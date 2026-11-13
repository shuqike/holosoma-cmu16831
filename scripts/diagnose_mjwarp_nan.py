"""Bounded, optimizer-free MJWarp probe; stop at the first invalid physics step."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from loguru import logger

from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac
from holosoma.config_values.simulator import mjwarp
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import setup_simulation_environment, simulation_resource_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--integrator", choices=("euler", "implicitfast"))
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--plane", action="store_true")
    parser.add_argument("--base-depth", type=float)
    parser.add_argument("--capture-spike", action="store_true")
    parser.add_argument("--spike-qvel", type=float, default=100.0)
    parser.add_argument(
        "--summary-only", action="store_true", help="Check every physics step without retaining every snapshot"
    )
    parser.add_argument("--action-std", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summary_only and args.capture_spike:
        parser.error("--summary-only and --capture-spike are mutually exclusive")
    if args.fps <= 0 or args.fps % 50:
        parser.error("--fps must be a positive multiple of 50 to preserve the control rate")
    logger.remove()
    if args.base_depth is not None:
        from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend

        original_init = WarpBackend.__init__

        def patched_init(self, model, *positional, **keywords):
            model.hfield_size[:, 3] = args.base_depth
            original_init(self, model, *positional, **keywords)

        WarpBackend.__init__ = patched_init
    # Use the checked-in task directly: no policy or experimental agent is needed.
    cfg = replace(g1_29dof_fast_sac, simulator=mjwarp, nightly=None)
    terrain = replace(cfg.terrain.terrain_term, scale_factor=args.scale)
    if args.plane:
        from holosoma.config_values.terrain import terrain_locomotion_plane

        terrain = terrain_locomotion_plane.terrain_term
    sim = cfg.simulator.config.sim
    sim = replace(
        sim, fps=args.fps, control_decimation=args.fps // 50, mujoco=replace(sim.mujoco, integrator=args.integrator)
    )
    cfg = replace(
        cfg,
        training=replace(cfg.training, seed=args.seed, num_envs=args.num_envs),
        terrain=replace(cfg.terrain, terrain_term=terrain),
        simulator=replace(cfg.simulator, config=replace(cfg.simulator.config, sim=sim)),
    )
    env, device, app = setup_simulation_environment(cfg)
    from importlib import metadata

    import mujoco
    import mujoco_warp
    import warp as wp

    result = {
        "args": vars(args) | {"output": str(args.output)},
        "versions": {k: metadata.version(k) for k in ("mujoco", "mujoco-warp", "warp-lang", "torch")},
        "collision_source_sha256": hashlib.sha256(
            (Path(mujoco_warp.__file__).parent / "_src" / "collision_gjk.py").read_bytes()
        ).hexdigest(),
        "history": [],
        "summary": {
            "physics_steps": 0,
            "max_qvel": 0.0,
            "max_qacc": 0.0,
            "min_contact_dist": 0.0,
            "control_steps": 0,
            "resets": 0,
        },
    }
    backend = env.simulator.backend
    fields = ("qpos", "qvel", "qacc", "ctrl", "qfrc_constraint", "qfrc_smooth")

    def snapshot():
        out = {}
        for name in fields:
            value = wp.to_torch(getattr(backend.mjw_data, name))
            good = torch.isfinite(value)
            out[name] = {
                "finite": bool(good.all()),
                "max_abs": float(value[good].abs().max()) if good.any() else None,
                "bad_worlds": (~good).reshape(args.num_envs, -1).any(1).nonzero().flatten().tolist(),
            }
        for name in ("nacon", "nefc"):
            value = wp.to_torch(getattr(backend.mjw_data, name))
            out[name] = int(value.max())
        return out

    class InvalidPhysicsError(RuntimeError):
        pass

    original_step = backend.step
    physics_step = 0

    def checked_step():
        nonlocal physics_step
        if args.sync:
            wp.synchronize()
            torch.cuda.synchronize()
        if args.summary_only:
            original_step()
            wp.synchronize()
            values = torch.cat([wp.to_torch(getattr(backend.mjw_data, name)).reshape(-1) for name in fields])
            count = int(wp.to_torch(backend.mjw_data.nacon)[0])
            contacts = backend.mjw_data.contact
            contact_dist = wp.to_torch(contacts.dist)[:count]
            contact_min = contact_dist.min() if count else values.new_zeros(())
            finite = torch.isfinite(values).all() & torch.isfinite(contact_dist).all()
            finite &= torch.isfinite(wp.to_torch(contacts.pos)[:count]).all()
            finite &= torch.isfinite(wp.to_torch(contacts.frame)[:count]).all()
            metrics = (
                torch.stack(
                    (
                        finite.float(),
                        backend.qvel_t.abs().max(),
                        backend.qacc_t.abs().max(),
                        contact_min,
                    )
                )
                .cpu()
                .tolist()
            )
            physics_step += 1
            summary = result["summary"]
            summary["physics_steps"] = physics_step
            if not metrics[0]:
                result["history"].append({"physics_step": physics_step, "after": snapshot()})
                raise InvalidPhysicsError(f"first invalid physics step {physics_step}")
            summary["max_qvel"] = max(summary["max_qvel"], metrics[1])
            summary["max_qacc"] = max(summary["max_qacc"], metrics[2])
            summary["min_contact_dist"] = min(summary["min_contact_dist"], metrics[3])
            return
        before = snapshot()
        previous = (
            {
                k: wp.to_torch(getattr(backend.mjw_data, k)).clone()
                for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied")
            }
            if args.capture_spike and "spike" not in result
            else None
        )
        original_step()
        wp.synchronize()
        after = snapshot()
        physics_step += 1
        result["history"].append({"physics_step": physics_step, "before": before, "after": after})
        result["summary"]["physics_steps"] = physics_step
        for field in ("qvel", "qacc"):
            maximum = after[field]["max_abs"]
            if maximum is not None:
                key = f"max_{field}"
                result["summary"][key] = max(result["summary"][key], maximum)
        if previous is not None and after["qvel"]["max_abs"] > args.spike_qvel:
            import copy

            import numpy as np

            velocities = backend.qvel_t.abs().amax(1)
            world = int(velocities.argmax())
            cpu_model = copy.copy(m)
            for name in (
                "body_mass",
                "body_inertia",
                "body_ipos",
                "geom_friction",
                "dof_damping",
                "body_pos",
                "body_quat",
                "geom_solref",
                "geom_solimp",
            ):
                values = wp.to_torch(getattr(backend.mjw_model, name))
                getattr(cpu_model, name)[:] = values[world if values.shape[0] > 1 else 0].cpu().numpy()
            cpu_data = mujoco.MjData(cpu_model)
            for name, value in previous.items():
                getattr(cpu_data, name)[:] = value[world].cpu().numpy()
            mujoco.mj_forward(cpu_model, cpu_data)
            gpu_acc = backend.qacc_t[world].cpu().numpy()
            contacts = backend.mjw_data.contact
            count = int(wp.to_torch(backend.mjw_data.nacon)[0])
            mask = wp.to_torch(contacts.worldid)[:count] == world
            result["spike"] = {
                "world": world,
                "step": physics_step,
                "before_qvel": previous["qvel"][world].cpu().tolist(),
                "gpu_qacc": gpu_acc.tolist(),
                "cpu_qacc": cpu_data.qacc.tolist(),
                "cpu_max_qacc": float(np.abs(cpu_data.qacc).max()),
                "gpu_contacts_dist": wp.to_torch(contacts.dist)[:count][mask].cpu().tolist(),
                "gpu_contacts_geom": wp.to_torch(contacts.geom)[:count][mask].cpu().tolist(),
                "gpu_contacts_pos": wp.to_torch(contacts.pos)[:count][mask].cpu().tolist(),
                "geom_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)],
                "cpu_contacts_dist": cpu_data.contact.dist.tolist(),
            }
            np.savez(args.output.with_suffix(".npz"), **{k: v[world].cpu().numpy() for k, v in previous.items()})
            mujoco.mj_saveModel(cpu_model, str(args.output.with_suffix(".mjb")))
        if any(not after[k]["finite"] for k in fields):
            ids = sorted({i for k in fields for i in after[k]["bad_worlds"]})
            result["bad_worlds"] = ids
            result["bad_origins"] = env.terrain_manager.get_state("locomotion_terrain").env_origins[ids].cpu().tolist()
            raise InvalidPhysicsError(f"first invalid physics step {physics_step}")

    backend.step = checked_step
    m = env.simulator.root_model
    result["model"] = {
        "timestep": m.opt.timestep,
        "integrator": int(m.opt.integrator),
        "solver": int(m.opt.solver),
        "iterations": m.opt.iterations,
        "hfield_size": m.hfield_size.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with simulation_resource_session(env, app), torch.no_grad():
        # Preserve the original probe's joint-range-scaled action distribution.
        nominal = env.default_dof_pos_base.squeeze(0)
        robot = env.robot_config
        lower = nominal.new_tensor(robot.dof_pos_lower_limit_list)
        upper = nominal.new_tensor(robot.dof_pos_upper_limit_list)
        action_bound = torch.maximum(nominal - lower, upper - nominal) / robot.control.action_scale
        generator = torch.Generator(device="cpu").manual_seed(1638)
        tape = (torch.randn(args.steps, args.num_envs, env.num_dof, generator=generator) * args.action_std).to(device)
        tape = (tape * action_bound).clamp(-action_bound, action_bound)
        try:
            start = time.perf_counter()
            env.reset_all()
            for step, action in enumerate(tape, 1):
                observations, reward, done, _ = env.step({"actions": action})
                returned = torch.cat([reward.reshape(-1), *[value.reshape(-1) for value in observations.values()]])
                if not torch.isfinite(returned).all():
                    raise InvalidPhysicsError(f"non-finite observation/reward at control step {step}")
                result["summary"]["control_steps"] = step
                result["summary"]["resets"] += int(done.sum())
                if step % args.log_every == 0:
                    print(
                        json.dumps({"progress": result["summary"], "seconds": time.perf_counter() - start}), flush=True
                    )
            result["status"] = "completed"
        except InvalidPhysicsError as exc:
            result["status"] = str(exc)
        finally:
            result["elapsed_seconds"] = time.perf_counter() - start
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
