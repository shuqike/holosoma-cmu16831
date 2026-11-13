"""Bounded G1 PPO training with first-failure checks before environment resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import mujoco_warp
import warp as wp

from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof
from holosoma.config_values.simulator import mjwarp
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend
from holosoma.train_agent import train
from holosoma.utils.safe_torch_import import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "physics_steps": 0,
        "control_steps": 0,
        "max_qvel": 0.0,
        "collision_source_sha256": hashlib.sha256(
            (Path(mujoco_warp.__file__).parent / "_src" / "collision_gjk.py").read_bytes()
        ).hexdigest(),
    }
    step = WarpBackend.step
    env_step = BaseTask.step

    def checked_physics(backend):
        step(backend)
        wp.synchronize()
        fields = ("qpos", "qvel", "qacc", "ctrl", "qfrc_constraint", "qfrc_smooth")
        values = torch.cat([wp.to_torch(getattr(backend.mjw_data, key)).reshape(-1) for key in fields])
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"Invalid training physics at step {result['physics_steps'] + 1}")
        count = int(wp.to_torch(backend.mjw_data.nacon)[0])
        contact = backend.mjw_data.contact
        contact_values = torch.cat(
            [wp.to_torch(getattr(contact, key))[:count].reshape(-1) for key in ("dist", "pos", "frame")]
        )
        if not torch.isfinite(contact_values).all():
            raise FloatingPointError("Invalid training contact")
        result["physics_steps"] += 1
        result["max_qvel"] = max(result["max_qvel"], float(backend.qvel_t.abs().max()))

    def checked_environment(env, actor_state):
        if "model" not in result:
            model = env.simulator.root_model
            result["model"] = {
                "timestep": model.opt.timestep,
                "integrator": int(model.opt.integrator),
                "hfield_size": model.hfield_size.tolist(),
            }
        if not torch.isfinite(actor_state["actions"]).all():
            raise FloatingPointError("Invalid policy action")
        observation, reward, done, extras = env_step(env, actor_state)
        if not torch.isfinite(torch.cat([reward.reshape(-1), *[v.reshape(-1) for v in observation.values()]])).all():
            raise FloatingPointError("Invalid training observation/reward")
        result["control_steps"] += 1
        return observation, reward, done, extras

    WarpBackend.step = checked_physics
    BaseTask.step = checked_environment
    cfg = replace(
        g1_29dof,
        simulator=mjwarp,
        training=replace(
            g1_29dof.training,
            num_envs=args.num_envs,
            seed=args.seed,
            project="mjwarp-nan-validation",
            name="ppo_fullterrain",
        ),
        logger=DisabledLoggerConfig(base_dir=str(args.output.parent / "train")),
        algo=replace(
            g1_29dof.algo,
            config=replace(
                g1_29dof.algo.config, num_learning_iterations=args.iterations, save_interval=args.iterations
            ),
        ),
    )
    start = time.perf_counter()
    checkpoint_root = args.output.parent / "train"
    existing_checkpoints = set(checkpoint_root.rglob("model_*.pt"))
    try:
        train(cfg)
        checkpoints = set(checkpoint_root.rglob("model_*.pt")) - existing_checkpoints
        if not checkpoints:
            raise RuntimeError("Training returned without a new checkpoint")
        checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["iter"] != args.iterations - 1:
            raise RuntimeError(f"Training stopped at iteration {state['iter']}")

        def check_tensors(value, key="checkpoint"):
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite tensor: {key}")
            elif isinstance(value, dict):
                for name, child in value.items():
                    check_tensors(child, f"{key}.{name}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    check_tensors(child, f"{key}[{index}]")

        check_tensors(state)
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_tensors_finite"] = True
        result["status"] = "completed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        WarpBackend.step = step
        BaseTask.step = env_step
        result["elapsed_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
