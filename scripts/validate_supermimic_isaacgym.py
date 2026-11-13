"""Staged checks under the unchanged legacy G1 task; never launches a benchmark."""
import argparse
import hashlib
import json
from dataclasses import asdict, replace, fields as dataclass_fields
from pathlib import Path

from train_supermimic_isaacgym import make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration-only", action="store_true", help="Compatibility flag; this entry point only performs physical/critic calibration")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output = (repo / args.output).resolve()
    if args.checkpoint:
        args.checkpoint = (repo / args.checkpoint).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args.output, args.num_envs, 31415, args.iterations)
    from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac
    from holosoma.agents.supermimic.supermimic import SuperMimicAgent
    from holosoma.utils.common import seeding
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.sim_utils import setup_simulation_environment
    torch.set_float32_matmul_precision("high")
    if args.checkpoint:
        dimensions = torch.load(args.checkpoint, map_location="cpu")
        known = {field.name for field in dataclass_fields(cfg.algo.config)}
        restored = {key:value for key,value in dimensions.get("experiment_config",{}).get("algo",{}).get("config",{}).items() if key in known}
        restored["implementation"] = dimensions.get("implementation","legacy_v4")
        restored["model_hidden_dim"] = dimensions["world_model"]["network.0.weight"].shape[0]
        cfg = replace(cfg, algo=replace(cfg.algo, config=replace(cfg.algo.config,
            **restored)))
        del dimensions

    # Compare task configuration, not just a hand-picked subset of reward weights.
    fields = ("robot", "terrain", "observation", "action", "termination", "randomization", "command", "curriculum", "reward")
    for name in fields:
        assert asdict(getattr(cfg, name)) == asdict(getattr(g1_29dof_fast_sac, name)), name
    report = {"protocol_equal": True, "task_fields": fields, "seed": 31415,
              "num_envs": args.num_envs, "iterations": args.iterations,
              "matmul_precision": torch.get_float32_matmul_precision(),
              "algorithm": asdict(cfg.algo.config), "benchmark_authorized_by_gates": False}
    source = Path(__file__).resolve().parents[1] / "src/holosoma/holosoma/agents/supermimic"
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    def save():
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    seeding(cfg.training.seed, torch_deterministic=False)
    env, device, _ = setup_simulation_environment(cfg)
    agent = SuperMimicAgent(env, cfg.algo.config, device, str(args.output))
    agent.setup()
    agent.attach_checkpoint_metadata(cfg)
    obs = env.reset_all()
    fk_error = {"separation_m": 0., "orientation": 0.}
    with torch.no_grad():
        for _ in range(32):
            obs, _, _, _ = env.step({"actions": agent.actor(obs["actor_obs"])})
            physical = agent.io.unpack(agent.io.state())
            sep, ori, _ = agent.world_model.geometry(physical.joint_position, physical.gravity)
            fk_error["separation_m"] = max(fk_error["separation_m"], (sep-physical.foot_separation).abs().max().item())
            fk_error["orientation"] = max(fk_error["orientation"], (ori-physical.foot_orientation_penalty).abs().max().item())
    report["urdf_vs_simulator"] = fk_error
    save()
    assert max(fk_error.values()) < 1e-3, fk_error
    # A zero rotation must have a finite Jacobian, including at the initial pose.
    x = torch.zeros(16, agent.io.state_scale.numel(), device=device)
    x[:, 8] = -1
    x.requires_grad_()
    a = torch.zeros(16, 29, device=device, requires_grad=True)
    c = agent.io.context()[:16]
    y, _ = agent.world_model(x, a, c)
    grads = torch.autograd.grad(y.square().sum(), (x, a))
    assert all(torch.isfinite(g).all() for g in grads)
    report["finite_model_jacobian"] = True
    save()
    if args.checkpoint:
        agent.load(str(args.checkpoint), load_history=False)
    else:
        agent.learn()
    report["training_iteration"] = agent.iteration
    checkpoint = args.checkpoint or args.output / ("model_" + str(agent.iteration) + ".pt")
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu")
    training_envs = saved.get("experiment_config", {}).get("training", {}).get("num_envs")
    report["training_num_envs"] = training_envs
    report["collected_control_steps_per_training_world"] = saved["collected_steps"]
    # Evaluation may use fewer worlds than training; never multiply the saved
    # per-world collection counter by the new evaluation environment count.
    report["collected_transitions"] = saved["collected_steps"] * training_envs if training_envs else None
    report["validation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    del saved
    # Calibration is independent of fitting: new rollout with no exploration.
    agent.writer = type(agent.writer)(log_dir=str(args.output / "validation-events"))
    agent.config = replace(agent.config, num_steps_per_env=1)
    agent.exploration_std.zero_()
    agent.actor.eval()
    agent.world_model.eval()
    env.model_io = agent.io
    obs = env.reset_all()
    io = agent.io
    if args.checkpoint:
        # Checkpoints contain weights, not replay; validation seeds must be fresh.
        obs = agent._collect(obs, steps=128)
    values, rewards, dones, terminals = [], [], [], []
    gravity_error = 0.0
    with torch.no_grad():
        for _ in range(1000):
            inputs = agent._value_input(io.state(), io.context(), env.command_manager.commands,
                io.gait.phase, io.gait.phase_dt, env.action_manager.action, io.delay())
            values.append(agent.critic(inputs).squeeze(-1))
            obs, reward, done, info = env.step({"actions": agent.actor(obs["actor_obs"])})
            rewards.append(reward.clone())
            dones.append(done.bool().clone())
            terminals.append((done.bool() & ~info["time_outs"].bool()).clone())
        batch = agent.replay.sample(512, 1, agent.SEED_FIELDS)
        s, c, command, phase, phase_dt, prev, delayed = (batch[n][:, 0] for n in agent.SEED_FIELDS)
        for _ in range(32):
            act = agent.actor(io.actor_observation(io.unpack(s), command, phase, prev))
            s, _ = agent.world_model(s, io.applied_action(act, prev, delayed), c)
            gravity_error = max(gravity_error, (s[:, 6:9].norm(dim=-1)-1).abs().max().item())
            phase = (phase + phase_dt + torch.pi) % (2 * torch.pi) - torch.pi
            prev = act
        target = torch.zeros_like(rewards[0])
        known = torch.zeros_like(dones[0])
        distance = torch.zeros_like(rewards[0])
        returns, valid, tail_discounts = [], [], []
        for reward, done, terminal in zip(reversed(rewards), reversed(dones), reversed(terminals)):
            target = reward + cfg.algo.config.gamma * (~done) * target
            # Keep long-lived trajectories: selecting only falls biases calibration.
            # A fall has exactly zero tail. At a timeout or the collection boundary
            # retain the long-window MC estimate only once its unobserved tail has
            # discount <= 1e-3. Report that approximation explicitly; never bootstrap
            # the calibration target from the critic being tested.
            known = torch.where(done, terminal, known)
            distance = torch.where(done, torch.ones_like(distance), distance + 1)
            tail_discount = torch.where(known, torch.zeros_like(distance), cfg.algo.config.gamma**distance)
            returns.append(target.clone())
            valid.append(tail_discount <= 1e-3)
            tail_discounts.append(tail_discount)
        valid = torch.stack(valid[::-1])[:128]
        observed = torch.stack(returns[::-1])[:128][valid]
        predicted = torch.stack(values)[:128][valid]
        scale = observed.square().mean().sqrt().clamp_min(1)
        rmse = (predicted-observed).square().mean().sqrt()
        report["gravity_max_norm_error_32_steps"] = gravity_error
        report["critic"] = dict(states=int(valid.sum()), coverage=valid.float().mean().item(),
            predicted_mean=predicted.mean().item(), observed_mean=observed.mean().item(),
            rmse=rmse.item(), normalized_rmse=(rmse/scale).item())
        tails = torch.stack(tail_discounts[::-1])[:128][valid]
        report["critic"].update(max_unobserved_tail_discount=tails.max().item(),
            truncated_fraction=(tails > 0).float().mean().item(),
            target="independent long-window Monte Carlo, zero tail at falls; truncated tails explicitly reported")
        torch.save(dict(predicted=predicted.cpu(), observed=observed.cpu(), tail_discount=tails.cpu()),
                   args.output / "critic_calibration.pt")
        report["critic"]["calibration_bins"] = [dict(count=int(indices.numel()),
            predicted_mean=predicted[indices].mean().item(), observed_mean=observed[indices].mean().item())
            for indices in predicted.argsort().tensor_split(10) if indices.numel()]
        report["physical_gate"] = gravity_error < 1e-4
        report["critic_gate"] = bool(valid.float().mean() > 0.95 and rmse / scale < 0.25)
        report["diagnostic_control_steps"] = 32 + (128 if args.checkpoint else 0) + 1000
        report["diagnostic_simulator_transitions"] = report["diagnostic_control_steps"] * args.num_envs
        report["diagnostic_count_scope"] = "Explicit physical-check, warmup and MC env.step calls; excludes reset implementation and any optional training"
        report["physical_gate_scope"] = "gravity direction only; full kinematic consistency still pending"
    save()
    report["physical_gate_scope"] = "gravity norm; URDF/simulator geometry agreement measured independently"
    report["benchmark_authorized_by_gates"] = False
    save()
    agent.writer.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
