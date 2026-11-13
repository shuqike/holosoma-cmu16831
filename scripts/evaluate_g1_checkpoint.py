"""Independent first-episode evaluation under the unchanged G1 task.

Both algorithms start in fresh environments under the same seed and task settings.
Only policy weights and observation normalization are loaded. Training curriculum
state is deliberately not imported into this common evaluation distribution.
Random commands, observation noise, domain randomization and pushes remain enabled.
"""
import argparse
from dataclasses import asdict, replace, fields as dataclass_fields
import hashlib
import json
import math
from pathlib import Path

from train_supermimic_isaacgym import make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("supermimic", "fastsac"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--smoke-steps", type=int, default=0, help="Short incomplete plumbing check, not evaluation")
    parser.add_argument("--metadata-only", action="store_true", help="Verify checkpoint/task metadata without creating a simulator")
    parser.add_argument("--actor-update", type=Path, help="Diagnostic only: apply a recorded complete SuperMimic optimizer step")
    parser.add_argument("--step-scale", type=float, default=1.)
    parser.add_argument("--load-curriculum", action="store_true", help="Diagnostic only: use the checkpoint's reward curriculum for paired update checks")
    parser.add_argument("--matmul-precision", choices=("highest","high"), default="highest",
                        help="Complete-update diagnostics use training's high setting; common policy evaluation defaults to highest")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.checkpoint = (repo / args.checkpoint).resolve()
    args.output = (repo / args.output).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.actor_update and args.algorithm != "supermimic":
        parser.error("--actor-update is a SuperMimic diagnostic")
    cfg = make_config(args.output, args.num_envs, args.seed, 1)
    from holosoma.agents.supermimic.supermimic import SuperMimicAgent
    from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
    from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac
    from holosoma.managers.observation.terms.locomotion import get_base_lin_vel, get_base_ang_vel
    from holosoma.managers.reward.terms.locomotion import tracking_lin_vel, tracking_ang_vel
    from holosoma.utils.common import seeding
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.sim_utils import setup_simulation_environment
    import yaml
    torch.set_float32_matmul_precision(args.matmul_precision)

    training_cfg = yaml.safe_load((args.checkpoint.parent / "holosoma_config.yaml").read_text())
    fields = ("simulator", "robot", "terrain", "observation", "action", "command", "reward",
              "termination", "randomization", "curriculum", "scene", "sensor", "plugin")
    # Compare serialized schemas (JSON normalizes tuples to the YAML list representation).
    task = {}
    for key in fields:
        value = getattr(cfg, key, None)
        task[key] = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    task = json.loads(json.dumps(task))
    assert all(task[key] == training_cfg.get(key) for key in fields), "Evaluation task differs from training task"
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.algorithm == "supermimic":
        known = {field.name for field in dataclass_fields(cfg.algo.config)}
        restored = {key:value for key,value in training_cfg["algo"]["config"].items() if key in known}
        restored["implementation"] = checkpoint.get("implementation", "legacy_v4")
        cfg = replace(cfg,algo=replace(cfg.algo,config=replace(cfg.algo.config,**restored)))
    training_envs = training_cfg["training"]["num_envs"]
    # Legacy FastSAC collects at step zero and saves before incrementing the counter.
    training_steps = checkpoint["collected_steps"] if args.algorithm == "supermimic" else checkpoint["global_step"] + 1
    report = dict(algorithm=args.algorithm, checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        training_transitions=training_steps * training_envs, seed=args.seed, num_envs=args.num_envs,
        protocol_equal=True, task_sha256=hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
        scope=__doc__, smoke_test=bool(args.smoke_steps), metadata_only=args.metadata_only)
    report.update(curriculum_from_checkpoint=args.load_curriculum, actor_update=str(args.actor_update) if args.actor_update else None,
                  step_scale=args.step_scale if args.actor_update else 0., matmul_precision=args.matmul_precision)
    if args.load_curriculum:
        report["scope"] = "Paired full-G1-protocol update diagnostic using the checkpoint curriculum; not the common-curriculum algorithm benchmark"
    if args.metadata_only:
        (args.output / "metadata.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return
    seeding(args.seed, torch_deterministic=False)
    env, device, _ = setup_simulation_environment(cfg)
    evaluation_env_state = checkpoint.get("env_state")
    if args.algorithm == "supermimic":
        agent = SuperMimicAgent(env, cfg.algo.config, device, str(args.output / "events"))
        agent.setup()
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.actor.eval()
        if args.actor_update:
            update = torch.load((repo / args.actor_update).resolve(), map_location=device)
            evaluation_env_state = update.get("evaluation_env_state",evaluation_env_state)
            assert update["checkpoint_sha256"] == report["checkpoint_sha256"], "Update belongs to another checkpoint"
            assert len(update["delta"]) == len(list(agent.actor.parameters()))
            with torch.no_grad():
                for parameter, delta in zip(agent.actor.parameters(), update["delta"]):
                    assert parameter.shape == delta.shape
                    parameter.add_(delta, alpha=args.step_scale)
    else:
        agent = FastSACAgent(env, g1_29dof_fast_sac.algo.config, device, str(args.output / "events"))
        agent.setup()
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    if args.load_curriculum:
        agent._restore_env_state(evaluation_env_state)
    torch.set_float32_matmul_precision(args.matmul_precision)
    policy = agent.get_inference_policy()
    # Agent initialization draws different amounts of random data; isolate reset RNG.
    seeding(args.seed + 1, torch_deterministic=False)
    obs = env.reset_all()
    initial_state = dict(roots=env.simulator.all_root_states.clone().cpu(),
        joints=env.simulator.dof_state.clone().cpu(), command=env.command_manager.commands.clone().cpu(),
        actor_obs=obs["actor_obs"].clone().cpu())
    report["initial_penalty_scale"] = float(env.reward_penalty_scale)
    metrics = {}
    compute_reward = env._compute_reward
    lin_params = env.reward_manager.get_term_cfg("tracking_lin_vel").params
    ang_params = env.reward_manager.get_term_cfg("tracking_ang_vel").params

    def record_reward():
        compute_reward()
        metrics["linear_score"] = tracking_lin_vel(env, **lin_params).clone()
        metrics["angular_score"] = tracking_ang_vel(env, **ang_params).clone()
        metrics["linear_squared_error"] = (env.command_manager.commands[:, :2] - get_base_lin_vel(env)[:, :2]).square().sum(-1)
        metrics["yaw_squared_error"] = (env.command_manager.commands[:, 2] - get_base_ang_vel(env)[:, 2]).square()
        metrics["action_rate"] = (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)
    env._compute_reward = record_reward
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    length = torch.zeros(args.num_envs, device=device)
    episode_return = torch.zeros_like(length)
    discounted_return = torch.zeros_like(length)
    totals = {key: length.clone() for key in ("linear_score", "angular_score", "linear_squared_error", "yaw_squared_error", "action_rate")}
    timeouts = torch.zeros_like(active)
    max_seconds = cfg.simulator.config.sim.max_episode_length_s
    steps = args.smoke_steps or math.ceil(max_seconds / env.dt) + 2
    rng_signatures = []
    with torch.no_grad():
        for step in range(steps):
            if args.actor_update:
                # Identical reset seeds need not keep random streams aligned:
                # action-dependent resets can consume different numbers of draws.
                signature = torch.get_rng_state().numpy().tobytes() + torch.cuda.get_rng_state().numpy().tobytes()
                rng_signatures.append(hashlib.sha256(signature).hexdigest())
            obs, reward, done, info = env.step({"actions": policy(obs)})
            length += active
            episode_return += reward * active
            discounted_return += (cfg.algo.config.gamma**step) * reward * active
            for key in totals:
                totals[key] += metrics[key] * active
            finished = active & done.bool()
            timeouts |= finished & info["time_outs"].bool()
            active &= ~done.bool()
            if not active.any():
                break
    if not args.smoke_steps:
        assert not active.any(), "Some first episodes were censored by the evaluation boundary"

    def mean_se(values):
        return dict(mean=values.mean().item(), standard_error=(values.std() / math.sqrt(args.num_envs)).item())
    report.update(completed_episodes=int((~active).sum()), censored_episodes=int(active.sum()),
        simulated_control_steps=step+1, evaluation_simulator_transitions=(step+1)*args.num_envs,
        final_penalty_scale=float(env.reward_penalty_scale),
        timeout_fraction=timeouts.float().mean().item(),
        episode_return=mean_se(episode_return),
        discounted_episode_return=mean_se(discounted_return),
        episode_seconds=mean_se(length * env.dt),
        linear_velocity_episode_score=mean_se(totals["linear_score"] / max_seconds),
        angular_velocity_episode_score=mean_se(totals["angular_score"] / max_seconds),
        action_rate_episode_penalty=mean_se(totals["action_rate"] / max_seconds),
        episode_planar_velocity_rmse_mps=mean_se((totals["linear_squared_error"] / length).sqrt()),
        episode_yaw_velocity_rmse_radps=mean_se((totals["yaw_squared_error"] / length).sqrt()))
    torch.save({"length": length.cpu(), "timeouts": timeouts.cpu(), "episode_return": episode_return.cpu(),
                "discounted_return": discounted_return.cpu(), "initial_state": initial_state,
                "torch_rng_signatures": rng_signatures,
                **{key: value.cpu() for key, value in totals.items()}}, args.output / "episodes.pt")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    agent.writer.close()


if __name__ == "__main__":
    main()
