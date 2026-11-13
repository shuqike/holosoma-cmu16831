"""Prepare one complete checkpoint optimizer update from saved production FIFO seeds.

The G1 task remains unchanged. Uses saved production FIFO seeds when available.
Records reward/kinematic consistency, complete
model-objective finite differences, the actual update, and all source hashes.
Real-policy improvement is checked separately in fresh full-protocol episodes.
"""
import argparse
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path

from train_supermimic_isaacgym import make_config
from supermimic_validation import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--collect-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output, args.checkpoint = (repo / args.output).resolve(), (repo / args.checkpoint).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = make_config(args.output / "events", args.num_envs, args.seed, 1)
    from holosoma.agents.supermimic.supermimic import SuperMimicAgent
    from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.common import seeding
    from holosoma.utils.sim_utils import setup_simulation_environment
    torch.set_float32_matmul_precision("high")
    state = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_env_state = state.get("env_state")
    saved_cfg = state.get("experiment_config", {}).get("algo", {}).get("config", {})
    keys = {field.name for field in fields(cfg.algo.config)}
    overrides = {k:v for k,v in saved_cfg.items() if k in keys}
    overrides.update(model_hidden_dim=state["world_model"]["network.0.weight"].shape[0], model_history_fraction=0.)
    overrides["implementation"] = state.get("implementation", "legacy_v4")
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    cfg = replace(cfg, algo=replace(cfg.algo, config=replace(cfg.algo.config, **overrides)))
    task_fields = ("robot", "terrain", "observation", "action", "termination", "randomization", "command", "curriculum", "reward")
    task = {key: asdict(getattr(cfg, key)) for key in task_fields}
    assert all(task[key] == asdict(getattr(g1_29dof_fast_sac, key)) for key in task_fields)
    del state
    seeding(args.seed, torch_deterministic=False)
    env, device, _ = setup_simulation_environment(cfg)
    agent = SuperMimicAgent(env, cfg.algo.config, device, str(args.output / "events"))
    agent.setup()
    agent.load(str(args.checkpoint), load_history=False)
    agent.actor.eval()
    agent.world_model.eval()
    io = agent.io
    env.model_io = io
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        source_sha256={p.name:digest(p) for p in (repo / "src/holosoma/holosoma/agents/supermimic").glob("*.py")},
        runner_sha256=digest(Path(__file__)), protocol_equal=True, task=task,
        algorithm_config=asdict(agent.config), seed=args.seed, num_envs=args.num_envs,
        production_matmul_precision=torch.get_float32_matmul_precision(),
        collect_steps=args.collect_steps, status="collecting", physical_max_errors={}, model_finite_differences=[],
        physics_context_encoding=io.context_encoding, initial_penalty_scale=float(env.reward_penalty_scale),
        actor_update_from_training_replay=False)
    report["optimizer_kind"] = "muon_adam" if agent.reference else "adam"
    def write():
        (args.output / "prepared_update.json").write_text(json.dumps(report, indent=2) + "\n")
    write()
    original_reward = env._compute_reward
    errors = dict(reward=0., gravity=0., foot_separation=0., foot_orientation=0., root_gravity=0.)
    def checked_reward():
        original_reward()
        with torch.no_grad():
            physical = io.unpack(io.state())
            from holosoma.agents.supermimic.reference_components import quaternion_rotate
            root_pose = io.root_pose()
            gravity_world = root_pose.new_tensor([0.,0.,-1.]).expand(root_pose.shape[0],3)
            expected_gravity = quaternion_rotate(root_pose[:,3:7],gravity_world,inverse=True)
            errors["root_gravity"] = max(errors["root_gravity"],float((expected_gravity-physical.gravity).abs().max()))
            expected = io.reward()(physical, env.command_manager.commands, io.gait.phase,
                env.action_manager.action, env.action_manager.prev_action)
            errors["reward"] = max(errors["reward"], float((expected - env.rew_buf).abs().max()))
            errors["gravity"] = max(errors["gravity"], float((physical.gravity.norm(dim=-1)-1).abs().max()))
            separation, orientation, _ = agent.world_model.geometry(physical.joint_position, physical.gravity)
            errors["foot_separation"] = max(errors["foot_separation"], float((separation-physical.foot_separation).abs().max()))
            errors["foot_orientation"] = max(errors["foot_orientation"], float((orientation-physical.foot_orientation_penalty).abs().max()))
    env._compute_reward = checked_reward
    obs = env.reset_all()
    for offset in range(0, args.collect_steps, 32):
        obs = agent._collect(obs, steps=min(32, args.collect_steps-offset))
        if offset % 256 == 0:
            print("Prepared replay", min(offset+32,args.collect_steps), flush=True)
    report.update(status="differentiating", physical_max_errors=errors,
                  replay_transitions=len(agent.replay)*args.num_envs,
                  physical_collection_final_penalty_scale=float(env.reward_penalty_scale))
    # The physical-check rollout is extra diagnostic data. Its curriculum
    # evolution must not change the objective associated with this checkpoint.
    agent._restore_env_state(checkpoint_env_state)
    report["gradient_penalty_scale"] = float(env.reward_penalty_scale)
    write()
    seed_path = args.checkpoint.with_name("actor_seeds_"+args.checkpoint.name)
    if seed_path.is_file():
        saved_seeds = torch.load(seed_path,map_location="cpu")
        assert saved_seeds["checkpoint_sha256"]==report["checkpoint_sha256"]
        assert saved_seeds["source_sha256"]==report["source_sha256"]
        assert saved_seeds["physics_context_encoding"]==io.context_encoding
        assert len(saved_seeds["seeds"]["state"])>=agent.config.batch_size
        seeds = {key:saved_seeds["seeds"][key][:agent.config.batch_size].to(device) for key in agent.SEED_FIELDS}
        report.update(actor_update_from_training_replay=True,production_replay_snapshot={
            **{key:value for key,value in saved_seeds.items() if key!="seeds"},"path":str(seed_path)})
        del saved_seeds
    else:
        seeds = agent.replay.sample(agent.config.batch_size, 1, agent.SEED_FIELDS, agent.train_worlds)
        report["replay_scope"] = "Fresh diagnostic replay fallback; cannot authorize the production-update benchmark gate"
    parameters = list(agent.actor.parameters())
    originals = [p.detach().clone() for p in parameters]
    agent.world_model.requires_grad_(False)
    agent.critic.requires_grad_(False)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    rollout = agent._imagine_policy(seeds)
    gradients = torch.autograd.grad(rollout["objective"].mean(), parameters, retain_graph=True)
    reward_gradients = torch.autograd.grad(rollout["reward_objective"].mean(), parameters, retain_graph=True)
    bootstrap_gradients = torch.autograd.grad(rollout["bootstrap"].mean(), parameters, retain_graph=True)
    report["objective_mean"] = float(rollout["objective"].mean())
    report["terminal_value_mean"] = float(rollout["bootstrap"].mean())
    report["gradient_norm"] = float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
    reward_norm = float(torch.stack([g.square().sum() for g in reward_gradients]).sum().sqrt())
    bootstrap_norm = float(torch.stack([g.square().sum() for g in bootstrap_gradients]).sum().sqrt())
    component_dot = sum(float((g*h).sum()) for g,h in zip(reward_gradients,bootstrap_gradients))
    report["gradient_components"] = dict(reward_norm=reward_norm,bootstrap_norm=bootstrap_norm,
        reward_bootstrap_cosine=component_dot/max(reward_norm*bootstrap_norm,1e-30),
        interpretation="Magnitude/alignment diagnostic, not evidence that either component matches real physics")
    agent._optimize(-rollout["objective"].mean(), agent.actor_optimizer, agent.actor)
    delta = [p.detach()-old for p,old in zip(parameters, originals)]
    derivative = sum(float((g*d).sum()) for g,d in zip(gradients, delta))
    report.update(optimizer_step_norm=float(torch.stack([d.square().sum() for d in delta]).sum().sqrt()),
                  model_autograd_directional_derivative=derivative,
                  learning_rates=[group["lr"] for group in agent.actor_optimizer.param_groups])
    report["gradient_components"].update(
        reward_dot_optimizer_step=sum(float((g*d).sum()) for g,d in zip(reward_gradients,delta)),
        bootstrap_dot_optimizer_step=sum(float((g*d).sum()) for g,d in zip(bootstrap_gradients,delta)))
    del rollout, gradients, reward_gradients, bootstrap_gradients
    @torch.no_grad()
    def model_value(step_scale):
        for p,old,d in zip(parameters, originals, delta):
            p.copy_(old + step_scale*d)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        return agent._imagine_policy(seeds)["objective"].detach().cpu()
    baseline = model_value(0.)
    values = {0.: baseline}
    for step_scale in (.01, .1, .25, .5, 1.):
        plus, minus = model_value(step_scale), model_value(-step_scale)
        values[step_scale], values[-step_scale] = plus, minus
        finite = float((plus-minus).mean()) / (2*step_scale)
        report["model_finite_differences"].append(dict(step_scale=step_scale,
            finite_difference=finite, autograd=derivative,
            relative_error=abs(finite-derivative)/max(abs(derivative),1e-8),
            plus_gain=float((plus-baseline).mean()), minus_gain=float((minus-baseline).mean())))
        write()
    model_value(0.)
    # Float64 avoids treating TF32 rounding at tiny parameter displacements as
    # an autodiff bug. This smaller diagnostic batch tests the SAME full formula;
    # the recorded optimizer update above still uses the production batch and dtype.
    for network in (agent.actor, agent.world_model, agent.critic):
        network.double()
    exact_seeds = {key:value[:64].double() if value.is_floating_point() else value[:64]
                   for key,value in seeds.items()}
    exact_original = [p.detach().clone() for p in parameters]
    exact_delta = [d.double() for d in delta]
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    objective = agent._imagine_policy(exact_seeds)["objective"].mean()
    exact_grad = torch.autograd.grad(objective, parameters)
    exact_derivative = sum(float((g*d).sum()) for g,d in zip(exact_grad, exact_delta))
    exact_checks = []
    with torch.no_grad():
        for epsilon in (.0001, .001, .01):
            endpoints = []
            for sign in (1, -1):
                for p,old,d in zip(parameters, exact_original, exact_delta):
                    p.copy_(old + sign*epsilon*d)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                endpoints.append(float(agent._imagine_policy(exact_seeds)["objective"].mean()))
            measured = (endpoints[0]-endpoints[1])/(2*epsilon)
            error = abs(measured-exact_derivative)
            exact_checks.append(dict(epsilon=epsilon, finite_difference=measured, autograd=exact_derivative,
                absolute_error=error, passed=error <= 1e-7 + .001*abs(exact_derivative)))
    report["float64_full_objective_check"] = dict(batch_size=len(exact_seeds["state"]), measurements=exact_checks)
    record = dict(checkpoint_sha256=report["checkpoint_sha256"], delta=[d.cpu() for d in delta],
                  evaluation_env_state=checkpoint_env_state,
                  source_sha256=report["source_sha256"], seed=args.seed, model_values=values)
    torch.save(record, args.output / "update.pt")
    report.update(status="completed", update=str(args.output / "update.pt"),
        context_encoding_gate=io.context_encoding=="isaacgym_physical_v1",
        physical_gate=max(errors.values()) < 1e-3 and io.context_encoding=="isaacgym_physical_v1",
        physical_gate_scope="Analytic reward parity, gravity norm, URDF/simulator foot geometry and fixed physical context convention; predictive contact dynamics are tested by the held-out rollouts and real-simulator update checks",
        numerical_model_gradient_gate=sum(row["passed"] for row in exact_checks) >= 2,
        scope=__doc__, benchmark_authorized_by_gates=False)
    assert {p.name:digest(p) for p in (repo / "src/holosoma/holosoma/agents/supermimic").glob("*.py")} == report["source_sha256"]
    write()
    agent.writer.close()
    print(json.dumps({k:report[k] for k in ("status", "physical_gate", "numerical_model_gradient_gate", "actor_update_from_training_replay", "model_autograd_directional_derivative")}), flush=True)


if __name__ == "__main__":
    main()
