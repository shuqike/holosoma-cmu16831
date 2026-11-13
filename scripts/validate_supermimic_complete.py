"""Complete-update validation, with physical checks, critic MC and independent full-task episodes."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from supermimic_validation import digest


VALIDATION_SCRIPTS = ("validate_supermimic_complete.py", "prepare_supermimic_update.py",
    "validate_supermimic_isaacgym.py", "evaluate_g1_checkpoint.py",
    "supermimic_validation.py", "train_supermimic_isaacgym.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seeds", nargs="+", type=int, default=[31415, 271828, 161803])
    parser.add_argument("--prepared-update", type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        parser.error("Complete validation requires at least three distinct simulator seeds")
    repo = Path(__file__).resolve().parents[1]
    checkpoint, output = (repo / args.checkpoint).resolve(), (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(validation_schema="complete_actor_update_v1", checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), status="running", num_envs=args.num_envs, seeds=args.seeds,
        source_sha256={p.name:digest(p) for p in (repo / "src/holosoma/holosoma/agents/supermimic").glob("*.py")},
        validation_script_sha256={name:digest(repo / "scripts" / name) for name in VALIDATION_SCRIPTS},
        benchmark_authorized_by_gates=False, stages=[])
    def write():
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    def stage(name, script, arguments):
        report["active_stage"] = name
        write()
        print("Validation:", name, flush=True)
        with (output / (name + ".log")).open("w") as stream:
            subprocess.run([sys.executable, "-u", str(repo / "scripts" / script)] + list(map(str,arguments)),
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        report["stages"].append(name)
        write()
    try:
        prepared = (repo / args.prepared_update).resolve() if args.prepared_update else output / "prepared"
        if not args.prepared_update:
            stage("prepare", "prepare_supermimic_update.py", ["--checkpoint", checkpoint, "--output", prepared,
                                                              "--num-envs", args.num_envs])
        preparation = json.loads((prepared / "prepared_update.json").read_text())
        assert preparation["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert preparation["source_sha256"] == report["source_sha256"]
        assert preparation["status"] == "completed"
        assert preparation["algorithm_config"]["batch_size"] == 4096, "Use the production actor batch"
        assert preparation["actor_update_from_training_replay"], "Use the saved production FIFO actor seeds"
        assert preparation["production_replay_snapshot"]["retained_replay_steps"]==preparation["algorithm_config"]["replay_steps"], "The production FIFO must be filled"
        report.update(preparation=str(prepared / "prepared_update.json"),
                      physical_gate=preparation["physical_gate"],
                      physical_gate_scope=preparation["physical_gate_scope"],
                      context_encoding_gate=preparation["context_encoding_gate"],
                      production_replay_snapshot=preparation["production_replay_snapshot"],
                      numerical_model_gradient_gate=preparation["numerical_model_gradient_gate"],
                      protocol_equal=preparation["protocol_equal"])
        stage("critic", "validate_supermimic_isaacgym.py", ["--checkpoint", checkpoint,
            "--output", output / "critic", "--num-envs", 512, "--iterations", 1, "--calibration-only"])
        calibration = json.loads((output / "critic/validation.json").read_text())
        assert calibration["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert calibration["source_sha256"] == report["source_sha256"]
        report["critic_gate"] = calibration["critic_gate"]
        report["critic"] = calibration["critic"]
        report["physical_gate"] &= calibration["physical_gate"]
        write()
        # Complete the diagnostic even if calibration fails; no training/comparison is launched here.
        conditions = [(seed, scale, False) for seed in args.seeds for scale in (0., 1., -1.)]
        conditions.append((args.seeds[0], 0., True))
        paths = {}
        for seed, scale, repeat in conditions:
            name = "seed%d_step%s%s" % (seed, scale, "_repeat" if repeat else "")
            path = output / name
            stage(name, "evaluate_g1_checkpoint.py", ["--algorithm", "supermimic", "--checkpoint", checkpoint,
                "--output", path, "--seed", seed, "--num-envs", args.num_envs, "--load-curriculum",
                "--actor-update", prepared / "update.pt", "--step-scale", scale, "--matmul-precision", "high"])
            paths[(seed,scale,repeat)] = path
        report["diagnostic_simulator_transitions"] = dict(
            preparation=preparation["collect_steps"]*preparation["num_envs"],
            critic=calibration["diagnostic_simulator_transitions"],
            complete_update=sum(json.loads((path / "evaluation.json").read_text())["evaluation_simulator_transitions"]
                                for path in paths.values()))
        report["diagnostic_transition_count_scope"] = "Additional validation interactions, not policy training or free pretraining; counts explicit simulator control calls including background worlds in first-episode evaluations"
        from holosoma.utils.safe_torch_import import torch
        paired, gains, central, initial_errors = [], [], [], []
        for seed in args.seeds:
            episodes = [torch.load(paths[(seed,scale,False)] / "episodes.pt", map_location="cpu") for scale in (0.,1.,-1.)]
            before, plus, minus = episodes
            for variant in (plus, minus):
                initial_errors.extend(float((value-variant["initial_state"][key]).abs().max())
                    for key,value in before["initial_state"].items())
            gain = plus["discounted_return"] - before["discounted_return"]
            direction = plus["discounted_return"] - minus["discounted_return"]
            gains.append(gain)
            central.append(direction)
            paired.append(dict(seed=seed, episodes=len(gain), mean_gain=float(gain.mean()),
                gain_se=float(gain.std()/len(gain)**.5), mean_central_change=float(direction.mean()),
                undiscounted_mean_gain=float((plus["episode_return"]-before["episode_return"]).mean())))
            paired[-1]["torch_rng_common_prefix_steps"] = [next((i for i,(a,b) in enumerate(zip(
                before["torch_rng_signatures"],variant["torch_rng_signatures"])) if a!=b),
                min(len(before["torch_rng_signatures"]),len(variant["torch_rng_signatures"]))) for variant in (plus,minus)]
        first = torch.load(paths[(args.seeds[0],0.,False)] / "episodes.pt", map_location="cpu")
        repeated = torch.load(paths[(args.seeds[0],0.,True)] / "episodes.pt", map_location="cpu")
        noise = (first["discounted_return"] - repeated["discounted_return"]).abs()
        gain, direction = torch.cat(gains), torch.cat(central)
        mean, se = float(gain.mean()), float(gain.std()/len(gain)**.5)
        dmean, dse = float(direction.mean()), float(direction.std()/len(direction)**.5)
        seed_mean = statistics.mean(row["mean_gain"] for row in paired)
        seed_se = statistics.stdev(row["mean_gain"] for row in paired)/len(paired)**.5
        seed_central = statistics.mean(row["mean_central_change"] for row in paired)
        seed_central_se = statistics.stdev(row["mean_central_change"] for row in paired)/len(paired)**.5
        initial_match = max(initial_errors) == 0
        report.update(paired_seeds=paired, initial_state_match=initial_match,
            repeat_max_error=float(noise.max()), repeat_mean_error=float(noise.mean()),
            complete_update=dict(mean_gain=mean, standard_error=se, central_change=dmean, central_standard_error=dse,
                seed_mean_gain=seed_mean, seed_standard_error=seed_se,
                seed_central_change=seed_central, seed_central_standard_error=seed_central_se,
                episodes=len(gain), interpretation="Paired evaluation episodes across fixed independent simulator seeds, conditional on this training checkpoint"),
            gate_interpretation="Engineering guard, not a theorem or a paper hyperparameter: require positive full-step and central returns beyond two SE at both episode and simulator-seed levels; evaluate the exact optimizer step, without selecting a smaller successful step",
            full_actor_update_gate=bool(initial_match and float(noise.max()) < 1e-5 and mean > 2*se
                and dmean > 2*dse and seed_mean > 2*seed_se and seed_central > 2*seed_central_se
                and all(row["mean_gain"] > 0 for row in paired)))
        report["benchmark_authorized_by_gates"] = all(report[key] for key in
            ("protocol_equal", "physical_gate", "critic_gate", "numerical_model_gradient_gate", "full_actor_update_gate"))
        report["status"] = "passed" if report["benchmark_authorized_by_gates"] else "failed_or_inconclusive"
        assert {p.name:digest(p) for p in (repo / "src/holosoma/holosoma/agents/supermimic").glob("*.py")} == report["source_sha256"]
        assert {name:digest(repo / "scripts" / name) for name in VALIDATION_SCRIPTS} == report["validation_script_sha256"]
        write()
    except BaseException:
        report["status"] = "execution_failed"
        write()
        raise
    print(json.dumps({key:report[key] for key in ("status", "physical_gate", "critic_gate", "numerical_model_gradient_gate", "full_actor_update_gate", "complete_update")},indent=2),flush=True)


if __name__ == "__main__":
    main()
