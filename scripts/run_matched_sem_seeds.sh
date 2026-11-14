#!/usr/bin/env bash
# EC2: bash scripts/run_matched_sem_seeds.sh --seconds 1800 --seeds 1 2
# Completed pairs are skipped; interrupted attempts restart with a fresh budget.
set -eo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
mode=run
for argument in "$@"; do
  if [[ $argument == --status || $argument == --help || $argument == -h ]]; then
    mode=status
  fi
done
if [[ $mode == run ]]; then
  mkdir -p logs/sem_experiments
  exec 9>logs/sem_experiments/.matched-seeds.lock
  if ! flock -n 9; then
    echo 'Another matched-seed batch is active. Wait for it to finish.' >&2
    exit 1
  fi
  # Conda activation is not compatible with nounset in every installation.
  source scripts/source_isaacgym_setup.sh
fi

# All Python imports here are standard-library or local protocol helpers.
python3 - "$repo_dir" "$@" <<'PY'
import argparse
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "scripts"))
from sem_experiment_utils import DEFAULT_ITERATIONS, PROTOCOL, default_output_root, timestamps, training_source_fingerprint

parser = argparse.ArgumentParser(description="Run only incomplete matched pairs for a timed experiment")
parser.add_argument("--status", action="store_true")
parser.add_argument("--seconds", type=float, default=900)
parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5])
parser.add_argument("--timestamps-seconds", type=float, nargs="+")
parser.add_argument("--timestamp-window-seconds", type=float, default=5.0)
parser.add_argument("--logging-interval", type=int, default=100)
parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
parser.add_argument("--output-root", type=Path)
ARGS = parser.parse_args(sys.argv[2:])
try:
    MARKS = timestamps(ARGS.seconds, ARGS.timestamps_seconds)
except ValueError as error:
    parser.error(str(error))
if (len(set(ARGS.seeds)) != len(ARGS.seeds) or any(seed < 0 for seed in ARGS.seeds)
        or ARGS.iterations <= 0 or ARGS.logging_interval <= 0
        or not math.isfinite(ARGS.timestamp_window_seconds) or ARGS.timestamp_window_seconds < 0):
    parser.error("Use unique nonnegative seeds, positive iteration/logging limits, and a finite nonnegative window")
STATUS_ONLY = ARGS.status
RESULTS = (REPO / (ARGS.output_root or default_output_root(ARGS.seconds))).resolve()
TRAINING_SOURCE = training_source_fingerprint(REPO)
PLAN = [(seed, method) for index, seed in enumerate(ARGS.seeds)
        for method in (("baseline", "sem") if index % 2 == 0 else ("sem", "baseline"))]


def active_jobs():
    """Include orphan trainers and manually launched jobs in this checkout."""
    jobs = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = proc.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
            cwd = proc.joinpath("cwd").resolve(strict=True)
            for arg in argv:
                if Path(arg).name in ("run_sem_experiment.py", "train_agent.py"):
                    if REPO in (cwd / arg).resolve().parents:
                        jobs.append((proc.name, " ".join(argv).strip()))
                        break
        except (OSError, RuntimeError):
            pass  # Process disappeared or belongs to another user.
    return jobs


def argument(command, flag):
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def classify(seed, method):
    complete, unfinished = [], []
    for path in sorted(RESULTS.glob("*/run.json")):
        try:
            run = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # A partial/corrupt record cannot prove completion.
        if run.get("seed") != seed or run.get("variant") != method:
            continue
        command = run.get("command", [])
        preset = "g1-29dof-fast-sac" + ("-sem" if method == "sem" else "")
        expected = {
            "--training.num-envs": "4096", "--training.seed": str(seed),
            "--terrain.terrain-term.scale-factor": "1.0",
            "--algo.config.num-learning-iterations": str(ARGS.iterations),
            "--algo.config.logging-interval": str(ARGS.logging_interval),
        }
        if not (run.get("smoke_test") is False and run.get("budget_s") == ARGS.seconds
                and run.get("protocol") == PROTOCOL and run.get("timestamps_s") == MARKS
                and run.get("training_source_sha256") == TRAINING_SOURCE
                and run.get("timestamp_window_s") == ARGS.timestamp_window_seconds
                and str(run.get("base_revision", "")).startswith("9c238cf8")
                and run.get("simulator") == "Isaac Gym Preview 4"
                and run.get("torch_compile_cache") == "cold"
                and run.get("simulator_cache") == "shared"
                and "exp:" + preset in command and "simulator:isaacgym" in command
                and all(argument(command, key) == value for key, value in expected.items())
                and argument(command, "--algo.config.compile") in (None, "True")
                and "--algo.config.no-compile" not in command):
            continue  # Smoke tests or different settings do not fill this pair.
        try:
            if run.get("status") != "finished" or run.get("return_code") != 0:
                raise ValueError("failed/interrupted")
            timing_files = list(path.parent.glob("training/*/*/timing.json"))
            if len(timing_files) != 1:
                raise ValueError("missing/ambiguous timing.json")
            timing = json.loads(timing_files[0].read_text())
            elapsed = float(timing["training_elapsed_wall_time_s"])
            step = timing["global_step"]
            # Subtracting monotonic timestamps can yield 900.0000000000009.
            # Allow one microsecond of representation error, not a shorter run.
            budget_matches = math.isclose(float(timing["budget_s"]), ARGS.seconds,
                                          rel_tol=0.0, abs_tol=1e-6)
            if (not budget_matches or not math.isfinite(elapsed)
                    or elapsed < ARGS.seconds or not isinstance(step, int) or step <= 10):
                raise ValueError("incomplete budget or missing training metrics")
            training_dir = timing_files[0].parent
            schedule = json.loads((training_dir / "timestamps.json").read_text())
            if not schedule.get("finished") or schedule.get("timestamps_s") != MARKS:
                raise ValueError("incomplete timestamp manifest")
            captured = {r["requested_elapsed_s"]: r for r in schedule["records"]}
            for record in schedule.get("pre_target_records", []):
                for key in ("checkpoint", "onnx"):
                    artifact = training_dir / record[key]
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        raise ValueError("missing pre-target " + key)
            for mark in MARKS:
                record = captured.get(mark)
                if record is None or not record.get("observation_available"):
                    raise ValueError("missing observation at requested timestamp %gs" % mark)
                for key in ("checkpoint", "onnx"):
                    artifact = training_dir / record[key]
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        raise ValueError("missing timestamp " + key)
            for extension in ("pt", "onnx"):
                artifact = training_dir / ("model_%07d.%s" % (step, extension))
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError("missing final " + extension)
            if not any(event.stat().st_size > 88
                       for event in training_dir.glob("events.out.tfevents.*")):
                raise ValueError("missing TensorBoard events")
        except (OSError, ValueError, KeyError, TypeError) as error:
            unfinished.append("%s (%s)" % (path.parent.name, error))
        else:
            complete.append(path.parent.name)
    return complete, unfinished


def stop_group(process):
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main():
    print("Results: %s; budget=%gs; timestamps=%s" % (RESULTS, ARGS.seconds, MARKS), flush=True)
    jobs = active_jobs()
    for pid, command in jobs:
        print("ACTIVE pid=%s: %s" % (pid, command), flush=True)
    if jobs and not STATUS_ONLY:
        print("Training is already active. Rerun after it finishes.", flush=True)
        return 1
    for seed, method in PLAN:
        complete, unfinished = classify(seed, method)
        print("seed=%s method=%s: %s" % (
            seed, method, "COMPLETE (skip)" if complete else "PENDING (fresh attempt)"), flush=True)
        for name in complete:
            print("  completed: " + name, flush=True)
        for name in unfinished:
            print("  unfinished: " + name, flush=True)
        if len(complete) > 1:
            print("  Multiple completed runs: select explicitly when plotting.", flush=True)
    if STATUS_ONLY:
        return 0

    signal.signal(signal.SIGTERM, interrupted)
    for seed, method in PLAN:
        if classify(seed, method)[0]:
            continue
        if active_jobs():
            print("Another trainer appeared; stopping to avoid concurrent runs.", flush=True)
            return 1
        print("\nSTART seed=%s method=%s at %s" % (
            seed, method, datetime.now(timezone.utc).isoformat()), flush=True)
        process = subprocess.Popen([
            sys.executable, str(REPO / "scripts/run_sem_experiment.py"),
            "--variant", method, "--seed", str(seed), "--seconds", str(ARGS.seconds),
            "--output-root", str(RESULTS), "--iterations", str(ARGS.iterations),
            "--logging-interval", str(ARGS.logging_interval),
            "--timestamp-window-seconds", str(ARGS.timestamp_window_seconds),
            "--timestamps-seconds", *[str(mark) for mark in MARKS],
        ], cwd=str(REPO), start_new_session=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            stop_group(process)
            raise
        if code != 0:
            print("Run failed. Inspect console.log, fix the cause, then rerun this script.", flush=True)
            return code if code > 0 else 1
        complete, unfinished = classify(seed, method)
        if not complete:
            print("Run exited successfully but outputs/budget are incomplete; stopping.", flush=True)
            for reason in unfinished:
                print("  " + reason, flush=True)
            return 1
    print("All %d seed/method combinations complete. Perform the final sync before stopping EC2." % len(PLAN), flush=True)
    return 0


try:
    sys.exit(main())
except KeyboardInterrupt:
    print("\nBatch interrupted. Rerun this script to skip completed pairs and retry unfinished ones.", flush=True)
    sys.exit(130)
PY
