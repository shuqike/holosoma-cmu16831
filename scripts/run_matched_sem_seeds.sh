#!/usr/bin/env bash
# EC2, inside tmux: bash scripts/run_matched_sem_seeds.sh [--status]
# Completed pairs are skipped; interrupted attempts restart with a fresh budget.
set -eo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
mode=run
if [[ ${1:-} == --status && $# == 1 ]]; then
  mode=status
elif [[ $# != 0 ]]; then
  echo 'Usage: bash scripts/run_matched_sem_seeds.sh [--status]' >&2
  exit 2
fi
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

# Bundled standard-library Python keeps this a single transferable script.
python3 - "$repo_dir" "$mode" <<'PY'
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
STATUS_ONLY = sys.argv[2] == "status"
RESULTS = REPO / "logs/sem_experiments"
PLAN = [(seed, method) for seed in (2, 3, 4, 5)
        for method in (("baseline", "sem") if seed % 2 == 0 else ("sem", "baseline"))]


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
            "--algo.config.num-learning-iterations": "50000",
        }
        if not (run.get("smoke_test") is False and run.get("budget_s") == 900
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
            if (timing.get("budget_s") != 900 or not math.isfinite(elapsed)
                    or elapsed < 900 or not isinstance(step, int) or step < 100):
                raise ValueError("incomplete budget or missing training metrics")
            training_dir = timing_files[0].parent
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
            "--variant", method, "--seed", str(seed), "--seconds", "900",
        ], cwd=str(REPO), start_new_session=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            stop_group(process)
            raise
        if code != 0:
            print("Run failed. Inspect console.log, fix the cause, then rerun this script.", flush=True)
            return code if code > 0 else 1
        if not classify(seed, method)[0]:
            print("Run exited successfully but outputs/budget are incomplete; stopping.", flush=True)
            return 1
    print("All eight pairs complete. Perform the final local sync before stopping EC2.", flush=True)
    return 0


try:
    sys.exit(main())
except KeyboardInterrupt:
    print("\nBatch interrupted. Rerun this script to skip completed pairs and retry unfinished ones.", flush=True)
    sys.exit(130)
PY
