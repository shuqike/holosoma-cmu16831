"""Launch a single-GPU FastSAC experiment with startup-inclusive timing and local logs.

Run with the activated Holosoma Isaac Gym Preview 4 environment (hsgym).
No simulator imports occur here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    from .sem_experiment_utils import (
        DEFAULT_ITERATIONS, PROTOCOL, atomic_json, default_output_root, timestamps, training_source_fingerprint,
    )
except ImportError:
    from sem_experiment_utils import (
        DEFAULT_ITERATIONS, PROTOCOL, atomic_json, default_output_root, timestamps, training_source_fingerprint,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sem", "baseline"), default="sem")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="Safety ceiling; normally the wall-clock budget stops training")
    parser.add_argument("--timestamps-seconds", type=float, nargs="+",
                        help="Capture marks; defaults to 900s when in range, plus the end")
    parser.add_argument("--timestamp-window-seconds", type=float, default=5.0,
                        help="Log every completed iteration in this window before each mark")
    parser.add_argument("--logging-interval", type=int, default=100)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.num_envs <= 0 or args.iterations <= 0 or args.logging_interval <= 0:
        parser.error("--num-envs, --iterations, and --logging-interval must be positive")
    if not math.isfinite(args.timestamp_window_seconds) or args.timestamp_window_seconds < 0:
        parser.error("--timestamp-window-seconds must be finite and nonnegative")
    try:
        marks = timestamps(args.seconds, args.timestamps_seconds)
    except ValueError as error:
        parser.error(str(error))
    args.output_root = args.output_root or default_output_root(args.seconds)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run one experiment on one GPU; do not launch with torchrun")

    repo = Path(__file__).resolve().parents[1]
    base_revision = subprocess.run(
        ["git", "rev-parse", "9c238cf8"], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", base_revision, "HEAD"], cwd=repo)
    if ancestor.returncode != 0:
        parser.error("The experiment must be based on teammate revision 9c238cf8")
    if not (repo / "scripts/setup_isaacgym.sh").is_file():
        parser.error("This checkout does not contain the required Isaac Gym backend")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (args.output_root / f"{stamp}_{args.variant}_seed{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    preset = "g1-29dof-fast-sac-sem" if args.variant == "sem" else "g1-29dof-fast-sac"
    command = [
        sys.executable, "-u", str(repo / "src/holosoma/holosoma/train_agent.py"),
        f"exp:{preset}", "simulator:isaacgym", "logger:disabled",
        "--logger.base-dir", str(run_dir / "training"),
        "--training.seed", str(args.seed), "--training.num-envs", str(args.num_envs),
        "--training.project", "cmu16831-sem", "--training.name", f"{args.variant}_seed{args.seed}",
        "--terrain.terrain-term.scale-factor", "1.0",
        "--algo.config.num-learning-iterations", str(args.iterations),
        "--algo.config.save-interval", "1000", "--algo.config.logging-interval", str(args.logging_interval),
    ]
    if args.no_compile:
        # 9c238cf8 uses tyro.conf.FlagConversionOff: booleans require an explicit value.
        command.extend(["--algo.config.compile", "False"])

    # Record both the base revision and local changes: uploading need not require a commit/push.
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    patch = subprocess.run(["git", "diff", base_revision, "--binary"], cwd=repo, capture_output=True, check=True)
    (run_dir / "source.patch").write_bytes(patch.stdout)
    # Also capture untracked experiment files, which git diff does not include.
    (run_dir / "launcher.py").write_bytes(Path(__file__).read_bytes())
    for source_dir in ("scripts", "docs", "src/holosoma/holosoma/agents/fast_sac/tests"):
        for source in (repo / source_dir).glob("*sem*"):
            if source.is_file():
                target = run_dir / "source" / source.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    source = repo / "src/holosoma/holosoma/agents/fast_sac/timed_experiment.py"
    target = run_dir / "source" / source.relative_to(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    (run_dir / "nvidia-smi.txt").write_text(gpu.stdout + gpu.stderr)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Every run starts with a fresh Torch/Triton cache; simulator caches remain shared.
    env["TORCHINDUCTOR_CACHE_DIR"] = str(run_dir / ".cache/torchinductor")
    env["TRITON_CACHE_DIR"] = str(run_dir / ".cache/triton")
    metadata = {
        "revision": revision.stdout.strip(), "variant": args.variant, "seed": args.seed,
        "base_revision": base_revision, "simulator": "Isaac Gym Preview 4",
        "budget_s": args.seconds, "command": command, "python": sys.version,
        "torch_compile_cache": "cold", "simulator_cache": "shared",
        "smoke_test": args.smoke_test or args.num_envs != 4096 or args.no_compile,
        "protocol": PROTOCOL, "timestamps_s": marks,
        "training_source_sha256": training_source_fingerprint(repo),
        "timestamp_window_s": args.timestamp_window_seconds,
        "logging_interval": args.logging_interval, "iteration_limit": args.iterations,
        "timing": "training interpreter launch through cooperative stop; cleanup/export timed separately by duration",
        "status": "running",
    }
    print(f"Outputs: {run_dir}", flush=True)
    started = time.monotonic()
    metadata["started_unix_s"] = time.time()
    env["HOLOSOMA_RUN_STARTED_MONOTONIC"] = str(started)
    env["HOLOSOMA_RUN_DEADLINE_MONOTONIC"] = str(started + args.seconds)
    env["HOLOSOMA_RUN_BUDGET_SECONDS"] = str(args.seconds)
    env["HOLOSOMA_TIMESTAMPS_SECONDS"] = json.dumps(marks)
    env["HOLOSOMA_TIMESTAMP_WINDOW_SECONDS"] = str(args.timestamp_window_seconds)
    atomic_json(run_dir / "run.json", metadata)
    with (run_dir / "console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            return_code = process.wait()
    metadata.update(
        status="finished" if return_code == 0 else "failed",
        return_code=return_code,
        process_duration_s=time.monotonic() - started,
    )
    atomic_json(run_dir / "run.json", metadata)
    print(f"Exit code {return_code}; outputs: {run_dir}", flush=True)
    if return_code != 0:
        print("Last 60 lines of console.log:", flush=True)
        with (run_dir / "console.log").open(errors="replace") as log:
            print("".join(deque(log, maxlen=60)), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
