"""Record timed observations and policy snapshots without assuming exact deadlines.

This module uses only the standard library. Snapshot/export callbacks belong to
the learner. All I/O overhead remains inside the running wall-clock budget.
"""

import json
import math
import os
import time
from pathlib import Path


def json_scalar(value, tag):
    """Convert Tensor/NumPy scalar values without importing either library."""
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "item", None)):
        try:
            value = value.item()
        except (ValueError, RuntimeError) as error:
            raise ValueError("Diagnostic %r must contain one scalar value" % tag) from error
    if not isinstance(value, (int, float)):
        raise TypeError("Diagnostic %r has unsupported scalar type %s" % (tag, type(value).__name__))
    if not math.isfinite(value):
        raise ValueError("Diagnostic %r must be finite" % tag)
    return value


class TimedExperimentRecorder:
    REQUIRED_SCALARS = ("Train/num_samples", "Train/mean_reward", "Train/mean_episode_length",
                        "Episode/rew_tracking_lin_vel", "Episode/rew_tracking_ang_vel")

    def __init__(self, log_dir, started, budget, marks, window=5.0):
        self.log_dir = Path(log_dir)
        self.started = started
        self.budget = budget
        self.marks = sorted(set(marks))
        self.window = window
        self.records = []
        self.pre_records = []
        self.latest = None
        self.latest_complete = None
        self.last_saved_step = None
        self.last_artifacts = None

    @classmethod
    def from_environment(cls, log_dir):
        text = os.environ.get("HOLOSOMA_TIMESTAMPS_SECONDS")
        if text is None:
            return None
        started = float(os.environ["HOLOSOMA_RUN_STARTED_MONOTONIC"])
        budget = float(os.environ["HOLOSOMA_RUN_BUDGET_SECONDS"])
        marks = json.loads(text)
        window = float(os.environ["HOLOSOMA_TIMESTAMP_WINDOW_SECONDS"])
        if (not math.isfinite(window) or window < 0 or not marks
                or any(not math.isfinite(t) or not 0 < t <= budget for t in marks)):
            raise ValueError("Invalid timed experiment schedule")
        return cls(log_dir, started, budget, marks, window)

    def elapsed(self):
        return time.monotonic() - self.started

    def due(self, elapsed):
        recorded = {r["requested_elapsed_s"] for r in self.records}
        return [t for t in self.marks if t not in recorded and elapsed >= t]

    def should_log(self, elapsed):
        recorded = {r["requested_elapsed_s"] for r in self.records}
        return any(t not in recorded and elapsed >= t - self.window for t in self.marks)

    def observe(self, step, elapsed, scalars):
        observation = {
            "step": int(step), "elapsed_wall_time_s": float(elapsed),
            "scalars": {tag: json_scalar(value, tag) for tag, value in scalars.items()},
        }
        # Serialize before opening the file, and retain only plain numbers in
        # memory so timestamp manifests cannot contain tensors or live graphs.
        line = json.dumps(observation, allow_nan=False) + "\n"
        # One line is one complete observation of all available diagnostics.
        with (self.log_dir / "observations.jsonl").open("a") as stream:
            stream.write(line)
        self.latest = observation
        if all(tag in observation["scalars"] for tag in self.REQUIRED_SCALARS):
            self.latest_complete = observation

    def publish(self, finished=False):
        recorded = {r["requested_elapsed_s"] for r in self.records}
        document = {
            "protocol": "timed-milestones-v1", "budget_s": self.budget,
            "timestamps_s": self.marks, "dense_window_s": self.window,
            "finished": finished, "records": self.records,
            "pre_target_records": self.pre_records,
            "unreached_timestamps_s": [t for t in self.marks if t not in recorded],
        }
        path = self.log_dir / "timestamps.json"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def capture_due(self, step, save, export):
        elapsed = self.elapsed()
        due = self.due(elapsed)
        prepared = {r["requested_elapsed_s"] for r in self.pre_records}
        before = [t for t in self.marks if t not in prepared and t - self.window <= elapsed < t]
        if not due and not before:
            return False
        # Several marks crossed during a long blocking call share one actual
        # policy. We report that fact rather than inventing intermediate states.
        if self.last_saved_step != step:
            checkpoint = self.log_dir / ("model_%07d.pt" % step)
            onnx = checkpoint.with_suffix(".onnx")
            metadata = save(str(checkpoint))
            export(str(onnx))
            self.last_saved_step = step
            self.last_artifacts = {
                "checkpoint": checkpoint.name, "onnx": onnx.name,
                "checkpoint_elapsed_wall_time_s": metadata["elapsed_wall_time_s"],
            }
        # A single snapshot near the start of the dense window gives strict
        # budget evaluations a nearby pre-target policy when work is timely.
        for target in before:
            snapshot_time = self.last_artifacts["checkpoint_elapsed_wall_time_s"]
            self.pre_records.append({
                "requested_elapsed_s": target, "step": step, **self.last_artifacts,
                "checkpoint_offset_s": snapshot_time - target,
                "checkpoint_within_requested_time": snapshot_time <= target,
            })
        for target in due:
            snapshot_time = self.last_artifacts["checkpoint_elapsed_wall_time_s"]
            scalars = self.latest["scalars"] if self.latest is not None else {}
            missing = [tag for tag in self.REQUIRED_SCALARS if tag not in scalars]
            self.records.append({
                "requested_elapsed_s": target, "step": step,
                **self.last_artifacts,
                "checkpoint_offset_s": snapshot_time - target,
                "checkpoint_within_requested_time": snapshot_time <= target,
                "latest_observation": self.latest,
                "latest_complete_observation": self.latest_complete,
                "observation_available": self.latest_complete is not None,
                "missing_required_scalars": missing,
            })
        self.publish()
        return True
