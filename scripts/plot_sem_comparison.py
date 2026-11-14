"""Plot matched FastSAC/SEM training runs without importing Holosoma or loading policies.

Dependencies: numpy, matplotlib, tensorboard, PyYAML.
Input is inferred from --budget under the repository's aws-results directory.
Only successful full runs are selected. Duplicate successful runs require explicit exclusion.
All data are clipped to the training budget; aggregation never extrapolates outside observed data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    from .sem_experiment_utils import discover_results_root, timestamps
    from .sem_postprocessing import interval_mean, trailing_window
except ImportError:
    from sem_experiment_utils import discover_results_root, timestamps
    from sem_postprocessing import interval_mean, trailing_window

METHODS = ("baseline", "sem")
LABELS = {"baseline": "FastSAC", "sem": "FastSAC + SEM"}
COLORS = {"baseline": "#2878b5", "sem": "#d95f02"}
METRICS = {
    "linear_tracking": ("Episode/rew_tracking_lin_vel", "Linear velocity tracking reward"),
    "angular_tracking": ("Episode/rew_tracking_ang_vel", "Angular velocity tracking reward"),
    "episode_duration": ("Train/mean_episode_length", "Mean episode duration (s)"),
    "episode_reward": ("Train/mean_reward", "Mean episode reward"),
}


@dataclass
class Run:
    path: Path
    metadata: dict
    config: dict
    # For each metric: columns step, elapsed seconds, logged environment samples, metric value.
    series: dict
    all_series: dict = None
    scalars: dict = None


def read_run(path: Path, metadata: dict, budget: float) -> Run:
    configs = list(path.rglob("holosoma_config.yaml"))
    if len(configs) != 1:
        raise ValueError(f"{path.name}: expected one synced training config; found {len(configs)}")
    config = yaml.safe_load(configs[0].read_text())
    method = metadata["variant"]
    if bool(config["algo"]["config"].get("use_actor_sem", False)) != (method == "sem"):
        raise ValueError(f"{path.name}: method label and saved SEM setting disagree")
    if config["training"]["seed"] != metadata["seed"] or config["training"]["checkpoint"] is not None:
        raise ValueError(f"{path.name}: seed mismatch or resumed training")
    if config["training"]["num_envs"] != 4096 or config["simulator"]["config"]["name"] != "isaacgym":
        raise ValueError(f"{path.name}: not the requested 4096-environment Isaac Gym experiment")
    if not math.isclose(float(metadata["budget_s"]), budget, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{path.name}: budget differs from requested {budget:g} seconds")
    if not list(configs[0].parent.glob("events.out.tfevents.*")):
        raise ValueError(f"{path.name}: TensorBoard events have not been synced")
    events = EventAccumulator(str(configs[0].parent), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags().get("scalars", []))
    all_scalars = {tag: {event.step: event.value for event in events.Scalars(tag)} for tag in tags}
    samples = {e.step: e.value for e in events.Scalars("Train/num_samples")}
    timing = (
        {e.step: e.value for e in events.Scalars("Timing/elapsed_wall_time_s")}
        if "Timing/elapsed_wall_time_s" in tags else {}
    )
    dt = config["simulator"]["config"]["sim"]["control_decimation"] / config["simulator"]["config"]["sim"]["fps"]
    series, all_series = {}, {}
    for name, (tag, _) in METRICS.items():
        if tag not in tags:
            raise ValueError(f"{path.name}: missing metric {tag}")
        points = {}
        for event in events.Scalars(tag):
            elapsed = timing.get(event.step, event.wall_time - metadata["started_unix_s"])
            value = event.value * dt if name == "episode_duration" else event.value
            row = (event.step, elapsed, samples.get(event.step, np.nan), value)
            if not np.all(np.isfinite(row)):
                raise ValueError(f"{path.name}: non-finite or unaligned {tag} at step {event.step}")
            if elapsed >= 0:
                points[event.step] = row
        array = np.asarray([points[s] for s in sorted(points)], dtype=float).reshape(-1, 4)
        all_series[name] = array
        array = array[array[:, 1] <= budget]
        if array.shape[0] < 2:
            raise ValueError(f"{path.name}: fewer than two in-budget points for {tag}")
        if np.any(np.diff(array[:, 1]) <= 0) or np.any(np.diff(array[:, 2]) <= 0):
            raise ValueError(f"{path.name}: non-monotonic elapsed time or sample counts")
        series[name] = array
    timing_paths = list(configs[0].parent.glob("timing.json"))
    if len(timing_paths) != 1:
        raise ValueError(f"{path.name}: missing timing.json")
    timing_record = json.loads(timing_paths[0].read_text())
    elapsed = float(timing_record["training_elapsed_wall_time_s"])
    if not math.isfinite(elapsed) or elapsed + 1e-6 < budget:
        raise ValueError(f"{path.name}: training stopped before its requested budget")
    return Run(path, metadata, config, series, all_series, all_scalars)


def comparison_config(run: Run) -> dict:
    """Ignore run identity and the intended SEM intervention, not task/optimizer differences."""
    config = copy.deepcopy(run.config)
    for key in ("seed", "name"):
        config["training"].pop(key, None)
    config["logger"].pop("base_dir", None)
    config["algo"]["config"].pop("use_actor_sem", None)
    return config


def select_runs(root: Path, seeds: list, excluded: list, budget: float) -> list:
    selected = {}
    for meta_path in sorted(root.rglob("run.json")):
        if meta_path.parent.name in excluded:
            continue
        metadata = json.loads(meta_path.read_text())
        if not math.isclose(float(metadata.get("budget_s", 0)), budget, rel_tol=0, abs_tol=1e-6):
            continue
        if metadata.get("smoke_test", True) or metadata.get("variant") not in METHODS:
            continue
        if metadata.get("seed") not in seeds:
            continue
        if metadata.get("status") != "finished" or metadata.get("return_code") != 0:
            print(f"Skipping unfinished/failed run: {meta_path.parent.name}", file=sys.stderr)
            continue
        key = (metadata["variant"], metadata["seed"])
        if key in selected:
            raise ValueError(
                f"Duplicate successful runs for {key}: {selected[key].path.name}, {meta_path.parent.name}. "
                "Use --exclude-run DIRECTORY_NAME to make an explicit selection."
            )
        selected[key] = read_run(meta_path.parent, metadata, budget)
    missing = [(method, seed) for method in METHODS for seed in seeds if (method, seed) not in selected]
    if missing:
        raise ValueError(f"Missing successful full runs: {missing}. Wait for sync, or explicitly request fewer --seeds.")
    runs = [selected[(method, seed)] for method in METHODS for seed in seeds]
    reference = comparison_config(runs[0])
    for run in runs[1:]:
        if comparison_config(run) != reference:
            raise ValueError(f"{run.path.name}: saved task/optimizer configuration differs across runs")
        for key in ("base_revision", "python", "torch_compile_cache", "simulator_cache", "simulator",
                    "protocol", "timestamps_s", "timestamp_window_s", "logging_interval"):
            if run.metadata.get(key) != runs[0].metadata.get(key):
                raise ValueError(f"{run.path.name}: run metadata differs for {key}")
        if run.metadata.get("training_source_sha256") != runs[0].metadata.get("training_source_sha256"):
            raise ValueError(f"{run.path.name}: training source differs across runs")
    return runs


def align(runs: list, metric: str, column: int) -> tuple:
    """Use only the interval observed in every selected run; never pad missing endpoints."""
    if any(len(run.series[metric]) < 2 for run in runs):
        raise ValueError(f"Insufficient supported observations for {metric}")
    lower = max(run.series[metric][0, column] for run in runs)
    upper = min(run.series[metric][-1, column] for run in runs)
    if lower >= upper:
        raise ValueError(f"No shared observed interval for {metric}")
    grid = np.linspace(lower, upper, 200)
    matrices = {}
    for method in METHODS:
        matrices[method] = np.asarray([
            np.interp(grid, run.series[metric][:, column], run.series[metric][:, 3])
            for run in runs if run.metadata["variant"] == method
        ])
    return grid, matrices


def estimate(values: np.ndarray, indices: np.ndarray) -> tuple:
    mean = values.mean(axis=0)
    if values.shape[0] == 1:
        undefined = np.full_like(mean, np.nan, dtype=float)
        return mean, undefined, undefined
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    return mean, low, high


def save_figure(fig, output: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list, fieldnames=None) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rolling_runs(runs, window, max_lag, min_coverage, max_gap):
    """Evaluate trailing means at each run's actual observation times."""
    transformed, records = [], []
    for run in runs:
        series = {}
        for metric, raw in run.series.items():
            averaged = raw.copy()
            for index, row in enumerate(raw):
                result = trailing_window([raw], row[1], window, max_lag, min_coverage, max_gap)
                averaged[index, 3] = result.values[0]
                records.append(dict(
                    run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"],
                    metric=metric, step=int(row[0]), elapsed_s=float(row[1]), samples=float(row[2]),
                    window_s=window, interval_start_s=result.start, interval_end_s=result.end,
                    coverage_fraction=result.coverage, max_interpolation_gap_s=float(result.max_gaps[0]),
                    status=result.statuses[0], value=float(result.values[0]),
                ))
            finite = np.flatnonzero(np.isfinite(averaged[:, 3]))
            # Retain internal/trailing gaps as NaN, never interpolate across
            # missing rolling estimates after dropping them from the series.
            series[metric] = averaged[finite[0]:] if len(finite) else np.empty((0, 4))
        transformed.append(Run(run.path, run.metadata, run.config, series))
    return transformed, records


def make_plots(runs: list, seeds: list, output: Path, budget: float, resamples: int,
               marks=(), window=0.0, suffix="") -> None:
    n = len(seeds)
    # Reuse the same seed resampling for both methods and their paired difference.
    indices = np.random.default_rng(0).integers(0, n, size=(resamples, n))
    summaries, paired, aligned_rows = [], [], []
    difference_fig, difference_axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis_index, (axis_name, column, scale, xlabel) in enumerate((
        ("wall_time", 1, 60.0, "Elapsed time including startup (minutes)"),
        ("samples", 2, 1e6, "Logged environment transitions (millions)"),
    )):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, (metric, (_, ylabel)) in zip(axes.flat, METRICS.items()):
            statistic = "trailing_time_weighted_mean" if window else "raw_logged_statistic"
            try:
                grid, matrices = align(runs, metric, column)
            except ValueError:
                ax.text(0.5, 0.5, "Insufficient common window coverage", ha="center", transform=ax.transAxes)
                ax.set(xlabel=xlabel, ylabel=ylabel)
                if metric == "linear_tracking":
                    difference_axes[axis_index].text(0.5, 0.5, "Insufficient common window coverage",
                                                     ha="center", transform=difference_axes[axis_index].transAxes)
                for method in METHODS:
                    summaries.append(dict(
                        metric=metric, axis=axis_name, method=method, n_seeds=n,
                        statistic=statistic, window_s=window, status="incomplete",
                        interval_start=np.nan, interval_end=np.nan,
                        endpoint_mean=np.nan, endpoint_ci_low=np.nan, endpoint_ci_high=np.nan,
                        normalized_auc_mean=np.nan, normalized_auc_ci_low=np.nan, normalized_auc_ci_high=np.nan,
                    ))
                paired.append(dict(
                    metric=metric, axis=axis_name, n_pairs=n, statistic=statistic, window_s=window,
                    status="incomplete", interval_start=np.nan, interval_end=np.nan,
                    endpoint_sem_minus_baseline=np.nan, endpoint_ci_low=np.nan, endpoint_ci_high=np.nan,
                    normalized_auc_sem_minus_baseline=np.nan, normalized_auc_ci_low=np.nan, normalized_auc_ci_high=np.nan,
                ))
                continue
            method_aucs = {}
            for method in METHODS:
                for run in runs:
                    if run.metadata["variant"] == method:
                        raw = run.series[metric]
                        ax.plot(raw[:, column] / scale, raw[:, 3], color=COLORS[method], alpha=0.18, lw=0.8)
                values = matrices[method]
                mean, low, high = estimate(values, indices)
                ax.plot(grid / scale, mean, color=COLORS[method], label=LABELS[method], lw=2)
                if n > 1:
                    ax.fill_between(grid / scale, low, high, color=COLORS[method], alpha=0.2)
                # The 200-point display grid can miss dense bursts. Integrate
                # each complete source curve on its own original knots.
                auc = np.asarray([
                    interval_mean(run.series[metric][:, column], run.series[metric][:, 3], grid[0], grid[-1])
                    for run in runs if run.metadata["variant"] == method
                ])
                method_aucs[method] = auc
                endpoint_stats = estimate(values[:, -1], indices)
                auc_stats = estimate(auc, indices)
                summaries.append(dict(
                    metric=metric, axis=axis_name, method=method, n_seeds=n,
                    statistic=statistic, window_s=window,
                    status="available" if np.all(np.isfinite(values)) and np.all(np.isfinite(auc)) else "incomplete",
                    interval_start=grid[0], interval_end=grid[-1],
                    endpoint_mean=float(endpoint_stats[0]), endpoint_ci_low=float(endpoint_stats[1]),
                    endpoint_ci_high=float(endpoint_stats[2]), normalized_auc_mean=float(auc_stats[0]),
                    normalized_auc_ci_low=float(auc_stats[1]), normalized_auc_ci_high=float(auc_stats[2]),
                ))
                for seed, curve in zip(seeds, values):
                    for x, y in zip(grid, curve):
                        aligned_rows.append(dict(metric=metric, axis=axis_name, method=method, seed=seed, x=x, value=y))
            delta = matrices["sem"] - matrices["baseline"]
            endpoint_delta = estimate(delta[:, -1], indices)
            area_delta = estimate(method_aucs["sem"] - method_aucs["baseline"], indices)
            paired.append(dict(
                metric=metric, axis=axis_name, n_pairs=n, interval_start=grid[0], interval_end=grid[-1],
                statistic=statistic, window_s=window,
                status="available" if np.all(np.isfinite(delta)) and np.all(np.isfinite(method_aucs["sem"] - method_aucs["baseline"])) else "incomplete",
                endpoint_sem_minus_baseline=float(endpoint_delta[0]),
                endpoint_ci_low=float(endpoint_delta[1]), endpoint_ci_high=float(endpoint_delta[2]),
                normalized_auc_sem_minus_baseline=float(area_delta[0]),
                normalized_auc_ci_low=float(area_delta[1]), normalized_auc_ci_high=float(area_delta[2]),
            ))
            if metric == "linear_tracking":
                dax = difference_axes[axis_index]
                mean, low, high = estimate(delta, indices)
                for curve in delta:
                    dax.plot(grid / scale, curve, color="#555555", alpha=0.2, lw=0.8)
                dax.plot(grid / scale, mean, color="#333333", lw=2)
                if n > 1:
                    dax.fill_between(grid / scale, low, high, color="#555555", alpha=0.2)
                dax.axhline(0, color="black", lw=0.8, ls="--")
                dax.set(xlabel=xlabel, ylabel="Linear tracking: SEM minus FastSAC")
                dax.grid(alpha=0.25)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            if axis_name == "wall_time":
                ax.set_xlim(0, budget / 60)
                for mark in marks:
                    ax.axvline(mark / 60, color="gray", lw=0.6, ls=":", alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
        interval_label = "95% pointwise seed-bootstrap intervals" if n > 1 else "one pair; no uncertainty estimate"
        description = f"{window:g}s time-weighted trailing means" if window else "RAW diagnostics; averaging windows vary near capture marks"
        fig.suptitle(f"FastSAC vs FastSAC + SEM | {n} matched seeds | {description}\n{interval_label}; thin lines: individual runs")
        save_figure(fig, output, f"comparison_{axis_name}{suffix}")
    difference_fig.suptitle(f"Paired linear tracking difference | {'trailing means' if window else 'raw diagnostics'} | positive favors SEM")
    save_figure(difference_fig, output, "paired_linear_difference" + suffix)
    write_csv(output / ("summary" + suffix + ".csv"), summaries)
    write_csv(output / ("paired_summary" + suffix + ".csv"), paired)
    write_csv(output / ("aligned_metrics" + suffix + ".csv"), aligned_rows,
              ["metric", "axis", "method", "seed", "x", "value"])


def at_or_before(run: Run, metric: str, target: float, max_lag: float) -> tuple:
    """Use an actual in-budget observation, never a future value or extrapolation."""
    values = run.series[metric]
    eligible = values[values[:, 1] <= target]
    if len(eligible) == 0:
        return None, "missing"
    row = eligible[-1]
    return row, "available" if target - row[1] <= max_lag + 1e-6 else "stale"


def make_timestamp_tables(runs, seeds, output, marks, max_lag, resamples,
                          window=60.0, sensitivity=(30.0, 120.0), min_coverage=0.95, max_gap=30.0):
    indices = np.random.default_rng(0).integers(0, len(seeds), size=(resamples, len(seeds)))
    observations, summaries, paired, diagnostics, checkpoints, last_observations = [], [], [], [], [], []
    windows = sorted(set([window] + list(sensitivity)))
    for target in marks:
        for metric in METRICS:
            for run in runs:
                row, status = at_or_before(run, metric, target, max_lag)
                last_observations.append(dict(
                    run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"], metric=metric,
                    requested_elapsed_s=target, statistic="last_observation", status=status,
                    observed_elapsed_s=float(row[1]) if row is not None else float("nan"),
                    lag_s=target - float(row[1]) if row is not None else float("nan"),
                    step=int(row[0]) if row is not None else -1,
                    samples=float(row[2]) if row is not None else float("nan"),
                    value=float(row[3]) if row is not None else float("nan"),
                ))
            for width in windows:
                result = trailing_window([run.series[metric] for run in runs], target, width,
                                         max_lag, min_coverage, max_gap)
                values = {method: {} for method in METHODS}
                interval = dict(
                    statistic="time_weighted_logged_trajectory", window_s=width,
                    requested_elapsed_s=target, requested_start_s=max(0.0, target - width),
                    interval_start_s=result.start, interval_end_s=result.end,
                    coverage_fraction=result.coverage, end_lag_s=result.end_lag,
                )
                for index, run in enumerate(runs):
                    method, seed = run.metadata["variant"], run.metadata["seed"]
                    observations.append(dict(
                        run=run.path.name, method=method, seed=seed, metric=metric,
                        **interval, status=result.statuses[index],
                        n_observations=int(result.counts[index]),
                        max_interpolation_gap_s=float(result.max_gaps[index]),
                        value=float(result.values[index]),
                    ))
                    values[method][seed] = result.values[index]
                for method in METHODS:
                    array = np.asarray([values[method][seed] for seed in seeds])
                    available = int(np.isfinite(array).sum())
                    stats = estimate(array, indices) if available == len(seeds) else (np.nan,) * 3
                    summaries.append(dict(
                        metric=metric, method=method, **interval,
                        n_requested=len(seeds), n_available=available,
                        status="available" if available == len(seeds) else "incomplete",
                        mean=float(stats[0]), ci_low=float(stats[1]), ci_high=float(stats[2]),
                    ))
                delta = np.asarray([values["sem"][seed] - values["baseline"][seed] for seed in seeds])
                available = int(np.isfinite(delta).sum())
                stats = estimate(delta, indices) if available == len(seeds) else (np.nan,) * 3
                paired.append(dict(
                    metric=metric, **interval, n_requested=len(seeds), n_available=available,
                    status="available" if available == len(seeds) else "incomplete",
                    sem_minus_baseline=float(stats[0]), ci_low=float(stats[1]), ci_high=float(stats[2]),
                ))
        for run in runs:
            row, status = at_or_before(run, "linear_tracking", target, max_lag)
            if row is not None:
                for tag, by_step in (run.scalars or {}).items():
                    if int(row[0]) in by_step:
                        diagnostics.append(dict(
                            run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"],
                            requested_elapsed_s=target, observed_elapsed_s=float(row[1]),
                            lag_s=target - float(row[1]), status=status,
                            tag=tag, value=by_step[int(row[0])],
                        ))
            manifests = list(run.path.rglob("timestamps.json"))
            record = None
            pre_record = None
            if len(manifests) == 1:
                manifest = json.loads(manifests[0].read_text())
                record = next((r for r in manifest["records"]
                               if math.isclose(r["requested_elapsed_s"], target, rel_tol=0, abs_tol=1e-6)), None)
                pre_record = next((r for r in manifest.get("pre_target_records", [])
                                   if math.isclose(r["requested_elapsed_s"], target, rel_tol=0, abs_tol=1e-6)), None)
            checkpoints.append(dict(
                run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"],
                requested_elapsed_s=target, status="collected" if record is not None else "not_collected",
                checkpoint=record["checkpoint"] if record is not None else "",
                onnx=record["onnx"] if record is not None else "",
                checkpoint_elapsed_s=record["checkpoint_elapsed_wall_time_s"] if record is not None else float("nan"),
                checkpoint_offset_s=record["checkpoint_offset_s"] if record is not None else float("nan"),
                within_requested_time=record["checkpoint_within_requested_time"] if record is not None else "",
                pre_target_checkpoint=pre_record["checkpoint"] if pre_record is not None else "",
                pre_target_onnx=pre_record["onnx"] if pre_record is not None else "",
                pre_target_elapsed_s=pre_record["checkpoint_elapsed_wall_time_s"] if pre_record is not None else float("nan"),
                pre_target_within_requested_time=pre_record["checkpoint_within_requested_time"] if pre_record is not None else "",
            ))
    write_csv(output / "timestamp_metrics.csv", [r for r in observations if r["window_s"] == window])
    write_csv(output / "timestamp_summary.csv", [r for r in summaries if r["window_s"] == window])
    write_csv(output / "paired_timestamp_summary.csv", [r for r in paired if r["window_s"] == window])
    write_csv(output / "timestamp_window_sensitivity_metrics.csv", observations)
    write_csv(output / "timestamp_window_sensitivity_summary.csv", summaries)
    write_csv(output / "paired_timestamp_window_sensitivity.csv", paired)
    write_csv(output / "timestamp_last_observations.csv", last_observations)
    write_csv(output / "timestamp_checkpoints.csv", checkpoints)
    if diagnostics:
        write_csv(output / "timestamp_scalars.csv", diagnostics)
    if any(row["status"] == "incomplete" and row["window_s"] == window for row in paired):
        print("Some primary windows have insufficient support; see timestamp_metrics.csv. No incomplete aggregate was estimated.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--budget", type=float, default=900,
                        help="Run duration in seconds; selects aws-results/budget_<budget>s (default: 900)")
    parser.add_argument("--timestamps-seconds", type=float, nargs="+")
    parser.add_argument("--timestamp-max-lag-seconds", type=float, default=30.0,
                        help="Maximum age of an actual observation at a requested timestamp")
    parser.add_argument("--summary-window-seconds", type=float, default=60.0,
                        help="Primary time-weighted trailing window for curves and timestamp summaries")
    parser.add_argument("--sensitivity-windows-seconds", type=float, nargs="*", default=[30.0, 120.0],
                        help="Additional timestamp windows; default: 30 and 120 seconds")
    parser.add_argument("--min-window-coverage", type=float, default=0.95,
                        help="Required fraction of the requested window supported by observations")
    parser.add_argument("--max-interpolation-gap-seconds", type=float, default=30.0,
                        help="Reject windows needing interpolation across wider observation gaps")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--exclude-run", action="append", default=[], metavar="DIRECTORY_NAME")
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if len(seeds) != len(args.seeds) or args.bootstrap_samples < 100 or not np.isfinite(args.budget) or args.budget <= 0:
        parser.error("Use unique seeds, a positive finite budget, and at least 100 bootstrap samples")
    if not np.isfinite(args.timestamp_max_lag_seconds) or args.timestamp_max_lag_seconds < 0:
        parser.error("--timestamp-max-lag-seconds must be finite and nonnegative")
    widths = [args.summary_window_seconds] + args.sensitivity_windows_seconds
    if any(not np.isfinite(width) or width <= 0 for width in widths):
        parser.error("Summary/sensitivity windows must be finite and positive")
    if not np.isfinite(args.min_window_coverage) or not 0 < args.min_window_coverage <= 1:
        parser.error("--min-window-coverage must be in (0, 1]")
    if not np.isfinite(args.max_interpolation_gap_seconds) or args.max_interpolation_gap_seconds <= 0:
        parser.error("--max-interpolation-gap-seconds must be finite and positive")
    try:
        marks = timestamps(args.budget, args.timestamps_seconds)
        results_root = Path(__file__).resolve().parents[1] / "aws-results"
        input_dir = discover_results_root(results_root, args.budget, seeds, args.exclude_run)
        args.output = args.output or input_dir / "comparison"
        print(f"Using results: {input_dir.resolve()}")
        runs = select_runs(input_dir, seeds, args.exclude_run, args.budget)
        if len(seeds) < 5:
            print("Exploratory comparison: fewer than five matched seeds.", file=sys.stderr)
        args.output.mkdir(parents=True, exist_ok=True)
        make_plots(runs, seeds, args.output, args.budget, args.bootstrap_samples, marks, suffix="_raw")
        averaged, window_rows = rolling_runs(runs, args.summary_window_seconds, args.timestamp_max_lag_seconds,
                                             args.min_window_coverage, args.max_interpolation_gap_seconds)
        make_plots(averaged, seeds, args.output, args.budget, args.bootstrap_samples, marks,
                   window=args.summary_window_seconds)
        write_csv(args.output / "windowed_metrics.csv", window_rows)
        make_timestamp_tables(
            runs, seeds, args.output, marks, args.timestamp_max_lag_seconds, args.bootstrap_samples,
            args.summary_window_seconds, args.sensitivity_windows_seconds,
            args.min_window_coverage, args.max_interpolation_gap_seconds,
        )
        raw_rows = []
        for run in runs:
            for metric, values in run.series.items():
                for step, elapsed, samples, value in values:
                    raw_rows.append(dict(
                        run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"],
                        metric=metric, step=int(step), elapsed_s=elapsed, samples=samples, value=value,
                    ))
        write_csv(args.output / "raw_metrics.csv", raw_rows)
        all_rows = []
        for run in runs:
            for metric, values in run.all_series.items():
                for step, elapsed, samples, value in values:
                    all_rows.append(dict(
                        run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"],
                        metric=metric, step=int(step), elapsed_s=elapsed, samples=samples, value=value,
                        within_budget=elapsed <= args.budget,
                    ))
        write_csv(args.output / "all_metrics.csv", all_rows)
        manifest = {
            "seeds": seeds, "budget_s": args.budget, "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_rng_seed": 0, "runs": [dict(path=str(run.path.resolve()), **run.metadata) for run in runs],
            "timestamps_s": marks, "timestamp_max_lag_s": args.timestamp_max_lag_seconds,
            "input_directory": str(input_dir.resolve()),
            "postprocessing": "time-weighted-windows-v2",
            "summary_window_s": args.summary_window_seconds,
            "timestamp_windows_s": sorted(set(widths)),
            "min_window_coverage": args.min_window_coverage,
            "max_interpolation_gap_s": args.max_interpolation_gap_seconds,
            "analysis_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes() + Path(__file__).with_name("sem_postprocessing.py").read_bytes()
            ).hexdigest(),
            "notes": [
                "Training metrics, not independent policy evaluation.",
                "Curve aggregates include only observations at or before the selected budget; no endpoint extrapolation.",
                "Primary curves are time-weighted trailing means evaluated at each actual observation, then aligned for display.",
                "Timestamp summaries use one shared trailing interval across all requested runs, excluding every observation after the target.",
                "Window integrals use every original knot, with interpolation only at observed-supported boundaries; dense logs are not equally weighted votes.",
                "Missing/stale/low-coverage/large-gap windows remain unavailable; incomplete aggregates never drop seeds.",
                "Raw last observations and *_raw plots/summaries are diagnostic only; capture-time averaging windows vary.",
                "Logged means lack episode counts: time-weighted summaries do not reconstruct exact episode-weighted rewards.",
                "Sample-axis primary plots show the same time-window means against samples, not fixed-sample-window means.",
                "Post-budget final diagnostics are retained in all_metrics.csv, but do not enter in-budget comparisons.",
                "Means and paired differences use the interval observed in every selected run for that metric/axis.",
                "Normalized AUC is the integral divided by interval length; initial startup gaps are not filled.",
                "AUC is integrated on original source knots, independently of the display grid.",
                "Pointwise percentile bootstrap intervals resample matched seeds, not individual time points.",
                "Five seeds provide limited uncertainty estimation; intervals are not a significance test.",
                "Sample counts come from Holosoma's Train/num_samples logger, not iteration count.",
                "Hardware must be held fixed externally; verify nvidia-smi.txt for the selected runs.",
            ],
        }
        (args.output / "selected_runs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote PNG/PDF plots, CSV metrics, and run manifest to {args.output.resolve()}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(1, f"Comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
