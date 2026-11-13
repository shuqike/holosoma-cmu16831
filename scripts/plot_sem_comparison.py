"""Plot matched FastSAC/SEM training runs without importing Holosoma or loading policies.

Dependencies: numpy, matplotlib, tensorboard, PyYAML. Default input: aws-results.
Only successful full runs are selected. Duplicate successful runs require explicit exclusion.
All data are clipped to the training budget; aggregation never extrapolates outside observed data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
    if float(metadata["budget_s"]) != budget:
        raise ValueError(f"{path.name}: budget differs from requested {budget:g} seconds")
    if not list(configs[0].parent.glob("events.out.tfevents.*")):
        raise ValueError(f"{path.name}: TensorBoard events have not been synced")
    events = EventAccumulator(str(configs[0].parent), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags().get("scalars", []))
    samples = {e.step: e.value for e in events.Scalars("Train/num_samples")}
    timing = (
        {e.step: e.value for e in events.Scalars("Timing/elapsed_wall_time_s")}
        if "Timing/elapsed_wall_time_s" in tags else {}
    )
    dt = config["simulator"]["config"]["sim"]["control_decimation"] / config["simulator"]["config"]["sim"]["fps"]
    series = {}
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
            if 0 <= elapsed <= budget:
                points[event.step] = row
        array = np.asarray([points[s] for s in sorted(points)], dtype=float)
        if array.shape[0] < 2:
            raise ValueError(f"{path.name}: fewer than two in-budget points for {tag}")
        if np.any(np.diff(array[:, 1]) <= 0) or np.any(np.diff(array[:, 2]) <= 0):
            raise ValueError(f"{path.name}: non-monotonic elapsed time or sample counts")
        series[name] = array
    return Run(path, metadata, config, series)


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
    for meta_path in sorted(root.glob("*/run.json")):
        if meta_path.parent.name in excluded:
            continue
        metadata = json.loads(meta_path.read_text())
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
        for key in ("base_revision", "python", "torch_compile_cache", "simulator_cache", "simulator"):
            if run.metadata.get(key) != runs[0].metadata.get(key):
                raise ValueError(f"{run.path.name}: run metadata differs for {key}")
    return runs


def align(runs: list, metric: str, column: int) -> tuple:
    """Use only the interval observed in every selected run; never pad missing endpoints."""
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


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(runs: list, seeds: list, output: Path, budget: float, resamples: int) -> None:
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
            grid, matrices = align(runs, metric, column)
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
                integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                auc = integrate(values, grid, axis=1) / (grid[-1] - grid[0])
                endpoint_stats = estimate(values[:, -1], indices)
                auc_stats = estimate(auc, indices)
                summaries.append(dict(
                    metric=metric, axis=axis_name, method=method, n_seeds=n,
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
            area_delta = estimate(integrate(delta, grid, axis=1) / (grid[-1] - grid[0]), indices)
            paired.append(dict(
                metric=metric, axis=axis_name, n_pairs=n, interval_start=grid[0], interval_end=grid[-1],
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
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
        interval_label = "95% pointwise seed-bootstrap intervals" if n > 1 else "one pair; no uncertainty estimate"
        fig.suptitle(f"FastSAC vs FastSAC + SEM | {n} matched seeds\n{interval_label}; thin lines: individual runs")
        save_figure(fig, output, f"comparison_{axis_name}")
    difference_fig.suptitle("Paired linear tracking difference | positive values favor SEM")
    save_figure(difference_fig, output, "paired_linear_difference")
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_summary.csv", paired)
    write_csv(output / "aligned_metrics.csv", aligned_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("aws-results"))
    parser.add_argument("--output", type=Path, default=Path("aws-results/comparison"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--budget", type=float, default=900)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--exclude-run", action="append", default=[], metavar="DIRECTORY_NAME")
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if len(seeds) != len(args.seeds) or args.bootstrap_samples < 100 or not np.isfinite(args.budget) or args.budget <= 0:
        parser.error("Use unique seeds, a positive finite budget, and at least 100 bootstrap samples")
    try:
        runs = select_runs(args.input, seeds, args.exclude_run, args.budget)
        if len(seeds) < 5:
            print("Exploratory comparison: fewer than five matched seeds.", file=sys.stderr)
        args.output.mkdir(parents=True, exist_ok=True)
        make_plots(runs, seeds, args.output, args.budget, args.bootstrap_samples)
        raw_rows = []
        for run in runs:
            for metric, values in run.series.items():
                for step, elapsed, samples, value in values:
                    raw_rows.append(dict(
                        run=run.path.name, method=run.metadata["variant"], seed=run.metadata["seed"],
                        metric=metric, step=int(step), elapsed_s=elapsed, samples=samples, value=value,
                    ))
        write_csv(args.output / "raw_metrics.csv", raw_rows)
        manifest = {
            "seeds": seeds, "budget_s": args.budget, "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_rng_seed": 0, "runs": [dict(path=str(run.path.resolve()), **run.metadata) for run in runs],
            "notes": [
                "Training metrics, not independent policy evaluation.",
                "Only observed points at or before the budget are included; no extrapolation to 900 seconds.",
                "Means and paired differences use the interval observed in every selected run for that metric/axis.",
                "Normalized AUC is the integral divided by interval length; initial startup gaps are not filled.",
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
