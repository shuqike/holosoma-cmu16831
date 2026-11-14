"""Checks for aggregation boundaries and paired seed uncertainty; no policy execution."""

from pathlib import Path

import numpy as np
import pytest

from scripts.plot_sem_comparison import Run, align, at_or_before, comparison_config, estimate, make_timestamp_tables, rolling_runs


def make_run(method, seed, start, end):
    config = {
        "training": {"seed": seed, "name": f"{method}_{seed}", "num_envs": 4096},
        "logger": {"base_dir": f"/tmp/{method}/{seed}"},
        "algo": {"config": {"use_actor_sem": method == "sem", "gamma": 0.97}},
        "terrain": {"scale_factor": 1.0},
    }
    values = np.asarray([[100, start, 1000, 0.1], [200, end, 2000, 0.9]], dtype=float)
    return Run(Path(f"{method}_{seed}"), {"variant": method, "seed": seed}, config, {"linear_tracking": values})


def test_alignment_never_extrapolates():
    runs = [make_run("baseline", 1, 10, 900), make_run("sem", 1, 30, 880)]
    grid, matrices = align(runs, "linear_tracking", 1)
    assert grid[0] == 30
    assert grid[-1] == 880
    assert matrices["baseline"].shape == (1, 200)
    assert matrices["sem"][0, 0] == pytest.approx(0.1)
    assert matrices["sem"][0, -1] == pytest.approx(0.9)


def test_alignment_rejects_disjoint_intervals():
    runs = [make_run("baseline", 1, 10, 20), make_run("sem", 1, 30, 40)]
    with pytest.raises(ValueError, match="No shared observed interval"):
        align(runs, "linear_tracking", 1)


def test_bootstrap_preserves_constant_paired_difference():
    baseline = np.arange(15, dtype=float).reshape(5, 3)
    sem = baseline + 2.0
    indices = np.random.default_rng(0).integers(0, 5, size=(100, 5))
    mean, low, high = estimate(sem - baseline, indices)
    np.testing.assert_allclose(mean, [2, 2, 2])
    np.testing.assert_allclose(low, mean)
    np.testing.assert_allclose(high, mean)


def test_only_run_identity_and_sem_flag_are_ignored():
    baseline = make_run("baseline", 1, 0, 900)
    sem = make_run("sem", 2, 0, 900)
    assert comparison_config(baseline) == comparison_config(sem)
    sem.config["terrain"]["scale_factor"] = 0.5
    assert comparison_config(baseline) != comparison_config(sem)


def test_timestamp_never_uses_a_future_observation():
    run = make_run("sem", 1, 890, 900.1)
    row, status = at_or_before(run, "linear_tracking", 900, 30)
    assert row[1] == 890
    assert row[3] == 0.1
    assert status == "available"


def test_timestamp_reports_missing_and_stale():
    run = make_run("sem", 1, 100, 200)
    assert at_or_before(run, "linear_tracking", 50, 30) == (None, "missing")
    row, status = at_or_before(run, "linear_tracking", 900, 30)
    assert row[1] == 200 and status == "stale"


def test_timestamp_tables_do_not_drop_an_unavailable_seed(tmp_path):
    import csv
    from scripts.plot_sem_comparison import METRICS

    runs = [make_run(method, seed, 100, 890 if seed == 1 else 800)
            for method in ("baseline", "sem") for seed in (1, 2)]
    for run in runs:
        run.series = {key: run.series["linear_tracking"].copy() for key in METRICS}
    make_timestamp_tables(runs, [1, 2], tmp_path, [900], 30, 100)
    rows = list(csv.DictReader((tmp_path / "paired_timestamp_summary.csv").open()))
    assert all(row["n_available"] == "0" for row in rows)
    assert all(row["status"] == "incomplete" and row["sem_minus_baseline"] == "nan" for row in rows)


def test_timestamp_tables_use_primary_window_and_export_sensitivity(tmp_path):
    import csv
    from scripts.plot_sem_comparison import METRICS

    runs = []
    for method in ("baseline", "sem"):
        run = make_run(method, 1, 0, 180)
        times = np.arange(0, 181, 10, dtype=float)
        values = np.column_stack((np.arange(len(times)), times, times * 100 + 1,
                                  np.full(len(times), 1 if method == "baseline" else 3)))
        run.series = {metric: values.copy() for metric in METRICS}
        runs.append(run)
    make_timestamp_tables(runs, [1], tmp_path, [180], 5, 100)
    paired = list(csv.DictReader((tmp_path / "paired_timestamp_summary.csv").open()))
    assert all(row["window_s"] == "60.0" and float(row["sem_minus_baseline"]) == 2 for row in paired)
    assert all(row["status"] == "available" for row in paired)
    sensitivity = list(csv.DictReader((tmp_path / "paired_timestamp_window_sensitivity.csv").open()))
    assert {float(row["window_s"]) for row in sensitivity} == {30, 60, 120}
    assert (tmp_path / "timestamp_last_observations.csv").is_file()


def test_rolling_curves_preserve_internal_unsupported_gaps():
    run = make_run("baseline", 1, 0, 280)
    times = np.asarray([0, 20, 40, 60, 80, 100, 200, 220, 240, 260, 280], dtype=float)
    run.series["linear_tracking"] = np.column_stack((
        np.arange(len(times)), times, times * 100 + 1, np.ones(len(times)),
    ))
    transformed, records = rolling_runs([run], 60, 30, .95, 30)
    curve = transformed[0].series["linear_tracking"]
    assert np.isnan(curve[curve[:, 1] == 200, 3]).all()
    assert curve[curve[:, 1] == 260, 3][0] == 1
    assert next(r for r in records if r["elapsed_s"] == 200)["status"] == "interpolation_gap"
