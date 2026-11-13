"""Checks for aggregation boundaries and paired seed uncertainty; no policy execution."""

from pathlib import Path

import numpy as np
import pytest

from scripts.plot_sem_comparison import Run, align, comparison_config, estimate


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
