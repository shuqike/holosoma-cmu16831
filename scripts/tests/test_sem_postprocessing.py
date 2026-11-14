"""Analytical checks for irregular sampling, fixed windows, and missing data."""

import numpy as np
import pytest

from scripts.sem_postprocessing import interval_mean, trailing_window


def series(times, values):
    times = np.asarray(times, dtype=float)
    return np.column_stack((np.arange(len(times)), times, times * 4096 + 4096, values))


def test_time_weighting_is_invariant_to_dense_logging_of_same_curve():
    sparse = series([0, 30, 60], [0, 1, 0])
    dense_times = [0, 30, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    dense = series(dense_times, np.interp(dense_times, sparse[:, 1], sparse[:, 3]))
    result = trailing_window([sparse, dense], 60)
    np.testing.assert_allclose(result.values, [.5, .5])
    assert not np.isclose(sparse[:, 3].mean(), dense[:, 3].mean())


def test_no_observation_after_target_enters_window():
    data = series([0, 30, 59, 60.1], [0, 0, 0, 1000])
    result = trailing_window([data], 60, max_lag=5)
    assert result.statuses == ["available"]
    assert result.end == 59
    assert result.end_lag == 1
    assert result.coverage == pytest.approx(59 / 60)
    assert result.values[0] == 0


def test_all_runs_use_same_supported_interval():
    a = series([0, 20, 40, 60], np.asarray([0, 20, 40, 60]) / 60)
    b = series([1, 21, 41, 59], np.asarray([1, 21, 41, 59]) / 60)
    result = trailing_window([a, b], 60, max_lag=5)
    assert (result.start, result.end) == (1, 59)
    assert result.coverage == pytest.approx(58 / 60)
    np.testing.assert_allclose(result.values, [.5, .5])


def test_missing_seed_does_not_improve_cohort_coverage():
    result = trailing_window([series([0, 30, 60], [1, 1, 1]), np.empty((0, 4))], 60)
    assert result.statuses == ["missing", "missing"]
    assert np.isnan(result.values).all()


def test_long_interpolation_gap_is_not_hidden_by_good_coverage():
    result = trailing_window([series([0, 60], [1, 1])], 60, max_gap=30)
    assert result.coverage == 1
    assert result.statuses == ["interpolation_gap"]
    assert result.max_gaps[0] == 60
    assert np.isnan(result.values[0])


def test_insufficient_coverage_and_stale_endpoints_are_explicit():
    data = series([20, 40, 60], [1, 1, 1])
    incomplete = trailing_window([data], 60)
    assert incomplete.statuses == ["insufficient_coverage"]
    assert incomplete.coverage == pytest.approx(2 / 3)
    stale = trailing_window([data], 100, max_lag=5)
    assert stale.statuses == ["stale"]
    assert np.isnan(stale.values[0])


def test_auc_uses_original_knots_even_when_display_grid_misses_spike():
    x = np.asarray([0, 49.99, 50, 50.01, 100])
    y = np.asarray([0, 0, 100, 0, 0])
    assert interval_mean(x, y, 0, 100) == pytest.approx(.01)
    display_x = np.linspace(0, 100, 200)
    assert np.interp(display_x, x, y).max() == 0


@pytest.mark.parametrize("kwargs", [
    {"window": 0}, {"window": np.nan}, {"max_lag": -1},
    {"min_coverage": 0}, {"min_coverage": 1.1}, {"max_gap": 0},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match="Invalid trailing-window"):
        trailing_window([series([0, 30, 60], [1, 1, 1])], 60, **kwargs)
