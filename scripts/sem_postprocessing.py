"""Time-weighted summaries of logged trajectories, independent of log density.

These are piecewise-linear summaries of recorded statistics, not reconstructed
episode-weighted rewards or independent policy evaluations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowSummary:
    values: np.ndarray
    statuses: list
    start: float
    end: float
    coverage: float
    end_lag: float
    max_gaps: np.ndarray
    counts: np.ndarray


def interval_mean(x, y, start, end):
    """Integrate over every original knot, with interpolation only at boundaries."""
    if end <= start or start < x[0] or end > x[-1]:
        raise ValueError("Requested interval is outside observed support")
    inside = (x > start) & (x < end)
    knots = np.concatenate(([start], x[inside], [end]))
    values = np.interp(knots, x, y)
    if not np.all(np.isfinite(values)):
        return float("nan")
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(values, knots) / (end - start))


def trailing_window(series, target, window=60.0, max_lag=30.0, min_coverage=0.95, max_gap=30.0):
    """Summarize Nx4 step/time/sample/value arrays on one shared trailing interval.

    Exclude every point after target before choosing support or interpolating.
    All requested runs contribute to the shared boundaries. No seed is dropped
    to improve coverage. Unsupported estimates remain NaN with explicit status.
    """
    if (not np.isfinite(target) or target < 0 or not np.isfinite(window) or window <= 0
            or not np.isfinite(max_lag) or max_lag < 0
            or not np.isfinite(min_coverage) or not 0 < min_coverage <= 1
            or not np.isfinite(max_gap) or max_gap <= 0 or not series):
        raise ValueError("Invalid trailing-window parameters")
    eligible = [array[array[:, 1] <= target] for array in series]
    n = len(series)
    result = WindowSummary(np.full(n, np.nan), ["missing"] * n, float("nan"), float("nan"),
                           0.0, float("nan"), np.full(n, np.nan), np.zeros(n, dtype=int))
    if target == 0 or any(len(array) == 0 for array in eligible):
        return result
    nominal_start = max(0.0, target - window)
    result.start = max(nominal_start, max(array[0, 1] for array in eligible))
    result.end = min(target, min(array[-1, 1] for array in eligible))
    result.end_lag = target - result.end
    result.coverage = max(0.0, result.end - result.start) / window
    common_status = "available"
    if result.end <= result.start:
        common_status = "missing"
    elif result.end_lag > max_lag + 1e-6:
        common_status = "stale"
    elif result.coverage + 1e-12 < min_coverage:
        common_status = "insufficient_coverage"
    for index, array in enumerate(eligible):
        result.statuses[index] = common_status
        if result.end <= result.start:
            continue
        x = array[:, 1]
        left = max(0, int(np.searchsorted(x, result.start, side="right")) - 1)
        right = min(len(x) - 1, int(np.searchsorted(x, result.end, side="left")))
        support = x[left:right + 1]
        result.max_gaps[index] = float(np.diff(support).max()) if len(support) > 1 else 0.0
        result.counts[index] = int(((x >= result.start) & (x <= result.end)).sum())
        if common_status != "available":
            continue
        if result.max_gaps[index] > max_gap + 1e-6:
            result.statuses[index] = "interpolation_gap"
            continue
        result.values[index] = interval_mean(x, array[:, 3], result.start, result.end)
        if not np.isfinite(result.values[index]):
            result.statuses[index] = "nonfinite"
    return result
