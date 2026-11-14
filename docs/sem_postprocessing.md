# Time-weighted postprocessing

The `time-weighted-windows-v2` analysis corrects the sensitivity of single
endpoint observations to dense capture-time logging. It changes only local
postprocessing, not training, checkpoints, source fingerprints, or run matching.
The existing pilot can be pooled with seeds 3–5 collected with the same trainer.

## Primary statistic

The default primary statistic is the time-weighted mean of the logged
trajectory over a trailing 60-second window. For each requested timestamp,
the analysis first excludes every observation after that timestamp. It then
chooses one common observed interval for all requested methods and seeds,
inside the requested window. The estimate is the piecewise-linear integral
over that interval divided by its duration.

Every original observation knot contributes to integration. Twenty-five
observations over five seconds therefore do not count as twenty-five equal
votes against a handful of observations covering the preceding minute.
Interpolation is used only inside observed support, never to extend the
endpoint to the requested deadline or to fill startup with zeros.

Defaults require at least 95% coverage of the requested window, an endpoint
age no greater than `--timestamp-max-lag-seconds`, and no supporting input
gap larger than 30 seconds. Window bounds, coverage, gap sizes, and observation
counts are written to CSV. Unavailable estimates remain NaN with an explicit
status. Aggregate estimates require every requested seed/pair; seeds are never
dropped to improve the result. Early timestamps and sparse historical logs
may not satisfy these requirements.

The 15-minute table thus describes approximately minutes 14–15, and the
30-minute table approximately minutes 29–30. It does not claim to measure
an instantaneous policy at exactly either deadline.

## Commands

In the local plotting Conda environment:

```bash
conda activate holosoma-plots && python scripts/plot_sem_comparison.py --budget 1800 --seeds 1 2 --timestamps-seconds 900 1800 --timestamp-max-lag-seconds 5
```

Defaults already select a 60-second primary window and 30-/120-second
sensitivity windows. Explicit configuration:

```bash
python scripts/plot_sem_comparison.py --budget 1800 --seeds 1 2 --timestamps-seconds 900 1800 --summary-window-seconds 60 --sensitivity-windows-seconds 30 120 --min-window-coverage 0.95 --max-interpolation-gap-seconds 30 --timestamp-max-lag-seconds 5
```

Use all five seeds after collection. Input is inferred from the run budget;
outputs remain under `aws-results/budget_1800s/comparison/`. Fix the primary
window before examining additional seeds and report the sensitivity results
rather than selecting whichever window favors a method.

## Outputs

| File | Statistic |
|---|---|
| `comparison_wall_time.png/.pdf` | Primary trailing time-weighted means versus elapsed time |
| `comparison_samples.png/.pdf` | The same time-window means versus samples; these are not fixed-sample-window averages |
| `paired_linear_difference.png/.pdf` | Paired differences of the primary trailing means |
| `summary.csv`, `paired_summary.csv` | Endpoints/AUC of primary windowed curves; actual shared support is recorded |
| `windowed_metrics.csv` | Per-observation primary rolling estimates, coverage, and statuses |
| `timestamp_metrics.csv` | Per-run primary trailing-window estimates at requested timestamps |
| `timestamp_summary.csv`, `paired_timestamp_summary.csv` | Primary window method means and paired differences |
| `timestamp_window_sensitivity_metrics.csv` | Per-run estimates for all configured window lengths |
| `timestamp_window_sensitivity_summary.csv` | Method summaries for all window lengths |
| `paired_timestamp_window_sensitivity.csv` | Paired differences for all window lengths |
| `timestamp_last_observations.csv` | Original raw endpoint values and ages, explicitly diagnostic |
| `*_raw.png/.pdf`, `summary_raw.csv`, `paired_summary_raw.csv` | Unsmooth raw diagnostics; capture-time averaging windows vary |
| `raw_metrics.csv`, `all_metrics.csv` | Original metric values, respectively budget-clipped and including post-budget diagnostics |
| `timestamp_scalars.csv` | Original scalar diagnostics at raw selected observations, not window averages |
| `timestamp_checkpoints.csv` | Unchanged capture times and policy eligibility |

Windowed curves are evaluated at each actual observation before alignment
for display. Unsupported internal estimates remain gaps, rather than being
deleted and bridged. Curves start only when a supported window exists.
Panels without enough common support are labeled as unavailable. The
200-point display grid is not used to calculate AUC: integration uses the
original knots of each source curve, clipped to the shared interval.

`selected_runs.json` records analysis version, window settings, source hash,
and interpretation notes. The raw observations remain available for audit.
Rerunning plotting overwrites generated comparison outputs; it does not modify
run data or checkpoints. No training rerun is needed for these corrections.

## Limits

Time weighting reduces sensitivity to logging density, but the input values
are already averages over variable windows. Episode sums/counts were not
saved, so this analysis cannot reconstruct exact episode-weighted rewards.
It also cannot remove all differences in the underlying averaging windows.
These summaries describe logged training trajectories, not held-out policy
evaluations. Independently evaluating the eligible saved checkpoints remains
the stronger test of final policy quality.

Bootstrap intervals resample matched training seeds, using the same indices
for both methods and every paired comparison. Time points are not replicates.
Two pilot seeds provide very limited uncertainty information; five are still
a small sample. Neither smoothing nor extra log entries adds training seeds.

Runtime regression checks, when you execute them in the plotting environment:

```bash
python -m pytest --confcutdir=scripts/tests scripts/tests/test_sem_postprocessing.py scripts/tests/test_sem_plotting.py -q
```

`--confcutdir=scripts/tests` prevents loading the repository-root `conftest.py`,
which imports Holosoma/PyTorch for simulator tests. The standalone analysis
tests need only the plotting environment; Holosoma is not required on the Mac.

These cover irregular sampling, dense-log invariance, future-data exclusion,
shared window support, coverage/gap rejection, paired aggregation, and exact
integration on source knots. Project code/tests are not executed locally
during implementation; static checks are reported separately.
