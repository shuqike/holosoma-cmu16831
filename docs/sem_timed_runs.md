# Configurable budgets and timestamp comparisons

For the corrected trailing-window summaries and sensitivity analysis, see
[time-weighted postprocessing](sem_postprocessing.md). Raw last-observation
values remain available as diagnostics, but are no longer headline summaries.

The `timed-milestones-v1` protocol supports arbitrary positive run durations
and timestamps within those durations. All times are seconds since training
interpreter launch, including simulator/import startup and compilation.
The end is always added to the schedule. With no timestamp argument, the
schedule is 900 seconds (when in range), plus the end. Examples:

| Duration | Custom marks | Effective schedule |
|---|---|---|
| 1800 s | omitted | 900, 1800 s |
| 1200 s | 300, 600, 900 | 300, 600, 900, 1200 s |
| 120 s | omitted | 120 s |

Changing duration no longer marks a run as a smoke test. The default safety
ceiling is 2,147,483,647 iterations, so the old 50,000-iteration limit does
not prematurely stop long runs. `--iterations` remains configurable; stopping
before the requested duration does not qualify as a complete measured run.

## Timestamp collection and timing limits

Regular logging remains every 100 iterations (`--logging-interval`). Within
five seconds before each mark (`--timestamp-window-seconds`), every completed
iteration logs all available diagnostics. Sample counts reflect actual
collected rollouts, including these extra records.

The learner captures PT/ONNX once near the start of each dense window, and
again at the first safe boundary at/after the mark. The pre-target snapshot
provides a nearby eligible policy when captured in time. Each capture records
its actual time, offset from the requested
mark, and strict timing eligibility. Exact capture times cannot be guaranteed
during blocking simulator, compiler, GPU, or export work. Several marks crossed
during one blocking call share the actual available policy; no intermediate
policies are invented. Marks before startup/initial learning may have no
usable observations. Early stops retain unreached marks explicitly.

Intermediate logging and exports consume the budget. Final export is cleanup.
The stop is cooperative, with actual overrun reported. Metrics are rolling
training statistics and update-window averages, not independent policy
evaluations. Both methods must use identical logging/capture schedules.

## Upload: Mac

Upload the full checkout, including the new helper and recorder; transferring
only the batch script is insufficient. The existing editable `hsgym` install
uses the uploaded files. No new training dependencies are required.

```bash
rsync -az --exclude='logs/' --exclude='aws-results/' \
  --exclude='.venv/' --exclude='.venv-plots/' --exclude='__pycache__/' \
  --exclude='.DS_Store' ./ holosoma-aws:holosoma-cmu16831/
```

## 30-minute pilot: EC2, inside tmux

```bash
cd ~/holosoma-cmu16831
bash scripts/run_matched_sem_seeds.sh --seconds 1800 --seeds 1 2 \
  --timestamps-seconds 900 1800 --status
bash scripts/run_matched_sem_seeds.sh --seconds 1800 --seeds 1 2 \
  --timestamps-seconds 900 1800
```

This runs four serial jobs, starting with baseline for seed 1 and SEM for
seed 2. Order alternates by position in the provided seed list. For all five
seeds, use `--seeds 1 2 3 4 5`; compatible completed pairs are skipped.
Keep the seed list order stable across retries.

Default outputs are `logs/sem_experiments/budget_1800s/TIMESTAMP_METHOD_seedN/`.
Other durations receive their own folders. Existing top-level 15-minute
results are untouched. `--output-root logs/sem_experiments/my_study` overrides
the default and can separate studies with the same duration.

The batch matches budget, capture schedule, logging window/interval, iteration
ceiling, protocol, and training source fingerprint before skipping a run.
Completion requires successful timing, timestamp observations, and final and
timestamp artifacts. Failed attempts are preserved and restarted from scratch,
not resumed with an incomplete replay/RNG state. Changing training source
invalidates earlier matches; isolate such experiments in a fresh directory.

Single-run alternative (always creates a fresh attempt):

```bash
source scripts/source_isaacgym_setup.sh
python scripts/run_sem_experiment.py --variant sem --seed 1 \
  --seconds 1800 --timestamps-seconds 900 1800
```

`--smoke-test` explicitly excludes a diagnostic run from analysis. Reduced
environment counts and disabled compilation also imply smoke tests.

## Sync: Mac

Existing continuous sync pulls all budget subfolders recursively:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh
```

After training finishes, Ctrl-C and pull ONNX outputs too:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh --once
```

Optional paths allow syncing just one study:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh \
  --remote-dir holosoma-cmu16831/logs/sem_experiments/budget_1800s \
  --local-dir aws-results/budget_1800s
```

Use the same paths with `--once` for its final pull. An optional positional
SSH alias still defaults to `holosoma-aws`. Keep the Mac powered, lid open.

## Plot: Mac, Conda environment

```bash
conda activate holosoma-plots
python scripts/plot_sem_comparison.py \
  --budget 1800 --seeds 1 2 --timestamps-seconds 900 1800 \
  --timestamp-max-lag-seconds 5
```

Use all five seeds once collected. The script infers the input folder as
`aws-results/budget_<budget>s/` relative to the repository, independently of
the current working directory. Outputs default to that folder's `comparison/`
subdirectory. `--output` overrides the output location; no input path is needed.
If the canonical folder is absent, a uniquely matching custom study is
discovered from budget/seed metadata under `aws-results/`. Ambiguous studies
are rejected rather than chosen automatically; `--exclude-run DIRECTORY_NAME`
can exclude unwanted attempts. Duplicate successful runs are also rejected.

Future collection and comparison at 5, 10, 15, and 20 minutes:

```bash
# EC2
bash scripts/run_matched_sem_seeds.sh --seconds 1200 --seeds 1 2 3 4 5 \
  --timestamps-seconds 300 600 900 1200

# Mac
python scripts/plot_sem_comparison.py \
  --budget 1200 --seeds 1 2 3 4 5 --timestamps-seconds 300 600 900 1200 \
  --timestamp-max-lag-seconds 5
```

Analysis timestamps can also be chosen after collection using existing
observations. Extra checkpoints exist only at marks requested during
collection or ordinary periodic checkpoint steps.

## Artifacts and interpretation

Each training directory additionally contains:

- `observations.jsonl`: complete scalar dictionaries, steps, and actual
  observation times for every log event, including available final diagnostics.
- `timestamps.json`: capture schedule, actual checkpoint times/offsets,
  filenames, latest diagnostics, completion state, and unreached marks.
- PT/ONNX snapshots at requested marks, besides periodic/final outputs.

`run.json` adds protocol, schedule, logging window, iteration limit, and a
training source fingerprint. `timing.json` distinguishes budget and iteration
stops. Checkpoints preserve their elapsed time and strict budget eligibility.

The existing curves, shared-interval summaries, and paired bootstrap outputs
remain. Requested times appear as dotted lines on wall-clock plots. New files:

| File | Meaning |
|---|---|
| `timestamp_metrics.csv` | Four trailing-window metrics per seed/method/mark, shared interval, coverage, and status |
| `timestamp_summary.csv` | Method means and bootstrap intervals at each mark |
| `paired_timestamp_summary.csv` | Paired SEM-minus-FastSAC differences at each mark |
| `timestamp_scalars.csv` | All available scalar tags at those observations, in original units |
| `timestamp_checkpoints.csv` | Pre-target and boundary snapshot filenames, actual times, offsets, and eligibility |
| `all_metrics.csv` | Four metrics including post-budget diagnostics, with a `within_budget` flag |

Timestamp metrics now use a **time-weighted trailing window** (default 60 s)
supported by observations at or before the target. Single raw observations
are retained in `timestamp_last_observations.csv` for diagnosis.
`--timestamp-max-lag-seconds` limits acceptable observation age: default 30 s
supports old sparse logs; use 5 s with the new dense windows. Missing/stale
observations/windows are labeled. Aggregates require every requested seed/pair; no
unavailable seeds are silently dropped. Inspect per-run observation ages.

Boundary policies can be slightly late and are labeled accordingly. Use the
pre-target checkpoint only if its eligibility flag confirms timely capture;
blocking work can prevent this snapshot too. A late checkpoint is not an
eligible policy at a strict earlier budget. Dense
pre-target metrics support the training-curve comparison; separate policy
evaluation still needs an eligible checkpoint and controlled evaluation.

`raw_metrics.csv` and ordinary curves remain budget-clipped; post-budget
diagnostics do not enter those comparisons. Existing `summary.csv` endpoints
refer to the shared observed interval, which can end before the requested
mark. Use the windowed timestamp tables to compare the periods preceding requested times.
Bootstrap intervals resample matched seeds, not time points.

## Runtime verification

Implementation checks on the Mac are static only. On EC2, with pytest installed:

```bash
source scripts/source_isaacgym_setup.sh
python -m pytest --confcutdir=scripts/tests scripts/tests/test_timed_experiment.py -q
python -m pytest src/holosoma/holosoma/agents/fast_sac/tests/test_sem.py -q

# Small integration check, excluded from measured results
python scripts/run_sem_experiment.py --num-envs 64 --seconds 180 \
  --timestamps-seconds 90 180 --smoke-test --no-compile
```

Inspect observations, timestamp offsets, PT/ONNX artifacts, and final timing
before full collection. In the plotting environment, with pytest installed:

```bash
python -m pytest --confcutdir=scripts/tests scripts/tests/test_sem_plotting.py -q
```

The cutoff excludes the repository-root simulator `conftest.py` from these
standalone script tests; the plotting environment does not need Holosoma.
