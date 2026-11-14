# Matched seed collection and analysis

Primary figures and timestamp tables now use time-weighted trailing windows;
see [the corrected postprocessing guide](sem_postprocessing.md). Raw point
values and plots are retained separately as diagnostics.

For the current configurable-budget protocol and timestamp-specific outputs,
see [the timed-run guide](sem_timed_runs.md). Defaults below use 900 seconds
and the new `budget_900s/` directory. The original completed study can still
be plotted with `--budget 900`; do not mix protocol versions in one comparison.

Wait until the seed-1 job finishes. On EC2, inside tmux:

```bash
cd ~/holosoma-cmu16831
bash scripts/run_matched_sem_seeds.sh
```

The script activates `hsgym` and runs eight jobs sequentially on the same GPU:

| Seed | First | Second |
|---|---|---|
| 2 | FastSAC | FastSAC+SEM |
| 3 | FastSAC+SEM | FastSAC |
| 4 | FastSAC | FastSAC+SEM |
| 5 | FastSAC+SEM | FastSAC |

All jobs use the existing launcher's 900-second budget, cold Torch/Triton
cache, shared simulator cache, 4096 environments, and identical task settings.
The script stops on the first failed job; inspect its console.log before
continuing. Do not run concurrent training or change the source/configuration
mid-batch. Eight budgets total two hours, plus final exports/cleanup.
Rerun the same script after an interruption: it skips completed seed/method
pairs and starts fresh attempts for failed or unfinished pairs, preserving all
earlier logs. Completion requires successful full-run metadata, a completed
900-second budget, final PT/ONNX outputs, and TensorBoard events. Smoke tests
do not count. This retries a timed experiment from scratch; it does not resume
training from a checkpoint. Keep the source/configuration unchanged across retries.
The script prevents concurrent copies and refuses to start while an existing
launcher/trainer in this checkout is active. It reports existing duplicate
successful runs without deleting or choosing among them.

To inspect the plan without starting training:

```bash
bash scripts/run_matched_sem_seeds.sh --status
```

Keep the Mac's caffeinated sync running. After collection, Ctrl-C and run:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh --once
```

## Plot locally without installing Holosoma

Create a separate Conda plotting environment outside the Python 3.8 simulator environment:

```bash
conda create -n holosoma-plots -c conda-forge \
  python=3.11 numpy matplotlib tensorboard pyyaml -y
conda activate holosoma-plots
python scripts/plot_sem_comparison.py --budget 900 --seeds 1 2 3 4 5
```

For an initial comparison after both seed-1 methods finish:

```bash
python scripts/plot_sem_comparison.py --budget 900 --seeds 1
```

No Holosoma imports, policy inference, or checkpoint deserialization are used.
The script reads saved YAML/JSON and TensorBoard scalar events. Run from the
fork root. The script infers `aws-results/budget_<budget>s/` from `--budget`
(default 900 seconds), and writes its `comparison/` subdirectory.

## Outputs

- `comparison_wall_time.png/.pdf`: four metrics versus startup-inclusive minutes.
- `comparison_samples.png/.pdf`: the same metrics versus logged environment transitions.
- `paired_linear_difference.png/.pdf`: SEM minus FastSAC for matched seeds; positive favors SEM.
- `summary.csv`: endpoint and normalized area-under-curve estimates per method.
- `paired_summary.csv`: SEM-minus-FastSAC endpoint and normalized AUC estimates.
- `raw_metrics.csv`: observed in-budget data, with run/seed/step/time/sample identifiers.
- `aligned_metrics.csv`: interpolated values used for mean curves and differences.
- `selected_runs.json`: exact run selection, budgets, metadata, and interpretation notes.

Metrics: linear tracking reward, angular tracking reward, mean episode duration
(converted from steps using the saved control timestep), and total episode reward.
These are training-rollout metrics. They are not held-out policy evaluations.

## Comparison conventions

Only runs marked successful, full-sized, and non-smoke are accepted. Every
requested seed must have both methods. The script checks saved task/optimizer
configs, normalizing only seed, run name, logging directory, and the SEM flag.
It checks the base revision, Python version, and cache policy too. It rejects
duplicate successful runs instead of silently choosing a favorable result.
Resolve duplicates explicitly, for example:

```bash
python scripts/plot_sem_comparison.py --seeds 1 2 3 4 5 \
  --exclude-run DIRECTORY_NAME_OF_THE_RUN_YOU_ARE_EXCLUDING
```

Data after 900 seconds are excluded, using `Timing/elapsed_wall_time_s` where
available and event wall timestamps relative to launcher start otherwise.
`Train/num_samples` supplies the sample axis. The latter follows Holosoma's
logging counter and is not reconstructed from the current iteration number.

Every aggregate uses the interval observed in ALL selected runs, with linear
interpolation only inside that interval. No values are invented at time zero
or extended to 900 seconds. Means can therefore begin after startup and end
slightly before 15 minutes. Summary files explicitly record interval endpoints.
Normalized AUC is the integral over this shared interval divided by its length;
it is NOT a whole-budget AUC that treats the unobserved startup interval as zero.
Plots also show each seed's raw curve faintly, including its own observed range.

Shading is a 95% pointwise percentile bootstrap interval over matched seeds
(2000 resamples; fixed RNG seed). The same resampled seed indices are used for
both methods and paired differences. No temporal data points are treated as
independent replicates. Five seeds provide limited uncertainty estimation;
intervals are exploratory, not a significance test or a guarantee of improvement.
With one matched seed no interval is drawn and CSV interval fields are `nan`
(undefined), rather than reporting zero uncertainty. Verify that selected GPU/instance
types match using `nvidia-smi.txt`; this is not checked by the plotting script.

For a preliminary SEM improvement claim, prioritize linear tracking versus
elapsed time and the paired linear difference; sample curves help distinguish
sample efficiency from wall-time effects. Total reward depends on shaping and
should be secondary. Separate checkpoint evaluation is needed to assess final
policy behavior, and the final saved checkpoint may be slightly over budget.
