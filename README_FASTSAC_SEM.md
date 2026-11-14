# FastSAC + SEM

This branch compares FastSAC with FastSAC + actor-only simplicial embeddings
(SEM) on the Unitree G1 29-DoF velocity-tracking task. It measures learning
under matched wall-clock budgets using Holosoma revision `9c238cf8` and
Isaac Gym Preview 4.

## References

- Seo et al., [*Learning Sim-to-Real Humanoid Locomotion in 15 Minutes*](https://arxiv.org/abs/2512.01996): FastSAC and the humanoid learning setup. [Holosoma repository](https://github.com/amazon-far/holosoma).
- Obando-Ceron et al., [*Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents*](https://arxiv.org/abs/2510.13704): the representation method used for the actor extension.

## Algorithm

SEM replaces the final SiLU activation in the FastSAC actor trunk:

```text
FastSAC:       observations -> 512 -> 256 -> 128 -> SiLU -> mean/log-std heads
FastSAC + SEM: observations -> 512 -> 256 -> 128 -> SEM  -> mean/log-std heads

SEM: reshape 128 features into two groups of 64
     -> softmax within each group
     -> flatten to 128 features
```

Earlier activations and layer normalization are preserved. Each SEM group
has nonnegative coordinates that sum to one. SEM adds no trainable parameters.
The critics, replay buffer, optimizers, entropy tuning, and task are shared
between methods. This is a dimension-preserving, actor-only adaptation of SEM.

The `exp:g1-29dof-fast-sac-sem` preset enables `use_actor_sem=True` with
`sem_group_size=64`. The baseline preset is `exp:g1-29dof-fast-sac`.
The group size must be at least two and divide the final hidden width.
Checkpoint loading checks SEM settings because parameter shapes alone do
not distinguish the actors.

| Component | File under `src/holosoma/holosoma/` |
|---|---|
| Actor and SEM layer | `agents/fast_sac/fast_sac.py` |
| Algorithm configuration | `config_types/algo.py` |
| Experiment presets | `config_values/loco/g1/experiment.py`, `config_values/experiment.py` |
| Training and checkpoint timing | `agents/fast_sac/fast_sac_agent.py` |
| Timestamp collection | `agents/fast_sac/timed_experiment.py` |
| Atomic PT checkpoint saving | `agents/modules/logging_utils.py` |

## Experiment settings

| Setting | Value |
|---|---|
| Task | G1 29-DoF commanded velocity tracking |
| Simulator | Isaac Gym Preview 4, headless |
| Parallel environments | 4,096 |
| Terrain | Default mixed terrain, `scale_factor=1.0` |
| Perturbations | Random pushes and domain randomization enabled |
| Observations | State observations; no camera input |
| Methods | `baseline` = FastSAC; `sem` = FastSAC + SEM |
| Budget | Configurable with `--seconds`; default 900 seconds |
| Timestamps | Configurable; run end is always included |
| Logging | TensorBoard and files; W&B and video disabled |
| Regular logging / checkpoints | Every 100 / 1,000 iterations |
| Timestamp window | Dense logging in the preceding 5 seconds, with PT/ONNX captures |
| Execution | One GPU and one training process at a time |
| Caches | Fresh Torch/Triton cache per run; shared simulator caches |

Keep hardware, dependencies, task settings, and capture schedules fixed
across methods. Use matched seeds. The batch alternates which method runs
first by position in the supplied seed list.

The budget includes interpreter startup, imports, simulator initialization,
compilation, and training. Instance boot and installation are outside it.
Training stops between iterations, so the actual stop can be slightly late.
Final save/export and cleanup extend process duration.

Timestamp records include actual capture times and eligibility. For a strict
policy-at-budget comparison, use an eligible pre-target checkpoint. Boundary
and final checkpoints can be late. Filenames identify iteration counters,
not environment transitions.

## Environment setup

Commands assume the repository root. Training uses Python 3.8 in `hsgym`.
Plotting uses Python 3.11 and does not require Holosoma, Isaac Gym, or CUDA.

### Training

The AWS configuration is a `g6e.2xlarge`: one NVIDIA L40S GPU (48 GB),
8 vCPUs, 64 GiB RAM, and a 200 GiB gp3 volume. Use the x86_64 Deep Learning
Base AMI with Single CUDA (Ubuntu 24.04) and retain its GPU driver.
See [the EC2 guide](docs/sem_ec2.md) for launch and dependency details.

```bash
sudo apt-get update && sudo apt-get install -y tmux rsync git wget curl build-essential cmake libgl1 libegl1 libvulkan1 libglib2.0-0t64 libxrender1 libxi6 libxrandr2
```

```bash
set -o pipefail && bash scripts/setup_isaacgym.sh 2>&1 | tee setup-isaacgym.log
```

The installer creates `hsgym` under `~/.holosoma_deps/miniconda3` and installs
Isaac Gym Preview 4 and the repository. Source it in each training shell:

```bash
source scripts/source_isaacgym_setup.sh
```

Preserve dependency versions across runs. Save environment records with results:

```bash
conda list --explicit > training-conda-explicit.txt && python -m pip freeze > training-pip-freeze.txt
```

### Plotting

```bash
conda create -n holosoma-plots -c conda-forge python=3.11 numpy matplotlib tensorboard pyyaml pytest -y && conda activate holosoma-plots
```

## Workflow

### Transfer

Configure SSH alias `holosoma-aws` and verify access. Upload the checkout,
including `.git` for source records:

```bash
ssh holosoma-aws 'mkdir -p ~/holosoma-cmu16831/logs/sem_experiments' && rsync -az --exclude='logs/' --exclude='aws-results/' --exclude='.venv/' --exclude='.venv-plots/' --exclude='__pycache__/' --exclude='.DS_Store' ./ holosoma-aws:holosoma-cmu16831/
```

Use tmux on the training instance to keep runs active after SSH disconnects:

```bash
tmux new -s sem
```

### Integration checks

`scripts/run_sem_experiment.py` launches one run and records its command,
source, GPU, timing, and completion status. In the training environment:

```bash
python scripts/run_sem_experiment.py --num-envs 64 --iterations 20 --seconds 300 --no-compile && python scripts/run_sem_experiment.py --num-envs 64 --iterations 20 --seconds 300
```

These runs are marked as smoke tests and excluded from analysis.

### Matched runs

`scripts/run_matched_sem_seeds.sh` activates the training environment and
collects paired runs sequentially. For five seeds at 15 minutes:

```bash
bash scripts/run_matched_sem_seeds.sh --seconds 900 --seeds 1 2 3 4 5
```

For 30 minutes with captures at 15 and 30 minutes:

```bash
bash scripts/run_matched_sem_seeds.sh --seconds 1800 --seeds 1 2 3 4 5 --timestamps-seconds 900 1800
```

Durations and timestamps are arbitrary positive seconds; timestamps must
fall within the budget. Add `--status` to inspect completion without training.
Keep the seed-list order stable across retries. Defaults are 900 seconds and
seeds 2–5; the commands above specify all seeds.

The batch skips compatible completed runs, preserves failed attempts, and
starts incomplete combinations from scratch. Checks include budget, schedule,
logging settings, protocol, and training source fingerprint. Checkpoints omit
replay contents and full RNG state, so retries receive a fresh budget.
The script stops on failure and prevents concurrent training.

A single run always creates a fresh attempt:

```bash
python scripts/run_sem_experiment.py --variant sem --seed 1 --seconds 1800 --timestamps-seconds 900 1800
```

Outputs default to `logs/sem_experiments/budget_<seconds>s/`.
Use `--output-root` to separate studies that share a budget.

### Sync

`scripts/sync_ec2_results.sh` pulls results into `aws-results/` every
30 seconds and retries connection failures:

```bash
bash scripts/sync_ec2_results.sh
```

It excludes caches, temporary files, and live ONNX exports. After training
ends, stop the loop and download ONNX exports as well:

```bash
bash scripts/sync_ec2_results.sh --once
```

Use `--remote-dir`, `--local-dir`, or a positional SSH alias to override
paths. On macOS, prefix the command with `caffeinate -is` to prevent sleep
while syncing. Training completion and syncing do not stop the EC2 instance.

### Plot

`scripts/plot_sem_comparison.py` infers input from `--budget`, checks paired
configurations, and rejects missing pairs or duplicate successful runs.
GPU hardware should also match across runs.

```bash
conda activate holosoma-plots && python scripts/plot_sem_comparison.py --budget 900 --seeds 1 2 3 4 5
```

```bash
conda activate holosoma-plots && python scripts/plot_sem_comparison.py --budget 1800 --seeds 1 2 3 4 5 --timestamps-seconds 900 1800 --timestamp-max-lag-seconds 5
```

Outputs go to `aws-results/budget_<budget>s/comparison/`. Use `--output` to
change that location and `--exclude-run DIRECTORY_NAME` to exclude an attempt.
Plotting updates comparison files without changing checkpoints or run data.

## Outputs

Each `TIMESTAMP_METHOD_seedN/` directory contains:

| File or directory | Contents |
|---|---|
| `run.json` | Command, source revision, seed, budget, protocol, and status |
| `console.log`, `nvidia-smi.txt` | Training output and GPU information |
| `source.patch`, `launcher.py`, `source/` | Source changes and selected file snapshots |
| `training/cmu16831-sem/EXPERIMENT/` | Configuration, metrics, timing, and checkpoints |

The training directory contains `holosoma_config.yaml`, TensorBoard events,
`timing.json`, `observations.jsonl`, `timestamps.json`, and PT/ONNX checkpoints.
Available artifacts depend on the recorded protocol. Source snapshots do
not replace a dependency lock. Results and checkpoints are ignored by Git.

The comparison directory contains wall-time and sample plots, paired
differences, endpoint/AUC tables, timestamp summaries, sensitivity tables,
raw diagnostics, and `selected_runs.json` with analysis settings.
See [the results README](aws-results/README.md) for the directory layout.

## Metrics

The four metrics are linear tracking reward, angular tracking reward,
episode duration, and total episode reward. These are training-rollout
statistics; tracking rewards are not physical velocity errors.

Primary summaries use time-weighted trailing means (default 60 seconds),
with 30- and 120-second sensitivity windows. Analysis excludes observations
after each target and checks coverage, staleness, and gaps. Missing estimates
are labeled; incomplete seed groups are not silently averaged. Endpoint and
normalized AUC summaries use shared observed support, which can end before
the requested deadline.

Confidence intervals bootstrap matched seeds, not time points. Raw values
remain available as diagnostics. These summaries do not replace frozen-policy
evaluation or establish sim-to-real performance. Use verified transition
counts for sample-matched checkpoints and timing metadata for strict budgets.

See [timestamp collection](docs/sem_timed_runs.md) and
[postprocessing](docs/sem_postprocessing.md) for definitions and options.

## Tests

Run algorithm tests in the training environment:

```bash
python -m pytest src/holosoma/holosoma/agents/fast_sac/tests/test_sem.py -q
```

Run timing and analysis tests in the plotting environment:

```bash
conda activate holosoma-plots && python -m pytest --confcutdir=scripts/tests scripts/tests/test_timed_experiment.py scripts/tests/test_sem_postprocessing.py scripts/tests/test_sem_plotting.py -q
```

`--confcutdir` avoids simulator-specific test imports. ONNX parity tests need
ONNX and ONNX Runtime. For the broader framework, see [the upstream overview](README.md)
and [training guide](src/holosoma/README.md).
