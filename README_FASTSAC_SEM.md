# FastSAC + SEM: CMU 16-831 experiment branch

This branch studies whether adding **simplicial embeddings (SEM) to the
FastSAC actor** improves humanoid walking under a fixed training budget.
The comparison is FastSAC versus FastSAC + actor-only SEM on the Unitree G1
29-DoF velocity-tracking task. Both methods run in the same Holosoma checkout,
with the same task, hardware, and experiment instrumentation.

The branch is based on Holosoma revision
**`9c238cf80f531c0e65d818348c3c1a5cc2764f5b`** (`9c238cf8`). Its training
backend is **Isaac Gym Preview 4**. Training runs on an AWS Linux GPU
instance.
This README describes the experiment implementation and collection protocol

## Research context and references

1. **Younggyo Seo, Carmelo Sferrazza, Juyue Chen, Guanya Shi, Rocky Duan,
   and Pieter Abbeel (2025).**
   [*Learning Sim-to-Real Humanoid Locomotion in 15 Minutes*](https://arxiv.org/abs/2512.01996).
   The paper develops FastSAC/FastTD3 recipes for fast humanoid learning with
   massively parallel simulation. Holosoma provides the training framework.
   [Project page](https://younggyo.me/fastsac-humanoid/) ·
   [Upstream code](https://github.com/amazon-far/holosoma).
2. **Johan Obando-Ceron, Walter Mayor, Samuel Lavoie, Scott Fujimoto,
   Aaron Courville, and Pablo Samuel Castro (2025 preprint; revised 2026).**
   [*Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents*](https://arxiv.org/abs/2510.13704).
   SEM constrains learned features to groups of probability-simplex vectors.
   This paper motivates the representation change tested here.

Our implementation is a dimension-preserving, actor-only adaptation of SEM.
It does not reproduce every architectural choice or benchmark in the SEM
paper, nor the full sim-to-real evaluation in the Holosoma paper. Published
FastSAC curves provide context; a locally collected FastSAC baseline controls
for this checkout, GPU, environment, and timing protocol.

## What changes in the learning algorithm?

FastSAC remains the off-policy actor-critic learner. Its replay buffer,
distributional critics, update rules, entropy tuning, optimizer settings,
observation normalization, and task settings are inherited from the baseline.
The intervention replaces the **last actor trunk activation**:

```text
FastSAC:       observation -> 512 -> 256 -> 128 -> SiLU -> mean/log-std heads
FastSAC + SEM: observation -> 512 -> 256 -> 128 -> SEM  -> mean/log-std heads

SEM: reshape 128 features into (2, 64)
     -> softmax within each group of 64
     -> flatten back to 128
```

The diagram omits the earlier activations and layer normalization, which are
preserved. For each group, SEM computes `exp(z_i) / sum_j exp(z_j)`;
the coordinates are nonnegative and sum to one. Both Gaussian output heads
consume the resulting features. SEM introduces **no additional trainable
parameters**, although its operation can still affect runtime.

The configuration fields are `use_actor_sem` (default `False`) and
`sem_group_size` (default `64`). The final hidden width must be divisible by
the group size, which must be at least two. The new
`exp:g1-29dof-fast-sac-sem` preset inherits `exp:g1-29dof-fast-sac` and enables
SEM. Because SEM has no weights of its own, the checkpoint loader explicitly
checks SEM settings instead of relying only on matching parameter shapes.

Other branch changes support the experiment:

| Area | Implementation |
|---|---|
| Actor representation | `src/holosoma/holosoma/agents/fast_sac/fast_sac.py` |
| Configuration and preset registration | `config_types/algo.py`, `config_values/loco/g1/experiment.py`, `config_values/experiment.py` under `src/holosoma/holosoma/` |
| Startup-inclusive deadline, timing logs, checkpoint metadata | `src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py` |
| Atomic checkpoint publication for live syncing | `src/holosoma/holosoma/agents/modules/logging_utils.py` |
| Legacy Tyro parser compatibility | `src/holosoma/holosoma/config_types/simulator.py` |
| Collection, syncing, analysis, and tests | Scripts and test files described below |

The Tyro fix hides optional scene-file and rigid-object lists from CLI
expansion while retaining their Python configuration fields and `None`
defaults. It resolves parser construction on this legacy checkout without
adding scene assets or changing the locomotion task.

## Experiment protocol

| Setting | Value |
|---|---|
| Base checkout | Holosoma `9c238cf8` plus this branch's changes |
| Robot and task | Unitree G1, 29 DoF, commanded velocity tracking |
| Simulator | Isaac Gym Preview 4, headless |
| Parallel environments | 4,096 |
| Terrain | Default mixed terrain, `scale_factor=1.0` |
| Perturbations | Default random pushes and domain randomization enabled |
| Observations | Existing state observations; no camera/CNN input |
| Budget | 900 seconds, including training interpreter/simulator startup |
| Methods | `baseline` = FastSAC; `sem` = FastSAC + actor-only SEM |
| Seeds | Matched seeds 1, 2, 3, 4, 5 |
| Execution | One GPU, one training process at a time |
| Compilation caches | Fresh Torch/Triton cache per run; shared simulator caches |
| Logging | Local TensorBoard and files; W&B/video disabled |
| Logging/checkpoint intervals | Every 100 / 1,000 learner iterations |

Keep the hardware, package environment, code, and task configuration fixed
across methods and seeds. Warm shared simulator caches using the same smoke
checks before measured collection. Seeds are matched experimental replicates;
different methods will still produce different trajectories.

The launcher sets a 50,000-iteration ceiling; the wall-clock deadline normally
stops training first. The batch orders seeds 2–5 as follows:

| Seed | First | Second |
|---|---|---|
| 2 | FastSAC | FastSAC + SEM |
| 3 | FastSAC + SEM | FastSAC |
| 4 | FastSAC | FastSAC + SEM |
| 5 | FastSAC + SEM | FastSAC |

Seed 1 is collected separately. The remaining eight runs require two hours
of training budgets, plus final exports and cleanup.

### What the 15-minute budget means

The clock starts immediately before the training interpreter is launched.
Imports, simulator initialization, compilation, and training count toward it.
EC2 boot, package installation, and the launcher's initial provenance capture
are outside it. The learner checks the deadline between iterations and
synchronizes CUDA work for timed runs. An ongoing iteration or export can
cross the deadline; this is a cooperative stop, not a hard process kill.

`timing.json` records training elapsed time and overrun. `run.json` records
the longer process duration including final save/export/cleanup. Checkpoint
metadata records `elapsed_wall_time_s` and `within_wall_clock_budget`.
For a strict policy-at-budget comparison, select the latest checkpoint with
that flag set to true. The final checkpoint can be over budget, and periodic
checkpoint spacing can leave a substantial gap before the deadline.

## Environment setup

All commands below assume the repository root unless stated otherwise.

### AWS training machine

The collection workflow uses a **g6e.2xlarge**, with one NVIDIA L40S GPU
(48 GB), 8 vCPUs, and 64 GiB RAM, and a 200 GiB gp3 root volume. An approved
8-vCPU G/VT quota accommodates one such instance. Use the same instance type
for the baseline and SEM; changing hardware requires a new matched comparison.

The selected image for this experiment is the **Deep Learning Base AMI with
Single CUDA (Ubuntu 24.04), x86_64**, identified in the original launch choice
as `ami-03c172464e77cac8c`. AMI IDs depend on region and release; verify the
AWS publisher and x86 architecture in your region. See the
[AWS DLAMI release notes](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html).
Keep the image's GPU driver. The installer creates a separate Python 3.8
environment rather than using an AMI-provided ML environment.

Preview 4 is legacy software whose original
[NVIDIA prerequisites](https://developer.nvidia.com/isaac-gym/download)
list Ubuntu 18.04/20.04. Ubuntu 24.04 is the environment used for this project's
successful integration/full-run checks; that does not establish general
compatibility for arbitrary package versions. The historical installer has
unlocked dependencies, so preserve a working environment for all replicates.

On EC2, after uploading this checkout:

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y tmux rsync git wget curl build-essential cmake \
  libgl1 libegl1 libvulkan1 libglib2.0-0t64 libxrender1 libxi6 libxrandr2

tmux new -s sem
cd ~/holosoma-cmu16831
set -o pipefail
bash scripts/setup_isaacgym.sh 2>&1 | tee setup-isaacgym.log
source scripts/source_isaacgym_setup.sh
```

`setup_isaacgym.sh` creates Python 3.8 environment `hsgym` under
`~/.holosoma_deps/miniconda3`, installs Isaac Gym Preview 4 and this checkout,
and writes a setup sentinel under `~/.holosoma_deps`. Source
`source_isaacgym_setup.sh` in each new training shell. Do not replace it with
the IsaacSim setup or a plotting environment. Inspect setup failures before
launching experiments; long streams of old boto3 metadata can indicate pip
resolver backtracking. Reuse a known working dependency set rather than
changing the experimental algorithm to resolve installation problems.

Optionally capture package versions from the working training environment:

```bash
conda list --explicit > training-conda-explicit.txt
python -m pip freeze > training-pip-freeze.txt
```

These are supplementary reproducibility records; the launcher does not
automatically capture a complete dependency lock. Keep them with your results.

### Mac plotting environment: Conda

Plotting does not require Holosoma, Isaac Gym, CUDA, or loading a policy.
Use a separate Conda environment with your existing Miniconda installation:

```bash
conda create -n holosoma-plots -c conda-forge \
  python=3.11 numpy matplotlib tensorboard pyyaml -y
conda activate holosoma-plots
```

Subsequent sessions only need `conda activate holosoma-plots`.

## Scripts and end-to-end workflow

### 1. Upload the branch to EC2

Configure a local SSH alias named `holosoma-aws` with the instance IP,
`User ubuntu`, and your private-key path. Complete an interactive SSH
connection before starting unattended syncing. From the local repository:

```bash
ssh holosoma-aws 'mkdir -p ~/holosoma-cmu16831/logs/sem_experiments'
rsync -az --exclude='logs/' --exclude='aws-results/' \
  --exclude='.venv/' --exclude='.venv-plots/' --exclude='__pycache__/' \
  --exclude='.DS_Store' ./ holosoma-aws:holosoma-cmu16831/
```

The upload includes `.git`: the launcher records and verifies Git provenance.
Use a fresh destination when switching from a newer Holosoma checkout, since
rsync without deletion can leave obsolete files. Full SSH and EC2 details
are in [the EC2 guide](docs/sem_ec2.md).

### 2. Validate integration and collect seed 1: EC2

`scripts/run_sem_experiment.py` launches one timed training process. It has
only standard-library imports and records the run command, source changes,
GPU information, budget, and completion status. The defaults are the measured
4096-environment, compiled, 900-second setup; `--variant` defaults to `sem`.

First run the two small checks in the activated training environment:

```bash
python scripts/run_sem_experiment.py \
  --num-envs 64 --iterations 20 --seconds 300 --no-compile
python scripts/run_sem_experiment.py \
  --num-envs 64 --iterations 20 --seconds 300
```

These runs are marked `smoke_test: true` and excluded from the comparison.
Check each printed exit code and output directory. Then collect the seed-1
pair sequentially, if it has not already been collected:

```bash
python scripts/run_sem_experiment.py --variant baseline --seed 1 --seconds 900
python scripts/run_sem_experiment.py --variant sem --seed 1 --seconds 900
```

Each invocation creates a fresh timestamped directory; the single-run launcher
does not skip earlier successful runs. Do not repeat an already complete
seed-1 pair unintentionally. Detach tmux with Ctrl-b, then d; reconnect using
`tmux attach -t sem`. Closing SSH leaves tmux training active.

### 3. Collect only remaining pairs: EC2

`scripts/run_matched_sem_seeds.sh` activates `hsgym`, runs seeds 2–5 in the
alternating order above, and stops on the first failure. It is self-contained
apart from the existing setup and single-run launcher scripts.

```bash
bash scripts/run_matched_sem_seeds.sh --status
bash scripts/run_matched_sem_seeds.sh
```

`--status` prints completed/pending pairs without starting training. On later
invocations, completed pairs are skipped. Completion requires compatible
full-run metadata, successful exit, at least 900 seconds of recorded training
time, final nonempty PT/ONNX artifacts, and a TensorBoard event file. Smoke
tests and differently configured runs do not fill a pair.

An interrupted/failed pair receives a **fresh run and fresh budget** on the
next invocation, preserving old attempts. Existing checkpoints omit replay
contents and full RNG state, so resuming them would not reproduce an
uninterrupted timed replicate. A lock prevents duplicate batch scripts;
the script also checks for active launchers/trainers in this checkout.
Keep the source/environment fixed across retries. Duplicate completed runs
are reported, not deleted or automatically selected for plotting.

### 4. Pull results continuously: Mac

`scripts/sync_ec2_results.sh` polls every 30 seconds, pulling the remote
`~/holosoma-cmu16831/logs/sem_experiments/` into local `aws-results/`.
It retries connection failures and excludes compilation caches and temporary
checkpoint files. The script needs SSH/rsync, not a Conda environment.

In a separate local terminal, start this before measured training:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh
```

`caffeinate` stays attached to the sync process and releases its assertions
when it exits. Keep the Mac plugged in with its lid open; screen locking is
fine. The loop omits ONNX files because export is not atomic. PT checkpoints
are published atomically; console/event files are refreshed snapshots.
An optional positional argument replaces the default SSH alias.

After all training exits, stop the loop with Ctrl-C and run a final pull
that includes ONNX outputs:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh --once
```

Neither the sync script nor training completion stops EC2 billing. Stop the
instance after verifying the final download; retained EBS storage still bills.

### 5. Plot matched results: Mac

`scripts/plot_sem_comparison.py` reads JSON/YAML and TensorBoard events. It
checks that paired task/optimizer configurations agree apart from run identity
and the intended SEM flag, and that base revision, Python, simulator, and cache
policies match. It rejects missing pairs and duplicate successful runs.
Verify matching GPU hardware separately using each run's `nvidia-smi.txt`.

```bash
conda activate holosoma-plots
python scripts/plot_sem_comparison.py --input aws-results --seeds 1 2 3 4 5
```

For an exploratory plot after both seed-1 methods finish:

```bash
python scripts/plot_sem_comparison.py --input aws-results --seeds 1
```

If duplicate successful runs exist, explicitly exclude an unwanted directory
with `--exclude-run DIRECTORY_NAME` (repeatable). Other options include
`--output`, `--budget`, and `--bootstrap-samples`. The default output directory
is `aws-results/comparison/`. Rerun after additional results sync to refresh it.

| Output | Contents |
|---|---|
| `comparison_wall_time.png` / `.pdf` | Four metrics versus startup-inclusive minutes |
| `comparison_samples.png` / `.pdf` | The same metrics versus logged environment samples |
| `paired_linear_difference.png` / `.pdf` | SEM minus FastSAC linear tracking, on both axes |
| `summary.csv` | Method endpoint and normalized area-under-curve estimates |
| `paired_summary.csv` | Paired SEM-minus-baseline differences and intervals |
| `raw_metrics.csv` | In-budget observations with run/seed/step/time/sample identifiers |
| `aligned_metrics.csv` | Interpolated observations used for aggregates |
| `selected_runs.json` | Exact selected runs and analysis conventions |

For a local TensorBoard dashboard in the same Conda environment:

```bash
tensorboard --logdir aws-results --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006>; no EC2 TensorBoard port needs to be exposed.

## Run outputs and provenance

The remote launcher writes under `logs/sem_experiments/`; the sync script
mirrors its contents under local `aws-results/`. A typical run contains:

```text
TIMESTAMP_sem_seed1/
  run.json                    # Command, revision/base, seed, timing, exit status
  console.log                 # Training stdout/stderr
  nvidia-smi.txt              # GPU/driver information
  source.patch                # Tracked changes relative to 9c238cf8
  launcher.py                 # Launcher snapshot
  source/...                  # Selected SEM scripts/docs/test snapshots
  training/cmu16831-sem/EXPERIMENT_DIRECTORY/
    holosoma_config.yaml      # Saved experiment configuration
    events.out.tfevents.*     # TensorBoard metrics
    timing.json               # Training time, budget, overrun, final step
    model_XXXXXXX.pt          # Periodic/final training checkpoints
    model_XXXXXXX.onnx        # Exported deterministic policy
```

Git revision and patch capture tracked source changes, with selected SEM
files snapshotted as well; they are not a complete snapshot of all untracked
files or installed dependencies. Archive the working environment and commit
used for a final report. Results, caches, and checkpoints remain outside the
branch commit (`logs/` and `aws-results/` are ignored).

## Metrics and interpretation

The analysis uses four **training-rollout metrics**:

| TensorBoard tag | Interpretation |
|---|---|
| `Episode/rew_tracking_lin_vel` | Linear velocity tracking reward |
| `Episode/rew_tracking_ang_vel` | Angular velocity tracking reward |
| `Train/mean_episode_length` | Episode duration, converted to seconds using saved simulator/control timing |
| `Train/mean_reward` | Total episode reward |

Tracking rewards are reward signals, not velocity error in physical units.
Read their scale from this checkout's reward configuration. Prioritize linear
tracking versus elapsed time and the paired difference for the compute-budget
question; the sample axis helps separate sample efficiency from throughput.

The analysis clips observations to 900 seconds using
`Timing/elapsed_wall_time_s`, falling back to event timestamps relative to
launcher start if needed. Sample counts come from `Train/num_samples`.
For each metric/axis, aggregates use only the interval observed in **every**
selected run. Linear interpolation does not extrapolate to time zero or the
deadline. Consequently, curves can start after startup and end before minute
15. The endpoint summary refers to the end of this shared interval, and
normalized AUC is the integral over that interval divided by its length.
It is not an entire-budget AUC that fills unobserved startup gaps with zero.

Means show 95% pointwise percentile bootstrap intervals over matched seeds
(2,000 resamples, RNG seed 0). Both methods use the same resampled seed
indices; time points are not independent replicates. A single matched seed
has no uncertainty band and undefined CSV interval bounds. Five seeds still
provide limited uncertainty estimation; these intervals are exploratory,
not a significance test. Training curves do not establish held-out robustness,
real-robot transfer, or final deterministic policy quality; those require
separate controlled checkpoint evaluations.

## Tests and validation

The branch includes tests for SEM geometry/gradients, baseline preservation,
parameter counts, preset isolation, teammate task defaults, legacy CLI
construction, ONNX parity, aggregation boundaries, and paired bootstrapping:

```bash
# EC2: activated hsgym; pytest must be installed
python -m pytest src/holosoma/holosoma/agents/fast_sac/tests/test_sem.py -q

# Plotting environment; install pytest if needed
conda activate holosoma-plots
conda install -c conda-forge pytest -y
python -m pytest scripts/tests/test_sem_plotting.py -q
```

ONNX parity is skipped if its optional ONNX/ONNX Runtime dependencies are
absent. The tests are provided for execution in the appropriate environment;
this documentation does not assert that all tests have been run. Project
training and tests are not executed on the Mac during branch preparation;
local preparation checks are limited to syntax and diff validation.

For detailed collection/analysis conventions, see
[docs/sem_analysis.md](docs/sem_analysis.md). For the broader upstream training
and deployment framework, see [the upstream overview](README.md) and
[training guide](src/holosoma/README.md).
