# FastSAC + actor-only SEM on EC2

The `experiment/fastsac+sem` branch is based on **Holosoma 9c238cf8** and uses
**Isaac Gym Preview 4**. Do not use the former IsaacSim setup instructions.
The SEM preset preserves the G1 FastSAC task and replaces only the last actor
SiLU with grouped softmax. Its 128 features form two simplices of size 64,
shared by the Gaussian mean and log-std heads. There are no extra parameters.
This is a dimension-preserving adaptation of SEM, not a reproduction of every
network setting in the SEM paper.

## 1. Launch one machine

Use **g6e.2xlarge**, one NVIDIA L40S (48 GB), 8 vCPUs, 64 GiB RAM, in the
region where your quota is approved. Your 8-vCPU G/VT quota accommodates one.
The selected project image is **Deep Learning Base AMI with Single CUDA
(Ubuntu 24.04), x86_64**, originally identified as
`ami-03c172464e77cac8c`. Verify the AWS publisher, architecture, region, and
release before launching: AMI IDs vary by region/release. Use the x86 image
for this instance and the legacy installer. Keep its installed GPU driver.
This project's integration and full-run checks succeeded on that setup.
NVIDIA's original Preview 4 prerequisites list Ubuntu 18.04/20.04 and Python
3.6–3.8; Ubuntu 24.04 success here is not a general compatibility guarantee.
Preserve the working environment across all matched runs.
Use a 200 GiB gp3 root EBS volume, SSH restricted to your IP, and no paid
Marketplace software. No public notebook/TensorBoard ports are needed.

Start On-Demand for setup and the first experiment. Compare the actual Spot
quote before using Spot: it is not always a meaningful discount. Losing a
short timed run to interruption complicates the comparison. Quota approval
does not guarantee capacity. A10G g5.2xlarge is a cheaper fallback, but its
24 GB GPU/32 GiB RAM leave less headroom for GPU replay and 4096 environments.
If switching hardware, re-establish the baseline on the new machine.

Planning price: g6e.2xlarge starts at about $2.24/h On-Demand; confirm your
region/AMI in the console. A 15-minute process is about $0.56 in compute,
plus cleanup. Limit total paid instance time to 40 hours (~$90), reserve
$15 for storage/network, and leave ~$45 contingency below $150. Stop the
instance during long idle periods; EBS still bills while stopped. Do not
create Savings Plans or reservations for this project.

Sources (checked November 12, 2026):
- https://aws.amazon.com/ec2/instance-types/g6e/
- https://instances.vantage.sh/aws/ec2/g6e.2xlarge
- https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html
- https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html
- https://developer.nvidia.com/isaac-gym/download

## 2. Upload the local working tree (Mac)

Commands below assume the SSH alias `holosoma-aws`. Add an entry to your
local `~/.ssh/config`, substituting your instance's public IP and key path:

```sshconfig
Host holosoma-aws
    HostName YOUR_EC2_PUBLIC_IP
    User ubuntu
    IdentityFile ~/.ssh/YOUR_EC2_KEY.pem
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ConnectTimeout 15
```

From the local fork root:

```bash
cd /path/to/holosoma-cmu16831
ssh holosoma-aws 'mkdir -p ~/holosoma-cmu16831/logs/sem_experiments'
rsync -az --exclude='logs/' --exclude='aws-results/' --exclude='.venv/' \
  --exclude='.venv-plots/' \
  --exclude='__pycache__/' --exclude='.DS_Store' \
  ./ holosoma-aws:~/holosoma-cmu16831/
```

This transfers `.git` and working-tree changes, so a push is unnecessary.
Use a fresh remote directory for this legacy checkout: rsync without deletion
can leave obsolete IsaacSim files behind in an existing newer checkout. If
that directory already holds newer source, upload into a separate directory
and adjust the paths below. Do not use `--delete`. Connect once
interactively and verify the server fingerprint before starting unattended sync.

## 3. Install once on EC2

Fastest if your teammate can provide a working image for this exact simulator
version. Otherwise use the restored legacy installer:

```bash
ssh holosoma-aws
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

The installer creates a separate **Python 3.8 `hsgym`** environment, downloads
Preview 4, and installs this checkout. An old `hssim` environment or DLAMI's
preinstalled PyTorch environment is not a substitute. Ensure setup exits
successfully; `pipefail` exposes failures through the log pipeline. Inspect the
log and resolve failures before sourcing the environment. Setup/download time costs money but
is excluded from the 15-minute training budget. The setup script accepts the
Anaconda channel terms. Review NVIDIA's Isaac Gym license before downloading.
Because this historical commit leaves many dependency versions open, prefer
your teammate's package lock/container if dependency resolution fails; do not
silently change simulator or algorithm settings to get past an installation
problem. Reuse their PyTorch version as well as their Preview 4 package.

## 4. Validate, then run seed 1 (EC2, inside tmux)

These are execution instructions for AWS; the implementation was not run locally.

The experiment includes a legacy Tyro compatibility fix: the optional scene-file
and rigid-object lists remain available through Python presets but are hidden
from CLI expansion. Their defaults remain `None`; no scene assets are added to
the locomotion task. Without this fix, parser construction can fail with
`List[SceneFileConfig] ... requires a default value`. The launcher also passes
`--algo.config.compile False` for `--no-compile`, because the pinned CLI uses
`FlagConversionOff`.

```bash
python -m pytest src/holosoma/holosoma/agents/fast_sac/tests/test_sem.py -q
python scripts/run_sem_experiment.py --num-envs 64 --iterations 20 --seconds 300 --no-compile
python scripts/run_sem_experiment.py --num-envs 64 --iterations 20 --seconds 300
```

The small run validates integration and warms shared simulator/asset caches.
It is not a measured 4096-environment result. Check that `run.json` reports
`finished`, and that `.pt`/`.onnx` files appear. If compilation/export fails,
resolve that before spending money on a full run. Then:

```bash
python scripts/run_sem_experiment.py --variant sem --seed 1 --seconds 900
```

Detach with Ctrl-b then d; reconnect with `tmux attach -t sem`. Training keeps
running when SSH disconnects. To inspect logs in a second SSH session:

```bash
find ~/holosoma-cmu16831/logs/sem_experiments -name console.log
tail -f /path/printed/above/console.log
```

The launcher selects **Isaac Gym Preview 4**, headless G1-29DoF velocity tracking, 4096 environments,
mixed terrain scale 1.0, the existing pushes/curriculum/rewards, no CNN or
cameras, and disabled W&B/video. TensorBoard remains enabled. Torch/Triton
compilation uses a fresh cache per measured run; simulator caches are shared.
These cache conventions are explicitly recorded and should be agreed with
the team for timed comparisons; they were not specified in the initial message.
For future comparisons, warm shared caches equally, use identical hardware,
and alternate method order. The run is single GPU only.

## 5. Pull outputs continuously (second local Mac terminal)

Plug the Mac into power, keep its lid open, and keep this terminal running.
The screen may turn off and you may lock the screen. `caffeinate` prevents
idle/system sleep while its child sync script is alive; it does not guarantee
network connectivity or override lid-close sleep. Its `-s` assertion applies
on AC power; `-i` prevents idle sleep. Verify with `man caffeinate` if needed.

From the local fork root, start syncing before the full experiment:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh
```

`caffeinate` wraps the long-lived sync process, including its 30-second waits.
It releases its sleep-prevention assertions when that process exits. Ctrl-C
stops the sync and releases the assertions; no separate caffeinate PID needs
to be tracked. For a different SSH alias, pass it as the script's argument.
The script saves files under `aws-results/`, prints changed paths, and retries
failed connections. It uses noninteractive SSH, so complete the initial
interactive connection first; encrypted keys should be loaded into ssh-agent.

An initial missing-directory warning is harmless until the first launcher run.
`.pt` checkpoints are published atomically; `.tmp` files are excluded. Growing
event/console files are snapshots and will refresh on subsequent pulls. ONNX
files are skipped while running because export writes them in several steps.
The sync loop retries on the next iteration if the network disconnects.
The EC2 job continues in tmux independently of local sync. Neither caffeinate
nor the sync script stops EC2 billing when the experiment ends.
Do not use `--append` or `--delete` for syncing.

When the launcher prints its exit code, stop the loop with Ctrl-C and pull
everything except compilation caches:

```bash
caffeinate -is /bin/bash scripts/sync_ec2_results.sh --once
```

Optional local dashboard, in a local Python environment with TensorBoard installed:

```bash
tensorboard --logdir aws-results --host 127.0.0.1 --port 6006
```

Open http://127.0.0.1:6006. No AWS inbound port 6006 is required.

## Outputs and timing interpretation

Each run has `run.json` (base revision 9c238cf8, current revision, simulator,
exact command, cache policy, launch time,
duration, exit status), `source.patch` relative to 9c238cf8, source snapshots, GPU information,
`console.log`, and `training/.../` containing config, TensorBoard events,
checkpoints, ONNX, and `timing.json`.

The clock starts immediately before launching the training interpreter and
includes imports, simulator startup, and compilation. It does not include
EC2 boot, package installation, or pre-launch provenance collection. The
learner stops cooperatively between iterations. An in-progress compilation,
simulation call, update, or export can exceed 900 seconds. `timing.json`
records training overrun; `run.json` includes final save/export/cleanup too.
This is not a hard process kill or a guarantee of exactly 900 seconds.

Every checkpoint records `elapsed_wall_time_s` and
`within_wall_clock_budget`. For a strict budget comparison, use the latest
checkpoint with that flag true; the final checkpoint can be over budget.
Checkpoints are periodic (1000 iterations), so the latest eligible one may
precede the deadline substantially. A stricter boundary/snapshot protocol
would be a separate experimental improvement. TensorBoard's
`Timing/elapsed_wall_time_s` includes startup; the existing internal training
timer does not. Raw event `wall_time - run.json.started_unix_s` is another
startup-inclusive axis. Report the current checkout's reward scale rather
than assuming it is the same as the paper's figure.

Final `.pt` files support weight/optimizer/curriculum loading, but replay data
and full RNG state are not checkpointed by the existing learner. Restarting
after a Spot interruption is not an uninterrupted timed replicate. Rerun that
replicate for final comparisons. Keep periodic outputs locally regardless.

For a later matched baseline and five-seed comparison:

```bash
python scripts/run_sem_experiment.py --variant baseline --seed 1 --seconds 900
bash scripts/run_matched_sem_seeds.sh --status
bash scripts/run_matched_sem_seeds.sh
```

Collect each seed-1 method only if it is still missing. The batch handles
seeds 2–5, alternates which method runs first, and skips completed full runs
on subsequent invocations. See [the analysis guide](sem_analysis.md) for
retry behavior and the Conda plotting setup, and the
[branch README](../README_FASTSAC_SEM.md) for the complete experiment overview.

After the final sync, stop the instance in the EC2 console. Delete the EBS
volume when no longer needed. A billing alert is not an automatic cost cap.
