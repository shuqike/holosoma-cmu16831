#!/usr/bin/env bash
# Full terrain, unchanged 200 Hz physics / 50 Hz control, default randomization.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python_bin=${1:-python}
out=${2:-logs/mjwarp_nan_investigation/validation}
mkdir -p "$out"
"$python_bin" scripts/patch_mujoco_warp.py --check
for seed in 42 2718 1; do
    timeout -k 10s 900s "$python_bin" -u scripts/diagnose_mjwarp_nan.py \
        --summary-only --steps 5000 --seed "$seed" \
        --output "$out/seed${seed}_512.json" > "$out/seed${seed}_512.log" 2>&1
    tail -n 1 "$out/seed${seed}_512.log"
done
timeout -k 10s 1200s "$python_bin" -u scripts/diagnose_mjwarp_nan.py \
    --summary-only --steps 5000 --num-envs 4096 --seed 42 --action-std 0.5 \
    --output "$out/seed42_4096_stress.json" > "$out/seed42_4096_stress.log" 2>&1
tail -n 1 "$out/seed42_4096_stress.log"
