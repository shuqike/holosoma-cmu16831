#!/usr/bin/env bash
# Run on the Mac: caffeinate -is bash scripts/sync_ec2_results.sh
# After training: caffeinate -is bash scripts/sync_ec2_results.sh --once
set -u

once=false
if [[ ${1:-} == --once ]]; then
  once=true
  shift
fi
remote=${1:-holosoma-aws}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
local_dir="$repo_dir/aws-results"
mkdir -p "$local_dir"
trap 'exit 0' INT TERM

options=(-az --itemize-changes --partial --partial-dir=.rsync-partial --timeout=60
  --exclude=.cache/ --exclude='*.tmp')
if [[ $once == false ]]; then
  # ONNX publication is not atomic; include it only in the final pull.
  options+=(--exclude='*.onnx' --exclude='*.onnx.data')
fi

while true; do
  date -u '+Sync started: %Y-%m-%dT%H:%M:%SZ'
  rsync "${options[@]}" \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3' \
    "${remote}:holosoma-cmu16831/logs/sem_experiments/" "$local_dir/"
  sync_status=$?
  if [[ $once == true ]]; then
    exit "$sync_status"
  fi
  if [[ $sync_status != 0 ]]; then
    echo "Sync failed (exit $sync_status); retrying in 30 seconds."
  fi
  sleep 30
done
