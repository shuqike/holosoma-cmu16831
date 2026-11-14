#!/usr/bin/env bash
# Run on the Mac: caffeinate -is bash scripts/sync_ec2_results.sh
# After training: caffeinate -is bash scripts/sync_ec2_results.sh --once
set -u

once=false
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
local_dir="$repo_dir/aws-results"
remote=holosoma-aws
remote_dir=holosoma-cmu16831/logs/sem_experiments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) once=true; shift ;;
    --remote-dir|--local-dir)
      if [[ $# -lt 2 || -z $2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
      if [[ $1 == --remote-dir ]]; then remote_dir=$2; else local_dir=$2; fi
      shift 2 ;;
    --help|-h)
      echo 'Usage: bash scripts/sync_ec2_results.sh [--once] [--remote-dir PATH] [--local-dir PATH] [SSH_ALIAS]'
      exit 0 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) remote=$1; shift ;;
  esac
done
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
  layout_options=()
  if [[ -f "$local_dir/.legacy-rsync-excludes" ]]; then
    # These historical runs were already downloaded and moved into budget
    # folders locally. Do not recreate their former top-level locations.
    layout_options+=(--exclude-from="$local_dir/.legacy-rsync-excludes")
  fi
  rsync "${options[@]}" "${layout_options[@]}" \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3' \
    "${remote}:${remote_dir%/}/" "$local_dir/"
  sync_status=$?
  if [[ $once == true ]]; then
    exit "$sync_status"
  fi
  if [[ $sync_status != 0 ]]; then
    echo "Sync failed (exit $sync_status); retrying in 30 seconds."
  fi
  sleep 30
done
