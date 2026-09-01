#!/usr/bin/env bash
set -euo pipefail

attempt=/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979
run_state=$attempt/run-state
cd "$run_state"
sha256sum -c helper-sha256sums.txt
"$run_state/ledger.py" "$run_state/ledgers.json" salloc_attempts
exec /usr/bin/salloc \
  --account=vision-torralba-urops-meng \
  --qos=vision-torralba-interactive \
  --partition=vision-torralba-rtx3090 \
  --job-name=rot-phasea-corrected-owner \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --gres=gpu:1 \
  --mem=8G \
  --time=00:20:00 \
  "$run_state/srun-owner.sh"
