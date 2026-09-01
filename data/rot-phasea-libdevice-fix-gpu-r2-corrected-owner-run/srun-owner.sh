#!/usr/bin/env bash
set -euo pipefail

attempt=/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979
run_state=$attempt/run-state
"$run_state/ledger.py" "$run_state/ledgers.json" srun_attempts
exec /usr/bin/srun \
  --pty \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --gres=gpu:1 \
  --kill-on-bad-exit=1 \
  "$run_state/gpu-payload.sh"
