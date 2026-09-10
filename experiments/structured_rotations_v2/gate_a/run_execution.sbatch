#!/bin/bash
set -euo pipefail

stage="${SLURM_SUBMIT_DIR:?Slurm submit directory is required}"
expected_parent="/data/scratch-fast/kwen1/structured-rotations-v2/staging"
case "$stage" in
  "$expected_parent"/*-code) ;;
  *) echo "refusing non-stage submit directory: $stage" >&2; exit 2 ;;
esac
test -d "$stage/.git"
test -f "$stage/REPRODUCIBILITY_METADATA.json"
test -f "$stage/STAGE_FILE_MANIFEST.json"

export PATH="/data/scratch-fast/kwen1/structured-rotations-v2/envs/gate-a-quality/bin:/usr/bin:/bin"
export HF_HOME="/data/scratch-fast/kwen1/.cache/huggingface"
export PIP_CACHE_DIR="/data/scratch-fast/kwen1/.cache/pip"
export TRITON_CACHE_DIR="/data/scratch-fast/kwen1/.cache/triton/structured-rotations-v2-gate-a2"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$stage"

retained="/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1"
result="$retained/runs/slurm-${SLURM_JOB_ID:?Slurm job id is required}"
/usr/bin/mkdir -p "$retained/runs"

cd "$stage"
exec /data/scratch-fast/kwen1/structured-rotations-v2/envs/gate-a-quality/bin/python \
  -m experiments.structured_rotations_v2.gate_a.execution \
  --contract "$stage/experiments/structured_rotations_v2/gate_a/execution_contract.json" \
  --result-directory "$result"
