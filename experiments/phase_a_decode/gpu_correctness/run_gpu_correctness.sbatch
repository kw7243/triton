#!/bin/bash
#SBATCH --job-name=vq-phase-a-gpu-correctness-clean-rerun

set -euo pipefail
umask 077

SOURCE_REPO=${SOURCE_REPO:?SOURCE_REPO must name the standalone immutable source}
STAGED_REPO=${RESEARCH_REPRO_STAGED_DIR:?RESEARCH_REPRO_STAGED_DIR must come from stage_and_run.sh}
RESULT_DIR=${RESULT_DIR:?RESULT_DIR must be the fresh durable result root}
MANIFEST_PATH=${PHASE_A_MANIFEST_PATH:?PHASE_A_MANIFEST_PATH is required}
MANIFEST_SHA256=${PHASE_A_MANIFEST_SHA256:?PHASE_A_MANIFEST_SHA256 is required}
PYTHON=${PHASE_A_PYTHON:?PHASE_A_PYTHON is required}
PHASE_A_ROOT=/data/scratch-fast/kwen1/compute-native-vq
RESULT_PROTOCOL="$STAGED_REPO/experiments/phase_a_decode/gpu_correctness/result_protocol.py"

SOURCE_REPO=$(cd "$SOURCE_REPO" && pwd -P)
STAGED_REPO=$(cd "$STAGED_REPO" && pwd -P)
RESULT_DIR=$(cd "$RESULT_DIR" && pwd -P)
MANIFEST_PATH=$(realpath "$MANIFEST_PATH")
test "${SOURCE_REPO#"$PHASE_A_ROOT/standalone-sources/"}" != "$SOURCE_REPO"
test "${STAGED_REPO#"$PHASE_A_ROOT/staging/"}" != "$STAGED_REPO"
test "${RESULT_DIR#"$PHASE_A_ROOT/results/"}" != "$RESULT_DIR"
test "$SOURCE_REPO" != "$STAGED_REPO"
test "$STAGED_REPO" != "$RESULT_DIR"
test -d "$STAGED_REPO/.git"
test -f "$STAGED_REPO/REPRODUCIBILITY_METADATA.json"
test -f "$MANIFEST_PATH"
test -x "$PYTHON"

export PIP_CACHE_DIR="$PHASE_A_ROOT/cache/pip"
export TRITON_CACHE_DIR="$PHASE_A_ROOT/cache/triton"
export XDG_CACHE_HOME="$PHASE_A_ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PHASE_A_ROOT/cache/torchinductor"
export CUDA_CACHE_PATH="$PHASE_A_ROOT/cache/cuda"
export MPLCONFIGDIR="$PHASE_A_ROOT/cache/matplotlib"
export TMPDIR="$PHASE_A_ROOT/tmp/vq-phase-a-clean-rerun-$SLURM_JOB_ID"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

finalize() {
  payload_status=$?
  set +e
  "$PYTHON" "$RESULT_PROTOCOL" finalize \
    --result "$RESULT_DIR" \
    --payload-exit-code "$payload_status" \
    --source "$SOURCE_REPO" \
    --stage "$STAGED_REPO" \
    --manifest-sha256 "$MANIFEST_SHA256"
  finalizer_status=$?
  if test "$payload_status" -ne 0; then
    exit "$payload_status"
  fi
  exit "$finalizer_status"
}
trap finalize EXIT

"$PYTHON" "$RESULT_PROTOCOL" start \
  --result "$RESULT_DIR" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --source "$SOURCE_REPO" \
  --stage "$STAGED_REPO"

cp "$STAGED_REPO/REPRODUCIBILITY_METADATA.json" "$RESULT_DIR/REPRODUCIBILITY_METADATA.json"

"$PYTHON" "$STAGED_REPO/experiments/phase_a_decode/gpu_correctness/validate_environment.py" \
  --source "$SOURCE_REPO" \
  --stage "$STAGED_REPO" \
  --result "$RESULT_DIR" \
  --manifest "$MANIFEST_PATH" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --output "$RESULT_DIR/environment_validation.json"

"$PYTHON" -m pip freeze > "$RESULT_DIR/environment.lock.txt"

cat > "$RESULT_DIR/command.txt" <<EOF
cd $STAGED_REPO
$PYTHON $STAGED_REPO/experiments/phase_a_decode/benchmark.py --correctness-only --seed 0 --s 96 192 --roles 2 --kv-heads 8 --head-dim 128 --output $RESULT_DIR
EOF

cd "$STAGED_REPO"
"$PYTHON" "$STAGED_REPO/experiments/phase_a_decode/benchmark.py" \
  --correctness-only \
  --seed 0 \
  --s 96 192 \
  --roles 2 \
  --kv-heads 8 \
  --head-dim 128 \
  --output "$RESULT_DIR"

"$PYTHON" "$RESULT_PROTOCOL" complete \
  --result "$RESULT_DIR" \
  --manifest-sha256 "$MANIFEST_SHA256"
