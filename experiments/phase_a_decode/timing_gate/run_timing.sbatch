#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
umask 077

if [[ "$#" -ne 9 ]]; then
  echo "usage: run_timing.sbatch SOURCE STAGE RESULT MANIFEST MANIFEST_SHA256 PYTHON CACHE_ROOT TMP_CHILD ROUTE" >&2
  exit 2
fi

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SOURCE_REPO="$1"
STAGED_REPO="$2"
RESULT_DIR="$3"
MANIFEST_PATH="$4"
MANIFEST_SHA256="$5"
PYTHON="$6"
CACHE_ROOT="$7"
TMP_CHILD="$8"
LAUNCH_ROUTE="$9"
export SOURCE_REPO
export RESEARCH_REPRO_STAGED_DIR="$STAGED_REPO"
export RESULT_DIR
export PHASE_A_MANIFEST_PATH="$MANIFEST_PATH"
export PHASE_A_MANIFEST_SHA256="$MANIFEST_SHA256"
export PHASE_A_PYTHON="$PYTHON"
export PHASE_A_CACHE_ROOT="$CACHE_ROOT"
export PHASE_A_TMP_CHILD="$TMP_CHILD"
export PHASE_A_LAUNCH_ROUTE="$LAUNCH_ROUTE"
case "$LAUNCH_ROUTE" in
  batch-primary|interactive-fallback) ;;
  *) echo "invalid launch route: $LAUNCH_ROUTE" >&2; exit 2 ;;
esac
RUNTIME="$STAGED_REPO/experiments/phase_a_decode/timing_gate/runtime.py"
VALIDATOR="$STAGED_REPO/experiments/phase_a_decode/timing_gate/validate_environment.py"
BENCHMARK="$STAGED_REPO/experiments/phase_a_decode/timing_gate/run_timing.py"

printf 'launch_manifest_sha256=%s\n' "$MANIFEST_SHA256"

SOURCE_REPO=$(cd "$SOURCE_REPO" && pwd -P)
STAGED_REPO=$(cd "$STAGED_REPO" && pwd -P)
RESULT_DIR=$(cd "$RESULT_DIR" && pwd -P)
MANIFEST_PATH=$(realpath "$MANIFEST_PATH")
test -d "$STAGED_REPO/.git"
test ! -e "$STAGED_REPO/.git/commondir"
test ! -e "$STAGED_REPO/.git/objects/info/alternates"
test -f "$STAGED_REPO/REPRODUCIBILITY_METADATA.json"
test -f "$MANIFEST_PATH"
test -x "$PYTHON"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export PYTHONPATH="$STAGED_REPO/python:$STAGED_REPO/experiments/phase_a_decode"

"$PYTHON" -B "$RUNTIME" start \
  --result "$RESULT_DIR" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --source "$SOURCE_REPO" \
  --stage "$STAGED_REPO"

finalize() {
  payload_status=$?
  set +e
  "$PYTHON" -B "$RUNTIME" finalize \
    --result "$RESULT_DIR" \
    --payload-exit-code "$payload_status" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --source "$SOURCE_REPO" \
    --stage "$STAGED_REPO"
  finalizer_status=$?
  if test "$payload_status" -ne 0; then
    exit "$payload_status"
  fi
  exit "$finalizer_status"
}
trap finalize EXIT

test -n "${SLURM_TMPDIR:-}"
SLURM_TMPDIR=$(cd "$SLURM_TMPDIR" && pwd -P)
TMPDIR="$SLURM_TMPDIR/$TMP_CHILD"
test ! -e "$TMPDIR"
mkdir -m 700 "$TMPDIR"
export TMPDIR

test ! -e "$CACHE_ROOT"
mkdir -m 700 "$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -m 700 "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

"$PYTHON" -B "$VALIDATOR" \
  --manifest "$MANIFEST_PATH" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --output "$RESULT_DIR/environment_validation.json"

cd "$STAGED_REPO"
"$PYTHON" -B "$RUNTIME" payload-start \
  --result "$RESULT_DIR" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --source "$SOURCE_REPO" \
  --stage "$STAGED_REPO" \
  --route "$LAUNCH_ROUTE"
"$PYTHON" -B "$BENCHMARK" \
  --output "$RESULT_DIR" \
  --seed 0 \
  --s 96 192 \
  --t-kv 32768 \
  --roles 2 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype fp16 \
  --config b512-w8 \
  --warmup-ms 100 \
  --rep-ms 500 \
  --outer-trials 9
