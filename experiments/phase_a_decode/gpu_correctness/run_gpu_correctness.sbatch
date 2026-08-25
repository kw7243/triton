#!/bin/bash
#SBATCH --job-name=vq-phase-a-gpu-correctness-r1
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --qos=vision-torralba-interactive
#SBATCH --partition=vision-torralba-rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
umask 077

SOURCE_REPO=${SOURCE_REPO:?SOURCE_REPO must name the immutable source worktree}
STAGED_REPO=${RESEARCH_REPRO_STAGED_DIR:?RESEARCH_REPRO_STAGED_DIR must come from stage_and_run.sh}
RESULT_DIR=${RESULT_DIR:?RESULT_DIR must be the fresh durable result root}
PHASE_A_ROOT=/data/scratch-fast/kwen1/compute-native-vq
PYTHON="$PHASE_A_ROOT/env/phase-a/bin/python"

SOURCE_REPO=$(cd "$SOURCE_REPO" && pwd -P)
STAGED_REPO=$(cd "$STAGED_REPO" && pwd -P)
RESULT_DIR=$(cd "$RESULT_DIR" && pwd -P)
test "$SOURCE_REPO" = "$PHASE_A_ROOT/worktrees/vq-phase-a-gpu-correctness-r1"
test "${STAGED_REPO#"$PHASE_A_ROOT/staging/"}" != "$STAGED_REPO"
test "${RESULT_DIR#"$PHASE_A_ROOT/results/"}" != "$RESULT_DIR"
test "$SOURCE_REPO" != "$STAGED_REPO"
test -f "$STAGED_REPO/REPRODUCIBILITY_METADATA.json"
test -x "$PYTHON"

export PIP_CACHE_DIR="$PHASE_A_ROOT/cache/pip"
export TRITON_CACHE_DIR="$PHASE_A_ROOT/cache/triton"
export XDG_CACHE_HOME="$PHASE_A_ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PHASE_A_ROOT/cache/torchinductor"
export CUDA_CACHE_PATH="$PHASE_A_ROOT/cache/cuda"
export MPLCONFIGDIR="$PHASE_A_ROOT/cache/matplotlib"
export TMPDIR="$PHASE_A_ROOT/tmp/vq-phase-a-gpu-correctness-r1-$SLURM_JOB_ID"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

finalize() {
  payload_status=$?
  set +e
  "$PYTHON" - "$RESULT_DIR" "$payload_status" "$SOURCE_REPO" "$STAGED_REPO" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, exit_code, source, stage = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
correctness_path = root / "correctness.json"
correctness = None
if correctness_path.exists():
    try:
        correctness = json.loads(correctness_path.read_text())
    except Exception:
        correctness = {"status": "unreadable"}
if correctness is None:
    classification = "NO RESULT"
elif correctness.get("status") == "passed" and exit_code == 0:
    classification = "PASS"
else:
    classification = "FAIL"
files = {}
for path in sorted(root.iterdir()):
    if path.is_file() and path.name != "final_result_manifest.json":
        files[path.name] = {"bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
manifest = {
    "schema": "vq-phase-a-gpu-correctness-r1/v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "payload_exit_code": exit_code,
    "status": "passed" if exit_code == 0 else "failed",
    "correctness": classification,
    "source_repo": source,
    "staged_repo": stage,
    "result_root": str(root),
    "timing_or_decision_artifacts_present": sorted(
        name for name in ("timings.csv", "tuning.json", "trial_timings.json", "jh_speedup.png")
        if (root / name).exists()),
    "files": files,
}
temporary = root / "final_result_manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "final_result_manifest.json")
PY
  exit "$payload_status"
}
trap finalize EXIT

cp "$STAGED_REPO/REPRODUCIBILITY_METADATA.json" "$RESULT_DIR/REPRODUCIBILITY_METADATA.json"
"$PYTHON" -m pip freeze > "$RESULT_DIR/environment.lock.txt"

"$PYTHON" "$STAGED_REPO/experiments/phase_a_decode/gpu_correctness/validate_environment.py" \
  --source "$SOURCE_REPO" \
  --stage "$STAGED_REPO" \
  --result "$RESULT_DIR" \
  --output "$RESULT_DIR/environment_validation.json"

cat > "$RESULT_DIR/command.txt" <<EOF
cd $STAGED_REPO
$PYTHON experiments/phase_a_decode/benchmark.py --correctness-only --seed 0 --s 96 192 --roles 2 --kv-heads 8 --head-dim 128 --output $RESULT_DIR
EOF

cd "$STAGED_REPO"
"$PYTHON" experiments/phase_a_decode/benchmark.py \
  --correctness-only \
  --seed 0 \
  --s 96 192 \
  --roles 2 \
  --kv-heads 8 \
  --head-dim 128 \
  --output "$RESULT_DIR"

"$PYTHON" - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
forbidden = [name for name in ("timings.csv", "tuning.json", "trial_timings.json", "jh_speedup.png")
             if (root / name).exists()]
correctness = json.loads((root / "correctness.json").read_text())
metadata = json.loads((root / "run_metadata.json").read_text())
if correctness.get("status") != "passed":
    raise SystemExit("correctness did not pass")
if forbidden:
    raise SystemExit("forbidden timing artifacts exist: " + ", ".join(forbidden))
for key in ("timing_executed", "tuning_executed", "cuda_graphs_executed"):
    if metadata.get(key) is not False:
        raise SystemExit(f"{key} is not false")
PY
