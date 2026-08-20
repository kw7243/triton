#!/bin/bash
#SBATCH --job-name=phase-a-hurwitz
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --qos=vision-torralba-interactive
#SBATCH --partition=vision-torralba-rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-%j.out
#SBATCH --error=/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-%j.out

set -euo pipefail

SOURCE_REPO=/data/scratch-fast/kwen1/compute-native-vq/triton
RESULT_DIR="$SOURCE_REPO/results/2026-08-20-hurwitz-decode-baseline"
PHASE_A_ROOT=/data/scratch-fast/kwen1/compute-native-vq
PYTHON="$PHASE_A_ROOT/env/phase-a/bin/python"
STAGED_REPO="$(pwd -P)"

export PIP_CACHE_DIR="$PHASE_A_ROOT/cache/pip"
export TRITON_CACHE_DIR="$PHASE_A_ROOT/cache/triton"
export XDG_CACHE_HOME="$PHASE_A_ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$PHASE_A_ROOT/cache/torchinductor"
export CUDA_CACHE_PATH="$PHASE_A_ROOT/cache/cuda"
export MPLCONFIGDIR="$PHASE_A_ROOT/cache/matplotlib"
export TMPDIR="$PHASE_A_ROOT/tmp/phase-a-$SLURM_JOB_ID"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

mkdir -p "$RESULT_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"

printf '%s\n' "$STAGED_REPO" > "$RESULT_DIR/staged_snapshot.txt"
if test -f "$STAGED_REPO/REPRODUCIBILITY_METADATA.json"; then
  cp "$STAGED_REPO/REPRODUCIBILITY_METADATA.json" "$RESULT_DIR/REPRODUCIBILITY_METADATA.json"
fi
"$PYTHON" -m pip freeze > "$RESULT_DIR/environment.lock.txt"

{
  echo "recorded_before_benchmark=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname -f)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "staged_repo=$STAGED_REPO"
  echo "source_repo=$SOURCE_REPO"
  echo "node_shared=true (one-GPU non-exclusive allocation on a seven-GPU node)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "git_status_begin"
  git status --short --branch
  echo "git_status_end"
  echo "scontrol_job_begin"
  scontrol show job "$SLURM_JOB_ID"
  echo "scontrol_job_end"
  echo "node_jobs_begin"
  squeue --noheader --nodelist "$(hostname -s)" -o "%.18i %.10u %.2t %.10M %.6D %R" || true
  echo "node_jobs_end"
  echo "gpu_query_begin"
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,memory.free,power.limit,clocks.sm,clocks.mem --format=csv,noheader
  echo "gpu_query_end"
  echo "nvidia_smi_begin"
  nvidia-smi
  echo "nvidia_smi_end"
  echo "gpu_devices_begin"
  ls -l /dev/nvidia* || true
  echo "gpu_devices_end"
  echo "host_memory_begin"
  free -h
  grep -E "^(MemTotal|MemAvailable|SwapTotal|SwapFree):" /proc/meminfo
  echo "host_memory_end"
  "$PYTHON" - <<'PY'
import torch
import triton

properties = torch.cuda.get_device_properties(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(f"python_torch={torch.__version__}")
print(f"python_triton={triton.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"gpu_name={torch.cuda.get_device_name(0)}")
print(f"compute_capability={torch.cuda.get_device_capability(0)}")
print(f"vram_total_bytes={properties.total_memory}")
print(f"vram_free_bytes={free_bytes}")
print(f"vram_visible_total_bytes={total_bytes}")
print(f"multiprocessor_count={properties.multi_processor_count}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY
} > "$RESULT_DIR/system.txt" 2>&1

cat > "$RESULT_DIR/command.txt" <<EOF
$PYTHON experiments/phase_a_decode/benchmark.py --seed 0 --s 96 192 --t-kv 4096 16384 32768 --roles 2 --kv-heads 8 --head-dim 128 --dtype fp16 --cache-modes hot cold --warmup 25 --rep 200 --outer-trials 5 --output $RESULT_DIR
EOF

"$PYTHON" experiments/phase_a_decode/benchmark.py \
  --seed 0 \
  --s 96 192 \
  --t-kv 4096 16384 32768 \
  --roles 2 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype fp16 \
  --cache-modes hot cold \
  --warmup 25 \
  --rep 200 \
  --outer-trials 5 \
  --output "$RESULT_DIR"
