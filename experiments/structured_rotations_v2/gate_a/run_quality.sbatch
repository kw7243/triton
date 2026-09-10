#!/usr/bin/env bash
#SBATCH --job-name=rot-v2-gate-a-quality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

export HF_HOME=/data/scratch-fast/kwen1/.cache/huggingface
export HF_DATASETS_CACHE=/data/scratch-fast/kwen1/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data/scratch-fast/kwen1/.cache/huggingface/hub
export TORCH_HOME=/data/scratch-fast/kwen1/.cache/torch
export TRITON_CACHE_DIR=/data/scratch-fast/kwen1/.cache/triton/structured-rotations-v2-gate-a
export XDG_CACHE_HOME=/data/scratch-fast/kwen1/.cache
export PYTHONNOUSERSITE=1

stage_root="$(pwd -P)"
staging_parent=/data/scratch-fast/kwen1/structured-rotations-v2/staging
if [[ "$(dirname "$stage_root")" != "$staging_parent" ]]; then
  echo "job working directory is not a direct child of the frozen staging parent" >&2
  exit 2
fi
if [[ ! -d "$stage_root/.git" || -f "$stage_root/.git" ]]; then
  echo "job working directory is not a self-contained repository stage" >&2
  exit 2
fi
python_bin=/data/scratch-fast/kwen1/structured-rotations-v2/envs/gate-a-quality/bin/python
retained_root=/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-quality-r1
result_directory="$retained_root/runs/slurm-$SLURM_JOB_ID"

mkdir -p "$retained_root/runs"

hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"$python_bin" -c 'import pyarrow, torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__, pyarrow.__version__); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1'

exec "$python_bin" \
  "$stage_root/experiments/structured_rotations_v2/gate_a/quality.py" \
  --contract "$stage_root/experiments/structured_rotations_v2/gate_a/contract.json" \
  --result-directory "$result_directory"
