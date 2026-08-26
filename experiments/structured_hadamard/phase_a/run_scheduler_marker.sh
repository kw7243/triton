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

printf 'rot-scheduler-marker-r1 marker=e27feb7a0c3cd46a8a4bc35c70c87591 job_id=%s stage=%s\n' "$SLURM_JOB_ID" "$RESEARCH_REPRO_STAGED_DIR"
exit 0
