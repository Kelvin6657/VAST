#!/usr/bin/env bash
#SBATCH --job-name=VAST_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%1
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
set -euo pipefail
REPO_ROOT="${VAST_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
task="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array job}"
stage=$((task / 3))
seeds=(42 123 2026)
seed="${seeds[$((task % 3))]}"
bash "$REPO_ROOT/scripts/03_train.sh" "$stage" "$seed"
