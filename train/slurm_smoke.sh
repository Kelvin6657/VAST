#!/usr/bin/env bash
#SBATCH --job-name=VAST_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -euo pipefail
REPO_ROOT="${VAST_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
bash "$REPO_ROOT/scripts/02_smoke.sh"
