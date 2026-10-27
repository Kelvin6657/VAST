#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
if [[ -f "$REPO_ROOT/configs/paths.sh" ]]; then
    source "$REPO_ROOT/configs/paths.sh"
else
    source "$REPO_ROOT/configs/paths.example.sh"
fi
: "${PROJECT_ROOT:=$REPO_ROOT/private_workspace}"
: "${PYTHON_BIN:=python}"
: "${VAST_DEVICE:=cuda:0}"
if [[ -n "${CONDA_SH:-}" ]]; then
    set +u
    source "$CONDA_SH"
    set -u
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    set +u
    conda activate "$CONDA_ENV"
    set -u
fi
export PYTHONUNBUFFERED=1
