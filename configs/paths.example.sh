# Copy to configs/paths.sh and edit locally. paths.sh is excluded from Git.
# PROJECT_ROOT must contain data/ and pretrained_weights/ (see README.md, dataset setup).
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/private_workspace}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export VAST_DEVICE="${VAST_DEVICE:-cuda:0}"
# Optional Conda activation for batch jobs. Leave empty to use the current environment.
export CONDA_SH="${CONDA_SH:-}"
export CONDA_ENV="${CONDA_ENV:-}"
