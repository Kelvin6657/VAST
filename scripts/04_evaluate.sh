#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
checkpoint="${1:?Usage: bash scripts/04_evaluate.sh CHECKPOINT [evaluation options]}"
shift
"$PYTHON_BIN" "$REPO_ROOT/eval/evaluate.py" --checkpoint "$checkpoint" \
    --project-root "$PROJECT_ROOT" --device "$VAST_DEVICE" "$@"
