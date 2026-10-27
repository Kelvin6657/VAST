#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
for stage in 0 1 2 3 4; do
    export PYTHONHASHSEED=42
    "$PYTHON_BIN" "$REPO_ROOT/VAST/progressive.py" smoke --stage "$stage" --seed 42 \
        --project-root "$PROJECT_ROOT" --device "$VAST_DEVICE"
done
