#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON_BIN" "$REPO_ROOT/VAST/progressive.py" aggregate
