#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
stage="${1:-4}"
seed="${2:-42}"
case "$stage" in 0|1|2|3|4) ;; *) echo 'Stage must be 0-4' >&2; exit 2;; esac
case "$seed" in 42|123|2026) ;; *) echo 'Seed must be 42, 123 or 2026' >&2; exit 2;; esac
"$PYTHON_BIN" - "$REPO_ROOT/VAST/smoke/stage${stage}_seed42/result.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text())['status'] != 'smoke_passed':
    raise SystemExit('Run scripts/02_smoke.sh successfully before formal training.')
PY
export PYTHONHASHSEED="$seed"
"$PYTHON_BIN" "$REPO_ROOT/VAST/progressive.py" train --stage "$stage" --seed "$seed" \
    --project-root "$PROJECT_ROOT" --device "$VAST_DEVICE"
