#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_ROOT/src"
exec "$TASK_ROOT/venv/bin/python" "$TASK_ROOT/scripts/xbox_bluetooth.py" "$@"
