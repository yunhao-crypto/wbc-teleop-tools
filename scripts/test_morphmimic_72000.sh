#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/run_tracking_trial.sh" MorphMimic72000 "$@"
