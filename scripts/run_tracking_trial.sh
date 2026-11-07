#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LABEL=${1:-};case "$LABEL" in A31000|B25000) POLICY=supermimic ;; Sonic16000|Sonic20000) POLICY=sonic ;; MorphMimic72000) POLICY=morphmimic ;; *) echo 'Expected A31000, B25000, Sonic16000, Sonic20000 or MorphMimic72000' >&2; exit 2 ;; esac
shift
MODE=${1:-live};case "$MODE" in check|live) if (( $# )); then shift; fi ;; *) echo 'Expected check or live' >&2; exit 2 ;; esac
if (( $# )); then echo 'Pinned trial does not accept overrides.' >&2; exit 2; fi
TRIAL_ROOT="$ROOT/trials/tracking_20261106/$LABEL"
VERIFY_ARGS=(--root "$ROOT" --trial "$LABEL")
if [[ "$MODE" == check ]]; then VERIFY_ARGS+=(--offline-only); fi
"$ROOT/venv/bin/python" "$ROOT/scripts/verify_tracking_trial.py" "${VERIFY_ARGS[@]}"
export VITA_ACTOR_PYTHON="$TRIAL_ROOT/venv/bin/python"
LOG="$ROOT/outputs/tracking_20261106_${LABEL}_$(date +%Y%m%d_%H%M%S_%N).jsonl"
if [[ "$MODE" == check ]]; then
  exec bash "$ROOT/scripts/start_actor_motion.sh" "$POLICY" check --root "$TRIAL_ROOT" \
    --motion-config config/actor_motion_test.json --steps 100 --log "$LOG"
fi
exec bash "$ROOT/scripts/start_actor_motion.sh" "$POLICY" live --root "$TRIAL_ROOT" \
  --motion-config config/actor_motion_test.json --seconds 0 --record-observations --log "$LOG"
