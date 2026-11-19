#!/usr/bin/env bash
# Robot67 SuperMimic OP3 candidate, using the existing device's v4 SDK contract.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
for option in "$@"; do
  case "$option" in
    --planner-config|--planner-config=*)
      echo 'Planner is disabled in this SuperMimic OP3 deployment.' >&2
      exit 2 ;;
  esac
done
if [[ "${1:-}" == supermimic_op3 ]]; then shift; fi
set -- supermimic_op3 "$@"
export VITA_ACTOR_PYTHON="$ROOT/venv/bin/python"
MODE=${2:-auto}
[[ "$MODE" != --* ]] || MODE=auto
HELP=0
for option in "$@"; do
  [[ "$option" != --help && "$option" != -h ]] || HELP=1
done
if [[ $HELP == 0 ]]; then
  exec 8>/userdata/wtt_operator.lock
  flock -n 8 || { echo 'Another WTT model/session is using robot67.' >&2; exit 1; }
fi
if [[ $HELP == 0 && ( "$MODE" == auto || "$MODE" == live ) ]]; then
  "$VITA_ACTOR_PYTHON" - <<'PYCOMPAT'
from locomotion_aorta import body_types
if getattr(body_types, 'BODY_CMD_MOTOR_THETA_PROTOCOL_VERSION', None) != 4:
    raise SystemExit('Robot67 motor-theta EXTERNAL requires the BodyCmd v4 SDK.')
PYCOMPAT
fi
exec bash "$ROOT/scripts/start_actor_pico.sh" "$@" --no-camera
