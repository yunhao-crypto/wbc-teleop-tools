#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() { echo 'Usage: start_actor_motion.sh sonic|supermimic|morphmimic [check|live] [--motion-config PATH] [actor options]'; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
POLICY=${1:-}
case "$POLICY" in sonic|supermimic|morphmimic) shift ;; *) usage >&2; exit 2 ;; esac
MODE=live
if [[ "${1:-}" == check || "${1:-}" == live ]]; then MODE=$1; shift; fi
CONFIG="$ROOT/config/actor_motion.json"
OPTIONS=()
while (( $# )); do
  case "$1" in
    --motion-config) (( $# >= 2 )) || { usage >&2; exit 2; }; CONFIG=$2; shift 2 ;;
    --motion-config=*) CONFIG=${1#*=}; shift ;;
    --policy|--policy=*|--manual-pico|--wait-pico|--auto-pico|--stand-only|--input|--input=*)
      echo "Incompatible fixed-motion option: $1" >&2; exit 2 ;;
    *) OPTIONS+=("$1"); shift ;;
  esac
done
if [[ "$MODE" == check ]]; then
  exec bash "$ROOT/scripts/start_actor_pico.sh" "$POLICY" motion-check \
    --motion-config "$CONFIG" "${OPTIONS[@]}"
fi
exec bash "$ROOT/scripts/start_actor_pico.sh" "$POLICY" live \
  --auto-start --local-start --confirm-motion --motion-config "$CONFIG" "${OPTIONS[@]}"
