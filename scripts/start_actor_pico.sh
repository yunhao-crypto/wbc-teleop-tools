#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
USAGE='Usage: start_actor_pico.sh [sonic|supermimic] [check|replay|state-check|shadow|configure|live|auto] [options]'
# The actor policy is always an explicit choice: a positional name, --policy,
# or the interactive menu. There is no default and no substitute policy.
POLICY=
choose_policy() {
  case "$1" in
    sonic|supermimic) ;;
    *) echo "Unknown policy '$1'; choose sonic or supermimic. $USAGE" >&2; exit 2 ;;
  esac
  if [[ -n "$POLICY" && "$POLICY" != "$1" ]]; then
    echo "Conflicting policy selections: $POLICY and $1; nothing started." >&2; exit 2
  fi
  POLICY=$1
}
if [[ "${1:-}" == sonic || "${1:-}" == supermimic ]]; then
  choose_policy "$1"; shift
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  set -- check "$@"
fi
if [[ $# == 0 || "$1" == --* ]]; then
  # Device-bound defaults are private deployment data, never cached consent.
  set -- auto --local-start "$@"
fi
MODE=$1
case "$MODE" in
  auto|check|replay|state-check|shadow|live|configure) shift ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
if [[ "${1:-}" == sonic || "${1:-}" == supermimic ]]; then
  choose_policy "$1"; shift
fi
HELP=0
REST=()
while (( $# )); do
  case "$1" in
    --policy) (( $# >= 2 )) || { echo '--policy requires sonic or supermimic' >&2; exit 2; }
              choose_policy "$2"; shift 2 ;;
    --policy=*) choose_policy "${1#--policy=}"; shift ;;
    -h|--help) HELP=1; REST+=("$1"); shift ;;
    *) REST+=("$1"); shift ;;
  esac
done
set -- ${REST[@]+"${REST[@]}"}
if [[ "$MODE" == auto ]]; then
  MODE=live; set -- --auto-start --manual-pico --wait-pico --confirm-motion "$@"
fi
if [[ "$MODE" != configure && $HELP == 0 && -z "$POLICY" ]]; then
  if [[ -t 0 && -t 2 ]]; then
    {
      echo 'Select the actor policy (no default):'
      echo '  1) sonic       GR00T G1 actor     config/actor_pico.json'
      echo '  2) supermimic  SuperMimic actor   config/actor_pico_supermimic.json'
    } >&2
    read -r -p 'Policy [1/2]: ' answer || answer=
    case "$answer" in
      1|sonic) POLICY=sonic ;;
      2|supermimic) POLICY=supermimic ;;
      *) echo 'No policy selected; nothing started.' >&2; exit 2 ;;
    esac
  else
    echo "Select the actor policy explicitly; nothing started. $USAGE" >&2; exit 2
  fi
fi
if [[ -n "$POLICY" && "$MODE" != configure ]]; then
  echo "ACTOR_POLICY: $POLICY" >&2
  set -- --policy "$POLICY" "$@"
fi
if [[ "$MODE" == state-check || "$MODE" == shadow || "$MODE" == live ]]; then
  set +u
  source /app/script/env.sh >/dev/null
  set -u
fi
unset HOLOMOTION_REAL_ROBOT
unset PYTHONHOME
export PYTHONNOUSERSITE=1
# GMR runs in the isolated input process; keep legacy native preprocessors unset.
unset VITA_PREPROCESS_LIBRARY VITA_PREPROCESS_NATIVE_STAGES VITA_STATE_LIBRARY
SDK="$ROOT/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
export PYTHONPATH="$ROOT/src:$ROOT/src/deployment/unitree_g1_ros2_29dof/src:$ROOT/build/pico"
export LD_LIBRARY_PATH="$ROOT/venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs:$ROOT/build/pico:$SDK:$SDK/lib:$SDK/SDK/arm64:/usr/hobot/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=disable
export QT_PLUGIN_PATH="$SDK/plugins" QT_QPA_PLATFORM=offscreen
mkdir -p "$ROOT/outputs" "$ROOT/build/pico/actor_xdg"
chmod 700 "$ROOT/build/pico/actor_xdg"
export XDG_RUNTIME_DIR="$ROOT/build/pico/actor_xdg"
PYTHON=${VITA_ACTOR_PYTHON:-$ROOT/venv/bin/python}
if [[ "$MODE" == configure ]]; then
  exec "$PYTHON" "$ROOT/scripts/configure_actor_s100.py" "$@"
fi
if [[ "$MODE" == state-check || "$MODE" == shadow || "$MODE" == live ]]; then
  # Use the same operator lock as the existing P1/P2/P3 entry.
  exec 9>"$ROOT/build/operator.lock"
  flock -n 9 || { echo 'Another onboard operator is running.' >&2; exit 1; }
  "$PYTHON" "$ROOT/scripts/prepare_zenoh_session.py" \
    --source /app_param/zenoh/s100_session_peer.json5 \
    --output "$ROOT/outputs/actor_pico_zenoh_session.json5"
  export ZENOH_SESSION_CONFIG_URI="$ROOT/outputs/actor_pico_zenoh_session.json5"
fi
exec "$PYTHON" "$ROOT/scripts/actor_pico.py" "$MODE" "$@"
