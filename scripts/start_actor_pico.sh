#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# == 0 ]]; then
  # Device-bound defaults are private deployment data, never cached consent.
  set -- auto --local-start
fi
MODE=$1
case "$MODE" in
  auto) shift; MODE=live; set -- --auto-start --manual-pico --wait-pico --confirm-motion "$@" ;;
  check|replay|state-check|shadow|live) shift "$(( $# > 0 ? 1 : 0 ))" ;;
  *) echo 'Usage: start_actor_pico.sh {check|replay|state-check|shadow|live|auto} [options]' >&2; exit 2 ;;
esac
if [[ "$MODE" == state-check || "$MODE" == shadow || "$MODE" == live ]]; then
  set +u
  source /app/script/env.sh >/dev/null
  set -u
fi
unset HOLOMOTION_REAL_ROBOT
# This new orientation IK uses the NumPy Pico converter on both host and S100.
unset VITA_PREPROCESS_LIBRARY VITA_PREPROCESS_NATIVE_STAGES VITA_STATE_LIBRARY
SDK="$ROOT/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
export PYTHONPATH="$ROOT/src:$ROOT/src/deployment/unitree_g1_ros2_29dof/src:$ROOT/build/pico:$ROOT/model_dependencies_readonly"
export LD_LIBRARY_PATH="$ROOT/venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs:$ROOT/build/pico:$SDK:$SDK/lib:$SDK/SDK/arm64:/usr/hobot/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export QT_PLUGIN_PATH="$SDK/plugins" QT_QPA_PLATFORM=offscreen
mkdir -p "$ROOT/outputs" "$ROOT/build/pico/actor_xdg"
chmod 700 "$ROOT/build/pico/actor_xdg"
export XDG_RUNTIME_DIR="$ROOT/build/pico/actor_xdg"
PYTHON=${VITA_ACTOR_PYTHON:-$ROOT/venv/bin/python}
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
