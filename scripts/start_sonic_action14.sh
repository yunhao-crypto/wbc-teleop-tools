#!/usr/bin/env bash
# Reference-only action14 entry. --plan/--help never source device env, take
# the operator lock, start Pico, or load a model.
# With no explicit live flags, this is the one-click entry: confirm motion,
# follow the Pico head, and use the deployed tracking client in bin/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFE=0
EXPLICIT=0
for arg in "$@"; do
  case "$arg" in
    --plan|--help|-h) SAFE=1 ;;
    --confirm-motion|--bridge|--fixed-neck|--follow-head|--gripper-config) EXPLICIT=1 ;;
    --bridge=*|--gripper-config=*) EXPLICIT=1 ;;
  esac
done
if [[ "$SAFE" == 1 ]]; then
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "${PYTHON:-python3}" -m deployment.vita_s100_aorta_30dof.sonic_action14_cli "$@"
fi
if [[ "$EXPLICIT" == 0 ]]; then
  BRIDGE="$ROOT/bin/humanoid_tracking_client"
  if [[ ! -x "$BRIDGE" ]]; then
    echo "one-click bridge missing: $BRIDGE" >&2
    exit 1
  fi
  set -- --confirm-motion --follow-head --bridge "$BRIDGE" "$@"
fi
if [[ -f /app/script/env.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /app/script/env.sh >/dev/null
  set -u
fi
unset HOLOMOTION_REAL_ROBOT
unset PYTHONHOME
export PYTHONNOUSERSITE=1
unset VITA_PREPROCESS_LIBRARY VITA_PREPROCESS_NATIVE_STAGES VITA_STATE_LIBRARY
SDK="$ROOT/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
export PYTHONPATH="$ROOT/src:$ROOT/src/deployment/unitree_g1_ros2_29dof/src:$ROOT/build/pico${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs:$ROOT/build/pico:$SDK:$SDK/lib:$SDK/SDK/arm64:/usr/hobot/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=disable
export QT_PLUGIN_PATH="$SDK/plugins" QT_QPA_PLATFORM=offscreen
mkdir -p "$ROOT/build/pico/actor_xdg"
chmod 700 "$ROOT/build/pico/actor_xdg"
export XDG_RUNTIME_DIR="$ROOT/build/pico/actor_xdg"
PY="${PYTHON:-python3}"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  PY="$ROOT/venv/bin/python"
fi
LOCK=0
for arg in "$@"; do
  [[ "$arg" == --confirm-motion ]] && LOCK=1
done
if [[ "$LOCK" == 1 ]]; then
  mkdir -p "$ROOT/build" "$ROOT/outputs"
  if [[ -f "$ROOT/scripts/prepare_zenoh_session.py" && -f /app_param/zenoh/s100_session_peer.json5 ]]; then
    "$PY" "$ROOT/scripts/prepare_zenoh_session.py" \
      --source /app_param/zenoh/s100_session_peer.json5 \
      --output "$ROOT/outputs/actor_pico_zenoh_session.json5" \
      --cli-output "$ROOT/outputs/actor_pico_cli_session.json5"
    export ZENOH_SESSION_CONFIG_URI="$ROOT/outputs/actor_pico_zenoh_session.json5"
    export VITA_ACTOR_CLI_SESSION_CONFIG="$ROOT/outputs/actor_pico_cli_session.json5"
  fi
  exec 9>"$ROOT/build/operator.lock"
  flock -n 9 || { echo 'Another onboard operator or connect session is running; exit it before starting another session.' >&2; exit 1; }
  # Same lifetime as the original live entry: the vendor receiver is up before
  # policy startup and stays up until this script exits. A headset may connect
  # at any time; stand and action14 do not wait for it.
  PICO_LOG="$ROOT/outputs/pico_service_action14.log"
  # Own session so Ctrl-C reaches the Python handler first. The EXIT trap
  # stops the receiver only after that handler has finished.
  # Close the operator lock fd. Otherwise the receiver inherits it and keeps
  # the session lease after this script exits.
  (cd "$SDK" && setsid "$SDK/RoboticsServiceProcess" >>"$PICO_LOG" 2>&1 </dev/null 9>&- & echo $! >"$ROOT/outputs/pico_service_action14.pid")
  PICO_PID="$(cat "$ROOT/outputs/pico_service_action14.pid")"
  export VITA_PICO_SERVICE_PID=$PICO_PID
  echo 'PICO_WAIT: receiver starting; action14 stand does not depend on receiving Pico. Connect the headset to this robot on TCP 63901 at any time while this script is running.'
  trap 'kill "$PICO_PID" 2>/dev/null || true; wait "$PICO_PID" 2>/dev/null || true' EXIT
  trap '' INT TERM
  set +e
  "$PY" -m deployment.vita_s100_aorta_30dof.sonic_action14_cli "$@"
  status=$?
  set -e
  exit "$status"
fi
exec "$PY" -m deployment.vita_s100_aorta_30dof.sonic_action14_cli "$@"
