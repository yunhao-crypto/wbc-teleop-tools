#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-check}
case "$MODE" in
  check|input-check|suspended-test) shift "$(( $# > 0 ? 1 : 0 ))" ;;
  *) echo 'Usage: start_onboard_vita.sh {check|input-check|suspended-test} [options]' >&2; exit 2 ;;
esac
# Vendor environment is needed only on the robot. Restore our isolated Python paths.
set +u
source /app/script/env.sh >/dev/null
set -u
unset PYTHONPATH HOLOMOTION_REAL_ROBOT
export PYTHONPATH="$ROOT/src:$ROOT/build/pico"
export LD_LIBRARY_PATH="$ROOT/vendor_packages/ort1232/lib:/usr/hobot/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export VITA_PREPROCESS_LIBRARY="$ROOT/artifacts/preprocessing_native_arm_v2.so"
# Body31 is decoded by the pinned SDK Python decoder; old44 native is retained unused.
unset VITA_STATE_LIBRARY
mkdir -p "$ROOT/outputs" "$ROOT/logs/live"
# Keep vendor settings intact. Only our process disables automatic outgoing
# SHM allocation, whose allocator scan produced 50 ms publish stalls.
"$ROOT/venv/bin/python" "$ROOT/scripts/prepare_zenoh_session.py" \
  --source /app_param/zenoh/s100_session_peer.json5 \
  --output "$ROOT/outputs/zenoh_session.json5"
export ZENOH_SESSION_CONFIG_URI="$ROOT/outputs/zenoh_session.json5"
cd "$ROOT"
exec 9>"$ROOT/build/operator.lock"
flock -n 9 || { echo 'Another onboard operator process is running.' >&2; exit 1; }
if [[ "$MODE" == suspended-test ]]; then
  echo '吊装/支撑测试入口：保持摇杆居中；READY 后按 START 才会申请控制并发送 action。'
  export HOLOMOTION_REAL_ROBOT=1
fi
exec "$ROOT/venv/bin/python" "$ROOT/scripts/operator_service.py" "$MODE" "$@"
