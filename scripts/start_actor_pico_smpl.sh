#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f /app/script/env.sh ]]; then
  set +u
  source /app/script/env.sh >/dev/null
  set -u
fi
unset VITA_PREPROCESS_LIBRARY VITA_PREPROCESS_NATIVE_STAGES VITA_STATE_LIBRARY
sdk="$root/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
export PYTHONPATH="$root/src:$root/build/pico${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$root/venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs:$root/build/pico:$sdk:$sdk/lib:$sdk/SDK/arm64:/usr/hobot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Existing private peer profile only; never generate or modify robot config.
export ZENOH_SESSION_CONFIG_URI="${ZENOH_SESSION_CONFIG_URI:-/app_param/zenoh/s100_session_peer.json5}"
exec "$root/venv/bin/python" "$root/scripts/pico_smpl_workflow.py" "$@"
