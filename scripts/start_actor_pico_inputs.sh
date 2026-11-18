#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
route="${1:-}"
case "$route" in
  pico_smpl|pico_teleop) shift ;;
  *) echo 'Usage: start_actor_pico_inputs.sh {pico_smpl|pico_teleop} [auto|doctor|replay|shadow] [options]' >&2; exit 2 ;;
esac
if [[ $# -eq 0 ]]; then set -- auto; fi
actor_python="${VITA_ACTOR_PYTHON:-$root/venv/bin/python}"
if [[ -f /app/script/env.sh ]]; then
  set +u
  source /app/script/env.sh >/dev/null
  set -u
fi
unset VITA_PREPROCESS_LIBRARY VITA_PREPROCESS_NATIVE_STAGES VITA_STATE_LIBRARY
sdk="$root/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
export PYTHONPATH="$root/src:$root/build/pico${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$root/venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs:$root/build/pico:$sdk:$sdk/lib:$sdk/SDK/arm64:/usr/hobot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$actor_python" "$root/scripts/$route.py" "$@"
