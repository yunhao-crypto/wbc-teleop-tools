#!/usr/bin/env bash
set -euo pipefail
CAMERA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export ZENOH_SESSION_CONFIG_URI=${ZENOH_SESSION_CONFIG_URI:-/app_param/zenoh/factory.json5}
export AORTA_GROUP=default PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CAMERA_PYTHON=${VITA_CAMERA_PYTHON:-$CAMERA_ROOT/venv/bin/python}
exec "$CAMERA_PYTHON" "$CAMERA_ROOT/scripts/pico_camera_bridge.py" "$@"
