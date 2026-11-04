#!/usr/bin/env bash
set -euo pipefail
# IL post-processing otherwise fails on shared Linux hosts whose inotify
# instance quota is exhausted. These settings affect this build process only.
export DOTNET_USE_POLLING_FILE_WATCHER=${DOTNET_USE_POLLING_FILE_WATCHER:-1}
export DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE=${DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE:-false}
if (( $# < 2 || $# > 3 )); then
  echo 'Usage: build_pico_camera_viewer.sh /path/to/Unity /path/to/xrt-client [robot-IPv4]' >&2
  exit 2
fi
VIEWER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VIEWER_UNITY=$1
VIEWER_PROJECT=$(cd -- "$2" && pwd)
[[ -x "$VIEWER_UNITY" ]] || { echo 'Unity editor is missing or not executable.' >&2; exit 2; }
VIEWER_PREPARE=(--project "$VIEWER_PROJECT")
if (( $# == 3 )); then VIEWER_PREPARE+=(--robot-host "$3"); fi
python3 "$VIEWER_ROOT/scripts/prepare_pico_camera_viewer.py" "${VIEWER_PREPARE[@]}"
mkdir -p "$VIEWER_PROJECT/Builds/Android"
exec "$VIEWER_UNITY" -batchmode -nographics -quit \
  -projectPath "$VIEWER_PROJECT" -buildTarget Android \
  -executeMethod VitaCameraBuilder.BuildBatch \
  -logFile "$VIEWER_PROJECT/Builds/vita_camera_build.log"
