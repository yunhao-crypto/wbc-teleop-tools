#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${VITA_PYTHON:-$root/venv/bin/python}"
sdk="$root/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
pybind="$root/third_party/pico_binding"
out="$root/build/pico"
mkdir -p "$out"
[[ "$(uname -m)" == aarch64 ]]
[[ -f "$sdk/SDK/arm64/libPXREARobotSDK.so" ]] || {
  echo 'Run scripts/bootstrap_actor_s100.py first to unpack the pinned Pico runtime.' >&2
  exit 1
}
command -v g++ >/dev/null || { echo 'S100 source build requires g++ and Python 3.10 development headers.' >&2; exit 1; }
# Build the one absent runtime dependency locally; no dpkg/system install.
g++ -O2 -std=c++11 -fPIC -shared -Wl,-soname,libdouble-conversion.so.3 \
  "$root"/third_party/double-conversion/double-conversion/*.cc \
  -o "$out/libdouble-conversion.so.3"
read -r -a includes <<< "$("$python" -m pybind11 --includes)"
suffix="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
g++ -O3 -std=c++17 -shared -fPIC "${includes[@]}" \
  -I"$sdk/SDK/include" -I"$pybind/include" -I"$pybind/bindings" \
  "$pybind/bindings/py_bindings.cpp" \
  -L"$sdk/SDK/arm64" -lPXREARobotSDK \
  '-Wl,-rpath,$ORIGIN/../../vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice/SDK/arm64' \
  -o "$out/xrobotoolkit_sdk$suffix"
LD_LIBRARY_PATH="$out:$sdk:$sdk/lib:$sdk/SDK/arm64:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$out" "$python" -c 'import xrobotoolkit_sdk; assert callable(xrobotoolkit_sdk.get_body_pose_snapshot); print(xrobotoolkit_sdk.__file__)'
sha256sum "$out/libdouble-conversion.so.3" "$out/xrobotoolkit_sdk$suffix"
