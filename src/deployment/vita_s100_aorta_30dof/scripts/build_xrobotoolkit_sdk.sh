#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --source DIR --destination DIR --python FILE --pybind11-cmake DIR" >&2
}

source_dir=""
destination=""
python=""
pybind11_cmake=""
while (( $# > 0 )); do
  case "$1" in
    --source) source_dir=$2; shift 2 ;;
    --destination) destination=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --pybind11-cmake) pybind11_cmake=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -d "$source_dir" && -x "$python" && -d "$pybind11_cmake" ]] || {
  usage
  exit 2
}

build_root=$(mktemp -d /tmp/holomotion-xrobotoolkit.XXXXXX)
cleanup() {
  rm -rf -- "$build_root"
}
trap cleanup EXIT

cp -a "$source_dir/." "$build_root/source/"
mkdir -p "$build_root/output" "$destination/python" "$destination/lib"
(
  cd "$build_root/source"
  CMAKE_PREFIX_PATH="$pybind11_cmake" "$python" setup.py build_ext \
    --build-lib "$build_root/output"
)

mapfile -t bindings < <(
  find "$build_root/output" -maxdepth 1 -type f \
    -name 'xrobotoolkit_sdk*.so' -print
)
(( ${#bindings[@]} == 1 )) || {
  echo "expected one XRoboToolkit binding, found ${#bindings[@]}" >&2
  exit 1
}
install -m 0555 "${bindings[0]}" "$destination/python/"
install -m 0555 "$source_dir/lib/libPXREARobotSDK.so" "$destination/lib/"

LD_LIBRARY_PATH="$destination/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$destination/python${PYTHONPATH:+:$PYTHONPATH}" \
  "$python" -c "import xrobotoolkit_sdk; print(xrobotoolkit_sdk.__file__)"
