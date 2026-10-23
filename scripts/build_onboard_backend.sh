#!/usr/bin/env bash
set -euo pipefail
# Build only. Does not run inference, install globally, connect to SDK, or control hardware.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$root_dir/src/deployment/vita_s100_aorta_30dof/onboard_backend"
ort_root="${ORT_ROOT:-$root_dir/packages/ort1232}"
ucp_root="${UCP_ROOT:-/usr}"
output="$root_dir/build/libvita_backend.so"
syntax=0
while (($#)); do
 case "$1" in
  --ort-root) ort_root="$2"; shift 2;;
  --ucp-root) ucp_root="$2"; shift 2;;
  --output) output="$2"; shift 2;;
  --syntax-only) syntax=1; shift;;
  *) echo "Unknown argument: $1" >&2; exit 2;;
 esac
done
compiler="${CXX:-g++}"
flags=(-std=c++17 -O3 -Wall -Wextra -Werror -ffp-contract=fast -fPIC -pthread -I"$source_dir" -I"$ort_root/include" -I"$ucp_root/include")
if [[ "$(uname -m)" == aarch64 ]]; then flags+=(-march=armv8.2-a+simd); fi
if ((syntax)); then
 "$compiler" "${flags[@]}" -fsyntax-only "$source_dir/backend.cpp"
else
 mkdir -p "$(dirname "$output")"
 ucp_lib="${UCP_LIB_DIR:-$ucp_root/hobot/lib}"
 if [[ ! -d "$ucp_lib" ]]; then ucp_lib="$ucp_root/lib"; fi
 "$compiler" "${flags[@]}" -shared "$source_dir/backend.cpp" -o "$output" \
  -L"$ort_root/lib" -L"$ucp_lib" -Wl,-rpath,"$ort_root/lib:$ucp_lib" \
  -Wl,-soname,libvita_backend.so -lonnxruntime -ldnn -lhbucp
 sha256sum "$output"
fi
