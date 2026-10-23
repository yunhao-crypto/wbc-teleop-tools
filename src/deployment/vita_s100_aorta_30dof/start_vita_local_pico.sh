#!/usr/bin/env bash
set -euo pipefail

root=${HOLOMOTION_VITA_ROOT:-/home/dpx/holomotion_vita}
runtime=${HOLOMOTION_VITA_RUNTIME:-/home/dpx/start_holomotion_vita.sh}
release=${HOLOMOTION_VITA_RELEASE:-$root/release/current}
mode=suspended-validation
case "${1:-}" in
  --check-only) mode=check; shift ;;
  --help|-h)
    echo "Usage: $0 [--check-only]"
    echo "Use the checked current Velocity17000 + Motion20000 release."
    echo "START begins supported robot validation; BACK/Ctrl-C exits."
    exit 0 ;;
  '') ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
(( $# == 0 )) || { echo "Unexpected arguments" >&2; exit 2; }
[[ -x "$runtime" && -r "$release/bundle.yaml" ]] || {
  echo "Checked Vita release/runtime is missing" >&2; exit 1;
}
if [[ "$mode" != check ]]; then
  echo "SUSPENDED VALIDATION: START begins 3 s preparation then zero-speed Velocity."
  echo "A=sticks B=Pico Motion Y=Velocity X=Idle BACK/Ctrl-C=exit"
fi
exec env HOLOMOTION_VITA_BUNDLE="$release/bundle.yaml" bash "$runtime" "$mode"
