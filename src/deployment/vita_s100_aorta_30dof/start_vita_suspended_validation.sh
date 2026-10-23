#!/usr/bin/env bash
set -euo pipefail

root=${HOLOMOTION_VITA_ROOT:-/home/dpx/holomotion_vita}
runtime=${HOLOMOTION_VITA_RUNTIME:-/home/dpx/start_holomotion_vita.sh}
release=${HOLOMOTION_VITA_RELEASE:-$root/release/current}

[[ -x "$runtime" ]] || {
  echo "Vita runtime launcher is missing: $runtime" >&2
  exit 1
}
[[ -r "$release/bundle.yaml" ]] || {
  echo "Checked Vita release is missing: $release/bundle.yaml" >&2
  exit 1
}

echo "SUSPENDED VALIDATION: START sends robot commands."
echo "Keep the robot supported and the operator at the emergency stop."
echo "Default control request: 2 hours (7200 seconds), NOT unlimited; BACK/Ctrl-C exits earlier."
echo "START=Prepare then zero-speed Velocity; A=enable sticks B=Pico Y=Velocity X=Idle BACK/Ctrl-C=exit"
exec env HOLOMOTION_VITA_BUNDLE="$release/bundle.yaml" \
  bash "$runtime" suspended-validation
