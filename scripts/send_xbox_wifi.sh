#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 1 ]]; then
  echo 'Usage: send_xbox_wifi.sh <Vita ASUS-5G IPv4> [WiFi interface, default wlp0s20f3] [joystick, default /dev/input/js0]' >&2
  exit 2
fi
exec python3 "$ROOT/scripts/send_xbox_wifi.py" --host "$1" --interface "${2:-wlp0s20f3}" --joystick "${3:-/dev/input/js0}" --key-file "$ROOT/config/xbox_wifi.key"
