#!/usr/bin/env bash
# WTT entry for the separately installed native MimicLite actor/reference package.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: start_mimiclite_native.sh [options]
  --model student2000|knee-reg1000|knee-reg2000
      student2000 selects native v2 hist10 (1224 inputs), not the old 814-input model.
  --native-root ABSOLUTE_PATH   Override the installed native package directory.
  --seconds SECONDS             Duration; default 0 runs until Ctrl-C.
  --stand-only                  Stand without Pico following.
  --motion NPZ_PATH             Replay an existing NPZ instead of Pico.
  --plan                        Print the command only; do not check/start the package.
  --help, -h                    Show this help without accessing robot services.

Default package on robot67:
  student2000 / knee-reg1000: /userdata/mimiclite_native_20261117
  knee-reg2000:              /userdata/mimiclite_knee_reg2000_20261117

Normal startup delegates to that package's start.sh. Its operator lock, firmware
checks, posture/ownership/freshness guards, Pico toggle and PASSIVE cleanup remain
in force. This entry does not install/restart locomotion or run a WTT actor.
HELP
}

fail() { echo "MimicLite native: $*" >&2; exit 2; }
value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"; }

MODEL=student2000
NATIVE_ROOT=
MODEL_SET=0
ROOT_SET=0
PLAN=0
STAND=0
MOTION=0
ARGS=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --model)
      value "$@"
      [[ $MODEL_SET == 0 ]] || fail '--model may be specified only once'
      MODEL=$2; MODEL_SET=1; shift 2 ;;
    --native-root)
      value "$@"
      [[ $ROOT_SET == 0 ]] || fail '--native-root may be specified only once'
      NATIVE_ROOT=$2; ROOT_SET=1; shift 2 ;;
    --seconds)
      value "$@"
      [[ "$2" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail '--seconds must be nonnegative (0 means until Ctrl-C)'
      ARGS+=("$1" "$2"); shift 2 ;;
    --stand-only) STAND=1; ARGS+=("$1"); shift ;;
    --motion) value "$@"; MOTION=1; ARGS+=("$1" "$2"); shift 2 ;;
    --plan) PLAN=1; shift ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done
[[ $STAND == 0 || $MOTION == 0 ]] || fail '--stand-only and --motion are mutually exclusive'
case "$MODEL" in
  student2000|knee-reg1000) DEFAULT_ROOT=/userdata/mimiclite_native_20261117 ;;
  knee-reg2000) DEFAULT_ROOT=/userdata/mimiclite_knee_reg2000_20261117 ;;
  *) fail "unsupported native model: $MODEL" ;;
esac
[[ $ROOT_SET == 1 ]] || NATIVE_ROOT=$DEFAULT_ROOT
[[ "$NATIVE_ROOT" == /* ]] || fail '--native-root must be absolute'
ENTRY="$NATIVE_ROOT/start.sh"
if [[ $PLAN == 1 ]]; then
  printf '%q ' bash "$ENTRY" --model "$MODEL" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
for file in "$ENTRY" "$NATIVE_ROOT/contract_$MODEL.json" "$NATIVE_ROOT/$MODEL.onnx"; do
  [[ -f "$file" && -r "$file" ]] || fail "missing native package file: $file; see --help for package selection"
done
echo "MIMICLITE_NATIVE_ENTRY: model=$MODEL package=$NATIVE_ROOT"
exec bash "$ENTRY" --model "$MODEL" "${ARGS[@]}"
