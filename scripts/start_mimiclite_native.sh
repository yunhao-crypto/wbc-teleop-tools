#!/usr/bin/env bash
# Launch a native locomotion model using the installed WTT reference environment.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: start_mimiclite_native.sh --model NAME [options]
  --model NAME                  Native package model; NAME.onnx and contract_NAME.json must exist.
  --native-root ABSOLUTE_PATH   Installed native package directory (default: $MIMICLITE_NATIVE_ROOT).
  --seconds SECONDS             Duration; default 0 runs until Ctrl-C.
  --stand-only                  Stand without Pico following.
  --motion NPZ_PATH             Replay an existing NPZ instead of Pico.
  --camera                      Relay front-camera video during Pico following.
  --check                       Verify the native bundle without robot access.
  --plan                        Print the command only; do not check/start the package.
  --help, -h                    Show this help without accessing robot services.

Pico is the default. The WTT launcher and vendor service run on CPUs 0,2,3,
the GMR thread on CPU 1, and the locomotion inference thread is pinned to CPU 4.
Set MIMICLITE_INFERENCE_CPUS=off to skip the inference pin helper. The operator
lock is shared with the other WTT launchers in this deployment.

Provision the matching native locomotion build and selected model first. This
entry verifies the running daemon's model file and requests no service restart.
HELP
}

fail() { echo "MimicLite native: $*" >&2; exit 2; }
value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"; }

MODEL=
NATIVE_ROOT=${MIMICLITE_NATIVE_ROOT:-}
MODEL_SET=0
ROOT_SET=0
PLAN=0
CHECK=0
STAND=0
MOTION=0
CAMERA_OPTIONS=(--no-camera)
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
    --stand-only) STAND=1; shift ;;
    --motion) value "$@"; MOTION=1; ARGS+=("$1" "$2"); shift 2 ;;
    --camera) CAMERA_OPTIONS=(); shift ;;
    --check) CHECK=1; shift ;;
    --plan) PLAN=1; shift ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done
[[ $STAND == 0 || $MOTION == 0 ]] || fail '--stand-only and --motion are mutually exclusive'
[[ -n "$MODEL" ]] || fail '--model is required (see --help)'
[[ "$MODEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid native model name: $MODEL"
[[ -n "$NATIVE_ROOT" ]] || fail '--native-root or MIMICLITE_NATIVE_ROOT is required'
[[ "$NATIVE_ROOT" == /* ]] || fail '--native-root must be absolute'
[[ $PLAN == 0 || $CHECK == 0 ]] || fail '--plan and --check are mutually exclusive'

WTT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_FILE="$NATIVE_ROOT/$MODEL.onnx"
CONTRACT="$NATIVE_ROOT/contract_$MODEL.json"
BRIDGE="$NATIVE_ROOT/humanoid_tracking_client"
LAUNCHER="$NATIVE_ROOT/mimiclite_native.py"
PYTHON="$WTT_ROOT/venv/bin/python"
HOST_CPUS=0,2,3
PIN_CPUS=${MIMICLITE_INFERENCE_CPUS-4}
[[ $PIN_CPUS == off || $PIN_CPUS =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] ||
  fail 'MIMICLITE_INFERENCE_CPUS must be a CPU list or off'
LOG="$WTT_ROOT/outputs/mimiclite_native_$(date +%Y%m%d_%H%M%S)_$$.jsonl"
COMMAND=("$PYTHON" "$LAUNCHER" --wtt-root "$WTT_ROOT" --bridge "$BRIDGE"
         --model-path "$MODEL_FILE" --contract "$CONTRACT" --log "$LOG" "${ARGS[@]}")
if [[ $STAND == 0 && $MOTION == 0 ]]; then
  COMMAND=("$PYTHON" "$WTT_ROOT/scripts/pico_session_services.py" "${CAMERA_OPTIONS[@]}" --
           "${COMMAND[@]}" --pico)
elif ((${#CAMERA_OPTIONS[@]} == 0)); then
  fail '--camera requires Pico mode'
fi
if [[ $PLAN == 1 ]]; then
  printf '%q ' taskset -c "$HOST_CPUS" "${COMMAND[@]}"
  printf '\n# native inference thread CPUs: %s\n' "$PIN_CPUS"
  exit 0
fi

for file in "$BRIDGE" "$LAUNCHER" "$NATIVE_ROOT/mimiclite_native_control.py" \
    "$NATIVE_ROOT/mimiclite_reference_channel.py" "$NATIVE_ROOT/mimiclite_reference_pipeline.py" \
    "$MODEL_FILE" "$CONTRACT" "$NATIVE_ROOT/$MODEL.sha256"; do
  [[ -f "$file" && -r "$file" ]] || fail "missing native bundle file: $file"
done
[[ -x $BRIDGE ]] || fail 'humanoid_tracking_client must be executable'
(
  cd "$NATIVE_ROOT"
  mapfile -t sums < "$MODEL.sha256"
  [[ ${#sums[@]} == 2 && ${sums[0]} =~ ^[a-f0-9]{64}\ \  && ${sums[1]} =~ ^[a-f0-9]{64}\ \  ]] ||
    fail 'invalid model checksum file'
  [[ ${sums[0]:66} == "$MODEL.onnx" && ${sums[1]:66} == "contract_$MODEL.json" ]] ||
    fail 'checksum file does not name the selected model and contract'
  sha256sum --check --strict --status -- "$MODEL.sha256" || fail 'model or contract checksum mismatch'
)
if [[ $CHECK == 1 ]]; then
  echo "MimicLite native bundle verified: $MODEL"
  exit 0
fi

[[ -x $PYTHON ]] || fail 'run the WTT native reference bootstrap first'
[[ -f /app/script/env.sh ]] || fail 'the S100 application environment is missing'
set +u
source /app/script/env.sh >/dev/null
set -u
unset HOLOMOTION_REAL_ROBOT PYTHONHOME VITA_PREPROCESS_LIBRARY VITA_PREPROCESS_NATIVE_STAGES VITA_STATE_LIBRARY
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable
SDK="$WTT_ROOT/vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice"
export PYTHONPATH="$WTT_ROOT/src:$WTT_ROOT/src/deployment/unitree_g1_ros2_29dof/src:$WTT_ROOT/build/pico"
export LD_LIBRARY_PATH="$WTT_ROOT/venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs:$WTT_ROOT/build/pico:$SDK:$SDK/lib:$SDK/SDK/arm64:/usr/hobot/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$SDK/plugins" QT_QPA_PLATFORM=offscreen
export XDG_RUNTIME_DIR="$WTT_ROOT/build/pico/actor_xdg"
mkdir -p "$XDG_RUNTIME_DIR" "$WTT_ROOT/outputs"
chmod 700 "$XDG_RUNTIME_DIR"

exec 9>"$WTT_ROOT/build/operator.lock"
flock -n 9 || fail 'another WTT operator session is active'
pid=$(systemctl show -p MainPID --value locomotion)
[[ $pid =~ ^[1-9][0-9]*$ ]] || fail 'native locomotion must be running'
cmp -s -- "$MODEL_FILE" "/proc/$pid/cwd/models/vita02a/actor_mimiclite.onnx" ||
  fail 'selected ONNX differs from the model under the running locomotion working directory'
"$PYTHON" "$WTT_ROOT/scripts/prepare_zenoh_session.py" \
  --source "${MIMICLITE_ZENOH_SESSION_SOURCE:-/app_param/zenoh/s100_session_peer.json5}" \
  --output "$WTT_ROOT/outputs/mimiclite_data_session.json5" \
  --cli-output "$WTT_ROOT/outputs/mimiclite_cli_session.json5"
export ZENOH_SESSION_CONFIG_URI="$WTT_ROOT/outputs/mimiclite_data_session.json5"
export VITA_ACTOR_CLI_SESSION_CONFIG="$WTT_ROOT/outputs/mimiclite_cli_session.json5"
if [[ $PIN_CPUS != off ]]; then
  taskset -c "$HOST_CPUS" "$PYTHON" "$WTT_ROOT/scripts/pin_native_inference.py" \
    --cpus "$PIN_CPUS" --launcher-pid "$$" --activation-log "$LOG" --log "${LOG%.jsonl}_pin.jsonl" </dev/null 9>&- &
fi
exec taskset -c "$HOST_CPUS" "${COMMAND[@]}"
