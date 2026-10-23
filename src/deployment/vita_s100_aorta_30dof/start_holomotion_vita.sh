#!/usr/bin/env bash
set -euo pipefail

root=${HOLOMOTION_VITA_ROOT:-/home/dpx/holomotion_vita}
python=${HOLOMOTION_VITA_PYTHON:-/home/dpx/miniconda3/envs/holomotion_train_isaaclab/bin/python}
sdk=${HOLOMOTION_VITA_SDK:-/home/dpx/projects/vita_holomotion_deploy_20260911/python-sdk-0.1.0-rc.2}
source_root=${HOLOMOTION_VITA_SOURCE:-$root/src}
xrobotoolkit=${HOLOMOTION_XROBO_SDK:-$root/vendor/xrobotoolkit_sdk}
cuda_runtime=${HOLOMOTION_VITA_CUDA_LIB:-$root/vendor/cuda12_ort/lib}
bundle=${HOLOMOTION_VITA_BUNDLE:-$root/release/current/bundle.yaml}
bundled_config="$(dirname "$bundle")/deployment_config/config.yaml"
default_config=$source_root/deployment/vita_s100_aorta_30dof/config/vita02a_s100.yaml
if [[ -r "$bundled_config" ]]; then default_config=$bundled_config; fi
config=${HOLOMOTION_VITA_CONFIG:-$default_config}
aorta_profile=${HOLOMOTION_VITA_AORTA_PROFILE:-$root/config/pc_session_client.json5}
command_name=${1:-}
if (( $# > 0 )); then shift; fi

[[ -x "$python" ]] || { echo "Vita Python is missing: $python" >&2; exit 1; }
[[ -d "$sdk" ]] || { echo "Vita S100 SDK is missing: $sdk" >&2; exit 1; }
[[ -d "$source_root/deployment" ]] || { echo "Vita source is missing: $source_root" >&2; exit 1; }
[[ -r "$config" ]] || { echo "Vita config is missing: $config" >&2; exit 1; }

export PYTHONPATH="$source_root:$source_root/deployment/unitree_g1_ros2_29dof/src:$xrobotoolkit/python:$sdk${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$xrobotoolkit/lib:$cuda_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HOLOMOTION_SMPL_MODEL_PATH=${HOLOMOTION_SMPL_MODEL_PATH:-$root/models/smpl/smpl_neutral.pkl}

case "$command_name" in
  check)
    [[ -r "$bundle" ]] || { echo "Vita bundle is missing: $bundle" >&2; exit 1; }
    exec "$python" -m deployment.vita_s100_aorta_30dof \
      --config "$config" check --bundle "$bundle" "$@"
    ;;
  controller-test)
    exec "$python" -m deployment.vita_s100_aorta_30dof \
      --config "$config" controller-test "$@"
    ;;
  diagnostic-monitor)
    export ZENOH_SESSION_CONFIG_URI=$aorta_profile
    [[ -r "$ZENOH_SESSION_CONFIG_URI" ]] || { echo "Aorta profile is missing: $ZENOH_SESSION_CONFIG_URI" >&2; exit 1; }
    exec "$python" -m deployment.vita_s100_aorta_30dof \
      --config "$config" diagnostic-monitor "$@"
    ;;
  teleop|suspended-validation)
    [[ -r "$bundle" ]] || { echo "Vita bundle is missing: $bundle" >&2; exit 1; }
    [[ -r "$HOLOMOTION_SMPL_MODEL_PATH" ]] || { echo "SMPL model is missing: $HOLOMOTION_SMPL_MODEL_PATH" >&2; exit 1; }
    [[ -d "$xrobotoolkit/python" && -d "$xrobotoolkit/lib" ]] || { echo "XRoboToolkit runtime is missing: $xrobotoolkit" >&2; exit 1; }
    export ZENOH_SESSION_CONFIG_URI=$aorta_profile
    [[ -r "$ZENOH_SESSION_CONFIG_URI" ]] || { echo "Aorta profile is missing: $ZENOH_SESSION_CONFIG_URI" >&2; exit 1; }
    export HOLOMOTION_REAL_ROBOT=1
    extra=()
    if [[ "$command_name" == suspended-validation ]]; then
      extra+=(--suspended-validation)
    fi
    exec "$python" -m deployment.vita_s100_aorta_30dof \
      --config "$config" teleop --bundle "$bundle" "${extra[@]}" "$@"
    ;;
  *)
    echo "Usage: $0 {check|controller-test|diagnostic-monitor|suspended-validation|teleop}" >&2
    exit 2
    ;;
esac
