#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --destination DIR --motion-dir DIR --velocity-dir DIR --smpl FILE --aorta-profile FILE [--launcher-dir DIR]" >&2
}

destination=""
motion_dir=""
velocity_dir=""
smpl=""
profile=""
launcher_dir=${HOME:?HOME is required}
while (( $# > 0 )); do
  case "$1" in
    --destination) destination=$2; shift 2 ;;
    --motion-dir) motion_dir=$2; shift 2 ;;
    --velocity-dir) velocity_dir=$2; shift 2 ;;
    --smpl) smpl=$2; shift 2 ;;
    --aorta-profile) profile=$2; shift 2 ;;
    --launcher-dir) launcher_dir=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$destination" && -n "$motion_dir" && -n "$velocity_dir" && -n "$smpl" && -n "$profile" ]] || { usage; exit 2; }
for item in "$motion_dir" "$velocity_dir"; do [[ -d "$item" ]] || { echo "missing directory: $item" >&2; exit 1; }; done
for item in "$smpl" "$profile"; do [[ -f "$item" ]] || { echo "missing file: $item" >&2; exit 1; }; done
if grep -q 'REPLACE_WITH_' "$profile"; then
  echo "Aorta profile is an unconfigured SDK template: $profile" >&2
  exit 1
fi
if ! grep -Eq '"?mode"?[[:space:]]*:[[:space:]]*"client"' "$profile"; then
  echo "Aorta profile must use the vendor-approved client mode: $profile" >&2
  exit 1
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
mkdir -p "$destination/src/deployment" "$destination/models/motion_model20000" \
  "$destination/models/velocity_model17000" "$destination/models/smpl" \
  "$destination/config" "$destination/logs" \
  "$destination/src/deployment/unitree_g1_ros2_29dof"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' \
  "$repo/holomotion/" "$destination/src/holomotion/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' \
  "$repo/holosmpl/" "$destination/src/holosmpl/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' \
  "$repo/deployment/common/" "$destination/src/deployment/common/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' \
  "$repo/deployment/vita_s100_aorta_30dof/" \
  "$destination/src/deployment/vita_s100_aorta_30dof/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' \
  "$repo/deployment/unitree_g1_ros2_29dof/src/" \
  "$destination/src/deployment/unitree_g1_ros2_29dof/src/"
rsync -a "$motion_dir/" "$destination/models/motion_model20000/"
rsync -a "$velocity_dir/" "$destination/models/velocity_model17000/"
install -m 0444 "$smpl" "$destination/models/smpl/smpl_neutral.pkl"
install -m 0600 "$profile" "$destination/config/pc_session_client.json5"
mkdir -p "$launcher_dir"
install -m 0755 \
  "$repo/deployment/vita_s100_aorta_30dof/start_holomotion_vita.sh" \
  "$launcher_dir/start_holomotion_vita.sh"
install -m 0755 \
  "$repo/deployment/vita_s100_aorta_30dof/start_vita_local_pico.sh" \
  "$launcher_dir/start_vita_local_pico.sh"
install -m 0755 \
  "$repo/deployment/vita_s100_aorta_30dof/start_vita_suspended_validation.sh" \
  "$launcher_dir/start_vita_suspended_validation.sh"
echo "Installed Vita deployment at $destination"
