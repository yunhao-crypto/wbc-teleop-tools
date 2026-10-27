"""gmr_lite — a slim, robot-agnostic slice of GMR for the motion-stream bridge.

Vendors only the live-retarget path the bridge uses (`GeneralMotionRetargeting`
+ `XRobotStreamer`), so the `pico` extra needs just `mink mujoco scipy numpy`
instead of the full GMR package (torch/smplx/redis/opencv/imageio/…).

Per-robot tuning and the robot XML path live in the IK-config JSONs under
`configs/`, so a new or changed humanoid drops in as a config + XML with no code
edits. Robots ship under `robots/<name>/`.

Upstream: https://github.com/YanjieZe/GMR (MIT). See ../README.md for provenance.
"""

from .retarget import GeneralMotionRetargeting, resolve_config_path
from .streamer import XRobotStreamer

# Friendly robot-name → vendored config-name aliases. The bridge's --robot flag
# maps through this; pass a config name or path directly to bypass it.
ROBOT_CONFIGS = {
    "unitree_g1": "xrobot_to_g1",
    "vita02": "xrobot_to_vita02",
    "vita02_linkerhand": "xrobot_to_vita02_linkerhand",
    "vita02a": "xrobot_to_vita02a",
    "vita02a_linkerhand": "xrobot_to_vita02a_linkerhand",
    "vita02a_noshoes": "xrobot_to_vita02a_noshoes",
    # Legacy aliases — vt_human_v2 was renamed vita02 upstream (wrists added).
    "vt_human_v2": "xrobot_to_vita02",
    "vt_human_v2_linkerhand": "xrobot_to_vita02_linkerhand",
}


def config_for_robot(robot: str) -> str:
    """Resolve a robot alias to a config name, or pass through a config/path."""
    return ROBOT_CONFIGS.get(robot, robot)


__all__ = [
    "GeneralMotionRetargeting",
    "XRobotStreamer",
    "resolve_config_path",
    "config_for_robot",
    "ROBOT_CONFIGS",
]
