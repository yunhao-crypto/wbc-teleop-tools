"""Live XRoboToolkit/PICO body-tracking accessors.

Vendored from GMR (general_motion_retargeting/utils/xrobot_streamer.py) with the
single `quat_mul_np` helper it used inlined, so the bridge no longer depends on
the full GMR package (which drags in torch/smplx/redis/opencv/etc.).

XRoboToolkit stays an OPTIONAL runtime dependency: importing this module works on
machines without ``xrobotoolkit_sdk``; constructing ``XRobotStreamer`` requires
the SDK and the PC Service (a one-time machine-specific C++ build — see the
tool README).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import xrobotoolkit_sdk as xrt
except ImportError:  # pragma: no cover - hardware SDK is optional.
    xrt = None


def quat_mul_np(x, y, scalar_first=True):
    """Quaternion multiplication on arrays of quaternions.

    :param x: array of quaternions of shape (..., 4)
    :param y: array of quaternions of shape (..., 4)
    :param scalar_first: True if quaternions are in [w, x, y, z] format
    :return: quaternion product in the same format
    """
    if not scalar_first:  # convert to scalar-first
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
        x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
        x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
        x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0,
    ], axis=-1)

    if not scalar_first:
        res = res[..., [1, 2, 3, 0]]  # back to [w, x, y, z]

    return res


class XRobotStreamer:
    """Read live PICO/XRoboToolkit tracking data and convert it to GMR format."""

    body_joint_names = [
        "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
        "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
        "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder",
        "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
        "Left_Hand", "Right_Hand",
    ]

    def __init__(self):
        if xrt is None:
            raise RuntimeError(
                "xrobotoolkit_sdk is not installed; install XRoboToolkit PC Service "
                "and its Python bindings to use live PICO streaming."
            )
        xrt.init()

    def get_raw_body_data(self):
        if not xrt.is_body_data_available():
            return None
        return xrt.get_body_joints_pose()

    def get_processed_body_data(self):
        body_poses = self.get_raw_body_data()
        if body_poses is None:
            return None

        body_pose_dict = {}
        for index, joint_name in enumerate(self.body_joint_names):
            pose = body_poses[index]
            pos = [pose[0], pose[1], pose[2]]
            rot = [pose[6], pose[3], pose[4], pose[5]]  # SDK xyzw -> GMR wxyz
            # Right after a headset reconnect the SDK can report body data
            # "available" while the pose buffer is still all zeros. A
            # zero-norm quaternion is not a rotation (scipy raises on it),
            # so treat the whole frame as not-yet-arrived.
            if (rot[0] * rot[0] + rot[1] * rot[1]
                    + rot[2] * rot[2] + rot[3] * rot[3]) < 1e-6:
                return None
            body_pose_dict[joint_name] = [pos, rot]
        return self.coordinate_transform_unity_data(body_pose_dict)

    @staticmethod
    def coordinate_transform_unity_data(body_pose_dict):
        """Convert Unity-style coordinates into the right-handed GMR frame."""
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        for value in body_pose_dict.values():
            x, y, z = value[0]
            qw, qx, qy, qz = value[1]
            orientation = quat_mul_np(
                rotation_quat, np.array([qw, qx, qy, qz]), scalar_first=True
            )
            position = np.array([x, y, z]) @ rotation_matrix.T
            value[0] = position.tolist()
            value[1] = orientation.tolist()
        return body_pose_dict

    @staticmethod
    def _safe_call(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    def get_controller_data(self):
        return {
            "LeftController": {
                "index_trig": self._safe_call(xrt.get_left_trigger),
                "grip": self._safe_call(xrt.get_left_grip),
                "key_one": self._safe_call(xrt.get_X_button),
                "key_two": self._safe_call(xrt.get_Y_button),
                "axis": self._safe_call(xrt.get_left_axis),
                "axis_click": self._safe_call(xrt.get_left_axis_click),
            },
            "RightController": {
                "index_trig": self._safe_call(xrt.get_right_trigger),
                "grip": self._safe_call(xrt.get_right_grip),
                "key_one": self._safe_call(xrt.get_A_button),
                "key_two": self._safe_call(xrt.get_B_button),
                "axis": self._safe_call(xrt.get_right_axis),
                "axis_click": self._safe_call(xrt.get_right_axis_click),
            },
            "timestamp": self._safe_call(xrt.get_time_stamp_ns, 0),
        }

    def get_current_frame(self):
        body_pose_dict = self.get_processed_body_data()
        controller_data = self.get_controller_data()
        headset_pose = self._safe_call(xrt.get_headset_pose)
        return body_pose_dict, controller_data, headset_pose
