"""Adapter to the pinned motion-stream-bridge Vita02A GMR implementation.

The vendored solver, Unity conversion, XML and IK tuning are unchanged. This
adapter only validates raw frames, selects joints by name and anchors world
yaw at following entry. It does not calibrate an arbitrary first pose to home.
"""
import importlib
from pathlib import Path
import sys

import numpy as np


class PicoGmrRetargeter:
    def __init__(self, root, contract, math_api, settings):
        self.math = math_api
        self.contract = contract
        self.settings = settings
        vendor = Path(root)/'third_party/motion_stream_bridge'
        sys.path.insert(0, str(vendor))
        module = importlib.import_module('motion_stream_bridge.gmr_lite')
        if Path(module.__file__).resolve() != vendor.resolve()/'motion_stream_bridge/gmr_lite/__init__.py':
            raise RuntimeError('Expected the pinned motion-stream-bridge GMR package')
        self.streamer = module.XRobotStreamer
        self.engine = module.GeneralMotionRetargeting(
            src_human='xrobot', tgt_robot='vita02a',
            config=str(vendor/'motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'),
            actual_human_height=float(settings['human_height_m']), verbose=False)
        import mujoco
        self.indices = []
        for name in contract.names:
            joint = mujoco.mj_name2id(self.engine.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or self.engine.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError('GMR does not provide actor hinge joint: '+name)
            self.indices.append(int(self.engine.model.jnt_qposadr[joint]))
        # The bridge wire format permutes wrist yaw/pitch. Never use it as the
        # model action order: the qpos addresses above are resolved by name.
        self.last_qpos = None
        self.last_error = None

    def processed_body(self, body):
        body = np.asarray(body, dtype=np.float64)
        if (body.shape != (24, 7) or not np.isfinite(body).all()
                or np.any(np.linalg.norm(body[:, 3:7], axis=1) < 1e-3)):
            raise ValueError('Expected finite Pico [24,7] with valid xyzw quaternions')
        poses = {name: [row[:3].tolist(), row[[6, 3, 4, 5]].tolist()]
                 for name, row in zip(self.streamer.body_joint_names, body)}
        return self.streamer.coordinate_transform_unity_data(poses)

    def solve(self, body, *, robot_root=None):
        # robot_root is deliberately unused here. The GMR root is in its own
        # world frame; following entry aligns yaw once against current feedback.
        qpos = self.engine.retarget(self.processed_body(body), offset_to_ground=True)
        if qpos.shape != (self.engine.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError('Nonfinite/invalid GMR qpos')
        self.last_qpos = qpos.copy()
        self.last_error = max(float(self.engine.error1()), float(self.engine.error2()))
        # This is a mixed position/orientation weighted task norm, NOT radians.
        return qpos[self.indices].copy(), self.math.quaternion(qpos[3:7]), self.last_error


def create_retargeter(root, config, contract, math_api):
    settings = config['retargeting']
    if settings['implementation'] != 'motion_stream_bridge_gmr_vita02a':
        raise ValueError('Only the pinned Vita02A bridge GMR retargeter is supported')
    height = float(settings['human_height_m'])
    if not np.isfinite(height) or not .8 <= height <= 2.5:
        raise ValueError('Invalid GMR operator height in metres')
    return PicoGmrRetargeter(root, contract, math_api, settings)
