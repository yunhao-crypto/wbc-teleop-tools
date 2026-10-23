"""CUDA Pico body-pose to canonical SMPL conversion without robot IK."""

from __future__ import annotations

import numpy as np
import warp as wp

from holosmpl.converters.pico.adapter import PICO_SMPL_PARENTS_24
from holosmpl.converters.smpl_to_body_poses import _qmul_wxyz, _qnorm_wxyz


@wp.func
def _pico_joint_to_canonical_smpl(
    body: wp.array(dtype=wp.float32),
    joint: int,
):
    base = joint * 7
    bw, bx, by, bz = _qnorm_wxyz(
        body[base + 6],
        body[base + 3],
        body[base + 4],
        body[base + 5],
    )
    root = wp.float32(0.7071067811865476)
    tw, tx, ty, tz = _qmul_wxyz(root, root, 0.0, 0.0, bw, bx, by, bz)
    qw, qx, qy, qz = _qmul_wxyz(tw, tx, ty, tz, 0.0, 0.0, 1.0, 0.0)
    qw, qx, qy, qz = _qnorm_wxyz(qw, qx, qy, qz)
    return (
        body[base],
        -body[base + 2],
        body[base + 1],
        qw,
        qx,
        qy,
        qz,
    )


@wp.kernel
def _pico_to_smpl_pose_kernel(
    body: wp.array(dtype=wp.float32),
    smpl_parents: wp.array(dtype=wp.int32),
    pose_aa: wp.array2d(dtype=wp.float32),
    root_trans: wp.array2d(dtype=wp.float32),
    global_quat: wp.array2d(dtype=wp.float32),
) -> None:
    if wp.tid() != 0:
        return
    for joint in range(24):
        px, py, pz, qw, qx, qy, qz = _pico_joint_to_canonical_smpl(body, joint)
        qb = joint * 4
        global_quat[0, qb] = qw
        global_quat[0, qb + 1] = qx
        global_quat[0, qb + 2] = qy
        global_quat[0, qb + 3] = qz
        if joint == 0:
            root_trans[0, 0] = px
            root_trans[0, 1] = py
            root_trans[0, 2] = pz
    for joint in range(24):
        qb = joint * 4
        qw = global_quat[0, qb]
        qx = global_quat[0, qb + 1]
        qy = global_quat[0, qb + 2]
        qz = global_quat[0, qb + 3]
        parent = smpl_parents[joint]
        if parent >= 0:
            pb = parent * 4
            qw, qx, qy, qz = _qmul_wxyz(
                global_quat[0, pb],
                -global_quat[0, pb + 1],
                -global_quat[0, pb + 2],
                -global_quat[0, pb + 3],
                qw,
                qx,
                qy,
                qz,
            )
            qw, qx, qy, qz = _qnorm_wxyz(qw, qx, qy, qz)
        if qw < 0.0:
            qw, qx, qy, qz = -qw, -qx, -qy, -qz
        vector_norm = wp.sqrt(qx * qx + qy * qy + qz * qz)
        scale = wp.float32(2.0)
        if vector_norm > 1.0e-7:
            angle = 2.0 * wp.atan2(vector_norm, wp.max(qw, 0.0))
            scale = angle / vector_norm
        ab = joint * 3
        pose_aa[0, ab] = qx * scale
        pose_aa[0, ab + 1] = qy * scale
        pose_aa[0, ab + 2] = qz * scale


class PicoToSmplGpuConverter:
    """Convert one Pico ``body_poses[24,7]`` frame to formal SMPL on CUDA."""

    def __init__(self, device: str = "cuda:0") -> None:
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(
                f"Pico-to-SMPL conversion requires CUDA: {device}"
            )
        self.body_host_wp = wp.empty(
            24 * 7, dtype=wp.float32, device="cpu", pinned=True
        )
        self.body_host = self.body_host_wp.numpy()
        self.body_dev = wp.empty(24 * 7, dtype=wp.float32, device=self.device)
        self.parents_dev = wp.array(
            np.asarray(PICO_SMPL_PARENTS_24, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.pose_dev = wp.empty((1, 72), dtype=wp.float32, device=self.device)
        self.trans_dev = wp.empty((1, 3), dtype=wp.float32, device=self.device)
        self.global_quat_dev = wp.empty(
            (1, 24 * 4), dtype=wp.float32, device=self.device
        )

    def to_device(self, body_poses: np.ndarray):
        body = np.asarray(body_poses, dtype=np.float32)
        if body.shape != (24, 7):
            raise ValueError(
                f"body_poses must have shape (24,7), got {body.shape}"
            )
        if not np.isfinite(body).all():
            raise ValueError("body_poses contains NaN or Inf")
        if np.any(np.linalg.norm(body[:, 3:7], axis=-1) <= 1.0e-8):
            raise ValueError("body_poses contains a zero quaternion")
        np.copyto(self.body_host, body.reshape(-1))
        wp.copy(self.body_dev, self.body_host_wp)
        wp.launch(
            _pico_to_smpl_pose_kernel,
            dim=1,
            inputs=[
                self.body_dev,
                self.parents_dev,
                self.pose_dev,
                self.trans_dev,
                self.global_quat_dev,
            ],
            device=self.device,
        )
        return self.pose_dev, self.trans_dev

    def __call__(self, body_poses: np.ndarray):
        self.to_device(body_poses)
        return wp.to_torch(self.pose_dev), wp.to_torch(self.trans_dev)


__all__ = ["PicoToSmplGpuConverter"]
