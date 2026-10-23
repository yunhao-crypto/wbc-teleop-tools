"""Convert HoloSMPL frames to the Pico-like HoloRetarget input schema."""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.func
def _qmul_wxyz(
    aw: float,
    ax: float,
    ay: float,
    az: float,
    bw: float,
    bx: float,
    by: float,
    bz: float,
):
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@wp.func
def _qnorm_wxyz(qw: float, qx: float, qy: float, qz: float):
    norm = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    inv_norm = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-8))
    return qw * inv_norm, qx * inv_norm, qy * inv_norm, qz * inv_norm


@wp.func
def _axis_angle_to_quat_wxyz(x: float, y: float, z: float):
    theta2 = x * x + y * y + z * z
    if theta2 < wp.float32(1.0e-8):
        return (
            wp.float32(1.0),
            wp.float32(0.5) * x,
            wp.float32(0.5) * y,
            wp.float32(0.5) * z,
        )
    theta = wp.sqrt(theta2)
    half_theta = wp.float32(0.5) * theta
    scale = wp.sin(half_theta) / theta
    return wp.cos(half_theta), x * scale, y * scale, z * scale


@wp.func
def _qrot_wxyz(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    vx: float,
    vy: float,
    vz: float,
):
    tx = wp.float32(2.0) * (qy * vz - qz * vy)
    ty = wp.float32(2.0) * (qz * vx - qx * vz)
    tz = wp.float32(2.0) * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


@wp.func
def _match_matrix_to_quat_sign(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
):
    m00 = wp.float32(1.0) - wp.float32(2.0) * (qy * qy + qz * qz)
    m11 = wp.float32(1.0) - wp.float32(2.0) * (qx * qx + qz * qz)
    m22 = wp.float32(1.0) - wp.float32(2.0) * (qx * qx + qy * qy)
    trace = m00 + m11 + m22
    flip = wp.float32(1.0)
    if trace > wp.float32(0.0):
        if qw < wp.float32(0.0):
            flip = wp.float32(-1.0)
    elif m00 > m11 and m00 > m22:
        if qx < wp.float32(0.0):
            flip = wp.float32(-1.0)
    elif m11 > m22:
        if qy < wp.float32(0.0):
            flip = wp.float32(-1.0)
    else:
        if qz < wp.float32(0.0):
            flip = wp.float32(-1.0)
    return qw * flip, qx * flip, qy * flip, qz * flip


@wp.kernel
def _smpl_forward_kinematics_kernel(
    transl: wp.array2d(dtype=wp.float32),
    pose_aa: wp.array2d(dtype=wp.float32),
    smpl_parents: wp.array(dtype=wp.int32),
    smpl_offsets: wp.array(dtype=wp.float32),
    smpl_pos: wp.array3d(dtype=wp.float32),
    smpl_quat: wp.array3d(dtype=wp.float32),
) -> None:
    frame = wp.tid()

    for joint in range(24):
        aa = joint * 3
        lw, lx, ly, lz = _axis_angle_to_quat_wxyz(
            pose_aa[frame, aa],
            pose_aa[frame, aa + 1],
            pose_aa[frame, aa + 2],
        )
        if joint == 0:
            smpl_pos[frame, 0, 0] = transl[frame, 0]
            smpl_pos[frame, 0, 1] = transl[frame, 1]
            smpl_pos[frame, 0, 2] = transl[frame, 2]
            smpl_quat[frame, 0, 0] = lw
            smpl_quat[frame, 0, 1] = lx
            smpl_quat[frame, 0, 2] = ly
            smpl_quat[frame, 0, 3] = lz
        else:
            parent = smpl_parents[joint]
            pw = smpl_quat[frame, parent, 0]
            px = smpl_quat[frame, parent, 1]
            py = smpl_quat[frame, parent, 2]
            pz = smpl_quat[frame, parent, 3]
            gw, gx, gy, gz = _qmul_wxyz(pw, px, py, pz, lw, lx, ly, lz)
            gw, gx, gy, gz = _qnorm_wxyz(gw, gx, gy, gz)
            smpl_quat[frame, joint, 0] = gw
            smpl_quat[frame, joint, 1] = gx
            smpl_quat[frame, joint, 2] = gy
            smpl_quat[frame, joint, 3] = gz

            offset = joint * 3
            ox, oy, oz = _qrot_wxyz(
                pw,
                px,
                py,
                pz,
                smpl_offsets[offset],
                smpl_offsets[offset + 1],
                smpl_offsets[offset + 2],
            )
            smpl_pos[frame, joint, 0] = smpl_pos[frame, parent, 0] + ox
            smpl_pos[frame, joint, 1] = smpl_pos[frame, parent, 1] + oy
            smpl_pos[frame, joint, 2] = smpl_pos[frame, parent, 2] + oz


@wp.kernel
def _smpl_fk_to_pico_body_kernel(
    smpl_pos: wp.array3d(dtype=wp.float32),
    smpl_quat: wp.array3d(dtype=wp.float32),
    body_out: wp.array3d(dtype=wp.float32),
) -> None:
    frame = wp.tid()
    inv_uw = wp.float32(0.7071067811865476)
    inv_ux = wp.float32(-0.7071067811865476)
    y180_w = wp.float32(0.0)
    y180_y = wp.float32(1.0)
    zero = wp.float32(0.0)
    for joint in range(24):
        body_out[frame, joint, 0] = smpl_pos[frame, joint, 0]
        body_out[frame, joint, 1] = smpl_pos[frame, joint, 2]
        body_out[frame, joint, 2] = -smpl_pos[frame, joint, 1]

        tw, tx, ty, tz = _qmul_wxyz(
            inv_uw,
            inv_ux,
            zero,
            zero,
            smpl_quat[frame, joint, 0],
            smpl_quat[frame, joint, 1],
            smpl_quat[frame, joint, 2],
            smpl_quat[frame, joint, 3],
        )
        rw, rx, ry, rz = _qmul_wxyz(tw, tx, ty, tz, y180_w, zero, y180_y, zero)
        rw, rx, ry, rz = _qnorm_wxyz(rw, rx, ry, rz)
        rw, rx, ry, rz = _match_matrix_to_quat_sign(rw, rx, ry, rz)
        body_out[frame, joint, 3] = rx
        body_out[frame, joint, 4] = ry
        body_out[frame, joint, 5] = rz
        body_out[frame, joint, 6] = rw


def smpl_forward_kinematics_torch(
    pose_aa,
    transl,
    *,
    smpl_parents,
    smpl_rest_parent_offsets,
    validate_inputs: bool = True,
    output_positions=None,
    output_quats_wxyz=None,
):
    """Return the exact SMPL FK intermediates used by HoloSMPL-to-Pico.

    The returned tensors are native-SMPL world positions and WXYZ global
    rotations with shapes ``[T,24,3]`` and ``[T,24,4]``.  This is the single FK
    path shared by Pico packing and pure-SMPL policy features.
    """

    import torch

    wp.init()
    pose = torch.as_tensor(pose_aa, dtype=torch.float32)
    trans = torch.as_tensor(transl, dtype=torch.float32, device=pose.device)
    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"pose_aa must be [T,72], got {tuple(pose.shape)}")
    if trans.shape != (pose.shape[0], 3):
        raise ValueError(
            "transl must be [T,3], got "
            f"{tuple(trans.shape)} for T={pose.shape[0]}"
        )
    if validate_inputs and (
        not torch.isfinite(pose).all() or not torch.isfinite(trans).all()
    ):
        raise ValueError("SMPL pose/transl contains non-finite values")
    pose = pose.contiguous()
    trans = trans.contiguous()
    parents = torch.as_tensor(
        smpl_parents, dtype=torch.int32, device=pose.device
    ).contiguous()
    offsets = (
        torch.as_tensor(
            smpl_rest_parent_offsets, dtype=torch.float32, device=pose.device
        )
        .reshape(-1)
        .contiguous()
    )
    if parents.shape != (24,) or offsets.shape != (72,):
        raise ValueError(
            "SMPL skeleton must contain 24 parents and 24x3 parent offsets"
        )

    frame_count = int(pose.shape[0])
    if output_positions is None:
        positions = torch.empty(
            (frame_count, 24, 3), dtype=torch.float32, device=pose.device
        )
    else:
        positions = torch.as_tensor(
            output_positions, dtype=torch.float32, device=pose.device
        )
        if positions.shape != (frame_count, 24, 3):
            raise ValueError(
                "output_positions must be "
                f"{(frame_count, 24, 3)}, got {tuple(positions.shape)}"
            )
    if output_quats_wxyz is None:
        quats_wxyz = torch.empty(
            (frame_count, 24, 4), dtype=torch.float32, device=pose.device
        )
    else:
        quats_wxyz = torch.as_tensor(
            output_quats_wxyz, dtype=torch.float32, device=pose.device
        )
        if quats_wxyz.shape != (frame_count, 24, 4):
            raise ValueError(
                "output_quats_wxyz must be "
                f"{(frame_count, 24, 4)}, got {tuple(quats_wxyz.shape)}"
            )
    positions = positions.contiguous()
    quats_wxyz = quats_wxyz.contiguous()
    device = wp.device_from_torch(pose.device)
    launch_smpl_forward_kinematics_warp(
        pose_aa=wp.from_torch(pose, dtype=wp.float32),
        transl=wp.from_torch(trans, dtype=wp.float32),
        smpl_parents=wp.from_torch(parents, dtype=wp.int32),
        smpl_rest_parent_offsets=wp.from_torch(offsets, dtype=wp.float32),
        output_positions=wp.from_torch(positions, dtype=wp.float32),
        output_quats_wxyz=wp.from_torch(quats_wxyz, dtype=wp.float32),
        device=device,
    )
    return positions, quats_wxyz


def launch_smpl_forward_kinematics_warp(
    *,
    pose_aa,
    transl,
    smpl_parents,
    smpl_rest_parent_offsets,
    output_positions,
    output_quats_wxyz,
    device,
) -> None:
    """Launch the canonical SMPL FK kernel on existing Warp arrays."""

    wp.launch(
        _smpl_forward_kinematics_kernel,
        dim=int(pose_aa.shape[0]),
        inputs=[
            transl,
            pose_aa,
            smpl_parents,
            smpl_rest_parent_offsets,
            output_positions,
            output_quats_wxyz,
        ],
        device=device,
    )


class SmplToBodyPosesConverter:
    """Convert one SMPL frame into Pico-like ``body_poses[24, 7]``."""

    def __init__(self, retargeter) -> None:
        if retargeter._smpl_adapter is None:
            from types import SimpleNamespace

            from holomotion.src.motion_tracking.smpl_reference import (
                load_neutral_smpl_skeleton,
            )

            skeleton = load_neutral_smpl_skeleton()
            retargeter._smpl_adapter = SimpleNamespace(
                smpl_parents=skeleton.parents.numpy(),
                smpl_rest_parent_offsets=(
                    skeleton.rest_parent_offsets.numpy()
                ),
            )
        self.adapter = retargeter._smpl_adapter
        self.device = retargeter._runner.newton_solver.joint_q.device
        if not self.device.is_cuda:
            raise RuntimeError("GPU SMPL body conversion requires CUDA Warp")

        self.transl_host_wp = wp.empty(
            shape=(1, 3), dtype=wp.float32, device="cpu", pinned=True
        )
        self.pose_host_wp = wp.empty(
            shape=(1, 72), dtype=wp.float32, device="cpu", pinned=True
        )
        self.body_host_wp = wp.empty(
            shape=(1, 24, 7), dtype=wp.float32, device="cpu", pinned=True
        )
        self.transl_host = self.transl_host_wp.numpy()
        self.pose_host = self.pose_host_wp.numpy()
        self.body_host = self.body_host_wp.numpy()
        self.transl_dev = wp.empty(
            shape=(1, 3), dtype=wp.float32, device=self.device
        )
        self.pose_dev = wp.empty(
            shape=(1, 72), dtype=wp.float32, device=self.device
        )
        self.smpl_pos_dev = wp.empty(
            shape=(1, 24, 3), dtype=wp.float32, device=self.device
        )
        self.smpl_quat_dev = wp.empty(
            shape=(1, 24, 4), dtype=wp.float32, device=self.device
        )
        self.body_dev = wp.empty(
            shape=(1, 24, 7), dtype=wp.float32, device=self.device
        )
        self.parents_dev = wp.array(
            self.adapter.smpl_parents.astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.offsets_dev = wp.array(
            self.adapter.smpl_rest_parent_offsets.astype(np.float32).reshape(
                -1
            ),
            dtype=wp.float32,
            device=self.device,
        )

    def __call__(
        self,
        *,
        transl: np.ndarray,
        global_orient_aa: np.ndarray,
        body_pose_aa: np.ndarray,
    ) -> np.ndarray:
        body_pose = np.asarray(body_pose_aa, dtype=np.float32).reshape(-1)
        if body_pose.shape[0] not in {63, 69}:
            raise ValueError(
                "SMPL body pose must contain 63 or 69 values, got "
                f"{body_pose.shape[0]}"
            )
        np.copyto(
            self.transl_host[0],
            np.asarray(transl, dtype=np.float32).reshape(3),
        )
        self.pose_host.fill(0.0)
        self.pose_host[0, :3] = np.asarray(
            global_orient_aa, dtype=np.float32
        ).reshape(3)
        self.pose_host[0, 3 : 3 + body_pose.shape[0]] = body_pose
        wp.copy(self.transl_dev, self.transl_host_wp)
        wp.copy(self.pose_dev, self.pose_host_wp)
        wp.launch(
            _smpl_forward_kinematics_kernel,
            dim=1,
            inputs=[
                self.transl_dev,
                self.pose_dev,
                self.parents_dev,
                self.offsets_dev,
                self.smpl_pos_dev,
                self.smpl_quat_dev,
            ],
            device=self.device,
        )
        wp.launch(
            _smpl_fk_to_pico_body_kernel,
            dim=1,
            inputs=[
                self.smpl_pos_dev,
                self.smpl_quat_dev,
                self.body_dev,
            ],
            device=self.device,
        )
        wp.copy(self.body_host_wp, self.body_dev)
        wp.synchronize()
        return self.body_host[0].copy()


__all__ = [
    "SmplToBodyPosesConverter",
    "launch_smpl_forward_kinematics_warp",
    "smpl_forward_kinematics_torch",
]
