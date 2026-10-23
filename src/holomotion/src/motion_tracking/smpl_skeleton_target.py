"""Pure-SMPL targets for the scale-invariant semantic skeleton reward.

Unlike :mod:`smpl_g1_task_target`, this builder does not contain robot link
names, per-part scales, virtual points, GMR costs, or SMPL-to-robot rotation
offsets.  It emits raw SMPL semantic joints plus orientations in the existing
robot-independent SMPL command frame.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from holomotion.src.motion_tracking.semantic_skeleton import (
    FOOT_INDICES,
    SEMANTIC_NODE_NAMES,
    SEMANTIC_SKELETON_VERSION,
    WRIST_INDICES,
    nominal_leg_scale,
)
from holomotion.src.motion_tracking.smpl_reference import (
    SMPL_TO_COMMAND_FRAME,
    SmplSkeletonSpec,
    load_neutral_smpl_skeleton,
    rotvec_to_matrix,
)
from holomotion.src.utils import torch_utils

SMPL_SKELETON_TARGET_VERSION: Final[str] = "smpl_skeleton_target_si_v1"
SMPL_SKELETON_TASK_FPS: Final[float] = 50.0
SMPL_SEMANTIC_JOINT_INDICES: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    4,
    5,
    10,
    11,
    9,
    16,
    17,
    18,
    19,
    20,
    21,
)
ENDPOINT_INDICES: Final[tuple[int, ...]] = FOOT_INDICES + WRIST_INDICES


@dataclass(frozen=True)
class SmplSkeletonTarget:
    """Full-clip tensors in the unanchored SMPL world frame."""

    position_h: torch.Tensor
    pelvis_rotation_xyzw_h: torch.Tensor
    torso_rotation_xyzw_h: torch.Tensor
    endpoint_rotation_xyzw_h: torch.Tensor
    head_rotation_xyzw_h: torch.Tensor
    root_linear_velocity_h: torch.Tensor
    root_angular_velocity_h: torch.Tensor
    nominal_scale_h: torch.Tensor


def _continuous_xyzw_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    quat_wxyz = torch_utils.quat_from_matrix(matrix)
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]].clone()
    if quat_xyzw.shape[0] == 0:
        return quat_xyzw
    quat_xyzw[0] = torch.where(
        quat_xyzw[0, ..., 3:4] < 0.0,
        -quat_xyzw[0],
        quat_xyzw[0],
    )
    for frame in range(1, quat_xyzw.shape[0]):
        flip = (
            torch.sum(
                quat_xyzw[frame - 1] * quat_xyzw[frame],
                dim=-1,
                keepdim=True,
            )
            < 0.0
        )
        quat_xyzw[frame] = torch.where(
            flip,
            -quat_xyzw[frame],
            quat_xyzw[frame],
        )
    return quat_xyzw


def _rotation_vector_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    quat = torch_utils.quat_from_matrix(matrix)
    quat = quat / torch.linalg.vector_norm(
        quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    scalar = quat[..., 0].clamp(0.0, 1.0)
    vector = quat[..., 1:]
    magnitude = torch.linalg.vector_norm(vector, dim=-1)
    angle = 2.0 * torch.atan2(magnitude, scalar)
    scale = torch.where(
        magnitude < 1.0e-7,
        2.0 + magnitude * magnitude / 3.0,
        angle / magnitude.clamp_min(1.0e-12),
    )
    return vector * scale[..., None]


def _finite_difference_linear(
    position: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    velocity = torch.zeros_like(position)
    if position.shape[0] <= 1:
        return velocity
    velocity[0] = (position[1] - position[0]) * fps
    velocity[-1] = (position[-1] - position[-2]) * fps
    if position.shape[0] > 2:
        velocity[1:-1] = (position[2:] - position[:-2]) * (0.5 * fps)
    return velocity


def _finite_difference_angular(
    rotation: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    velocity = torch.zeros(
        rotation.shape[:-2] + (3,),
        dtype=rotation.dtype,
        device=rotation.device,
    )
    if rotation.shape[0] <= 1:
        return velocity
    velocity[0] = (
        _rotation_vector_from_matrix(
            rotation[1] @ rotation[0].transpose(-1, -2)
        )
        * fps
    )
    velocity[-1] = (
        _rotation_vector_from_matrix(
            rotation[-1] @ rotation[-2].transpose(-1, -2)
        )
        * fps
    )
    if rotation.shape[0] > 2:
        velocity[1:-1] = _rotation_vector_from_matrix(
            rotation[2:] @ rotation[:-2].transpose(-1, -2)
        ) * (0.5 * fps)
    return velocity


class SmplSkeletonTargetBuilder:
    """Build robot-independent semantic skeleton targets from SMPL."""

    def __init__(
        self,
        *,
        fps: float = SMPL_SKELETON_TASK_FPS,
        smpl_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.skeleton = load_neutral_smpl_skeleton(smpl_model_path)
        self.nominal_scale = self._build_nominal_scale(self.skeleton)

    @property
    def feature_spec_hash(self) -> str:
        payload = "|".join(
            (
                SMPL_SKELETON_TARGET_VERSION,
                SEMANTIC_SKELETON_VERSION,
                repr(SEMANTIC_NODE_NAMES),
                repr(SMPL_SEMANTIC_JOINT_INDICES),
                repr(ENDPOINT_INDICES),
                repr(SMPL_TO_COMMAND_FRAME),
                str(self.fps),
                self.skeleton.skeleton_hash,
                repr(float(self.nominal_scale)),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_nominal_scale(
        skeleton: SmplSkeletonSpec,
    ) -> float:
        parents = skeleton.parents.to(dtype=torch.long, device="cpu")
        offsets = skeleton.rest_parent_offsets.to(
            dtype=torch.float32, device="cpu"
        )
        rest_all = torch.zeros((24, 3), dtype=torch.float32)
        for joint in range(1, 24):
            parent = int(parents[joint].item())
            rest_all[joint] = rest_all[parent] + offsets[joint]
        indices = torch.tensor(SMPL_SEMANTIC_JOINT_INDICES, dtype=torch.long)
        rest = rest_all.index_select(0, indices)
        scale = float(nominal_leg_scale(rest).item())
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid neutral SMPL nominal scale {scale}")
        return scale

    def _forward_kinematics(
        self,
        pose: torch.Tensor,
        translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = pose.shape[0]
        local_rotation = rotvec_to_matrix(pose.reshape(frames, 24, 3))
        global_rotation = torch.empty_like(local_rotation)
        relative_position = torch.empty(
            (frames, 24, 3),
            dtype=pose.dtype,
            device=pose.device,
        )
        parents = self.skeleton.parents.to(device=pose.device)
        offsets = self.skeleton.rest_parent_offsets.to(
            device=pose.device,
            dtype=pose.dtype,
        )
        global_rotation[:, 0] = local_rotation[:, 0]
        relative_position[:, 0] = 0.0
        for joint in range(1, 24):
            parent = int(parents[joint].item())
            global_rotation[:, joint] = (
                global_rotation[:, parent] @ local_rotation[:, joint]
            )
            relative_position[:, joint] = relative_position[:, parent] + (
                global_rotation[:, parent] @ offsets[joint, :, None]
            ).squeeze(-1)
        return relative_position + translation[:, None], global_rotation

    def build(
        self,
        human_pose_aa: torch.Tensor | np.ndarray,
        human_root_trans: torch.Tensor | np.ndarray,
        *,
        output_start: int = 0,
        output_length: int | None = None,
    ) -> SmplSkeletonTarget:
        pose = torch.as_tensor(
            human_pose_aa,
            dtype=torch.float32,
            device="cpu",
        )
        translation = torch.as_tensor(
            human_root_trans,
            dtype=torch.float32,
            device="cpu",
        )
        if pose.ndim != 2 or pose.shape[1] != 72:
            raise ValueError(
                f"human_pose_aa must be [T,72], got {tuple(pose.shape)}"
            )
        if translation.shape != (pose.shape[0], 3):
            raise ValueError(
                "human_root_trans must be [T,3] with the same T as pose, got "
                f"{tuple(translation.shape)} versus {tuple(pose.shape)}"
            )
        if (
            not torch.isfinite(pose).all()
            or not torch.isfinite(translation).all()
        ):
            raise ValueError("SMPL pose/root input contains non-finite values")

        total = int(pose.shape[0])
        start = int(output_start)
        length = total - start if output_length is None else int(output_length)
        end = start + length
        if start < 0 or length <= 0 or end > total:
            raise ValueError(
                "Invalid output slice "
                f"start={start}, length={length}, context={total}"
            )

        position_all, rotation_all = self._forward_kinematics(
            pose,
            translation,
        )
        indices = torch.tensor(
            SMPL_SEMANTIC_JOINT_INDICES,
            dtype=torch.long,
            device=pose.device,
        )
        position = position_all.index_select(1, indices)
        rotation = rotation_all.index_select(1, indices)

        command_frame = torch.tensor(
            SMPL_TO_COMMAND_FRAME,
            dtype=pose.dtype,
            device=pose.device,
        )
        pelvis_rotation = rotation_all[:, 0] @ command_frame
        torso_rotation = rotation_all[:, 9] @ command_frame
        head_rotation = rotation_all[:, 15] @ command_frame
        endpoint_indices = torch.tensor(
            ENDPOINT_INDICES,
            dtype=torch.long,
            device=pose.device,
        )
        endpoint_rotation = rotation.index_select(1, endpoint_indices)
        endpoint_rotation = endpoint_rotation @ command_frame

        root_linear_velocity = _finite_difference_linear(
            position_all[:, 0],
            self.fps,
        )
        root_angular_velocity = _finite_difference_angular(
            rotation_all[:, 0],
            self.fps,
        )
        scale = torch.full(
            (total, 1),
            float(self.nominal_scale),
            dtype=pose.dtype,
            device=pose.device,
        )

        return SmplSkeletonTarget(
            position_h=position[start:end].contiguous(),
            pelvis_rotation_xyzw_h=_continuous_xyzw_from_matrix(
                pelvis_rotation
            )[start:end].contiguous(),
            torso_rotation_xyzw_h=_continuous_xyzw_from_matrix(torso_rotation)[
                start:end
            ].contiguous(),
            endpoint_rotation_xyzw_h=_continuous_xyzw_from_matrix(
                endpoint_rotation
            )[start:end].contiguous(),
            head_rotation_xyzw_h=_continuous_xyzw_from_matrix(head_rotation)[
                start:end
            ].contiguous(),
            root_linear_velocity_h=root_linear_velocity[
                start:end
            ].contiguous(),
            root_angular_velocity_h=root_angular_velocity[
                start:end
            ].contiguous(),
            nominal_scale_h=scale[start:end].contiguous(),
        )


__all__ = [
    "ENDPOINT_INDICES",
    "SMPL_SEMANTIC_JOINT_INDICES",
    "SMPL_SKELETON_TARGET_VERSION",
    "SmplSkeletonTarget",
    "SmplSkeletonTargetBuilder",
]
