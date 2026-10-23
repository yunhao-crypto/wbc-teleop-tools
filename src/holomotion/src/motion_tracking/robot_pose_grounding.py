"""Robot-only FK used to place a reset pose above the ground.

This module deliberately contains no SMPL-to-robot decoder, IK objective, or
Robot-reference data.  It evaluates the G1 URDF at the selected reset joint
pose and returns a bounded root-height correction from the two semantic feet.
"""

from __future__ import annotations

import torch
from torch import nn

from holomotion.src.motion_tracking.robot_skeleton_adapter import (
    RobotSkeletonAdapterSpec,
)
from holomotion.src.motion_tracking.semantic_skeleton import FOOT_INDICES
from holomotion.src.training.reference_fk import URDFParser
from holomotion.src.utils import torch_utils


class _DifferentiableRobotFootFK(nn.Module):
    """Compact batched URDF FK that returns semantic foot positions."""

    def __init__(
        self,
        *,
        robot_file_path: str,
        adapter_spec: RobotSkeletonAdapterSpec,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        (
            body_names,
            parents,
            local_translation,
            local_rotation,
            _body_joint_order,
            _joint_body_index,
            _joint_axis,
            dof_names,
            dof_axis,
            body_dof_indices,
        ) = URDFParser(robot_file_path).parse()
        missing = sorted(
            set(adapter_spec.required_link_names) - set(body_names)
        )
        if missing:
            raise ValueError(
                f"Semantic links are absent from robot FK: {missing}"
            )

        self.body_names = tuple(body_names)
        self.dof_names = tuple(dof_names)
        self.num_dof = len(dof_names)
        self._body_dof_indices = tuple(tuple(x) for x in body_dof_indices)
        name_to_index = {name: index for index, name in enumerate(body_names)}
        foot_names = adapter_spec.foot_link_names
        foot_sites = tuple(
            adapter_spec.node_sites[index] for index in FOOT_INDICES
        )
        unsupported = [
            site.kind for site in foot_sites if site.kind == "midpoint"
        ]
        if unsupported:
            raise ValueError(
                "Pose grounding does not support midpoint foot sites"
            )
        self.register_buffer(
            "_foot_body_indices",
            torch.tensor(
                [name_to_index[name] for name in foot_names],
                dtype=torch.long,
                device=device,
            ),
        )
        self.register_buffer(
            "_foot_local_offsets",
            torch.tensor(
                [site.local_offset for site in foot_sites],
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer("_parents", parents.to(device=device))
        self.register_buffer(
            "_local_translation",
            local_translation.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "_local_rotation_matrix",
            torch_utils.matrix_from_quat(
                local_rotation.to(device=device, dtype=dtype)
            ),
        )
        self.register_buffer(
            "_dof_axis_unit",
            torch.nn.functional.normalize(
                dof_axis.to(device=device, dtype=dtype), dim=-1
            ),
        )

    def _dof_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        cosine = torch.cos(q)
        sine = torch.sin(q)
        one_minus_cosine = 1.0 - cosine
        x, y, z = self._dof_axis_unit.unbind(dim=-1)
        x = x[None]
        y = y[None]
        z = z[None]
        entries = (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        )
        return torch.stack(entries, dim=-1).reshape(*q.shape, 3, 3)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 2 or q.shape[-1] != self.num_dof:
            raise ValueError(
                f"q must be [B,{self.num_dof}], got {tuple(q.shape)}"
            )
        batch_size = q.shape[0]
        joint_rotation = self._dof_rotation_matrix(q)
        positions: list[torch.Tensor] = []
        rotations: list[torch.Tensor] = []
        root_position = torch.zeros(
            batch_size, 3, device=q.device, dtype=q.dtype
        )
        root_rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(
            batch_size, -1, -1
        )
        for body_index in range(len(self.body_names)):
            parent = int(self._parents[body_index].item())
            if parent < 0:
                position = root_position
                rotation = root_rotation
            else:
                parent_position = positions[parent]
                parent_rotation = rotations[parent]
                position = parent_position + torch.einsum(
                    "bij,j->bi",
                    parent_rotation,
                    self._local_translation[body_index],
                )
                rotation = (
                    parent_rotation @ self._local_rotation_matrix[body_index]
                )
                for dof_index in self._body_dof_indices[body_index]:
                    rotation = rotation @ joint_rotation[:, dof_index]
            positions.append(position)
            rotations.append(rotation)
        body_position = torch.stack(positions, dim=1)
        body_rotation = torch.stack(rotations, dim=1)
        foot_position = body_position.index_select(
            1, self._foot_body_indices
        )
        foot_rotation = body_rotation.index_select(
            1, self._foot_body_indices
        )
        return foot_position + torch.einsum(
            "bfij,fj->bfi", foot_rotation, self._foot_local_offsets
        )


class RobotPoseGroundingRuntime(nn.Module):
    """Return a bounded root-height correction for an arbitrary robot pose."""

    def __init__(
        self,
        *,
        robot_file_path: str,
        adapter_spec: RobotSkeletonAdapterSpec,
        expected_dof_names: tuple[str, ...],
        q_default_urdf: torch.Tensor,
        device: torch.device | str,
        root_clearance_m: float = 0.03,
        max_root_height_adjustment_m: float = 0.30,
    ) -> None:
        super().__init__()
        if root_clearance_m < 0.0:
            raise ValueError("root_clearance_m must be non-negative")
        if max_root_height_adjustment_m <= 0.0:
            raise ValueError(
                "max_root_height_adjustment_m must be strictly positive"
            )
        self.robot_fk = _DifferentiableRobotFootFK(
            robot_file_path=robot_file_path,
            adapter_spec=adapter_spec,
            device=device,
        )
        runtime_dof_names = tuple(expected_dof_names)
        fk_dof_names = tuple(self.robot_fk.dof_names)
        if len(set(runtime_dof_names)) != len(runtime_dof_names):
            raise ValueError(
                "Pose-grounding runtime DoF names must be unique: "
                f"runtime={runtime_dof_names}"
            )
        if len(set(fk_dof_names)) != len(fk_dof_names):
            raise ValueError(
                "Pose-grounding FK DoF names must be unique: "
                f"grounding={fk_dof_names}"
            )
        if set(fk_dof_names) != set(runtime_dof_names):
            missing_from_runtime = sorted(
                set(fk_dof_names) - set(runtime_dof_names)
            )
            missing_from_fk = sorted(
                set(runtime_dof_names) - set(fk_dof_names)
            )
            raise ValueError(
                "Pose-grounding FK and runtime must contain the same "
                "DoF names: "
                f"missing_from_runtime={missing_from_runtime}, "
                f"missing_from_fk={missing_from_fk}, "
                f"grounding={fk_dof_names}, runtime={runtime_dof_names}"
            )
        runtime_name_to_index = {
            name: index for index, name in enumerate(runtime_dof_names)
        }
        self.register_buffer(
            "_fk_from_runtime_indices",
            torch.tensor(
                [runtime_name_to_index[name] for name in fk_dof_names],
                dtype=torch.long,
                device=device,
            ),
        )
        if q_default_urdf.ndim != 1 or q_default_urdf.shape[0] != len(
            expected_dof_names
        ):
            raise ValueError(
                "q_default_urdf must be [num_dof], got "
                f"{tuple(q_default_urdf.shape)}"
            )
        self.register_buffer(
            "q_default", q_default_urdf.detach().clone().to(device)
        )
        self.root_clearance_m = float(root_clearance_m)
        self.max_root_height_adjustment_m = float(max_root_height_adjustment_m)
        with torch.no_grad():
            self.register_buffer(
                "default_foot_min",
                self.robot_fk(self._to_fk_order(self.q_default.unsqueeze(0)))
                .select(-1, 2)
                .amin(dim=-1)
                .detach(),
            )

    def _to_fk_order(self, q_runtime: torch.Tensor) -> torch.Tensor:
        """Map controller/runtime joints into URDF FK order by name."""

        return q_runtime.index_select(-1, self._fk_from_runtime_indices)

    @staticmethod
    def _matrix_from_quat_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        w, x, y, z = torch.unbind(quaternion, dim=-1)
        return torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(quaternion.shape[:-1] + (3, 3))

    @torch.no_grad()
    def forward(
        self,
        q_urdf: torch.Tensor,
        root_rotation_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        foot_local = self.robot_fk(self._to_fk_order(q_urdf))
        rotation = self._matrix_from_quat_wxyz(root_rotation_wxyz)
        foot_world_relative = torch.einsum(
            "bij,bkj->bki", rotation, foot_local
        )
        solved_foot_min = foot_world_relative[..., 2].amin(dim=-1)
        adjustment = (
            self.default_foot_min - solved_foot_min + self.root_clearance_m
        )
        return adjustment.clamp(
            -self.max_root_height_adjustment_m,
            self.max_root_height_adjustment_m,
        )


__all__ = ["RobotPoseGroundingRuntime"]
