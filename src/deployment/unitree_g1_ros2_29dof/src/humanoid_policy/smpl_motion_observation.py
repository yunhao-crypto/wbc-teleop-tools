"""Warp CUDA observation adapter for pure-SMPL motion policies."""

from __future__ import annotations

import ctypes

import numpy as np
import warp as wp

from holomotion.src.motion_tracking.reference_observation import (
    RootReferenceKinematics,
    derive_root_reference_kinematics_warp,
)
from holomotion.src.motion_tracking.smpl_actor_observation import (
    launch_smpl_actor_observation_pack_warp,
    smpl_motion_actor_observation_dim,
)
from holomotion.src.motion_tracking.smpl_reference import SMPL_BODY_COMMAND_DIM
from humanoid_policy.gpu_motion_observation import WarpCudaObservation


class GpuSmplMotionObservationBuilder:
    """Build the Actor input directly from SMPL root/body and robot state."""

    def __init__(
        self,
        *,
        n_future_frames: int,
        default_dof_pos,
        policy_hz: float,
        device: str = "cuda:0",
    ) -> None:
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(f"Warp CUDA device is unavailable: {device}")
        self.n_future_frames = int(n_future_frames)
        self.sequence_frames = self.n_future_frames + 3
        self.policy_hz = float(policy_hz)
        defaults = np.ascontiguousarray(default_dof_pos, dtype=np.float32)
        if defaults.ndim != 1 or defaults.size <= 0:
            raise ValueError("default_dof_pos must be a non-empty vector")
        self.robot_dof_dim = int(defaults.size)

        widths = {
            "robot_quat": 4,
            "robot_angvel": 3,
            "robot_dof_pos": self.robot_dof_dim,
            "robot_dof_vel": self.robot_dof_dim,
            "last_action": self.robot_dof_dim,
            "yaw_alignment": 4,
        }
        self._slices: dict[str, slice] = {}
        offset = 0
        for name, width in widths.items():
            self._slices[name] = slice(offset, offset + width)
            offset += width
        self._host = wp.empty(
            offset, dtype=wp.float32, device="cpu", pinned=True
        )
        host_buffer = (ctypes.c_float * offset).from_address(self._host.ptr)
        self._host_numpy = np.ctypeslib.as_array(host_buffer)
        self._host_numpy.fill(0.0)
        self._device_staging = wp.empty(
            offset, dtype=wp.float32, device=self.device
        )
        self._device_views = {
            name: self._device_staging[field_slice].reshape(shape)
            for (name, shape), field_slice in zip(
                (
                    ("robot_quat", (1, 4)),
                    ("robot_angvel", (1, 3)),
                    ("robot_dof_pos", (1, self.robot_dof_dim)),
                    ("robot_dof_vel", (1, self.robot_dof_dim)),
                    ("last_action", (1, self.robot_dof_dim)),
                    ("yaw_alignment", (1, 4)),
                ),
                self._slices.values(),
                strict=True,
            )
        }
        self._default_dof_pos = wp.from_numpy(
            defaults.reshape(1, self.robot_dof_dim),
            dtype=wp.float32,
            device=self.device,
        )
        self._kinematics = RootReferenceKinematics(
            *[
                wp.empty(
                    (1, self.sequence_frames, 3),
                    dtype=wp.float32,
                    device=self.device,
                )
                for _ in range(5)
            ]
        )
        self._output = wp.empty(
            (
                1,
                smpl_motion_actor_observation_dim(
                    SMPL_BODY_COMMAND_DIM,
                    self.n_future_frames,
                    self.robot_dof_dim,
                ),
            ),
            dtype=wp.float32,
            device=self.device,
        )

    def _host_view(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        return self._host_numpy[self._slices[name]].reshape(shape)

    def build(
        self,
        *,
        smpl_reference,
        robot_root_quat_wxyz,
        robot_root_angvel_local,
        robot_dof_pos,
        robot_dof_vel,
        last_action,
        reference_yaw_alignment_wxyz=None,
        reference_frame_index: int | None = None,
    ):
        root_state, sample_time, body = (
            smpl_reference.device_observation_sequence(
                n_frames=self.n_future_frames,
                fps=self.policy_hz,
                include_derivative_tail=True,
                frame_index=reference_frame_index,
            )
        )
        np.copyto(
            self._host_view("robot_quat", (4,)),
            np.asarray(robot_root_quat_wxyz, dtype=np.float32),
        )
        np.copyto(
            self._host_view("robot_angvel", (3,)),
            np.asarray(robot_root_angvel_local, dtype=np.float32),
        )
        for name, value in (
            ("robot_dof_pos", robot_dof_pos),
            ("robot_dof_vel", robot_dof_vel),
            ("last_action", last_action),
        ):
            np.copyto(
                self._host_view(name, (self.robot_dof_dim,)),
                np.asarray(value, dtype=np.float32),
            )
        alignment = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
            if reference_yaw_alignment_wxyz is None
            else reference_yaw_alignment_wxyz,
            dtype=np.float32,
        )
        np.copyto(self._host_view("yaw_alignment", (4,)), alignment)
        wp.copy(self._device_staging, self._host)
        derive_root_reference_kinematics_warp(
            root_state,
            sample_time,
            outputs=self._kinematics,
            device=self.device,
        )
        launch_smpl_actor_observation_pack_warp(
            root_state_wp=root_state,
            root_kinematics=self._kinematics,
            body_current_future_wp=body,
            robot_root_quat_wp=self._device_views["robot_quat"],
            robot_root_angvel_wp=self._device_views["robot_angvel"],
            robot_dof_pos_wp=self._device_views["robot_dof_pos"],
            robot_dof_vel_wp=self._device_views["robot_dof_vel"],
            last_action_wp=self._device_views["last_action"],
            default_dof_pos_wp=self._default_dof_pos,
            reference_yaw_alignment_wp=self._device_views["yaw_alignment"],
            use_yaw_alignment=True,
            body_dim=SMPL_BODY_COMMAND_DIM,
            robot_dof_dim=self.robot_dof_dim,
            current_index=1,
            num_future_frames=self.n_future_frames,
            output_wp=self._output,
            device=self.device,
        )
        return WarpCudaObservation(self._output)


__all__ = ["GpuSmplMotionObservationBuilder"]
