"""Warp CUDA formal-SMPL reference used by online pure-SMPL deployment."""

from __future__ import annotations

import ctypes
import time

import numpy as np
import warp as wp

from holomotion.src.motion_tracking.smpl_reference import (
    SMPL_TO_COMMAND_FRAME,
    launch_smpl_reference_intermediates_pack_warp,
    load_neutral_smpl_skeleton,
)
from holosmpl.converters.pico_to_smpl_gpu import PicoToSmplGpuConverter
from holosmpl.converters.smpl_to_body_poses import (
    launch_smpl_forward_kinematics_warp,
)


@wp.func
def _quat_mul_wxyz(q0: wp.vec4, q1: wp.vec4) -> wp.vec4:
    w0 = q0[0]
    x0 = q0[1]
    y0 = q0[2]
    z0 = q0[3]
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]
    return wp.vec4(
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    )


@wp.func
def _normalize_quat_wxyz(q: wp.vec4) -> wp.vec4:
    norm = wp.sqrt(wp.dot(q, q))
    q = q / wp.max(norm, 1.0e-9)
    if q[0] < 0.0:
        q = -q
    return q


@wp.func
def _quat_conjugate_wxyz(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def _yaw_from_quat_wxyz(q: wp.vec4) -> float:
    return wp.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )


@wp.kernel
def _store_smpl_frame_kernel(
    body: wp.array2d(dtype=wp.float32),
    pose_aa: wp.array2d(dtype=wp.float32),
    foot_z: wp.array(dtype=wp.float32),
    root_trans: wp.array2d(dtype=wp.float32),
    global_quat_wxyz: wp.array3d(dtype=wp.float32),
    sample_time: float,
    slot: int,
    body_ring: wp.array2d(dtype=wp.float32),
    foot_z_ring: wp.array(dtype=wp.float32),
    trans_ring: wp.array2d(dtype=wp.float32),
    root_quat_ring: wp.array2d(dtype=wp.float32),
    time_ring: wp.array(dtype=wp.float32),
) -> None:
    index = wp.tid()
    if index < 69:
        body_ring[slot, index] = body[0, index]
    elif index < 132:
        body_ring[slot, index] = pose_aa[0, index - 66]
    elif index < 258:
        rotation_index = index - 132
        joint = 1 + rotation_index // 6
        component = rotation_index % 6
        root = wp.vec4(
            global_quat_wxyz[0, 0, 0],
            global_quat_wxyz[0, 0, 1],
            global_quat_wxyz[0, 0, 2],
            global_quat_wxyz[0, 0, 3],
        )
        joint_rotation = wp.vec4(
            global_quat_wxyz[0, joint, 0],
            global_quat_wxyz[0, joint, 1],
            global_quat_wxyz[0, joint, 2],
            global_quat_wxyz[0, joint, 3],
        )
        command = wp.vec4(0.5, -0.5, -0.5, -0.5)
        relative = _quat_mul_wxyz(_quat_conjugate_wxyz(root), joint_rotation)
        semantic = _normalize_quat_wxyz(
            _quat_mul_wxyz(
                _quat_mul_wxyz(_quat_conjugate_wxyz(command), relative),
                command,
            )
        )
        w = semantic[0]
        x = semantic[1]
        y = semantic[2]
        z = semantic[3]
        if component == 0:
            value = 1.0 - 2.0 * (y * y + z * z)
        elif component == 1:
            value = 2.0 * (x * y - z * w)
        elif component == 2:
            value = 2.0 * (x * y + z * w)
        elif component == 3:
            value = 1.0 - 2.0 * (x * x + z * z)
        elif component == 4:
            value = 2.0 * (x * z - y * w)
        else:
            value = 2.0 * (y * z + x * w)
        body_ring[slot, index] = value
    if index < 3:
        trans_ring[slot, index] = root_trans[0, index]
    if index == 0:
        foot_z_ring[slot] = foot_z[0]
        time_ring[slot] = sample_time
        root = wp.vec4(
            global_quat_wxyz[0, 0, 0],
            global_quat_wxyz[0, 0, 1],
            global_quat_wxyz[0, 0, 2],
            global_quat_wxyz[0, 0, 3],
        )
        # Quaternion of SMPL_TO_COMMAND_FRAME.
        command = wp.vec4(0.5, -0.5, -0.5, -0.5)
        packed = _normalize_quat_wxyz(_quat_mul_wxyz(root, command))
        for component in range(4):
            root_quat_ring[slot, component] = packed[component]


@wp.kernel
def _gather_smpl_observation_kernel(
    body_ring: wp.array2d(dtype=wp.float32),
    foot_z_ring: wp.array(dtype=wp.float32),
    trans_ring: wp.array2d(dtype=wp.float32),
    root_quat_ring: wp.array2d(dtype=wp.float32),
    time_ring: wp.array(dtype=wp.float32),
    oldest_slot: int,
    capacity: int,
    ground_history_frames: int,
    sequence_frames: int,
    future_frames: int,
    use_existing_anchor: int,
    anchor: wp.array(dtype=wp.float32),
    qpos_output: wp.array3d(dtype=wp.float32),
    sample_time_output: wp.array2d(dtype=wp.float32),
    body_output: wp.array3d(dtype=wp.float32),
) -> None:
    if wp.tid() != 0:
        return

    output_start = ground_history_frames - 1
    anchor_chronological = output_start + 1
    anchor_slot = (oldest_slot + anchor_chronological) % capacity
    if use_existing_anchor == 0:
        anchor[0] = trans_ring[anchor_slot, 0]
        anchor[1] = trans_ring[anchor_slot, 1]
        anchor_quat = wp.vec4(
            root_quat_ring[anchor_slot, 0],
            root_quat_ring[anchor_slot, 1],
            root_quat_ring[anchor_slot, 2],
            root_quat_ring[anchor_slot, 3],
        )
        anchor[2] = _yaw_from_quat_wxyz(anchor_quat)

    negative_yaw = -anchor[2]
    yaw_quat = wp.vec4(
        wp.cos(0.5 * negative_yaw),
        0.0,
        0.0,
        wp.sin(0.5 * negative_yaw),
    )
    cos_yaw = wp.cos(negative_yaw)
    sin_yaw = wp.sin(negative_yaw)
    first_time_slot = (oldest_slot + output_start) % capacity
    first_time = time_ring[first_time_slot]
    previous_root_quat = wp.vec4(1.0, 0.0, 0.0, 0.0)

    for frame in range(sequence_frames):
        chronological = output_start + frame
        slot = (oldest_slot + chronological) % capacity
        ground = wp.float32(1.0e30)
        for history_offset in range(ground_history_frames):
            ground_chronological = chronological - history_offset
            if ground_chronological >= 0:
                ground_slot = (oldest_slot + ground_chronological) % capacity
                ground = wp.min(ground, foot_z_ring[ground_slot])

        shifted_x = trans_ring[slot, 0] - anchor[0]
        shifted_y = trans_ring[slot, 1] - anchor[1]
        qpos_output[0, frame, 0] = cos_yaw * shifted_x - sin_yaw * shifted_y
        qpos_output[0, frame, 1] = sin_yaw * shifted_x + cos_yaw * shifted_y
        qpos_output[0, frame, 2] = trans_ring[slot, 2] - ground
        root_quat = wp.vec4(
            root_quat_ring[slot, 0],
            root_quat_ring[slot, 1],
            root_quat_ring[slot, 2],
            root_quat_ring[slot, 3],
        )
        normalized = _normalize_quat_wxyz(_quat_mul_wxyz(yaw_quat, root_quat))
        if frame > 0 and wp.dot(previous_root_quat, normalized) < 0.0:
            normalized = -normalized
        for component in range(4):
            qpos_output[0, frame, 3 + component] = normalized[component]
        previous_root_quat = normalized
        sample_time_output[0, frame] = time_ring[slot] - first_time

    for body_frame in range(future_frames + 1):
        chronological = output_start + 1 + body_frame
        slot = (oldest_slot + chronological) % capacity
        for component in range(258):
            body_output[0, body_frame, component] = body_ring[slot, component]


@wp.kernel
def _gather_offline_smpl_observation_kernel(
    all_qpos: wp.array2d(dtype=wp.float32),
    all_body: wp.array2d(dtype=wp.float32),
    frame_index: int,
    frame_count: int,
    policy_hz: float,
    sequence_frames: int,
    future_frames: int,
    qpos_output: wp.array3d(dtype=wp.float32),
    sample_time_output: wp.array2d(dtype=wp.float32),
    body_output: wp.array3d(dtype=wp.float32),
) -> None:
    index = wp.tid()
    if index < sequence_frames * 7:
        output_frame = index // 7
        component = index - output_frame * 7
        source_frame = wp.clamp(
            frame_index - 1 + output_frame, 0, frame_count - 1
        )
        qpos_output[0, output_frame, component] = all_qpos[
            source_frame, component
        ]
        if component == 0:
            sample_time_output[0, output_frame] = (
                wp.float32(output_frame) / policy_hz
            )
    if index < (future_frames + 1) * 258:
        output_frame = index // 258
        component = index - output_frame * 258
        source_frame = wp.clamp(frame_index + output_frame, 0, frame_count - 1)
        body_output[0, output_frame, component] = all_body[
            source_frame, component
        ]


class GpuSmplReference:
    """Common online reference interface consumed by policy runtime."""

    def __init__(
        self,
        *,
        n_future_frames: int,
        device: str,
        policy_hz: float,
    ) -> None:
        wp.init()
        self.n_fut_frames = max(int(n_future_frames), 0)
        self.sequence_frames = self.n_fut_frames + 3
        self.policy_hz = float(policy_hz)
        if self.policy_hz <= 0.0:
            raise ValueError("policy_hz must be positive")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(f"Warp CUDA device is unavailable: {device}")
        self.seen_frames = 0
        self.received = False
        self.last_receive_time: float | None = None
        self.latest_frame_index: int | None = None
        self.kinematics = None

    @property
    def has_reference(self) -> bool:
        return self.received

    @property
    def latest_qpos(self):
        return None

    @property
    def root_rot_queue(self):
        return None

    def data_age(self, current_time: float) -> float:
        if self.last_receive_time is None:
            return float("inf")
        return float(current_time) - self.last_receive_time

    def is_ready_for_motion(
        self, enable_teleop_reference: bool, delay_frames: int
    ) -> bool:
        del delay_frames
        return bool(enable_teleop_reference and self.has_future_sequence())

    def current_root_rot(self) -> np.ndarray | None:
        raise NotImplementedError


class GpuOnlineSmplReference(GpuSmplReference):
    """Formal SMPL input, shared SMPL kernels, and Warp CUDA ring buffers."""

    def __init__(
        self,
        *,
        n_future_frames: int,
        policy_hz: float,
        device: str = "cuda:0",
        ground_history_frames: int = 50,
    ) -> None:
        super().__init__(
            n_future_frames=n_future_frames,
            device=device,
            policy_hz=policy_hz,
        )
        self.ground_history_frames = int(ground_history_frames)
        if self.ground_history_frames <= 0:
            raise ValueError("ground_history_frames must be positive")
        self.capacity = self.ground_history_frames - 1 + self.sequence_frames
        skeleton = load_neutral_smpl_skeleton()
        self.body_dim = 258
        self._parents = wp.from_numpy(
            np.asarray(skeleton.parents, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._offsets = wp.from_numpy(
            np.asarray(
                skeleton.rest_parent_offsets,
                dtype=np.float32,
            ).reshape(-1),
            dtype=wp.float32,
            device=self.device,
        )
        self._command_frame = wp.from_numpy(
            np.asarray(SMPL_TO_COMMAND_FRAME, dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self._pose_host = wp.empty(
            (1, 72), dtype=wp.float32, device="cpu", pinned=True
        )
        self._trans_host = wp.empty(
            (1, 3), dtype=wp.float32, device="cpu", pinned=True
        )
        pose_buffer = (ctypes.c_float * 72).from_address(self._pose_host.ptr)
        trans_buffer = (ctypes.c_float * 3).from_address(self._trans_host.ptr)
        self._pose_host_numpy = np.ctypeslib.as_array(pose_buffer)
        self._trans_host_numpy = np.ctypeslib.as_array(trans_buffer)
        self._pose = wp.empty((1, 72), dtype=wp.float32, device=self.device)
        self._trans = wp.empty((1, 3), dtype=wp.float32, device=self.device)
        self._zero_trans = wp.zeros(
            (1, 3), dtype=wp.float32, device=self.device
        )
        self._global_pos = wp.empty(
            (1, 24, 3), dtype=wp.float32, device=self.device
        )
        self._global_quat = wp.empty(
            (1, 24, 4), dtype=wp.float32, device=self.device
        )
        self._body = wp.empty((1, 69), dtype=wp.float32, device=self.device)
        self._foot_z = wp.empty(1, dtype=wp.float32, device=self.device)
        self._unused_root_rot = wp.empty(
            (1, 3, 3), dtype=wp.float32, device=self.device
        )
        self._body_ring = wp.empty(
            (self.capacity, self.body_dim),
            dtype=wp.float32,
            device=self.device,
        )
        self._foot_z_ring = wp.empty(
            self.capacity, dtype=wp.float32, device=self.device
        )
        self._trans_ring = wp.empty(
            (self.capacity, 3), dtype=wp.float32, device=self.device
        )
        self._root_quat_ring = wp.empty(
            (self.capacity, 4), dtype=wp.float32, device=self.device
        )
        self._time_ring = wp.empty(
            self.capacity, dtype=wp.float32, device=self.device
        )
        self._anchor = wp.empty(3, dtype=wp.float32, device=self.device)
        self._qpos_sequence = wp.empty(
            (1, self.sequence_frames, 7),
            dtype=wp.float32,
            device=self.device,
        )
        self._sample_time_sequence = wp.empty(
            (1, self.sequence_frames),
            dtype=wp.float32,
            device=self.device,
        )
        self._body_sequence = wp.empty(
            (1, self.n_fut_frames + 1, self.body_dim),
            dtype=wp.float32,
            device=self.device,
        )
        self._current_quat_host = wp.empty(
            4, dtype=wp.float32, device="cpu", pinned=True
        )
        quat_buffer = (ctypes.c_float * 4).from_address(
            self._current_quat_host.ptr
        )
        self._current_quat_numpy = np.ctypeslib.as_array(quat_buffer)
        self._pico_converter = PicoToSmplGpuConverter(str(self.device))
        self._write_slot = 0
        self._last_sample_time: float | None = None
        self._sample_time_origin: float | None = None
        self._version = 0
        self._gathered_version = -1
        self._anchor_ready = False

    def store_pico(
        self,
        body_poses: np.ndarray,
        *,
        sample_time: float | None = None,
        current_time: float | None = None,
        frame_index: int | None = None,
    ) -> bool:
        pose, trans = self._pico_converter.to_device(body_poses)
        receive_time = float(
            time.time() if current_time is None else current_time
        )
        return self._store_device(
            pose,
            trans,
            sample_time=(
                receive_time if sample_time is None else float(sample_time)
            ),
            current_time=receive_time,
            frame_index=frame_index,
        )

    def store_smpl(
        self,
        human_pose_aa: np.ndarray,
        human_root_trans: np.ndarray,
        *,
        sample_time: float,
        current_time: float | None = None,
        frame_index: int | None = None,
    ) -> bool:
        pose = np.asarray(human_pose_aa, dtype=np.float32)
        trans = np.asarray(human_root_trans, dtype=np.float32)
        if pose.shape not in {(72,), (1, 72)}:
            raise ValueError(f"human_pose_aa must be [72], got {pose.shape}")
        if trans.shape not in {(3,), (1, 3)}:
            raise ValueError(
                f"human_root_trans must be [3], got {trans.shape}"
            )
        if not np.isfinite(pose).all() or not np.isfinite(trans).all():
            raise ValueError("received SMPL frame contains NaN or Inf")
        np.copyto(self._pose_host_numpy, pose.reshape(72))
        np.copyto(self._trans_host_numpy, trans.reshape(3))
        wp.copy(self._pose, self._pose_host)
        wp.copy(self._trans, self._trans_host)
        return self._store_device(
            self._pose,
            self._trans,
            sample_time=sample_time,
            current_time=current_time,
            frame_index=frame_index,
        )

    def _store_device(
        self,
        pose,
        trans,
        *,
        sample_time: float,
        current_time: float | None,
        frame_index: int | None,
    ) -> bool:
        sample_time = float(sample_time)
        if not np.isfinite(sample_time):
            raise ValueError("sample_time must be finite")
        if (
            self._last_sample_time is not None
            and sample_time <= self._last_sample_time
        ):
            return False
        if self._sample_time_origin is None:
            self._sample_time_origin = sample_time

        launch_smpl_forward_kinematics_warp(
            pose_aa=pose,
            transl=self._zero_trans,
            smpl_parents=self._parents,
            smpl_rest_parent_offsets=self._offsets,
            output_positions=self._global_pos,
            output_quats_wxyz=self._global_quat,
            device=self.device,
        )
        launch_smpl_reference_intermediates_pack_warp(
            relative_world_pos=self._global_pos,
            global_quat_wxyz=self._global_quat,
            root_trans=trans,
            command_frame=self._command_frame,
            body_command=self._body,
            foot_z=self._foot_z,
            root_command_rot=self._unused_root_rot,
            device=self.device,
        )
        wp.launch(
            _store_smpl_frame_kernel,
            dim=self.body_dim,
            inputs=[
                self._body,
                pose,
                self._foot_z,
                trans,
                self._global_quat,
                sample_time - self._sample_time_origin,
                self._write_slot,
                self._body_ring,
                self._foot_z_ring,
                self._trans_ring,
                self._root_quat_ring,
                self._time_ring,
            ],
            device=self.device,
        )
        self._write_slot = (self._write_slot + 1) % self.capacity
        self._last_sample_time = sample_time
        self.seen_frames += 1
        self.received = True
        self.last_receive_time = float(
            time.time() if current_time is None else current_time
        )
        self.latest_frame_index = (
            None if frame_index is None else int(frame_index)
        )
        self._version += 1
        return True

    def _sequence_ready(self) -> bool:
        return self.seen_frames >= self.capacity

    def has_future_sequence(
        self,
        n_frames: int | None = None,
        *,
        include_derivative_tail: bool = False,
    ) -> bool:
        del include_derivative_tail
        count = (
            self.n_fut_frames if n_frames is None else max(int(n_frames), 0)
        )
        return count == self.n_fut_frames and self._sequence_ready()

    def reset_root_anchor(self) -> None:
        self._anchor_ready = False
        self._gathered_version = -1

    def device_observation_sequence(
        self,
        *,
        n_frames: int | None = None,
        include_derivative_tail: bool = True,
        frame_index: int | None = None,
        fps: float | None = None,
    ):
        del frame_index
        count = (
            self.n_fut_frames if n_frames is None else max(int(n_frames), 0)
        )
        if count != self.n_fut_frames:
            raise ValueError(
                f"reference allocated for {self.n_fut_frames} futures, got "
                f"{count}"
            )
        if fps is not None and abs(float(fps) - self.policy_hz) > 1.0e-5:
            raise ValueError(
                f"reference fps={self.policy_hz}, requested fps={fps}"
            )
        if not include_derivative_tail:
            raise ValueError("SMPL observation requires derivative tail")
        if not self._sequence_ready():
            raise RuntimeError("SMPL reference sequence is unavailable")
        if self._gathered_version != self._version:
            wp.launch(
                _gather_smpl_observation_kernel,
                dim=1,
                inputs=[
                    self._body_ring,
                    self._foot_z_ring,
                    self._trans_ring,
                    self._root_quat_ring,
                    self._time_ring,
                    self._write_slot,
                    self.capacity,
                    self.ground_history_frames,
                    self.sequence_frames,
                    self.n_fut_frames,
                    int(self._anchor_ready),
                    self._anchor,
                    self._qpos_sequence,
                    self._sample_time_sequence,
                    self._body_sequence,
                ],
                device=self.device,
            )
            self._anchor_ready = True
            self._gathered_version = self._version
        return (
            self._qpos_sequence,
            self._sample_time_sequence,
            self._body_sequence,
        )

    def current_root_rot(self) -> np.ndarray | None:
        if not self._sequence_ready():
            return None
        self.device_observation_sequence()
        wp.copy(
            self._current_quat_host,
            self._qpos_sequence,
            src_offset=7 + 3,
            count=4,
        )
        wp.synchronize_device(self.device)
        return self._current_quat_numpy.copy()


class GpuOfflineSmplReference:
    """Precomputed formal-SMPL playback using the training feature builder."""

    def __init__(
        self,
        human_pose_aa: np.ndarray,
        human_root_trans: np.ndarray,
        *,
        n_future_frames: int,
        policy_hz: float,
        device: str = "cuda:0",
    ) -> None:
        import torch

        from holomotion.src.motion_tracking.smpl_reference import (
            SMPL_BODY_COMMAND_MODE,
            SmplReferenceBuilder,
        )

        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(f"Warp CUDA device is unavailable: {device}")
        self.n_fut_frames = max(int(n_future_frames), 0)
        self.sequence_frames = self.n_fut_frames + 3
        self.policy_hz = float(policy_hz)
        builder = SmplReferenceBuilder(
            SMPL_BODY_COMMAND_MODE, fps=self.policy_hz
        )
        reference = builder.build(human_pose_aa, human_root_trans)
        frame_count = int(reference.body_command.shape[0])
        if frame_count <= 0:
            raise ValueError("offline SMPL clip is empty")
        qpos = torch.empty((frame_count, 7), dtype=torch.float32)
        qpos[:, :3] = reference.root_pos
        qpos[:, 3:7] = reference.root_rot_xyzw[:, [3, 0, 1, 2]]
        self._qpos_torch = qpos.to(str(self.device)).contiguous()
        self._body_torch = reference.body_command.to(
            str(self.device)
        ).contiguous()
        self._all_qpos = wp.from_torch(self._qpos_torch, dtype=wp.float32)
        self._all_body = wp.from_torch(self._body_torch, dtype=wp.float32)
        self._initial_root_rot = qpos[0, 3:7].numpy().copy()
        self.frame_count = frame_count
        self._qpos_sequence = wp.empty(
            (1, self.sequence_frames, 7),
            dtype=wp.float32,
            device=self.device,
        )
        self._sample_time_sequence = wp.empty(
            (1, self.sequence_frames),
            dtype=wp.float32,
            device=self.device,
        )
        self._body_sequence = wp.empty(
            (1, self.n_fut_frames + 1, 258),
            dtype=wp.float32,
            device=self.device,
        )
        self._last_frame_index: int | None = None

    @property
    def has_reference(self) -> bool:
        return True

    def has_future_sequence(self, *args, **kwargs) -> bool:
        del args, kwargs
        return True

    def device_observation_sequence(
        self,
        *,
        n_frames: int | None = None,
        include_derivative_tail: bool = True,
        frame_index: int | None = None,
        fps: float | None = None,
    ):
        count = self.n_fut_frames if n_frames is None else int(n_frames)
        if count != self.n_fut_frames:
            raise ValueError(
                f"reference allocated for {self.n_fut_frames} futures, got {count}"
            )
        if not include_derivative_tail:
            raise ValueError("SMPL observation requires derivative tail")
        if fps is not None and abs(float(fps) - self.policy_hz) > 1.0e-5:
            raise ValueError(
                f"reference fps={self.policy_hz}, requested fps={fps}"
            )
        current = 0 if frame_index is None else int(frame_index)
        current = max(0, min(current, self.frame_count - 1))
        if current != self._last_frame_index:
            size = max(
                self.sequence_frames * 7,
                (self.n_fut_frames + 1) * 258,
            )
            wp.launch(
                _gather_offline_smpl_observation_kernel,
                dim=size,
                inputs=[
                    self._all_qpos,
                    self._all_body,
                    current,
                    self.frame_count,
                    self.policy_hz,
                    self.sequence_frames,
                    self.n_fut_frames,
                    self._qpos_sequence,
                    self._sample_time_sequence,
                    self._body_sequence,
                ],
                device=self.device,
            )
            self._last_frame_index = current
        return (
            self._qpos_sequence,
            self._sample_time_sequence,
            self._body_sequence,
        )

    def current_root_rot(self) -> np.ndarray | None:
        return self._initial_root_rot.copy()


__all__ = ["GpuOfflineSmplReference", "GpuOnlineSmplReference"]
