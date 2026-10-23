"""Actor-only observation terms derived from a pure-SMPL reference."""

from __future__ import annotations
from dataclasses import dataclass

import torch
import warp as wp

from holomotion.src.motion_tracking.reference_observation import (
    RootReferenceKinematics,
    derive_root_reference_kinematics_torch,
)

SMPL_ACTOR_CURRENT_TERMS = (
    "actor_smpl_gravity_projection_cur",
    "actor_smpl_base_linvel_cur",
    "actor_smpl_base_angvel_cur",
    "actor_smpl_body_command_cur",
    "actor_smpl_root_height_cur",
    "actor_smpl_robot_yaw_error_sin_cos",
)

SMPL_ACTOR_FUTURE_TERMS = (
    "actor_smpl_body_command_fut",
    "actor_smpl_root_height_fut",
    "actor_smpl_gravity_projection_fut",
    "actor_smpl_base_linvel_fut",
    "actor_smpl_base_angvel_fut",
    "actor_smpl_future_yaw_delta_sin_cos",
    "actor_smpl_future_root_ori_robot_frame_6d",
)


@dataclass(frozen=True)
class SmplActorObservationInput:
    """Inputs after the paired cache has gathered one environment batch."""

    root_state_with_context: torch.Tensor
    body_current_future: torch.Tensor
    robot_root_quat_wxyz: torch.Tensor


@dataclass(frozen=True)
class SmplMotionActorObservationInput:
    """Complete deployed Actor input with SMPL reference and robot state."""

    root_state_with_context: torch.Tensor
    body_current_future: torch.Tensor
    robot_root_quat_wxyz: torch.Tensor
    robot_root_angvel_local: torch.Tensor
    robot_dof_pos: torch.Tensor
    robot_dof_vel: torch.Tensor
    last_action: torch.Tensor
    default_dof_pos: torch.Tensor
    reference_sample_time: torch.Tensor | None = None
    reference_yaw_alignment_wxyz: torch.Tensor | None = None


@wp.func
def _wp_normalize_quat_wxyz(q: wp.vec4) -> wp.vec4:
    norm = wp.sqrt(wp.dot(q, q))
    return q / wp.max(norm, 1.0e-9)


@wp.func
def _wp_quat_mul_wxyz(q0: wp.vec4, q1: wp.vec4) -> wp.vec4:
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
def _wp_quat_inverse_wxyz(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def _wp_yaw_from_quat_wxyz(q: wp.vec4) -> float:
    return wp.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )


@wp.func
def _wp_projected_gravity_component(q: wp.vec4, component: int) -> float:
    value = 2.0 * (-q[3] * q[1] + q[0] * q[2])
    if component == 1:
        value = -2.0 * (q[3] * q[2] + q[0] * q[1])
    elif component == 2:
        value = 1.0 - 2.0 * (q[0] * q[0] + q[3] * q[3])
    return value


@wp.func
def _wp_rot6d_component_wxyz(q: wp.vec4, component: int) -> float:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    value = 1.0 - 2.0 * (y * y + z * z)
    if component == 1:
        value = 2.0 * (x * y - z * w)
    elif component == 2:
        value = 2.0 * (x * y + z * w)
    elif component == 3:
        value = 1.0 - 2.0 * (x * x + z * z)
    elif component == 4:
        value = 2.0 * (x * z - y * w)
    elif component == 5:
        value = 2.0 * (y * z + x * w)
    return value


@wp.kernel
def _pack_smpl_actor_observation_kernel(
    root_state: wp.array3d(dtype=wp.float32),
    root_linvel_local: wp.array3d(dtype=wp.float32),
    root_angvel_local: wp.array3d(dtype=wp.float32),
    projected_gravity: wp.array3d(dtype=wp.float32),
    body_current_future: wp.array3d(dtype=wp.float32),
    robot_root_quat_wxyz: wp.array2d(dtype=wp.float32),
    robot_root_angvel_local: wp.array2d(dtype=wp.float32),
    robot_dof_pos: wp.array2d(dtype=wp.float32),
    robot_dof_vel: wp.array2d(dtype=wp.float32),
    last_action: wp.array2d(dtype=wp.float32),
    default_dof_pos: wp.array2d(dtype=wp.float32),
    reference_yaw_alignment_wxyz: wp.array2d(dtype=wp.float32),
    use_yaw_alignment: int,
    body_dim: int,
    robot_dof_dim: int,
    current_index: int,
    num_future_frames: int,
    output: wp.array2d(dtype=wp.float32),
) -> None:
    index = wp.tid()
    current_dim = body_dim + 18 + 3 * robot_dof_dim
    future_dim = body_dim + 18
    output_dim = current_dim + num_future_frames * future_dim
    batch = index // output_dim
    element = index - batch * output_dim
    current = current_index

    if element < 3:
        output[batch, element] = projected_gravity[batch, current, element]
    elif element < 6:
        output[batch, element] = root_linvel_local[batch, current, element - 3]
    elif element < 9:
        output[batch, element] = root_angvel_local[batch, current, element - 6]
    elif element < 9 + body_dim:
        output[batch, element] = body_current_future[batch, 0, element - 9]
    elif element == 9 + body_dim:
        output[batch, element] = root_state[batch, current, 2]
    elif element < 12 + body_dim:
        robot_quat = _wp_normalize_quat_wxyz(
            wp.vec4(
                robot_root_quat_wxyz[batch, 0],
                robot_root_quat_wxyz[batch, 1],
                robot_root_quat_wxyz[batch, 2],
                robot_root_quat_wxyz[batch, 3],
            )
        )
        ref_quat = _wp_normalize_quat_wxyz(
            wp.vec4(
                root_state[batch, current, 3],
                root_state[batch, current, 4],
                root_state[batch, current, 5],
                root_state[batch, current, 6],
            )
        )
        if use_yaw_alignment != 0:
            alignment = _wp_normalize_quat_wxyz(
                wp.vec4(
                    reference_yaw_alignment_wxyz[batch, 0],
                    reference_yaw_alignment_wxyz[batch, 1],
                    reference_yaw_alignment_wxyz[batch, 2],
                    reference_yaw_alignment_wxyz[batch, 3],
                )
            )
            ref_quat = _wp_normalize_quat_wxyz(
                _wp_quat_mul_wxyz(alignment, ref_quat)
            )
        yaw_error = _wp_yaw_from_quat_wxyz(ref_quat) - _wp_yaw_from_quat_wxyz(
            robot_quat
        )
        if element == 10 + body_dim:
            output[batch, element] = wp.sin(yaw_error)
        else:
            output[batch, element] = wp.cos(yaw_error)
    elif element < 15 + body_dim:
        robot_quat = _wp_normalize_quat_wxyz(
            wp.vec4(
                robot_root_quat_wxyz[batch, 0],
                robot_root_quat_wxyz[batch, 1],
                robot_root_quat_wxyz[batch, 2],
                robot_root_quat_wxyz[batch, 3],
            )
        )
        output[batch, element] = _wp_projected_gravity_component(
            robot_quat, element - 12 - body_dim
        )
    elif element < 18 + body_dim:
        output[batch, element] = robot_root_angvel_local[
            batch, element - 15 - body_dim
        ]
    elif element < 18 + body_dim + robot_dof_dim:
        dof = element - 18 - body_dim
        output[batch, element] = (
            robot_dof_pos[batch, dof] - default_dof_pos[batch, dof]
        )
    elif element < 18 + body_dim + 2 * robot_dof_dim:
        dof = element - 18 - body_dim - robot_dof_dim
        output[batch, element] = robot_dof_vel[batch, dof]
    elif element < current_dim:
        dof = element - 18 - body_dim - 2 * robot_dof_dim
        output[batch, element] = last_action[batch, dof]
    else:
        future_element = element - current_dim
        body_count = num_future_frames * body_dim
        height_count = num_future_frames
        vec3_count = num_future_frames * 3
        yaw_count = num_future_frames * 2
        if future_element < body_count:
            future = future_element // body_dim
            component = future_element - future * body_dim
            output[batch, element] = body_current_future[
                batch, future + 1, component
            ]
        elif future_element < body_count + height_count:
            future = future_element - body_count
            output[batch, element] = root_state[batch, current + 1 + future, 2]
        elif future_element < body_count + height_count + vec3_count:
            local = future_element - body_count - height_count
            future = local // 3
            component = local - 3 * future
            output[batch, element] = projected_gravity[
                batch, current + 1 + future, component
            ]
        elif future_element < body_count + height_count + 2 * vec3_count:
            local = future_element - body_count - height_count - vec3_count
            future = local // 3
            component = local - 3 * future
            output[batch, element] = root_linvel_local[
                batch, current + 1 + future, component
            ]
        elif future_element < body_count + height_count + 3 * vec3_count:
            local = future_element - body_count - height_count - 2 * vec3_count
            future = local // 3
            component = local - 3 * future
            output[batch, element] = root_angvel_local[
                batch, current + 1 + future, component
            ]
        elif (
            future_element
            < body_count + height_count + 3 * vec3_count + yaw_count
        ):
            local = future_element - body_count - height_count - 3 * vec3_count
            future = local // 2
            component = local - 2 * future
            current_quat = _wp_normalize_quat_wxyz(
                wp.vec4(
                    root_state[batch, current, 3],
                    root_state[batch, current, 4],
                    root_state[batch, current, 5],
                    root_state[batch, current, 6],
                )
            )
            frame = current + 1 + future
            future_quat = _wp_normalize_quat_wxyz(
                wp.vec4(
                    root_state[batch, frame, 3],
                    root_state[batch, frame, 4],
                    root_state[batch, frame, 5],
                    root_state[batch, frame, 6],
                )
            )
            yaw_delta = _wp_yaw_from_quat_wxyz(
                future_quat
            ) - _wp_yaw_from_quat_wxyz(current_quat)
            if component == 0:
                output[batch, element] = wp.sin(yaw_delta)
            else:
                output[batch, element] = wp.cos(yaw_delta)
        else:
            local = (
                future_element
                - body_count
                - height_count
                - 3 * vec3_count
                - yaw_count
            )
            future = local // 6
            component = local - 6 * future
            frame = current + 1 + future
            robot_quat = _wp_normalize_quat_wxyz(
                wp.vec4(
                    robot_root_quat_wxyz[batch, 0],
                    robot_root_quat_wxyz[batch, 1],
                    robot_root_quat_wxyz[batch, 2],
                    robot_root_quat_wxyz[batch, 3],
                )
            )
            ref_quat = _wp_normalize_quat_wxyz(
                wp.vec4(
                    root_state[batch, frame, 3],
                    root_state[batch, frame, 4],
                    root_state[batch, frame, 5],
                    root_state[batch, frame, 6],
                )
            )
            if use_yaw_alignment != 0:
                alignment = _wp_normalize_quat_wxyz(
                    wp.vec4(
                        reference_yaw_alignment_wxyz[batch, 0],
                        reference_yaw_alignment_wxyz[batch, 1],
                        reference_yaw_alignment_wxyz[batch, 2],
                        reference_yaw_alignment_wxyz[batch, 3],
                    )
                )
                ref_quat = _wp_normalize_quat_wxyz(
                    _wp_quat_mul_wxyz(alignment, ref_quat)
                )
            relative = _wp_normalize_quat_wxyz(
                _wp_quat_mul_wxyz(_wp_quat_inverse_wxyz(robot_quat), ref_quat)
            )
            output[batch, element] = _wp_rot6d_component_wxyz(
                relative, component
            )


def smpl_motion_actor_observation_dim(
    body_dim: int,
    num_future_frames: int,
    robot_dof_dim: int = 29,
) -> int:
    """Return the flattened P+A+R Actor observation dimension."""

    current = int(body_dim) + 18 + 3 * int(robot_dof_dim)
    future = int(body_dim) + 18
    return current + int(num_future_frames) * future


def launch_smpl_actor_observation_pack_warp(
    *,
    root_state_wp,
    root_kinematics: RootReferenceKinematics,
    body_current_future_wp,
    robot_root_quat_wp,
    robot_root_angvel_wp,
    robot_dof_pos_wp,
    robot_dof_vel_wp,
    last_action_wp,
    default_dof_pos_wp,
    reference_yaw_alignment_wp,
    use_yaw_alignment: bool,
    body_dim: int,
    robot_dof_dim: int,
    current_index: int,
    num_future_frames: int,
    output_wp,
    device,
) -> None:
    """Pack pure-SMPL root/body targets and actual robot state directly."""

    wp.launch(
        _pack_smpl_actor_observation_kernel,
        dim=(
            root_state_wp.shape[0]
            * smpl_motion_actor_observation_dim(
                body_dim, num_future_frames, robot_dof_dim
            )
        ),
        inputs=[
            root_state_wp,
            root_kinematics.root_linvel_local,
            root_kinematics.root_angvel_local,
            root_kinematics.projected_gravity,
            body_current_future_wp,
            robot_root_quat_wp,
            robot_root_angvel_wp,
            robot_dof_pos_wp,
            robot_dof_vel_wp,
            last_action_wp,
            default_dof_pos_wp,
            reference_yaw_alignment_wp,
            int(bool(use_yaw_alignment)),
            int(body_dim),
            int(robot_dof_dim),
            int(current_index),
            int(num_future_frames),
            output_wp,
        ],
        device=device,
    )


def build_smpl_motion_actor_observation_torch(
    inputs: SmplMotionActorObservationInput,
    *,
    fps: float = 50.0,
    current_index: int = 1,
    num_future_frames: int = 10,
) -> torch.Tensor:
    """Build the complete Actor input without a dummy robot-reference pose."""

    root_state = inputs.root_state_with_context
    if root_state.ndim == 2:
        root_state = root_state.unsqueeze(0)
    root_state = root_state.to(dtype=torch.float32).contiguous()
    if root_state.ndim != 3 or root_state.shape[-1] != 7:
        raise ValueError(
            "root_state_with_context must be [B,T,7], got "
            f"{tuple(root_state.shape)}"
        )
    body = inputs.body_current_future.to(
        device=root_state.device, dtype=torch.float32
    ).contiguous()
    if body.ndim == 2:
        body = body.unsqueeze(0)
    expected_frames = 1 + int(num_future_frames)
    if (
        body.ndim != 3
        or body.shape[0] != root_state.shape[0]
        or body.shape[1] != expected_frames
    ):
        raise ValueError(
            "body_current_future must be [B,1+n_future,body_dim], got "
            f"{tuple(body.shape)}"
        )
    body_dim = int(body.shape[2])
    robot_tensors = (
        inputs.robot_dof_pos,
        inputs.robot_dof_vel,
        inputs.last_action,
        inputs.default_dof_pos,
    )
    robot_dof_dim = int(robot_tensors[0].shape[-1])
    if robot_dof_dim <= 0 or any(
        tensor.shape != (root_state.shape[0], robot_dof_dim)
        for tensor in robot_tensors
    ):
        raise ValueError(
            "robot_dof_pos, robot_dof_vel, last_action and default_dof_pos "
            "must all be [B,robot_dof_dim]"
        )
    robot_quat = inputs.robot_root_quat_wxyz.to(
        device=root_state.device, dtype=torch.float32
    ).contiguous()
    robot_angvel = inputs.robot_root_angvel_local.to(
        device=root_state.device, dtype=torch.float32
    ).contiguous()
    if robot_quat.shape != (root_state.shape[0], 4):
        raise ValueError("robot_root_quat_wxyz must be [B,4]")
    if robot_angvel.shape != (root_state.shape[0], 3):
        raise ValueError("robot_root_angvel_local must be [B,3]")
    sample_time = inputs.reference_sample_time
    kinematics = derive_root_reference_kinematics_torch(
        root_state,
        sample_time=sample_time,
        fps=fps,
    )
    if inputs.reference_yaw_alignment_wxyz is None:
        alignment = torch.zeros(
            (root_state.shape[0], 4),
            dtype=torch.float32,
            device=root_state.device,
        )
        alignment[:, 0] = 1.0
        use_alignment = False
    else:
        alignment = inputs.reference_yaw_alignment_wxyz.to(
            device=root_state.device, dtype=torch.float32
        ).contiguous()
        if alignment.shape != (root_state.shape[0], 4):
            raise ValueError("reference_yaw_alignment_wxyz must be [B,4]")
        use_alignment = True
    output = torch.empty(
        (
            root_state.shape[0],
            smpl_motion_actor_observation_dim(
                body_dim, num_future_frames, robot_dof_dim
            ),
        ),
        dtype=torch.float32,
        device=root_state.device,
    )
    device = wp.device_from_torch(root_state.device)
    launch_smpl_actor_observation_pack_warp(
        root_state_wp=wp.from_torch(root_state, dtype=wp.float32),
        root_kinematics=RootReferenceKinematics(
            *[
                wp.from_torch(value, dtype=wp.float32)
                for value in (
                    kinematics.root_linvel_world,
                    kinematics.root_angvel_world,
                    kinematics.root_linvel_local,
                    kinematics.root_angvel_local,
                    kinematics.projected_gravity,
                )
            ]
        ),
        body_current_future_wp=wp.from_torch(body, dtype=wp.float32),
        robot_root_quat_wp=wp.from_torch(robot_quat, dtype=wp.float32),
        robot_root_angvel_wp=wp.from_torch(robot_angvel, dtype=wp.float32),
        robot_dof_pos_wp=wp.from_torch(
            robot_tensors[0]
            .to(device=root_state.device, dtype=torch.float32)
            .contiguous(),
            dtype=wp.float32,
        ),
        robot_dof_vel_wp=wp.from_torch(
            robot_tensors[1]
            .to(device=root_state.device, dtype=torch.float32)
            .contiguous(),
            dtype=wp.float32,
        ),
        last_action_wp=wp.from_torch(
            robot_tensors[2]
            .to(device=root_state.device, dtype=torch.float32)
            .contiguous(),
            dtype=wp.float32,
        ),
        default_dof_pos_wp=wp.from_torch(
            robot_tensors[3]
            .to(device=root_state.device, dtype=torch.float32)
            .contiguous(),
            dtype=wp.float32,
        ),
        reference_yaw_alignment_wp=wp.from_torch(alignment, dtype=wp.float32),
        use_yaw_alignment=use_alignment,
        body_dim=body_dim,
        robot_dof_dim=robot_dof_dim,
        current_index=current_index,
        num_future_frames=num_future_frames,
        output_wp=wp.from_torch(output, dtype=wp.float32),
        device=device,
    )
    return output


def _normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(
        quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_wxyz(quat)
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _quat_mul_wxyz(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = torch.unbind(q0, dim=-1)
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def _rot6d_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.unbind(_normalize_quat_wxyz(quat), dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
        ),
        dim=-1,
    )


def derive_smpl_actor_terms_torch(
    inputs: SmplActorObservationInput,
    *,
    fps: float = 50.0,
    current_index: int = 1,
    num_future_frames: int = 10,
) -> dict[str, torch.Tensor]:
    """Build named Actor command terms while preserving legacy timing rules."""

    root_state = inputs.root_state_with_context
    body = inputs.body_current_future
    robot_quat = inputs.robot_root_quat_wxyz
    if root_state.ndim != 3 or root_state.shape[-1] != 7:
        raise ValueError(
            "root_state_with_context must be [B,T,7], got "
            f"{tuple(root_state.shape)}"
        )
    batch = root_state.shape[0]
    expected_body_frames = 1 + int(num_future_frames)
    if body.ndim != 3 or body.shape[:2] != (batch, expected_body_frames):
        raise ValueError(
            "body_current_future must be [B,1+n_future,Bdim], got "
            f"{tuple(body.shape)}"
        )
    if robot_quat.shape != (batch, 4):
        raise ValueError(
            "robot_root_quat_wxyz must be "
            f"{(batch, 4)}, got {tuple(robot_quat.shape)}"
        )
    future_end = int(current_index) + 1 + int(num_future_frames)
    if current_index < 0 or future_end > root_state.shape[1]:
        raise ValueError(
            f"root_state has {root_state.shape[1]} frames but requires "
            f"{future_end}"
        )

    root_state = root_state.to(dtype=torch.float32).contiguous()
    body = body.to(device=root_state.device, dtype=torch.float32).contiguous()
    robot_quat = _normalize_quat_wxyz(
        robot_quat.to(device=root_state.device, dtype=torch.float32)
    )
    kinematics = derive_root_reference_kinematics_torch(
        root_state, fps=float(fps)
    )
    cur = int(current_index)
    fut = slice(cur + 1, future_end)
    ref_quat_cur = _normalize_quat_wxyz(root_state[:, cur, 3:7])
    ref_quat_fut = _normalize_quat_wxyz(root_state[:, fut, 3:7])

    yaw_error = _yaw_from_quat_wxyz(ref_quat_cur) - _yaw_from_quat_wxyz(
        robot_quat
    )
    future_yaw_delta = _yaw_from_quat_wxyz(ref_quat_fut) - _yaw_from_quat_wxyz(
        ref_quat_cur
    ).unsqueeze(1)
    robot_inv = robot_quat.clone()
    robot_inv[:, 1:] *= -1.0
    relative_future = _quat_mul_wxyz(
        robot_inv[:, None].expand_as(ref_quat_fut), ref_quat_fut
    )

    return {
        "actor_smpl_gravity_projection_cur": kinematics.projected_gravity[
            :, cur
        ],
        "actor_smpl_base_linvel_cur": kinematics.root_linvel_local[:, cur],
        "actor_smpl_base_angvel_cur": kinematics.root_angvel_local[:, cur],
        "actor_smpl_body_command_cur": body[:, 0],
        "actor_smpl_root_height_cur": root_state[:, cur, 2:3],
        "actor_smpl_robot_yaw_error_sin_cos": torch.stack(
            (torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1
        ),
        "actor_smpl_body_command_fut": body[:, 1:],
        "actor_smpl_root_height_fut": root_state[:, fut, 2:3],
        "actor_smpl_gravity_projection_fut": kinematics.projected_gravity[
            :, fut
        ],
        "actor_smpl_base_linvel_fut": kinematics.root_linvel_local[:, fut],
        "actor_smpl_base_angvel_fut": kinematics.root_angvel_local[:, fut],
        "actor_smpl_future_yaw_delta_sin_cos": torch.stack(
            (torch.sin(future_yaw_delta), torch.cos(future_yaw_delta)), dim=-1
        ),
        "actor_smpl_future_root_ori_robot_frame_6d": _rot6d_from_quat_wxyz(
            relative_future
        ),
    }


__all__ = [
    "SMPL_ACTOR_CURRENT_TERMS",
    "SMPL_ACTOR_FUTURE_TERMS",
    "SmplActorObservationInput",
    "SmplMotionActorObservationInput",
    "build_smpl_motion_actor_observation_torch",
    "derive_smpl_actor_terms_torch",
    "launch_smpl_actor_observation_pack_warp",
    "smpl_motion_actor_observation_dim",
]
