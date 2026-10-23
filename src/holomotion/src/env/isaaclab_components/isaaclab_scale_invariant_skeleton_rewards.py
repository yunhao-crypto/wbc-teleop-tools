"""Scale-invariant SMPL-to-robot semantic skeleton rewards."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Protocol

import torch

from holomotion.src.motion_tracking.semantic_skeleton import (
    EDGE_GROUPS,
    FOOT_INDICES,
    LOCAL_CHAIN_EDGE_PAIRS,
    POSITION_GROUPS,
    local_bend_plane_features,
    pelvis_local_positions,
    unit_edge_directions,
)
from holomotion.src.motion_tracking.skeleton_reward_diagnostics import (
    record_skeleton_diagnostic,
)
from holomotion.src.utils import torch_utils

LOCAL_STRUCTURE_STD = 0.45
GLOBAL_STRUCTURE_STD_RAD = 0.35
NORMALIZED_POSITION_STD_BODY = 0.25
FOOT_SUPPORT_POSITION_STD_BODY = 0.35
FOOT_CENTER_POSITION_STD_BODY = 0.35
FOOT_RELATIVE_POSITION_STD_BODY = 0.35
ORIENTATION_STD_RAD = 0.50
ROOT_LINEAR_VELOCITY_STD_INV_S = 1.0
ROOT_ANGULAR_VELOCITY_STD_RADPS = 1.0
ROOT_YAW_RATE_STD_RADPS = 1.50
ROOT_GRAVITY_STD_RAD = 0.50


class SkeletonStateLike(Protocol):
    position_w: torch.Tensor
    pelvis_rotation_wxyz: torch.Tensor
    torso_rotation_wxyz: torch.Tensor
    endpoint_rotation_wxyz: torch.Tensor
    head_rotation_wxyz: torch.Tensor | None
    root_linear_velocity_w: torch.Tensor
    root_angular_velocity_w: torch.Tensor
    nominal_scale: torch.Tensor


def _field(state: SkeletonStateLike | Mapping[str, Any], name: str):
    if isinstance(state, Mapping):
        try:
            return state[name]
        except KeyError as error:
            raise AttributeError(
                f"Skeleton state mapping has no field {name!r}"
            ) from error
    try:
        return getattr(state, name)
    except AttributeError as error:
        raise AttributeError(
            f"Skeleton state object has no field {name!r}"
        ) from error


def _bounded_exp(error: torch.Tensor, *, std: float) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("std must be positive")
    return torch.exp(-error / (float(std) ** 2))


def _normalized_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four WXYZ values")
    return quaternion / torch.linalg.vector_norm(
        quaternion,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)


def _quaternion_geodesic_angle(
    actual_wxyz: torch.Tensor,
    target_wxyz: torch.Tensor,
) -> torch.Tensor:
    if actual_wxyz.shape != target_wxyz.shape:
        raise ValueError("actual and target quaternion shapes must match")
    actual = _normalized_quaternion(actual_wxyz)
    target = _normalized_quaternion(target_wxyz)
    cosine_half = torch.abs(torch.sum(actual * target, dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(cosine_half)


def _pelvis_matrix(
    state: SkeletonStateLike | Mapping[str, Any],
) -> torch.Tensor:
    return torch_utils.matrix_from_quat(_field(state, "pelvis_rotation_wxyz"))


def _mean_grouped(
    per_item: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    group_errors = []
    for group in groups:
        indices = torch.tensor(
            group,
            dtype=torch.long,
            device=per_item.device,
        )
        group_errors.append(per_item.index_select(-1, indices).mean(dim=-1))
    return torch.stack(group_errors, dim=-1).mean(dim=-1)


def _normalized_pelvis_local_positions(
    state: SkeletonStateLike | Mapping[str, Any],
) -> torch.Tensor:
    scale = _field(state, "nominal_scale")
    if scale.shape[-1:] != (1,):
        raise ValueError("nominal_scale must have shape [B,1]")
    if torch.any(scale <= 0.0):
        raise ValueError("nominal_scale must be positive")
    return (
        pelvis_local_positions(
            _field(state, "position_w"),
            _pelvis_matrix(state),
        )
        / scale[..., None]
    )


def local_structure_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = LOCAL_STRUCTURE_STD,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_direction, actual_valid = unit_edge_directions(
        _field(actual, "position_w"),
        _pelvis_matrix(actual),
    )
    target_direction, target_valid = unit_edge_directions(
        _field(target, "position_w"),
        _pelvis_matrix(target),
    )
    actual_bend, actual_plane, actual_plane_valid = local_bend_plane_features(
        actual_direction, actual_valid
    )
    target_bend, target_plane, target_plane_valid = local_bend_plane_features(
        target_direction, target_valid
    )
    proximal = torch.tensor(
        [pair[0] for pair in LOCAL_CHAIN_EDGE_PAIRS],
        dtype=torch.long,
        device=actual_valid.device,
    )
    distal = torch.tensor(
        [pair[1] for pair in LOCAL_CHAIN_EDGE_PAIRS],
        dtype=torch.long,
        device=actual_valid.device,
    )
    pair_valid = (
        actual_valid.index_select(-1, proximal)
        & actual_valid.index_select(-1, distal)
        & target_valid.index_select(-1, proximal)
        & target_valid.index_select(-1, distal)
    )
    plane_valid = actual_plane_valid & target_plane_valid & pair_valid
    bend_error = torch.square(actual_bend - target_bend)
    # A limb plane is unoriented: n and -n describe the same plane.  The
    # absolute cosine also avoids a discontinuous penalty when a nearly
    # straight limb crosses through zero bend.
    plane_error = 1.0 - torch.abs(
        torch.sum(actual_plane * target_plane, dim=-1)
    ).clamp(0.0, 1.0)
    plane_weight = plane_valid.to(dtype=bend_error.dtype)
    chain_error = (bend_error + plane_weight * plane_error) / (
        1.0 + plane_weight
    )
    # Invalid semantic edges must not silently look perfect.
    chain_error = torch.where(
        pair_valid,
        chain_error,
        torch.ones_like(chain_error),
    )
    mean_error = chain_error.mean(dim=-1)
    reward = _bounded_exp(mean_error, std=std)
    return reward, torch.sqrt(mean_error.clamp_min(0.0))


def global_structure_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = GLOBAL_STRUCTURE_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_direction, actual_valid = unit_edge_directions(
        _field(actual, "position_w"),
        _pelvis_matrix(actual),
    )
    target_direction, target_valid = unit_edge_directions(
        _field(target, "position_w"),
        _pelvis_matrix(target),
    )
    valid = actual_valid & target_valid
    # 2(1-cos(theta)) is smooth at zero and approximates theta^2.
    per_edge = 2.0 * (
        1.0
        - torch.sum(actual_direction * target_direction, dim=-1).clamp(
            -1.0, 1.0
        )
    )
    per_edge = torch.where(valid, per_edge, torch.ones_like(per_edge))
    mean_error = _mean_grouped(per_edge, EDGE_GROUPS)
    reward = _bounded_exp(mean_error, std=std)
    return reward, torch.sqrt(mean_error.clamp_min(0.0))


def normalized_position_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = NORMALIZED_POSITION_STD_BODY,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_local = _normalized_pelvis_local_positions(actual)
    target_local = _normalized_pelvis_local_positions(target)
    per_node = torch.sum(torch.square(actual_local - target_local), dim=-1)
    mean_error = _mean_grouped(per_node, POSITION_GROUPS)
    reward = _bounded_exp(mean_error, std=std)
    return reward, torch.sqrt(mean_error.clamp_min(0.0))


def foot_support_position_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = FOOT_SUPPORT_POSITION_STD_BODY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare bilateral foot placement without morphology-specific scales."""
    actual_local = _normalized_pelvis_local_positions(actual)
    target_local = _normalized_pelvis_local_positions(target)
    foot_indices = torch.tensor(
        FOOT_INDICES,
        dtype=torch.long,
        device=actual_local.device,
    )
    delta = actual_local.index_select(
        -2, foot_indices
    ) - target_local.index_select(-2, foot_indices)
    per_foot = torch.sum(torch.square(delta), dim=-1)
    mean_error = per_foot.mean(dim=-1)
    reward = _bounded_exp(mean_error, std=std)
    return reward, torch.sqrt(mean_error.clamp_min(0.0))


def _normalized_foot_positions(
    state: SkeletonStateLike | Mapping[str, Any],
) -> torch.Tensor:
    positions = _normalized_pelvis_local_positions(state)
    foot_indices = torch.tensor(
        FOOT_INDICES,
        dtype=torch.long,
        device=positions.device,
    )
    return positions.index_select(-2, foot_indices)


def foot_center_position_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = FOOT_CENTER_POSITION_STD_BODY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare the bilateral foot center in normalized pelvis-local space."""
    actual_center = _normalized_foot_positions(actual).mean(dim=-2)
    target_center = _normalized_foot_positions(target).mean(dim=-2)
    squared_error = torch.sum(
        torch.square(actual_center - target_center),
        dim=-1,
    )
    reward = _bounded_exp(squared_error, std=std)
    return reward, torch.sqrt(squared_error.clamp_min(0.0))


def foot_relative_position_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = FOOT_RELATIVE_POSITION_STD_BODY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare signed left-minus-right foot placement and crossing."""
    actual_feet = _normalized_foot_positions(actual)
    target_feet = _normalized_foot_positions(target)
    actual_relative = actual_feet[:, 0] - actual_feet[:, 1]
    target_relative = target_feet[:, 0] - target_feet[:, 1]
    squared_error = torch.sum(
        torch.square(actual_relative - target_relative),
        dim=-1,
    )
    reward = _bounded_exp(squared_error, std=std)
    return reward, torch.sqrt(squared_error.clamp_min(0.0))


def pelvis_torso_orientation_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = ORIENTATION_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_rotation = torch.stack(
        (
            _field(actual, "pelvis_rotation_wxyz"),
            _field(actual, "torso_rotation_wxyz"),
        ),
        dim=1,
    )
    target_rotation = torch.stack(
        (
            _field(target, "pelvis_rotation_wxyz"),
            _field(target, "torso_rotation_wxyz"),
        ),
        dim=1,
    )
    angles = _quaternion_geodesic_angle(actual_rotation, target_rotation)
    mean_error = torch.square(angles).mean(dim=-1)
    reward = _bounded_exp(mean_error, std=std)
    return reward, torch.sqrt(mean_error.clamp_min(0.0))


def endpoint_orientation_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    endpoint_slice: slice,
    std: float = ORIENTATION_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_rotation = _field(
        actual,
        "endpoint_rotation_wxyz",
    )[:, endpoint_slice]
    target_rotation = _field(
        target,
        "endpoint_rotation_wxyz",
    )[:, endpoint_slice]
    angles = _quaternion_geodesic_angle(actual_rotation, target_rotation)
    mean_error = torch.square(angles).mean(dim=-1)
    reward = _bounded_exp(mean_error, std=std)
    return reward, torch.sqrt(mean_error.clamp_min(0.0))


def head_orientation_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = ORIENTATION_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare full head orientation for embodiments with an actuated head."""

    actual_rotation = _field(actual, "head_rotation_wxyz")
    target_rotation = _field(target, "head_rotation_wxyz")
    if actual_rotation is None or target_rotation is None:
        raise ValueError(
            "Head orientation reward requires both robot and SMPL head frames"
        )
    angles = _quaternion_geodesic_angle(actual_rotation, target_rotation)
    squared_error = torch.square(angles)
    reward = _bounded_exp(squared_error, std=std)
    return reward, angles


def head_yaw_pitch_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = ORIENTATION_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare torso-local head direction without penalizing head roll.

    A two-axis robot head can express yaw and pitch but cannot independently
    express roll.  Comparing the torso-local forward axes measures exactly
    those two observable degrees of freedom and is invariant to head roll.
    """

    actual_head = _field(actual, "head_rotation_wxyz")
    target_head = _field(target, "head_rotation_wxyz")
    if actual_head is None or target_head is None:
        raise ValueError(
            "Head yaw/pitch reward requires both robot and SMPL head frames"
        )
    actual_torso = torch_utils.matrix_from_quat(
        _field(actual, "torso_rotation_wxyz")
    )
    target_torso = torch_utils.matrix_from_quat(
        _field(target, "torso_rotation_wxyz")
    )
    actual_relative = actual_torso.transpose(-1, -2) @ (
        torch_utils.matrix_from_quat(actual_head)
    )
    target_relative = target_torso.transpose(-1, -2) @ (
        torch_utils.matrix_from_quat(target_head)
    )
    actual_forward = actual_relative[..., :, 0]
    target_forward = target_relative[..., :, 0]
    cosine = torch.sum(actual_forward * target_forward, dim=-1).clamp(
        -1.0, 1.0
    )
    forward_angle = torch.acos(cosine)
    squared_chord_error = 2.0 * (1.0 - cosine)
    reward = _bounded_exp(squared_chord_error, std=std)

    actual_yaw = torch.atan2(actual_forward[..., 1], actual_forward[..., 0])
    target_yaw = torch.atan2(target_forward[..., 1], target_forward[..., 0])
    yaw_error = torch.atan2(
        torch.sin(actual_yaw - target_yaw),
        torch.cos(actual_yaw - target_yaw),
    ).abs()
    actual_pitch = torch.atan2(
        -actual_forward[..., 2],
        torch.linalg.vector_norm(actual_forward[..., :2], dim=-1),
    )
    target_pitch = torch.atan2(
        -target_forward[..., 2],
        torch.linalg.vector_norm(target_forward[..., :2], dim=-1),
    )
    pitch_error = (actual_pitch - target_pitch).abs()
    return reward, forward_angle, yaw_error, pitch_error


def head_yaw_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = ORIENTATION_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare only torso-local head yaw for a one-axis robot head."""

    actual_head = _field(actual, "head_rotation_wxyz")
    target_head = _field(target, "head_rotation_wxyz")
    if actual_head is None or target_head is None:
        raise ValueError(
            "Head yaw reward requires both robot and SMPL head frames"
        )
    actual_torso = torch_utils.matrix_from_quat(
        _field(actual, "torso_rotation_wxyz")
    )
    target_torso = torch_utils.matrix_from_quat(
        _field(target, "torso_rotation_wxyz")
    )
    actual_relative = actual_torso.transpose(-1, -2) @ (
        torch_utils.matrix_from_quat(actual_head)
    )
    target_relative = target_torso.transpose(-1, -2) @ (
        torch_utils.matrix_from_quat(target_head)
    )
    actual_forward = actual_relative[..., :, 0]
    target_forward = target_relative[..., :, 0]
    actual_yaw = torch.atan2(actual_forward[..., 1], actual_forward[..., 0])
    target_yaw = torch.atan2(target_forward[..., 1], target_forward[..., 0])
    yaw_error = torch.atan2(
        torch.sin(actual_yaw - target_yaw),
        torch.cos(actual_yaw - target_yaw),
    ).abs()
    return _bounded_exp(torch.square(yaw_error), std=std), yaw_error


def root_velocity_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    field_name: str,
    std: float,
    normalize_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_velocity = _field(actual, field_name)
    target_velocity = _field(target, field_name)
    actual_local = (
        _pelvis_matrix(actual).transpose(-1, -2)
        @ actual_velocity.unsqueeze(-1)
    ).squeeze(-1)
    target_local = (
        _pelvis_matrix(target).transpose(-1, -2)
        @ target_velocity.unsqueeze(-1)
    ).squeeze(-1)
    if normalize_scale:
        actual_local = actual_local / _field(actual, "nominal_scale")
        target_local = target_local / _field(target, "nominal_scale")
    squared_error = torch.sum(
        torch.square(actual_local - target_local),
        dim=-1,
    )
    reward = _bounded_exp(squared_error, std=std)
    return reward, torch.sqrt(squared_error.clamp_min(0.0))


def root_yaw_rate_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = ROOT_YAW_RATE_STD_RADPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare signed root angular velocity around the world vertical axis."""
    actual_yaw_rate = _field(actual, "root_angular_velocity_w")[:, 2]
    target_yaw_rate = _field(target, "root_angular_velocity_w")[:, 2]
    squared_error = torch.square(actual_yaw_rate - target_yaw_rate)
    reward = _bounded_exp(squared_error, std=std)
    return reward, torch.sqrt(squared_error.clamp_min(0.0))


def root_gravity_reward_and_error(
    actual: SkeletonStateLike | Mapping[str, Any],
    target: SkeletonStateLike | Mapping[str, Any],
    *,
    std: float = ROOT_GRAVITY_STD_RAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare pelvis tilt without penalizing global heading or body scale."""
    actual_matrix = _pelvis_matrix(actual)
    target_matrix = _pelvis_matrix(target)
    gravity_w = actual_matrix.new_tensor((0.0, 0.0, -1.0))
    actual_gravity = (
        actual_matrix.transpose(-1, -2)
        @ gravity_w.expand(actual_matrix.shape[0], -1).unsqueeze(-1)
    ).squeeze(-1)
    target_gravity = (
        target_matrix.transpose(-1, -2)
        @ gravity_w.expand(target_matrix.shape[0], -1).unsqueeze(-1)
    ).squeeze(-1)
    cosine = torch.sum(actual_gravity * target_gravity, dim=-1).clamp(
        -1.0,
        1.0,
    )
    # Squared chord distance is smooth at zero and approximates angle squared.
    squared_chord_error = 2.0 * (1.0 - cosine)
    reward = _bounded_exp(squared_chord_error, std=std)
    return reward, torch.acos(cosine)


def _states(env: Any, command_name: str):
    command = env.command_manager.get_term(command_name)
    return (
        command.get_semantic_skeleton_robot_state(),
        command.get_semantic_skeleton_target_immediate_next(),
    )


def _record(
    env: Any,
    *,
    name: str,
    result: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    reward, error = result
    record_skeleton_diagnostic(
        env,
        name=name,
        error=error,
        kernel=reward,
    )
    return reward


def skeleton_si_local_structure_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="local_structure",
        result=local_structure_reward_and_error(actual, target, std=std),
    )


def skeleton_si_global_structure_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="global_structure",
        result=global_structure_reward_and_error(actual, target, std=std),
    )


def skeleton_si_normalized_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="normalized_position",
        result=normalized_position_reward_and_error(actual, target, std=std),
    )


def skeleton_si_foot_support_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="foot_support_position",
        result=foot_support_position_reward_and_error(
            actual,
            target,
            std=std,
        ),
    )


def skeleton_si_foot_center_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="foot_center_position",
        result=foot_center_position_reward_and_error(
            actual,
            target,
            std=std,
        ),
    )


def skeleton_si_foot_relative_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="foot_relative_position",
        result=foot_relative_position_reward_and_error(
            actual,
            target,
            std=std,
        ),
    )


def skeleton_si_pelvis_torso_orientation_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="pelvis_torso_orientation",
        result=pelvis_torso_orientation_reward_and_error(
            actual,
            target,
            std=std,
        ),
    )


def skeleton_si_foot_orientation_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="foot_orientation",
        result=endpoint_orientation_reward_and_error(
            actual,
            target,
            endpoint_slice=slice(0, len(FOOT_INDICES)),
            std=std,
        ),
    )


def skeleton_si_wrist_orientation_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="wrist_orientation",
        result=endpoint_orientation_reward_and_error(
            actual,
            target,
            endpoint_slice=slice(len(FOOT_INDICES), None),
            std=std,
        ),
    )


def skeleton_si_head_orientation_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="head_orientation",
        result=head_orientation_reward_and_error(actual, target, std=std),
    )


def skeleton_si_head_yaw_pitch_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    reward, forward_error, yaw_error, pitch_error = (
        head_yaw_pitch_reward_and_error(actual, target, std=std)
    )
    for name, error in (
        ("head_forward", forward_error),
        ("head_yaw", yaw_error),
        ("head_pitch", pitch_error),
    ):
        record_skeleton_diagnostic(
            env,
            name=name,
            error=error,
            kernel=reward,
        )
    return reward


def skeleton_si_head_yaw_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="head_yaw",
        result=head_yaw_reward_and_error(actual, target, std=std),
    )


def skeleton_si_root_linear_velocity_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="root_linear_velocity",
        result=root_velocity_reward_and_error(
            actual,
            target,
            field_name="root_linear_velocity_w",
            std=std,
            normalize_scale=True,
        ),
    )


def skeleton_si_root_angular_velocity_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="root_angular_velocity",
        result=root_velocity_reward_and_error(
            actual,
            target,
            field_name="root_angular_velocity_w",
            std=std,
            normalize_scale=False,
        ),
    )


def skeleton_si_root_yaw_rate_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="root_yaw_rate",
        result=root_yaw_rate_reward_and_error(
            actual,
            target,
            std=std,
        ),
    )


def skeleton_si_root_gravity_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _states(env, command_name)
    return _record(
        env,
        name="root_gravity",
        result=root_gravity_reward_and_error(actual, target, std=std),
    )


__all__ = [
    "foot_center_position_reward_and_error",
    "foot_relative_position_reward_and_error",
    "foot_support_position_reward_and_error",
    "global_structure_reward_and_error",
    "head_orientation_reward_and_error",
    "head_yaw_reward_and_error",
    "head_yaw_pitch_reward_and_error",
    "local_structure_reward_and_error",
    "normalized_position_reward_and_error",
    "pelvis_torso_orientation_reward_and_error",
    "root_gravity_reward_and_error",
    "root_yaw_rate_reward_and_error",
    "root_velocity_reward_and_error",
    "skeleton_si_foot_center_position_tracking_exp",
    "skeleton_si_foot_orientation_tracking_exp",
    "skeleton_si_foot_relative_position_tracking_exp",
    "skeleton_si_foot_support_position_tracking_exp",
    "skeleton_si_global_structure_tracking_exp",
    "skeleton_si_head_orientation_tracking_exp",
    "skeleton_si_head_yaw_tracking_exp",
    "skeleton_si_head_yaw_pitch_tracking_exp",
    "skeleton_si_local_structure_tracking_exp",
    "skeleton_si_normalized_position_tracking_exp",
    "skeleton_si_pelvis_torso_orientation_tracking_exp",
    "skeleton_si_root_angular_velocity_tracking_exp",
    "skeleton_si_root_gravity_tracking_exp",
    "skeleton_si_root_linear_velocity_tracking_exp",
    "skeleton_si_root_yaw_rate_tracking_exp",
    "skeleton_si_wrist_orientation_tracking_exp",
]
