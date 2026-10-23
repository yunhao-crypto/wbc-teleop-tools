# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Task-space tracking rewards shared by robot-reference and SMPL targets.

The command term owns target construction, the persistent episode anchor, and
the robot semantic-state convention (including virtual toes).  This module
only compares two state objects returned by the command term:

``command.get_task_space_target_immediate_next()``
``command.get_task_space_robot_state()``

Both objects must expose ``position_w``, ``rotation_w_wxyz``,
``linear_velocity_w``, and ``angular_velocity_w`` tensors in the semantic body
order defined by :data:`TASK_SPACE_BODY_NAMES`.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch

from holomotion.src.motion_tracking.task_space_diagnostics import (
    record_task_space_diagnostic,
)

TASK_SPACE_BODY_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "torso",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
"""Canonical 14-node task-space order used by command providers."""

PELVIS_INDEX = 0
FOOT_INDICES = (5, 6)
WRIST_INDICES = (12, 13)

SEMANTIC_POSITION_STD = 0.311324713
SEMANTIC_ROTATION_STD = 0.50
ROOT_VELOCITY_STD = 1.0
ROOT_VELOCITY_EPS = 0.1
PRECISION_POSITION_STD = 0.30
LINK_LINEAR_VELOCITY_STD = 1.0
LINK_ANGULAR_VELOCITY_STD = 3.14

# GMR-inspired Stage-2 relative importance, normalized only by its sum in the
# reward kernel.  The outer reward weight remains one.
TASK_SPACE_ROTATION_WEIGHTS = (
    8.5,  # pelvis
    8.5,  # left hip
    8.5,  # right hip
    8.5,  # left knee
    8.5,  # right knee
    39.8,  # left foot
    39.8,  # right foot
    8.5,  # torso
    4.5,  # left shoulder
    4.5,  # right shoulder
    4.0,  # left elbow
    4.0,  # right elbow
    16.2,  # left wrist
    16.2,  # right wrist
)
TASK_SPACE_ROTATION_WEIGHT_SUM = 180.0

_CONSTANT_TENSOR_CACHE: dict[
    tuple[tuple[float, ...], torch.device, torch.dtype],
    torch.Tensor,
] = {}


class TaskSpaceStateLike(Protocol):
    """Structural interface consumed by the environment wrappers."""

    position_w: torch.Tensor
    rotation_w_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    angular_velocity_w: torch.Tensor


def _validate_std(std: float) -> None:
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")


def _validate_state_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    trailing_shape: tuple[int, int],
) -> None:
    if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != trailing_shape:
        raise ValueError(
            f"{name} must have shape [B, {trailing_shape[0]}, "
            f"{trailing_shape[1]}], got {tuple(tensor.shape)}."
        )
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}.")


def _validate_vector_pair(
    actual: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> None:
    expected = (len(TASK_SPACE_BODY_NAMES), 3)
    _validate_state_tensor(
        actual, name=f"actual {name}", trailing_shape=expected
    )
    _validate_state_tensor(
        target, name=f"target {name}", trailing_shape=expected
    )
    if actual.shape != target.shape:
        raise ValueError(
            f"actual and target {name} shapes differ: "
            f"{tuple(actual.shape)} vs {tuple(target.shape)}."
        )


def _validate_quaternion_pair(
    actual: torch.Tensor,
    target: torch.Tensor,
) -> None:
    expected = (len(TASK_SPACE_BODY_NAMES), 4)
    _validate_state_tensor(
        actual,
        name="actual rotation_w_wxyz",
        trailing_shape=expected,
    )
    _validate_state_tensor(
        target,
        name="target rotation_w_wxyz",
        trailing_shape=expected,
    )
    if actual.shape != target.shape:
        raise ValueError(
            "actual and target rotation shapes differ: "
            f"{tuple(actual.shape)} vs {tuple(target.shape)}."
        )


def _normalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    return quaternion_wxyz / torch.clamp(
        norm,
        min=torch.finfo(quaternion_wxyz.dtype).eps,
    )


def _quat_apply_inverse(
    quaternion_wxyz: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``vector`` by the inverse of a WXYZ quaternion."""
    quaternion_wxyz = _normalize_quaternion(quaternion_wxyz)
    scalar = quaternion_wxyz[..., :1]
    imaginary = -quaternion_wxyz[..., 1:]
    first_cross = torch.cross(imaginary, vector, dim=-1)
    return vector + 2.0 * torch.cross(
        imaginary,
        first_cross + scalar * vector,
        dim=-1,
    )


def _quaternion_geodesic_angle(
    actual_wxyz: torch.Tensor,
    target_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return the sign-invariant SO(3) geodesic angle in radians."""
    actual_wxyz = _normalize_quaternion(actual_wxyz)
    target_wxyz = _normalize_quaternion(target_wxyz)
    absolute_dot = torch.abs(torch.sum(actual_wxyz * target_wxyz, dim=-1))
    absolute_dot = torch.clamp(absolute_dot, min=0.0, max=1.0)
    return 2.0 * torch.acos(absolute_dot)


def _bounded_exp_reward(
    squared_error: torch.Tensor,
    *,
    std: float,
) -> torch.Tensor:
    _validate_std(std)
    reward = torch.exp(-squared_error / std**2)
    return torch.clamp(reward, min=0.0, max=1.0)


def _constant_tensor(
    values: Sequence[float],
    reference: torch.Tensor,
) -> torch.Tensor:
    values = tuple(float(value) for value in values)
    key = (values, reference.device, reference.dtype)
    tensor = _CONSTANT_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = torch.tensor(
            values,
            device=reference.device,
            dtype=reference.dtype,
        )
        _CONSTANT_TENSOR_CACHE[key] = tensor
    return tensor


def _select_bodies(
    tensor: torch.Tensor,
    body_indices: tuple[int, ...],
) -> torch.Tensor:
    start = body_indices[0]
    if body_indices == tuple(range(start, start + len(body_indices))):
        return tensor[:, start : start + len(body_indices)]
    indices = torch.tensor(
        body_indices,
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(1, indices)


def _root_local_positions(
    position_w: torch.Tensor,
    rotation_w_wxyz: torch.Tensor,
) -> torch.Tensor:
    root_relative_w = (
        position_w - position_w[:, PELVIS_INDEX : PELVIS_INDEX + 1]
    )
    inverse_root = rotation_w_wxyz[:, PELVIS_INDEX : PELVIS_INDEX + 1].expand(
        -1, position_w.shape[1], -1
    )
    return _quat_apply_inverse(inverse_root, root_relative_w)


def semantic_position_tracking_reward(
    actual_position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    std: float = SEMANTIC_POSITION_STD,
) -> torch.Tensor:
    """Track 13 non-pelvis positions in each entity's full pelvis frame."""
    reward, _ = _semantic_position_reward_and_error(
        actual_position_w,
        target_position_w,
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
        std=std,
    )
    return reward


def _semantic_position_reward_and_error(
    actual_position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_vector_pair(
        actual_position_w,
        target_position_w,
        name="position_w",
    )
    _validate_quaternion_pair(
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
    )
    actual_local = _root_local_positions(
        actual_position_w,
        actual_rotation_w_wxyz,
    )
    target_local = _root_local_positions(
        target_position_w,
        target_rotation_w_wxyz,
    )
    per_body_squared_error = torch.sum(
        torch.square(actual_local - target_local),
        dim=-1,
    )
    mean_squared_error = per_body_squared_error[:, 1:].mean(dim=-1)
    reward = _bounded_exp_reward(mean_squared_error, std=std)
    return reward, torch.sqrt(torch.clamp(mean_squared_error, min=0.0))


def semantic_rotation_tracking_reward(
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    std: float = SEMANTIC_ROTATION_STD,
    body_weights: Sequence[float] = TASK_SPACE_ROTATION_WEIGHTS,
) -> torch.Tensor:
    """Track fixed-anchor global SO(3) rotations using frozen W18 weights."""
    reward, _ = _semantic_rotation_reward_and_error(
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
        std=std,
        body_weights=body_weights,
    )
    return reward


def _semantic_rotation_reward_and_error(
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    std: float,
    body_weights: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_quaternion_pair(
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
    )
    body_weights = tuple(float(weight) for weight in body_weights)
    if len(body_weights) != len(TASK_SPACE_BODY_NAMES):
        raise ValueError(
            "body_weights must contain one value for each task-space body."
        )
    weight_sum = sum(body_weights)
    if any(weight < 0.0 for weight in body_weights) or weight_sum <= 0.0:
        raise ValueError(
            "body_weights must be non-negative with positive sum."
        )
    weights = _constant_tensor(body_weights, actual_rotation_w_wxyz)
    angles = _quaternion_geodesic_angle(
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
    )
    weighted_squared_error = (
        torch.sum(
            torch.square(angles) * weights[None, :],
            dim=-1,
        )
        / weight_sum
    )
    reward = _bounded_exp_reward(weighted_squared_error, std=std)
    error = torch.sqrt(torch.clamp(weighted_squared_error, min=0.0))
    return reward, error


def root_velocity_tracking_reward(
    actual_velocity_w: torch.Tensor,
    target_velocity_w: torch.Tensor,
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    std: float = ROOT_VELOCITY_STD,
    eps: float = ROOT_VELOCITY_EPS,
) -> torch.Tensor:
    """Track root velocity with the baseline relative-ratio kernel."""
    reward, _ = _root_velocity_reward_and_error(
        actual_velocity_w,
        target_velocity_w,
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
        std=std,
        eps=eps,
    )
    return reward


def _root_velocity_reward_and_error(
    actual_velocity_w: torch.Tensor,
    target_velocity_w: torch.Tensor,
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    std: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_vector_pair(
        actual_velocity_w,
        target_velocity_w,
        name="velocity_w",
    )
    _validate_quaternion_pair(
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
    )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    actual_root_velocity = _quat_apply_inverse(
        actual_rotation_w_wxyz[:, PELVIS_INDEX],
        actual_velocity_w[:, PELVIS_INDEX],
    )
    target_root_velocity = _quat_apply_inverse(
        target_rotation_w_wxyz[:, PELVIS_INDEX],
        target_velocity_w[:, PELVIS_INDEX],
    )
    error_norm = torch.linalg.vector_norm(
        actual_root_velocity - target_root_velocity,
        dim=-1,
    )
    target_norm = torch.linalg.vector_norm(target_root_velocity, dim=-1)
    relative_error = error_norm / (target_norm + eps)
    reward = _bounded_exp_reward(torch.square(relative_error), std=std)
    return reward, error_norm


def precision_position_tracking_reward(
    actual_position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    body_indices: Sequence[int],
    std: float = PRECISION_POSITION_STD,
) -> torch.Tensor:
    """Track selected endpoints in each entity's full pelvis frame."""
    reward, _ = _precision_position_reward_and_error(
        actual_position_w,
        target_position_w,
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
        body_indices=body_indices,
        std=std,
    )
    return reward


def _precision_position_reward_and_error(
    actual_position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    actual_rotation_w_wxyz: torch.Tensor,
    target_rotation_w_wxyz: torch.Tensor,
    *,
    body_indices: Sequence[int],
    std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_vector_pair(
        actual_position_w,
        target_position_w,
        name="position_w",
    )
    _validate_quaternion_pair(
        actual_rotation_w_wxyz,
        target_rotation_w_wxyz,
    )
    body_indices = tuple(int(index) for index in body_indices)
    if not body_indices:
        raise ValueError("body_indices must be a non-empty 1-D sequence.")
    if any(
        index < 0 or index >= len(TASK_SPACE_BODY_NAMES)
        for index in body_indices
    ):
        raise ValueError("body_indices contains an out-of-range index.")
    actual_local = _root_local_positions(
        actual_position_w,
        actual_rotation_w_wxyz,
    )
    target_local = _root_local_positions(
        target_position_w,
        target_rotation_w_wxyz,
    )
    per_body_squared_error = torch.sum(
        torch.square(
            _select_bodies(actual_local, body_indices)
            - _select_bodies(target_local, body_indices)
        ),
        dim=-1,
    )
    mean_squared_error = per_body_squared_error.mean(dim=-1)
    reward = _bounded_exp_reward(mean_squared_error, std=std)
    return reward, torch.sqrt(torch.clamp(mean_squared_error, min=0.0))


def link_velocity_tracking_reward(
    actual_velocity_w: torch.Tensor,
    target_velocity_w: torch.Tensor,
    *,
    std: float,
) -> torch.Tensor:
    """Track 14 semantic link velocities directly in the world frame."""
    reward, _ = _link_velocity_reward_and_error(
        actual_velocity_w,
        target_velocity_w,
        std=std,
    )
    return reward


def _link_velocity_reward_and_error(
    actual_velocity_w: torch.Tensor,
    target_velocity_w: torch.Tensor,
    *,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_vector_pair(
        actual_velocity_w,
        target_velocity_w,
        name="velocity_w",
    )
    per_body_squared_error = torch.sum(
        torch.square(actual_velocity_w - target_velocity_w),
        dim=-1,
    )
    mean_squared_error = per_body_squared_error.mean(dim=-1)
    reward = _bounded_exp_reward(mean_squared_error, std=std)
    return reward, torch.sqrt(torch.clamp(mean_squared_error, min=0.0))


def _state_field(state: TaskSpaceStateLike | Mapping[str, Any], name: str):
    if isinstance(state, Mapping):
        try:
            return state[name]
        except KeyError as error:
            raise AttributeError(
                f"Task-space state mapping has no field {name!r}."
            ) from error
    try:
        return getattr(state, name)
    except AttributeError as error:
        raise AttributeError(
            f"Task-space state object has no field {name!r}."
        ) from error


def _task_space_states(env: Any, command_name: str):
    command = env.command_manager.get_term(command_name)
    target = command.get_task_space_target_immediate_next()
    actual = command.get_task_space_robot_state()
    return actual, target


def task_space_semantic_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _semantic_position_reward_and_error(
        _state_field(actual, "position_w"),
        _state_field(target, "position_w"),
        _state_field(actual, "rotation_w_wxyz"),
        _state_field(target, "rotation_w_wxyz"),
        std=std,
    )
    record_task_space_diagnostic(
        env,
        name="semantic_position",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_semantic_rotation_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _semantic_rotation_reward_and_error(
        _state_field(actual, "rotation_w_wxyz"),
        _state_field(target, "rotation_w_wxyz"),
        std=std,
        body_weights=TASK_SPACE_ROTATION_WEIGHTS,
    )
    record_task_space_diagnostic(
        env,
        name="semantic_rotation",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_root_linear_velocity_tracking_exp(
    env: Any,
    std: float,
    eps: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _root_velocity_reward_and_error(
        _state_field(actual, "linear_velocity_w"),
        _state_field(target, "linear_velocity_w"),
        _state_field(actual, "rotation_w_wxyz"),
        _state_field(target, "rotation_w_wxyz"),
        std=std,
        eps=eps,
    )
    record_task_space_diagnostic(
        env,
        name="root_linear_velocity",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_root_angular_velocity_tracking_exp(
    env: Any,
    std: float,
    eps: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _root_velocity_reward_and_error(
        _state_field(actual, "angular_velocity_w"),
        _state_field(target, "angular_velocity_w"),
        _state_field(actual, "rotation_w_wxyz"),
        _state_field(target, "rotation_w_wxyz"),
        std=std,
        eps=eps,
    )
    record_task_space_diagnostic(
        env,
        name="root_angular_velocity",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_foot_precision_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _precision_position_reward_and_error(
        _state_field(actual, "position_w"),
        _state_field(target, "position_w"),
        _state_field(actual, "rotation_w_wxyz"),
        _state_field(target, "rotation_w_wxyz"),
        body_indices=FOOT_INDICES,
        std=std,
    )
    record_task_space_diagnostic(
        env,
        name="foot_precision_position",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_wrist_precision_position_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _precision_position_reward_and_error(
        _state_field(actual, "position_w"),
        _state_field(target, "position_w"),
        _state_field(actual, "rotation_w_wxyz"),
        _state_field(target, "rotation_w_wxyz"),
        body_indices=WRIST_INDICES,
        std=std,
    )
    record_task_space_diagnostic(
        env,
        name="wrist_precision_position",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_link_linear_velocity_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _link_velocity_reward_and_error(
        _state_field(actual, "linear_velocity_w"),
        _state_field(target, "linear_velocity_w"),
        std=std,
    )
    record_task_space_diagnostic(
        env,
        name="link_linear_velocity",
        error=error,
        kernel=reward,
    )
    return reward


def task_space_link_angular_velocity_tracking_exp(
    env: Any,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    actual, target = _task_space_states(env, command_name)
    reward, error = _link_velocity_reward_and_error(
        _state_field(actual, "angular_velocity_w"),
        _state_field(target, "angular_velocity_w"),
        std=std,
    )
    record_task_space_diagnostic(
        env,
        name="link_angular_velocity",
        error=error,
        kernel=reward,
    )
    return reward
