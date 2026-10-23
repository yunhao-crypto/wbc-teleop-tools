"""Behavior metrics used to gate velocity-policy acquisition curricula."""

from __future__ import annotations

import torch


def command_tracking_score(
    commands: torch.Tensor,
    linear_velocity: torch.Tensor,
    yaw_velocity: torch.Tensor,
    *,
    idle_linear_velocity_tolerance: float = 0.2,
    idle_yaw_velocity_tolerance: float = 0.3,
    command_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return a command-relative tracking score in ``[0, 1]``.

    A stationary robot receives zero for any non-zero commanded axis.  Motion
    perpendicular to the planar command and motion on an idle yaw axis are
    bounded by absolute tolerances, so one well-tracked component cannot hide
    an uncontrolled component.
    """

    if commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("commands must have shape (N, 3)")
    if linear_velocity.ndim != 2 or linear_velocity.shape[0] != len(commands):
        raise ValueError("linear_velocity must have shape (N, D)")
    if linear_velocity.shape[1] < 2:
        raise ValueError("linear_velocity must provide x/y components")
    if yaw_velocity.shape != commands.shape[:1]:
        raise ValueError("yaw_velocity must have shape (N,)")
    if idle_linear_velocity_tolerance <= 0.0:
        raise ValueError("idle_linear_velocity_tolerance must be positive")
    if idle_yaw_velocity_tolerance <= 0.0:
        raise ValueError("idle_yaw_velocity_tolerance must be positive")
    if command_epsilon <= 0.0:
        raise ValueError("command_epsilon must be positive")

    command_xy = commands[:, :2]
    measured_xy = linear_velocity[:, :2]
    command_linear_magnitude = torch.linalg.vector_norm(command_xy, dim=1)
    command_yaw_magnitude = commands[:, 2].abs()
    linear_active = command_linear_magnitude > command_epsilon
    yaw_active = command_yaw_magnitude > command_epsilon

    safe_linear_magnitude = command_linear_magnitude.clamp_min(command_epsilon)
    command_direction = command_xy / safe_linear_magnitude.unsqueeze(1)
    measured_parallel_speed = torch.sum(measured_xy * command_direction, dim=1)
    measured_perpendicular = (
        measured_xy - measured_parallel_speed.unsqueeze(1) * command_direction
    )

    parallel_relative_score = 1.0 - (
        (command_linear_magnitude - measured_parallel_speed).abs()
        / safe_linear_magnitude
    )
    perpendicular_idle_score = 1.0 - (
        torch.linalg.vector_norm(measured_perpendicular, dim=1)
        / float(idle_linear_velocity_tolerance)
    )
    active_linear_score = torch.minimum(
        parallel_relative_score, perpendicular_idle_score
    )

    yaw_relative_score = 1.0 - (
        (commands[:, 2] - yaw_velocity).abs()
        / command_yaw_magnitude.clamp_min(command_epsilon)
    )
    idle_linear_score = 1.0 - (
        torch.linalg.vector_norm(measured_xy, dim=1)
        / float(idle_linear_velocity_tolerance)
    )
    idle_yaw_score = 1.0 - (
        yaw_velocity.abs() / float(idle_yaw_velocity_tolerance)
    )

    linear_score = torch.where(
        linear_active, active_linear_score, idle_linear_score
    )
    yaw_score = torch.where(yaw_active, yaw_relative_score, idle_yaw_score)
    return torch.minimum(linear_score, yaw_score).clamp_(0.0, 1.0)
