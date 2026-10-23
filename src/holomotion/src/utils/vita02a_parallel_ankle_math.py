# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Torch kinematics for the Vita02A two-motor parallel ankle."""

from __future__ import annotations

import numpy as np
import torch

VITA02A_LX = 0.05
VITA02A_LY = 0.02175
VITA02A_Z_TOP = 0.31
VITA02A_Z_BOT = 0.235


def _stack3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, y, z), dim=-1)


def _constant_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(reference, value)


def _rotate_y(angle: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x = cosine * vector[..., 0] + sine * vector[..., 2]
    y = vector[..., 1].expand_as(x)
    z = -sine * vector[..., 0] + cosine * vector[..., 2]
    return _stack3(x, y, z)


def _rotate_x(angle: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x = vector[..., 0].expand_as(cosine)
    y = cosine * vector[..., 1] - sine * vector[..., 2]
    z = sine * vector[..., 1] + cosine * vector[..., 2]
    return _stack3(x, y, z)


def _ankle_motor_angles(
    pitch: torch.Tensor,
    roll: torch.Tensor,
    *,
    lx: float,
    ly: float,
    z_top: float,
    z_bot: float,
) -> torch.Tensor:
    zero = _constant_like(pitch, 0.0)
    top = _stack3(
        zero, _constant_like(pitch, ly), _constant_like(pitch, z_top)
    )
    bottom = _stack3(
        zero,
        _constant_like(pitch, -ly),
        _constant_like(pitch, z_bot),
    )
    top_zero = _stack3(
        _constant_like(pitch, -lx),
        _constant_like(pitch, ly),
        zero,
    )
    bottom_zero = _stack3(
        _constant_like(pitch, -lx),
        _constant_like(pitch, -ly),
        zero,
    )

    angles = []
    for anchor, crank_zero, rod_length in (
        (top, top_zero, z_top),
        (bottom, bottom_zero, z_bot),
    ):
        crank = _rotate_y(pitch, _rotate_x(roll, crank_zero))
        delta = crank - anchor
        a = delta[..., 0]
        b = delta[..., 2]
        c = -(
            rod_length * rod_length
            - lx * lx
            - torch.sum(delta * delta, dim=-1)
        ) / (2.0 * lx)
        discriminant = b * b * c * c - (a * a + b * b) * (c * c - a * a)
        sine = (b * c + torch.sqrt(torch.clamp(discriminant, min=0.0))) / (
            a * a + b * b
        )
        valid = (discriminant >= -1.0e-12) & (torch.abs(sine) <= 1.0 + 1.0e-5)
        angles.append(
            torch.where(
                valid, torch.asin(torch.clamp(sine, -1.0, 1.0)), float("nan")
            )
        )
    return torch.stack(angles, dim=-1)


def vita02a_ankle_jacobian(
    pitch: torch.Tensor,
    roll: torch.Tensor,
    *,
    lx: float = VITA02A_LX,
    ly: float = VITA02A_LY,
    z_top: float = VITA02A_Z_TOP,
    z_bot: float = VITA02A_Z_BOT,
) -> torch.Tensor:
    """Return motor-angle Jacobian with columns ``[roll, pitch]``."""

    angles = _ankle_motor_angles(
        pitch,
        roll,
        lx=lx,
        ly=ly,
        z_top=z_top,
        z_bot=z_bot,
    )
    zero = _constant_like(pitch, 0.0)
    top = _stack3(
        zero, _constant_like(pitch, ly), _constant_like(pitch, z_top)
    )
    bottom = _stack3(
        zero,
        _constant_like(pitch, -ly),
        _constant_like(pitch, z_bot),
    )
    crank_arm = _stack3(_constant_like(pitch, -lx), zero, zero)
    top_zero = _stack3(
        _constant_like(pitch, -lx),
        _constant_like(pitch, ly),
        zero,
    )
    bottom_zero = _stack3(
        _constant_like(pitch, -lx),
        _constant_like(pitch, -ly),
        zero,
    )

    rows = []
    roll_axis = _stack3(torch.cos(pitch), zero, -torch.sin(pitch))
    pitch_axis = _stack3(zero, _constant_like(pitch, 1.0), zero)
    motor_axis = pitch_axis
    for anchor, crank_zero, angle in (
        (top, top_zero, angles[..., 0]),
        (bottom, bottom_zero, angles[..., 1]),
    ):
        bar_end = anchor + _rotate_y(angle, crank_arm)
        crank_end = _rotate_y(pitch, _rotate_x(roll, crank_zero))
        rod = crank_end - bar_end
        bar = bar_end - anchor
        denominator = torch.sum(
            motor_axis * torch.cross(bar, rod, dim=-1), dim=-1
        )
        rows.append(
            torch.stack(
                (
                    torch.sum(
                        torch.cross(crank_end, rod, dim=-1) * roll_axis,
                        dim=-1,
                    )
                    / denominator,
                    torch.sum(
                        torch.cross(crank_end, rod, dim=-1) * pitch_axis,
                        dim=-1,
                    )
                    / denominator,
                ),
                dim=-1,
            )
        )
    return torch.stack(rows, dim=-2)


def project_vita02a_ankle_torque(
    pitch: torch.Tensor,
    roll: torch.Tensor,
    tau_pitch: torch.Tensor,
    tau_roll: torch.Tensor,
    *,
    tau_motor_max: float,
    lx: float = VITA02A_LX,
    ly: float = VITA02A_LY,
    z_top: float = VITA02A_Z_TOP,
    z_bot: float = VITA02A_Z_BOT,
    return_valid: bool = False,
):
    """Apply physical motor caps; invalid samples produce zero ankle effort.

    ``return_valid=True`` also returns the mask used by the actuator's fault
    counter. No host synchronization or whole-batch failure on a bad sample.
    """
    if not np.isfinite(tau_motor_max) or tau_motor_max <= 0:
        raise ValueError("tau_motor_max must be finite and positive")

    jacobian = vita02a_ankle_jacobian(
        pitch,
        roll,
        lx=lx,
        ly=ly,
        z_top=z_top,
        z_bot=z_bot,
    )
    virtual_torque = torch.stack((tau_roll, tau_pitch), dim=-1).unsqueeze(-1)
    determinant = (
        jacobian[..., 0, 0] * jacobian[..., 1, 1]
        - jacobian[..., 0, 1] * jacobian[..., 1, 0]
    )
    valid = (
        torch.isfinite(jacobian).all(dim=-1).all(dim=-1)
        & (torch.abs(determinant) > 1.0e-6)
        & torch.isfinite(virtual_torque).all(dim=-1).all(dim=-1)
    )
    safe_jacobian = torch.where(
        valid[..., None, None],
        jacobian,
        torch.eye(2, device=jacobian.device, dtype=jacobian.dtype),
    )
    safe_virtual = torch.where(valid[..., None, None], virtual_torque, 0.0)
    motor_torque = torch.linalg.solve(
        safe_jacobian.transpose(-1, -2), safe_virtual
    ).squeeze(-1)
    motor_torque = torch.clamp(
        motor_torque, min=-tau_motor_max, max=tau_motor_max
    )
    projected = torch.matmul(
        safe_jacobian.transpose(-1, -2), motor_torque.unsqueeze(-1)
    ).squeeze(-1)
    if return_valid:
        return projected[..., 1], projected[..., 0], valid
    return projected[..., 1], projected[..., 0]


def vita02a_ankle_jacobian_numpy(
    pitch: float,
    roll: float,
    *,
    lx: float = VITA02A_LX,
    ly: float = VITA02A_LY,
    z_top: float = VITA02A_Z_TOP,
    z_bot: float = VITA02A_Z_BOT,
) -> np.ndarray:
    """NumPy equivalent used by the CPU MuJoCo evaluator."""

    def rotate_y(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
        )

    def rotate_x(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
        )

    rotation = rotate_y(pitch) @ rotate_x(roll)
    anchors = (
        np.asarray((0.0, ly, z_top)),
        np.asarray((0.0, -ly, z_bot)),
    )
    crank_zeros = (
        np.asarray((-lx, ly, 0.0)),
        np.asarray((-lx, -ly, 0.0)),
    )
    crank_arm = np.asarray((-lx, 0.0, 0.0))
    motor_angles = []
    for anchor, crank_zero, rod_length in zip(
        anchors, crank_zeros, (z_top, z_bot), strict=True
    ):
        delta = rotation @ crank_zero - anchor
        a, _, b = delta
        c = -(rod_length**2 - lx**2 - delta @ delta) / (2.0 * lx)
        discriminant = b * b * c * c - (a * a + b * b) * (c * c - a * a)
        if not np.isfinite(discriminant) or discriminant < -1.0e-12:
            raise ValueError("Invalid parallel ankle linkage state")
        discriminant = max(discriminant, 0.0)
        sine = (b * c + np.sqrt(discriminant)) / (a * a + b * b)
        if not np.isfinite(sine) or abs(sine) > 1.0 + 1.0e-5:
            raise ValueError("Invalid parallel ankle motor angle")
        motor_angles.append(np.arcsin(np.clip(sine, -1.0, 1.0)))

    roll_axis = np.asarray((np.cos(pitch), 0.0, -np.sin(pitch)))
    pitch_axis = np.asarray((0.0, 1.0, 0.0))
    rows = []
    for anchor, crank_zero, angle in zip(
        anchors, crank_zeros, motor_angles, strict=True
    ):
        bar_end = anchor + rotate_y(float(angle)) @ crank_arm
        crank_end = rotation @ crank_zero
        rod = crank_end - bar_end
        bar = bar_end - anchor
        denominator = pitch_axis @ np.cross(bar, rod)
        if not np.isfinite(denominator) or abs(denominator) < 1.0e-10:
            raise ValueError("Singular parallel ankle linkage state")
        rows.append(
            (
                np.cross(crank_end, rod) @ roll_axis / denominator,
                np.cross(crank_end, rod) @ pitch_axis / denominator,
            )
        )
    jacobian = np.asarray(rows, dtype=np.float64)
    if (
        not np.isfinite(jacobian).all()
        or abs(np.linalg.det(jacobian)) <= 1.0e-6
    ):
        raise ValueError("Singular parallel ankle Jacobian")
    return jacobian


def project_vita02a_ankle_torque_numpy(
    pitch: float,
    roll: float,
    tau_pitch: float,
    tau_roll: float,
    *,
    tau_motor_max: float,
    lx: float = VITA02A_LX,
    ly: float = VITA02A_LY,
    z_top: float = VITA02A_Z_TOP,
    z_bot: float = VITA02A_Z_BOT,
) -> tuple[float, float]:
    """Project one MuJoCo virtual ankle torque through both motor caps."""

    if (
        not np.isfinite([tau_motor_max, tau_pitch, tau_roll]).all()
        or tau_motor_max <= 0
    ):
        raise ValueError("Invalid parallel ankle torque input")

    jacobian = vita02a_ankle_jacobian_numpy(
        pitch,
        roll,
        lx=lx,
        ly=ly,
        z_top=z_top,
        z_bot=z_bot,
    )
    virtual = np.asarray((tau_roll, tau_pitch), dtype=np.float64)
    motor = np.linalg.solve(jacobian.T, virtual)
    motor = np.clip(motor, -tau_motor_max, tau_motor_max)
    projected = jacobian.T @ motor
    return float(projected[1]), float(projected[0])
