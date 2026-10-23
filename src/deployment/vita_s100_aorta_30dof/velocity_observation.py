"""Exact ten-frame observation used by the official Vita02A walk policy."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from deployment.vita_s100_aorta_30dof.model_contract import (
    VITA_VELOCITY_ACTION_DIM,
    VITA_VELOCITY_OBSERVATION_DIM,
    VitaPolicyContract,
)
from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState


@dataclass(frozen=True)
class VitaVelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def as_array(self) -> np.ndarray:
        result = np.asarray(
            [self.vx, self.vy, self.yaw_rate], dtype=np.float32
        )
        if not np.isfinite(result).all():
            raise ValueError("velocity command contains NaN or Inf")
        return result


def projected_gravity_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float32).reshape(4)
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def _floats(contract: VitaPolicyContract, key: str, size: int) -> np.ndarray:
    result = np.asarray(
        [float(part) for part in contract.metadata[key].split(",")],
        dtype=np.float32,
    )
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"velocity metadata {key!r} is invalid")
    return result


class VitaVelocityObservationBuilder:
    """Build frame-major ``10 x 99 = 990`` official-policy inputs."""

    def __init__(self, contract: VitaPolicyContract) -> None:
        if contract.kind != "velocity":
            raise ValueError(
                "velocity observation requires a velocity contract"
            )
        self.contract = contract
        self.history_length = int(contract.metadata["history_length"])
        self.frame_dim = int(contract.metadata["frame_dim"])
        if self.history_length != 10 or self.frame_dim != 99:
            raise ValueError("official Vita velocity ABI must be 10 x 99")
        self.history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self.policy_step = 0
        self.gait_cycle = float(contract.metadata["gait_cycle"])
        self.gait_offsets = _floats(contract, "gait_phase_offsets", 2)
        self.phase_ratio = _floats(contract, "gait_phase_ratios", 2)
        self.standing_phase = _floats(contract, "gait_phase_standing_value", 2)
        self.linear_deadband = float(
            contract.metadata["command_standing_linear_deadband"]
        )
        self.yaw_deadband = float(
            contract.metadata["command_standing_yaw_deadband"]
        )
        self.clip_observations = float(contract.metadata["clip_observations"])
        self.policy_dt = float(contract.metadata["policy_dt"])
        self.scales = {
            name: float(contract.metadata[f"obs_scale_{name}"])
            for name in (
                "base_ang_vel",
                "projected_gravity",
                "command",
                "joint_pos",
                "joint_vel",
                "actions",
            )
        }

    def reset(self) -> None:
        self.history.clear()
        self.policy_step = 0

    def _command(
        self, value: VitaVelocityCommand | Sequence[float]
    ) -> np.ndarray:
        result = (
            value.as_array()
            if isinstance(value, VitaVelocityCommand)
            else np.asarray(value, dtype=np.float32).reshape(-1)
        )
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(
                "velocity command must be finite [vx, vy, yaw_rate]"
            )
        for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            lower, upper = _floats(self.contract, f"command_range_{name}", 2)
            if result[axis] < lower - 1.0e-6 or result[axis] > upper + 1.0e-6:
                raise ValueError(
                    f"velocity command {name}={result[axis]} is out of range"
                )
        standing = bool(
            abs(float(result[0])) < self.linear_deadband
            and abs(float(result[1])) < self.linear_deadband
            and abs(float(result[2])) < self.yaw_deadband
        )
        return np.zeros(3, dtype=np.float32) if standing else result.copy()

    def build(
        self,
        state: VitaRobotState,
        command: VitaVelocityCommand | Sequence[float],
        last_action: Sequence[float],
    ) -> np.ndarray:
        previous = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if (
            previous.shape != (VITA_VELOCITY_ACTION_DIM,)
            or not np.isfinite(previous).all()
        ):
            raise ValueError("last velocity action must be finite [28]")
        command_array = self._command(command)
        standing = not np.any(command_array)
        if standing:
            phase = self.standing_phase
        else:
            phase = (
                self.policy_step * self.policy_dt / self.gait_cycle
                + self.gait_offsets
            ) % 1.0
        q = state.select_joint_pos(self.contract.action_joint_names)
        qd = state.select_joint_vel(self.contract.action_joint_names)
        frame = np.concatenate(
            (
                state.root_ang_vel_b * self.scales["base_ang_vel"],
                projected_gravity_wxyz(state.root_quat_wxyz)
                * self.scales["projected_gravity"],
                command_array * self.scales["command"],
                (q - self.contract.default_joint_pos)
                * self.scales["joint_pos"],
                qd * self.scales["joint_vel"],
                previous * self.scales["actions"],
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
                self.phase_ratio,
            )
        ).astype(np.float32, copy=False)
        if frame.shape != (self.frame_dim,):
            raise AssertionError(
                f"Vita velocity frame has shape {frame.shape}"
            )
        if not self.history:
            self.history.extend(
                frame.copy() for _ in range(self.history_length)
            )
        else:
            self.history.append(frame)
        observation = np.clip(
            np.concatenate(tuple(self.history)),
            -self.clip_observations,
            self.clip_observations,
        ).astype(self.contract.observation_dtype, copy=False)
        if observation.shape != (VITA_VELOCITY_OBSERVATION_DIM,):
            raise AssertionError("Vita velocity observation is not 990-D")
        self.policy_step += 1
        return observation.reshape(1, -1)


__all__ = [
    "VitaVelocityCommand",
    "VitaVelocityObservationBuilder",
    "projected_gravity_wxyz",
]
