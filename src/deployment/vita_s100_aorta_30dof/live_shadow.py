"""Pure helpers for read-only live Vita policy shadow evaluation."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from deployment.vita_s100_aorta_30dof.shadow_adapter import (
    alignment_quaternion_from_contract,
    quaternion_inverse_wxyz,
    quaternion_multiply_wxyz,
    rotate_vector_wxyz,
)
from deployment.vita_s100_aorta_30dof.state_adapter import (
    IMU_LOCATION_HIP,
    IMU_STATUS_OK,
    VitaRawLowState,
)


@dataclass(frozen=True)
class VitaLivePolicyState:
    """One validated robot state expressed in the policy coordinate order."""

    state_id: int
    source_timestamp_us: int
    received_monotonic_ns: int
    robot_root_quat_wxyz: np.ndarray
    robot_root_angvel_local: np.ndarray
    robot_dof_pos: np.ndarray
    robot_dof_vel: np.ndarray


class FixedRateSourceSelector:
    """Select the first source frame at or after each fixed-rate deadline."""

    def __init__(self, rate_hz: float) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive and finite")
        self.period_us = int(round(1.0e6 / rate_hz))
        if self.period_us <= 0:
            raise ValueError("rate_hz produces an invalid period")
        self.next_timestamp_us: int | None = None
        self.last_timestamp_us: int | None = None
        self.skipped_deadlines = 0
        self.reordered_frames = 0

    def accept(self, source_timestamp_us: int) -> bool:
        """Return whether this source timestamp should drive a policy frame."""

        timestamp = int(source_timestamp_us)
        if timestamp <= 0:
            raise ValueError("source timestamp must be positive")
        if (
            self.last_timestamp_us is not None
            and timestamp <= self.last_timestamp_us
        ):
            self.reordered_frames += 1
            return False
        self.last_timestamp_us = timestamp
        if self.next_timestamp_us is None:
            self.next_timestamp_us = timestamp + self.period_us
            return True
        if timestamp < self.next_timestamp_us:
            return False
        elapsed_periods = (
            timestamp - self.next_timestamp_us
        ) // self.period_us
        self.skipped_deadlines += int(elapsed_periods)
        self.next_timestamp_us += (int(elapsed_periods) + 1) * self.period_us
        return True


def live_alignment_quaternion(
    alignment: Mapping[str, Any],
    *,
    robot_serial: str | None = None,
) -> np.ndarray:
    """Validate a contract specifically approved for live read-only shadow."""

    approval = alignment.get("approval")
    if not isinstance(approval, dict) or not approval.get(
        "live_readonly_shadow"
    ):
        raise ValueError(
            "robot alignment is not approved for live read-only shadow"
        )
    configured_serial = alignment.get("robot_serial")
    if robot_serial is not None and configured_serial != robot_serial:
        raise ValueError(
            "alignment robot_serial does not match the expected robot serial"
        )
    return alignment_quaternion_from_contract(alignment, {})


def policy_state_from_lowstate(
    state: VitaRawLowState,
    *,
    base_from_hip_quaternion_wxyz: np.ndarray,
) -> VitaLivePolicyState:
    """Convert one decoded live LowState into motion-policy inputs."""

    if not state.attitude_valid:
        raise ValueError("LowState attitude is invalid")
    hip = state.require_imu(
        IMU_LOCATION_HIP,
        allowed_statuses={IMU_STATUS_OK},
    )
    base_from_hip = np.asarray(base_from_hip_quaternion_wxyz, dtype=np.float32)
    hip_from_base = quaternion_inverse_wxyz(base_from_hip)
    root_quaternion = quaternion_multiply_wxyz(
        hip.normalized_quaternion_wxyz(), hip_from_base
    )
    root_angular_velocity = rotate_vector_wxyz(
        base_from_hip, hip.gyroscope_rad_s
    )
    dof_position, dof_velocity = state.observation_joint_state()
    return VitaLivePolicyState(
        state_id=state.state_id,
        source_timestamp_us=state.ts_state_real_us,
        received_monotonic_ns=state.received_monotonic_ns,
        robot_root_quat_wxyz=root_quaternion,
        robot_root_angvel_local=root_angular_velocity,
        robot_dof_pos=dof_position,
        robot_dof_vel=dof_velocity,
    )


__all__ = [
    "FixedRateSourceSelector",
    "VitaLivePolicyState",
    "live_alignment_quaternion",
    "policy_state_from_lowstate",
]
