"""Pure 42-slot command conditioning shared by active and inert output paths."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from deployment.common.policy_runtime import PolicyStep
from deployment.vita_s100_aorta_30dof.command_guard import CommandEnvelope
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_ACTION_TO_COMMAND_SLOTS, VITA_COMMAND_SLOT_COUNT
from deployment.vita_s100_aorta_30dof.robot_adapter import VITA_CONTROLLER_JOINT_NAMES

class InitialPoseError(RuntimeError):
    """The measured pose cannot initialize a bounded position command."""


def validate_gains(kp, kd, envelope: CommandEnvelope) -> None:
    for label, values, maximum in (
        ("KP", np.asarray(kp), envelope.max_kp),
        ("KD", np.asarray(kd), envelope.max_kd),
    ):
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > maximum)
        ):
            raise ValueError(
                f"{label} outside robot supported range [0, {maximum}]; "
                "gains were not clipped"
            )


class VitaCommandAdapter:
    """Map named policy steps to 42 SDK slots and apply SDK conditioning."""

    def __init__(self, envelope: CommandEnvelope) -> None:
        self.envelope = envelope
        self.policy_slots = np.asarray(
            VITA_ACTION_TO_COMMAND_SLOTS, dtype=np.int64
        )
        self.previous_target: np.ndarray | None = None
        self.position_clip_count = 0
        self.slew_clip_count = 0
        self.gain_clip_count = 0

    def arm(self, current_q: Sequence[float]) -> None:
        value = np.asarray(current_q, dtype=np.float64).reshape(-1)
        if (
            value.shape != (VITA_COMMAND_SLOT_COUNT,)
            or not np.isfinite(value).all()
        ):
            raise ValueError("current_q must be finite [42]")
        if np.any(value < self.envelope.lower_rad) or np.any(
            value > self.envelope.upper_rad
        ):
            bad = np.flatnonzero(
                (value < self.envelope.lower_rad)
                | (value > self.envelope.upper_rad)
            )
            details = "; ".join(
                f"{self.envelope.joint_names[i]}={value[i]:.5f} outside "
                f"[{self.envelope.lower_rad[i]:.5f}, "
                f"{self.envelope.upper_rad[i]:.5f}]"
                for i in bad
            )
            raise InitialPoseError(details)
        self.previous_target = value.copy()

    def condition(
        self,
        step: PolicyStep,
        *,
        hold_q: Sequence[float],
        period_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
        if step.joint_names != VITA_CONTROLLER_JOINT_NAMES:
            raise ValueError("policy step joint order is not Vita 30-DoF")
        hold = np.asarray(hold_q, dtype=np.float64).reshape(-1)
        if hold.shape != (VITA_COMMAND_SLOT_COUNT,):
            raise ValueError("hold_q must contain 42 SDK slots")
        # Both policies leave fingers open, matching the official profile.
        target = np.zeros(VITA_COMMAND_SLOT_COUNT, dtype=np.float64)
        kp = np.zeros(VITA_COMMAND_SLOT_COUNT, dtype=np.float64)
        kd = np.zeros(VITA_COMMAND_SLOT_COUNT, dtype=np.float64)
        target[self.policy_slots] = step.target_joint_pos
        kp[self.policy_slots] = step.joint_stiffness
        kd[self.policy_slots] = step.joint_damping
        safe_lower = self.envelope.lower_rad.copy()
        safe_upper = self.envelope.upper_rad.copy()
        # Only round inward when float32 serialization would cross a bound.
        lower32 = safe_lower.astype(np.float32)
        upper32 = safe_upper.astype(np.float32)
        safe_lower = np.where(
            lower32.astype(np.float64) < safe_lower,
            np.nextafter(lower32, np.float32(np.inf)),
            lower32,
        ).astype(np.float64)
        safe_upper = np.where(
            upper32.astype(np.float64) > safe_upper,
            np.nextafter(upper32, np.float32(-np.inf)),
            upper32,
        ).astype(np.float64)
        if np.any(
            safe_lower[self.policy_slots] >= safe_upper[self.policy_slots]
        ):
            raise RuntimeError("SDK position envelope is too narrow")
        clipped = np.clip(target, safe_lower, safe_upper)
        position_modification = float(np.max(np.abs(clipped - target)))
        if position_modification > 0.0:
            self.position_clip_count += 1
        target = clipped
        if self.previous_target is None:
            raise RuntimeError("command adapter is not armed")
        if (
            not self.envelope.min_period_s
            <= period_s
            <= self.envelope.max_period_s
        ):
            raise RuntimeError("control period is outside the SDK envelope")
        validate_gains(kp, kd, self.envelope)
        slew_modification = 0.0
        gain_modification = 0.0
        self.previous_target = target.copy()
        return (
            target,
            kp,
            kd,
            {
                "position_clip_max_rad": position_modification,
                "slew_clip_max_rad": slew_modification,
                "position_clip_cycles": self.position_clip_count,
                "slew_clip_cycles": self.slew_clip_count,
                "gain_clip_max": gain_modification,
                "gain_clip_cycles": self.gain_clip_count,
            },
        )
