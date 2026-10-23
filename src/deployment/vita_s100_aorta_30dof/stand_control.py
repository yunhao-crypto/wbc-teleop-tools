"""Action-sink-free START preparation followed by zero-command Velocity."""

from __future__ import annotations
from typing import Any

import numpy as np

from deployment.common.policy_runtime import PolicyStep


class VitaStandEntry:
    """Interpolate position only, then run Velocity without another blend."""

    def __init__(
        self,
        initial: PolicyStep,
        target: PolicyStep,
        *,
        started: float,
        prepare_seconds: float,
    ) -> None:
        if (
            not np.isfinite([started, prepare_seconds]).all()
            or prepare_seconds <= 0
        ):
            raise ValueError("stand timing must be finite and positive")
        if initial.joint_names != target.joint_names:
            raise ValueError("stand joint orders differ")
        self.initial = initial
        self.target = target
        self.started = started
        self.prepare_seconds = prepare_seconds
        self.policy_started: float | None = None
        self.complete = False
        self.phase = "prepare"

    def step(self, runtime: Any, state: Any, *, now: float) -> PolicyStep:
        if self.policy_started is None:
            alpha = np.clip(
                (now - self.started) / self.prepare_seconds, 0.0, 1.0
            )
            if alpha < 1.0:
                return PolicyStep(
                    mode="stand",
                    joint_names=self.target.joint_names,
                    raw_action=self.target.raw_action,
                    target_joint_pos=self.initial.target_joint_pos
                    * (1.0 - alpha)
                    + self.target.target_joint_pos * alpha,
                    joint_stiffness=self.target.joint_stiffness.copy(),
                    joint_damping=self.target.joint_damping.copy(),
                )
            runtime.reset()
            self.policy_started = now
            self.complete = True
            self.phase = "zero_velocity"
        return runtime.step_velocity(state, (0.0, 0.0, 0.0))
