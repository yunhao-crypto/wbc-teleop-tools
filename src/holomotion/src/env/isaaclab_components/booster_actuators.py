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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Booster actuator models aligned with the official K1 training stack."""

from __future__ import annotations

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with Booster torque-speed clipping."""

    cfg: "BoosterDelayedPDActuatorCfg"

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.knee_point_velocity = self._parse_joint_parameter(
            cfg.knee_point_velocity, self.velocity_limit
        )
        self.knee_point_velocity = torch.clamp(
            self.knee_point_velocity, min=0.0
        )
        self.knee_point_velocity = torch.minimum(
            self.knee_point_velocity, self.velocity_limit
        )
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._torque_speed_denom = (
            self.velocity_limit - self.knee_point_velocity
        ).clamp(min=1.0e-6)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_vel_abs = self._joint_vel.abs()
        max_effort = (
            self.effort_limit
            * (self.velocity_limit - joint_vel_abs)
            / self._torque_speed_denom
        )
        max_effort = max_effort.clamp(min=0.0)
        max_effort = torch.minimum(max_effort, self.effort_limit)
        max_effort = torch.where(
            ~torch.isfinite(self.velocity_limit),
            self.effort_limit,
            max_effort,
        )
        max_effort = torch.where(
            self.velocity_limit <= 0.0,
            torch.zeros_like(max_effort),
            max_effort,
        )
        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`BoosterDelayedPDActuator`."""

    class_type: type = BoosterDelayedPDActuator
    knee_point_velocity: dict[str, float] | float | None = None
    """Velocity where available torque starts decreasing toward zero."""
