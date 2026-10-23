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

"""Vita02A delayed PD actuator with a coupled two-motor ankle limit.

The URDF exposes serial ankle-pitch and ankle-roll coordinates, while each
physical foot is driven by two motors through a parallel linkage.  This model
keeps all controlled joints in one actuator group so Isaac Lab samples one
delay per environment for the complete target packet.  It then projects each
virtual ankle torque into motor space, clips both motors, and maps the result
back into virtual joint space.
"""

from __future__ import annotations
from collections.abc import Sequence

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from holomotion.src.utils.vita02a_parallel_ankle_math import (
    VITA02A_LX,
    VITA02A_LY,
    VITA02A_Z_BOT,
    VITA02A_Z_TOP,
    project_vita02a_ankle_torque,
)


class Vita02AParallelAnkleActuator(DelayedPDActuator):
    """One delayed-PD group for all joints plus coupled ankle motor caps."""

    cfg: "Vita02AParallelAnkleActuatorCfg"

    def __init__(
        self, cfg: "Vita02AParallelAnkleActuatorCfg", *args, **kwargs
    ):
        super().__init__(cfg, *args, **kwargs)
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError("Vita02A actuator joint names must be unique.")
        joint_index = {
            name: index for index, name in enumerate(self.joint_names)
        }
        try:
            self._ankle_pairs = tuple(
                (
                    joint_index[f"{side}_ankle_pitch_joint"],
                    joint_index[f"{side}_ankle_roll_joint"],
                )
                for side in ("left", "right")
            )
        except KeyError as error:
            raise ValueError(
                "Vita02A parallel ankles require left/right pitch and roll "
                "joints in the same actuator group."
            ) from error
        self._parallel_ankle_torque_clip_count = torch.zeros(
            (), dtype=torch.long, device=self._device
        )
        self.parallel_ankle_invalid_state_count = torch.zeros(
            (), dtype=torch.long, device=self._device
        )

    @property
    def parallel_ankle_torque_clip_count(self) -> torch.Tensor:
        """Running number of environment-foot samples changed by projection."""

        return self._parallel_ankle_torque_clip_count

    def reset(self, env_ids: Sequence[int]) -> None:
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            self._parallel_ankle_torque_clip_count.zero_()

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        effort = control_action.joint_efforts
        for pitch_index, roll_index in self._ankle_pairs:
            pitch_input = effort[:, pitch_index].clone()
            roll_input = effort[:, roll_index].clone()
            pitch, roll, valid = project_vita02a_ankle_torque(
                joint_pos[:, pitch_index],
                joint_pos[:, roll_index],
                pitch_input,
                roll_input,
                tau_motor_max=self.cfg.tau_motor_max,
                lx=self.cfg.lx,
                ly=self.cfg.ly,
                z_top=self.cfg.z_top,
                z_bot=self.cfg.z_bot,
                return_valid=True,
            )
            self.parallel_ankle_invalid_state_count += torch.count_nonzero(
                ~valid
            )
            changed = (torch.abs(pitch - pitch_input) > 1.0e-6) | (
                torch.abs(roll - roll_input) > 1.0e-6
            )
            self._parallel_ankle_torque_clip_count += torch.count_nonzero(
                changed
            )
            effort[:, pitch_index] = pitch
            effort[:, roll_index] = roll
        control_action.joint_efforts = effort
        self.applied_effort = effort
        return control_action


@configclass
class Vita02AParallelAnkleActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`Vita02AParallelAnkleActuator`."""

    class_type: type = Vita02AParallelAnkleActuator
    tau_motor_max: float = 50.0
    qd_motor_max: float = 7.85
    lx: float = VITA02A_LX
    ly: float = VITA02A_LY
    z_top: float = VITA02A_Z_TOP
    z_bot: float = VITA02A_Z_BOT
