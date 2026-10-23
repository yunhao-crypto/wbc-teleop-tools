"""Config-driven six-term HoloMotion Velocity input for deployment.

Reuse the shared G1 deployment history builder; do not impose the legacy
Vbot-lab gait clock, command deadband, frame-major history or action clipping.
The training config must be a hash-checked artifact in the model bundle.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

from deployment.unitree_g1_ros2_29dof.src.humanoid_policy.obs_builder import (
    PolicyObsBuilder,
    get_gravity_orientation,
)


class UnifiedVelocityObservationBuilder:
    TERM_FUNCTIONS = {
        "actor_velocity_command": "velocity_command",
        "actor_projected_gravity": "projected_gravity",
        "actor_rel_robot_root_ang_vel": "rel_robot_root_ang_vel",
        "actor_dof_pos": "dof_pos",
        "actor_dof_vel": "dof_vel",
        "actor_last_action": "last_action",
    }

    def __init__(self, contract: Any, training_config: str | Path) -> None:
        self.contract = contract
        cfg = OmegaConf.load(training_config)
        schema = cfg.modules.actor.obs_schema
        if list(schema) != ["flattened_obs"]:
            raise ValueError("unsupported unified velocity actor schema")
        terms = list(schema.flattened_obs.terms)
        if (
            set(terms) != {f"unified/{name}" for name in self.TERM_FUNCTIONS}
            or len(terms) != 6
        ):
            raise ValueError("unsupported unified velocity observation terms")
        by_name = {}
        for entry in cfg.obs.obs_groups.unified.atomic_obs_list:
            for name in entry:
                if name in self.TERM_FUNCTIONS:
                    by_name[name] = OmegaConf.to_container(
                        entry[name], resolve=True
                    )
        widths = {
            name: len(contract.action_joint_names)
            for name in self.TERM_FUNCTIONS
        }
        widths.update(
            actor_velocity_command=4,
            actor_projected_gravity=3,
            actor_rel_robot_root_ang_vel=3,
        )
        atomic = []
        dimension = 0
        for key in terms:
            name = key.split("/")[1]
            term = by_name[name]
            if (
                term.get("func") != self.TERM_FUNCTIONS[name]
                or term.get("clip") is not None
            ):
                raise ValueError(
                    f"unsupported velocity term transform: {name}"
                )
            if name in {"actor_dof_pos", "actor_dof_vel"}:
                if (
                    tuple(term.get("params", {}).get("joint_names", ()))
                    != contract.action_joint_names
                ):
                    raise ValueError(
                        f"velocity {name} joint order disagrees with ONNX"
                    )
            history = int(term.get("history_length", 0))
            if history != int(schema.flattened_obs.seq_len) or history < 1:
                raise ValueError("velocity term/schema history disagreement")
            dimension += widths[name] * history
            atomic.append({name: term})
        if contract.observation_shape != (1, dimension):
            raise ValueError(
                "velocity training observation dimension disagrees with ONNX"
            )
        # This ABI has unbounded raw actions. Never borrow the official policy
        # clip_actions setting or silently add a deployment-only action clip.
        if OmegaConf.select(cfg, "env.config.clip_actions") is not None:
            raise ValueError(
                "unified velocity action clipping requires an explicit adapter"
            )
        physics = OmegaConf.to_container(cfg.robot.joint_physics, resolve=True)
        if (
            physics.get("joint_friction") != "zero"
            or physics.get("passive_damping") != 0.0
        ):
            raise ValueError(
                "velocity config physics disagrees with nominal contract"
            )
        defaults = dict(
            zip(contract.joint_names, contract.default_joint_pos, strict=True)
        )
        self.defaults = np.array(
            [defaults[n] for n in contract.action_joint_names],
            dtype=np.float32,
        )
        self.atomic_obs_list = atomic
        self.terms: dict[str, np.ndarray] = {}
        self.builder = PolicyObsBuilder(
            contract.joint_names,
            contract.default_joint_pos,
            evaluator=self,
            obs_policy_cfg={"atomic_obs_list": atomic},
        )

    def __getattr__(self, name: str) -> Any:
        prefix = "_get_obs_"
        if (
            name.startswith(prefix)
            and name[len(prefix) :] in self.TERM_FUNCTIONS
        ):
            return lambda: self.terms[name[len(prefix) :]]
        raise AttributeError(name)

    def reset(self) -> None:
        self.builder.reset()

    def build(
        self, state: Any, command: Any, last_action: Sequence[float]
    ) -> np.ndarray:
        cmd = np.asarray(
            command.as_array() if hasattr(command, "as_array") else command,
            dtype=np.float32,
        ).reshape(-1)
        previous = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if cmd.shape != (3,) or not np.isfinite(cmd).all():
            raise ValueError(
                "velocity command must be finite [vx, vy, yaw_rate]"
            )
        if (
            previous.shape != (len(self.contract.action_joint_names),)
            or not np.isfinite(previous).all()
        ):
            raise ValueError(
                "last velocity action has invalid shape or values"
            )
        self.terms = {
            "actor_velocity_command": np.concatenate(
                (
                    np.array(
                        [float(np.linalg.norm(cmd) > 0.1)], dtype=np.float32
                    ),
                    cmd,
                )
            ),
            "actor_projected_gravity": get_gravity_orientation(
                state.root_quat_wxyz
            ),
            "actor_rel_robot_root_ang_vel": np.asarray(
                state.root_ang_vel_b, dtype=np.float32
            ),
            "actor_dof_pos": state.select_joint_pos(
                self.contract.action_joint_names
            )
            - self.defaults,
            "actor_dof_vel": state.select_joint_vel(
                self.contract.action_joint_names
            ),
            "actor_last_action": previous,
        }
        out = self.builder.build_policy_obs().reshape(1, -1)
        if not np.isfinite(out).all():
            raise ValueError("non-finite unified velocity observation")
        return out.astype(self.contract.observation_dtype, copy=False)
