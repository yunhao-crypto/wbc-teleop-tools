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

"""Fail-closed checks for the shared cross-embodiment training contract."""

from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping

from omegaconf import OmegaConf

PROFILE_NAME = "holomotion_unified_200hz_v1"
PHYSICS_HZ = 200
POLICY_HZ = 50
CONTROL_DECIMATION = 4
VELOCITY_CONTEXT_LENGTH = 8


VELOCITY_REWARD_WEIGHTS = {
    "stand_still_action_rate": -1.0,
    "feet_contact_without_cmd": 1.0,
    "track_stand_still_exp": 5.0,
    "track_lin_vel_xy_heading_aligned_frame_exp": 3.0,
    "track_ang_vel_z_heading_aligned_frame_exp": 3.0,
    "is_alive": 0.15,
    "lin_vel_z_l2": -1.0,
    "ang_vel_xy_l2": -5.0e-2,
    "joint_acc_l2": -1.0e-6,
    "action_rate_l2": -1.0e-1,
    "joint_pos_limits": -5.0,
    "feet_air_time_v4": 1.0,
    "fly": -1.0,
    "feet_too_near": -10.0,
    "joint_deviation_l1_arms": -0.3,
    "joint_deviation_l1_legs_yaw": -0.15,
    "joint_deviation_l1_legs": -0.02,
    "flat_orientation_l2": -5.0,
    "base_height_l2": -10.0,
    "feet_slide": -1.0,
    "undesired_contacts": -1.0,
    "torso_xy_ang_vel_l2_penalty": -1.0,
    "torso_upright_l2_penalty": -1.0,
}

VELOCITY_OBSERVATION_TERMS = [
    "actor_velocity_command",
    "actor_projected_gravity",
    "actor_rel_robot_root_ang_vel",
    "actor_dof_pos",
    "actor_dof_vel",
    "actor_last_action",
    "critic_velocity_command",
    "critic_rel_robot_root_lin_vel",
    "critic_rel_robot_root_ang_vel",
    "critic_root_rel_robot_bodylink_pos_flat",
    "critic_root_rel_robot_bodylink_rot_mat_flat",
    "critic_dof_pos",
    "critic_dof_vel",
    "critic_last_action",
]

VELOCITY_PPO_FIELDS = {
    "num_steps_per_env": 32,
    "num_learning_epochs": 3,
    "num_mini_batches": 4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "value_loss_coef": 1.0,
    "entropy_coef": 5.0e-3,
    "max_grad_norm": 1.0,
    "init_noise_std": 1.0,
    "optimizer_type": "AdamW",
    "schedule": "adaptive",
    "actor_learning_rate": 3.0e-4,
    "critic_learning_rate": 5.0e-4,
}

VELOCITY_COMMAND_FIELDS = {
    "resampling_time_range": [3, 10.0],
    "rel_standing_envs": 0.20,
    "rel_yaw_envs": 0.30,
    "heading_command": False,
    "heading_control_stiffness": 0.5,
    "ranges": {
        "lin_vel_x": [-0.6, 1.0],
        "lin_vel_y": [-0.5, 0.5],
        "ang_vel_z": [-1.0, 1.0],
        "heading": [-3.14, 3.14],
    },
}


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _require_equal(name: str, actual, expected) -> None:
    actual_plain = _plain(actual)
    expected_plain = _plain(expected)
    if actual_plain != expected_plain:
        raise ValueError(
            f"Unified HoloMotion contract mismatch for {name}: "
            f"actual={actual_plain!r}, expected={expected_plain!r}"
        )


def _atomic_observation_names(config) -> list[str]:
    atomic = config.obs.obs_groups.unified.atomic_obs_list
    names: list[str] = []
    for item in atomic:
        item_plain = _plain(item)
        if not isinstance(item_plain, Mapping) or len(item_plain) != 1:
            raise ValueError(
                "Each unified observation entry must contain exactly one term"
            )
        names.append(str(next(iter(item_plain))))
    return names


def _module_signature(module) -> dict:
    return {
        "type": str(module.type),
        "hidden_norm": str(module.hidden_norm),
        "hidden_dims": [
            int(value) for value in module.layer_config.hidden_dims
        ],
        "activation": str(module.layer_config.activation),
        "obs_norm": {
            "enabled": bool(module.obs_norm.enabled),
            "update_method": str(module.obs_norm.update_method),
            "ema_momentum": float(module.obs_norm.ema_momentum),
            "enable_clipping": bool(module.obs_norm.enable_clipping),
            "clip_range": float(module.obs_norm.clip_range),
        },
    }


def _expected_module_signature() -> dict:
    return {
        "type": "MLP",
        "hidden_norm": "rmsnorm",
        "hidden_dims": [512, 512, 512],
        "activation": "SiLU",
        "obs_norm": {
            "enabled": True,
            "update_method": "ema",
            "ema_momentum": 1.0e-4,
            "enable_clipping": True,
            "clip_range": 10.0,
        },
    }


def _delay_receipt(config, sim_freq: int) -> dict:
    domain_delay = _plain(config.domain_rand.get("action_delay", {})) or {}
    domain_enabled = bool(domain_delay.get("enabled", False))
    domain_min = int(domain_delay.get("min_delay", 0)) if domain_enabled else 0
    domain_max = int(domain_delay.get("max_delay", 0)) if domain_enabled else 0

    embedded: dict[str, dict[str, float | int]] = {}
    actuator_cfg = _plain(config.robot.actuators)
    for group_name, group in actuator_cfg.items():
        if not isinstance(group, Mapping):
            continue
        if "min_delay" not in group and "max_delay" not in group:
            continue
        min_steps = int(group.get("min_delay", 0))
        max_steps = int(group.get("max_delay", 0))
        if min_steps < 0 or max_steps < min_steps:
            raise ValueError(
                f"Invalid actuator delay range for {group_name}: "
                f"[{min_steps}, {max_steps}]"
            )
        embedded[str(group_name)] = {
            "min_steps": min_steps,
            "max_steps": max_steps,
            "min_ms": 1000.0 * min_steps / sim_freq,
            "max_ms": 1000.0 * max_steps / sim_freq,
        }

    embedded_enabled = any(
        int(values["max_steps"]) > 0 for values in embedded.values()
    )
    if domain_enabled and domain_max > 0 and embedded_enabled:
        raise ValueError(
            "Unified HoloMotion forbids applying environment and actuator "
            "delay simultaneously"
        )
    return {
        "domain": {
            "enabled": domain_enabled,
            "min_steps": domain_min,
            "max_steps": domain_max,
            "min_ms": 1000.0 * domain_min / sim_freq,
            "max_ms": 1000.0 * domain_max / sim_freq,
        },
        "actuator_groups": embedded,
    }


def _velocity_algorithm_payload(config) -> dict:
    reward_weights = {
        str(name): float(value.weight)
        for name, value in config.rewards.items()
    }
    ppo = {
        name: _plain(config.algo.config.get(name))
        for name in VELOCITY_PPO_FIELDS
    }
    commands = config.env.config.commands.base_velocity.params
    command_payload = {
        name: _plain(commands.get(name)) for name in VELOCITY_COMMAND_FIELDS
    }
    return {
        "clock": {
            "physics_hz": int(config.env.config.simulation.sim_freq),
            "control_decimation": int(
                config.env.config.simulation.control_decimation
            ),
            "policy_hz": float(
                config.env.config.simulation.sim_freq
                / config.env.config.simulation.control_decimation
            ),
        },
        "observation": {
            "context_length": int(config.obs.context_length),
            "target_fps": int(config.obs.target_fps),
            "terms": _atomic_observation_names(config),
        },
        "modules": {
            "actor": _module_signature(config.modules.actor),
            "critic": _module_signature(config.modules.critic),
        },
        "ppo": ppo,
        "reward_weights": reward_weights,
        "command": command_payload,
    }


def _validated_velocity_overrides(contract) -> dict:
    overrides = _plain(contract.get("robot_specific_overrides", {})) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("robot_specific_overrides must be a mapping")

    allowed = {
        "ppo": {"init_noise_std"},
        "reward_weights": {"is_terminated"},
    }
    result = {"ppo": {}, "reward_weights": {}}
    unknown_sections = set(overrides) - set(allowed)
    if unknown_sections:
        raise ValueError(
            "Unsupported robot-specific override sections: "
            f"{sorted(unknown_sections)}"
        )
    for section, values in overrides.items():
        if not isinstance(values, Mapping):
            raise ValueError(
                f"robot_specific_overrides.{section} must be a mapping"
            )
        unknown_fields = set(values) - allowed[section]
        if unknown_fields:
            raise ValueError(
                f"Unsupported robot-specific {section} overrides: "
                f"{sorted(unknown_fields)}"
            )
        result[section] = dict(values)
    return result


def _validate_velocity_algorithm(payload: dict, contract) -> dict:
    _require_equal(
        "velocity context_length",
        payload["observation"]["context_length"],
        VELOCITY_CONTEXT_LENGTH,
    )
    _require_equal(
        "velocity target_fps", payload["observation"]["target_fps"], POLICY_HZ
    )
    _require_equal(
        "velocity observation terms",
        payload["observation"]["terms"],
        VELOCITY_OBSERVATION_TERMS,
    )
    expected_module = _expected_module_signature()
    _require_equal(
        "velocity Actor module", payload["modules"]["actor"], expected_module
    )
    _require_equal(
        "velocity Critic module", payload["modules"]["critic"], expected_module
    )
    overrides = _validated_velocity_overrides(contract)
    expected_ppo = dict(VELOCITY_PPO_FIELDS)
    expected_ppo.update(overrides["ppo"])
    _require_equal("velocity PPO", payload["ppo"], expected_ppo)
    expected_rewards = dict(VELOCITY_REWARD_WEIGHTS)
    expected_rewards.update(overrides["reward_weights"])
    _require_equal(
        "velocity reward weights",
        payload["reward_weights"],
        expected_rewards,
    )
    _require_equal(
        "velocity command distribution",
        payload["command"],
        VELOCITY_COMMAND_FIELDS,
    )
    return overrides


def validate_holomotion_training_contract(config) -> dict | None:
    """Validate and describe an opted-in shared HoloMotion configuration."""

    contract = config.get("holomotion_training_contract", None)
    if contract is None or not bool(contract.get("required", False)):
        return None

    profile = str(contract.get("profile", ""))
    _require_equal("profile", profile, PROFILE_NAME)
    task = str(contract.get("task", ""))
    if task not in {"velocity_tracking", "motion_tracking"}:
        raise ValueError(f"Unsupported unified HoloMotion task: {task!r}")

    simulation = config.env.config.simulation
    sim_freq = int(simulation.sim_freq)
    decimation = int(simulation.control_decimation)
    _require_equal("physics frequency", sim_freq, PHYSICS_HZ)
    _require_equal("control decimation", decimation, CONTROL_DECIMATION)
    _require_equal("policy frequency", sim_freq / decimation, POLICY_HZ)

    algorithm_payload: dict
    if task == "velocity_tracking":
        algorithm_payload = _velocity_algorithm_payload(config)
        robot_specific_overrides = _validate_velocity_algorithm(
            algorithm_payload, contract
        )
    else:
        target_fps = int(config.obs.target_fps)
        _require_equal("motion target_fps", target_fps, POLICY_HZ)
        algorithm_payload = {
            "clock": {
                "physics_hz": sim_freq,
                "control_decimation": decimation,
                "policy_hz": sim_freq / decimation,
                "target_fps": target_fps,
            },
            "actor_body_mode": str(contract.get("actor_body_mode", "")),
            "reset_profile": str(contract.get("reset_profile", "")),
        }

    policy_action_dim = int(contract.policy_action_dim)
    controller_target_dim = int(contract.controller_target_dim)
    actual_action_dim = int(config.robot.actions_dim)
    _require_equal(
        "policy action dimension", actual_action_dim, policy_action_dim
    )
    if controller_target_dim != len(config.robot.dof_names):
        raise ValueError(
            "controller_target_dim must match robot.dof_names: "
            f"{controller_target_dim} != {len(config.robot.dof_names)}"
        )

    payload_json = json.dumps(
        algorithm_payload, sort_keys=True, separators=(",", ":")
    )
    return {
        "schema": "holomotion_training_contract_v1",
        "profile": profile,
        "task": task,
        "embodiment": str(contract.embodiment),
        "policy_action_dim": policy_action_dim,
        "controller_target_dim": controller_target_dim,
        "robot_specific_overrides": (
            robot_specific_overrides if task == "velocity_tracking" else {}
        ),
        "algorithm_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
        "algorithm": algorithm_payload,
        "actuator_delay": _delay_receipt(config, sim_freq),
    }
