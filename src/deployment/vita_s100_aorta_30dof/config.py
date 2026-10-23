"""Configuration ownership for the Vita02A S100 deployment."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from deployment.common.controller_input import VelocityLimits


@dataclass(frozen=True)
class VitaDeploymentConfig:
    robot_serial: str
    aorta_group: str
    aorta_profile: Path
    joystick: Path
    policy_hz: float
    state_max_age: float
    pico_max_age: float
    stand_transition_seconds: float
    aorta_startup_timeout: float
    sdk_rpc_timeout: float
    sdk_lease_duration_seconds: float
    logs: Path
    velocity_limits: VelocityLimits
    lateral_command_supported: bool
    interface_contract: Path
    imu_alignment_contract: Path
    stand_profile: Path
    policy_to_sdk_slots: Path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive(value: Any, label: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def load_vita_config(path: str | Path) -> VitaDeploymentConfig:
    source = Path(path).expanduser().resolve()
    raw = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")), "config"
    )
    robot = _mapping(raw.get("robot"), "robot")
    runtime = _mapping(raw.get("runtime"), "runtime")
    controller = _mapping(raw.get("controller"), "controller")
    velocity = _mapping(controller.get("velocity"), "controller.velocity")
    contracts = _mapping(raw.get("contracts"), "contracts")

    def contract(name: str) -> Path:
        value = Path(str(contracts[name])).expanduser()
        return (
            value.resolve()
            if value.is_absolute()
            else (source.parent / value).resolve()
        )

    config = VitaDeploymentConfig(
        robot_serial=str(robot.get("serial", "")).strip(),
        aorta_group=str(robot.get("aorta_group", "")).strip(),
        aorta_profile=Path(str(robot.get("aorta_profile", ""))).expanduser(),
        joystick=Path(str(robot.get("joystick", ""))).expanduser(),
        policy_hz=_positive(runtime.get("policy_hz"), "runtime.policy_hz"),
        state_max_age=_positive(
            runtime.get("state_max_age"), "runtime.state_max_age"
        ),
        pico_max_age=_positive(
            runtime.get("pico_max_age"), "runtime.pico_max_age"
        ),
        stand_transition_seconds=_positive(
            runtime.get("stand_transition_seconds"),
            "runtime.stand_transition_seconds",
        ),
        aorta_startup_timeout=_positive(
            runtime.get("aorta_startup_timeout"),
            "runtime.aorta_startup_timeout",
        ),
        sdk_rpc_timeout=_positive(
            runtime.get("sdk_rpc_timeout"),
            "runtime.sdk_rpc_timeout",
        ),
        sdk_lease_duration_seconds=_positive(
            runtime.get("sdk_lease_duration_seconds"),
            "runtime.sdk_lease_duration_seconds",
        ),
        logs=Path(str(runtime.get("logs", ""))).expanduser(),
        velocity_limits=VelocityLimits(
            forward=float(velocity["forward"]),
            backward=float(velocity["backward"]),
            lateral=float(velocity["lateral"]),
            yaw=float(velocity["yaw"]),
            deadzone=float(velocity.get("deadzone")),
        ),
        lateral_command_supported=bool(
            controller.get("lateral_command_supported")
        ),
        interface_contract=contract("interface"),
        imu_alignment_contract=contract("imu_alignment"),
        stand_profile=contract("stand_profile"),
        policy_to_sdk_slots=contract("policy_to_sdk_slots"),
    )
    if not config.robot_serial or not config.aorta_group:
        raise ValueError("robot serial and Aorta group are required")
    if abs(config.policy_hz - 50.0) > 1.0e-6:
        raise ValueError("Vita policy_hz must be 50")
    if config.lateral_command_supported:
        raise ValueError(
            "official model_17000 does not support lateral velocity"
        )
    for required in (
        config.interface_contract,
        config.imu_alignment_contract,
        config.stand_profile,
        config.policy_to_sdk_slots,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Vita deployment contract is missing: {required}"
            )
    return config


__all__ = ["VitaDeploymentConfig", "load_vita_config"]
