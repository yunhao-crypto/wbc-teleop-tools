"""Fail-closed command line for the Vita02A S100 deployment."""

from __future__ import annotations
import argparse
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any, Sequence

from deployment.common.model_contract import DeploymentScope, load_model_bundle
from deployment.common.xbox_joystick import LocalXboxController
from deployment.vita_s100_aorta_30dof.aorta_adapter import (
    load_json,
    suspended_validation_alignment_quaternion,
)
from deployment.vita_s100_aorta_30dof.command_guard import (
    command_envelope_from_contract,
)
from deployment.vita_s100_aorta_30dof.config import load_vita_config
from deployment.vita_s100_aorta_30dof.diagnostic_monitor import (
    run_diagnostic_monitor,
)

DEFAULT_CONFIG = Path(__file__).with_name("config") / "vita02a_s100.yaml"


def _check(args: argparse.Namespace) -> int:
    if os.environ.get("HOLOMOTION_REAL_ROBOT"):
        raise RuntimeError("real-robot authorization leaked into check mode")
    config = load_vita_config(args.config)
    bundle = load_model_bundle(args.bundle)
    from deployment.vita_s100_aorta_30dof.runtime import VitaDualPolicyRuntime

    runtime = VitaDualPolicyRuntime.from_bundle(
        bundle,
        require_trained=True,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
            "CPUExecutionProvider",
        ],
        require_cuda=True,
        velocity_providers=["CPUExecutionProvider"],
        velocity_require_cuda=False,
    )
    contracts = runtime.contracts
    envelope = command_envelope_from_contract(
        load_json(config.interface_contract)
    )
    for mode in ("velocity", "motion"):
        runtime.command_envelope(envelope, mode)
    warmup = runtime.warmup()
    alignment = load_json(config.imu_alignment_contract)
    active_approved = bool(
        alignment.get("scope") == "robot-active-approved"
        and isinstance(alignment.get("approval"), dict)
        and alignment["approval"].get("active_control")
    )
    release_approved = bundle.deployment_scope is DeploymentScope.GROUND
    gate_reasons = []
    if not active_approved:
        gate_reasons.append("imu_alignment_not_active_approved")
    if not release_approved:
        gate_reasons.append("bundle_scope_not_ground")
    suspended_ready = False
    if bundle.deployment_scope is DeploymentScope.SUSPENDED_VALIDATION:
        suspended_validation_alignment_quaternion(
            config.imu_alignment_contract, robot_serial=config.robot_serial
        )
        suspended_ready = True
    receipt = {
        "schema": "holomotion_vita_no_action_check_v1",
        "no_robot_action": True,
        "robot": bundle.robot,
        "policy_hz": bundle.policy_hz,
        "motion_observation": list(contracts.motion.observation_shape),
        "motion_action": list(contracts.motion.action_shape),
        "velocity_observation": list(contracts.velocity.observation_shape),
        "velocity_action": list(contracts.velocity.action_shape),
        "velocity_lateral_supported": config.lateral_command_supported,
        "runtime_warmup": warmup,
        "deployment_scope": bundle.deployment_scope.value,
        "active_imu_alignment_approved": active_approved,
        "active_gate": "pass" if not gate_reasons else "blocked",
        "active_gate_reasons": gate_reasons,
        "suspended_validation_contract_ready": suspended_ready,
    }
    print(json.dumps(receipt, sort_keys=True))
    print("Vita check PASSED. No robot action was sent.")
    if gate_reasons:
        print("Ground release remains blocked: " + ", ".join(gate_reasons))
    if suspended_ready:
        print("Suspended validation contract ready; START remains manual.")
    return 0


def _controller_test(args: argparse.Namespace) -> int:
    if os.environ.get("HOLOMOTION_REAL_ROBOT"):
        raise RuntimeError(
            "real-robot authorization leaked into controller-test"
        )
    config = load_vita_config(args.config)
    stop_requested = False

    def stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old = signal.signal(signal.SIGINT, stop)
    try:
        with LocalXboxController(config.joystick) as controller:
            print("Controller test running; press Ctrl-C to finish.")
            while not stop_requested:
                sample = controller.sample(timeout=0.1)
                print(
                    json.dumps(
                        {
                            "sequence": sample.sequence,
                            "buttons": sorted(
                                button.value for button in sample.buttons
                            ),
                            "axes": sample.axes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        signal.signal(signal.SIGINT, old)
    return 0


def _diagnostic(args: argparse.Namespace) -> int:
    if os.environ.get("HOLOMOTION_REAL_ROBOT"):
        raise RuntimeError(
            "real-robot authorization leaked into diagnostic monitor"
        )
    output = run_diagnostic_monitor(
        load_vita_config(args.config), output_root=args.output
    )
    print(f"Vita read-only recording: {output}")
    return 0


def _teleop(args: argparse.Namespace) -> int:
    if os.environ.get("HOLOMOTION_REAL_ROBOT") != "1":
        raise RuntimeError(
            "teleop is available only through the installed launcher"
        )
    config = load_vita_config(args.config)
    bundle = load_model_bundle(args.bundle)
    suspended_validation = bool(getattr(args, "suspended_validation", False))
    required_scope = (
        DeploymentScope.SUSPENDED_VALIDATION
        if suspended_validation
        else DeploymentScope.GROUND
    )
    if bundle.deployment_scope is not required_scope:
        raise RuntimeError(
            "suspended validation requires a suspended-validation bundle"
            if suspended_validation
            else "real teleop requires a ground-approved immutable bundle"
        )
    from deployment.vita_s100_aorta_30dof.real_teleop import run_real_teleop
    from deployment.vita_s100_aorta_30dof.runtime import VitaDualPolicyRuntime

    runtime = VitaDualPolicyRuntime.from_bundle(
        bundle,
        require_trained=True,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
            "CPUExecutionProvider",
        ],
        require_cuda=True,
        velocity_providers=["CPUExecutionProvider"],
        velocity_require_cuda=False,
    )
    run_real_teleop(
        config=config,
        runtime=runtime,
        suspended_validation=suspended_validation,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="offline model/contract check")
    check.add_argument("--bundle", type=Path, required=True)
    check.set_defaults(handler=_check)
    controller = commands.add_parser(
        "controller-test", help="read direct USB controller until Ctrl-C"
    )
    controller.set_defaults(handler=_controller_test)
    diagnostic = commands.add_parser(
        "diagnostic-monitor", help="record LowState read-only until Ctrl-C"
    )
    diagnostic.add_argument("--output", type=Path)
    diagnostic.set_defaults(handler=_diagnostic)
    teleop = commands.add_parser(
        "teleop",
        help="run direct local Pico teleoperation until operator exit",
    )
    teleop.add_argument("--bundle", type=Path, required=True)
    teleop.add_argument(
        "--suspended-validation",
        action="store_true",
        help="fixture-only validation of a suspended-validation bundle",
    )
    teleop.set_defaults(handler=_teleop)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
