"""Fail-closed handoff from Unitree ai_sport to the G1 LowCmd controller."""

from __future__ import annotations
import argparse
import importlib
import os
import time
from typing import Any

CONFIRMATION = "I_CONFIRM_G1_LOW_LEVEL_TAKEOVER"
AI_SPORT_SERVICE = "ai_sport"
ROBOT_STATE_CLIENT_MODULES = (
    "unitree_sdk2py.g1.robot_state.robot_state_client",
    "unitree_sdk2py.go2.robot_state.robot_state_client",
    "unitree_sdk2py.b2.robot_state.robot_state_client",
)


def load_robot_state_client_type() -> type:
    errors = []
    for module_name in ROBOT_STATE_CLIENT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        return module.RobotStateClient
    raise RuntimeError(
        "Unitree RobotStateClient is unavailable: " + "; ".join(errors)
    )


def _return_code(result: Any) -> int:
    if isinstance(result, tuple):
        result = result[0]
    return int(result)


def check_ai_sport(client: Any) -> int:
    result = client.ServiceList()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            f"unexpected RobotState ServiceList result: {result!r}"
        )
    code, services = result
    if int(code) != 0 or services is None:
        raise RuntimeError(
            f"RobotState ServiceList failed: code={code!r}, data={services!r}"
        )
    matches = [item for item in services if item.name == AI_SPORT_SERVICE]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {AI_SPORT_SERVICE} service, found {len(matches)}"
        )
    status = int(matches[0].status)
    if status not in (0, 1):
        raise RuntimeError(f"unexpected {AI_SPORT_SERVICE} status: {status}")
    print(f"service={AI_SPORT_SERVICE} status={status}")
    return status


def release_ai_sport(
    client: Any, *, timeout: float, interval: float = 0.25
) -> None:
    if timeout <= 0.0:
        raise ValueError("release timeout must be positive")
    status = check_ai_sport(client)
    deadline = time.monotonic() + timeout
    while status != 0:
        code = _return_code(client.ServiceSwitch(AI_SPORT_SERVICE, False))
        if code != 0:
            raise RuntimeError(f"ai_sport ServiceSwitch(false) failed: {code}")
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out disabling Unitree ai_sport")
        time.sleep(interval)
        status = check_ai_sport(client)
    print("G1 ai_sport is disabled; custom LowCmd may start.")


def create_client(network_interface: str) -> Any:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    client_type = load_robot_state_client_type()
    ChannelFactoryInitialize(0, network_interface)
    client = client_type()
    client.SetTimeout(5.0)
    client.Init()
    return client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or disable the Unitree G1 ai_sport service."
    )
    parser.add_argument(
        "command", choices=("dependencies", "check", "release")
    )
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--confirm", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dependencies":
        load_robot_state_client_type()
        print("G1 takeover dependency check PASSED. No channel was opened.")
        return 0
    if args.command == "release" and (
        os.environ.get("HOLOMOTION_REAL_ROBOT") != "1"
        or args.confirm != CONFIRMATION
    ):
        raise RuntimeError(
            "G1 release is available only through the real launcher"
        )
    client = create_client(args.network_interface)
    if args.command == "check":
        check_ai_sport(client)
        print("G1 takeover check PASSED. No service was changed.")
        return 0
    release_ai_sport(client, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
