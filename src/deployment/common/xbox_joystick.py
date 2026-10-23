"""Read an Xbox-compatible Linux joystick without network relays."""

from __future__ import annotations
import os
import select
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from deployment.common.controller_input import Button, ControllerSample

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT = struct.Struct("<IhBB")

DEFAULT_AXIS_MAP = {
    0: "left_x",
    1: "left_y",
    3: "right_x",
    4: "right_y",
}
DEFAULT_BUTTON_MAP = {
    0: Button.A,
    1: Button.B,
    2: Button.X,
    3: Button.Y,
    4: Button.LB,
    5: Button.RB,
    6: Button.BACK,
    7: Button.START,
}


def normalize_joystick_axis(value: int) -> float:
    """Map the Linux joystick signed-16 range to [-1, 1]."""

    value = max(-32767, min(32767, int(value)))
    return float(value) / 32767.0


@dataclass
class XboxJoystickState:
    """Mutable state assembled from Linux joystick events."""

    deadman_button: Button = Button.RB
    axis_map: Mapping[int, str] = field(
        default_factory=lambda: dict(DEFAULT_AXIS_MAP)
    )
    button_map: Mapping[int, Button] = field(
        default_factory=lambda: dict(DEFAULT_BUTTON_MAP)
    )
    axes: dict[str, float] = field(
        default_factory=lambda: {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
        }
    )
    buttons: set[Button] = field(default_factory=set)

    def apply_event(self, event_type: int, number: int, value: int) -> None:
        """Apply one Linux joystick event."""

        event_type &= ~JS_EVENT_INIT
        if event_type == JS_EVENT_AXIS and number in self.axis_map:
            self.axes[self.axis_map[number]] = normalize_joystick_axis(value)
        elif event_type == JS_EVENT_BUTTON and number in self.button_map:
            button = self.button_map[number]
            if value:
                self.buttons.add(button)
            else:
                self.buttons.discard(button)

    def sample(self, sequence: int) -> ControllerSample:
        """Snapshot the directly attached controller."""

        return ControllerSample(
            sequence=sequence,
            monotonic_time=time.monotonic(),
            buttons=frozenset(self.buttons),
            axes=dict(self.axes),
        )


class LinuxJoystick:
    """Non-blocking reader for the Linux ``/dev/input/js*`` API."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        descriptor = os.open(
            self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC
        )
        self.descriptor = descriptor

    def __enter__(self) -> "LinuxJoystick":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the joystick device."""

        os.close(self.descriptor)

    def read_ready_events(self, timeout: float) -> list[tuple[int, int, int]]:
        """Return all currently available ``(type, number, value)`` events."""

        readable, _, _ = select.select([self.descriptor], [], [], timeout)
        if not readable:
            return []
        result: list[tuple[int, int, int]] = []
        while True:
            try:
                raw = os.read(self.descriptor, JS_EVENT.size)
            except BlockingIOError:
                break
            if not raw:
                raise OSError(f"joystick disconnected: {self.path}")
            if len(raw) != JS_EVENT.size:
                raise OSError("short Linux joystick event")
            _milliseconds, value, event_type, number = JS_EVENT.unpack(raw)
            result.append((event_type, number, value))
        return result


class LocalXboxController:
    """Direct USB gamepad source sampled by the robot control process."""

    def __init__(self, path: Path) -> None:
        self.joystick = LinuxJoystick(path)
        self.state = XboxJoystickState()
        self.sequence = 0

    def close(self) -> None:
        self.joystick.close()

    def __enter__(self) -> "LocalXboxController":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def sample(self, timeout: float = 0.0) -> ControllerSample:
        for event_type, number, value in self.joystick.read_ready_events(
            timeout
        ):
            self.state.apply_event(event_type, number, value)
        result = self.state.sample(self.sequence)
        self.sequence += 1
        return result


__all__ = [
    "DEFAULT_AXIS_MAP",
    "DEFAULT_BUTTON_MAP",
    "JS_EVENT_AXIS",
    "JS_EVENT_BUTTON",
    "JS_EVENT_INIT",
    "LinuxJoystick",
    "LocalXboxController",
    "XboxJoystickState",
    "normalize_joystick_axis",
]
