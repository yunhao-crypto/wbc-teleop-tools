"""Robot-independent controller samples, edges, and velocity mapping."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class Button(str, Enum):
    """Logical controls used by the deployment state machine."""

    START = "start"
    A = "a"
    B = "b"
    X = "x"
    Y = "y"
    BACK = "back"
    LB = "lb"
    RB = "rb"
    LT = "lt"
    RT = "rt"


@dataclass(frozen=True)
class ControllerSample:
    """One normalized controller report."""

    sequence: int
    monotonic_time: float
    buttons: frozenset[Button] = field(default_factory=frozenset)
    axes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("controller sequence must be non-negative")
        if not math.isfinite(self.monotonic_time):
            raise ValueError("controller sample time must be finite")
        unknown = set(self.axes) - {"left_x", "left_y", "right_x", "right_y"}
        if unknown:
            raise ValueError(f"unknown controller axes: {sorted(unknown)}")
        for name, value in self.axes.items():
            if not math.isfinite(float(value)) or abs(float(value)) > 1.0:
                raise ValueError(f"controller axis {name} must be in [-1, 1]")


class ControllerEdgeTracker:
    """Convert level-valued controller samples into rising button edges."""

    def __init__(self) -> None:
        self._last_sequence: int | None = None
        self._buttons: frozenset[Button] = frozenset()

    def reset(self) -> None:
        self._last_sequence = None
        self._buttons = frozenset()

    def update(self, sample: ControllerSample) -> frozenset[Button]:
        if (
            self._last_sequence is not None
            and sample.sequence <= self._last_sequence
        ):
            raise ValueError(
                "controller sequence must increase monotonically: "
                f"{sample.sequence} <= {self._last_sequence}"
            )
        pressed = frozenset(sample.buttons - self._buttons)
        self._buttons = sample.buttons
        self._last_sequence = sample.sequence
        return pressed


@dataclass(frozen=True)
class VelocityLimits:
    """Joystick limits matching the velocity-policy training envelope."""

    forward: float = 1.0
    backward: float = 0.6
    lateral: float = 0.5
    yaw: float = 1.0
    deadzone: float = 0.15

    def __post_init__(self) -> None:
        values = (self.forward, self.backward, self.lateral, self.yaw)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("velocity limits must be finite and non-negative")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("controller deadzone must be in [0, 1)")


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply a continuous one-dimensional deadzone to a normalized axis."""

    value = float(value)
    if not math.isfinite(value) or abs(value) > 1.0:
        raise ValueError("normalized controller axis must be in [-1, 1]")
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(min(scaled, 1.0), value)


def velocity_command_from_axes(
    *,
    left_x: float,
    left_y: float,
    right_x: float,
    limits: VelocityLimits,
) -> tuple[float, float, float]:
    """Map conventional gamepad axes to ``vx, vy, yaw_rate``."""

    forward_axis = -apply_deadzone(left_y, limits.deadzone)
    forward_limit = limits.forward if forward_axis >= 0.0 else limits.backward
    return (
        forward_axis * forward_limit,
        -apply_deadzone(left_x, limits.deadzone) * limits.lateral,
        -apply_deadzone(right_x, limits.deadzone) * limits.yaw,
    )


__all__ = [
    "Button",
    "ControllerEdgeTracker",
    "ControllerSample",
    "VelocityLimits",
    "apply_deadzone",
    "velocity_command_from_axes",
]
