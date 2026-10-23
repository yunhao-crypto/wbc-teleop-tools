"""G1/K1/R1 Start/A/B/Y/X/Back mode semantics for Vita02A."""

from __future__ import annotations
from dataclasses import dataclass

from deployment.common.controller_input import (
    Button,
    ControllerEdgeTracker,
    ControllerSample,
    VelocityLimits,
    apply_deadzone,
)
from deployment.common.mode_fsm import (
    ModeEvent,
    TeleopGuards,
    TeleopMode,
    TeleopModeMachine,
    TransitionResult,
)


@dataclass(frozen=True)
class VitaEngineOutput:
    mode: TeleopMode
    velocity_command: tuple[float, float, float]
    transition: TransitionResult | None


class VitaTeleopEngine:
    """Direct-controller state machine; the official walk policy has no vy."""

    _MODE_EVENTS = {
        Button.START: ModeEvent.START,
        Button.A: ModeEvent.VELOCITY,
        Button.B: ModeEvent.MOTION,
        Button.Y: ModeEvent.RETURN_VELOCITY,
    }

    def __init__(self, limits: VelocityLimits) -> None:
        self.limits = limits
        self.edges = ControllerEdgeTracker()
        self.modes = TeleopModeMachine()
        self._last_sequence: int | None = None

    def update(
        self,
        sample: ControllerSample,
        *,
        robot_ready: bool,
        motion_reference_ready: bool,
    ) -> VitaEngineOutput:
        if (
            self._last_sequence is not None
            and sample.sequence < self._last_sequence
        ):
            raise ValueError("controller sequence moved backwards")
        if sample.sequence == self._last_sequence:
            pressed = frozenset()
        else:
            pressed = self.edges.update(sample)
            self._last_sequence = sample.sequence
        axes = sample.axes
        forward_axis = -apply_deadzone(
            float(axes.get("left_y", 0.0)), self.limits.deadzone
        )
        forward_limit = (
            self.limits.forward
            if forward_axis >= 0.0
            else self.limits.backward
        )
        command = (
            forward_axis * forward_limit,
            0.0,
            -apply_deadzone(
                float(axes.get("right_x", 0.0)), self.limits.deadzone
            )
            * self.limits.yaw,
        )
        sticks_centered = all(
            apply_deadzone(float(axes.get(name, 0.0)), self.limits.deadzone)
            == 0.0
            for name in ("left_x", "left_y", "right_x", "right_y")
        )
        guards = TeleopGuards(
            robot_ready=robot_ready,
            sticks_centered=sticks_centered,
            motion_reference_ready=motion_reference_ready,
        )
        transition = None
        if Button.BACK in pressed:
            transition = self.modes.handle(ModeEvent.ESTOP, guards)
        elif Button.X in pressed:
            transition = self.modes.handle(ModeEvent.IDLE, guards)
        else:
            events = [
                event
                for button, event in self._MODE_EVENTS.items()
                if button in pressed
            ]
            if len(events) > 1:
                transition = self.modes.handle(ModeEvent.FAULT, guards)
            elif events:
                transition = self.modes.handle(events[0], guards)
        if self.modes.mode is TeleopMode.MOTION and not motion_reference_ready:
            transition = self.modes.handle(ModeEvent.REFERENCE_LOST, guards)
        if self.modes.mode is not TeleopMode.VELOCITY:
            command = (0.0, 0.0, 0.0)
        return VitaEngineOutput(self.modes.mode, command, transition)


__all__ = ["VitaEngineOutput", "VitaTeleopEngine"]
