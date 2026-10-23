"""Small explicit state machine shared by real and shadow teleoperation."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class TeleopMode(str, Enum):
    IDLE = "idle"
    STAND = "stand"
    VELOCITY = "velocity"
    MOTION = "motion"
    ESTOP = "estop"


class ModeEvent(str, Enum):
    START = "start"
    VELOCITY = "velocity"
    MOTION = "motion"
    RETURN_VELOCITY = "return_velocity"
    IDLE = "idle"
    REFERENCE_LOST = "reference_lost"
    ESTOP = "estop"
    FAULT = "fault"


@dataclass(frozen=True)
class TeleopGuards:
    robot_ready: bool = False
    sticks_centered: bool = False
    motion_reference_ready: bool = False


@dataclass(frozen=True)
class TransitionResult:
    previous: TeleopMode
    current: TeleopMode
    event: ModeEvent
    accepted: bool
    reason: str

    @property
    def changed(self) -> bool:
        return self.previous is not self.current


class TeleopModeMachine:
    """G1-style Start/A/B/Y/X/Back mode semantics."""

    def __init__(self, initial: TeleopMode = TeleopMode.IDLE) -> None:
        self._mode = TeleopMode(initial)

    @property
    def mode(self) -> TeleopMode:
        return self._mode

    def _result(
        self,
        previous: TeleopMode,
        event: ModeEvent,
        *,
        accepted: bool,
        reason: str,
    ) -> TransitionResult:
        return TransitionResult(
            previous=previous,
            current=self._mode,
            event=event,
            accepted=accepted,
            reason=reason,
        )

    def handle(
        self, event: ModeEvent, guards: TeleopGuards
    ) -> TransitionResult:
        event = ModeEvent(event)
        previous = self._mode

        if previous is TeleopMode.ESTOP:
            return self._result(
                previous,
                event,
                accepted=False,
                reason="e-stop is latched; restart the deployment",
            )
        if event in {ModeEvent.ESTOP, ModeEvent.FAULT}:
            self._mode = TeleopMode.ESTOP
            reason = "operator e-stop" if event is ModeEvent.ESTOP else "fault"
            return self._result(previous, event, accepted=True, reason=reason)
        if event is ModeEvent.IDLE:
            self._mode = TeleopMode.IDLE
            return self._result(
                previous,
                event,
                accepted=True,
                reason="normal damping/idle requested",
            )
        if event is ModeEvent.START:
            if previous not in {TeleopMode.IDLE, TeleopMode.STAND}:
                return self._result(
                    previous,
                    event,
                    accepted=False,
                    reason="Start is only valid from idle",
                )
            if not guards.robot_ready:
                return self._result(
                    previous,
                    event,
                    accepted=False,
                    reason="robot state is not ready",
                )
            self._mode = TeleopMode.STAND
            return self._result(
                previous,
                event,
                accepted=True,
                reason="stand requested",
            )
        if event is ModeEvent.VELOCITY:
            if previous is not TeleopMode.STAND:
                return self._result(
                    previous,
                    event,
                    accepted=False,
                    reason="Velocity requires stand mode",
                )
            if not guards.robot_ready:
                reason = "robot state is not ready"
            elif not guards.sticks_centered:
                reason = "center both sticks before entering Velocity"
            else:
                self._mode = TeleopMode.VELOCITY
                return self._result(
                    previous,
                    event,
                    accepted=True,
                    reason="velocity requested",
                )
            return self._result(previous, event, accepted=False, reason=reason)
        if event is ModeEvent.MOTION:
            if previous is not TeleopMode.VELOCITY:
                reason = "Pico Motion requires Velocity mode"
            elif not guards.robot_ready:
                reason = "robot state is not ready"
            elif not guards.sticks_centered:
                reason = "center both sticks before entering Pico Motion"
            elif not guards.motion_reference_ready:
                reason = "Pico motion reference is not ready"
            else:
                self._mode = TeleopMode.MOTION
                return self._result(
                    previous,
                    event,
                    accepted=True,
                    reason="Pico Motion requested",
                )
            return self._result(previous, event, accepted=False, reason=reason)
        if event is ModeEvent.RETURN_VELOCITY:
            if previous is not TeleopMode.MOTION:
                reason = "Y is only valid in Pico Motion"
            elif not guards.sticks_centered:
                reason = "center both sticks before returning to Velocity"
            else:
                self._mode = TeleopMode.VELOCITY
                return self._result(
                    previous,
                    event,
                    accepted=True,
                    reason="return to Velocity requested",
                )
            return self._result(previous, event, accepted=False, reason=reason)
        if event is ModeEvent.REFERENCE_LOST:
            if previous is not TeleopMode.MOTION:
                return self._result(
                    previous,
                    event,
                    accepted=False,
                    reason="reference loss is only relevant in Pico Motion",
                )
            self._mode = TeleopMode.VELOCITY
            return self._result(
                previous,
                event,
                accepted=True,
                reason="Pico reference lost; returned to Velocity",
            )
        raise AssertionError(f"unhandled mode event: {event}")


__all__ = [
    "ModeEvent",
    "TeleopGuards",
    "TeleopMode",
    "TeleopModeMachine",
    "TransitionResult",
]
