"""Pure, fail-closed command checks for future Vita S100 deployment."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_COMMAND_SLOT_COUNT,
    VITA_TARGET_TO_COMMAND_SLOTS,
)

# Adapter slew limiting intentionally lands on the documented jump boundary.
# Permit only floating-point reconstruction noise when the guard recomputes
# abs(target - previous); this is many orders below any physical command.
_JUMP_COMPARISON_ATOL_RAD = 1.0e-12


class UnsafeCommandError(ValueError):
    """Raised when a command or transition violates the frozen contract."""


@dataclass(frozen=True)
class CommandEnvelope:
    joint_names: tuple[str, ...]
    lower_rad: np.ndarray
    upper_rad: np.ndarray
    max_input_velocity_rad_s: np.ndarray
    jump_margin_rad: np.ndarray
    min_period_s: float
    max_period_s: float
    max_kp: float = 600.0
    max_kd: float = 60.0


def command_envelope_from_contract(
    contract: Mapping[str, Any],
) -> CommandEnvelope:
    slots = sorted(contract["slots"], key=lambda item: item["slot"])
    if [item["slot"] for item in slots] != list(
        range(VITA_COMMAND_SLOT_COUNT)
    ):
        raise ValueError("interface contract must contain ordered slots 0..41")
    jump_guard = contract["external"]["jump_guard"]
    default_velocity = float(jump_guard["default_max_input_velocity_rad_s"])
    default_margin = float(jump_guard["default_margin_rad"])
    reduced_velocity = float(
        jump_guard["neck_and_finger_max_input_velocity_rad_s"]
    )
    reduced_margin = float(jump_guard["neck_and_finger_margin_rad"])
    names = tuple(str(item["joint"]) for item in slots)
    reduced = np.asarray(
        ["neck_" in name or "finger_" in name for name in names],
        dtype=np.bool_,
    )
    period_ms = contract["external"]["interpolation_period_ms"]
    return CommandEnvelope(
        joint_names=names,
        lower_rad=np.asarray(
            [item["q_rad"][0] for item in slots], dtype=np.float64
        ),
        upper_rad=np.asarray(
            [item["q_rad"][1] for item in slots], dtype=np.float64
        ),
        max_input_velocity_rad_s=np.where(
            reduced, reduced_velocity, default_velocity
        ),
        jump_margin_rad=np.where(reduced, reduced_margin, default_margin),
        min_period_s=float(period_ms[0]) / 1000.0,
        max_period_s=float(period_ms[1]) / 1000.0,
        max_kp=float(contract["external"]["limits"]["max_kp"]),
        max_kd=float(contract["external"]["limits"]["max_kd"]),
    )


def _command_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (VITA_COMMAND_SLOT_COUNT,):
        raise UnsafeCommandError(
            f"{label} must have {VITA_COMMAND_SLOT_COUNT} values, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise UnsafeCommandError(f"{label} contains NaN or Inf")
    return array


def policy_target_to_command_slots(
    policy_target: Sequence[float], hold_q_by_slot: Sequence[float]
) -> np.ndarray:
    target = np.asarray(policy_target, dtype=np.float64).reshape(-1)
    if target.shape != (len(VITA_TARGET_TO_COMMAND_SLOTS),):
        raise UnsafeCommandError(
            "policy_target must match the 30-joint Vita target DOF contract"
        )
    if not np.isfinite(target).all():
        raise UnsafeCommandError("policy_target contains NaN or Inf")
    complete = _command_vector(hold_q_by_slot, "hold_q_by_slot").copy()
    complete[np.asarray(VITA_TARGET_TO_COMMAND_SLOTS, dtype=np.int64)] = target
    return complete


def position_limit_violations(
    target_q: Sequence[float],
    envelope: CommandEnvelope,
    *,
    tolerance_rad: float = 0.0,
) -> list[dict[str, float | int | str]]:
    if tolerance_rad < 0.0:
        raise ValueError("tolerance_rad must be non-negative")
    target = _command_vector(target_q, "target_q")
    violations = []
    for slot, value in enumerate(target):
        lower = float(envelope.lower_rad[slot])
        upper = float(envelope.upper_rad[slot])
        if value < lower - tolerance_rad or value > upper + tolerance_rad:
            violations.append(
                {
                    "slot": slot,
                    "joint": envelope.joint_names[slot],
                    "value_rad": float(value),
                    "lower_rad": lower,
                    "upper_rad": upper,
                }
            )
    return violations


def jump_guard_violations(
    previous_q: Sequence[float],
    target_q: Sequence[float],
    *,
    period_s: float,
    envelope: CommandEnvelope,
) -> list[dict[str, float | int | str]]:
    if not math.isfinite(period_s) or not (
        envelope.min_period_s <= period_s <= envelope.max_period_s
    ):
        raise UnsafeCommandError(
            "period_s is outside the documented interpolation interval"
        )
    previous = _command_vector(previous_q, "previous_q")
    target = _command_vector(target_q, "target_q")
    delta = np.abs(target - previous)
    allowed = envelope.max_input_velocity_rad_s * period_s
    allowed = allowed + envelope.jump_margin_rad
    return [
        {
            "slot": int(slot),
            "joint": envelope.joint_names[int(slot)],
            "delta_rad": float(delta[int(slot)]),
            "allowed_rad": float(allowed[int(slot)]),
        }
        for slot in np.flatnonzero(delta > allowed + _JUMP_COMPARISON_ATOL_RAD)
    ]


def minimum_smoothstep_duration_seconds(
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    max_peak_velocity_rad_s: float,
    minimum_duration_s: float = 2.0,
) -> float:
    if (
        not math.isfinite(max_peak_velocity_rad_s)
        or max_peak_velocity_rad_s <= 0
    ):
        raise ValueError("max_peak_velocity_rad_s must be positive")
    if not math.isfinite(minimum_duration_s) or minimum_duration_s <= 0:
        raise ValueError("minimum_duration_s must be positive")
    start = _command_vector(start_q, "start_q")
    target = _command_vector(target_q, "target_q")
    required = 1.5 * float(np.max(np.abs(target - start)))
    required /= max_peak_velocity_rad_s
    return max(minimum_duration_s, required)


@dataclass(frozen=True)
class SessionIdentity:
    client_id: str
    lease_id: str
    generation: int
    reject_count: int


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reasons: tuple[str, ...]


class FailClosedCommandGuard:
    """Latch on state, ownership, or structural command failures.

    SDK target rejection is telemetry, not a local ownership failure.  The
    S100 contract permits recovery after a later valid frame while EXTERNAL is
    still owned, including its TIMEOUT_HOLD phase.
    """

    def __init__(
        self,
        envelope: CommandEnvelope,
        *,
        max_state_age_ms: float = 100.0,
        initial_position_tolerance_rad: float = 0.0,
        state_id_bits: int = 16,
    ) -> None:
        if max_state_age_ms <= 0.0:
            raise ValueError("max_state_age_ms must be positive")
        if initial_position_tolerance_rad < 0.0:
            raise ValueError(
                "initial_position_tolerance_rad must be non-negative"
            )
        if state_id_bits not in (16, 64):
            raise ValueError("state_id_bits must be 16 or 64")
        self.state_id_bits = state_id_bits
        self.envelope = envelope
        self.max_state_age_ns = int(max_state_age_ms * 1.0e6)
        self.initial_position_tolerance_rad = initial_position_tolerance_rad
        self._identity: SessionIdentity | None = None
        self._previous_target: np.ndarray | None = None
        self._last_state_id: int | None = None
        self._fault_reasons: tuple[str, ...] = ()

    def _state_reversed(self, state_id: int) -> bool:
        if self.state_id_bits == 64:
            if (isinstance(state_id, bool) or not isinstance(state_id, (int, np.integer))
                    or not 0 <= state_id < (1 << 64)):
                return True
            return self._last_state_id is not None and int(state_id) < self._last_state_id
        if self._last_state_id is None:
            return False
        return (int(state_id) - self._last_state_id) % 65536 >= 32768

    @property
    def fault_reasons(self) -> tuple[str, ...]:
        return self._fault_reasons

    @property
    def last_accepted_target(self) -> np.ndarray:
        """Return a copy of the last target accepted by the guard."""

        if self._previous_target is None:
            raise RuntimeError("command guard must be armed before hold")
        return self._previous_target.copy()

    def arm(
        self, identity: SessionIdentity, current_q: Sequence[float]
    ) -> GuardDecision:
        if self._identity is not None or self._fault_reasons:
            raise RuntimeError("command guard cannot be armed twice")
        reasons = []
        if not identity.client_id or not identity.lease_id:
            reasons.append("missing_session_identity")
        if identity.generation <= 0:
            reasons.append("invalid_session_generation")
        current = _command_vector(current_q, "current_q")
        if position_limit_violations(
            current,
            self.envelope,
            tolerance_rad=self.initial_position_tolerance_rad,
        ):
            reasons.append("initial_position_outside_documented_limits")
        if reasons:
            self._fault_reasons = tuple(reasons)
            return GuardDecision(False, self._fault_reasons)
        self._identity = identity
        self._previous_target = current.copy()
        return GuardDecision(True, ())

    def evaluate(
        self,
        *,
        target_q: Sequence[float],
        period_s: float,
        now_monotonic_ns: int,
        lowstate_received_monotonic_ns: int,
        state_id: int,
        attitude_valid: bool,
        current_action: str,
        phase: str,
        client_id: str,
        lease_id: str,
        generation: int,
        reject_count: int,
    ) -> GuardDecision:
        if self._fault_reasons:
            return GuardDecision(False, self._fault_reasons)
        if self._identity is None or self._previous_target is None:
            raise RuntimeError("command guard must be armed before evaluation")

        reasons = []
        if not attitude_valid:
            reasons.append("attitude_invalid")
        state_age_ns = now_monotonic_ns - lowstate_received_monotonic_ns
        if state_age_ns < 0 or state_age_ns > self.max_state_age_ns:
            reasons.append("lowstate_stale")
        identity = self._identity
        if current_action != "EXTERNAL" or phase not in {
            "RUNNING",
            "TIMEOUT_HOLD",
        }:
            reasons.append("external_not_running")
        if (
            client_id != identity.client_id
            or lease_id != identity.lease_id
            or generation != identity.generation
        ):
            reasons.append("external_ownership_changed")
        if self._state_reversed(state_id):
            reasons.append("lowstate_not_advancing")

        target = _command_vector(target_q, "target_q")
        if position_limit_violations(target, self.envelope):
            reasons.append("target_outside_documented_limits")

        if reasons:
            self._fault_reasons = tuple(dict.fromkeys(reasons))
            return GuardDecision(False, self._fault_reasons)
        self._previous_target = target.copy()
        self._last_state_id = int(state_id)
        return GuardDecision(True, ())

    def evaluate_hold_last(
        self,
        *,
        now_monotonic_ns: int,
        lowstate_received_monotonic_ns: int,
        state_id: int,
        attitude_valid: bool,
        current_action: str,
        phase: str,
        client_id: str,
        lease_id: str,
        generation: int,
        reject_count: int,
    ) -> GuardDecision:
        """Validate an unchanged last-target hold."""

        if self._fault_reasons:
            return GuardDecision(False, self._fault_reasons)
        if self._identity is None or self._previous_target is None:
            raise RuntimeError("command guard must be armed before evaluation")

        reasons = []
        if not attitude_valid:
            reasons.append("attitude_invalid")
        state_age_ns = now_monotonic_ns - lowstate_received_monotonic_ns
        if state_age_ns < 0 or state_age_ns > self.max_state_age_ns:
            reasons.append("lowstate_stale")
        identity = self._identity
        if current_action != "EXTERNAL" or phase not in {
            "RUNNING",
            "TIMEOUT_HOLD",
        }:
            reasons.append("external_not_running")
        if (
            client_id != identity.client_id
            or lease_id != identity.lease_id
            or generation != identity.generation
        ):
            reasons.append("external_ownership_changed")
        if self._state_reversed(state_id):
            reasons.append("lowstate_not_advancing")

        if reasons:
            self._fault_reasons = tuple(dict.fromkeys(reasons))
            return GuardDecision(False, self._fault_reasons)
        self._last_state_id = int(state_id)
        return GuardDecision(True, ())


__all__ = [
    "CommandEnvelope",
    "FailClosedCommandGuard",
    "GuardDecision",
    "SessionIdentity",
    "UnsafeCommandError",
    "command_envelope_from_contract",
    "jump_guard_violations",
    "minimum_smoothstep_duration_seconds",
    "policy_target_to_command_slots",
    "position_limit_violations",
]
