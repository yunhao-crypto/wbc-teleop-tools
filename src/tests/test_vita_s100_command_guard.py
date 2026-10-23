from __future__ import annotations

import numpy as np

from deployment.common.policy_runtime import PolicyStep
from deployment.vita_s100_aorta_30dof.aorta_adapter import VitaCommandAdapter
from deployment.vita_s100_aorta_30dof.command_guard import (
    CommandEnvelope,
    FailClosedCommandGuard,
    SessionIdentity,
    command_envelope_from_contract,
    jump_guard_violations,
    minimum_smoothstep_duration_seconds,
    policy_target_to_command_slots,
)
from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_TO_COMMAND_SLOTS,
    VITA_TARGET_TO_COMMAND_SLOTS,
)
from deployment.vita_s100_aorta_30dof.robot_adapter import (
    VITA_CONTROLLER_JOINT_NAMES,
)


def _contract():
    return {
        "external": {
            "interpolation_period_ms": [2, 100],
            "limits": {"max_kp": 600.0, "max_kd": 60.0},
            "jump_guard": {
                "default_max_input_velocity_rad_s": 3.0,
                "default_margin_rad": 0.02,
                "neck_and_finger_max_input_velocity_rad_s": 1.0,
                "neck_and_finger_margin_rad": 0.01,
            },
        },
        "slots": [
            {"slot": slot, "joint": f"joint_{slot}", "q_rad": [-2.0, 2.0]}
            for slot in range(40)
        ]
        + [
            {"slot": 40, "joint": "neck_pitch_joint", "q_rad": [-1.0, 1.0]},
            {"slot": 41, "joint": "neck_yaw_joint", "q_rad": [-1.0, 1.0]},
        ],
    }


def test_policy_target_expands_with_hold_values():
    hold = np.arange(42, dtype=np.float64)
    policy = np.arange(30, dtype=np.float64) * 0.01
    complete = policy_target_to_command_slots(policy, hold)
    np.testing.assert_allclose(
        complete[[25, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38, 39]],
        hold[[25, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38, 39]],
    )


def test_policy_target_uses_dof_order_not_raw_action_order():
    assert VITA_TARGET_TO_COMMAND_SLOTS != VITA_ACTION_TO_COMMAND_SLOTS
    hold = np.full(42, -1.0, dtype=np.float64)
    target = np.arange(30, dtype=np.float64)

    complete = policy_target_to_command_slots(target, hold)

    for target_index, command_slot in enumerate(VITA_TARGET_TO_COMMAND_SLOTS):
        assert complete[command_slot] == target[target_index]
    assert complete[41] == target[9]
    assert complete[18] == target[10]


def test_jump_guard_uses_reduced_neck_limit():
    envelope = command_envelope_from_contract(_contract())
    previous = np.zeros(42)
    target = previous.copy()
    target[0] = 0.081
    target[40] = 0.031
    violations = jump_guard_violations(
        previous, target, period_s=0.02, envelope=envelope
    )
    assert [item["slot"] for item in violations] == [0, 40]
    assert all(isinstance(item["slot"], int) for item in violations)


def test_jump_guard_accepts_slew_limited_boundary_roundoff():
    envelope = command_envelope_from_contract(_contract())
    previous = np.zeros(42)
    previous[3] = 0.4123712480068207
    target = previous.copy()
    target[3] = previous[3] + 0.08

    # This exact pair reconstructs as 0.08000000000000002 in float64,
    # although the adapter applied the documented 0.08-rad limit.
    assert abs(target[3] - previous[3]) > 0.08
    assert not jump_guard_violations(
        previous, target, period_s=0.02, envelope=envelope
    )


def test_jump_guard_still_rejects_a_real_boundary_excess():
    envelope = command_envelope_from_contract(_contract())
    previous = np.zeros(42)
    target = previous.copy()
    target[3] = 0.080000001

    violations = jump_guard_violations(
        previous, target, period_s=0.02, envelope=envelope
    )
    assert [item["slot"] for item in violations] == [3]


def _adapter_step(target: float) -> PolicyStep:
    count = len(VITA_CONTROLLER_JOINT_NAMES)
    return PolicyStep(
        mode="test",
        joint_names=VITA_CONTROLLER_JOINT_NAMES,
        raw_action=np.zeros(1),
        target_joint_pos=np.full(count, target),
        joint_stiffness=np.zeros(count),
        joint_damping=np.zeros(count),
    )


def _adapter_envelope(*, position_limit: float = 2.0) -> CommandEnvelope:
    return CommandEnvelope(
        joint_names=tuple(f"joint_{slot}" for slot in range(42)),
        lower_rad=np.full(42, -position_limit),
        upper_rad=np.full(42, position_limit),
        max_input_velocity_rad_s=np.full(42, 3.0),
        jump_margin_rad=np.full(42, 0.02),
        min_period_s=0.002,
        max_period_s=0.1,
    )


def test_command_adapter_does_not_apply_extra_slew_limit():
    envelope = _adapter_envelope()
    adapter = VitaCommandAdapter(envelope)
    previous = np.zeros(42)
    adapter.arm(previous)

    target, _kp, _kd, conditioning = adapter.condition(
        _adapter_step(1.0), hold_q=previous, period_s=0.02
    )

    delta = np.abs(target - previous)
    assert float(np.max(delta)) == 1.0
    assert conditioning["slew_clip_cycles"] == 0


def test_command_adapter_keeps_policy_positions_inside_sdk_boundary():
    envelope = _adapter_envelope(position_limit=0.1)
    adapter = VitaCommandAdapter(envelope)
    previous = np.zeros(42)
    adapter.arm(previous)

    target, _kp, _kd, conditioning = adapter.condition(
        _adapter_step(1.0), hold_q=previous, period_s=0.1
    )

    controlled = np.asarray(VITA_ACTION_TO_COMMAND_SLOTS, dtype=np.int64)
    assert np.all(target[controlled] < envelope.upper_rad[controlled])
    assert np.all(target[controlled] > envelope.lower_rad[controlled])
    assert conditioning["position_clip_cycles"] == 1


def test_command_adapter_preserves_training_damping_within_sdk_limit():
    envelope = _adapter_envelope()
    adapter = VitaCommandAdapter(envelope)
    previous = np.zeros(42)
    adapter.arm(previous)
    count = len(VITA_CONTROLLER_JOINT_NAMES)
    step = PolicyStep(
        mode="test",
        joint_names=VITA_CONTROLLER_JOINT_NAMES,
        raw_action=np.zeros(1),
        target_joint_pos=np.zeros(count),
        joint_stiffness=np.full(count, 176.8663),
        joint_damping=np.full(count, 5.62983),
    )

    _target, kp, kd, _conditioning = adapter.condition(
        step, hold_q=previous, period_s=0.02
    )

    controlled = np.asarray(VITA_ACTION_TO_COMMAND_SLOTS, dtype=np.int64)
    np.testing.assert_allclose(kp[controlled], 176.8663)
    np.testing.assert_allclose(kd[controlled], 5.62983)


def test_invalid_gains_are_reported_not_silently_clipped():
    from dataclasses import replace
    import pytest

    adapter = VitaCommandAdapter(_adapter_envelope())
    adapter.arm(np.zeros(42))
    step = replace(_adapter_step(0.0), joint_stiffness=np.full(30, 601.0))
    with pytest.raises(ValueError, match="gains were not clipped"):
        adapter.condition(step, hold_q=np.zeros(42), period_s=0.02)


def test_duplicate_fresh_state_is_allowed_but_expired_state_is_not():
    guard = FailClosedCommandGuard(command_envelope_from_contract(_contract()))
    assert guard.arm(
        SessionIdentity("client", "lease", 1, 0), np.zeros(42)
    ).allowed
    frame = dict(
        target_q=np.zeros(42),
        period_s=0.02,
        now_monotonic_ns=1_000_000_000,
        lowstate_received_monotonic_ns=999_000_000,
        state_id=1,
        attitude_valid=True,
        current_action="EXTERNAL",
        phase="RUNNING",
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=0,
    )
    assert guard.evaluate(**frame).allowed
    frame["now_monotonic_ns"] += 20_000_000
    assert guard.evaluate(**frame).allowed
    frame["now_monotonic_ns"] += 100_000_000
    assert guard.evaluate(**frame).reasons == ("lowstate_stale",)


def test_smoothstep_duration_matches_peak_velocity_limit():
    start = np.zeros(42)
    target = start.copy()
    target[3] = 1.0
    assert (
        minimum_smoothstep_duration_seconds(
            start, target, max_peak_velocity_rad_s=0.1
        )
        == 15.0
    )


def test_guard_latches_runtime_failure_without_auto_recovery():
    envelope = command_envelope_from_contract(_contract())
    guard = FailClosedCommandGuard(envelope)
    identity = SessionIdentity("client", "lease", 1, 0)
    assert guard.arm(identity, np.zeros(42)).allowed
    denied = guard.evaluate(
        target_q=np.zeros(42),
        period_s=0.02,
        now_monotonic_ns=1_000_000_000,
        lowstate_received_monotonic_ns=999_000_000,
        state_id=1,
        attitude_valid=True,
        current_action="PASSIVE",
        phase="RUNNING",
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=0,
    )
    assert not denied.allowed
    assert denied.reasons == ("external_not_running",)
    still_denied = guard.evaluate(
        target_q=np.zeros(42),
        period_s=0.02,
        now_monotonic_ns=1_020_000_000,
        lowstate_received_monotonic_ns=1_019_000_000,
        state_id=2,
        attitude_valid=True,
        current_action="EXTERNAL",
        phase="RUNNING",
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=0,
    )
    assert not still_denied.allowed
    assert still_denied.reasons == ("external_not_running",)


def test_guard_allows_only_last_target_hold_without_deadman():
    envelope = command_envelope_from_contract(_contract())
    guard = FailClosedCommandGuard(envelope)
    identity = SessionIdentity("client", "lease", 1, 0)
    target = np.zeros(42)
    target[3] = 0.01
    assert guard.arm(identity, np.zeros(42)).allowed
    assert guard.evaluate(
        target_q=target,
        period_s=0.02,
        now_monotonic_ns=1_000_000_000,
        lowstate_received_monotonic_ns=999_000_000,
        state_id=1,
        attitude_valid=True,
        current_action="EXTERNAL",
        phase="RUNNING",
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=0,
    ).allowed

    held = guard.last_accepted_target
    held[3] = 1.0
    assert guard.last_accepted_target[3] == 0.01
    assert guard.evaluate_hold_last(
        now_monotonic_ns=1_020_000_000,
        lowstate_received_monotonic_ns=1_019_000_000,
        state_id=2,
        attitude_valid=True,
        current_action="EXTERNAL",
        phase="RUNNING",
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=0,
    ).allowed


def test_guard_hold_last_still_latches_ownership_loss():
    envelope = command_envelope_from_contract(_contract())
    guard = FailClosedCommandGuard(envelope)
    assert guard.arm(
        SessionIdentity("client", "lease", 1, 0), np.zeros(42)
    ).allowed

    denied = guard.evaluate_hold_last(
        now_monotonic_ns=1_000_000_000,
        lowstate_received_monotonic_ns=999_000_000,
        state_id=1,
        attitude_valid=True,
        current_action="EXTERNAL",
        phase="RUNNING",
        client_id="other",
        lease_id="lease",
        generation=1,
        reject_count=0,
    )

    assert not denied.allowed
    assert denied.reasons == ("external_ownership_changed",)


def test_guard_allows_sdk_single_frame_recovery_in_timeout_hold():
    envelope = command_envelope_from_contract(_contract())
    guard = FailClosedCommandGuard(envelope)
    assert guard.arm(
        SessionIdentity("client", "lease", 1, 0), np.zeros(42)
    ).allowed

    decision = guard.evaluate(
        target_q=np.zeros(42),
        period_s=0.02,
        now_monotonic_ns=1_000_000_000,
        lowstate_received_monotonic_ns=999_000_000,
        state_id=1,
        attitude_valid=True,
        current_action="EXTERNAL",
        phase="TIMEOUT_HOLD",
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=1,
    )

    assert decision.allowed


def test_guard_rejects_initial_pose_outside_documented_limits():
    envelope = command_envelope_from_contract(_contract())
    current = np.zeros(42)
    current[40] = 1.001
    guard = FailClosedCommandGuard(envelope)
    decision = guard.arm(SessionIdentity("client", "lease", 1, 0), current)
    assert not decision.allowed
    assert decision.reasons == ("initial_position_outside_documented_limits",)
