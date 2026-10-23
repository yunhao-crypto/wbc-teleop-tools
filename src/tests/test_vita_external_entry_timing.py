"""Exercise START through the real loop with a delayed, offline SDK double."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from deployment.common.controller_input import Button, ControllerSample
from deployment.common.policy_runtime import PolicyStep
from deployment.vita_s100_aorta_30dof import real_teleop as teleop
from deployment.vita_s100_aorta_30dof.aorta_adapter import (
    VITA_CONTROLLER_JOINT_NAMES,
)
from deployment.vita_s100_aorta_30dof.config import load_vita_config
from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_TO_COMMAND_SLOTS,
)


@pytest.mark.parametrize("handshake_seconds", [0.25, 2.0])
@pytest.mark.parametrize("feedback", ["fresh", "stale", "missing"])
@pytest.mark.parametrize(
    "scenario", ["stand", "velocity", "motion", "pico_loss"]
)
def test_start_rebases_feedback_and_ramp_after_handshake(
    tmp_path, monkeypatch, handshake_seconds, feedback, scenario
):
    continue_velocity = scenario != "stand"
    switches = scenario in {"motion", "pico_loss"}
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(
        teleop,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            monotonic_ns=lambda: round(clock.now * 1e9),
            time_ns=lambda: round(clock.now * 1e9),
            thread_time_ns=lambda: round(clock.now * 1e9),
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    config = replace(
        load_vita_config(
            "deployment/vita_s100_aorta_30dof/config/vita02a_s100.yaml"
        ),
        logs=tmp_path / "logs",
    )
    if continue_velocity:
        config = replace(
            config,
            stand_transition_seconds=0.08,
        )
    status = SimpleNamespace(
        current_action="QUIET_PASSIVE",
        phase="IDLE",
        client_id="test",
        lease_id="lease",
        generation=1,
        reject_count=0,
    )
    published = []
    events = []
    reads = []
    input_order = []

    class Session:
        def __init__(self, **kwargs):
            self.status = status

        def wait_ready(self, timeout):
            return status

        def require_available(self, status):
            pass

        def enter(self, timeout, **kwargs):
            clock.now += handshake_seconds
            status.current_action = "EXTERNAL"
            status.phase = "RUNNING"
            return status

        def publish(self, target, kp, kd):
            published.append((np.copy(target), np.copy(kp), np.copy(kd)))
            clock.now += 0.001  # SDK call only; cycle clock owns pacing.
            return SimpleNamespace(
                command_id=len(published),
                published_monotonic_ns=round(clock.now * 1e9),
                command_build_started_wall_ns=round((clock.now - 0.001) * 1e9),
                publish_started_wall_ns=round((clock.now - 0.001) * 1e9),
                published_wall_ns=round(clock.now * 1e9),
                period_s=0.02,
            )

        def exit(self, timeout):
            status.current_action = "QUIET_PASSIVE"

        def close(self):
            pass

    class Source:
        group = config.aorta_group

        def read(self, timeout):
            after_entry = status.current_action == "EXTERNAL"
            if after_entry and feedback == "missing":
                raise TimeoutError("no post-entry LowState")
            clock.now += 0.002
            q = np.zeros(44)
            q[0] = 0.01 if after_entry else 0.0
            raw = SimpleNamespace(
                q_by_slot=q,
                state_id=len(reads) + 1,
                attitude_valid=True,
                received_monotonic_ns=round(
                    (
                        clock.now
                        - (1.0 if after_entry and feedback == "stale" else 0)
                    )
                    * 1e9
                ),
            )
            reads.append(raw)
            input_order.append(("state", raw.state_id))
            return raw

    class Controller:
        count = 0

        def sample(self, timeout):
            self.count += 1
            # The deployed loop polls buttons once more during the START handshake.
            # Keep one stand tick before BACK, preserving the interpolation assertion.
            mapping = {1: {Button.START}, 4: {Button.BACK}}
            if continue_velocity:
                mapping = {
                    1: {Button.START},
                    15: {Button.START},
                    20: {Button.A},
                    23: {Button.BACK},
                }
            if switches:
                mapping.pop(23, None)
                mapping[23] = {Button.B}
                if scenario == "motion":
                    mapping[26] = {Button.Y}
                mapping[30] = {Button.BACK}
            buttons = mapping.get(self.count, set())
            return ControllerSample(
                sequence=self.count,
                monotonic_time=clock.now,
                buttons=frozenset(buttons),
                axes={"left_y": -0.5}
                if continue_velocity and self.count in {17, 21, 22}
                else {},
            )

    def convert(raw, **kwargs):
        return SimpleNamespace(
            joint_pos=raw.q_by_slot[list(VITA_ACTION_TO_COMMAND_SLOTS)],
            joint_vel=np.zeros(30),
            attitude_valid=True,
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            root_ang_vel_b=np.zeros(3),
        )

    target = PolicyStep(
        mode="stand",
        joint_names=VITA_CONTROLLER_JOINT_NAMES,
        raw_action=np.zeros(30),
        target_joint_pos=np.full(30, 0.1),
        joint_stiffness=np.full(30, 10.0),
        joint_damping=np.ones(30),
    )
    policy_calls = []
    anchor_resets = []
    reference = SimpleNamespace(
        reset_root_anchor=lambda: anchor_resets.append(clock.now),
        current_root_rot=lambda: np.array([1.0, 0.0, 0.0, 0.0]),
    )
    motion_target = replace(
        target,
        mode="motion",
        target_joint_pos=np.full(30, 0.2),
        joint_stiffness=np.full(30, 25.0),
    )
    runtime = SimpleNamespace(
        bundle=SimpleNamespace(
            manifest_path=tmp_path / "bundle.yaml",
            artifacts={
                name: SimpleNamespace(
                    path=tmp_path / f"{name}.onnx", sha256="0" * 64
                )
                for name in ("velocity", "motion")
            },
        ),
        contracts=SimpleNamespace(velocity=target, motion=target),
        command_envelope=lambda base, mode: base,
        warmup=lambda: None,
        create_online_reference=lambda: reference,
        stand_step=lambda: target,
        reset=lambda: policy_calls.append("reset"),
        step_motion=lambda *a, **kw: (
            policy_calls.append("motion") or motion_target
        ),
        step_velocity=lambda state, command: (
            policy_calls.append(command) or target
        ),
    )
    monkeypatch.setattr(teleop, "VitaExternalSession", Session)
    monkeypatch.setattr(
        teleop, "VitaRawStateSource", lambda **kw: nullcontext(Source())
    )
    monkeypatch.setattr(
        teleop, "LocalXboxController", lambda *a: nullcontext(Controller())
    )
    monkeypatch.setattr(
        teleop, "ControlProcessLock", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(
        teleop,
        "JsonlSessionRecorder",
        lambda *a, **kw: nullcontext(SimpleNamespace(write=events.append)),
    )
    monkeypatch.setattr(
        teleop, "_wait_for_neutral_controller", lambda *a, **kw: None
    )
    monkeypatch.setattr(teleop, "robot_state_from_raw", convert)
    controller = Controller()
    monkeypatch.setattr(
        teleop, "LocalXboxController", lambda *a: nullcontext(controller)
    )
    pico = SimpleNamespace(start=lambda: None, stop=lambda: None)
    monkeypatch.setattr(
        teleop, "create_pico_source", lambda **kw: pico if switches else None
    )
    monkeypatch.setattr(
        teleop, "store_latest_pico", lambda *a: input_order.append(("pico", None))
    )
    monkeypatch.setattr(
        teleop,
        "_reference_ready",
        lambda *a: scenario != "pico_loss" or controller.count < 25,
    )

    if feedback != "fresh":
        with pytest.raises((RuntimeError, TimeoutError), match="LowState"):
            teleop.run_real_teleop(
                config=config, runtime=runtime, suspended_validation=True
            )
        assert not published
    else:
        teleop.run_real_teleop(
            config=config, runtime=runtime, suspended_validation=True
        )
        if not continue_velocity:
            assert len(published) == 2
        else:
            assert policy_calls.count("reset") == (3 if switches else 1)
            controls = [e for e in events if e.get("record_type") == "control"]
            stand = [e for e in controls if e["mode"] == "stand"]
            velocity = [e for e in controls if e["mode"] == "velocity"]
            assert stand[-1]["stand_phase"] == "zero_velocity"
            assert all(e["velocity_command"] == (0.0, 0.0, 0.0) for e in stand)
            assert velocity[0]["velocity_command"] == (0.0, 0.0, 0.0)
            assert any(e["velocity_command"][0] > 0 for e in velocity)
            if switches:
                motion = [e for e in controls if e["mode"] == "motion"]
                assert motion and len(anchor_resets) == 1
                for frame in motion:
                    read_index = input_order.index(("state", frame["state_id"]))
                    assert input_order[read_index - 1][0] == "pico"
                np.testing.assert_allclose(motion[0]["target_policy"], 0.2)
                np.testing.assert_allclose(
                    motion[0]["kp_sdk"][list(VITA_ACTION_TO_COMMAND_SLOTS)],
                    25.0,
                )
                np.testing.assert_allclose(velocity[-1]["target_policy"], 0.1)
                assert velocity[-1]["velocity_command"] == (0.0, 0.0, 0.0)
            np.testing.assert_array_equal(
                stand[-1]["target_policy"], velocity[0]["target_policy"]
            )
        assert published[0][0][0] == pytest.approx(0.01)
        np.testing.assert_array_equal(
            published[0][1][list(VITA_ACTION_TO_COMMAND_SLOTS)],
            np.full(30, 10.0),
        )
        # Interpolation now reads feedback on the next 20 ms tick, rather
        # than computing early and holding that target until publication.
        expected = 0.01 + (0.1 - 0.01) * (
            0.02 / config.stand_transition_seconds
        )
        assert published[1][0][0] == pytest.approx(expected, abs=1e-8)
        assert published[1][1].max() == 10.0
        entry = next(e for e in events if e.get("event") == "external_running")
        control = next(e for e in events if e.get("record_type") == "control")
        assert entry["state_id"] > entry["pre_entry_state_id"]
        assert control["state_id"] > entry["state_id"]
        controls = [e for e in events if e.get("record_type") == "control"]
        assert all(
            e["timing_ms"]["state_to_publish_ms"] == 1.0 for e in controls
        )
        assert all(e["cycle_skipped_ticks"] == 0 for e in controls)
        assert len(
            [e for e in events if e.get("record_type") == "cycle_timing"]
        ) == len(controls)
    assert status.current_action == "QUIET_PASSIVE"
