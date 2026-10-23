from __future__ import annotations
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deployment.common.controller_input import (
    Button,
    ControllerSample,
    VelocityLimits,
)
from deployment.common.mode_fsm import TeleopMode
from deployment.vita_s100_aorta_30dof.aorta_adapter import (
    VitaExternalSession,
    active_alignment_quaternion,
    suspended_validation_alignment_quaternion,
)
from deployment.vita_s100_aorta_30dof.cli import _teleop
from deployment.vita_s100_aorta_30dof.config import load_vita_config
from deployment.vita_s100_aorta_30dof.real_teleop import (
    _wait_for_initial_lowstate,
)
from deployment.vita_s100_aorta_30dof.teleop_engine import VitaTeleopEngine

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "vita_s100_aorta_30dof"
)


def _sample(
    sequence: int,
    *buttons: Button,
    axes: dict[str, float] | None = None,
) -> ControllerSample:
    return ControllerSample(
        sequence=sequence,
        monotonic_time=time.monotonic(),
        buttons=frozenset(buttons),
        axes=axes or {},
    )


def test_vita_uses_shared_g1_r1_mode_semantics() -> None:
    engine = VitaTeleopEngine(VelocityLimits())
    output = engine.update(
        _sample(0, Button.START),
        robot_ready=True,
        motion_reference_ready=False,
    )
    assert output.mode is TeleopMode.STAND

    engine.update(_sample(1), robot_ready=True, motion_reference_ready=False)
    output = engine.update(
        _sample(2, Button.A),
        robot_ready=True,
        motion_reference_ready=False,
    )
    assert output.mode is TeleopMode.VELOCITY

    output = engine.update(
        _sample(3, axes={"left_x": 1.0, "left_y": -1.0}),
        robot_ready=True,
        motion_reference_ready=False,
    )
    assert output.velocity_command == (1.0, 0.0, 0.0)

    engine.update(_sample(4), robot_ready=True, motion_reference_ready=True)
    output = engine.update(
        _sample(5, Button.B),
        robot_ready=True,
        motion_reference_ready=True,
    )
    assert output.mode is TeleopMode.MOTION

    engine.update(_sample(6), robot_ready=True, motion_reference_ready=True)
    output = engine.update(
        _sample(7, Button.Y),
        robot_ready=True,
        motion_reference_ready=True,
    )
    assert output.mode is TeleopMode.VELOCITY

    engine.update(_sample(8), robot_ready=True, motion_reference_ready=True)
    output = engine.update(
        _sample(9, Button.X),
        robot_ready=True,
        motion_reference_ready=True,
    )
    assert output.mode is TeleopMode.IDLE


def test_product_has_no_application_session_deadline() -> None:
    config = load_vita_config(PROFILE / "config" / "vita02a_s100.yaml")
    assert config.policy_hz == 50.0
    assert config.aorta_startup_timeout == 30.0
    assert config.sdk_rpc_timeout == 5.0
    assert config.sdk_lease_duration_seconds == 7200.0

    for name in (
        "start_holomotion_vita.sh",
        "start_vita_local_pico.sh",
        "start_vita_suspended_validation.sh",
    ):
        source = (PROFILE / name).read_text(encoding="utf-8")
        assert "--seconds" not in source
        assert "maximum_duration" not in source
        assert "active_timeout" not in source


def test_product_defaults_to_vendor_approved_client_profile() -> None:
    config = load_vita_config(PROFILE / "config" / "vita02a_s100.yaml")
    assert config.aorta_profile == Path(
        "/home/dpx/holomotion_vita/config/pc_session_client.json5"
    )

    launcher = (PROFILE / "start_holomotion_vita.sh").read_text(
        encoding="utf-8"
    )
    assert "pc_session_client.json5" in launcher
    assert "HOLOMOTION_VITA_AORTA_PROFILE" in launcher
    assert "${ZENOH_SESSION_CONFIG_URI:-" not in launcher

    installer = (PROFILE / "scripts" / "install_release.sh").read_text(
        encoding="utf-8"
    )
    assert '"client"' in installer
    assert "pc_session_client.json5" in installer


def test_command_session_waits_for_its_own_subscriptions() -> None:
    status = SimpleNamespace(
        current_action="QUIET_PASSIVE",
        phase=SimpleNamespace(value="IDLE"),
        client_id="",
        lease_id="",
        generation=0,
        reject_count=0,
    )

    class FakeController:
        def __init__(self) -> None:
            self.status = status
            self.lowstate_timeout = None

        def wait_lowstate(self, timeout: float) -> object:
            self.lowstate_timeout = timeout
            return object()

    session = VitaExternalSession.__new__(VitaExternalSession)
    session.controller = FakeController()

    assert session.wait_ready(30.0) is status
    assert 0.0 < session.controller.lowstate_timeout <= 30.0


def test_policy_lowstate_is_primed_with_startup_timeout() -> None:
    expected = SimpleNamespace(state_id=42)

    class FakeStateSource:
        timeout = None

        def read(self, timeout: float) -> object:
            self.timeout = timeout
            return expected

    source = FakeStateSource()
    assert _wait_for_initial_lowstate(source, timeout=30.0) is expected
    assert source.timeout == 30.0


def test_policy_lowstate_startup_timeout_has_clear_context() -> None:
    class FakeStateSource:
        @staticmethod
        def read(_timeout: float) -> object:
            raise TimeoutError("raw subscriber timeout")

    with pytest.raises(
        TimeoutError,
        match="initial policy Vita LowState after 30.0 s",
    ):
        _wait_for_initial_lowstate(FakeStateSource(), timeout=30.0)


def test_command_session_rejects_existing_external_owner() -> None:
    status = SimpleNamespace(
        current_action="EXTERNAL",
        phase=SimpleNamespace(value="RUNNING"),
        client_id="other-client",
        lease_id="other-lease",
    )
    with pytest.raises(RuntimeError, match="active EXTERNAL owner"):
        VitaExternalSession.require_available(status)


def test_command_session_forwards_reviewed_lease_duration() -> None:
    entered = SimpleNamespace(
        current_action="EXTERNAL",
        phase=SimpleNamespace(value="RUNNING"),
        client_id="client",
        lease_id="lease",
        generation=1,
        reject_count=0,
    )

    class FakeController:
        def __init__(self) -> None:
            self.status = SimpleNamespace(
                current_action="QUIET_PASSIVE",
                phase=SimpleNamespace(value="IDLE"),
                client_id="",
                lease_id="",
            )
            self.call = None

        def enter_external(
            self, timeout: float, *, lease_duration_s: float
        ) -> object:
            self.call = (timeout, lease_duration_s)
            return entered

    session = VitaExternalSession.__new__(VitaExternalSession)
    session.controller = FakeController()
    session.entered = False

    assert session.enter(5.0, lease_duration_s=7200.0) is entered
    assert session.controller.call == (5.0, 7200.0)
    assert session.entered


def test_command_session_never_catches_up_faster_than_policy_period() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now_ns = 1_005_000_000
            self.sleep_calls = []

        def monotonic_ns(self) -> int:
            return self.now_ns

        def sleep(self, seconds: float) -> None:
            self.sleep_calls.append(seconds)
            self.now_ns += round(seconds * 1.0e9)

    class FakeController:
        def __init__(self, clock: FakeClock) -> None:
            self.clock = clock
            self.publish_times_ns = []

        def publish(self, _command: object) -> None:
            self.publish_times_ns.append(self.clock.monotonic_ns())

    clock = FakeClock()
    session = VitaExternalSession.__new__(VitaExternalSession)
    session._LowCmd = lambda **values: values
    session._MotorCommand = lambda **values: values
    session.controller = FakeController(clock)
    session.command_id = 0
    session.minimum_publish_period_s = 0.02
    session.last_publish_monotonic_ns = 1_000_000_000
    session._monotonic_ns = clock.monotonic_ns
    session._sleep = clock.sleep

    result = session.publish(np.zeros(42), np.zeros(42), np.zeros(42))

    assert result.command_id == 1
    assert result.published_monotonic_ns == 1_020_000_000
    assert result.period_s == pytest.approx(0.02)
    assert clock.sleep_calls == [pytest.approx(0.015)]
    assert session.controller.publish_times_ns == [1_020_000_000]


def test_active_control_stays_blocked_by_readonly_imu_contract() -> None:
    config = load_vita_config(PROFILE / "config" / "vita02a_s100.yaml")
    with pytest.raises(RuntimeError, match="not active-control approved"):
        active_alignment_quaternion(
            config.imu_alignment_contract,
            robot_serial=config.robot_serial,
        )


def test_reviewed_alignment_is_limited_to_suspended_validation() -> None:
    config = load_vita_config(PROFILE / "config" / "vita02a_s100.yaml")
    value = suspended_validation_alignment_quaternion(
        config.imu_alignment_contract,
        robot_serial=config.robot_serial,
    )
    assert value.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_teleop_rejects_suspended_bundle_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    placeholder_sha = "0" * 64
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(
        "\n".join(
            (
                "schema: holomotion_model_bundle_v1",
                "robot: vita02a_s100_30dof",
                "policy_hz: 50.0",
                "deployment_scope: suspended_validation",
                "artifacts:",
                "  motion:",
                "    path: motion.onnx",
                f'    sha256: "{placeholder_sha}"',
                "    status: trained",
                "  velocity:",
                "    path: velocity.onnx",
                f'    sha256: "{placeholder_sha}"',
                "    status: trained",
                "  smpl:",
                "    path: smpl.pkl",
                f'    sha256: "{placeholder_sha}"',
                "    status: licensed_asset",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOLOMOTION_REAL_ROBOT", "1")
    with pytest.raises(RuntimeError, match="ground-approved"):
        _teleop(
            SimpleNamespace(
                config=PROFILE / "config" / "vita02a_s100.yaml",
                bundle=bundle,
                suspended_validation=False,
            )
        )


def test_suspended_validation_requires_suspended_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    placeholder_sha = "0" * 64
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(
        "\n".join(
            (
                "schema: holomotion_model_bundle_v1",
                "robot: vita02a_s100_30dof",
                "policy_hz: 50.0",
                "deployment_scope: ground",
                "artifacts:",
                "  motion:",
                "    path: motion.onnx",
                f'    sha256: "{placeholder_sha}"',
                "    status: trained",
                "  velocity:",
                "    path: velocity.onnx",
                f'    sha256: "{placeholder_sha}"',
                "    status: trained",
                "  smpl:",
                "    path: smpl.pkl",
                f'    sha256: "{placeholder_sha}"',
                "    status: licensed_asset",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOLOMOTION_REAL_ROBOT", "1")
    with pytest.raises(RuntimeError, match="suspended-validation bundle"):
        _teleop(
            SimpleNamespace(
                config=PROFILE / "config" / "vita02a_s100.yaml",
                bundle=bundle,
                suspended_validation=True,
            )
        )
