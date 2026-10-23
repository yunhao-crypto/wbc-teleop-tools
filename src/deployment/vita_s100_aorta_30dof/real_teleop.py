"""Pure Start/A/B/Y/X/Back real-robot teleoperation for Vita02A."""

from __future__ import annotations
import logging
import signal
import time
from typing import Any, Sequence

import numpy as np

from deployment.common.controller_input import Button
from deployment.common.mode_fsm import ModeEvent, TeleopGuards, TeleopMode
from deployment.common.policy_runtime import PolicyStep
from deployment.common.process_lock import ControlProcessLock
from deployment.common.xbox_joystick import LocalXboxController
from deployment.vita_s100_aorta_30dof.aorta_adapter import (
    InitialPoseError,
    VitaCommandAdapter,
    VitaExternalSession,
    VitaRawStateSource,
    active_alignment_quaternion,
    load_json,
    readonly_alignment_quaternion,
    robot_state_from_raw,
    suspended_validation_alignment_quaternion,
    validate_gains,
)
from deployment.vita_s100_aorta_30dof.command_guard import (
    FailClosedCommandGuard,
    SessionIdentity,
    command_envelope_from_contract,
)
from deployment.vita_s100_aorta_30dof.config import VitaDeploymentConfig
from deployment.vita_s100_aorta_30dof.pico_input import (
    create_pico_source,
    store_latest_pico,
)
from deployment.vita_s100_aorta_30dof.recording import JsonlSessionRecorder
from deployment.vita_s100_aorta_30dof.runtime import VitaDualPolicyRuntime
from deployment.vita_s100_aorta_30dof.stand_control import (
    VitaStandEntry,
)
from deployment.vita_s100_aorta_30dof.teleop_engine import VitaTeleopEngine
from deployment.vita_s100_aorta_30dof.timing import PolicyCycleClock

LOGGER = logging.getLogger("holomotion.vita.teleop")


class _CancelBeforeInputClose:
    """Attempt ownership cancellation before subscriber/controller teardown."""
    def __init__(self, session_getter, timeout, recorder=None):
        self.session_getter = session_getter
        self.timeout = timeout
        self.recorder = recorder
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, traceback):
        if exc is not None and self.recorder is not None:
            try:
                self.recorder.write({
                    "record_type": "event", "event": "control_loop_failed",
                    "wall_ns": time.time_ns(),
                    "clock": "CLOCK_REALTIME",
                    "error_type": exc_type.__name__, "error": str(exc),
                })
            except Exception as record_error:
                LOGGER.error("Failed to record loop exception: %s", record_error)
        current = self.session_getter()
        if current is not None:
            try:
                current.exit(self.timeout)
            except Exception as exc:
                LOGGER.error(
                    "EXTERNAL cancellation before input cleanup failed: %s",
                    exc,
                )


class _PicoLogger:
    def info(self, message: str) -> None:
        LOGGER.info("[Pico] %s", message)

    def warn(self, message: str) -> None:
        LOGGER.warning("[Pico] %s", message)

    def error(self, message: str) -> None:
        LOGGER.error("[Pico] %s", message)


def yaw_alignment_wxyz(
    robot_quaternion: Sequence[float], reference_quaternion: Sequence[float]
) -> np.ndarray:
    def yaw(quaternion: Sequence[float]) -> float:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
        return float(
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        )

    delta = yaw(robot_quaternion) - yaw(reference_quaternion)
    return np.asarray(
        [np.cos(delta / 2.0), 0.0, 0.0, np.sin(delta / 2.0)],
        dtype=np.float32,
    )


def _reference_ready(reference: Any, maximum_age: float) -> bool:
    return bool(
        reference.has_future_sequence(10, include_derivative_tail=True)
        and reference.data_age(time.time()) <= maximum_age
    )


def _phase(status: Any) -> str:
    return str(getattr(status.phase, "value", status.phase))


def _identity(status: Any) -> SessionIdentity:
    return SessionIdentity(
        client_id=str(status.client_id),
        lease_id=str(status.lease_id),
        generation=int(status.generation),
        reject_count=int(status.reject_count),
    )


def _status_payload(status: Any) -> dict[str, Any]:
    payload = {
        "current_action": str(status.current_action),
        "phase": _phase(status),
        "client_id": str(status.client_id),
        "lease_id": str(status.lease_id),
        "generation": int(status.generation),
        "reject_count": int(status.reject_count),
    }
    for name in (
        "last_reject_code",
        "last_accepted_cmd_id",
        "estimated_input_period_us",
        "ingress_drop_count",
        "timeout_hold",
        "lease_deadline_monotonic_ns",
    ):
        value = getattr(status, name, None)
        if value is not None:
            payload[name] = value
    return payload


def _wait_for_neutral_controller(
    controller: LocalXboxController, *, stop: Any, deadzone: float
) -> None:
    neutral_since = None
    while not stop():
        sample = controller.sample(timeout=0.02)
        neutral = not sample.buttons and all(
            abs(float(value)) <= deadzone for value in sample.axes.values()
        )
        now = time.monotonic()
        neutral_since = (
            now if neutral and neutral_since is None else neutral_since
        )
        if not neutral:
            neutral_since = None
        if neutral_since is not None and now - neutral_since >= 0.5:
            return
    raise KeyboardInterrupt


def _wait_for_initial_lowstate(
    states: VitaRawStateSource, *, timeout: float
) -> Any:
    """Prime policy state before advertising operator readiness."""

    try:
        return states.read(timeout)
    except TimeoutError as exc:
        raise TimeoutError(
            "timed out waiting for the initial policy Vita LowState "
            f"after {timeout:.1f} s"
        ) from exc


def run_real_teleop(
    *,
    config: VitaDeploymentConfig,
    runtime: VitaDualPolicyRuntime,
    suspended_validation: bool = False,
    state_source_factory: Any = None,
    state_converter: Any = None,
    session_factory: Any = None,
    controller_factory: Any = None,
    recorder_factory: Any = None,
) -> None:
    """Run until Back, Ctrl-C, or a safety fault terminates the process."""

    state_source_factory = state_source_factory or VitaRawStateSource
    state_converter = state_converter or robot_state_from_raw
    session_factory = session_factory or VitaExternalSession
    controller_factory = controller_factory or LocalXboxController
    recorder_factory = recorder_factory or JsonlSessionRecorder
    LOGGER.info("Warming Velocity and Motion policies without robot action")
    warmup = runtime.warmup()
    LOGGER.info("Policy warmup and execution providers: %s", warmup)
    reference = runtime.create_online_reference()
    pico = None

    base_from_hip = readonly_alignment_quaternion(
        config.imu_alignment_contract, robot_serial=config.robot_serial
    )
    envelope = command_envelope_from_contract(
        load_json(config.interface_contract)
    )
    envelopes = {
        name: runtime.command_envelope(envelope, name)
        for name in ("velocity", "motion")
    }
    envelope = envelopes["velocity"]
    for name in ("velocity", "motion"):
        contract = getattr(runtime.contracts, name)
        validate_gains(
            contract.joint_stiffness, contract.joint_damping, envelope
        )
    engine = VitaTeleopEngine(config.velocity_limits)
    stop_requested = False

    def request_stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {
        number: signal.signal(number, request_stop)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    period = 1.0 / config.policy_hz
    cycle_clock = PolicyCycleClock(
        period, monotonic_ns=time.monotonic_ns, sleep=time.sleep
    )
    lock_path = config.logs.parent / "holomotion_vita_control.lock"
    session: VitaExternalSession | None = None
    external = False
    guard: FailClosedCommandGuard | None = None
    adapter: VitaCommandAdapter | None = None
    stand_start: PolicyStep | None = None
    stand_target = runtime.stand_step()
    stand_entry: VitaStandEntry | None = None
    stand_complete = False
    yaw_alignment: np.ndarray | None = None
    last_controller_link = None
    last_status_log = 0.0
    last_sdk_reject_count: int | None = None

    try:
        try:
            pico = create_pico_source(
                logger=_PicoLogger(), maximum_age=config.pico_max_age
            )
            pico.start()
        except Exception as exc:
            if pico is not None:
                pico.stop()
            pico = None
            LOGGER.warning(
                "Pico is unavailable; Stand/Velocity remain available: %s", exc
            )
        with (
            ControlProcessLock(lock_path, purpose="vita-holomotion-teleop"),
            recorder_factory(
                config.logs / "sessions", prefix="vita"
            ) as recorder,
            controller_factory(config.joystick) as controller,
            state_source_factory(group=config.aorta_group) as states,
            _CancelBeforeInputClose(lambda: session, config.sdk_rpc_timeout, recorder),
        ):
            LOGGER.warning(
                "CONTROL SCOPE: %s",
                "SUSPENDED VALIDATION"
                if suspended_validation
                else "GROUND RELEASE",
            )
            if states.group != config.aorta_group:
                raise RuntimeError(
                    "effective Aorta group does not match config"
                )
            recorder.write(
                {
                    "record_type": "event",
                    "event": "runtime_started",
                    "wall_ns": time.time_ns(),
                    "clock": "CLOCK_REALTIME",
                    "bundle": str(runtime.bundle.manifest_path),
                    "runtime_warmup": warmup,
                    "motion_selection": getattr(runtime, "motion_selection", None),
                    "policies": {
                        name: {
                            "path": str(runtime.bundle.artifacts[name].path),
                            "sha256": runtime.bundle.artifacts[name].sha256,
                        }
                        for name in ("velocity", "motion")
                    },
                    "deployment_scope": (
                        "suspended_validation"
                        if suspended_validation
                        else "ground"
                    ),
                }
            )
            for name in ("velocity", "motion"):
                artifact = runtime.bundle.artifacts[name]
                LOGGER.info(
                    "POLICY %s: %s SHA256=%s",
                    name,
                    artifact.path,
                    artifact.sha256,
                )
            LOGGER.info(
                "Waiting up to %.1f s for the S100 command session",
                config.aorta_startup_timeout,
            )
            session = session_factory(group=config.aorta_group)
            try:
                ready_status = session.wait_ready(config.aorta_startup_timeout)
                bind_session = getattr(states, 'bind_session', None)
                if bind_session is not None:
                    bind_session(session)
            except Exception as exc:
                recorder.write(
                    {
                        "record_type": "event",
                        "event": "controller_startup_failed",
                        "wall_ns": time.time_ns(),
                        "clock": "CLOCK_REALTIME",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                raise
            recorder.write(
                {
                    "record_type": "event",
                    "event": "controller_ready",
                    "wall_ns": time.time_ns(),
                    "clock": "CLOCK_REALTIME",
                    "status": _status_payload(ready_status),
                }
            )
            _wait_for_neutral_controller(
                controller,
                stop=lambda: stop_requested,
                deadzone=config.velocity_limits.deadzone,
            )
            initial_raw = _wait_for_initial_lowstate(
                states, timeout=config.aorta_startup_timeout
            )
            recorder.write(
                {
                    "record_type": "event",
                    "event": "policy_lowstate_ready",
                    "wall_ns": time.time_ns(),
                    "clock": "CLOCK_REALTIME",
                    "state_id": initial_raw.state_id,
                    "startup_timeout_s": config.aorta_startup_timeout,
                }
            )
            LOGGER.info(
                "Policy LowState ready: state_id=%s",
                initial_raw.state_id,
            )
            recorder.write(
                {
                    "record_type": "event",
                    "event": "operator_ready",
                    "wall_ns": time.time_ns(),
                    "clock": "CLOCK_REALTIME",
                }
            )
            LOGGER.info(
                "READY: START=Prepare then zero-speed Velocity, "
                "A=enable sticks, "
                "B=Pico Motion, Y=Velocity, "
                "X=Idle/PASSIVE, BACK=exit; Ctrl-C also exits"
            )
            pending_raw = initial_raw
            pending_control = None
            while not stop_requested:
                wait_started_wall_ns = time.time_ns()
                tick = cycle_clock.wait() if external else None
                wait_finished_wall_ns = time.time_ns()
                if stop_requested:
                    break
                cycle_started = time.monotonic()
                cycle_started_ns = time.time_ns()
                cycle_started_wall_ns = cycle_started_ns
                timing_ms = {
                    "cycle_wait_ms": (
                        (wait_finished_wall_ns - wait_started_wall_ns) / 1e6
                        if tick else 0.0
                    ),
                }
                # Prepare the human reference before selecting robot feedback.
                # This avoids ageing a selected LowState while Pico is decoded.
                pico_started_ns = time.time_ns()
                if pico is not None:
                    store_latest_pico(pico, reference)
                pico_done_ns = time.time_ns()
                state_read_started_ns = time.time_ns()
                if pending_raw is not None:
                    raw = pending_raw
                    pending_raw = None
                else:
                    raw = states.read(config.state_max_age)
                state_read_done_ns = time.time_ns()
                state_read_ms = (state_read_done_ns - state_read_started_ns) / 1e6
                state_read_done_wall_ns = state_read_done_ns
                state = state_converter(
                    raw, base_from_hip_quaternion_wxyz=base_from_hip
                )
                if time.monotonic_ns() - raw.received_monotonic_ns > int(
                    config.state_max_age * 1.0e9
                ):
                    raise RuntimeError("Vita LowState is stale")
                state_map_done_ns = time.time_ns()
                state_map_done_wall_ns = state_map_done_ns
                # Check reference freshness at the mode decision, as before.
                pico_ready = bool(
                    pico is not None
                    and _reference_ready(reference, config.pico_max_age)
                )
                sample = (
                    pending_control
                    if pending_control is not None
                    else controller.sample(timeout=0.0)
                )
                pending_control = None
                controller_link = getattr(controller, "link_status", None)
                if controller_link is not None:
                    signature = tuple(
                        controller_link.get(k)
                        for k in (
                            "connected",
                            "controls_enabled",
                            "reason",
                            "generation",
                        )
                    )
                    if signature != last_controller_link:
                        recorder.write(
                            {
                                "record_type": "event",
                                "event": "controller_link",
                                "wall_ns": time.time_ns(),
                                "clock": "CLOCK_REALTIME",
                                **controller_link,
                            }
                        )
                        LOGGER.warning("Controller link: %s", controller_link)
                        last_controller_link = signature
                controller_done_ns = time.time_ns()
                timing_ms.update(
                    state_read_ms=state_read_ms,
                    state_mapping_ms=(state_map_done_ns - state_read_done_ns)
                    / 1e6,
                    pico_poll_ms=(pico_done_ns - pico_started_ns) / 1e6,
                    controller_ms=(controller_done_ns - state_map_done_ns) / 1e6,
                )
                stationary = bool(np.max(np.abs(state.joint_vel)) < 0.2)
                mode_before = engine.modes.mode
                robot_ready = bool(
                    state.attitude_valid
                    and (
                        (mode_before is TeleopMode.IDLE and stationary)
                        or stand_complete
                        or mode_before
                        in {TeleopMode.VELOCITY, TeleopMode.MOTION}
                    )
                )
                output = engine.update(
                    sample,
                    robot_ready=robot_ready,
                    motion_reference_ready=pico_ready,
                )
                transition = output.transition
                if transition is not None:
                    LOGGER.log(
                        logging.INFO
                        if transition.accepted
                        else logging.WARNING,
                        "MODE %s -> %s: %s",
                        transition.previous.value,
                        transition.current.value,
                        transition.reason,
                    )
                    if transition.accepted and transition.changed:
                        if transition.current is TeleopMode.ESTOP:
                            break
                        if transition.current is TeleopMode.IDLE:
                            if session is not None and external:
                                session.exit(config.sdk_rpc_timeout)
                                recorder.write(
                                    {
                                        "record_type": "event",
                                        "event": "external_exited",
                                        "wall_ns": time.time_ns(),
                                        "clock": "CLOCK_REALTIME",
                                        "reason": "operator_idle",
                                    }
                                )
                            external = False
                            guard = None
                            adapter = None
                            last_sdk_reject_count = None
                            stand_complete = False
                            stand_entry = None
                            yaw_alignment = None
                        elif transition.current is TeleopMode.STAND:
                            if suspended_validation:
                                suspended_validation_alignment_quaternion(
                                    config.imu_alignment_contract,
                                    robot_serial=config.robot_serial,
                                )
                            else:
                                active_alignment_quaternion(
                                    config.imu_alignment_contract,
                                    robot_serial=config.robot_serial,
                                )
                            if session is None:
                                raise RuntimeError(
                                    "S100 command session is unavailable"
                                )
                            hold_q = np.asarray(
                                raw.q_by_slot[:42], dtype=np.float64
                            )
                            envelope = envelopes["velocity"]
                            adapter = VitaCommandAdapter(envelope)
                            try:
                                adapter.arm(hold_q)
                            except InitialPoseError as exc:
                                LOGGER.warning(
                                    "START not accepted; remaining IDLE: %s",
                                    exc,
                                )
                                recorder.write(
                                    {
                                        "record_type": "event",
                                        "event": "entry_pose_rejected",
                                        "reason": str(exc),
                                    }
                                )
                                engine.modes.handle(
                                    ModeEvent.IDLE, TeleopGuards()
                                )
                                adapter = None
                                continue
                            session.require_available(session.status)
                            LOGGER.info(
                                "Requesting EXTERNAL for %.0f seconds; "
                                "BACK/Ctrl-C exits earlier",
                                config.sdk_lease_duration_seconds,
                            )
                            recorder.write(
                                {
                                    "record_type": "event",
                                    "event": "prepare_started",
                                    "wall_ns": time.time_ns(),
                                    "clock": "CLOCK_REALTIME",
                                    "lease_duration_seconds": (
                                        config.sdk_lease_duration_seconds
                                    ),
                                    "status": _status_payload(session.status),
                                }
                            )
                            try:
                                entered = session.enter(
                                    config.sdk_rpc_timeout,
                                    lease_duration_s=(
                                        config.sdk_lease_duration_seconds
                                    ),
                                )
                            except Exception as exc:
                                recorder.write(
                                    {
                                        "record_type": "event",
                                        "event": "prepare_failed",
                                        "wall_ns": time.time_ns(),
                                        "clock": "CLOCK_REALTIME",
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                    }
                                )
                                raise
                            external = True
                            # PREPARE/COMMIT can outlast fresh LowState.
                            # Rebase takeover on feedback received after the
                            # blocking handshake, not on the pre-START sample.
                            entry_state_id = raw.state_id
                            raw = states.read(config.state_max_age)
                            state_age_ns = (
                                time.monotonic_ns() - raw.received_monotonic_ns
                            )
                            if (
                                not 0
                                <= state_age_ns
                                <= int(config.state_max_age * 1.0e9)
                            ):
                                raise RuntimeError(
                                    "Vita LowState is stale "
                                    "after EXTERNAL entry"
                                )
                            if not raw.attitude_valid:
                                raise RuntimeError(
                                    "Vita attitude is invalid "
                                    "after EXTERNAL entry"
                                )
                            state = state_converter(
                                raw,
                                base_from_hip_quaternion_wxyz=base_from_hip,
                            )
                            hold_q = np.asarray(
                                raw.q_by_slot[:42], dtype=np.float64
                            )
                            try:
                                adapter.arm(hold_q)
                            except InitialPoseError as exc:
                                session.exit(config.sdk_rpc_timeout)
                                external = False
                                LOGGER.warning(
                                    "Entry pose changed; returned to IDLE: %s",
                                    exc,
                                )
                                recorder.write(
                                    {
                                        "record_type": "event",
                                        "event": "entry_pose_rejected",
                                        "reason": str(exc),
                                    }
                                )
                                engine.modes.handle(
                                    ModeEvent.IDLE, TeleopGuards()
                                )
                                adapter = None
                                continue
                            last_sdk_reject_count = int(entered.reject_count)
                            recorder.write(
                                {
                                    "record_type": "event",
                                    "event": "external_running",
                                    "wall_ns": time.time_ns(),
                                    "clock": "CLOCK_REALTIME",
                                    "status": _status_payload(entered),
                                    "pre_entry_state_id": entry_state_id,
                                    "state_id": raw.state_id,
                                    "state_age_ms": ((time.time_ns() - raw.received_wall_ns) / 1.0e6
                                                     if getattr(raw, "received_wall_ns", 0) else None),
                                }
                            )
                            guard = FailClosedCommandGuard(
                                envelope,
                                state_id_bits=64,
                                max_state_age_ms=config.state_max_age * 1000.0,
                            )
                            armed = guard.arm(_identity(entered), hold_q)
                            if not armed.allowed:
                                raise RuntimeError(
                                    "command guard rejected entry: "
                                    + ",".join(armed.reasons)
                                )
                            # First target equals post-handshake feedback, with
                            # model gains already active. No zero-gain gap.
                            stand_start = PolicyStep(
                                mode="stand-entry",
                                joint_names=stand_target.joint_names,
                                raw_action=np.zeros(30, dtype=np.float32),
                                target_joint_pos=state.joint_pos.copy(),
                                joint_stiffness=stand_target.joint_stiffness.copy(),
                                joint_damping=stand_target.joint_damping.copy(),
                            )
                            (
                                initial_q,
                                initial_kp,
                                initial_kd,
                                initial_conditioning,
                            ) = adapter.condition(
                                stand_start, hold_q=hold_q, period_s=period
                            )
                            # Ownership can change while post-entry feedback is
                            # acquired. Apply the same live guard before the very
                            # first hold as before every subsequent command.
                            initial_status = session.status
                            if initial_status is None:
                                raise RuntimeError("S100 status missing before initial hold")
                            initial_decision = guard.evaluate(
                                target_q=initial_q,
                                period_s=period,
                                now_monotonic_ns=time.monotonic_ns(),
                                lowstate_received_monotonic_ns=raw.received_monotonic_ns,
                                state_id=raw.state_id,
                                attitude_valid=raw.attitude_valid,
                                current_action=initial_status.current_action,
                                phase=_phase(initial_status),
                                client_id=initial_status.client_id,
                                lease_id=initial_status.lease_id,
                                generation=initial_status.generation,
                                reject_count=initial_status.reject_count,
                            )
                            if not initial_decision.allowed:
                                raise RuntimeError("initial command guard rejected target: "
                                                   + ",".join(initial_decision.reasons))
                            if stop_requested:
                                break
                            # Handshake can block: recheck link and abort buttons
                            # immediately before any actuator frame is emitted.
                            pending_control = controller.sample(timeout=0.0)
                            if pending_control.buttons & {
                                Button.BACK,
                                Button.X,
                            }:
                                break
                            initial_publish = session.publish(
                                initial_q, initial_kp, initial_kd,
                                input_state_id=raw.state_id,
                                input_state_ts_real_us=raw.ts_state_real_us,
                            )
                            cycle_clock.reset(
                                after_ns=initial_publish.published_monotonic_ns
                            )
                            recorder.write(
                                {
                                    "record_type": "event",
                                    "event": "initial_hold_published",
                                    "target_sdk": initial_q,
                                    "kp_sdk": initial_kp,
                                    "kd_sdk": initial_kd,
                                    "conditioning": initial_conditioning,
                                    "command_id": initial_publish.command_id,
                                    "command_id_semantics": "local diagnostic; wire key is session_id/sequence",
                                    "session_id": initial_publish.session_id,
                                    "sequence": initial_publish.sequence,
                                    "input_state_id": initial_publish.input_state_id,
                                    "input_state_ts_real_us": initial_publish.input_state_ts_real_us,
                                    "ts_client_send_us": initial_publish.ts_client_send_us,
                                    "published_wall_ns": initial_publish.published_wall_ns,
                                    "publish_period_s": (
                                        initial_publish.period_wall_s
                                    ),
                                }
                            )
                            stand_entry = VitaStandEntry(
                                stand_start,
                                stand_target,
                                started=time.monotonic(),
                                prepare_seconds=config.stand_transition_seconds,
                            )
                            stand_complete = False
                            # The measured-pose hold supplied this cycle's
                            # command. Start interpolation on a fresh cycle.
                            continue
                        elif transition.current in {
                            TeleopMode.VELOCITY,
                            TeleopMode.MOTION,
                        }:
                            # START already runs Velocity. B/Y (including
                            # reference loss) reset caches and switch directly.
                            if transition.previous is not TeleopMode.STAND:
                                runtime.reset()
                            if transition.current is TeleopMode.MOTION:
                                reference.reset_root_anchor()
                                reference_root = reference.current_root_rot()
                                if reference_root is None:
                                    raise RuntimeError(
                                        "Pico root orientation is unavailable"
                                    )
                                yaw_alignment = yaw_alignment_wxyz(
                                    state.root_quat_wxyz, reference_root
                                )
                            else:
                                yaw_alignment = None

                mode = engine.modes.mode
                if mode is TeleopMode.IDLE:
                    time.sleep(0.01)
                    continue
                if (
                    not external
                    or session is None
                    or adapter is None
                    or guard is None
                ):
                    raise RuntimeError("active mode has no EXTERNAL ownership")
                policy_started_ns = time.time_ns()
                policy_started_wall_ns = policy_started_ns
                timing_ms["mode_handling_ms"] = (
                    policy_started_ns - controller_done_ns
                ) / 1e6
                if mode is TeleopMode.STAND:
                    if stand_entry is None:
                        raise RuntimeError("stand entry state is unavailable")
                    prior_phase = stand_entry.phase
                    step = stand_entry.step(runtime, state, now=cycle_started)
                    stand_complete = stand_entry.complete
                    if stand_entry.phase != prior_phase:
                        LOGGER.info("START phase=%s", stand_entry.phase)
                        recorder.write(
                            {
                                "record_type": "event",
                                "event": "stand_phase_changed",
                                "phase": stand_entry.phase,
                                "wall_ns": time.time_ns(),
                                "clock": "CLOCK_REALTIME",
                            }
                        )
                elif mode is TeleopMode.VELOCITY:
                    step = runtime.step_velocity(
                        state, output.velocity_command
                    )
                elif mode is TeleopMode.MOTION:
                    if yaw_alignment is None:
                        raise RuntimeError("Pico yaw alignment is unavailable")
                    step = runtime.step_motion(
                        state,
                        reference,
                        reference_yaw_alignment_wxyz=yaw_alignment,
                    )
                else:
                    break
                policy_done_ns = time.time_ns()
                policy_done_wall_ns = policy_done_ns
                timing_ms["policy_total_ms"] = (
                    policy_done_ns - policy_started_ns
                ) / 1e6
                if step.mode in {"velocity", "motion"}:
                    timing_ms.update(getattr(runtime, "last_timing_ms", {}))
                envelope = envelopes[
                    "motion" if mode is TeleopMode.MOTION else "velocity"
                ]
                adapter.envelope = envelope
                guard.envelope = envelope
                hold_q = np.asarray(raw.q_by_slot[:42], dtype=np.float64)
                target, kp, kd, conditioning = adapter.condition(
                    step, hold_q=hold_q, period_s=period
                )
                conditioning_done_ns = time.time_ns()
                conditioning_done_wall_ns = conditioning_done_ns
                timing_ms["conditioning_ms"] = (
                    conditioning_done_ns - policy_done_ns
                ) / 1e6
                status = session.status
                if status is None:
                    raise RuntimeError("S100 control status is unavailable")
                current_reject_count = int(status.reject_count)
                if (
                    last_sdk_reject_count is not None
                    and current_reject_count != last_sdk_reject_count
                ):
                    reject_payload = _status_payload(status)
                    recorder.write(
                        {
                            "record_type": "event",
                            "event": "sdk_command_rejected",
                            "wall_ns": time.time_ns(),
                            "clock": "CLOCK_REALTIME",
                            "reject_count_previous": last_sdk_reject_count,
                            "reject_count_current": current_reject_count,
                            "status": reject_payload,
                        }
                    )
                    LOGGER.warning(
                        "S100 rejected command frame(s): reject_count %d -> "
                        "%d, code=%s, last_accepted_cmd_id=%s; continuing "
                        "while EXTERNAL ownership is valid",
                        last_sdk_reject_count,
                        current_reject_count,
                        reject_payload.get("last_reject_code", "unknown"),
                        reject_payload.get("last_accepted_cmd_id", "unknown"),
                    )
                last_sdk_reject_count = current_reject_count
                decision = guard.evaluate(
                    target_q=target,
                    period_s=period,
                    now_monotonic_ns=time.monotonic_ns(),
                    lowstate_received_monotonic_ns=raw.received_monotonic_ns,
                    state_id=raw.state_id,
                    attitude_valid=raw.attitude_valid,
                    current_action=status.current_action,
                    phase=_phase(status),
                    client_id=status.client_id,
                    lease_id=status.lease_id,
                    generation=status.generation,
                    reject_count=status.reject_count,
                )
                if not decision.allowed:
                    raise RuntimeError(
                        "command guard rejected target: "
                        + ",".join(decision.reasons)
                    )
                guard_done_ns = time.time_ns()
                guard_done_wall_ns = guard_done_ns
                timing_ms["status_guard_ms"] = (
                    guard_done_ns - conditioning_done_ns
                ) / 1e6
                if stop_requested:
                    break
                # Keep the newly sampled controls for next FSM update so
                # this freshness check does not consume button edges.
                pending_control = controller.sample(timeout=0.0)
                if pending_control.buttons & {Button.BACK, Button.X}:
                    if Button.BACK in pending_control.buttons:
                        break
                    session.exit(config.sdk_rpc_timeout)
                    continue
                check_recorder = getattr(recorder, "check_health", None)
                if check_recorder is not None:
                    check_recorder()
                publish_cpu_started_ns = time.thread_time_ns()
                publish_result = session.publish(
                    target, kp, kd, input_state_id=raw.state_id,
                    input_state_ts_real_us=raw.ts_state_real_us)
                timing_ms["command_build_and_publish_cpu_ms"] = (
                    time.thread_time_ns() - publish_cpu_started_ns
                ) / 1e6
                timing_ms.update(
                    command_build_ms=getattr(
                        publish_result, "command_build_ms", None
                    ),
                    sdk_publish_ms=getattr(
                        publish_result, "sdk_publish_ms", None
                    ),
                    state_to_publish_ms=(
                        publish_result.published_wall_ns
                        - getattr(raw, "received_wall_ns", 0)
                    ) / 1e6 if getattr(raw, "received_wall_ns", 0) else None,
                )
                record_started_ns = time.time_ns()
                recorder.write(
                    {
                        "record_type": "control",
                        "wall_ns": time.time_ns(),
                        "clock": "CLOCK_REALTIME",
                        "deployment_scope": (
                            "suspended_validation"
                            if suspended_validation
                            else "ground"
                        ),
                        "state_id": raw.state_id,
                        "applied_session_id": raw.applied_session_id,
                        "applied_sequence": raw.applied_sequence,
                        "ts_first_output_us": raw.ts_first_output_us,
                        "source_imu_timestamps_us": raw.imu_timestamps_us,
                        "source_joint_timestamps_us": raw.joint_timestamps_us,
                        "mode": mode.value,
                        "stand_phase": stand_entry.phase
                        if mode is TeleopMode.STAND and stand_entry is not None
                        else None,
                        "policy_mode": step.mode,
                        "controller": {
                            "sequence": sample.sequence,
                            "buttons": sorted(
                                button.value for button in sample.buttons
                            ),
                            "axes": sample.axes,
                        },
                        "velocity_command": output.velocity_command,
                        "pico_ready": pico_ready,
                        "root_quat_wxyz": state.root_quat_wxyz,
                        "root_ang_vel_b": state.root_ang_vel_b,
                        "q_policy": state.joint_pos,
                        "dq_policy": state.joint_vel,
                        "raw_action": step.raw_action,
                        "onnx_raw_action": getattr(
                            runtime, "last_raw_action", None
                        )
                        if step.mode in {"velocity", "motion"}
                        else None,
                        "policy_observation": getattr(
                            runtime, "last_observation", None
                        )
                        if step.mode in {"velocity", "motion"}
                        else None,
                        "lowstate_age_ms": (
                            (time.time_ns() - raw.received_wall_ns) / 1e6
                            if getattr(raw, "received_wall_ns", 0) else None
                        ),
                        "state_read_ms": state_read_ms,
                        "timing_ms": timing_ms,
                        "cycle_started_wall_ns": cycle_started_ns,
                        "timing_wall_ns": {
                            "clock": "CLOCK_REALTIME",
                            "cycle_started_wall_ns": cycle_started_wall_ns,
                            "state_received_wall_ns": getattr(raw, "received_wall_ns", 0),
                            "state_read_selected_wall_ns": getattr(raw, "read_selected_wall_ns", 0),
                            "state_decode_finished_wall_ns": getattr(raw, "read_finished_wall_ns", 0),
                            "state_read_done_wall_ns": state_read_done_wall_ns,
                            "state_map_done_wall_ns": state_map_done_wall_ns,
                            "policy_started_wall_ns": policy_started_wall_ns,
                            "policy_done_wall_ns": policy_done_wall_ns,
                            "conditioning_done_wall_ns": conditioning_done_wall_ns,
                            "guard_done_wall_ns": guard_done_wall_ns,
                            "command_build_started_wall_ns": publish_result.command_build_started_wall_ns,
                            "sdk_publish_started_wall_ns": publish_result.publish_started_wall_ns,
                            "sdk_returned_wall_ns": publish_result.published_wall_ns,
                            **(getattr(runtime, "last_timing_wall_ns", {})
                               if step.mode in {"velocity", "motion"} else {}),
                        },
                        "cycle_skipped_ticks": tick.skipped_ticks
                        if tick
                        else 0,
                        "state_received_wall_ns": getattr(raw, "received_wall_ns", None),
                        "state_real_us": getattr(
                            raw, "ts_state_real_us", None
                        ),
                        "state_pub_us": getattr(raw, "ts_state_pub_us", None),
                        "native_state_receive_stamp_ns": getattr(
                            raw, "native_receive_stamp_ns", 0
                        ),
                        "native_state_receive_clock": "CLOCK_REALTIME (Aorta SystemTime UNIX_EPOCH)",
                        "state_timestamp_clock": "eth0 PHC disciplined to CLOCK_REALTIME; no offset applied",
                        "command_transport_profile": getattr(
                            session, "last_publish_profile", None
                        ),
                        "sdk_publish_started_wall_ns": publish_result.publish_started_wall_ns,
                        "limit_source": "motion_urdf"
                        if mode is TeleopMode.MOTION
                        else "velocity_metadata",
                        "target_policy": step.target_joint_pos,
                        "target_sdk": target,
                        "kp_sdk": kp,
                        "kd_sdk": kd,
                        "conditioning": conditioning,
                        "sdk_status_before_publish": _status_payload(status),
                        "sdk_reject_count": current_reject_count,
                        "command_id": publish_result.command_id,
                        "command_id_semantics": "local diagnostic; wire key is session_id/sequence",
                        "session_id": publish_result.session_id,
                        "sequence": publish_result.sequence,
                        "input_state_id": publish_result.input_state_id,
                        "input_state_ts_real_us": publish_result.input_state_ts_real_us,
                        "ts_client_send_us": publish_result.ts_client_send_us,
                        "published_wall_ns": publish_result.published_wall_ns,
                        "publish_period_s": publish_result.period_wall_s,
                    }
                )
                record_done_ns = time.time_ns()
                # Measures the recorder call: serialization/write for the sync
                # recorder, snapshot/enqueue for the process recorder, not fsync.
                recorder.write(
                    {
                        "record_type": "cycle_timing",
                        "command_id": publish_result.command_id,
                        "command_id_semantics": "local diagnostic; wire key is session_id/sequence",
                        "session_id": publish_result.session_id,
                        "sequence": publish_result.sequence,
                        "input_state_id": publish_result.input_state_id,
                        "input_state_ts_real_us": publish_result.input_state_ts_real_us,
                        "ts_client_send_us": publish_result.ts_client_send_us,
                        "record_write_ms": (record_done_ns - record_started_ns)
                        / 1e6,
                        "record_write_semantics": getattr(
                            recorder, "write_semantics", "serialize_and_write"
                        ),
                        "cycle_through_record_ms": (
                            record_done_ns - cycle_started_ns
                        )
                        / 1e6,
                        "wall_ns": record_done_ns,
                        "clock": "CLOCK_REALTIME",
                    }
                )
                if cycle_started - last_status_log >= 1.0:
                    LOGGER.info(
                        "STATUS mode=%s vx=%.2f vy=%.2f yaw=%.2f pico=%s "
                        "clips(position/slew/gain)=%d/%d/%d",
                        mode.value,
                        *output.velocity_command,
                        "ready" if pico_ready else "waiting",
                        adapter.position_clip_count,
                        adapter.slew_clip_count,
                        adapter.gain_clip_count,
                    )
                    last_status_log = cycle_started
            LOGGER.info("Operator stop requested; leaving EXTERNAL")
            if session is not None and external:
                session.exit(config.sdk_rpc_timeout)
                external = False
                recorder.write(
                    {
                        "record_type": "event",
                        "event": "external_exited",
                        "wall_ns": time.time_ns(),
                        "clock": "CLOCK_REALTIME",
                        "reason": "operator_stop",
                    }
                )
    finally:
        if session is not None:
            try:
                session.exit(config.sdk_rpc_timeout)
            except Exception as exc:
                LOGGER.error("EXTERNAL exit failed: %s", exc)
            session.close()
        if pico is not None:
            pico.stop()
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


__all__ = ["run_real_teleop", "yaw_alignment_wxyz"]
