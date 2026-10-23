"""Deferred Aorta state/command adapters for Vita02A."""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from deployment.vita_s100_aorta_30dof.command_conditioning import (
    InitialPoseError, VitaCommandAdapter, validate_gains,
)
from deployment.vita_s100_aorta_30dof.live_shadow import (
    live_alignment_quaternion,
    policy_state_from_lowstate,
)
from deployment.vita_s100_aorta_30dof.robot_adapter import (
    VITA_CONTROLLER_JOINT_NAMES,
    VitaRobotState,
)
from deployment.vita_s100_aorta_30dof.state_adapter import (
    VitaRawLowState,

)

from .sdkmain_adapter import (raw_state_from_arrays, verify_state_decoder,
    make_body_command, RealtimeBodyTransport, validate_robot_info)

LOWSTATE_TOPIC = "/sdk/foot_humanoid/body_state"
_ACTIVE_EXTERNAL_PHASES = frozenset(
    ("PREPARING", "ARMED", "TRANSITIONING", "RUNNING", "TIMEOUT_HOLD")
)


@dataclass(frozen=True)
class PublishResult:
    """Timing identity for one command handed to the SDK publisher."""

    command_id: int
    published_monotonic_ns: int
    period_s: float | None
    publish_started_monotonic_ns: int
    command_build_ms: float
    sdk_publish_ms: float
    command_build_started_wall_ns: int = 0
    publish_started_wall_ns: int = 0
    published_wall_ns: int = 0
    period_wall_s: float | None = None
    session_id: int = 0
    sequence: int = 0
    input_state_id: int = 0
    input_state_ts_real_us: int = 0
    ts_client_send_us: int = 0


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON contract must be an object: {path}")
    return value


def active_alignment_quaternion(
    path: str | Path, *, robot_serial: str
) -> np.ndarray:
    """Require a reviewed alignment explicitly promoted for active control."""

    contract = load_json(path)
    approval = contract.get("approval")
    if (
        contract.get("scope") != "robot-active-approved"
        or not isinstance(approval, dict)
        or not approval.get("active_control")
    ):
        raise RuntimeError(
            "HIP IMU alignment is read-only-reviewed, "
            "not active-control approved"
        )
    return live_alignment_quaternion(contract, robot_serial=robot_serial)


def suspended_validation_alignment_quaternion(
    path: str | Path, *, robot_serial: str
) -> np.ndarray:
    """Load the reviewed alignment only for an explicit fixture test."""

    contract = load_json(path)
    approval = contract.get("approval")
    if (
        contract.get("scope") != "robot-readonly-reviewed"
        or not isinstance(approval, dict)
        or not approval.get("live_readonly_shadow")
        or not approval.get("suspended_active_validation")
        or approval.get("active_control")
    ):
        raise RuntimeError(
            "HIP IMU alignment is not approved for suspended validation"
        )
    return live_alignment_quaternion(contract, robot_serial=robot_serial)


def readonly_alignment_quaternion(
    path: str | Path, *, robot_serial: str
) -> np.ndarray:
    return live_alignment_quaternion(
        load_json(path), robot_serial=robot_serial
    )


class VitaRawStateSource:
    """SDK depth-one native arrays, polled only by the main policy thread."""

    def __init__(self, *, group: str) -> None:
        from locomotion_aorta.pull import PullStateSource
        from .sdk021_guard import verify_vendor_sources
        verify_vendor_sources()
        verify_state_decoder()
        self._source = PullStateSource(group=group,
            node_name=f"holomotion-vita-state-{uuid.uuid4()}")
        self._owns_source = True
        self._closed = False
        self._latest_arrays = self._latest_raw = None
        self.received = 0
        try:
            self.group = self._effective_group(self._source)
            if self.group != group:
                raise RuntimeError('effective Aorta group does not match config')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _effective_group(source):
        group = source.transport.node.group
        return group() if callable(group) else group

    def bind_session(self, session):
        """Reuse the ready controller's ring; never take ownership of it."""
        if self._closed:
            raise RuntimeError('State source closed')
        source = session.controller
        if self._effective_group(source) != self.group:
            raise RuntimeError('session/state Aorta groups differ')
        if self._owns_source:
            self._source.close()
        self._source = source
        self._owns_source = False
        self._latest_arrays = self._latest_raw = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_source:
                self._source.close()

    def __enter__(self) -> "VitaRawStateSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read(self, timeout: float, *, wait_for_new: bool = True) -> VitaRawLowState:
        """Select one advancing frame within the original total read budget.

        Ready frames return immediately; gaps wait for the remaining timeout,
        without returning a cached frame to the control loop. Sleep between
        empty SDK polls because wait_state returns its cached snapshot. Publication
        age uses the same-host REALTIME source stamp, not the dequeue clock.
        Receipt MONOTONIC remains a separate local check, with no clock offset.
        """
        import math
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('state timeout must be finite and positive')
        if self._closed:
            raise RuntimeError('State source closed')
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('No advancing local LowState')
            pop_before = time.time_ns()
            arrays = self._source.wait_state(timeout if self._latest_arrays is None else 0)
            if self._closed:
                raise RuntimeError('State source closed')
            if self._latest_arrays is None or arrays.state_id > self._latest_arrays.state_id:
                break
            remaining = deadline - time.monotonic()
            if not wait_for_new or remaining <= 0:
                raise TimeoutError('No advancing local LowState')
            time.sleep(min(.001, remaining))
        received = int(arrays.received_monotonic_ns)
        raw = raw_state_from_arrays(arrays, received_monotonic_ns=received)
        mapped_wall = time.time_ns()
        published_ns = int(arrays.ts_state_pub_us) * 1000
        maximum_age = int(timeout * 1e9)
        if (published_ns <= 0 or not 0 <= mapped_wall - published_ns <= maximum_age
                or not 0 <= time.monotonic_ns() - received <= maximum_age):
            raise RuntimeError('Latest local LowState is stale')
        self._pop_before_wall_ns = pop_before
        self._mapped_wall_ns = mapped_wall
        self._latest_arrays, self._latest_raw = arrays, raw
        self.received += 1
        return raw

def robot_state_from_raw(
    raw: VitaRawLowState, *, base_from_hip_quaternion_wxyz: np.ndarray
) -> VitaRobotState:
    state = policy_state_from_lowstate(
        raw,
        base_from_hip_quaternion_wxyz=base_from_hip_quaternion_wxyz,
    )
    return VitaRobotState(
        joint_names=VITA_CONTROLLER_JOINT_NAMES,
        joint_pos=state.robot_dof_pos,
        joint_vel=state.robot_dof_vel,
        root_quat_wxyz=state.robot_root_quat_wxyz,
        root_ang_vel_b=state.robot_root_angvel_local,
        state_id=state.state_id,
        received_monotonic_ns=state.received_monotonic_ns,
        attitude_valid=raw.attitude_valid,
    )


class VitaExternalSession:
    """Thin command session; imports the robot SDK only when instantiated."""

    def __init__(self, *, group: str) -> None:
        from locomotion_aorta.pull import PullController
        from .sdk021_guard import verify_vendor_sources
        self.vendor_source_hashes = verify_vendor_sources()

        node_name = f"holomotion-vita-command-{os.getpid()}"
        transport = RealtimeBodyTransport(group=group, node_name=node_name)
        try:
            self.controller = PullController(
                group=group, node_name=node_name, _transport=transport)
            self.controller._capture_publish_timing = True
        except BaseException:
            transport.close()
            raise
        self.command_id = 0
        self.entered = False
        self.last_publish_monotonic_ns: int | None = None
        self.last_publish_wall_ns: int | None = None
        self._monotonic_ns = time.monotonic_ns

    def close(self) -> None:
        self.controller.close()

    def __enter__(self) -> "VitaExternalSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def status(self) -> Any:
        return self.controller.status

    def wait_ready(self, timeout: float) -> Any:
        """Wait for this controller's own LowState and status subscriptions."""

        deadline = time.monotonic() + float(timeout)
        self.controller.wait_state(max(0.001, deadline - time.monotonic()))
        while time.monotonic() < deadline:
            status = self.controller.status
            if status is not None:
                self.require_available(status)
                validate_robot_info(self.controller.get_robot_info(
                    max(0.001, deadline - time.monotonic())))
                return status
            time.sleep(0.01)
        raise TimeoutError("timed out waiting for S100 control status")

    @staticmethod
    def require_available(status: Any) -> None:
        if status is None:
            raise RuntimeError("S100 control status is not ready")
        phase = str(getattr(status.phase, "value", status.phase))
        if (
            str(status.current_action) == "EXTERNAL"
            and phase in _ACTIVE_EXTERNAL_PHASES
        ):
            raise RuntimeError(
                "S100 already has an active EXTERNAL owner: "
                f"client_id={status.client_id!r}, "
                f"lease_id={status.lease_id!r}, phase={phase}"
            )

    def enter(self, timeout: float, *, lease_duration_s: float) -> Any:
        status_before = self.controller.status
        if status_before is None:
            raise RuntimeError("S100 control status is not ready")
        self.require_available(status_before)
        # rc.3 owns lease negotiation; retain the adapter argument for callers.
        status = self.controller.enter_external(timeout)
        self.entered = True
        return status

    def exit(self, timeout: float) -> None:
        if self.entered:
            self.controller.exit_external(timeout)
            self.entered = False

    def publish(
        self,
        target_q: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        *,
        input_state_id: int = 0,
        input_state_ts_real_us: int = 0,
    ) -> PublishResult:
        # The policy loop owns pacing before state acquisition. Sending a
        # completed target must not introduce a second, post-inference wait.
        build_started_wall_ns = time.time_ns()
        self.command_id = self.command_id % 65535 + 1
        command = make_body_command(target_q, kp, kd,
                                    created_us=build_started_wall_ns // 1000,
                                    input_state_id=input_state_id,
                                    input_state_ts_real_us=input_state_ts_real_us)
        publish_started_ns = int(self._monotonic_ns())
        publish_started_wall_ns = time.time_ns()
        sent = self.controller.publish(command)
        wire_command = getattr(self.controller.transport, 'last_body_command', None)
        # The SDK return object still has its pre-transport monotonic send stamp.
        wire_send_us = (wire_command.ts_client_send_us
                        if wire_command is not None and
                        (wire_command.session_id, wire_command.sequence) == (sent.session_id, sent.sequence)
                        else 0)
        published_wall_ns = time.time_ns()
        published_monotonic_ns = int(self._monotonic_ns())
        period_s = (
            None
            if self.last_publish_monotonic_ns is None
            else (published_monotonic_ns - self.last_publish_monotonic_ns)
            / 1.0e9
        )
        self.last_publish_monotonic_ns = published_monotonic_ns
        period_wall_s = (
            None if self.last_publish_wall_ns is None
            else (published_wall_ns - self.last_publish_wall_ns) / 1.0e9
        )
        self.last_publish_wall_ns = published_wall_ns
        return PublishResult(
            command_id=self.command_id,
            published_monotonic_ns=published_monotonic_ns,
            period_s=period_s,
            publish_started_monotonic_ns=publish_started_ns,
            command_build_ms=(publish_started_wall_ns - build_started_wall_ns) / 1e6,
            sdk_publish_ms=(published_wall_ns - publish_started_wall_ns) / 1e6,
            command_build_started_wall_ns=build_started_wall_ns,
            publish_started_wall_ns=publish_started_wall_ns,
            published_wall_ns=published_wall_ns,
            period_wall_s=period_wall_s,
            session_id=sent.session_id,
            sequence=sent.sequence,
            input_state_id=sent.input_state_id,
            input_state_ts_real_us=sent.input_state_ts_real_us,
            ts_client_send_us=wire_send_us,
        )


__all__ = [
    "LOWSTATE_TOPIC",
    "PublishResult",
    "VitaCommandAdapter",
    "VitaExternalSession",
    "VitaRawStateSource",
    "active_alignment_quaternion",
    "load_json",
    "readonly_alignment_quaternion",
    "robot_state_from_raw",
    "suspended_validation_alignment_quaternion",
]
