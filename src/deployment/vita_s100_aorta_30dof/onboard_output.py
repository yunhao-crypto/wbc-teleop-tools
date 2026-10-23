"""Condition, guard and serialize outputs entirely in memory.

This module has no robot SDK, network, publisher or lease dependency. Ownership
checks use a clearly simulated identity. Bytes omit external ownership fields
and Aorta headers; they are an inert LowCmd schema fixture, not a ready command
for an active session. Actual ownership and SDK publication remain untested.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .command_conditioning import VitaCommandAdapter, validate_gains
from .command_guard import (
    CommandEnvelope, FailClosedCommandGuard, GuardDecision, SessionIdentity,
    UnsafeCommandError,
)


@dataclass(frozen=True)
class InertOutput:
    command_id: int
    target_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    conditioning: dict
    decision: GuardDecision
    payload: bytes | None
    timing_ms: dict
    ownership_scope: str = "simulated-local-only"
    action_sent: bool = False


def serialize_inert_lowcmd(command_id: int, target_q, kp, kd) -> bytes:
    """Encode the verified lowlevel.LowCmd/MotorCmd layout without SDK imports.

    All 42 MotorCmd structs have mode=10, dq=tau=0 and float32 q/kp/kd.
    cmd_id is the only nonzero scalar table field; no external session metadata
    or network header is fabricated. Schema round-trip tests use vendor getters.
    """
    import flatbuffers

    if not 1 <= command_id <= 65535:
        raise ValueError("command_id must be in [1,65535]")
    vectors = [np.asarray(x, np.float32) for x in (target_q, kp, kd)]
    if any(x.shape != (42,) or not np.isfinite(x).all() for x in vectors):
        raise ValueError("LowCmd vectors must be finite [42] float32")
    if any(np.any(x < 0) for x in vectors[1:]):
        raise ValueError("LowCmd gains must be nonnegative")
    builder = flatbuffers.Builder(1200)
    builder.StartVector(24, 42, 4)
    for q, p, d in zip(*(x[::-1] for x in vectors), strict=True):
        builder.Prep(4, 24)
        for value in (d, p, 0., 0., q):
            builder.PrependFloat32(float(value))
        builder.Pad(3)
        builder.PrependUint8(10)
    motors = builder.EndVector()
    builder.StartObject(13)
    builder.PrependUOffsetTRelativeSlot(1, motors, 0)
    builder.PrependUint16Slot(3, command_id, 0)
    builder.Finish(builder.EndObject())
    return bytes(builder.Output())


class InertCommandSink:
    """Run the shared conditioning/guard path, returning bytes to the caller.

    Construct once, then arm from a complete measured 42-slot pose. Retain the
    sink across policy switches and change envelope via process(envelope=...).
    Guard faults latch until a new sink is explicitly constructed, as in the
    original active path. No method can send the returned bytes anywhere.
    """
    def __init__(self, envelope: CommandEnvelope, *, max_state_age_ms=100.):
        self.adapter = VitaCommandAdapter(envelope)
        self.guard = FailClosedCommandGuard(envelope, max_state_age_ms=max_state_age_ms)
        self.identity = SessionIdentity("inert-shadow", "simulated-no-lease", 1, 0)
        self.command_id = 0
        self.armed = False

    def arm(self, current_q) -> GuardDecision:
        if self.armed:
            raise RuntimeError("inert sink is already armed")
        self.adapter.arm(current_q)
        decision = self.guard.arm(self.identity, current_q)
        self.armed = decision.allowed
        return decision

    def process(self, step, *, hold_q, period_s: float,
                now_monotonic_ns: int, lowstate_received_monotonic_ns: int,
                state_id: int, attitude_valid: bool,
                envelope: CommandEnvelope | None = None) -> InertOutput:
        if not self.armed:
            raise RuntimeError("inert sink must be armed before processing")
        if not 0 <= int(state_id) <= 65535:
            raise UnsafeCommandError("state_id must be uint16")
        started = time.monotonic_ns()
        if envelope is not None:
            self.adapter.envelope = envelope
            self.guard.envelope = envelope
        target, kp, kd, conditioning = self.adapter.condition(
            step, hold_q=hold_q, period_s=period_s)
        # Validate the exact values that will survive serialization. Position
        # clipping already rounds inward at float32 envelope boundaries.
        target, kp, kd = (np.asarray(x, np.float32) for x in (target, kp, kd))
        validate_gains(kp, kd, self.adapter.envelope)
        conditioned = time.monotonic_ns()
        decision = self.guard.evaluate(
            target_q=target, period_s=period_s,
            now_monotonic_ns=now_monotonic_ns,
            lowstate_received_monotonic_ns=lowstate_received_monotonic_ns,
            state_id=state_id, attitude_valid=attitude_valid,
            current_action="EXTERNAL", phase="RUNNING",
            client_id=self.identity.client_id, lease_id=self.identity.lease_id,
            generation=self.identity.generation, reject_count=0)
        checked = time.monotonic_ns()
        payload = None
        if decision.allowed:
            self.command_id = self.command_id % 65535 + 1
            payload = serialize_inert_lowcmd(self.command_id, target, kp, kd)
        completed = time.monotonic_ns()
        return InertOutput(self.command_id, target, kp, kd, conditioning,
                           decision, payload, {
                               "conditioning_ms": (conditioned-started)/1e6,
                               "guard_ms": (checked-conditioned)/1e6,
                               "serialization_ms": (completed-checked)/1e6,
                               "output_total_ms": (completed-started)/1e6,
                           })


def load_output_envelopes(path, *, expected_sha256: str) -> dict[str, CommandEnvelope]:
    """Load original exported limits after checking a caller-pinned SHA256.

    Also check the copied original interface and URDF provenance files. The
    caller supplies the expected artifact digest from deployment configuration,
    rather than accepting an unverified hash embedded in the artifact itself.
    """
    import hashlib
    import json
    from pathlib import Path

    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Output envelope artifact SHA256 mismatch")
    record = json.loads(raw)
    if record.get('schema') != 'vita_original_output_envelopes_v1':
        raise ValueError("Unsupported output envelope schema")
    for key in ('motion_urdf', 'deployment_vita02a_a_sample_full_interface'):
        source = record['sources'][key]
        filename = source['copied_file']
        if Path(filename).name != filename:
            raise ValueError("Envelope provenance filename must be local")
        if hashlib.sha256((path.parent / filename).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError("Output envelope provenance SHA256 mismatch: " + key)
    result = {}
    vector_keys = ('lower_rad', 'upper_rad', 'max_input_velocity_rad_s', 'jump_margin_rad')
    for name in ('base', 'velocity', 'motion'):
        values = dict(record['envelopes'][name])
        values['joint_names'] = tuple(values['joint_names'])
        if len(values['joint_names']) != 42 or len(set(values['joint_names'])) != 42:
            raise ValueError("Envelope must have 42 unique slot names")
        for key in vector_keys:
            values[key] = np.asarray(values[key], dtype=np.float64)
            if values[key].shape != (42,) or not np.isfinite(values[key]).all():
                raise ValueError("Invalid envelope vector: " + key)
        env = CommandEnvelope(**values)
        if (np.any(env.lower_rad >= env.upper_rad)
                or np.any(env.max_input_velocity_rad_s <= 0)
                or np.any(env.jump_margin_rad < 0)
                or not np.isfinite([env.min_period_s, env.max_period_s, env.max_kp, env.max_kd]).all()
                or not 0 < env.min_period_s <= env.max_period_s
                or env.max_kp <= 0 or env.max_kd <= 0):
            raise ValueError("Invalid envelope bounds/limits")
        result[name] = env
    return result
