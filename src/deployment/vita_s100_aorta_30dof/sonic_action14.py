"""Native action14 reference encoding; import has no hardware or model side effects."""
import hashlib
import json
import math
from pathlib import Path
import secrets
import time

CONFIG = Path(__file__).resolve().parents[3] / 'config' / 'actor_pico.json'
MODEL_SHA256 = '2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d'
NECK_NAMES = ('neck_yaw_joint', 'neck_pitch_joint')
# Mechanical travel already used by HUMANOID_TELEOP. Head control clips tighter.
NECK_LIMITS = {
    'neck_yaw_joint': (-1.396263, 1.396263),
    'neck_pitch_joint': (-0.785398, 0.523599),
}


def contract():
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    slots = tuple(config['body_profile']['body_joint_order'])
    if len(slots) != 31 or len(set(slots)) != 31 or slots[14] != 'waist_roll_joint':
        raise ValueError('invalid public body roster')
    if config['artifacts']['model']['sha256'] != MODEL_SHA256:
        raise ValueError('unexpected SONIC model SHA256')
    return slots, hashlib.sha256(raw).hexdigest()


class ReferenceEncoder:
    """Monotonic source-sequence gate; never turns a repeat into a new sample."""
    def __init__(self, runtime_id, control_id=None, *, max_age_s=0.6, session_id=None):
        # SDK 0.3.0 refuses to encode control_id 0. Tracking admission uses
        # runtime_id only; this local id is not an EXTERNAL control lease.
        if control_id is None:
            control_id = secrets.randbits(64) or 1
        if type(runtime_id) is not int or not 0 < runtime_id < 2**64:
            raise ValueError('action14 requires a nonzero RobotInfo runtime_id')
        if type(control_id) is not int or not 0 < control_id < 2**64:
            raise ValueError('action14 BodyCmd requires a nonzero control_id')
        self.runtime_id, self.control_id = runtime_id, control_id
        if session_id is not None and (type(session_id) is not int or not 0 < session_id < 2**64):
            raise ValueError('invalid source session identity')
        if type(max_age_s) not in (int, float) or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('invalid source freshness limit')
        self.session_id = session_id if session_id is not None else (secrets.randbits(64) or 1)
        self.max_age_s = max_age_s
        self.slots, _ = contract()
        self.sequence = 0
        self.last_source_ns = 0
        self.last_source_realtime_us = 0  # local diagnostics only, never BodyState identity

    def encode(self, named_q, root_wxyz, *, source_ns, received_monotonic,
               now_monotonic=None, source_realtime_us=0, sequence=None, neck_q=None):
        from locomotion_aorta import BodyCmd, BodyJointCmd
        from lowlevel.BodyCmdAction import BodyCmdAction
        slots = self.slots
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if not math.isfinite(received_monotonic) or not 0 <= now-received_monotonic <= self.max_age_s:
            raise ValueError('stale/future source receipt')
        if type(source_ns) is not int or source_ns <= self.last_source_ns:
            raise ValueError('duplicate or reversed source timestamp')
        expected = set(slots) - {'waist_roll_joint', 'neck_pitch_joint', 'neck_yaw_joint'}
        if set(named_q) != expected:
            raise ValueError('expected exactly 28 named virtual joints')
        if len(root_wxyz) != 4 or not all(math.isfinite(x) for x in root_wxyz):
            raise ValueError('invalid root quaternion')
        norm = math.sqrt(sum(x*x for x in root_wxyz))
        if abs(norm-1) > 0.001:
            raise ValueError('root quaternion must be unit wxyz')
        if not all(math.isfinite(v) for v in named_q.values()):
            raise ValueError('nonfinite reference')
        if neck_q is None:
            neck = {name: 0.0 for name in NECK_NAMES}
        else:
            if set(neck_q) != set(NECK_NAMES):
                raise ValueError('neck follow requires both neck joints')
            neck = {}
            for name, (low, high) in NECK_LIMITS.items():
                value = float(neck_q[name])
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError('neck target outside mechanical travel')
                neck[name] = value
        joints = tuple(BodyJointCmd(q=float(named_q.get(name, neck.get(name, 0.0)))) for name in slots)
        if type(source_realtime_us) is not int or source_realtime_us < 0:
            raise ValueError('invalid source realtime diagnostics')
        next_sequence = self.sequence+1 if sequence is None else sequence
        if type(next_sequence) is not int or not self.sequence < next_sequence < 2**64:
            raise ValueError('source grid sequence must increase within uint64')
        self.sequence = next_sequence
        self.last_source_ns = source_ns
        self.last_source_realtime_us = source_realtime_us
        now_us = time.time_ns() // 1000
        return BodyCmd(joints=joints, action=BodyCmdAction.HUMANOID_TRACKING,
            protocol_version=3, runtime_id=self.runtime_id, control_id=self.control_id,
            session_id=self.session_id, sequence=self.sequence,
            input_state_id=0, input_state_ts_real_us=0,
            ts_client_created_us=now_us, ts_client_send_us=now_us,
            root_quat=tuple(root_wxyz))


def plan():
    slots, digest = contract()
    return dict(backend='HUMANOID_TRACKING/action14', enabled='conditional',
                reason='requires matching native profile/model, explicit motion, exactly one neck mode, and correlated action/event results; not hardware-validated',
                config_sha256=digest, pinned_model_sha256=MODEL_SHA256,
                public_slots=len(slots), reference_joints=28,
                head_following='opt-in --follow-head; omitted neck targets stay zero',
                gripper='opt-in --gripper-config on the action14 transport; no EXTERNAL request',
                clock='Pico timestamp and receipt retained locally; BodyState dependency IDs zero; creation/send realtime us')
