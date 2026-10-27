"""SDK 0.2.1 native-array boundary; retain internal policy30/conditioning42.

Names come from H1 add17f2 PublicBodyJointNames(). The unused legacy hand and
reserved slots below are zero placeholders, never measurements or hand commands.
No Node, publisher or controller is constructed by importing this module.
"""
from dataclasses import replace
import hashlib
import os
import time

import numpy as np
from locomotion_aorta import BodyCmd, BodyJointCmd, JointError
from locomotion_aorta._body_codec import decode_body_state
from locomotion_aorta.transport import _AortaTransport, BODY_CMD_TOPIC
from locomotion_aorta.pull import PullTransport
from .policy_contract import VITA_COMMAND_SLOT_NAMES, VITA_EXPECTED_STATE_SLOT_ROLES
from .state_adapter import VitaRawLowState, VitaImuSample

from .sdk021_guard import BODY_STATE_SCHEMA_HASH
BODY_NAMES = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_yaw_joint',
    'left_wrist_pitch_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'neck_pitch_joint', 'neck_yaw_joint',
)
BODY_TO_LEGACY = tuple(VITA_COMMAND_SLOT_NAMES.index(n) if n in VITA_COMMAND_SLOT_NAMES
                       else None for n in BODY_NAMES)
BODY_POPULATED = np.array([i for i, slot in enumerate(BODY_TO_LEGACY) if slot is not None])
LEGACY_POPULATED = np.array([slot for slot in BODY_TO_LEGACY if slot is not None])


def verify_state_decoder():
    import body_state_schema_meta as schema
    if os.environ.get('VITA_STATE_LIBRARY'):
        raise RuntimeError('SDK-main uses the SDK BODY2 decoder; unset VITA_STATE_LIBRARY')
    if (schema.SCHEMA_HASH_HEX != BODY_STATE_SCHEMA_HASH or
            hashlib.sha256(schema.SCHEMA_BFBS).hexdigest() != BODY_STATE_SCHEMA_HASH):
        raise RuntimeError('Unexpected BodyState schema')
    from locomotion_aorta._native_body_codec import _state_decoder
    _state_decoder()  # Check the installed native decoder's BFBS identity too.


def validate_robot_info(reply):
    from aorta.services.common.ServiceStatus import ServiceStatus
    info = reply.info
    if (reply.status not in (ServiceStatus.SUCCESS, ServiceStatus.PARTIAL_SUCCESS)
            or info is None or info.protocol_version != 1):
        raise RuntimeError('Body31 robot discovery failed')
    body = info.body
    if (body is None or body.slot_count != 31 or body.joint_names != BODY_NAMES
            or len(body.joints) != 31):
        raise RuntimeError('Unexpected body31 joint roster')
    for index, joint in enumerate(body.joints):
        if joint.body_index != index or joint.populated != (index != 14):
            raise RuntimeError('H1 requires policy30 and unpopulated waist_roll')


def raw_state_from_body(state, *, received_monotonic_ns):
    if state.protocol_version != 2 or len(state.joints) != 31:
        raise ValueError('Expected protocol2 body31 state')
    if not state.attitude_valid:
        raise ValueError('BodyState attitude is invalid')
    q, dq = np.zeros(44, np.float32), np.zeros(44, np.float32)
    for index, joint in enumerate(state.joints):
        legacy = BODY_TO_LEGACY[index]
        if legacy is None:
            # Installation is checked against BodyInfo, not inferred from NONE.
            continue
        if joint.error_code != JointError.NONE:
            raise ValueError('Policy joint Q/Dq unavailable')
        q[legacy], dq[legacy] = joint.q, joint.dq
    if not 1 <= len(state.imu_states) <= 2:
        raise ValueError('Expected one or two known IMUs')
    imus, seen = [], set()
    for sample in state.imu_states:
        if sample.location not in (0, 1) or sample.location in seen or sample.state is None:
            raise ValueError('Missing/duplicate/unknown IMU')
        seen.add(sample.location)
        if sample.location == 0 and (sample.status != 1):
            raise ValueError('HIP IMU quaternion/gyro unavailable')
        imu = sample.state
        imus.append(VitaImuSample(sample.location, sample.status,
            imu.quaternion, imu.gyroscope, imu.accelerometer, imu.rpy,
            sample.ts_sampled_us, sample.ts_acquired_hw_us, sample.ts_ingested_us,
            sample.ts_assembled_us))
    if 0 not in seen:
        raise ValueError('Missing HIP IMU')
    # Only Q/Dq, IMU and state identity are used by this legacy policy container.
    # Other legacy slots/telemetry are placeholders and must not be logged as measured.
    return VitaRawLowState(state.state_id, state.ts_state_real_us, state.ts_state_pub_us,
        state.ts_consumed_us, int(received_monotonic_ns), state.attitude_valid,
        q, dq, np.zeros(44, np.float32), np.zeros(44, np.float32),
        np.zeros(44, np.int64), np.asarray(VITA_EXPECTED_STATE_SLOT_ROLES), tuple(imus),
        state_id_bits=64, applied_session_id=state.applied_session_id,
        applied_sequence=state.applied_sequence, ts_first_output_us=state.ts_first_output_us,
        joint_timestamps_us=tuple((j.ts_acquired_hw_us, j.ts_ingested_us, j.ts_assembled_us)
                                  for j in state.joints))


def decode_body_raw(payload, *, received_monotonic_ns):
    from locomotion_aorta.state_arrays import decode_body_state_arrays
    return raw_state_from_arrays(decode_body_state_arrays(payload),
                                 received_monotonic_ns=received_monotonic_ns)


def validate_body_arrays(state):
    """Check native Body31 inputs without building an unused legacy44 state."""
    if state.protocol_version != 2 or not state.attitude_valid:
        raise ValueError('Expected valid-attitude protocol2 body31 state')
    if (isinstance(state.state_id, bool) or not isinstance(state.state_id, (int, np.integer))
            or not 0 <= state.state_id < 1 << 64):
        raise ValueError('state_id must fit uint64')
    for name, shape, dtype in (
        ('q', (31,), np.float32), ('dq', (31,), np.float32),
        ('error_code', (31,), np.uint32),
        ('joint_stamps_us', (31, 3), np.uint64),
    ):
        value = getattr(state, name)
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError('Unexpected native array: ' + name)
    if np.any(state.error_code[BODY_POPULATED] != JointError.NONE):
        raise ValueError('Policy joint Q/Dq unavailable')
    if not np.isfinite(state.q[BODY_POPULATED]).all() or not np.isfinite(state.dq[BODY_POPULATED]).all():
        raise ValueError('Body q/dq contains NaN or Inf')
    count = len(state.imu_location)
    if not 1 <= count <= 2:
        raise ValueError('Expected one or two known IMUs')
    if (state.imu_values.shape != (count, 13) or
            state.imu_stamps_us.shape != (count, 4) or
            state.imu_stamps_us.dtype != np.uint64 or
            state.imu_status.shape != (count,) or
            state.imu_state_present.shape != (count,)):
        raise ValueError('Unexpected native IMU arrays')
    if not np.isfinite(state.imu_values).all():
        raise ValueError('IMU contains NaN or Inf')
    seen = set()
    for i in range(count):
        location, status = int(state.imu_location[i]), int(state.imu_status[i])
        if location not in (0, 1) or location in seen or not state.imu_state_present[i]:
            raise ValueError('Missing/duplicate/unknown IMU')
        seen.add(location)
        if location == 0 and status != 1:
            raise ValueError('HIP IMU quaternion/gyro unavailable')
    if 0 not in seen:
        raise ValueError('Missing HIP IMU')


def raw_state_from_arrays(state, *, received_monotonic_ns):
    """Convert only for legacy consumers; actor inference stays on Body31."""
    validate_body_arrays(state)
    q, dq = np.zeros(44, np.float32), np.zeros(44, np.float32)
    q[LEGACY_POPULATED], dq[LEGACY_POPULATED] = state.q[BODY_POPULATED], state.dq[BODY_POPULATED]
    imus = tuple(VitaImuSample(int(state.imu_location[i]), int(state.imu_status[i]),
        values[:4], values[4:7], values[7:10], values[10:13],
        *(int(v) for v in state.imu_stamps_us[i])) for i,values in enumerate(state.imu_values))
    return VitaRawLowState(state.state_id, state.ts_state_real_us, state.ts_state_pub_us,
        state.ts_consumed_us, int(received_monotonic_ns), state.attitude_valid,
        q, dq, np.zeros(44, np.float32), np.zeros(44, np.float32),
        np.zeros(44, np.int64), np.asarray(VITA_EXPECTED_STATE_SLOT_ROLES), tuple(imus),
        state_id_bits=64, applied_session_id=state.applied_session_id,
        applied_sequence=state.applied_sequence, ts_first_output_us=state.ts_first_output_us,
        joint_timestamps_us=tuple(tuple(int(v) for v in row) for row in state.joint_stamps_us))


def make_body_command(target_q, kp, kd, *, created_us, input_state_id=0, input_state_ts_real_us=0):
    vectors = tuple(np.asarray(v) for v in (target_q, kp, kd))
    if any(v.shape != (42,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError('Expected three finite internal42 vectors')
    # The frozen conditioner sets all fingers to q=kp=kd=0. Refuse silent loss
    # of a new hand-control intent: hands require their own explicit SDK API.
    body_slots = {i for i in BODY_TO_LEGACY if i is not None}
    if any(v[i] != 0 for v in vectors for i in range(42) if i not in body_slots):
        raise ValueError('Hand commands are outside the body31 policy path')
    joints = tuple(BodyJointCmd() if i is None else BodyJointCmd(
        q=float(vectors[0][i]), kp=float(vectors[1][i]), kd=float(vectors[2][i]))
        for i in BODY_TO_LEGACY)
    return BodyCmd(joints=joints, ts_client_created_us=int(created_us),
                   input_state_id=int(input_state_id),
                   input_state_ts_real_us=int(input_state_ts_real_us))


class RealtimeBodyTransport(PullTransport):
    """Keep SDK session/sequence/validation; wire client stamps use REALTIME.

    The controller assigns a monotonic send stamp. Replace that field only at
    this transport boundary, preserving the native publisher and SDK trace.
    """
    def __init__(self, *, node_name, group):
        super().__init__(node_name=node_name, group=group)
        self._capture_publish_timing = True

    def publish(self, topic, message):
        if topic == BODY_CMD_TOPIC:
            message = replace(message, ts_client_send_us=time.time_ns() // 1000)
        result = super().publish(topic, message)
        if topic == BODY_CMD_TOPIC:
            # Capture the actual REALTIME-stamped wire object, after successful send.
            self.last_body_command = message
        return result
