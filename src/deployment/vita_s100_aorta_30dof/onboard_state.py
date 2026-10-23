"""Selected LowState decoder for policy ticks; no SDK transport or control handles.

Checks all 44 slot identities and roles but decodes Q/Dq only for 30 policy
joints. Unused torque, temperature, counters and secondary IMU vectors are
intentionally outside this policy-input contract. Generated schema accessors
are used rather than assuming fixed byte offsets in FlatBuffers messages.
"""
from __future__ import annotations
import time
import os
import ctypes
import hashlib
from functools import lru_cache
import numpy as np
from .policy_contract import VITA_POLICY_STATE_TO_STATE_SLOTS, VITA_EXPECTED_STATE_SLOT_ROLES
from .live_shadow import VitaLivePolicyState
from .shadow_adapter import quaternion_inverse_wxyz, quaternion_multiply_wxyz, rotate_vector_wxyz, normalize_quaternion_wxyz

_SLOT_TO_POLICY = {slot: index for index, slot in enumerate(VITA_POLICY_STATE_TO_STATE_SLOTS)}


def decode_policy_lowstate_buffer(payload, *, base_from_hip_quaternion_wxyz,
                                  received_monotonic_ns=None):
    native_path = os.environ.get("VITA_STATE_LIBRARY")
    if native_path:
        return _decode_native(payload, native_path, base_from_hip_quaternion_wxyz, received_monotonic_ns)
    from lowlevel.HumanoidLowState import HumanoidLowState
    try:
        view = HumanoidLowState.GetRootAs(payload, 0)
        return decode_policy_lowstate_view(view,
            base_from_hip_quaternion_wxyz=base_from_hip_quaternion_wxyz,
            received_monotonic_ns=received_monotonic_ns)
    except (IndexError, TypeError, OverflowError) as exc:
        raise ValueError('Malformed LowState buffer') from exc


def decode_policy_lowstate_view(view, *, base_from_hip_quaternion_wxyz,
                                received_monotonic_ns=None):
    if int(view.MotorStatesLength()) != 44:
        raise ValueError('LowState must contain 44 motor slots')
    if not bool(view.AttitudeValid()):
        raise ValueError('LowState attitude is invalid')
    q = np.empty(30, np.float32)
    dq = np.empty(30, np.float32)
    seen = set()
    for index in range(44):
        motor = view.MotorStates(index)
        if motor is None:
            raise ValueError('Missing motor state')
        slot = int(motor.SlotIndex())
        if slot < 0 or slot >= 44 or slot in seen:
            raise ValueError('Invalid or duplicate motor slot')
        seen.add(slot)
        if int(motor.SlotRole()) != VITA_EXPECTED_STATE_SLOT_ROLES[slot]:
            raise ValueError(f'Unexpected motor role at slot {slot}')
        policy_index = _SLOT_TO_POLICY.get(slot)
        if policy_index is not None:
            q[policy_index] = motor.Q()
            dq[policy_index] = motor.Dq()
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError('Nonfinite policy Q/Dq')
    count = int(view.ImuStatesLength())
    if not 1 <= count <= 2:
        raise ValueError('LowState must contain one or two known IMUs')
    locations = set()
    hip = None
    for index in range(count):
        sample = view.ImuStates(index)
        if sample is None:
            raise ValueError('Missing IMU sample')
        location = int(sample.Location())
        if location not in (0, 1) or location in locations:
            raise ValueError('Unknown or duplicate IMU location')
        locations.add(location)
        state = sample.State()
        if state is None:
            raise ValueError('Missing IMU state')
        if location == 0:
            if int(sample.Status()) != 1:
                raise ValueError('HIP IMU status is not OK')
            hip = state
    if hip is None:
        raise ValueError('Missing HIP IMU')
    if hip.QuaternionLength() != 4 or hip.GyroscopeLength() != 3:
        raise ValueError('Invalid HIP quaternion/gyro vector size')
    quaternion = np.asarray([hip.Quaternion(i) for i in range(4)], np.float32)
    gyro = np.asarray([hip.Gyroscope(i) for i in range(3)], np.float32)
    if not np.isfinite(quaternion).all() or not np.isfinite(gyro).all():
        raise ValueError('Nonfinite HIP quaternion/gyro')
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise ValueError('HIP quaternion has zero norm')
    normalized = np.ascontiguousarray(quaternion / norm, dtype=np.float32)
    alignment = np.asarray(base_from_hip_quaternion_wxyz, np.float32)
    root = quaternion_multiply_wxyz(normalized, quaternion_inverse_wxyz(alignment))
    angular = rotate_vector_wxyz(alignment, gyro)
    state_id = int(view.StateId())
    if not 0 <= state_id < 65536:
        raise ValueError('state_id must fit uint16')
    return VitaLivePolicyState(state_id=state_id, source_timestamp_us=int(view.TsStateReal()),
        received_monotonic_ns=time.monotonic_ns() if received_monotonic_ns is None else int(received_monotonic_ns),
        robot_root_quat_wxyz=root, robot_root_angvel_local=angular,
        robot_dof_pos=q, robot_dof_vel=dq)


_SCHEMA_HASH = 'de514c40e02f4cd730a3154970e57f48f8f6ecbeaf0af9ed3f19b2df79bf3765'
_NATIVE_SLOTS = (ctypes.c_int32 * 30)(*VITA_POLICY_STATE_TO_STATE_SLOTS)
_NATIVE_ROLES = (ctypes.c_int32 * 44)(*VITA_EXPECTED_STATE_SLOT_ROLES)


@lru_cache(maxsize=4)
def _native_library(path):
    import humanoid_low_state_schema_meta as schema
    if schema.SCHEMA_HASH_HEX != _SCHEMA_HASH or hashlib.sha256(schema.SCHEMA_BFBS).hexdigest() != _SCHEMA_HASH:
        raise RuntimeError('Native state decoder requires the pinned LowState schema')
    lib = ctypes.CDLL(path)
    lib.vita_state_schema_hash.restype = ctypes.c_char_p
    if lib.vita_state_schema_hash().decode('ascii') != _SCHEMA_HASH:
        raise RuntimeError('Native state library schema mismatch')
    lib.vita_state_decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_char_p, ctypes.c_size_t]
    lib.vita_state_decode.restype = ctypes.c_int
    # Configure ctypes once, rather than rebuilding the live signature each tick.
    if hasattr(lib, 'vita_state_decode_live'):
        lib.vita_state_decode_live.argtypes = lib.vita_state_decode.argtypes[:6] + [ctypes.POINTER(ctypes.c_float)] + lib.vita_state_decode.argtypes[6:]
        lib.vita_state_decode_live.restype = ctypes.c_int
    return lib


def _decode_native(payload, path, alignment, received, *, live=False):
    lib = _native_library(path)
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    values = np.empty(67, np.float32)
    metadata = (ctypes.c_uint64 * (3 if live else 2))()
    error = ctypes.create_string_buffer(192)
    if live:
        # New live ABI is required: never silently fall back to duplicate decoding.
        fn = lib.vita_state_decode_live
        all_q = np.empty(44, np.float32)
        result = fn(payload, len(payload), _NATIVE_SLOTS, _NATIVE_ROLES,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), metadata,
            all_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), error, len(error))
    else:
        result = lib.vita_state_decode(payload, len(payload), _NATIVE_SLOTS, _NATIVE_ROLES,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), metadata, error, len(error))
    if result:
        raise ValueError(error.value.decode('utf8', errors='replace'))
    quaternion, gyro = values[60:64], values[64:67]
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError('HIP quaternion has zero or nonfinite norm')
    normalized = np.ascontiguousarray(quaternion / norm, dtype=np.float32)
    alignment = np.asarray(alignment, np.float32)
    if np.array_equal(alignment, (1.0, 0.0, 0.0, 0.0)):
        # Preserve BOTH float64-normalize/float32-round stages from multiply.
        # Only the identity rotation and its repeated construction disappear.
        root = normalize_quaternion_wxyz(normalize_quaternion_wxyz(normalized))
        angular = gyro.copy()
    else:
        root = quaternion_multiply_wxyz(normalized, quaternion_inverse_wxyz(alignment))
        angular = rotate_vector_wxyz(alignment, gyro)
    state = VitaLivePolicyState(state_id=int(metadata[0]), source_timestamp_us=int(metadata[1]),
        received_monotonic_ns=time.monotonic_ns() if received is None else int(received),
        robot_root_quat_wxyz=root, robot_root_angvel_local=angular,
        robot_dof_pos=values[:30], robot_dof_vel=values[30:60])

    return (state, all_q, int(metadata[2])) if live else state
