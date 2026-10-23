"""Decode raw Vita HumanoidLowState views without creating Aorta sessions."""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_TO_COMMAND_SLOTS,
    VITA_DOF_TO_STATE_SLOTS,
    VITA_EXPECTED_STATE_SLOT_ROLES,
    VITA_POLICY_STATE_TO_STATE_SLOTS,
    VITA_STATE_SLOT_COUNT,
)

IMU_LOCATION_HIP = 0
IMU_LOCATION_TRUNK = 1
IMU_STATUS_OK = 1
KNOWN_IMU_LOCATIONS = frozenset((IMU_LOCATION_HIP, IMU_LOCATION_TRUNK))


def _finite_vector(
    value: Sequence[float], size: int, label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,):
        raise ValueError(
            f"{label} must have shape ({size},), got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return np.ascontiguousarray(result)


def _view_vector(view: Any, accessor: str, size: int) -> tuple[float, ...]:
    method = getattr(view, accessor)
    values = tuple(float(method(index)) for index in range(size))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{accessor} contains NaN or Inf")
    return values


def policy_state_to_state_slots(
    values: Sequence[float] | np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    expected = len(VITA_POLICY_STATE_TO_STATE_SLOTS)
    if array.ndim not in (1, 2) or array.shape[-1] != expected:
        raise ValueError(
            "policy state must have shape [30] or [frames, 30], "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("policy state contains NaN or Inf")
    expanded = np.full(
        (*array.shape[:-1], VITA_STATE_SLOT_COUNT),
        fill_value,
        dtype=np.float32,
    )
    expanded[..., np.asarray(VITA_POLICY_STATE_TO_STATE_SLOTS)] = array
    return np.ascontiguousarray(expanded)


@dataclass(frozen=True)
class VitaImuSample:
    location: int
    status: int
    quaternion_wxyz: np.ndarray
    gyroscope_rad_s: np.ndarray
    accelerometer_m_s2: np.ndarray
    rpy_deg: np.ndarray
    ts_sampled_us: int
    ts_acquired_hw_us: int
    ts_ingested_us: int
    ts_assembled_us: int
    device_timestamp_ms: int | None = None
    host_timestamp_us: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quaternion_wxyz",
            _finite_vector(self.quaternion_wxyz, 4, "quaternion_wxyz"),
        )
        object.__setattr__(
            self,
            "gyroscope_rad_s",
            _finite_vector(self.gyroscope_rad_s, 3, "gyroscope_rad_s"),
        )
        object.__setattr__(
            self,
            "accelerometer_m_s2",
            _finite_vector(self.accelerometer_m_s2, 3, "accelerometer_m_s2"),
        )
        object.__setattr__(
            self,
            "rpy_deg",
            _finite_vector(self.rpy_deg, 3, "rpy_deg"),
        )

    def normalized_quaternion_wxyz(self) -> np.ndarray:
        norm = float(np.linalg.norm(self.quaternion_wxyz))
        if not math.isfinite(norm) or norm < 1.0e-6:
            raise ValueError(
                f"IMU location {self.location} has a near-zero quaternion"
            )
        return np.ascontiguousarray(
            self.quaternion_wxyz / norm, dtype=np.float32
        )


@dataclass(frozen=True)
class VitaRawLowState:
    state_id: int
    ts_state_real_us: int
    ts_state_pub_us: int
    ts_consumed_us: int
    received_monotonic_ns: int
    attitude_valid: bool
    q_by_slot: np.ndarray
    dq_by_slot: np.ndarray
    tau_est_by_slot: np.ndarray
    temperature_by_slot: np.ndarray
    lost_by_slot: np.ndarray
    slot_role_by_slot: np.ndarray
    imu_samples: tuple[VitaImuSample, ...]
    state_id_bits: int = 16  # Legacy44 is uint16; public BODY is uint64.
    applied_session_id: int = 0
    applied_sequence: int = 0
    ts_first_output_us: int = 0
    joint_timestamps_us: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "q_by_slot",
            "dq_by_slot",
            "tau_est_by_slot",
            "temperature_by_slot",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(
                    getattr(self, field_name),
                    VITA_STATE_SLOT_COUNT,
                    field_name,
                ),
            )
        lost = np.asarray(self.lost_by_slot, dtype=np.int64).reshape(-1)
        roles = np.asarray(self.slot_role_by_slot, dtype=np.int64).reshape(-1)
        if lost.shape != (VITA_STATE_SLOT_COUNT,):
            raise ValueError("lost_by_slot must contain 44 values")
        if roles.shape != (VITA_STATE_SLOT_COUNT,):
            raise ValueError("slot_role_by_slot must contain 44 values")
        if np.any(lost < 0):
            raise ValueError("lost_by_slot contains a negative counter")
        object.__setattr__(self, "lost_by_slot", np.ascontiguousarray(lost))
        object.__setattr__(
            self, "slot_role_by_slot", np.ascontiguousarray(roles)
        )
        expected_roles = np.asarray(
            VITA_EXPECTED_STATE_SLOT_ROLES, dtype=np.int64
        )
        if not np.array_equal(roles, expected_roles):
            mismatches = np.flatnonzero(roles != expected_roles).tolist()
            raise ValueError(f"unexpected slot roles at slots {mismatches}")
        if self.state_id_bits not in (16, 64):
            raise ValueError("state_id_bits must be 16 or 64")
        if (self.state_id_bits == 64 and
                (isinstance(self.state_id, bool) or not isinstance(self.state_id, (int, np.integer)))):
            raise ValueError("state_id must be an integer")
        if not 0 <= self.state_id < (1 << self.state_id_bits):
            raise ValueError(f"state_id must fit uint{self.state_id_bits}")

    def observation_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        slots = np.asarray(VITA_POLICY_STATE_TO_STATE_SLOTS, dtype=np.int64)
        return (
            np.ascontiguousarray(self.q_by_slot[slots], dtype=np.float32),
            np.ascontiguousarray(self.dq_by_slot[slots], dtype=np.float32),
        )

    def dof_order_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        slots = np.asarray(VITA_DOF_TO_STATE_SLOTS, dtype=np.int64)
        return (
            np.ascontiguousarray(self.q_by_slot[slots], dtype=np.float32),
            np.ascontiguousarray(self.dq_by_slot[slots], dtype=np.float32),
        )

    def action_order_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        slots = np.asarray(VITA_ACTION_TO_COMMAND_SLOTS, dtype=np.int64)
        return (
            np.ascontiguousarray(self.q_by_slot[slots], dtype=np.float32),
            np.ascontiguousarray(self.dq_by_slot[slots], dtype=np.float32),
        )

    def require_imu(
        self, location: int, *, allowed_statuses: set[int] | None = None
    ) -> VitaImuSample:
        matches = [
            sample
            for sample in self.imu_samples
            if sample.location == location
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one IMU at location {location}, "
                f"found {len(matches)}"
            )
        sample = matches[0]
        if (
            allowed_statuses is not None
            and sample.status not in allowed_statuses
        ):
            raise RuntimeError(
                f"IMU location {location} has rejected status {sample.status}"
            )
        return sample


def decode_humanoid_lowstate_view(
    view: Any, *, received_monotonic_ns: int | None = None
) -> VitaRawLowState:
    motor_count = int(view.MotorStatesLength())
    if motor_count != VITA_STATE_SLOT_COUNT:
        raise ValueError(
            f"HumanoidLowState has {motor_count} motor slots; expected 44"
        )

    q = np.empty(VITA_STATE_SLOT_COUNT, dtype=np.float32)
    dq = np.empty(VITA_STATE_SLOT_COUNT, dtype=np.float32)
    tau_est = np.empty(VITA_STATE_SLOT_COUNT, dtype=np.float32)
    temperature = np.empty(VITA_STATE_SLOT_COUNT, dtype=np.float32)
    lost = np.empty(VITA_STATE_SLOT_COUNT, dtype=np.int64)
    slot_roles = np.empty(VITA_STATE_SLOT_COUNT, dtype=np.int64)
    seen_slots: set[int] = set()
    for vector_index in range(motor_count):
        motor = view.MotorStates(vector_index)
        if motor is None:
            raise ValueError(f"motor state {vector_index} is missing")
        slot = int(motor.SlotIndex())
        if slot < 0 or slot >= VITA_STATE_SLOT_COUNT:
            raise ValueError(f"motor state has out-of-range slot index {slot}")
        if slot in seen_slots:
            raise ValueError(f"duplicate motor slot index {slot}")
        seen_slots.add(slot)
        q[slot] = float(motor.Q())
        dq[slot] = float(motor.Dq())
        tau_est[slot] = float(motor.TauEst())
        temperature[slot] = float(motor.Temperature())
        lost[slot] = int(motor.Lost())
        slot_roles[slot] = int(motor.SlotRole())
    if seen_slots != set(range(VITA_STATE_SLOT_COUNT)):
        missing = sorted(set(range(VITA_STATE_SLOT_COUNT)) - seen_slots)
        raise ValueError(f"HumanoidLowState is missing motor slots {missing}")

    imu_samples = []
    imu_locations: set[int] = set()
    for index in range(int(view.ImuStatesLength())):
        sample = view.ImuStates(index)
        if sample is None:
            raise ValueError(f"IMU sample {index} is missing")
        state = sample.State()
        if state is None:
            raise ValueError(f"IMU sample {index} has no state")
        location = int(sample.Location())
        if location not in KNOWN_IMU_LOCATIONS:
            raise ValueError(f"unknown IMU location {location}")
        if location in imu_locations:
            raise ValueError(f"duplicate IMU location {location}")
        imu_locations.add(location)
        imu_samples.append(
            VitaImuSample(
                location=location,
                status=int(sample.Status()),
                quaternion_wxyz=_view_vector(state, "Quaternion", 4),
                gyroscope_rad_s=_view_vector(state, "Gyroscope", 3),
                accelerometer_m_s2=_view_vector(state, "Accelerometer", 3),
                rpy_deg=_view_vector(state, "Rpy", 3),
                ts_sampled_us=int(sample.TsSampledUs()),
                ts_acquired_hw_us=int(sample.TsAcquiredHwUs()),
                ts_ingested_us=int(sample.TsIngestedUs()),
                ts_assembled_us=int(sample.TsAssembledUs()),
                device_timestamp_ms=int(sample.DeviceTimestampMs()),
                host_timestamp_us=int(sample.HostTimestampUs()),
            )
        )
    if IMU_LOCATION_HIP not in imu_locations:
        raise ValueError("HumanoidLowState is missing the required HIP IMU")

    return VitaRawLowState(
        state_id=int(view.StateId()),
        ts_state_real_us=int(view.TsStateReal()),
        ts_state_pub_us=int(view.TsStatePub()),
        ts_consumed_us=int(view.TsConsumedUs()),
        received_monotonic_ns=(
            time.monotonic_ns()
            if received_monotonic_ns is None
            else int(received_monotonic_ns)
        ),
        attitude_valid=bool(view.AttitudeValid()),
        q_by_slot=q,
        dq_by_slot=dq,
        tau_est_by_slot=tau_est,
        temperature_by_slot=temperature,
        lost_by_slot=lost,
        slot_role_by_slot=slot_roles,
        imu_samples=tuple(imu_samples),
    )


def decode_humanoid_lowstate_buffer(
    payload: bytes | bytearray | memoryview,
    *,
    received_monotonic_ns: int | None = None,
) -> VitaRawLowState:
    from lowlevel.HumanoidLowState import HumanoidLowState

    view = HumanoidLowState.GetRootAs(payload, 0)
    return decode_humanoid_lowstate_view(
        view, received_monotonic_ns=received_monotonic_ns
    )


__all__ = [
    "IMU_LOCATION_HIP",
    "IMU_LOCATION_TRUNK",
    "IMU_STATUS_OK",
    "VitaImuSample",
    "VitaRawLowState",
    "decode_humanoid_lowstate_buffer",
    "decode_humanoid_lowstate_view",
    "policy_state_to_state_slots",
]
