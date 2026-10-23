"""Offline adapters for recorded Vita LowState shadow evaluation."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_EXPECTED_STATE_SLOT_ROLES,
    VITA_POLICY_STATE_TO_STATE_SLOTS,
    VITA_STATE_SLOT_COUNT,
)
from deployment.vita_s100_aorta_30dof.state_adapter import (
    IMU_LOCATION_HIP,
    IMU_STATUS_OK,
)

LOWSTATE_TRACE_SCHEMA = "vita_s100_lowstate_trace_v1"
IMU_ALIGNMENT_SCHEMA = "vita02a.imu_alignment.v1"
ROBOT_ALIGNMENT_SCOPES = frozenset(
    ("robot-readonly-reviewed", "robot-active-approved")
)


def load_lowstate_trace(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        trace = {name: np.asarray(payload[name]) for name in payload.files}
    validate_lowstate_trace(trace)
    return trace


def trace_metadata(trace: Mapping[str, np.ndarray]) -> dict[str, Any]:
    value = np.asarray(trace["metadata"]).reshape(()).item()
    metadata = json.loads(str(value))
    if not isinstance(metadata, dict):
        raise ValueError("LowState trace metadata must be an object")
    return metadata


def alignment_quaternion_from_contract(
    alignment: Mapping[str, Any], lowstate_metadata: Mapping[str, Any]
) -> np.ndarray:
    if alignment.get("schema") != IMU_ALIGNMENT_SCHEMA:
        raise ValueError("unsupported IMU alignment schema")
    if alignment.get("imu_location") != "HIP":
        raise ValueError("only explicit HIP IMU alignment is supported")
    if alignment.get("imu_quaternion_semantics") != "world_from_imu":
        raise ValueError("IMU quaternion semantics must be world_from_imu")
    if alignment.get("gyroscope_frame") != "imu":
        raise ValueError("gyroscope frame must be imu")

    scope = alignment.get("scope")
    source_kind = lowstate_metadata.get("source_kind")
    if source_kind == "mujoco_policy_trace_fixture":
        if scope != "simulation-only":
            raise ValueError(
                "MuJoCo fixture requires a simulation-only IMU alignment"
            )
    else:
        if scope not in ROBOT_ALIGNMENT_SCOPES:
            raise ValueError(
                "robot trace requires a reviewed robot IMU alignment"
            )
        approval = alignment.get("approval")
        if not isinstance(approval, dict) or not approval.get(
            "offline_recorded_trace_shadow"
        ):
            raise ValueError(
                "robot alignment is not approved for recorded-trace shadow"
            )
    return normalize_quaternion_wxyz(
        np.asarray(
            alignment["base_from_imu_quaternion_wxyz"], dtype=np.float32
        )
    )


def validate_lowstate_trace(trace: Mapping[str, np.ndarray]) -> None:
    required = {
        "metadata": (),
        "imu_locations": (2,),
        "state_id": None,
        "ts_state_real_us": None,
        "received_monotonic_ns": None,
        "attitude_valid": None,
        "q_by_slot": (VITA_STATE_SLOT_COUNT,),
        "dq_by_slot": (VITA_STATE_SLOT_COUNT,),
        "tau_est_by_slot": (VITA_STATE_SLOT_COUNT,),
        "temperature_by_slot": (VITA_STATE_SLOT_COUNT,),
        "lost_by_slot": (VITA_STATE_SLOT_COUNT,),
        "slot_role_by_slot": (VITA_STATE_SLOT_COUNT,),
        "imu_present": (2,),
        "imu_status": (2,),
        "imu_quaternion_wxyz": (2, 4),
        "imu_gyroscope_rad_s": (2, 3),
        "imu_accelerometer_m_s2": (2, 3),
        "imu_rpy_deg": (2, 3),
    }
    missing = sorted(set(required) - set(trace))
    if missing:
        raise ValueError(f"LowState trace is missing fields: {missing}")
    metadata = trace_metadata(trace)
    if metadata.get("schema_version") != LOWSTATE_TRACE_SCHEMA:
        raise ValueError(
            "unsupported LowState trace schema: "
            f"{metadata.get('schema_version')!r}"
        )

    frames = int(np.asarray(trace["state_id"]).shape[0])
    if frames <= 0:
        raise ValueError("LowState trace contains no frames")
    for name, trailing_shape in required.items():
        if trailing_shape is None or name == "metadata":
            continue
        actual = np.asarray(trace[name]).shape
        expected = (
            trailing_shape
            if name == "imu_locations"
            else (frames, *trailing_shape)
        )
        if actual != expected:
            raise ValueError(f"{name} has shape {actual}, expected {expected}")
    for name in (
        "ts_state_real_us",
        "received_monotonic_ns",
        "attitude_valid",
    ):
        if np.asarray(trace[name]).shape != (frames,):
            raise ValueError(f"{name} must contain one value per frame")

    locations = np.asarray(trace["imu_locations"], dtype=np.int64)
    hip_matches = np.flatnonzero(locations == IMU_LOCATION_HIP)
    if hip_matches.size != 1:
        raise ValueError("LowState trace must contain one HIP IMU column")
    expected_roles = np.asarray(VITA_EXPECTED_STATE_SLOT_ROLES, dtype=np.int64)
    roles = np.asarray(trace["slot_role_by_slot"], dtype=np.int64)
    if not np.all(roles == expected_roles[None, :]):
        raise ValueError(
            "LowState trace slot roles do not match the Vita contract"
        )


def select_policy_frames(
    trace: Mapping[str, np.ndarray], *, policy_hz: float = 50.0
) -> tuple[np.ndarray, str]:
    if not math.isfinite(policy_hz) or policy_hz <= 0.0:
        raise ValueError("policy_hz must be positive")
    state_us = np.asarray(trace["ts_state_real_us"], dtype=np.int64)
    arrival_ns = np.asarray(trace["received_monotonic_ns"], dtype=np.int64)
    if np.all(state_us > 0) and np.all(np.diff(state_us) > 0):
        timeline_ns = state_us * 1000
        source = "ts_state_real_us"
    elif np.all(np.diff(arrival_ns) > 0):
        timeline_ns = arrival_ns
        source = "received_monotonic_ns"
    else:
        raise ValueError("LowState trace has no strictly increasing time base")

    period_ns = int(round(1_000_000_000.0 / policy_hz))
    targets = np.arange(
        int(timeline_ns[0]),
        int(timeline_ns[-1]) + 1,
        period_ns,
        dtype=np.int64,
    )
    indices = np.searchsorted(timeline_ns, targets, side="right") - 1
    indices = np.unique(np.clip(indices, 0, timeline_ns.size - 1))
    return np.ascontiguousarray(indices, dtype=np.int64), source


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("quaternion has zero or non-finite norm")
    return np.asarray(value / norm, dtype=np.float32)


def quaternion_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.asarray([w, -x, -y, -z], dtype=np.float32)


def quaternion_multiply_wxyz(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_w, left_x, left_y, left_z = normalize_quaternion_wxyz(left)
    right_w, right_x, right_y, right_z = normalize_quaternion_wxyz(right)
    return normalize_quaternion_wxyz(
        np.asarray(
            [
                left_w * right_w
                - left_x * right_x
                - left_y * right_y
                - left_z * right_z,
                left_w * right_x
                + left_x * right_w
                + left_y * right_z
                - left_z * right_y,
                left_w * right_y
                - left_x * right_z
                + left_y * right_w
                + left_z * right_x,
                left_w * right_z
                + left_x * right_y
                - left_y * right_x
                + left_z * right_w,
            ],
            dtype=np.float32,
        )
    )


def rotate_vector_wxyz(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    rotation = np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(
        rotation @ np.asarray(vector, dtype=np.float32)
    )


def policy_state_from_trace(
    trace: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    base_from_hip_quaternion_wxyz: np.ndarray,
) -> dict[str, np.ndarray]:
    locations = np.asarray(trace["imu_locations"], dtype=np.int64)
    hip_column = int(np.flatnonzero(locations == IMU_LOCATION_HIP)[0])
    present = np.asarray(trace["imu_present"], dtype=np.bool_)[
        indices, hip_column
    ]
    statuses = np.asarray(trace["imu_status"], dtype=np.int64)[
        indices, hip_column
    ]
    attitudes = np.asarray(trace["attitude_valid"], dtype=np.bool_)[indices]
    if not np.all(present):
        raise ValueError("selected LowState frames contain a missing HIP IMU")
    if not np.all(statuses == IMU_STATUS_OK):
        rejected = sorted(
            set(int(value) for value in statuses if value != IMU_STATUS_OK)
        )
        raise ValueError(
            f"selected HIP IMU frames have rejected statuses {rejected}"
        )
    if not np.all(attitudes):
        raise ValueError("selected LowState frames contain invalid attitude")

    policy_state_slots = np.asarray(
        VITA_POLICY_STATE_TO_STATE_SLOTS, dtype=np.int64
    )
    q = np.asarray(trace["q_by_slot"], dtype=np.float32)[indices][
        :, policy_state_slots
    ]
    dq = np.asarray(trace["dq_by_slot"], dtype=np.float32)[indices][
        :, policy_state_slots
    ]
    hip_quaternions = np.asarray(
        trace["imu_quaternion_wxyz"], dtype=np.float32
    )[indices, hip_column]
    hip_gyroscope = np.asarray(trace["imu_gyroscope_rad_s"], dtype=np.float32)[
        indices, hip_column
    ]
    base_from_hip = normalize_quaternion_wxyz(base_from_hip_quaternion_wxyz)
    hip_from_base = quaternion_inverse_wxyz(base_from_hip)
    root_quaternions = np.stack(
        [
            quaternion_multiply_wxyz(value, hip_from_base)
            for value in hip_quaternions
        ]
    )
    root_angular_velocity = np.stack(
        [rotate_vector_wxyz(base_from_hip, value) for value in hip_gyroscope]
    )
    arrays = {
        "source_frame_index": np.asarray(indices, dtype=np.int64),
        "state_id": np.asarray(trace["state_id"], dtype=np.uint16)[indices],
        "robot_root_quat_wxyz": root_quaternions,
        "robot_root_angvel_local": root_angular_velocity,
        "robot_dof_pos": np.ascontiguousarray(q),
        "robot_dof_vel": np.ascontiguousarray(dq),
    }
    nonfinite = [
        name
        for name, values in arrays.items()
        if values.dtype.kind in "fc" and not np.isfinite(values).all()
    ]
    if nonfinite:
        raise ValueError(
            f"policy state contains non-finite arrays: {nonfinite}"
        )
    return arrays


__all__ = [
    "IMU_ALIGNMENT_SCHEMA",
    "LOWSTATE_TRACE_SCHEMA",
    "ROBOT_ALIGNMENT_SCOPES",
    "alignment_quaternion_from_contract",
    "load_lowstate_trace",
    "normalize_quaternion_wxyz",
    "policy_state_from_trace",
    "quaternion_inverse_wxyz",
    "quaternion_multiply_wxyz",
    "rotate_vector_wxyz",
    "select_policy_frames",
    "trace_metadata",
    "validate_lowstate_trace",
]
