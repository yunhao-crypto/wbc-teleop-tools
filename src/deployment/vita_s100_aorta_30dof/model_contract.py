"""Exact ONNX ABI contracts for the Vita02A policy pair."""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from deployment.common.model_contract import ModelBundle, verify_model_bundle
from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_JOINT_NAMES,
    VITA_OBSERVATION_JOINT_NAMES,
)

VITA_MOTION_OBSERVATION_DIM = 3126
VITA_MOTION_ACTION_DIM = 30
VITA_MOTION_KV_SHAPE = (1, 2, 1, 32, 4, 64)
VITA_VELOCITY_OBSERVATION_DIM = 990
VITA_UNIFIED_VELOCITY_OBSERVATION_DIM = 752
VITA_VELOCITY_ACTION_DIM = 28
VITA_VELOCITY_JOINT_NAMES = VITA_ACTION_JOINT_NAMES[:28]


@dataclass(frozen=True)
class VitaPolicyContract:
    kind: str
    observation_name: str
    observation_shape: tuple[int | str | None, ...]
    observation_dtype: np.dtype
    action_name: str
    action_shape: tuple[int | str | None, ...]
    action_dtype: np.dtype
    joint_names: tuple[str, ...]
    action_joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    metadata: Mapping[str, str]
    kv_input_name: str | None = None
    kv_output_name: str | None = None
    step_index_name: str | None = None
    rope_max_seq_len: int | None = None


@dataclass(frozen=True)
class VitaBundleContracts:
    motion: VitaPolicyContract
    velocity: VitaPolicyContract


def _shape(value: Any) -> tuple[int | str | None, ...]:
    return tuple(value.shape)


def _dtype(value: Any) -> np.dtype:
    values = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(int64)": np.dtype(np.int64),
    }
    try:
        return values[str(value.type)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ONNX tensor type {value.type!r}"
        ) from exc


def _names(metadata: Mapping[str, str], key: str) -> tuple[str, ...]:
    if key not in metadata:
        raise ValueError(f"ONNX metadata is missing {key!r}")
    result = tuple(part.strip() for part in metadata[key].split(","))
    if not result or not all(result) or len(set(result)) != len(result):
        raise ValueError(f"ONNX metadata {key!r} is invalid")
    return result


def _floats(
    metadata: Mapping[str, str], key: str, expected: int
) -> np.ndarray:
    if key not in metadata:
        raise ValueError(f"ONNX metadata is missing {key!r}")
    result = np.asarray(
        [float(part.strip()) for part in metadata[key].split(",")],
        dtype=np.float32,
    )
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError(
            f"ONNX metadata {key!r} must contain {expected} values"
        )
    return result


def _require_tensor(
    tensor: Any,
    *,
    shape: Sequence[int],
    dtypes: Sequence[np.dtype],
    label: str,
) -> None:
    actual_shape = _shape(tensor)
    actual_dtype = _dtype(tensor)
    allowed = tuple(np.dtype(value) for value in dtypes)
    if actual_shape != tuple(shape):
        raise ValueError(
            f"{label} shape is {actual_shape}, expected {tuple(shape)}"
        )
    if actual_dtype not in allowed:
        raise ValueError(
            f"{label} dtype is {actual_dtype}, expected {allowed}"
        )


def inspect_vita_session(session: Any, kind: str) -> VitaPolicyContract:
    """Inspect one loaded ONNX Runtime session without robot I/O."""

    if kind not in {"motion", "velocity"}:
        raise ValueError(f"unsupported policy kind {kind!r}")
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    obs = inputs.get("obs")
    actions = outputs.get("actions")
    if obs is None or actions is None:
        raise ValueError(f"{kind} model must expose obs/actions")

    if kind == "motion":
        joint_names = _names(metadata, "joint_names")
        action_names = _names(metadata, "action_joint_names")
        if joint_names != VITA_OBSERVATION_JOINT_NAMES:
            raise ValueError(
                "motion joint_names do not match Vita observation order"
            )
        if action_names != VITA_ACTION_JOINT_NAMES:
            raise ValueError(
                "motion action_joint_names do not match Vita action order"
            )
        _require_tensor(
            obs,
            shape=(1, VITA_MOTION_OBSERVATION_DIM),
            dtypes=(np.float16, np.float32),
            label="motion obs",
        )
        _require_tensor(
            actions,
            shape=(1, VITA_MOTION_ACTION_DIM),
            dtypes=(_dtype(obs),),
            label="motion actions",
        )
        kv_input = inputs.get("past_key_values")
        kv_output = outputs.get("present_key_values")
        step_index = inputs.get("step_idx")
        if kv_input is None or kv_output is None or step_index is None:
            raise ValueError("motion model is missing KV-cache tensors")
        _require_tensor(
            kv_input,
            shape=VITA_MOTION_KV_SHAPE,
            dtypes=(_dtype(obs),),
            label="motion past_key_values",
        )
        _require_tensor(
            kv_output,
            shape=VITA_MOTION_KV_SHAPE,
            dtypes=(_dtype(obs),),
            label="motion present_key_values",
        )
        _require_tensor(
            step_index,
            shape=(1,),
            dtypes=(np.int64,),
            label="motion step_idx",
        )
        rope_max = int(metadata.get("rope_max_seq_len", "0"))
        if rope_max <= 0:
            raise ValueError("motion rope_max_seq_len must be positive")
        extra = dict(
            kv_input_name=kv_input.name,
            kv_output_name=kv_output.name,
            step_index_name=step_index.name,
            rope_max_seq_len=rope_max,
        )
    elif _shape(obs) == (1, VITA_UNIFIED_VELOCITY_OBSERVATION_DIM):
        joint_names = _names(metadata, "joint_names")
        action_names = _names(metadata, "action_joint_names")
        if joint_names != VITA_OBSERVATION_JOINT_NAMES:
            raise ValueError(
                "unified velocity joint_names do not match "
                "Vita observation order"
            )
        if action_names != VITA_VELOCITY_JOINT_NAMES:
            raise ValueError(
                "unified velocity action_joint_names do not match "
                "Vita 28-DoF order"
            )
        physics = json.loads(metadata.get("joint_physics_contract", "{}"))
        if (
            physics.get("joint_friction") != "zero"
            or physics.get("passive_damping") != 0.0
        ):
            raise ValueError(
                "unified velocity requires zero friction and passive damping"
            )
        _require_tensor(
            obs,
            shape=(1, VITA_UNIFIED_VELOCITY_OBSERVATION_DIM),
            dtypes=(np.float16, np.float32),
            label="unified velocity obs",
        )
        _require_tensor(
            actions,
            shape=(1, VITA_VELOCITY_ACTION_DIM),
            dtypes=(_dtype(obs),),
            label="unified velocity actions",
        )
        extra = {}
    else:
        if metadata.get("policy_format") != "tienkung_lab_vita02a_velocity_v1":
            raise ValueError("velocity policy_format is not Vita02A v1")
        if metadata.get("contract_version") != "2":
            raise ValueError("velocity contract_version must be 2")
        joint_names = _names(metadata, "joint_names")
        action_names = joint_names
        if joint_names != VITA_VELOCITY_JOINT_NAMES:
            raise ValueError(
                "velocity joint_names do not match Vita 28-DoF order"
            )
        _require_tensor(
            obs,
            shape=(1, VITA_VELOCITY_OBSERVATION_DIM),
            dtypes=(np.float16, np.float32),
            label="velocity obs",
        )
        _require_tensor(
            actions,
            shape=(1, VITA_VELOCITY_ACTION_DIM),
            dtypes=(_dtype(obs),),
            label="velocity actions",
        )
        if float(metadata.get("sim_dof_frictionloss", "nan")) != 0.0:
            raise ValueError(
                "velocity policy was not exported with zero joint friction"
            )
        if int(metadata.get("history_length", "0")) != 10:
            raise ValueError("velocity history_length must be 10")
        if metadata.get("history_order") != "oldest_to_newest":
            raise ValueError("velocity history_order is unsupported")
        if metadata.get("history_reset") != "backfill_first_frame":
            raise ValueError("velocity history_reset is unsupported")
        extra = {}

    stiffness = _floats(metadata, "joint_stiffness", len(joint_names))
    damping = _floats(metadata, "joint_damping", len(joint_names))
    action_scale = _floats(metadata, "action_scale", len(action_names))
    if np.any(stiffness <= 0.0) or np.any(damping <= 0.0):
        raise ValueError(f"{kind} PD gains must be positive")
    if np.any(action_scale <= 0.0):
        raise ValueError(f"{kind} action_scale must be positive")
    return VitaPolicyContract(
        kind=kind,
        observation_name=obs.name,
        observation_shape=_shape(obs),
        observation_dtype=_dtype(obs),
        action_name=actions.name,
        action_shape=_shape(actions),
        action_dtype=_dtype(actions),
        joint_names=joint_names,
        action_joint_names=action_names,
        default_joint_pos=_floats(
            metadata, "default_joint_pos", len(joint_names)
        ),
        joint_stiffness=stiffness,
        joint_damping=damping,
        action_scale=action_scale,
        metadata=metadata,
        **extra,
    )


def inspect_vita_bundle(
    bundle: ModelBundle,
    *,
    motion_session: Any,
    velocity_session: Any,
    verify_files: bool = True,
) -> VitaBundleContracts:
    if bundle.robot != "vita02a_s100_30dof":
        raise ValueError(f"expected vita02a_s100_30dof, got {bundle.robot}")
    if abs(bundle.policy_hz - 50.0) > 1.0e-6:
        raise ValueError("Vita policies must run at 50 Hz")
    if verify_files:
        verify_model_bundle(bundle, require_trained=False)
    return VitaBundleContracts(
        motion=inspect_vita_session(motion_session, "motion"),
        velocity=inspect_vita_session(velocity_session, "velocity"),
    )


__all__ = [
    "VITA_MOTION_ACTION_DIM",
    "VITA_MOTION_KV_SHAPE",
    "VITA_MOTION_OBSERVATION_DIM",
    "VITA_VELOCITY_ACTION_DIM",
    "VITA_VELOCITY_JOINT_NAMES",
    "VITA_VELOCITY_OBSERVATION_DIM",
    "VitaBundleContracts",
    "VitaPolicyContract",
    "inspect_vita_bundle",
    "inspect_vita_session",
]
