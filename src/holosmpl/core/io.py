from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holosmpl.core.processing.derived_fields import derive_formal_fields
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
)
from holosmpl.core.validation.formal_check import (
    FORBIDDEN_FORMAL_FIELDS,
    REQUIRED_FORMAL_FIELDS,
    ROBOT_LIKE_KEYWORDS,
)


@dataclass(frozen=True)
class FormalClip:
    """One validated HoloSMPL formal NPZ clip."""

    path: Path
    human_pose_aa: Any
    human_shape_beta: Any
    human_root_trans: Any
    human_root_height: Any
    human_gravity_projection: Any
    metadata: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return int(self.human_pose_aa.shape[0])

    @property
    def target_fps(self) -> float:
        return float(self.metadata["target_fps"])


def load_formal_clip(path: str | Path) -> FormalClip:
    """Load and validate one HoloSMPL ``formal_human_npz_v1`` clip."""

    import numpy as np

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        fields = set(data.files)
        missing = sorted(REQUIRED_FORMAL_FIELDS - fields)
        forbidden = sorted(fields & FORBIDDEN_FORMAL_FIELDS)
        robot_like = [
            field
            for field in fields
            if any(keyword in field.lower() for keyword in ROBOT_LIKE_KEYWORDS)
        ]
        if missing:
            raise ValueError(
                f"HoloSMPL formal NPZ is missing fields: {missing}"
            )
        if forbidden:
            raise ValueError(
                f"canonical fields leaked into formal NPZ: {forbidden}"
            )
        if robot_like:
            raise ValueError(
                f"robot-like fields in HoloSMPL formal NPZ: {robot_like}"
            )
        metadata = json.loads(str(data["metadata"].item()))
        clip = FormalClip(
            path=path,
            human_pose_aa=np.asarray(data["human_pose_aa"]),
            human_shape_beta=np.asarray(data["human_shape_beta"]),
            human_root_trans=np.asarray(data["human_root_trans"]),
            human_root_height=np.asarray(data["human_root_height"]),
            human_gravity_projection=np.asarray(
                data["human_gravity_projection"]
            ),
            metadata=metadata,
        )

    arrays = {
        "human_pose_aa": (clip.human_pose_aa, 72),
        "human_root_trans": (clip.human_root_trans, 3),
        "human_root_height": (clip.human_root_height, 1),
        "human_gravity_projection": (
            clip.human_gravity_projection,
            3,
        ),
    }
    for name, (array, width) in arrays.items():
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must be [T,{width}], got {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(
                f"{name} dtype must be float32, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if clip.human_shape_beta.ndim != 1 or clip.human_shape_beta.size <= 0:
        raise ValueError(
            "human_shape_beta must be a non-empty clip-level [B] array"
        )
    if clip.human_shape_beta.dtype != np.float32:
        raise ValueError(
            "human_shape_beta dtype must be float32, got "
            f"{clip.human_shape_beta.dtype}"
        )
    if not np.isfinite(clip.human_shape_beta).all():
        raise ValueError("human_shape_beta contains NaN or Inf")
    if clip.n_frames <= 0:
        raise ValueError("HoloSMPL formal clip has zero frames")
    for name, (array, _) in arrays.items():
        if int(array.shape[0]) != clip.n_frames:
            raise ValueError(
                f"{name} frame count {array.shape[0]} != {clip.n_frames}"
            )

    metadata = clip.metadata
    if metadata.get("schema_version") != "formal_human_npz_v1":
        raise ValueError("metadata schema_version must be formal_human_npz_v1")
    if metadata.get("canonical_coordinate_system") != (
        CANONICAL_COORDINATE_FRAME
    ):
        raise ValueError("formal NPZ coordinate system is not canonical")
    if metadata.get("up_axis") != CANONICAL_UP_AXIS:
        raise ValueError("formal NPZ up_axis is not canonical")
    if metadata.get("unit") != CANONICAL_UNIT:
        raise ValueError("formal NPZ unit is not canonical")
    if metadata.get("slice_policy") != "none":
        raise ValueError("metadata slice_policy must be none")
    if metadata.get("formal_pose_72_policy") is None:
        raise ValueError("metadata missing formal_pose_72_policy")
    if int(metadata.get("frame_count", -1)) != clip.n_frames:
        raise ValueError("metadata frame_count mismatch")
    if clip.target_fps <= 0.0:
        raise ValueError("metadata target_fps must be positive")

    derived = derive_formal_fields(clip.human_pose_aa, clip.human_root_trans)
    if not np.allclose(
        clip.human_root_height,
        derived["human_root_height"],
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise ValueError("human_root_height does not match human_root_trans")
    if not np.allclose(
        clip.human_gravity_projection,
        derived["human_gravity_projection"],
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise ValueError(
            "human_gravity_projection does not match human_pose_aa"
        )
    return clip


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
