"""Policy-order robot state and target helpers for Vita02A."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_JOINT_NAMES,
)

VITA_CONTROLLER_JOINT_NAMES = VITA_ACTION_JOINT_NAMES


def _finite(values: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{size}]")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class VitaRobotState:
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat_wxyz: np.ndarray
    root_ang_vel_b: np.ndarray
    state_id: int = 0
    received_monotonic_ns: int = 0
    attitude_valid: bool = True

    def __post_init__(self) -> None:
        if self.joint_names != VITA_CONTROLLER_JOINT_NAMES:
            raise ValueError(
                "Vita robot state joint order is not the 30-DoF contract"
            )
        object.__setattr__(
            self, "joint_pos", _finite(self.joint_pos, 30, "joint_pos")
        )
        object.__setattr__(
            self, "joint_vel", _finite(self.joint_vel, 30, "joint_vel")
        )
        quaternion = _finite(self.root_quat_wxyz, 4, "root_quat_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-6:
            raise ValueError("root quaternion has zero norm")
        object.__setattr__(self, "root_quat_wxyz", quaternion / norm)
        object.__setattr__(
            self,
            "root_ang_vel_b",
            _finite(self.root_ang_vel_b, 3, "root_ang_vel_b"),
        )

    def _indices(self, names: Sequence[str]) -> np.ndarray:
        index = {
            name: position for position, name in enumerate(self.joint_names)
        }
        try:
            return np.asarray([index[name] for name in names], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"unknown Vita joint {exc.args[0]!r}") from exc

    def select_joint_pos(self, names: Sequence[str]) -> np.ndarray:
        return np.ascontiguousarray(self.joint_pos[self._indices(names)])

    def select_joint_vel(self, names: Sequence[str]) -> np.ndarray:
        return np.ascontiguousarray(self.joint_vel[self._indices(names)])


def values_in_order(
    source_names: Sequence[str],
    values: Sequence[float],
    target_names: Sequence[str],
) -> np.ndarray:
    source = tuple(source_names)
    array = _finite(values, len(source), "joint metadata")
    mapping = dict(zip(source, array, strict=True))
    try:
        return np.asarray(
            [mapping[name] for name in target_names], dtype=np.float32
        )
    except KeyError as exc:
        raise ValueError(f"joint metadata is missing {exc.args[0]!r}") from exc


__all__ = ["VITA_CONTROLLER_JOINT_NAMES", "VitaRobotState", "values_in_order"]
