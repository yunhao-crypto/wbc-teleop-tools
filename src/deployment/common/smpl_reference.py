"""Stable import boundary for the mature G1 pure-SMPL CUDA implementation."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any


def _load_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Load installed classes or the source-tree compatibility location."""

    try:
        from humanoid_policy.smpl_motion_observation import (
            GpuSmplMotionObservationBuilder,
        )
        from humanoid_policy.smpl_reference import (
            GpuOfflineSmplReference,
            GpuOnlineSmplReference,
        )
    except ModuleNotFoundError:
        source_root = (
            Path(__file__).resolve().parents[1]
            / "unitree_g1_ros2_29dof"
            / "src"
        )
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from humanoid_policy.smpl_motion_observation import (
            GpuSmplMotionObservationBuilder,
        )
        from humanoid_policy.smpl_reference import (
            GpuOfflineSmplReference,
            GpuOnlineSmplReference,
        )
    return (
        GpuSmplMotionObservationBuilder,
        GpuOnlineSmplReference,
        GpuOfflineSmplReference,
    )


def create_motion_observation_builder(**kwargs: Any) -> Any:
    builder, _online, _offline = _load_classes()
    return builder(**kwargs)


def create_online_smpl_reference(**kwargs: Any) -> Any:
    _builder, online, _offline = _load_classes()
    return online(**kwargs)


def create_offline_smpl_reference(*args: Any, **kwargs: Any) -> Any:
    _builder, _online, offline = _load_classes()
    return offline(*args, **kwargs)


__all__ = [
    "create_motion_observation_builder",
    "create_offline_smpl_reference",
    "create_online_smpl_reference",
]
