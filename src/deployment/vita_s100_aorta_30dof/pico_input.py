"""Unified local XRoboToolkit/Pico source for Vita02A."""

from __future__ import annotations
from typing import Any


def create_pico_source(*, logger: Any, maximum_age: float,
                       duplicate_poll_interval_s: float = 0.0001) -> Any:
    from humanoid_policy.local_smpl_source import LocalPicoSmplSource

    return LocalPicoSmplSource(logger=logger, max_source_age=maximum_age,
                               duplicate_poll_interval_s=duplicate_poll_interval_s)


def store_latest_pico(source: Any, reference: Any) -> bool:
    result = source.step()
    if result is None:
        return False
    body_poses, frame_index, metadata = result
    stored = reference.store_pico(
        body_poses,
        sample_time=float(metadata["timestamp_ns"]) * 1.0e-9,
        current_time=float(metadata.get("timestamp_realtime", 0.0)) or None,
        frame_index=int(frame_index),
    )
    if not stored:
        raise RuntimeError("Pico produced an invalid or non-monotonic frame")
    return True


__all__ = ["create_pico_source", "store_latest_pico"]
