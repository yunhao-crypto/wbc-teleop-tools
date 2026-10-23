"""Shared freshness and fail-closed primitives for digital-twin runtimes."""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FreshnessGate:
    maximum_age: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_age) or self.maximum_age <= 0.0:
            raise ValueError("maximum_age must be finite and positive")

    def require(self, *, sample_time: float, now: float) -> float:
        age = float(now) - float(sample_time)
        if not math.isfinite(age) or age < 0.0:
            raise RuntimeError(f"invalid sample age: {age}")
        if age > self.maximum_age:
            raise RuntimeError(
                f"sample is stale: {age:.3f}s > {self.maximum_age:.3f}s"
            )
        return age


__all__ = ["FreshnessGate"]
