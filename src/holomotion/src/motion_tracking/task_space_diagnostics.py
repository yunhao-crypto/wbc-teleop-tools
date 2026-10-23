"""Windowed diagnostics for task-space tracking rewards.

The reward functions record one physical error and one bounded kernel value
per environment.  This module keeps the hot path on-device, aggregates exact
means over every environment-step in a logging window, and retains a small
uniform environment sample for a pooled P95 estimate.  No diagnostic changes
the reward returned to IsaacLab.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

import torch

TASK_SPACE_DIAGNOSTICS_ATTRIBUTE = "_holomotion_task_space_diagnostics"
TASK_SPACE_DIAGNOSTIC_SPECS = (
    ("semantic_position", "m"),
    ("semantic_rotation", "rad"),
    ("root_linear_velocity", "mps"),
    ("root_angular_velocity", "radps"),
    ("foot_precision_position", "m"),
    ("wrist_precision_position", "m"),
    ("link_linear_velocity", "mps"),
    ("link_angular_velocity", "radps"),
)
TASK_SPACE_DIAGNOSTIC_INDEX = {
    name: index
    for index, (name, _unit) in enumerate(TASK_SPACE_DIAGNOSTIC_SPECS)
}


@dataclass(frozen=True)
class TaskSpaceDiagnosticsSnapshot:
    """On-device statistics collected over one TensorBoard log window."""

    error_sums: torch.Tensor
    kernel_sums: torch.Tensor
    counts: torch.Tensor
    sampled_errors: torch.Tensor


class TaskSpaceDiagnosticsAccumulator:
    """Accumulate diagnostics without host synchronization."""

    def __init__(
        self,
        *,
        num_envs: int,
        window_steps: int,
        device: torch.device | str,
        sample_envs_per_rank: int = 256,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if window_steps <= 0:
            raise ValueError(
                f"window_steps must be positive, got {window_steps}."
            )
        if sample_envs_per_rank <= 0:
            raise ValueError(
                "sample_envs_per_rank must be positive, got "
                f"{sample_envs_per_rank}."
            )

        self.num_envs = int(num_envs)
        self.window_steps = int(window_steps)
        self.device = torch.device(device)
        self.sample_envs_per_rank = min(
            int(sample_envs_per_rank),
            self.num_envs,
        )

        num_terms = len(TASK_SPACE_DIAGNOSTIC_SPECS)
        self._error_sums = torch.zeros(
            num_terms,
            device=self.device,
            dtype=torch.float32,
        )
        self._kernel_sums = torch.zeros_like(self._error_sums)
        self._counts = torch.zeros_like(self._error_sums)

        # Midpoint sampling covers the full environment index range without
        # calling a random generator or perturbing training RNG state.
        sample_numbers = torch.arange(
            self.sample_envs_per_rank,
            device=self.device,
            dtype=torch.long,
        )
        self._sample_indices = torch.div(
            (2 * sample_numbers + 1) * self.num_envs,
            2 * self.sample_envs_per_rank,
            rounding_mode="floor",
        ).clamp_max(self.num_envs - 1)

        sample_capacity = self.window_steps * self.sample_envs_per_rank
        self._sampled_errors = torch.empty(
            (num_terms, sample_capacity),
            device=self.device,
            dtype=torch.float32,
        )
        self._sample_offsets = [0] * num_terms

    @torch.no_grad()
    def record(
        self,
        *,
        name: str,
        error: torch.Tensor,
        kernel: torch.Tensor,
    ) -> None:
        """Record one reward term for all environments in the current step."""

        try:
            term_index = TASK_SPACE_DIAGNOSTIC_INDEX[name]
        except KeyError as error_key:
            raise KeyError(
                f"Unknown task-space diagnostic term {name!r}."
            ) from error_key

        error = error.detach().reshape(-1)
        kernel = kernel.detach().reshape(-1)
        expected_shape = (self.num_envs,)
        if tuple(error.shape) != expected_shape:
            raise ValueError(
                f"{name} error must have shape {expected_shape}, "
                f"got {tuple(error.shape)}."
            )
        if tuple(kernel.shape) != expected_shape:
            raise ValueError(
                f"{name} kernel must have shape {expected_shape}, "
                f"got {tuple(kernel.shape)}."
            )
        if error.device != self.device or kernel.device != self.device:
            raise ValueError(
                f"{name} diagnostics must remain on {self.device}; got "
                f"{error.device} and {kernel.device}."
            )

        error_float = error.float()
        kernel_float = kernel.float()
        self._error_sums[term_index].add_(error_float.sum())
        self._kernel_sums[term_index].add_(kernel_float.sum())
        self._counts[term_index].add_(float(self.num_envs))

        sampled_error = error_float.index_select(0, self._sample_indices)
        start = self._sample_offsets[term_index]
        end = start + self.sample_envs_per_rank
        capacity = self._sampled_errors.shape[1]
        if end > capacity:
            raise RuntimeError(
                "Task-space diagnostic sample window overflow for "
                f"{name!r}: requested {end}, capacity {capacity}. "
                "The accumulator must be consumed at each log interval."
            )
        self._sampled_errors[term_index, start:end].copy_(sampled_error)
        self._sample_offsets[term_index] = end

    @torch.no_grad()
    def pop(self) -> TaskSpaceDiagnosticsSnapshot | None:
        """Return the current window and reset the accumulator in-place."""

        if not any(self._sample_offsets):
            return None
        first_offset = self._sample_offsets[0]
        if first_offset <= 0 or any(
            offset != first_offset for offset in self._sample_offsets
        ):
            recorded = {
                name: self._sample_offsets[index]
                for index, (name, _unit) in enumerate(
                    TASK_SPACE_DIAGNOSTIC_SPECS
                )
            }
            raise RuntimeError(
                "Task-space diagnostics require all eight reward terms once "
                f"per environment step; sampled counts were {recorded}."
            )

        snapshot = TaskSpaceDiagnosticsSnapshot(
            error_sums=self._error_sums.clone(),
            kernel_sums=self._kernel_sums.clone(),
            counts=self._counts.clone(),
            sampled_errors=(
                self._sampled_errors[:, :first_offset]
                .transpose(0, 1)
                .contiguous()
            ),
        )
        self._error_sums.zero_()
        self._kernel_sums.zero_()
        self._counts.zero_()
        self._sample_offsets = [0] * len(TASK_SPACE_DIAGNOSTIC_SPECS)
        return snapshot


def configure_task_space_diagnostics(
    env: Any,
    *,
    num_envs: int,
    window_steps: int,
    device: torch.device | str,
    sample_envs_per_rank: int = 256,
) -> TaskSpaceDiagnosticsAccumulator:
    """Attach a fresh diagnostics accumulator to an IsaacLab environment."""

    accumulator = TaskSpaceDiagnosticsAccumulator(
        num_envs=num_envs,
        window_steps=window_steps,
        device=device,
        sample_envs_per_rank=sample_envs_per_rank,
    )
    setattr(env, TASK_SPACE_DIAGNOSTICS_ATTRIBUTE, accumulator)
    return accumulator


def record_task_space_diagnostic(
    env: Any,
    *,
    name: str,
    error: torch.Tensor,
    kernel: torch.Tensor,
) -> None:
    """Record a term when training has configured diagnostics."""

    accumulator = getattr(env, TASK_SPACE_DIAGNOSTICS_ATTRIBUTE, None)
    if isinstance(accumulator, TaskSpaceDiagnosticsAccumulator):
        accumulator.record(name=name, error=error, kernel=kernel)


def pop_task_space_diagnostics(
    env: Any,
) -> TaskSpaceDiagnosticsSnapshot | None:
    """Pop diagnostics from a configured environment, if present."""

    accumulator = getattr(env, TASK_SPACE_DIAGNOSTICS_ATTRIBUTE, None)
    if not isinstance(accumulator, TaskSpaceDiagnosticsAccumulator):
        return None
    return accumulator.pop()


@torch.no_grad()
def summarize_task_space_diagnostics(
    snapshot: TaskSpaceDiagnosticsSnapshot,
    *,
    accelerator: Any,
    is_main_process: bool,
) -> dict[str, float]:
    """Build globally reduced TensorBoard scalars at a log boundary.

    Means use every environment-step on every rank.  P95 uses a deterministic
    uniform environment sample pooled across ranks and reports the nearest-rank
    order statistic.
    """

    packed_sums = torch.stack(
        (snapshot.error_sums, snapshot.kernel_sums, snapshot.counts),
        dim=0,
    )
    sampled_errors = snapshot.sampled_errors
    if int(getattr(accelerator, "num_processes", 1)) > 1:
        packed_sums = accelerator.reduce(packed_sums, reduction="sum")
        sampled_errors = accelerator.gather(sampled_errors)

    if not is_main_process:
        return {}

    counts = packed_sums[2].clamp_min(1.0)
    error_means = packed_sums[0] / counts
    kernel_means = packed_sums[1] / counts
    sample_count = int(sampled_errors.shape[0])
    if sample_count <= 0:
        raise RuntimeError(
            "Task-space diagnostic sample is unexpectedly empty."
        )
    p95_index = max(1, int(math.ceil(0.95 * sample_count)))
    error_p95 = torch.kthvalue(
        sampled_errors,
        p95_index,
        dim=0,
    ).values

    stacked_values = torch.stack(
        (error_means, error_p95, kernel_means),
        dim=1,
    )
    cpu_values = stacked_values.detach().to(device="cpu").tolist()

    metrics: dict[str, float] = {}
    prefix = "Metrics/TaskSpaceRaw"
    for (name, unit), values in zip(
        TASK_SPACE_DIAGNOSTIC_SPECS,
        cpu_values,
        strict=True,
    ):
        error_mean, sampled_p95, kernel_mean = values
        metrics[f"{prefix}/{name}/raw_error_mean_{unit}"] = float(error_mean)
        metrics[f"{prefix}/{name}/raw_error_p95_sampled_{unit}"] = float(
            sampled_p95
        )
        metrics[f"{prefix}/{name}/kernel_mean"] = float(kernel_mean)
    return metrics
