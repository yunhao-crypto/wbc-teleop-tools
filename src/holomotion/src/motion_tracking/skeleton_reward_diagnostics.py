"""Windowed diagnostics for scale-invariant skeleton rewards."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

import torch

SKELETON_DIAGNOSTICS_ATTRIBUTE = "_holomotion_skeleton_si_diagnostics"
SKELETON_DIAGNOSTIC_SPECS = (
    ("local_structure", "unit"),
    ("global_structure", "rad"),
    ("normalized_position", "body"),
    ("foot_support_position", "body"),
    ("pelvis_torso_orientation", "rad"),
    ("foot_orientation", "rad"),
    ("wrist_orientation", "rad"),
    ("head_orientation", "rad"),
    ("head_forward", "rad"),
    ("head_yaw", "rad"),
    ("head_pitch", "rad"),
    ("root_linear_velocity", "inv_s"),
    ("root_angular_velocity", "radps"),
    ("root_gravity", "rad"),
    ("foot_center_position", "body"),
    ("foot_relative_position", "body"),
    ("root_yaw_rate", "radps"),
    ("root_yaw_progress", "rad"),
    ("root_yaw_progress_multiscale", "rad"),
    ("root_yaw_progress_short", "rad"),
    ("root_yaw_progress_medium", "rad"),
    ("root_yaw_progress_long", "rad"),
    ("root_yaw_progress_episode", "rad"),
)
SKELETON_DIAGNOSTIC_INDEX = {
    name: index
    for index, (name, _unit) in enumerate(SKELETON_DIAGNOSTIC_SPECS)
}


@dataclass(frozen=True)
class SkeletonDiagnosticsSnapshot:
    specs: tuple[tuple[str, str], ...]
    error_sums: torch.Tensor
    kernel_sums: torch.Tensor
    counts: torch.Tensor
    sampled_errors: torch.Tensor
    sampled_kernels: torch.Tensor


class SkeletonDiagnosticsAccumulator:
    """Accumulate exact means plus a bounded deterministic P95 sample."""

    def __init__(
        self,
        *,
        num_envs: int,
        window_steps: int,
        device: torch.device | str,
        sample_envs_per_rank: int = 256,
    ) -> None:
        if num_envs <= 0 or window_steps <= 0 or sample_envs_per_rank <= 0:
            raise ValueError(
                "num_envs, window_steps, and sample_envs_per_rank "
                "must be positive"
            )
        self.num_envs = int(num_envs)
        self.window_steps = int(window_steps)
        self.device = torch.device(device)
        self.sample_envs_per_rank = min(
            int(sample_envs_per_rank),
            self.num_envs,
        )
        term_count = len(SKELETON_DIAGNOSTIC_SPECS)
        self._error_sums = torch.zeros(
            term_count,
            dtype=torch.float32,
            device=self.device,
        )
        self._kernel_sums = torch.zeros_like(self._error_sums)
        self._counts = torch.zeros_like(self._error_sums)
        sample_numbers = torch.arange(
            self.sample_envs_per_rank,
            dtype=torch.long,
            device=self.device,
        )
        self._sample_indices = torch.div(
            (2 * sample_numbers + 1) * self.num_envs,
            2 * self.sample_envs_per_rank,
            rounding_mode="floor",
        ).clamp_max(self.num_envs - 1)
        self._sampled_errors = torch.empty(
            (
                term_count,
                self.window_steps * self.sample_envs_per_rank,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._sampled_kernels = torch.empty_like(self._sampled_errors)
        self._sample_offsets = [0] * term_count

    @torch.no_grad()
    def record(
        self,
        *,
        name: str,
        error: torch.Tensor,
        kernel: torch.Tensor,
    ) -> None:
        try:
            index = SKELETON_DIAGNOSTIC_INDEX[name]
        except KeyError as key_error:
            raise KeyError(
                f"Unknown skeleton diagnostic term {name!r}"
            ) from key_error
        error = error.detach().reshape(-1)
        kernel = kernel.detach().reshape(-1)
        expected = (self.num_envs,)
        if tuple(error.shape) != expected or tuple(kernel.shape) != expected:
            raise ValueError(
                f"{name} error/kernel must both have shape {expected}"
            )
        if error.device != self.device or kernel.device != self.device:
            raise ValueError(f"{name} diagnostics must stay on {self.device}")
        error = error.float()
        kernel = kernel.float()
        self._error_sums[index].add_(error.sum())
        self._kernel_sums[index].add_(kernel.sum())
        self._counts[index].add_(float(self.num_envs))
        sampled = error.index_select(0, self._sample_indices)
        start = self._sample_offsets[index]
        end = start + self.sample_envs_per_rank
        if end > self._sampled_errors.shape[1]:
            raise RuntimeError(
                f"Skeleton diagnostic sample overflow for {name!r}"
            )
        self._sampled_errors[index, start:end].copy_(sampled)
        self._sampled_kernels[index, start:end].copy_(
            kernel.index_select(0, self._sample_indices)
        )
        self._sample_offsets[index] = end

    @torch.no_grad()
    def pop(self) -> SkeletonDiagnosticsSnapshot | None:
        if not any(self._sample_offsets):
            return None
        active_indices = [
            index
            for index, offset in enumerate(self._sample_offsets)
            if offset > 0
        ]
        first = self._sample_offsets[active_indices[0]]
        if any(
            self._sample_offsets[index] != first for index in active_indices
        ):
            recorded = {
                name: self._sample_offsets[index]
                for index, (name, _unit) in enumerate(
                    SKELETON_DIAGNOSTIC_SPECS
                )
            }
            raise RuntimeError(
                "All active SI skeleton rewards must record once per step; "
                f"got {recorded}"
            )
        indices = torch.tensor(
            active_indices,
            dtype=torch.long,
            device=self.device,
        )
        snapshot = SkeletonDiagnosticsSnapshot(
            specs=tuple(
                SKELETON_DIAGNOSTIC_SPECS[index] for index in active_indices
            ),
            error_sums=self._error_sums.index_select(0, indices).clone(),
            kernel_sums=self._kernel_sums.index_select(0, indices).clone(),
            counts=self._counts.index_select(0, indices).clone(),
            sampled_errors=(
                self._sampled_errors.index_select(0, indices)[:, :first]
                .transpose(0, 1)
                .contiguous()
            ),
            sampled_kernels=(
                self._sampled_kernels.index_select(0, indices)[:, :first]
                .transpose(0, 1)
                .contiguous()
            ),
        )
        self._error_sums.zero_()
        self._kernel_sums.zero_()
        self._counts.zero_()
        self._sample_offsets = [0] * len(SKELETON_DIAGNOSTIC_SPECS)
        return snapshot


def configure_skeleton_diagnostics(
    env: Any,
    *,
    num_envs: int,
    window_steps: int,
    device: torch.device | str,
    sample_envs_per_rank: int = 256,
) -> SkeletonDiagnosticsAccumulator:
    accumulator = SkeletonDiagnosticsAccumulator(
        num_envs=num_envs,
        window_steps=window_steps,
        device=device,
        sample_envs_per_rank=sample_envs_per_rank,
    )
    setattr(env, SKELETON_DIAGNOSTICS_ATTRIBUTE, accumulator)
    return accumulator


def record_skeleton_diagnostic(
    env: Any,
    *,
    name: str,
    error: torch.Tensor,
    kernel: torch.Tensor,
) -> None:
    accumulator = getattr(env, SKELETON_DIAGNOSTICS_ATTRIBUTE, None)
    if isinstance(accumulator, SkeletonDiagnosticsAccumulator):
        accumulator.record(name=name, error=error, kernel=kernel)


def pop_skeleton_diagnostics(
    env: Any,
) -> SkeletonDiagnosticsSnapshot | None:
    accumulator = getattr(env, SKELETON_DIAGNOSTICS_ATTRIBUTE, None)
    if not isinstance(accumulator, SkeletonDiagnosticsAccumulator):
        return None
    return accumulator.pop()


@torch.no_grad()
def summarize_skeleton_diagnostics(
    snapshot: SkeletonDiagnosticsSnapshot,
    *,
    accelerator: Any,
    is_main_process: bool,
) -> dict[str, float]:
    packed = torch.stack(
        (snapshot.error_sums, snapshot.kernel_sums, snapshot.counts),
        dim=0,
    )
    sampled = snapshot.sampled_errors
    sampled_kernels = snapshot.sampled_kernels
    if int(getattr(accelerator, "num_processes", 1)) > 1:
        packed = accelerator.reduce(packed, reduction="sum")
        sampled = accelerator.gather(sampled)
        sampled_kernels = accelerator.gather(sampled_kernels)
    if not is_main_process:
        return {}
    counts = packed[2].clamp_min(1.0)
    means = packed[0] / counts
    kernels = packed[1] / counts
    if sampled.shape[0] <= 0:
        raise RuntimeError("Skeleton diagnostic sample is empty")
    p95_index = max(1, int(math.ceil(0.95 * int(sampled.shape[0]))))
    p95 = torch.kthvalue(sampled, p95_index, dim=0).values
    p10_index = max(
        1,
        int(math.ceil(0.10 * int(sampled_kernels.shape[0]))),
    )
    kernel_p10 = torch.kthvalue(
        sampled_kernels,
        p10_index,
        dim=0,
    ).values
    values = (
        torch.stack(
            (means, p95, kernels, kernel_p10),
            dim=1,
        )
        .cpu()
        .tolist()
    )
    metrics: dict[str, float] = {}
    prefix = "Metrics/SkeletonSI"
    for (name, unit), (
        mean,
        sampled_p95,
        kernel,
        sampled_kernel_p10,
    ) in zip(
        snapshot.specs,
        values,
        strict=True,
    ):
        metrics[f"{prefix}/{name}/raw_error_mean_{unit}"] = float(mean)
        metrics[f"{prefix}/{name}/raw_error_p95_sampled_{unit}"] = float(
            sampled_p95
        )
        metrics[f"{prefix}/{name}/kernel_mean"] = float(kernel)
        metrics[f"{prefix}/{name}/kernel_p10_sampled"] = float(
            sampled_kernel_p10
        )
    return metrics


__all__ = [
    "SKELETON_DIAGNOSTIC_SPECS",
    "SkeletonDiagnosticsAccumulator",
    "SkeletonDiagnosticsSnapshot",
    "configure_skeleton_diagnostics",
    "pop_skeleton_diagnostics",
    "record_skeleton_diagnostic",
    "summarize_skeleton_diagnostics",
]
