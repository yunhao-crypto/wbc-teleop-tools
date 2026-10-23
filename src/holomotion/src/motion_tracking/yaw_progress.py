"""Reset-safe unwrapped yaw-progress errors for scale-invariant tracking."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _normalize_heading_xy(heading_xy: torch.Tensor) -> torch.Tensor:
    if heading_xy.ndim != 2 or heading_xy.shape[-1] != 2:
        raise ValueError("heading_xy must have shape [B,2]")
    return heading_xy / torch.linalg.vector_norm(
        heading_xy,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)


def signed_heading_increment(
    previous_heading_xy: torch.Tensor,
    current_heading_xy: torch.Tensor,
) -> torch.Tensor:
    """Return the shortest signed planar angle from previous to current."""
    previous = _normalize_heading_xy(previous_heading_xy)
    current = _normalize_heading_xy(current_heading_xy)
    cross = previous[:, 0] * current[:, 1] - previous[:, 1] * current[:, 0]
    dot = torch.sum(previous * current, dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(cross, dot)


@dataclass(frozen=True)
class YawProgressSnapshot:
    """Signed unwrapped progress errors for finite and episode horizons."""

    signed_window_errors: torch.Tensor
    signed_episode_error: torch.Tensor | None


class MultiHorizonYawProgress:
    """Track signed yaw progress over several windows and the full episode.

    Each step accumulates the shortest signed heading increment.  Therefore a
    complete missed revolution remains ``2*pi`` in the episode accumulator even
    after the current headings become identical again.  Assignment changes,
    frame discontinuities, and explicit resets reseed all accumulators.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        window_steps: Sequence[int],
        include_episode: bool,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        windows = tuple(int(value) for value in window_steps)
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not windows or any(value <= 0 for value in windows):
            raise ValueError("window_steps must contain positive integers")
        if any(
            left >= right
            for left, right in zip(windows, windows[1:], strict=False)
        ):
            raise ValueError("window_steps must be strictly increasing")
        self.num_envs = int(num_envs)
        self.window_steps = windows
        self.include_episode = bool(include_episode)
        self.device = torch.device(device)
        self.dtype = dtype
        self._max_window_steps = max(windows)
        self._window_steps_tensor = torch.tensor(
            windows,
            device=self.device,
            dtype=torch.long,
        )

        self._delta_error_history = torch.zeros(
            (self.num_envs, self._max_window_steps),
            device=self.device,
            dtype=self.dtype,
        )
        self._window_error_sums = torch.zeros(
            (self.num_envs, len(self.window_steps)),
            device=self.device,
            dtype=self.dtype,
        )
        self._episode_error_sum = (
            torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=self.dtype,
            )
            if self.include_episode
            else None
        )
        self._write_index = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._previous_actual = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
            dtype=self.dtype,
        )
        self._previous_target = torch.zeros_like(self._previous_actual)
        self._last_clip = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._last_frame = torch.full_like(self._last_clip, -1)
        self._last_start_frame = torch.full_like(self._last_clip, -1)
        self._needs_reseed = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | None,
    ) -> None:
        if env_ids is None:
            self._needs_reseed[:] = True
            return
        if isinstance(env_ids, slice):
            self._needs_reseed[env_ids] = True
            return
        env_ids_tensor = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )
        self._needs_reseed[env_ids_tensor] = True

    def _clear(self, mask: torch.Tensor) -> None:
        self._delta_error_history[mask] = 0.0
        self._window_error_sums[mask] = 0.0
        if self._episode_error_sum is not None:
            self._episode_error_sum[mask] = 0.0
        self._write_index[mask] = 0

    @torch.no_grad()
    def update(
        self,
        *,
        actual_heading_xy: torch.Tensor,
        target_heading_xy: torch.Tensor,
        clip_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        start_frame_indices: torch.Tensor,
    ) -> YawProgressSnapshot:
        expected_heading_shape = (self.num_envs, 2)
        if tuple(actual_heading_xy.shape) != expected_heading_shape:
            raise ValueError(
                f"actual_heading_xy must have shape {expected_heading_shape}"
            )
        if tuple(target_heading_xy.shape) != expected_heading_shape:
            raise ValueError(
                f"target_heading_xy must have shape {expected_heading_shape}"
            )
        expected_index_shape = (self.num_envs,)
        for name, value in (
            ("clip_indices", clip_indices),
            ("frame_indices", frame_indices),
            ("start_frame_indices", start_frame_indices),
        ):
            if tuple(value.shape) != expected_index_shape:
                raise ValueError(
                    f"{name} must have shape {expected_index_shape}"
                )

        actual = _normalize_heading_xy(
            actual_heading_xy.to(device=self.device, dtype=self.dtype)
        )
        target = _normalize_heading_xy(
            target_heading_xy.to(device=self.device, dtype=self.dtype)
        )
        clip = clip_indices.to(device=self.device, dtype=torch.long)
        frame = frame_indices.to(device=self.device, dtype=torch.long)
        start = start_frame_indices.to(device=self.device, dtype=torch.long)

        same_assignment = (clip == self._last_clip) & (
            start == self._last_start_frame
        )
        duplicate = (
            (~self._needs_reseed)
            & same_assignment
            & (frame == self._last_frame)
        )
        advance = (
            (~self._needs_reseed)
            & same_assignment
            & (frame == self._last_frame + 1)
        )
        reseed = ~(duplicate | advance)
        self._clear(reseed)

        actual_increment = signed_heading_increment(
            self._previous_actual[advance],
            actual[advance],
        )
        target_increment = signed_heading_increment(
            self._previous_target[advance],
            target[advance],
        )
        delta_error = actual_increment - target_increment
        env_ids = torch.nonzero(advance, as_tuple=False).squeeze(-1)
        write_index = self._write_index[advance]
        evict_index = (
            write_index[:, None] - self._window_steps_tensor[None]
        ) % self._max_window_steps
        evicted = self._delta_error_history[env_ids[:, None], evict_index]
        self._window_error_sums[env_ids] += delta_error[:, None] - evicted
        self._delta_error_history[env_ids, write_index] = delta_error
        self._write_index[advance] = (write_index + 1) % self._max_window_steps
        if self._episode_error_sum is not None:
            self._episode_error_sum[advance] += delta_error

        changed = ~duplicate
        self._previous_actual[changed] = actual[changed]
        self._previous_target[changed] = target[changed]
        self._last_clip[changed] = clip[changed]
        self._last_frame[changed] = frame[changed]
        self._last_start_frame[changed] = start[changed]
        self._needs_reseed[changed] = False
        return YawProgressSnapshot(
            signed_window_errors=self._window_error_sums.clone(),
            signed_episode_error=(
                None
                if self._episode_error_sum is None
                else self._episode_error_sum.clone()
            ),
        )


class WindowedYawProgress:
    """Accumulate unwrapped target-minus-actual yaw over a rolling window."""

    def __init__(
        self,
        *,
        num_envs: int,
        window_steps: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        self.num_envs = int(num_envs)
        self.window_steps = int(window_steps)
        self.device = torch.device(device)
        self.dtype = dtype

        self._delta_error_history = torch.zeros(
            (self.num_envs, self.window_steps),
            device=self.device,
            dtype=self.dtype,
        )
        self._delta_error_sum = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=self.dtype,
        )
        self._write_index = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._previous_actual = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
            dtype=self.dtype,
        )
        self._previous_target = torch.zeros_like(self._previous_actual)
        self._last_clip = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._last_frame = torch.full_like(self._last_clip, -1)
        self._last_start_frame = torch.full_like(self._last_clip, -1)
        self._needs_reseed = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | None,
    ) -> None:
        if env_ids is None:
            self._needs_reseed[:] = True
            return
        if isinstance(env_ids, slice):
            self._needs_reseed[env_ids] = True
            return
        env_ids_tensor = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )
        self._needs_reseed[env_ids_tensor] = True

    def _clear(self, mask: torch.Tensor) -> None:
        self._delta_error_history[mask] = 0.0
        self._delta_error_sum[mask] = 0.0
        self._write_index[mask] = 0

    @torch.no_grad()
    def update(
        self,
        *,
        actual_heading_xy: torch.Tensor,
        target_heading_xy: torch.Tensor,
        clip_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        start_frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        expected_heading_shape = (self.num_envs, 2)
        if tuple(actual_heading_xy.shape) != expected_heading_shape:
            raise ValueError(
                f"actual_heading_xy must have shape {expected_heading_shape}"
            )
        if tuple(target_heading_xy.shape) != expected_heading_shape:
            raise ValueError(
                f"target_heading_xy must have shape {expected_heading_shape}"
            )
        expected_index_shape = (self.num_envs,)
        for name, value in (
            ("clip_indices", clip_indices),
            ("frame_indices", frame_indices),
            ("start_frame_indices", start_frame_indices),
        ):
            if tuple(value.shape) != expected_index_shape:
                raise ValueError(
                    f"{name} must have shape {expected_index_shape}"
                )

        actual = _normalize_heading_xy(
            actual_heading_xy.to(device=self.device, dtype=self.dtype)
        )
        target = _normalize_heading_xy(
            target_heading_xy.to(device=self.device, dtype=self.dtype)
        )
        clip = clip_indices.to(device=self.device, dtype=torch.long)
        frame = frame_indices.to(device=self.device, dtype=torch.long)
        start = start_frame_indices.to(device=self.device, dtype=torch.long)

        same_assignment = (clip == self._last_clip) & (
            start == self._last_start_frame
        )
        duplicate = (
            (~self._needs_reseed)
            & same_assignment
            & (frame == self._last_frame)
        )
        advance = (
            (~self._needs_reseed)
            & same_assignment
            & (frame == self._last_frame + 1)
        )
        reseed = ~(duplicate | advance)
        self._clear(reseed)

        actual_increment = signed_heading_increment(
            self._previous_actual[advance],
            actual[advance],
        )
        target_increment = signed_heading_increment(
            self._previous_target[advance],
            target[advance],
        )
        delta_error = actual_increment - target_increment
        env_ids = torch.nonzero(advance, as_tuple=False).squeeze(-1)
        write_index = self._write_index[advance]
        evicted = self._delta_error_history[env_ids, write_index]
        self._delta_error_sum[advance] += delta_error - evicted
        self._delta_error_history[env_ids, write_index] = delta_error
        self._write_index[advance] = (write_index + 1) % self.window_steps

        changed = ~duplicate
        self._previous_actual[changed] = actual[changed]
        self._previous_target[changed] = target[changed]
        self._last_clip[changed] = clip[changed]
        self._last_frame[changed] = frame[changed]
        self._last_start_frame[changed] = start[changed]
        self._needs_reseed[changed] = False
        return torch.abs(self._delta_error_sum).clone()


__all__ = [
    "MultiHorizonYawProgress",
    "WindowedYawProgress",
    "YawProgressSnapshot",
    "signed_heading_increment",
]
