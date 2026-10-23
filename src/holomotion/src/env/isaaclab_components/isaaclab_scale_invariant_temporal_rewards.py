"""Stateful temporal rewards for scale-invariant skeleton tracking."""

from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg

from holomotion.src.motion_tracking.skeleton_reward_diagnostics import (
    record_skeleton_diagnostic,
)
from holomotion.src.motion_tracking.yaw_progress import (
    MultiHorizonYawProgress,
    WindowedYawProgress,
)
from holomotion.src.utils import torch_utils


def _pelvis_heading_xy(state: Any) -> torch.Tensor:
    rotation = torch_utils.matrix_from_quat(state.pelvis_rotation_wxyz)
    return rotation[:, :2, 0]


class skeleton_si_windowed_yaw_progress_tracking_exp(  # noqa: N801
    ManagerTermBase
):
    """Track unwrapped signed yaw progress over a reset-safe rolling window."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        window_steps = int(cfg.params["window_steps"])
        self._window_steps = window_steps
        self._progress = WindowedYawProgress(
            num_envs=self.num_envs,
            window_steps=window_steps,
            device=self.device,
        )

    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | None = None,
    ) -> None:
        self._progress.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        window_steps: int,
        command_name: str = "ref_motion",
    ) -> torch.Tensor:
        if std <= 0.0:
            raise ValueError("std must be positive")
        if int(window_steps) != self._window_steps:
            raise ValueError("window_steps cannot change after initialization")

        command = env.command_manager.get_term(command_name)
        actual = command.get_semantic_skeleton_robot_state()
        target = command.get_semantic_skeleton_target_immediate_next()
        clip, frame, start = command.get_semantic_skeleton_progress_token()
        error = self._progress.update(
            actual_heading_xy=_pelvis_heading_xy(actual),
            target_heading_xy=_pelvis_heading_xy(target),
            clip_indices=clip,
            frame_indices=frame,
            start_frame_indices=start,
        )
        reward = torch.exp(-torch.square(error) / (float(std) ** 2))
        record_skeleton_diagnostic(
            env,
            name="root_yaw_progress",
            error=error,
            kernel=reward,
        )
        return reward


class skeleton_si_multiscale_yaw_progress_tracking_exp(  # noqa: N801
    ManagerTermBase
):
    """Track signed yaw progress at finite horizons and from episode reset."""

    _WINDOW_DIAGNOSTIC_NAMES = (
        "root_yaw_progress_short",
        "root_yaw_progress_medium",
        "root_yaw_progress_long",
    )

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._window_steps = tuple(
            int(value) for value in cfg.params["window_steps"]
        )
        self._window_weights = tuple(
            float(value) for value in cfg.params["window_weights"]
        )
        self._episode_weight = float(cfg.params["episode_weight"])
        if len(self._window_steps) > len(self._WINDOW_DIAGNOSTIC_NAMES):
            raise ValueError(
                "at most three yaw-progress windows are supported"
            )
        if len(self._window_weights) != len(self._window_steps):
            raise ValueError(
                "window_weights must have one value per window_steps entry"
            )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self._window_weights
        ):
            raise ValueError("window_weights must be finite and non-negative")
        if (
            not math.isfinite(self._episode_weight)
            or self._episode_weight < 0.0
        ):
            raise ValueError("episode_weight must be finite and non-negative")
        self._total_weight = sum(self._window_weights) + self._episode_weight
        if self._total_weight <= 0.0:
            raise ValueError(
                "at least one yaw-progress weight must be positive"
            )
        self._window_weights_tensor = torch.tensor(
            self._window_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self._progress = MultiHorizonYawProgress(
            num_envs=self.num_envs,
            window_steps=self._window_steps,
            include_episode=self._episode_weight > 0.0,
            device=self.device,
        )

    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | None = None,
    ) -> None:
        self._progress.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        window_steps: Sequence[int],
        window_weights: Sequence[float],
        episode_weight: float,
        command_name: str = "ref_motion",
    ) -> torch.Tensor:
        if std <= 0.0:
            raise ValueError("std must be positive")
        if tuple(int(value) for value in window_steps) != self._window_steps:
            raise ValueError("window_steps cannot change after initialization")
        if (
            tuple(float(value) for value in window_weights)
            != self._window_weights
        ):
            raise ValueError(
                "window_weights cannot change after initialization"
            )
        if float(episode_weight) != self._episode_weight:
            raise ValueError(
                "episode_weight cannot change after initialization"
            )

        command = env.command_manager.get_term(command_name)
        actual = command.get_semantic_skeleton_robot_state()
        target = command.get_semantic_skeleton_target_immediate_next()
        clip, frame, start = command.get_semantic_skeleton_progress_token()
        snapshot = self._progress.update(
            actual_heading_xy=_pelvis_heading_xy(actual),
            target_heading_xy=_pelvis_heading_xy(target),
            clip_indices=clip,
            frame_indices=frame,
            start_frame_indices=start,
        )
        std_square = float(std) ** 2
        window_errors = torch.abs(snapshot.signed_window_errors)
        weighted_square = torch.sum(
            torch.square(window_errors) * self._window_weights_tensor[None],
            dim=-1,
        )
        window_kernels = torch.exp(-torch.square(window_errors) / std_square)
        weighted_kernel = torch.sum(
            window_kernels * self._window_weights_tensor[None],
            dim=-1,
        )
        if snapshot.signed_episode_error is not None:
            episode_error = torch.abs(snapshot.signed_episode_error)
            episode_kernel = torch.exp(
                -torch.square(episode_error) / std_square
            )
            weighted_kernel = (
                weighted_kernel + self._episode_weight * episode_kernel
            )
            weighted_square = (
                weighted_square
                + self._episode_weight * torch.square(episode_error)
            )
        else:
            episode_error = None
            episode_kernel = None
        aggregate_error = torch.sqrt(
            (weighted_square / self._total_weight).clamp_min(0.0)
        )
        reward = weighted_kernel / self._total_weight
        record_skeleton_diagnostic(
            env,
            name="root_yaw_progress_multiscale",
            error=aggregate_error,
            kernel=reward,
        )
        for index, name in enumerate(self._WINDOW_DIAGNOSTIC_NAMES):
            if index >= len(self._window_steps):
                break
            error = window_errors[:, index]
            record_skeleton_diagnostic(
                env,
                name=name,
                error=error,
                kernel=torch.exp(-torch.square(error) / std_square),
            )
        if episode_error is not None and episode_kernel is not None:
            record_skeleton_diagnostic(
                env,
                name="root_yaw_progress_episode",
                error=episode_error,
                kernel=episode_kernel,
            )
        return reward


__all__ = [
    "skeleton_si_multiscale_yaw_progress_tracking_exp",
    "skeleton_si_windowed_yaw_progress_tracking_exp",
]
