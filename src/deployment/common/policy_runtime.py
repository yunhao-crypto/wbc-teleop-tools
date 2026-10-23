"""Action-sink-free ONNX policy primitives shared by robot deployments."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from deployment.common.cuda_io import (
    GpuObservationCaster,
    run_cuda_observation,
)


@dataclass(frozen=True)
class PolicyStep:
    """A named policy result that has not yet been sent to hardware."""

    mode: str
    joint_names: tuple[str, ...]
    raw_action: np.ndarray
    target_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        if count == 0 or len(set(self.joint_names)) != count:
            raise ValueError(
                "PolicyStep joint_names must be unique and non-empty"
            )
        for name in ("target_joint_pos", "joint_stiffness", "joint_damping"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(
                -1
            )
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f"PolicyStep {name} must be finite [{count}]")
            object.__setattr__(self, name, np.ascontiguousarray(value))
        raw = np.asarray(self.raw_action, dtype=np.float32).reshape(-1)
        if raw.size == 0 or not np.isfinite(raw).all():
            raise ValueError(
                "PolicyStep raw_action must be finite and non-empty"
            )
        object.__setattr__(self, "raw_action", np.ascontiguousarray(raw))


def create_onnx_session(
    path: str | Path,
    *,
    providers: Sequence[Any] | None = None,
    intra_op_threads: int = 1,
    require_cuda: bool = True,
) -> Any:
    """Create an ONNX session and fail if CUDA silently falls back to CPU."""

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(intra_op_threads)
    options.inter_op_num_threads = 1
    selected = list(
        providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=selected
    )
    if require_cuda and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            f"CUDAExecutionProvider was not enabled for {path}; "
            f"active providers={session.get_providers()}"
        )
    return session


def finite_action(value: Any, *, size: int, label: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (size,):
        raise RuntimeError(
            f"{label} returned {action.shape}; expected ({size},)"
        )
    if not np.isfinite(action).all():
        raise RuntimeError(f"{label} returned NaN or Inf")
    return np.ascontiguousarray(action)


class FeedForwardOnnxRunner:
    """Small allocation-conscious ONNX feed-forward runner."""

    def __init__(
        self,
        session: Any,
        *,
        input_name: str,
        output_name: str,
        input_shape: Sequence[int],
        input_dtype: np.dtype | type,
        action_size: int,
        label: str,
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = tuple(int(item) for item in input_shape)
        self.input_dtype = np.dtype(input_dtype)
        self.action_size = int(action_size)
        self.label = label

    def run(self, observation: Any) -> np.ndarray:
        value = np.asarray(observation, dtype=self.input_dtype).reshape(
            self.input_shape
        )
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(value)},
        )
        return finite_action(
            outputs[0], size=self.action_size, label=self.label
        )


class RecurrentOnnxRunner:
    """ONNX runner for a policy with explicit KV cache and step index."""

    def __init__(
        self,
        session: Any,
        *,
        observation_name: str,
        action_name: str,
        observation_shape: Sequence[int],
        observation_dtype: np.dtype | type,
        action_size: int,
        kv_input_name: str,
        kv_output_name: str,
        kv_shape: Sequence[int],
        kv_dtype: np.dtype | type,
        step_index_name: str,
        rope_max_seq_len: int,
        rope_reset_margin: int = 64,
    ) -> None:
        self.session = session
        self.observation_name = observation_name
        self.action_name = action_name
        self.observation_shape = tuple(int(item) for item in observation_shape)
        self.observation_dtype = np.dtype(observation_dtype)
        self.action_size = int(action_size)
        self.kv_input_name = kv_input_name
        self.kv_output_name = kv_output_name
        self.kv_cache = np.zeros(tuple(kv_shape), dtype=np.dtype(kv_dtype))
        self.step_index_name = step_index_name
        self.step_index = np.zeros(1, dtype=np.int64)
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.rope_reset_margin = max(int(rope_reset_margin), 0)
        self._caster = GpuObservationCaster()

    def reset(self) -> None:
        self.kv_cache.fill(0)
        self.step_index.fill(0)

    def _maybe_reset(self) -> None:
        reset_at = self.rope_max_seq_len - self.rope_reset_margin
        if reset_at <= 0:
            reset_at = self.rope_max_seq_len
        if int(self.step_index[0]) >= reset_at:
            self.reset()

    def run(self, observation: Any) -> np.ndarray:
        self._maybe_reset()
        cpu_inputs: Mapping[str, np.ndarray] = {
            self.kv_input_name: self.kv_cache,
            self.step_index_name: self.step_index,
        }
        output_names = [self.action_name, self.kv_output_name]
        if bool(getattr(observation, "is_cuda", False)):
            prepared = self._caster.cast(observation, self.observation_dtype)
            outputs = run_cuda_observation(
                self.session,
                input_name=self.observation_name,
                input_dtype=self.observation_dtype,
                observation=prepared,
                output_names=output_names,
                cpu_inputs=dict(cpu_inputs),
            )
        else:
            value = np.asarray(
                observation, dtype=self.observation_dtype
            ).reshape(self.observation_shape)
            outputs = self.session.run(
                output_names,
                {
                    self.observation_name: np.ascontiguousarray(value),
                    **cpu_inputs,
                },
            )
        if len(outputs) != 2:
            raise RuntimeError(
                "recurrent policy did not return action and KV cache"
            )
        next_cache = np.asarray(outputs[1])
        if next_cache.shape != self.kv_cache.shape:
            raise RuntimeError(
                f"KV output shape is {next_cache.shape}; "
                f"expected {self.kv_cache.shape}"
            )
        if next_cache.dtype != self.kv_cache.dtype:
            raise RuntimeError(
                f"KV output dtype is {next_cache.dtype}; "
                f"expected {self.kv_cache.dtype}"
            )
        np.copyto(self.kv_cache, next_cache)
        self.step_index[0] += 1
        return finite_action(
            outputs[0], size=self.action_size, label="motion policy"
        )


__all__ = [
    "FeedForwardOnnxRunner",
    "PolicyStep",
    "RecurrentOnnxRunner",
    "create_onnx_session",
    "finite_action",
]
