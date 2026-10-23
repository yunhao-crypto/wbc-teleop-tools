"""Dtype-aware CUDA observation binding shared by deployed policies."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover - host-only test environment
    wp = None


if wp is not None:

    @wp.kernel
    def _float32_to_float16(
        source: wp.array(dtype=wp.float32),
        target: wp.array(dtype=wp.float16),
    ) -> None:
        index = wp.tid()
        target[index] = wp.float16(source[index])


@dataclass(frozen=True)
class CudaObservation:
    array: Any
    numpy_dtype: np.dtype
    is_cuda: bool = True

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def buffer_ptr(self) -> int:
        return int(self.array.ptr)

    @property
    def device_id(self) -> int:
        return int(getattr(self.array.device, "ordinal", 0) or 0)

    def synchronize(self) -> None:
        if wp is None:  # pragma: no cover
            raise RuntimeError("warp-lang is unavailable")
        wp.synchronize_device(self.array.device)


class GpuObservationCaster:
    """Cast a Warp/Torch FP32 observation to the ONNX input dtype on GPU."""

    def __init__(self) -> None:
        self._warp_fp16 = None

    def cast(self, observation: Any, dtype: np.dtype | type) -> Any:
        dtype = np.dtype(dtype)
        if dtype == np.dtype(np.float32):
            return observation
        if dtype != np.dtype(np.float16):
            raise TypeError(f"unsupported CUDA observation dtype: {dtype}")
        if hasattr(observation, "array") and hasattr(
            observation.array, "device"
        ):
            if wp is None:  # pragma: no cover
                raise RuntimeError("warp-lang is required for Warp casting")
            source = observation.array
            if source.dtype == wp.float16:
                return CudaObservation(source, np.dtype(np.float16))
            if source.dtype != wp.float32:
                raise TypeError(f"expected Warp float32, got {source.dtype}")
            if self._warp_fp16 is None or tuple(
                self._warp_fp16.shape
            ) != tuple(source.shape):
                self._warp_fp16 = wp.empty(
                    source.shape,
                    dtype=wp.float16,
                    device=source.device,
                )
            wp.launch(
                _float32_to_float16,
                dim=source.size,
                inputs=[source.flatten(), self._warp_fp16.flatten()],
                device=source.device,
            )
            return CudaObservation(self._warp_fp16, np.dtype(np.float16))
        import torch

        if (
            not isinstance(observation, torch.Tensor)
            or not observation.is_cuda
        ):
            raise TypeError("observation must expose a CUDA buffer")
        return observation.to(dtype=torch.float16).contiguous()


def _binding_info(
    observation: Any, expected_dtype: np.dtype
) -> tuple[int, tuple[int, ...], int]:
    if hasattr(observation, "buffer_ptr"):
        observation.synchronize()
        actual_dtype = getattr(observation, "numpy_dtype", expected_dtype)
        if np.dtype(actual_dtype) != expected_dtype:
            raise TypeError(
                f"CUDA observation dtype is {actual_dtype}, "
                f"expected {expected_dtype}"
            )
        return (
            int(observation.buffer_ptr),
            tuple(observation.shape),
            int(getattr(observation, "device_id", 0)),
        )
    import torch

    if not isinstance(observation, torch.Tensor) or not observation.is_cuda:
        raise TypeError("observation must expose a CUDA buffer")
    expected_torch = (
        torch.float16
        if expected_dtype == np.dtype(np.float16)
        else torch.float32
    )
    if observation.dtype != expected_torch or not observation.is_contiguous():
        raise TypeError("CUDA Torch observation has wrong dtype or layout")
    torch.cuda.current_stream(observation.device).synchronize()
    return (
        int(observation.data_ptr()),
        tuple(observation.shape),
        int(observation.device.index or 0),
    )


def run_cuda_observation(
    session: Any,
    *,
    input_name: str,
    input_dtype: np.dtype | type,
    observation: Any,
    output_names: list[str],
    cpu_inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    dtype = np.dtype(input_dtype)
    pointer, shape, device_id = _binding_info(observation, dtype)
    binding = session.io_binding()
    binding.bind_input(
        name=input_name,
        device_type="cuda",
        device_id=device_id,
        element_type=dtype.type,
        shape=shape,
        buffer_ptr=pointer,
    )
    for name, value in cpu_inputs.items():
        binding.bind_cpu_input(name, np.ascontiguousarray(value))
    for name in output_names:
        binding.bind_output(name, device_type="cpu")
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()


__all__ = [
    "CudaObservation",
    "GpuObservationCaster",
    "run_cuda_observation",
]
