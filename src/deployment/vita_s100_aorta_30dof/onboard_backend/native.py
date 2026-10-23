"""ctypes adapter for the no-actuation Vita C ABI; all actions remain unclipped."""
from __future__ import annotations
import ctypes as C
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Mapping
import numpy as np

ABI = 1
F32P = C.POINTER(C.c_float)
I64P = C.POINTER(C.c_int64)


class BackendError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Vita backend status {status}: {message}")
        self.status = int(status)


class _Artifact(C.Structure):
    _fields_ = [("path", C.c_char_p), ("sha256", C.c_char_p)]


class _Config(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("struct_bytes", C.c_uint32)] + [
        (name, _Artifact) for name in ("front", "back", "moe", "router", "pool0", "pool1", "velocity", "velocity_hbm")
    ] + [("timeout_ms", C.c_uint32), ("parallel_router", C.c_uint32), ("deadline_ms", C.c_double)]


class _State(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("struct_bytes", C.c_uint32)] + [
        (name, C.c_uint64) for name in ("velocity_calls", "motion_calls", "explicit_resets", "rope_resets")
    ] + [("motion_step", C.c_int64), ("poisoned", C.c_uint32), ("velocity_hbm_available", C.c_uint32)]


class _Timing(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("struct_bytes", C.c_uint32), ("step_used", C.c_int64), ("step_next", C.c_int64)] + [
        (name, C.c_uint64) for name in ("velocity_calls", "motion_calls", "explicit_resets", "rope_resets")
    ] + [(name, C.c_double) for name in ("total_ms", "router_ms", "frontend_ms", "parallel_ms", "prepare_ms", "moe_ms", "backend_ms", "output_ms")] + [
        ("deadline_missed", C.c_uint32), ("automatic_reset", C.c_uint32)
    ]


def _struct(cls):
    value = cls()
    value.abi_version, value.struct_bytes = ABI, C.sizeof(cls)
    return value


def _dict(value):
    return {name: getattr(value, name) for name, _ in value._fields_}


def _observation(value, width):
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape not in ((width,), (1, width)):
        raise ValueError(f"Expected original FP32 observation [{width}] or [1,{width}], got {value.dtype} {value.shape}")
    value = np.ascontiguousarray(value).reshape(width)
    if not np.isfinite(value).all():
        raise ValueError("Observation contains nonfinite values")
    return value


_FILE_KEYS = {
    "front": "artifacts/model_motion/cuts/frontend.onnx",
    "back": "artifacts/model_motion/cuts/backend_after_moe.onnx",
    "moe": "artifacts/model_motion/moe/fp32_moe.hbm",
    "router": "artifacts/model_motion/cuts/router_parameters.bin",
    "pool0": "artifacts/model_motion/pools_nk/expert_pool_0_nk.f32",
    "pool1": "artifacts/model_motion/pools_nk/expert_pool_1_nk.f32",
    "velocity": "artifacts/model_velocity/original.onnx",
    "velocity_hbm": "artifacts/model_velocity/vpu/velocity_fp32_vpu.hbm",
}


def _artifacts(manifest, artifact_root=None):
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).resolve()
        data = json.loads(manifest_path.read_text())
        base = manifest_path.parent
    else:
        data = dict(manifest)
        base = None
    if "files" in data:
        if artifact_root is None:
            if base is None or base.name != "outputs":
                raise ValueError("File-list manifest requires artifact_root or an outputs/ manifest path")
            artifact_root = base.parent
        root = Path(artifact_root).resolve()
        entries = {item["path"]: item for item in data["files"]}
        records = {key: entries[path] for key, path in _FILE_KEYS.items() if path in entries}
    else:
        records = data.get("artifacts", data)
        if artifact_root is None and base is None:
            # Dict paths may all be absolute; reject ambiguity for relative paths.
            if any(not Path(item["path"]).is_absolute() for item in records.values() if item):
                raise ValueError("Relative artifact mapping requires artifact_root")
        root = Path(artifact_root).resolve() if artifact_root is not None else base
    result = {}
    for key in _FILE_KEYS:
        record = records.get(key)
        if record is None:
            if key == "velocity_hbm":
                continue
            raise ValueError(f"Missing required artifact {key}")
        path = Path(record["path"])
        if not path.is_absolute():
            if root is None:
                raise ValueError("Relative artifact path without root")
            path = root / path
        digest = record["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA256 for {key}")
        result[key] = {"path": str(path.resolve()), "sha256": digest}
    return result


@dataclass(frozen=True)
class InferenceResult:
    actions: np.ndarray
    timing: Mapping[str, Any]
    cache: np.ndarray | None = None
    routes: np.ndarray | None = None
    logits: np.ndarray | None = None


class NativeBackend:
    """Both policies resident; serialized calls; no clipping/mapping/control.

    copy=False returns persistent output views valid only until the next call.
    Reset clears native Motion KV/step only. The surrounding original runtime
    owns Velocity history/gait and both clipped last_action buffers.
    """
    def __init__(self, library, manifest, *, artifact_root=None, velocity_backend="ort",
                 parallel_router=True, timeout_ms=100, deadline_ms=20.0):
        if not 1 <= int(timeout_ms) <= 60000:
            raise ValueError("timeout_ms must be positive and <=60000")
        if velocity_backend not in ("ort", "hbm"):
            raise ValueError("velocity_backend must be ort or hbm")
        if not np.isfinite(deadline_ms) or deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive and finite")
        self.artifacts = _artifacts(manifest, artifact_root)
        if velocity_backend == "hbm" and "velocity_hbm" not in self.artifacts:
            raise ValueError("Velocity HBM was not supplied at startup")
        self.velocity_backend = velocity_backend
        self._lock = threading.Lock()
        self._handle = C.c_void_p()
        self._error = C.create_string_buffer(2048)
        self._lib = C.CDLL(str(Path(library).resolve()))
        self._configure_abi()
        self._velocity_actions = np.empty(28, dtype=np.float32)
        self._motion_actions = np.empty(30, dtype=np.float32)
        self._cache = np.empty((1, 2, 1, 32, 4, 64), dtype=np.float32)
        self._routes = np.empty((1, 1, 2), dtype=np.int64)
        self._logits = np.empty((1, 1, 1024), dtype=np.float32)
        self._timing = _struct(_Timing)
        cfg = _struct(_Config)
        cfg.timeout_ms, cfg.parallel_router, cfg.deadline_ms = int(timeout_ms), int(bool(parallel_router)), float(deadline_ms)
        for key, item in self.artifacts.items():
            setattr(cfg, key, _Artifact(item["path"].encode(), item["sha256"].encode()))
        self._check(self._lib.vita_create(C.byref(cfg), C.byref(self._handle), self._error, len(self._error)))
        self.build_info = self._lib.vita_build_info().decode()

    def _configure_abi(self):
        error = [C.c_void_p, C.c_size_t]
        self._lib.vita_create.argtypes = [C.POINTER(_Config), C.POINTER(C.c_void_p)] + error
        self._lib.vita_reset.argtypes = [C.c_void_p] + error
        self._lib.vita_state.argtypes = [C.c_void_p, C.POINTER(_State)] + error
        self._lib.vita_destroy.argtypes = [C.c_void_p] + error
        self._lib.vita_velocity.argtypes = [C.c_void_p, F32P, C.c_size_t, C.c_int, F32P, C.c_size_t, C.POINTER(_Timing)] + error
        self._lib.vita_motion.argtypes = [C.c_void_p, F32P, C.c_size_t, F32P, C.c_size_t, F32P, C.c_size_t, I64P, C.c_size_t, F32P, C.c_size_t, C.POINTER(_Timing)] + error
        for name in ("vita_create", "vita_reset", "vita_state", "vita_destroy", "vita_velocity", "vita_motion"):
            getattr(self._lib, name).restype = C.c_int
        self._lib.vita_build_info.argtypes = []
        self._lib.vita_build_info.restype = C.c_char_p

    def _check(self, status):
        if status:
            raise BackendError(status, self._error.value.decode(errors="replace"))

    def _enter(self):
        if not self._lock.acquire(blocking=False):
            raise BackendError(2, "Concurrent Python backend call rejected")
        if not self._handle.value:
            self._lock.release()
            raise BackendError(1, "Backend is closed")

    def reset(self):
        self._enter()
        try:
            self._check(self._lib.vita_reset(self._handle, self._error, len(self._error)))
        finally:
            self._lock.release()

    def state(self):
        self._enter()
        try:
            value = _struct(_State)
            self._check(self._lib.vita_state(self._handle, C.byref(value), self._error, len(self._error)))
            return _dict(value)
        finally:
            self._lock.release()

    def run_velocity(self, observation, *, backend=None, copy=True):
        obs = _observation(observation, 990)
        backend = self.velocity_backend if backend is None else backend
        if backend not in ("ort", "hbm"):
            raise ValueError("Unknown Velocity backend")
        self._enter()
        try:
            self._check(self._lib.vita_velocity(self._handle, obs.ctypes.data_as(F32P), obs.size,
                0 if backend == "ort" else 1, self._velocity_actions.ctypes.data_as(F32P), 28,
                C.byref(self._timing), self._error, len(self._error)))
            return InferenceResult(self._velocity_actions.copy() if copy else self._velocity_actions, _dict(self._timing))
        finally:
            self._lock.release()

    def run_motion(self, observation, *, diagnostics=False, copy=True):
        obs = _observation(observation, 3126)
        self._enter()
        try:
            self._check(self._lib.vita_motion(self._handle, obs.ctypes.data_as(F32P), obs.size,
                self._motion_actions.ctypes.data_as(F32P), 30,
                self._cache.ctypes.data_as(F32P) if diagnostics else None, self._cache.size if diagnostics else 0,
                self._routes.ctypes.data_as(I64P) if diagnostics else None, self._routes.size if diagnostics else 0,
                self._logits.ctypes.data_as(F32P) if diagnostics else None, self._logits.size if diagnostics else 0,
                C.byref(self._timing), self._error, len(self._error)))
            take = (lambda x: x.copy()) if copy else (lambda x: x)
            return InferenceResult(take(self._motion_actions), _dict(self._timing),
                take(self._cache) if diagnostics else None, take(self._routes) if diagnostics else None,
                take(self._logits) if diagnostics else None)
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise BackendError(2, "Cannot close during inference")
        try:
            if self._handle.value:
                self._check(self._lib.vita_destroy(self._handle, self._error, len(self._error)))
                self._handle = C.c_void_p()
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        # If device work is unresolved, intentionally retain native resources;
        # releasing DMA memory would be unsafe. Explicit close exposes the error.
        try:
            if hasattr(self, "_handle") and self._handle.value:
                self.close()
        except Exception:
            pass


class NativeVelocityRunner:
    def __init__(self, backend: NativeBackend):
        self.backend = backend
        self.session = self
        self.last_timing = None

    def run(self, observation):
        result = self.backend.run_velocity(observation)
        self.last_timing = result.timing
        return result.actions

    def get_providers(self):
        return ["CPUExecutionProvider"] if self.backend.velocity_backend == "ort" else ["S100UCP-FP32-Mixed"]


class NativeMotionRunner:
    def __init__(self, backend: NativeBackend):
        self.backend = backend
        self.session = self
        self.last_timing = None

    def run(self, observation):
        result = self.backend.run_motion(observation)
        self.last_timing = result.timing
        return result.actions

    def reset(self):
        self.backend.reset()

    @property
    def step_index(self):
        return np.array([self.backend.state()["motion_step"]], dtype=np.int64)

    def get_providers(self):
        return ["S100UCP-FP32-MoE", "CPUExecutionProvider"]
