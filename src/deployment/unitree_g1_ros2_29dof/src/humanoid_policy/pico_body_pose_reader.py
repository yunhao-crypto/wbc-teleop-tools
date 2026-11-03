"""Latest-only Pico XRoboToolkit reader for the pure-SMPL runtime."""

from __future__ import annotations
import threading
import time
import math

import numpy as np


class PicoBodyPoseReader:
    def __init__(self, *, logger, sdk=None, duplicate_poll_interval_s=0.0001) -> None:
        if (isinstance(duplicate_poll_interval_s, bool)
                or not isinstance(duplicate_poll_interval_s, (int, float))
                or not math.isfinite(duplicate_poll_interval_s)
                or not 0 < duplicate_poll_interval_s <= .01):
            raise ValueError('Pico duplicate poll interval must be in (0, 0.01] seconds')
        self.duplicate_poll_interval_s = float(duplicate_poll_interval_s)
        if sdk is None:
            try:
                import xrobotoolkit_sdk as sdk
            except ImportError as exc:
                raise RuntimeError(
                    "reference_source=pico_local requires xrobotoolkit_sdk"
                ) from exc
        self.logger = logger
        self.sdk = sdk
        self._snapshot = getattr(sdk, "get_body_pose_snapshot", None)
        if not callable(self._snapshot):
            raise RuntimeError("Pico requires the atomic get_body_pose_snapshot SDK binding; rebuild the private binding")
        self._stop_event = threading.Event()
        self._thread = None
        self._latest = None
        self._last_stamp_ns = None
        self._fps_ema = 0.0
        self._read_errors = 0
        self._sdk_initialized = False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                available, stamp_ns, body_snapshot = self._snapshot()
                if not available:
                    self._stop_event.wait(0.001)
                    continue
                stamp_ns = int(stamp_ns)
                previous = self._last_stamp_ns
                if previous is not None and stamp_ns == previous:
                    # Cached snapshots still copy all 24x7 poses through the
                    # binding. Avoid competing with IK for CPU/GIL on repeats.
                    self._stop_event.wait(self.duplicate_poll_interval_s)
                    continue
                dt = (stamp_ns - previous) * 1.0e-9 if previous else 0.0
                if dt > 0.0:
                    fps = 1.0 / dt
                    self._fps_ema = (
                        fps
                        if self._fps_ema == 0.0
                        else 0.9 * self._fps_ema + 0.1 * fps
                    )
                body = np.asarray(
                    body_snapshot, dtype=np.float32
                )
                if body.shape != (24, 7) or not np.isfinite(body).all():
                    raise ValueError(
                        f"XRoboToolkit body_poses must be (24,7), got {body.shape}"
                    )
                self._last_stamp_ns = stamp_ns
                self._latest = {
                    "body_poses": np.ascontiguousarray(body),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": time.monotonic(),
                    "timestamp_ns": stamp_ns,
                    "dt": float(dt),
                    "fps": float(self._fps_ema),
                }
            except Exception as exc:
                self._read_errors += 1
                if self._read_errors == 1 or self._read_errors % 100 == 0:
                    self.logger.error(f"[Pico] body reader error: {exc}")
                self._stop_event.wait(0.01)

    def latest(self):
        return self._latest

    def start(self) -> None:
        if self._thread is not None:
            return
        init = getattr(self.sdk, "init", None)
        if callable(init):
            init()
            self._sdk_initialized = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="pico-body-reader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sdk_initialized:
            close = getattr(self.sdk, "close", None)
            if callable(close):
                close_error = []

                def close_sdk() -> None:
                    try:
                        close()
                    except Exception as exc:  # pragma: no cover - SDK only
                        close_error.append(exc)

                close_thread = threading.Thread(
                    target=close_sdk,
                    name="pico-sdk-close",
                    daemon=True,
                )
                close_thread.start()
                close_thread.join(timeout=2.0)
                if close_thread.is_alive():
                    self.logger.warn(
                        "[Pico] SDK close exceeded 2s; leaving daemon cleanup"
                    )
                elif close_error:
                    self.logger.error(
                        f"[Pico] SDK close failed: {close_error[0]}"
                    )
            self._sdk_initialized = False


__all__ = ["PicoBodyPoseReader"]
