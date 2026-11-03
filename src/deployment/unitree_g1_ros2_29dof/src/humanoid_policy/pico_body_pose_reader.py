"""Latest-only Pico XRoboToolkit reader for the pure-SMPL runtime."""

from __future__ import annotations
import threading
import time
import math
import json
import os
import queue
from pathlib import Path

import numpy as np


class PicoFrameLog:
    """Optional bounded capture of reader-visible frames, outside the reader loop."""
    def __init__(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / f'pico_reader_{os.getpid()}_{time.time_ns()}.jsonl'
        self.queue = queue.Queue(maxsize=2048)
        self.stop = threading.Event()
        self.dropped = self.written = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, name='pico-frame-log', daemon=True)
        self.thread.start()

    def append(self, sample):
        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            with self.path.open('x', buffering=1024*1024) as stream:
                stream.write(json.dumps(dict(record_type='header',
                    clock='Python reader first visibility; not kernel or SDK callback arrival'))+'\n')
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        sample = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    row = {k:(v.tolist() if isinstance(v, np.ndarray) else v)
                           for k,v in sample.items()}
                    stream.write(json.dumps(dict(record_type='pico_reader_frame', **row),
                                            allow_nan=False, separators=(',', ':'))+'\n')
                    self.written += 1
                stream.write(json.dumps(dict(record_type='summary', written=self.written,
                                             dropped=self.dropped))+'\n')
        except BaseException as error:
            self.error = repr(error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3.)


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
        self._frame_sequence = 0
        self._capture = None

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
                self._frame_sequence += 1
                realtime_ns, monotonic_ns = time.time_ns(), time.monotonic_ns()
                self._latest = {
                    "body_poses": np.ascontiguousarray(body),
                    "timestamp_realtime": realtime_ns*1e-9,
                    "timestamp_monotonic": monotonic_ns*1e-9,
                    "reader_realtime_ns": realtime_ns,
                    "reader_monotonic_ns": monotonic_ns,
                    "reader_frame_sequence": self._frame_sequence,
                    "timestamp_ns": stamp_ns,
                    "dt": float(dt),
                    "fps": float(self._fps_ema),
                }
                if self._capture is not None:
                    self._capture.append(self._latest)
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
        directory = os.environ.get('VITA_PICO_CAPTURE_DIR')
        if directory:
            self._capture = PicoFrameLog(directory)
        self._thread = threading.Thread(
            target=self._run, name="pico-body-reader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.close()
            if self._capture.error or self._capture.dropped or self._capture.thread.is_alive():
                self.logger.error(f'[Pico] frame capture incomplete: error={self._capture.error}, '
                                  f'dropped={self._capture.dropped}, writer_alive={self._capture.thread.is_alive()}')
            self._capture = None
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
