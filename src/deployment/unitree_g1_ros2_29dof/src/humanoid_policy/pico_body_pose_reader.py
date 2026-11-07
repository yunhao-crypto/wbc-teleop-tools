"""Latest-only Pico XRoboToolkit reader for the pure-SMPL runtime."""

from __future__ import annotations
import threading
import time
import math
import json
import os
import queue
import multiprocessing
import signal
from pathlib import Path

import numpy as np


def _write_frames(path, samples, result):
    """JSON conversion and file I/O run outside the reader/GMR interpreter."""
    written, error = 0, None
    try:
        # Ctrl-C belongs to the caller, which stops the reader then drains us.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Do not inherit a real-time policy if a caller has opted into one.
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        with path.open('x', buffering=1024*1024) as stream:
            stream.write(json.dumps(dict(record_type='header', writer_pid=os.getpid(),
                clock='Python reader first visibility; not kernel or SDK callback arrival'))+'\n')
            while True:
                sample = samples.get()
                if isinstance(sample, int):
                    dropped = sample  # Ordered end marker follows all accepted frames.
                    break
                row = {k:(v.tolist() if isinstance(v, np.ndarray) else v)
                       for k,v in sample.items()}
                stream.write(json.dumps(dict(record_type='pico_reader_frame', **row),
                                        allow_nan=False, separators=(',', ':'))+'\n')
                written += 1
            stream.write(json.dumps(dict(record_type='summary', written=written,
                                         dropped=dropped))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        error = repr(exc)
    finally:
        result.send((written, error))
        result.close()


class PicoFrameLog:
    """Optional bounded process capture; append never waits for disk or JSON."""
    def __init__(self, directory, *, capacity=2048):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Pico capture capacity must be a positive integer')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / f'pico_reader_{os.getpid()}_{time.time_ns()}.jsonl'
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(maxsize=capacity)
        self.result, writer_result = context.Pipe(duplex=False)
        self.dropped = self.written = 0
        self.accepted = 0
        self.error = None
        self.closed = False
        self.process = context.Process(target=_write_frames,
            args=(self.path, self.queue, writer_result), name='pico-frame-log', daemon=True)
        self.process.start()
        writer_result.close()

    def append(self, sample):
        if self.closed:
            raise RuntimeError('Pico frame capture is closed')
        if not self.process.is_alive():
            self.dropped += 1
            self.error = 'Pico frame writer exited before close'
            return
        try:
            # The reader publishes immutable arrays; the queue feeder only
            # pickles them. All tolist/JSON work belongs to the child process.
            self.queue.put_nowait(sample)
            self.accepted += 1
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic()+3.
        try:
            if self.process.is_alive():
                self.queue.put(self.dropped, timeout=max(0., deadline-time.monotonic()))
                self.process.join(timeout=max(0., deadline-time.monotonic()))
            if self.process.is_alive():
                self.error = 'Pico frame writer did not drain within 3s'
            elif self.result.poll():
                self.written, error = self.result.recv()
                self.error = error or self.error
                if self.written != self.accepted and self.error is None:
                    self.error = f'Pico capture count mismatch: {self.written}/{self.accepted}'
            else:
                self.error = self.error or 'Pico frame writer exited without acknowledgement'
        except (queue.Full, EOFError, OSError) as exc:
            self.error = repr(exc)
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1.)
            self.queue.close()
            self.queue.cancel_join_thread()
            self.result.close()


class PicoBodyPoseReader:
    def __init__(self, *, logger, sdk=None, duplicate_poll_interval_s=0.0001,
                 on_frame=None) -> None:
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
        self._on_frame = on_frame

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
                    "body_poses": np.array(body, copy=True, order='C'),
                    "timestamp_realtime": realtime_ns*1e-9,
                    "timestamp_monotonic": monotonic_ns*1e-9,
                    "reader_realtime_ns": realtime_ns,
                    "reader_monotonic_ns": monotonic_ns,
                    "reader_frame_sequence": self._frame_sequence,
                    "timestamp_ns": stamp_ns,
                    "dt": float(dt),
                    "fps": float(self._fps_ema),
                }
                if self._on_frame is not None:
                    self._on_frame(self._latest)
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
            if self._capture.error or self._capture.dropped or self._capture.process.is_alive():
                self.logger.error(f'[Pico] frame capture incomplete: error={self._capture.error}, '
                                  f'dropped={self._capture.dropped}, writer_alive={self._capture.process.is_alive()}')
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
