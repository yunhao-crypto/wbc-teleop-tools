"""Bounded process-backed JSONL recording, without robot/SDK imports.

The parent snapshots arrays before returning; JSON serialization and disk writes
run in a separate process, avoiding a Python writer thread competing for the GIL.
Overflow, worker errors and incomplete drains are errors, never silent drops.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
from typing import Mapping

import numpy as np

from .recording import JsonlSessionRecorder


def _snapshot(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported in control logs")
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported control log value: {type(value).__name__}")


def _writer(root, prefix, messages, replies):
    try:
        # Terminal Ctrl-C belongs to the control process: it cancels control
        # first, then drains this worker. Do not lose queued exit records.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Never inherit the control loop's real-time scheduling priority.
        if hasattr(os, "sched_setscheduler"):
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        with JsonlSessionRecorder(root, prefix=prefix) as recorder:
            replies.put(("ready", str(recorder.directory)))
            count = 0
            while True:
                sequence, payload = messages.get()
                if sequence != count:
                    raise RuntimeError(
                        f"Log sequence gap: {sequence} != {count}"
                    )
                if payload is None:
                    break
                recorder.write(payload)
                count += 1
        # Acknowledgement is after flush, fsync and close.
        replies.put(("closed", count))
    except BaseException as exc:
        replies.put(("error", repr(exc)))
        raise


class ProcessJsonlSessionRecorder:
    write_semantics = "snapshot_and_enqueue_to_process"

    def __init__(self, root, *, prefix, capacity=256, timeout=15.0):
        if capacity < 1 or timeout <= 0:
            raise ValueError("Positive recorder capacity and timeout required")
        ctx = mp.get_context("spawn")
        self._messages = ctx.Queue(maxsize=capacity)
        self._replies = ctx.Queue()
        self._timeout = timeout
        self._count = 0
        self._closed = False
        self._failure = None
        self._process = ctx.Process(
            target=_writer,
            args=(str(root), prefix, self._messages, self._replies),
            name="vita-log-writer",
            daemon=True,
        )
        self._process.start()
        try:
            kind, detail = self._replies.get(timeout=timeout)
            if kind != "ready":
                raise RuntimeError(f"Recorder startup failed: {detail}")
            self.directory = Path(detail)
            self.path = self.directory / "control.jsonl"
        except BaseException:
            self._dispose()
            raise

    def check_health(self):
        if self._closed:
            raise RuntimeError("Recorder is closed")
        if self._failure:
            raise RuntimeError(self._failure)
        try:
            kind, detail = self._replies.get_nowait()
        except queue.Empty:
            if self._process.is_alive():
                return
            self._failure = f"Recorder worker exited: {self._process.exitcode}"
        else:
            self._failure = f"Unexpected recorder response: {kind}: {detail}"
        raise RuntimeError(self._failure)

    def write(self, payload):
        self.check_health()
        record = _snapshot(payload)
        try:
            self._messages.put_nowait((self._count, record))
        except queue.Full as exc:
            self._failure = "Recorder queue full; no silent log loss permitted"
            raise RuntimeError(self._failure) from exc
        self._count += 1

    def _dispose(self):
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)
        self._messages.cancel_join_thread()
        self._messages.close()
        self._replies.close()
        self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            if not self._process.is_alive():
                raise RuntimeError("Recorder worker exited before log drain")
            self._messages.put((self._count, None), timeout=self._timeout)
            kind, detail = self._replies.get(timeout=self._timeout)
            if kind != "closed" or detail != self._count:
                raise RuntimeError(f"Incomplete log drain: {kind}: {detail}")
            self._process.join(timeout=self._timeout)
            if self._process.exitcode != 0:
                raise RuntimeError(
                    f"Recorder exit status: {self._process.exitcode}"
                )
        finally:
            self._dispose()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
