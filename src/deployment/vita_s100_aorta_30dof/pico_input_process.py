"""Bounded receive-only Pico SDK subprocess. No robot imports or publication."""
from contextlib import ExitStack
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np
from .actor_pico_buttons import DualTriggerToggle


def _worker(stop, lock, poses, clocks, buttons, status, sdk_dir, dll_dirs):
    sdk = None; initialized = False
    try:
        with ExitStack() as resources:
            if sdk_dir:
                sys.path.insert(0, sdk_dir)
                dll_dirs = [sdk_dir, *dll_dirs]
            if os.name == 'nt':
                for directory in dll_dirs:
                    resources.enter_context(os.add_dll_directory(directory))
            sdk = importlib.import_module('xrobotoolkit_sdk')
            for method in ('init', 'close', 'get_body_pose_snapshot', 'get_controller_trigger_snapshot'):
                if not callable(getattr(sdk, method, None)):
                    raise RuntimeError('Pico binding missing atomic interface: '+method)
            sdk.init()
            initialized = True
            status.put(dict(state='receiving', sdk_path=getattr(sdk, '__file__', None),
                binding_test_fixture=bool(getattr(sdk, 'CODEX_PICO_TEST_FIXTURE', False))))
            previous_body = None; previous_stamp = None
            trigger_toggle=DualTriggerToggle(.6)
            while not stop.is_set():
                available, stamp, body = sdk.get_body_pose_snapshot()
                now_ns = time.monotonic_ns()
                if available:
                    if isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer)) or not 0 < int(stamp) < 2**63:
                        raise ValueError('Pico body timestamp is not int64 nanoseconds')
                    body = np.asarray(body, dtype=np.float32)
                    if body.shape != (24, 7): raise ValueError('Pico body must be24x7')
                    if stamp != previous_stamp or not np.array_equal(body, previous_body):
                        # A single slot is latest-wins and retains first visibility.
                        # Holding the same packet never updates its receipt time.
                        with lock:
                            np.copyto(np.frombuffer(poses, dtype=np.float32).reshape(24, 7), body)
                            clocks[0:2] = [int(stamp), now_ns]; clocks[2] += 1
                        previous_body = body.copy(); previous_stamp = stamp
                available, stamp, age, left, right = sdk.get_controller_trigger_snapshot()
                now_ns = time.monotonic_ns()
                valid = (available and type(stamp) is int and 0 < stamp < 2**63 and
                    np.isfinite([age, left, right]).all() and 0 <= age <= .6 and 0 <= left <= 1 and 0 <= right <= 1)
                fired=trigger_toggle.update((available,stamp,age,left,right),now_ns/1_000_000_000)
                with lock:
                    clocks[3] = 1 if valid else 0
                    if valid and stamp != clocks[4]:
                        clocks[4:6] = [stamp, now_ns-int(round(age*1e9))]
                        buttons[:] = [float(left), float(right)]
                    if fired:clocks[6]+=1;clocks[7]=now_ns
                stop.wait(.002)
    except Exception as exc:
        try: status.put_nowait(dict(state='failed', error=f'{type(exc).__name__}: {exc}'))
        except queue.Full: pass
    finally:
        if initialized:
            # Parent enforces a bounded shutdown if a native SDK getter/close hangs.
            try: sdk.close()
            except Exception: pass


class PicoInputProcess:
    def __init__(self, *, sdk_dir=None, dll_dirs=(), library_dirs=()):
        for directory in ([sdk_dir] if sdk_dir else [])+list(dll_dirs):
            if not Path(directory).is_dir(): raise ValueError('Pico SDK/DLL directory missing')
        self.ctx = mp.get_context('spawn'); self.stop_event = self.ctx.Event()
        self.lock = self.ctx.Lock(); self.poses = self.ctx.RawArray('f', 24*7)
        self.clocks = self.ctx.RawArray('q', 8); self.buttons = self.ctx.RawArray('d', 2)
        self.status = self.ctx.Queue(maxsize=4); self.state = dict(state='starting')
        self.process = self.ctx.Process(target=_worker, args=(self.stop_event, self.lock, self.poses,
            self.clocks, self.buttons, self.status, str(Path(sdk_dir).resolve()) if sdk_dir else None,
            [str(Path(p).resolve()) for p in dll_dirs]), daemon=True, name='pico-input-receiver')
        self.last_sequence = 0; self.last_buttons = (False, 0, float('inf'), 0., 0.)
        self.close_report = None
        self.library_dirs = list(library_dirs)

    def start(self):
        # glibc reads LD_LIBRARY_PATH when the spawned interpreter starts.
        # Restore the parent's environment immediately after spawning.
        original = os.environ.get('LD_LIBRARY_PATH')
        try:
            if os.name != 'nt' and self.library_dirs:
                os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(self.library_dirs+([original] if original else []))
            self.process.start()
        finally:
            if os.name != 'nt' and self.library_dirs:
                if original is None: os.environ.pop('LD_LIBRARY_PATH', None)
                else: os.environ['LD_LIBRARY_PATH'] = original

    def poll(self, now_ns):
        while True:
            try: self.state = self.status.get_nowait()
            except queue.Empty: break
        pose = None
        if self.lock.acquire(False):
            try:
                stamp, received, sequence, available, button_stamp, button_received,event_seq,event_ns = self.clocks[:]
                if sequence != self.last_sequence:
                    pose = (np.frombuffer(self.poses, dtype=np.float32).reshape(24, 7).copy(), stamp, received)
                    self.last_sequence = sequence
                self.last_buttons = (bool(available), button_stamp, (now_ns-button_received)*1e-9, *self.buttons[:])
                self.state.update(trigger_event_seq=int(event_seq),trigger_event_monotonic_ns=int(event_ns))
            finally: self.lock.release()
        return pose, self.last_buttons, dict(self.state)

    def close(self):
        if self.close_report is not None: return self.close_report
        self.stop_event.set(); self.process.join(2.)
        forced = self.process.is_alive()
        if forced: self.process.terminate(); self.process.join(1.)
        if self.process.is_alive(): self.process.kill(); self.process.join(1.)
        self.status.close()
        self.close_report = dict(receiver_exitcode=self.process.exitcode, receiver_forced_shutdown=forced)
        return self.close_report
