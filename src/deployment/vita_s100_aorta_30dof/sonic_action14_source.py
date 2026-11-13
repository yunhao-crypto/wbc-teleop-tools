"""Reference-only Pico/GMR producer. Importing this module never opens hardware.

The worker owns its vendor service, never constructs an actor or runs inference.
Only newest bounded snapshots cross process boundaries; lost frames are not replayed.
"""
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import time
from types import SimpleNamespace

from .sonic_action14 import contract


def load_source_config(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config/actor_pico.json').read_text())
    slots, _ = contract()
    if config['body_profile']['body_joint_order'] != list(slots):
        raise ValueError('source and encoder joint contract mismatch')
    if config['artifacts']['model']['sha256'] != '2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d':
        raise ValueError('unexpected SONIC model identity')
    for name, item in config['artifacts'].items():
        if name == 'model':  # Weights live on robot; NEVER load them in source process.
            continue
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('missing pinned source artifact: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != item['sha256']:
            raise ValueError('source artifact digest mismatch: ' + name)
    from .actor_pico_stream import stream_settings, worker_settings
    stream_settings(config)
    worker_settings(config)
    return config


def reference_contract():
    slots, _ = contract()
    return SimpleNamespace(names=tuple(n for n in slots if n not in
        ('waist_roll_joint', 'neck_pitch_joint', 'neck_yaw_joint')))


def make_solver(root, config):
    from .actor_pico_gmr import create_retargeter
    # No SDK policy imports: GMR only calls math.quaternion.
    from .actor_pico_source_math import quaternion
    return create_retargeter(root, config, reference_contract(),
                             SimpleNamespace(quaternion=quaternion))


class SourceFrames:
    """Pure frame conditioner; a fake solver is sufficient for unit testing."""
    def __init__(self, solver, config):
        from .actor_pico_stream import StreamFilter, stream_settings
        self.solver = solver
        settings = stream_settings(config)
        self.filter = StreamFilter(settings) if settings is not None else None
        self.origin = None
        self.previous = 0
        self.sequence = 0

    def push(self, body, metadata):
        stamp = int(metadata['timestamp_ns'])
        receipt = float(metadata['timestamp_monotonic'])
        if stamp <= self.previous or stamp <= 0 or not 0 <= time.monotonic()-receipt <= 1.0:
            raise ValueError('invalid/stale Pico sample')
        q, root, error = self.solver.solve(body, stamp_ns=stamp)
        if self.origin is None:
            self.origin = stamp
        if self.filter is not None:
            q, _, root = self.filter((stamp-self.origin)*1e-9, q,
                                     self.solver.last_qpos[:3].copy(), root)
        self.previous = stamp
        self.sequence += 1
        return dict(state='ready', sequence=self.sequence, source_ns=stamp,
                    received_monotonic=receipt, source_realtime_us=int(
                        float(metadata.get('timestamp_realtime', 0)) * 1e6),
                    q=tuple(map(float, q)), root_wxyz=tuple(map(float, root)),
                    gmr_task_error=float(error))


def _latest(output, snapshot):
    try:
        output.put_nowait(snapshot)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        try:
            output.put_nowait(snapshot)
        except queue.Full:
            pass


def _head_snapshot(body, metadata):
    import numpy as np
    poses = np.asarray(body, dtype=float)
    if poses.shape != (24, 7) or not np.isfinite(poses).all():
        raise ValueError('head follow requires finite Pico body poses')
    return dict(raw_body_poses=poses.tolist(),
                received_monotonic=float(metadata['timestamp_monotonic']),
                pico_timestamp_ns=int(metadata['timestamp_ns']))


def _external_service_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _worker(root, output, controls, stop, follow_head=False):
    os.setsid()  # only descendants of this worker are eligible for group cleanup
    service = pico = triggers = log = None
    external_pid = os.environ.get('VITA_PICO_SERVICE_PID')
    external_pid = int(external_pid) if external_pid else None
    try:
        # The one-click shell already owns the receiver for the whole script.
        # A direct launch starts it here, still before the slow solver import,
        # and leaves it up until this worker stops. No headset timeout.
        if external_pid is None:
            service_dir = Path(root)/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
            log_path = Path(root)/'outputs'/'pico_service_action14.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, 'a')
            service = subprocess.Popen([str(service_dir/'RoboticsServiceProcess')],
                                       cwd=service_dir, stdout=log, stderr=subprocess.STDOUT)
        elif not _external_service_alive(external_pid):
            raise RuntimeError('Pico receiver exited before the reader started')
        config = load_source_config(root)
        frames = SourceFrames(make_solver(root, config), config)
        from .pico_input import create_pico_source
        from .actor_pico_buttons import TriggerReader
        from .actor_pico_stream import worker_settings
        pico = create_pico_source(logger=logging.getLogger('sonic-action14-pico'),
            maximum_age=config['pico_max_age_s'],
            duplicate_poll_interval_s=worker_settings(config).reader_poll_interval_s)
        pico.start()
        triggers = TriggerReader(pico.reader.sdk, controls, maximum_age=config['pico_max_age_s'])
        triggers.start()
        _latest(output, {'state': 'waiting_pico'})
        while not stop.is_set():
            if service is not None and service.poll() is not None:
                raise RuntimeError('owned vendor service exited')
            if external_pid is not None and not _external_service_alive(external_pid):
                raise RuntimeError('Pico receiver exited')
            sample = pico.step()
            if sample is None:
                stop.wait(.001)
                continue
            body, _, metadata = sample
            pushed = frames.push(body, metadata)
            if follow_head:
                pushed['head_snapshot'] = _head_snapshot(body, metadata)
            _latest(output, pushed)
    except BaseException as exc:
        _latest(output, {'state': 'failed', 'error': repr(exc)})
    finally:
        if triggers is not None:
            triggers.close()
        if pico is not None:
            pico.stop()
        if service is not None and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=2)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait(timeout=1)
        if log is not None:
            log.close()


class ReferenceSourceProcess:
    def __init__(self, root, *, context=None, follow_head=False):
        ctx = context or mp.get_context('spawn')
        self.output = ctx.Queue(maxsize=2)
        # Seven slots: trigger count/receipts plus grip receipt, stamp, left, right.
        # GripperRunner reads [3:7]; a shorter array disables that path.
        self.controls = ctx.Array('d', [0.]*7)
        self.stop = ctx.Event()
        self.follow_head = bool(follow_head)
        self.process = ctx.Process(target=_worker, args=(
            str(root), self.output, self.controls, self.stop, self.follow_head), daemon=False)

    def start(self):
        self.process.start()

    def latest(self):
        snapshot = None
        while True:
            try:
                snapshot = self.output.get_nowait()
            except queue.Empty:
                return snapshot

    def trigger(self):
        with self.controls.get_lock():
            return int(self.controls[0]), float(self.controls[1]), float(self.controls[2])

    def close(self):
        self.stop.set()
        if self.process.pid is None:
            return
        self.process.join(timeout=3)
        if self.process.is_alive():
            # The isolated process group contains only our worker and owned service.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.process.join(timeout=2)
        if self.process.is_alive():
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.join(timeout=2)
