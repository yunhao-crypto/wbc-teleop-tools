"""Optional Pico capture/IK in an isolated process, never a robot controller.

The process starts idle before EXTERNAL. Vendor initialization occurs only on
CONNECT, outside the control interpreter/GIL. Snapshots cross a bounded queue;
the controller keeps current policy history, SDK ownership and sequence.
"""
import logging
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import time

from .actor_supermimic import create_reference
from .actor_pico_gmr import create_retargeter
from .actor_pico_stream import SourceClock, StreamFilter, stream_settings, worker_settings
from .actor_pico_buttons import TriggerReader
from .actor_pico_runtime import sdk_math


def _latest(output, item):
    try:
        output.put_nowait(item)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            return
        try:
            output.put_nowait(item)
        except queue.Full:
            pass


def thread_affinity(cpus):
    """Restrict only the calling thread; children inherit its mask at creation."""
    available = os.sched_getaffinity(0)
    if cpus:
        if not set(cpus) <= available:
            raise ValueError(f'Requested Pico CPUs {list(cpus)} outside available CPUs {sorted(available)}')
        os.sched_setaffinity(0, cpus)
    return sorted(os.sched_getaffinity(0))


class ReferenceStream:
    """GMR plus causal reference for one Pico session; each frame yields a snapshot.

    With a ``pico_stream`` configuration the snapshot carries the conditioned
    raw samples and their clock offset (``stream``) for control-clock
    playback; otherwise it carries the window anchored at the newest sample.
    ``clock`` is the local monotonic clock shared with the control process.
    """
    def __init__(self, root, config, contract, math_api, source_log=None, source_path=None,
                 clock=time.monotonic):
        self.reference = create_reference(contract, math_api, config)
        self.retarget = create_retargeter(root, config, contract, math_api)
        self.source_log, self.source_path = source_log, source_path
        self.last_source_stamp = 0
        self.clock = clock
        self.settings = stream_settings(config)
        self.filter = StreamFilter(self.settings) if self.settings is not None else None
        self.source_clock = SourceClock()
        self.pending_log = None

    def push(self, body, metadata, anchor):
        reference, retarget = self.reference, self.retarget
        stamp = int(metadata['timestamp_ns'])
        if stamp <= self.last_source_stamp:
            raise ValueError('Pico timestamps must strictly increase')
        self.last_source_stamp = stamp
        begin, cpu = time.monotonic(), time.thread_time()
        q, root_quat, residual = retarget.solve(body, robot_root=anchor)
        retarget_ms, retarget_cpu_ms = (time.monotonic()-begin)*1000, (time.thread_time()-cpu)*1000
        raw_q, raw_root = q, root_quat
        base = retarget.last_qpos[:3].copy()
        received = float(metadata['timestamp_monotonic'])
        source_s = 0. if reference.origin_ns is None else (stamp-reference.origin_ns)*1e-9
        if self.filter is not None:
            q, base, root_quat = self.filter(source_s, q, base, root_quat)
        gap_info = dict(source_gap_s=0. if reference.last_stamp_ns is None else
            (stamp-reference.last_stamp_ns)*1e-9,
            receipt_gap_s=0. if reference.last_receive_monotonic is None else
            received-reference.last_receive_monotonic)
        reference.append(stamp,q,root_quat,received_monotonic=received,base=base)
        available = self.clock()
        offset = self.source_clock.update(source_s, available)
        # Written after the snapshot is published (flush_log), never inside
        # the measured retarget time or ahead of the control loop.
        self.pending_log = (stamp, received, available, body, retarget.last_qpos.copy(), residual)
        snapshot = dict(state='collecting', received_monotonic=received,
            **gap_info,
            pico_timestamp_ns=stamp, gmr_task_error=residual,
            raw_body_poses=body.tolist(),
            retarget_q=raw_q.tolist(), retarget_root_wxyz=list(map(float, raw_root)),
            source_log=None if self.source_path is None else str(self.source_path),
            retarget_ms=retarget_ms, retarget_cpu_ms=retarget_cpu_ms)
        if reference.ready():
            if self.settings is not None:
                snapshot.update(state='ready', stream=dict(reference.export(), clock_offset_s=offset))
            else:
                motion = reference.motion()
                snapshot.update(state='ready', q=motion.q, base=motion.base, quat=motion.quat)
        return snapshot

    def flush_log(self):
        if self.source_log is None or self.pending_log is None:
            self.pending_log = None
            return
        stamp, received, available, body, qpos, residual = self.pending_log
        self.pending_log = None
        self.source_log.write(json.dumps(dict(record_type='gmr_frame',
            timestamp_ns=stamp, received_monotonic=received, available_monotonic=available,
            body_poses=body.tolist(), qpos=qpos.tolist(),
            gmr_task_error=residual), allow_nan=False, separators=(',', ':'))+'\n')


def _capture(root, config, contract, lower, upper, commands, output, stop, controls=None):
    # All descendants (including the owned vendor service) belong to this
    # process group so the parent can clean up even if vendor init blocks.
    os.setsid()
    from .actor_pico_scheduling import input_worker_scheduler
    input_worker_scheduler()
    service = pico = log = buttons = source_log = None
    try:

        while not stop.is_set():
            try:
                anchor = commands.get(timeout=.1)
                break
            except queue.Empty:
                continue
        else:
            return
        from .pico_input import create_pico_source
        math_api = sdk_math(root, config)
        stream = ReferenceStream(root, config, contract, math_api)
        source_path = Path(root)/'outputs'/f'actor_pico_gmr_input_{time.time_ns()}.jsonl'
        source_log = source_path.open('x')
        sdk = Path(root)/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
        log = (Path(root)/'outputs/pico_service_actor.log').open('a')
        service = subprocess.Popen([str(sdk/'RoboticsServiceProcess')], cwd=sdk,
                                   stdout=log, stderr=subprocess.STDOUT)
        # The vendor process keeps the inherited mask. SDK and reader threads
        # inherit worker_cpus; narrow only the GMR thread after they start.
        settings = worker_settings(config)
        worker_affinity = thread_affinity(settings.worker_cpus)
        poll = settings.reader_poll_interval_s
        pico = create_pico_source(logger=logging.getLogger('actor-manual-pico'),
            maximum_age=config['pico_max_age_s'], duplicate_poll_interval_s=poll)
        pico.start()
        if controls is not None:
            buttons = TriggerReader(pico.reader.sdk,controls,maximum_age=config['pico_max_age_s'])
            buttons.start()
        gmr_affinity = thread_affinity(settings.gmr_cpus)
        source_log.write(json.dumps(dict(record_type='header',
            retargeting=config['retargeting'], model=config['artifacts']['model'],
            pico_stream=config.get('pico_stream'), pico_worker=config.get('pico_worker'),
            worker_affinity=worker_affinity,
            gmr_affinity=gmr_affinity, reader_poll_interval_s=poll))+'\n')
        stream.source_log, stream.source_path = source_log, source_path
        _latest(output, dict(state='waiting_pico'))
        while not stop.is_set():
            if service.poll() is not None:
                raise RuntimeError('Owned Pico service exited')
            sample = pico.step()
            if sample is not None:
                body, _, metadata = sample
                _latest(output, stream.push(body, metadata, anchor))
                stream.flush_log()
            else:
                # Only idle polls sleep: a processed frame never delays the next one.
                stop.wait(.001)
    except BaseException as error:
        _latest(output, dict(state='failed', error=repr(error)))
    finally:
        if buttons is not None:
            buttons.close()
        if pico is not None:
            pico.stop()
        if service is not None and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=2)
            except subprocess.TimeoutExpired:
                service.kill(); service.wait(timeout=1)
        if log is not None:
            log.close()
        if source_log is not None:
            source_log.close()


class PicoReferenceProcess:
    """Nonblocking control-side API. No Pico is initialized by start().

    buttons=False omits the controller trigger reader (automatic following).
    """
    def __init__(self, runtime, *, buttons=True):
        context = multiprocessing.get_context('spawn')
        self.commands, self.output = context.Queue(maxsize=1), context.Queue(maxsize=2)
        self.stop = context.Event()
        self.controls = context.Array('d',[0.,0.,0.]) if buttons else None
        self.process = context.Process(target=_capture, args=(str(runtime.root), runtime.config,
            runtime.model.contract, runtime.policy.lower, runtime.policy.upper,
            self.commands, self.output, self.stop,self.controls), daemon=True)
        self.connected = False
        self.closed = False
        self.previous = dict(state='idle')

    def start(self):
        self.process.start()

    def connect(self, robot_root):
        if self.connected:
            return False
        self.commands.put_nowait(tuple(float(x) for x in robot_root))
        self.connected = True
        self.previous = dict(state='connecting')
        return True

    def snapshot(self):
        # Drain at most the queue capacity; never spend a control cycle on a
        # backlog. A cached snapshot keeps its ORIGINAL source receipt clock.
        for _ in range(2):
            try:
                self.previous = self.output.get_nowait()
            except queue.Empty:
                break
        if not self.process.is_alive() and self.previous['state'] != 'failed':
            self.previous = dict(state='failed', error='Pico worker exited')
        if self.controls is None:
            return dict(self.previous)
        with self.controls.get_lock():
            count,event_time,controller_time = self.controls[:]
        return dict(self.previous,trigger_event_seq=int(count),
                    trigger_event_monotonic=event_time,controller_received_monotonic=controller_time)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                self.process.terminate()
            self.process.join(timeout=1)
        # Also clean up an owned service left by an already-exited worker.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.join(timeout=1)
        self.commands.close(); self.output.close()
        self.commands.cancel_join_thread(); self.output.cancel_join_thread()
