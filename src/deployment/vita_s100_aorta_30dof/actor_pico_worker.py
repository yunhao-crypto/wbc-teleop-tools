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

from .actor_pico_reference import CausalActorReference
from .actor_pico_gmr import create_retargeter
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


def _capture(root, config, contract, lower, upper, commands, output, stop, controls=None):
    # All descendants (including the owned vendor service) belong to this
    # process group so the parent can clean up even if vendor init blocks.
    os.setsid()
    from .actor_pico_scheduling import input_worker_scheduler
    input_worker_scheduler()
    service = pico = log = buttons = source_log = None
    last_source_stamp = 0
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
        reference = CausalActorReference(contract, math_api)
        retarget = create_retargeter(root, config, contract, math_api)
        source_path = Path(root)/'outputs'/f'actor_pico_gmr_input_{time.time_ns()}.jsonl'
        source_log = source_path.open('x')
        source_log.write(json.dumps(dict(record_type='header',
            retargeting=config['retargeting'], model=config['artifacts']['model']))+'\n')
        sdk = Path(root)/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
        log = (Path(root)/'outputs/pico_service_actor.log').open('a')
        service = subprocess.Popen([str(sdk/'RoboticsServiceProcess')], cwd=sdk,
                                   stdout=log, stderr=subprocess.STDOUT)
        pico = create_pico_source(logger=logging.getLogger('actor-manual-pico'), maximum_age=config['pico_max_age_s'])
        pico.start()
        if controls is not None:
            buttons = TriggerReader(pico.reader.sdk,controls,maximum_age=config['pico_max_age_s'])
            buttons.start()
        _latest(output, dict(state='waiting_pico'))
        while not stop.is_set():
            if service.poll() is not None:
                raise RuntimeError('Owned Pico service exited')
            sample = pico.step()
            if sample is not None:
                body, _, metadata = sample
                stamp = int(metadata['timestamp_ns'])
                if stamp <= last_source_stamp:
                    raise ValueError('Pico timestamps must strictly increase')
                last_source_stamp = stamp
                begin = time.monotonic()
                q, root_quat, residual = retarget.solve(body, robot_root=anchor)
                source_log.write(json.dumps(dict(record_type='gmr_frame',
                    timestamp_ns=stamp, received_monotonic=float(metadata['timestamp_monotonic']),
                    body_poses=body.tolist(), qpos=retarget.last_qpos.tolist(),
                    gmr_task_error=residual), allow_nan=False, separators=(',', ':'))+'\n')
                received = float(metadata['timestamp_monotonic'])
                gap_info = dict(source_gap_s=0. if reference.last_stamp_ns is None else
                    (stamp-reference.last_stamp_ns)*1e-9,
                    receipt_gap_s=0. if reference.last_receive_monotonic is None else
                    received-reference.last_receive_monotonic)
                reference.append(stamp,q,root_quat,received_monotonic=received)
                snapshot = dict(state='collecting', received_monotonic=received,
                    **gap_info,
                    pico_timestamp_ns=int(metadata['timestamp_ns']), gmr_task_error=residual,
                    raw_body_poses=body.tolist(),
                    retarget_q=q.tolist(), retarget_root_wxyz=root_quat.tolist(),
                    source_log=str(source_path),
                    retarget_ms=(time.monotonic()-begin)*1000)
                if reference.ready():
                    motion = reference.motion()
                    snapshot.update(state='ready', q=motion.q, base=motion.base, quat=motion.quat)
                _latest(output, snapshot)
            stop.wait(.005)
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
    """Nonblocking control-side API. No Pico is initialized by start()."""
    def __init__(self, runtime):
        context = multiprocessing.get_context('spawn')
        self.commands, self.output = context.Queue(maxsize=1), context.Queue(maxsize=2)
        self.stop = context.Event()
        self.controls = context.Array('d',[0.,0.,0.])
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
