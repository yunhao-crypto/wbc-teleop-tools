"""Optional Pico capture/IK in an isolated process, never a robot controller.

The process starts idle before EXTERNAL. Vendor initialization occurs only on
CONNECT, outside the control interpreter/GIL. Snapshots cross a bounded queue;
the controller keeps current policy history, SDK ownership and sequence.
"""
from dataclasses import replace
import logging
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import time

from .actor_pico_reference import CausalActorReference, PicoOrientationRetargeter, PicoReferenceGapError
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


def _capture(root, config, contract, lower, upper, commands, output, stop):
    # All descendants (including the owned vendor service) belong to this
    # process group so the parent can clean up even if vendor init blocks.
    os.setsid()
    service = pico = log = None
    epoch = 0
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
        reference = CausalActorReference(contract, math_api,
            maximum_gap=config['reference_maximum_gap_s'])
        retarget = PicoOrientationRetargeter(
            urdf=Path(root)/config['artifacts']['urdf']['path'],
            contract=replace(contract, lower=lower, upper=upper), mapping=config['segments'],
            math_api=math_api, iterations=config['ik_iterations'],
            max_residual_rad=config['ik_max_residual_rad'])
        sdk = Path(root)/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
        log = (Path(root)/'outputs/pico_service_actor.log').open('a')
        service = subprocess.Popen([str(sdk/'RoboticsServiceProcess')], cwd=sdk,
                                   stdout=log, stderr=subprocess.STDOUT)
        pico = create_pico_source(logger=logging.getLogger('actor-manual-pico'), maximum_age=.1)
        pico.start()
        _latest(output, dict(state='waiting_pico',reference_epoch=epoch))
        while not stop.is_set():
            if service.poll() is not None:
                raise RuntimeError('Owned Pico service exited')
            sample = pico.step()
            if sample is not None:
                body, _, metadata = sample
                begin = time.monotonic()
                q, root_quat, residual = retarget.solve(body, robot_root=anchor)
                received = float(metadata['timestamp_monotonic'])
                stamp = int(metadata['timestamp_ns'])
                gap_info = {}
                try:
                    reference.append(stamp, q, root_quat, received_monotonic=received)
                except PicoReferenceGapError as error:
                    # Keep the capture connection alive, but discard every
                    # sample before the gap. Following pins this epoch and
                    # cancels on a change, even if it misses 'collecting'.
                    gap_info = dict(source_gap_s=error.gap_s,
                        receipt_gap_s=received-reference.last_receive_monotonic)
                    epoch += 1
                    reference = CausalActorReference(contract, math_api,
                        maximum_gap=config['reference_maximum_gap_s'])
                    reference.append(stamp,q,root_quat,received_monotonic=received)
                snapshot = dict(state='collecting', received_monotonic=received,
                    reference_epoch=epoch,**gap_info,
                    pico_timestamp_ns=int(metadata['timestamp_ns']), ik_max_residual_rad=residual,
                    retarget_ms=(time.monotonic()-begin)*1000)
                if reference.ready():
                    motion = reference.motion()
                    snapshot.update(state='ready', q=motion.q, base=motion.base, quat=motion.quat)
                _latest(output, snapshot)
            stop.wait(.005)
    except BaseException as error:
        _latest(output, dict(state='failed', error=repr(error),reference_epoch=epoch))
    finally:
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


class PicoReferenceProcess:
    """Nonblocking control-side API. No Pico is initialized by start()."""
    def __init__(self, runtime):
        context = multiprocessing.get_context('spawn')
        self.commands, self.output = context.Queue(maxsize=1), context.Queue(maxsize=2)
        self.stop = context.Event()
        self.process = context.Process(target=_capture, args=(str(runtime.root), runtime.config,
            runtime.model.contract, runtime.policy.lower, runtime.policy.upper,
            self.commands, self.output, self.stop), daemon=True)
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
        return self.previous

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
