#!/usr/bin/env python3
"""Measure the production Pico capture worker with frozen GMR inputs, without actuation.

Real Pico packets exercise the SDK, body/trigger readers, IPC and raw logging.
Every processed packet solves the next frozen fixture frame, so baseline and
configured runs do identical IK work. No robot controller/transport is created.
--baseline uses 0.1 ms duplicate polling and unrestricted input/GMR threads;
it keeps the same IK, filters, playback and weights as the configured variant.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/deployment/unitree_g1_ros2_29dof/src')]


def capture(options, commands, output, stop, controls):
    sys.path.append(str(Path(options['runtime_root'])/'build/pico'))
    import numpy as np
    from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
    from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import SessionFixture
    from deployment.vita_s100_aorta_30dof import actor_pico_worker as worker
    from bench_pico_gmr import summary
    fixture = SessionFixture(options['fixture'])
    runtime = ActorPicoRuntime(options['runtime_root'], backend='auto', policy='supermimic',
                               human_height=fixture.human_height_m)
    config = json.loads((ROOT/'config/actor_pico_supermimic.json').read_text())
    if config['artifacts']['model']['sha256'] != fixture.meta['model']['sha256']:
        raise ValueError('Benchmark fixture differs from the configured model')
    config['retargeting']['human_height_m'] = fixture.human_height_m
    if options['baseline']:
        config = copy.deepcopy(config)
        config['pico_worker'].update(worker_cpus=[], gmr_cpus=[], reader_poll_interval_s=.0001)
    count = min(options['frames'], len(fixture.frame_stamp_ns))
    rows, paths, digest = [], [], hashlib.sha256()
    original = worker.ReferenceStream

    class FrozenStream(original):
        def push(self, body, metadata, anchor):
            i = len(rows)
            before, cpu = time.monotonic(), time.thread_time()
            fixed = dict(metadata, timestamp_ns=int(fixture.frame_stamp_ns[i]))
            snapshot = super().push(fixture.frame_body[i].astype(np.float64), fixed, anchor)
            full_cpu, full = (time.thread_time()-cpu)*1000, (time.monotonic()-before)*1000
            digest.update(self.retarget.last_qpos.astype('<f8', copy=False).tobytes())
            rows.append(dict(frame=i, live_stamp_ns=int(metadata['timestamp_ns']),
                retarget_ms=snapshot['retarget_ms'], retarget_cpu_ms=snapshot['retarget_cpu_ms'],
                snapshot_ms=full, snapshot_cpu_ms=full_cpu))
            if self.source_path is not None and not paths:
                paths.append(str(self.source_path))
            if len(rows) == count:
                stop.set()
            return snapshot

    worker.ReferenceStream = FrozenStream
    worker._capture(options['runtime_root'], config, runtime.model.contract,
                    runtime.policy.lower, runtime.policy.upper, commands, output, stop, controls)
    if len(rows) != count:
        raise RuntimeError(f'Capture ended after {len(rows)}/{count} benchmark frames; inspect worker output')
    header = json.loads(Path(paths[0]).read_text().splitlines()[0])
    report = dict(schema='vita_pico_live_gmr_benchmark_v1', actions_sent=0,
        variant='baseline' if options['baseline'] else 'configured', frames=len(rows),
        fixture_sha256=hashlib.sha256(Path(options['fixture']).read_bytes()).hexdigest(),
        qpos_sha256=digest.hexdigest(), pico_stream=config['pico_stream'], pico_worker=config['pico_worker'], capture_header=header,
        wall_ms=summary([r['retarget_ms'] for r in rows]),
        cpu_ms=summary([r['retarget_cpu_ms'] for r in rows]),
        snapshot_ms=summary([r['snapshot_ms'] for r in rows]),
        raw_gmr_logs=paths, rows=rows)
    with Path(options['output']).open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=ROOT,
                        help='installed SDK/model root; default is this checkout')
    parser.add_argument('--fixture', type=Path,
                        default=ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz')
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--startup-timeout', type=float, default=90.)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or not 0 < args.startup_timeout <= 300:
        parser.error('frames must be positive and startup-timeout must lie in (0, 300] seconds')
    if args.output.exists():
        parser.error('output already exists; choose a new report path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_root = args.runtime_root.resolve()
    sdk = args.runtime_root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='disable',
        QT_PLUGIN_PATH=str(sdk/'plugins'), QT_QPA_PLATFORM='offscreen',
        XDG_RUNTIME_DIR=str(args.runtime_root/'build/pico/actor_xdg'))
    os.environ['LD_LIBRARY_PATH'] = ':'.join(map(str, [
        args.runtime_root/'venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs',
        args.runtime_root/'build/pico', sdk, sdk/'lib', sdk/'SDK/arm64', '/usr/hobot/lib']))+':'+os.environ.get('LD_LIBRARY_PATH','')
    # SDK/model math is read-only; the same operator lock excludes active control.
    with (args.runtime_root/'build/operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        context = multiprocessing.get_context('spawn')
        commands, output = context.Queue(maxsize=1), context.Queue(maxsize=2)
        stop, controls = context.Event(), context.Array('d', [0.,0.,0.])
        process = context.Process(target=capture, args=(vars(args), commands, output, stop, controls))
        commands.put((1.,0.,0.,0.))
        start, first, last_notice = time.monotonic(), None, 0.
        failure = None
        process.start()
        try:
            while process.is_alive():
                now = time.monotonic()
                for _ in range(2):
                    try:
                        snapshot = output.get_nowait()
                    except queue.Empty:
                        break
                    if snapshot.get('state') == 'failed':
                        failure = snapshot.get('error');stop.set()
                    if snapshot.get('state') in ('collecting','ready') and first is None:
                        first = now
                if first is None and now-start > args.startup_timeout:
                    failure = 'Timed out waiting for live Pico body data';stop.set()
                if now-start > args.startup_timeout+max(60., args.frames*.2):
                    failure = 'Benchmark exceeded its total time budget';stop.set()
                if now-last_notice > 10:
                    print('GMR_BENCH: '+('measuring live receiver with frozen inputs' if first else
                        'waiting for live Pico; no robot control'), flush=True)
                    last_notice = now
                if failure:
                    break
                time.sleep(.02)
        finally:
            stop.set();process.join(timeout=6.)
            if process.is_alive():
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: process.terminate()
                process.join(timeout=3.)
            if process.is_alive():
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: process.kill()
                process.join(timeout=2.)
            for q in (commands, output): q.close();q.join_thread()
        if failure or process.exitcode:
            raise RuntimeError(failure or f'Benchmark worker exited with status {process.exitcode}')
    report = json.loads(args.output.read_text())
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
