#!/usr/bin/env python3
"""Pinned actor_g1 check/replay or Pico + Body31 read-only shadow inference."""
import argparse
from contextlib import ExitStack
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime


def check(runtime, args, write):
    c = runtime.model.contract
    motion = runtime.math.Motion.hold(c.default)
    for _ in range(args.steps):
        obs, row = runtime.infer_motion(motion, q=c.default, dq=np.zeros(28),
                                        root=[1., 0., 0., 0.], gyro=np.zeros(3))
        write(row, obs)
    runtime.policy.reset()
    return dict(mode='offline_model_check', steps=args.steps, hardware_inputs_checked=False)


def replay(runtime, args, write):
    """Replay explicitly named robot-state arrays plus raw 24x7 Pico frames."""
    if args.input is None:
        raise ValueError('replay requires --input with the documented NPZ schema')
    with np.load(args.input, allow_pickle=False) as f:
        body, stamps = f['body_poses'], f['sample_times_ns']
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f['robot_joint_names']]
        indices = [names.index(n) for n in runtime.model.contract.names]
        q, dq = f['robot_joint_pos'][:, indices], f['robot_joint_vel'][:, indices]
        root, gyro = f['robot_root_quat_wxyz'], f['robot_root_angvel_local']
    count = len(body)
    if (len(set(names)) != len(names) or body.shape != (count, 24, 7)
            or stamps.shape != (count,) or not np.issubdtype(stamps.dtype, np.integer)
            or q.shape != (count, 28) or dq.shape != (count, 28)
            or root.shape != (count, 4) or gyro.shape != (count, 3)):
        raise ValueError('Invalid actor/Pico replay shapes or timestamp dtype')
    completed = 0
    for i in range(count):
        t = (int(stamps[i])-int(stamps[0]))*1e-9
        try:
            runtime.push_pico(body[i], int(stamps[i]), received_monotonic=t, robot_root=root[i])
        except ValueError as exc:
            raise ValueError(f'Pico replay raw frame {i}: {exc}') from exc
        if not runtime.reference.ready():
            continue
        obs, row = runtime.infer(q=q[i], dq=dq[i], root=root[i], gyro=gyro[i], now_monotonic=t)
        write(dict(row, raw_frame=i, pico_timestamp_ns=int(stamps[i])), obs)
        completed += 1
        if completed >= args.steps:
            break
    if not completed:
        raise RuntimeError('Replay never supplied the complete future-reference window')
    return dict(mode='raw_pico_replay', steps=completed, input=str(args.input),
                reference_delay_s=runtime.reference.delay, hardware_inputs_checked=False)


def shadow(runtime, args, write):
    from deployment.vita_s100_aorta_30dof.pico_input import create_pico_source
    from deployment.vita_s100_aorta_30dof.sdkmain_adapter import raw_state_from_arrays
    from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources
    from locomotion_aorta.pull import PullStateSource
    if not os.environ.get('ZENOH_SESSION_CONFIG_URI'):
        raise RuntimeError('Use start_actor_pico.sh shadow with a reviewed robot peer profile')
    verify_vendor_sources()
    stop = False
    def stopped(*_):
        nonlocal stop
        stop = True
    previous = {n: signal.signal(n, stopped) for n in (signal.SIGINT, signal.SIGTERM)}
    sdk = runtime.root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    steps = 0
    try:
        with ExitStack() as resources:
            log = resources.enter_context((runtime.root/'outputs/pico_service_actor.log').open('a'))
            service = subprocess.Popen([str(sdk/'RoboticsServiceProcess')], cwd=sdk,
                                       stdout=log, stderr=subprocess.STDOUT)
            def close_service():
                if service.poll() is None:
                    service.terminate()
                    try:
                        service.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        service.kill()
                        service.wait(timeout=2)
            resources.callback(close_service)
            source = PullStateSource(group=args.group, node_name='vita-actor-pico-shadow')
            resources.callback(source.close)
            if source.group != args.group:
                raise RuntimeError('SDK source group mismatch')
            pico = create_pico_source(logger=logging.getLogger('actor-pico'), maximum_age=.6)
            resources.callback(pico.stop)
            pico.start()
            start = deadline = time.monotonic()
            last_state_ns = None
            print('ACTOR_SHADOW: waiting for Body31 and Pico; first Pico frame calibrates against actor default pose.', flush=True)
            while not stop and time.monotonic()-start < args.seconds:
                if service.poll() is not None:
                    raise RuntimeError('Pico service exited; see outputs/pico_service_actor.log')
                delay = deadline-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if stop:
                    break
                cycle_started = time.monotonic()
                try:
                    state = source.wait_state(timeout=min(1., args.seconds))
                except TimeoutError:
                    if not steps and time.monotonic()-start < args.startup_timeout:
                        continue
                    raise
                raw_state_from_arrays(state, received_monotonic_ns=state.received_monotonic_ns)
                now = time.monotonic()
                age = now-state.received_monotonic_ns*1e-9
                source_age = (time.time_ns()-state.ts_state_pub_us*1000)*1e-9
                if not 0 <= age <= .1 or not -.005 <= source_age <= .1:
                    raise TimeoutError('Body31 feedback is stale or clock contract differs')
                if state.protocol_version != 2 or state.roster_version != 1:
                    raise ValueError('Expected the pinned protocol2 roster1 body31 state')
                q, dq, root, gyro = runtime.policy.state(state)
                sample = pico.step()
                if sample is not None:
                    body, _, metadata = sample
                    runtime.push_pico(body, int(metadata['timestamp_ns']),
                                      received_monotonic=float(metadata['timestamp_monotonic']), robot_root=root)
                if runtime.reference.ready():
                    if last_state_ns is not None and state.ts_state_real_us < last_state_ns:
                        raise ValueError('Body31 source clock regressed')
                    obs, row = runtime.infer(q=q, dq=dq, root=root, gyro=gyro,
                                             now_monotonic=time.monotonic())
                    cycle_ms = (time.monotonic()-cycle_started)*1000
                    write(dict(row, state_id=int(state.state_id),
                               state_age_s=age, source_publication_age_s=source_age,
                               input_retarget_inference_ms=cycle_ms,
                               cycle_overrun=cycle_ms > runtime.model.contract.period*1000,
                               pico_timestamp_ns=runtime.reference.last_stamp_ns), obs)
                    steps += 1
                    if steps % 50 == 0:
                        print(json.dumps(dict(steps=steps, inference_ms=row['inference_ms'],
                                              retarget_ms=runtime.last_retarget_ms,
                                              ik_residual_rad=runtime.last_residual, actions_sent=0)), flush=True)
                elif now-start > args.startup_timeout:
                    raise TimeoutError('Pico has not provided a complete actor reference window')
                last_state_ns = state.ts_state_real_us
                deadline += runtime.model.contract.period
                now = time.monotonic()
                if deadline < now:
                    skipped = math.floor((now-deadline)/runtime.model.contract.period)+1
                    deadline += skipped*runtime.model.contract.period
                    write(dict(record_type='cycle_overrun', skipped_ticks=skipped,
                               completed_cycle_with_log_ms=(now-cycle_started)*1000, actions_sent=0))
    finally:
        for n, handler in previous.items():
            signal.signal(n, handler)
    if not steps and not stop:
        raise RuntimeError('No Pico-driven actor inference completed')
    return dict(mode='s100_pico_body31_shadow', steps=steps, stopped_by_signal=stop,
                reference_delay_s=runtime.reference.delay, hardware_inputs_checked=steps > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'replay', 'shadow'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--backend', choices=('auto', 'sdk', 'ort'), default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--startup-timeout', type=float, default=30)
    parser.add_argument('--group', default='default')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--record-observations', action='store_true')
    args = parser.parse_args()
    if (args.steps < 1 or not math.isfinite(args.seconds) or not 1 <= args.seconds <= 7200
            or not math.isfinite(args.startup_timeout) or not 1 <= args.startup_timeout <= 120):
        parser.error('Invalid step/duration/startup budget')
    logging.basicConfig(level=logging.INFO)
    runtime = ActorPicoRuntime(args.root, backend=args.backend, threads=args.threads)
    args.log = args.log or args.root/'outputs'/f'actor_pico_{args.mode}_{time.time_ns()}.jsonl'
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('x', buffering=1) as stream:
        def write(row, observation=None):
            if args.record_observations and observation is not None:
                row = dict(row, observation=observation.tolist())
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
        write(dict(record_type='header', model=runtime.config['artifacts']['model'],
                   contract=dict(input_dim=1520, action_dim=28, policy_hz=50),
                   scope='inference_only', retargeter='calibrated_segment_orientation_ik',
                   reference_delay_s=runtime.reference.delay, actions_sent=0))
        try:
            report = {'check': check, 'replay': replay, 'shadow': shadow}[args.mode](runtime, args, write)
        except Exception as exc:
            write(dict(record_type='error', error=repr(exc), actions_sent=0))
            raise
        report.update(actions_sent=0, control_handles_created=0, log=str(args.log),
                      s100_inference_timing='not_measured' if args.mode != 'shadow' else 'see_trace')
        write(dict(record_type='summary', **report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
