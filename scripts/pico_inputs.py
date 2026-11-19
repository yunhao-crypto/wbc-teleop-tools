#!/usr/bin/env python3
"""pico_smpl / pico_teleop continuous shadow inference, with zero actuation."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def load_replay(path, expected):
    import numpy as np
    from deployment.vita_s100_aorta_30dof.pico_threepoint_frontend import sha
    if sha(path) != expected: raise ValueError('Pico replay SHA256 mismatch')
    with np.load(path, allow_pickle=False) as z:
        body, stamps, received = (z[k].copy() for k in ('body_poses', 'sample_times_ns', 'receive_monotonic_ns'))
        meta = json.loads(z['metadata'].item())
    n = len(body)
    if (body.shape != (n, 24, 7) or not 2 <= n <= 180000 or stamps.shape != (n,) or received.shape != (n,)
            or stamps.dtype != np.int64 or received.dtype != np.int64 or not np.isfinite(body).all()
            or np.any(stamps <= 0) or np.any(received <= 0) or np.any(np.diff(received) <= 0)
            or meta.get('schema') != 'pico_body24_record_v1' or meta.get('complete') is not True):
        raise ValueError('Need complete, finite raw Pico24 replay with int64 device/receive clocks and provenance')
    # Unlike offline conversion, replay deliberately sends device gaps/backward
    # clocks to the same fault state machine; timestamps are never repaired.
    return body, stamps, received, meta


def synthetic_feedback(flow):
    import numpy as np
    from deployment.vita_s100_aorta_30dof.pico_realtime import Feedback
    return Feedback(np.array(flow.contract.default), np.zeros(28), np.array([1., 0., 0., 0.]), np.zeros(3))


def main(route=None):
    if route is not None and len(sys.argv) == 1: sys.argv.append('auto')
    parser = argparse.ArgumentParser(description=__doc__)
    if route is None: parser.add_argument('route', choices=('pico_smpl', 'pico_teleop'))
    parser.add_argument('mode', choices=('doctor', 'replay', 'shadow', 'auto', 'live'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--feedback', choices=('synthetic-stand', 'body31'))
    parser.add_argument('--wrist-mode', choices=('stand',))
    parser.add_argument('--backend', choices=('auto', 'ort', 'sdk'), default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--startup-timeout', type=float)
    parser.add_argument('--sdk-dir', type=Path)
    parser.add_argument('--dll-dir', action='append', type=Path, default=[])
    parser.add_argument('--group', default='default')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--trigger-script', type=Path, help='Replay-only explicit button fault/transition scenario JSON')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--record-observations', action='store_true')
    parser.add_argument('--require-tracking', action='store_true', help='Fail validation unless following inference actually ran')
    parser.add_argument('--trace-capacity', type=int, default=3000, help='Keep only the most recent inference samples in the NPZ trace')
    parser.add_argument('--pico-service', choices=('auto', 'external'), default='auto')
    parser.add_argument('--service-exe', type=Path, help='Installed platform-compatible RoboticsServiceProcess')
    parser.add_argument('--service-arg', action='append', default=[], help='One service argument, passed without a shell')
    parser.add_argument('--service-port', type=int, default=63901)
    parser.add_argument('--service-ready-timeout', type=float, default=10.)
    args = parser.parse_args()
    route = route or args.route
    requested_mode = args.mode
    # Refuse live before importing any inference, SDK or transport module.
    if args.mode == 'live':
        parser.error('Motion disabled: these two routes only validate no-action inference. Existing live gates/protections remain in force.')
    from deployment.vita_s100_aorta_30dof.pico_service import default_sdk_directory
    if args.sdk_dir is None: args.sdk_dir = default_sdk_directory(args.root)
    if args.mode == 'doctor':
        from deployment.vita_s100_aorta_30dof.pico_service import resolve_executable, listening
        try:
            service_path = str(resolve_executable(args.root, args.service_exe, args.sdk_dir)); service_error = None
        except (ValueError, OSError) as exc: service_path = None; service_error = str(exc)
        if args.sdk_dir: sys.path.insert(0, str(args.sdk_dir.resolve()))
        spec = importlib.util.find_spec('xrobotoolkit_sdk')
        print(json.dumps(dict(route=route, sdk_path=spec.origin if spec else None,
            pico_service_mode=args.pico_service, service_executable=service_path, service_error=service_error,
            service_port_listening=listening(args.service_port), service_started=False,
            sdk_initialized=False, actions_sent=0, control_handles_created=0,
            status='SDK_FOUND_NOT_INITIALIZED' if spec else 'MISSING_PICO_BINDING'), indent=2)); return
    if args.mode == 'auto':
        args.mode = 'shadow'
        if args.feedback is None: args.feedback = 'synthetic-stand'
        if route == 'pico_smpl' and args.wrist_mode is None: args.wrist_mode = 'stand'
        if args.out is None: args.out = args.root/'outputs'/f'{route}_shadow_{time.time_ns()}'
    if args.seconds is None: args.seconds = 0. if requested_mode == 'auto' else 30.
    if args.startup_timeout is None: args.startup_timeout = 180. if requested_mode == 'auto' else 15.
    if args.feedback is None: parser.error('Declare --feedback synthetic-stand or body31; no silent feedback fallback')
    if route == 'pico_smpl' and args.wrist_mode != 'stand': parser.error('pico_smpl requires --wrist-mode stand')
    if route == 'pico_teleop' and args.wrist_mode is not None: parser.error('--wrist-mode applies only to pico_smpl')
    if not 0 <= args.seconds <= 3600 or not 0 < args.startup_timeout <= 180 or args.threads < 1 or not 1 <= args.trace_capacity <= 30000:
        parser.error('Invalid duration/startup/thread budget')
    if args.out is None or args.out.exists(): parser.error('Use a new --out directory')
    if args.mode == 'replay' and (args.input is None or not args.sha256 or args.feedback != 'synthetic-stand'):
        parser.error('replay requires --input, --sha256 and explicit synthetic-stand feedback; no robot connection')
    if args.mode == 'shadow' and (args.input is not None or args.sha256 or args.trigger_script is not None):
        parser.error('Live-device shadow cannot inject replay inputs or trigger scripts')
    if args.mode == 'replay' and (args.service_exe is not None or args.service_arg or args.pico_service != 'auto'):
        parser.error('Offline replay never starts or connects a Pico service')
    if not 1 <= args.service_port <= 65535 or not 0 < args.service_ready_timeout <= 60:
        parser.error('Invalid service port/readiness budget')
    import numpy as np
    from deployment.vita_s100_aorta_30dof.pico_realtime import PicoFlow, PERIOD_NS, next_deadline_ns
    from deployment.vita_s100_aorta_30dof.pico_threepoint_frontend import sha
    if args.feedback == 'synthetic-stand':
        from deployment.vita_s100_aorta_30dof.pico_shadow_math import prepare
        prepare(args.root, route)
    flow = PicoFlow(args.root, route, backend=args.backend, threads=args.threads, wrist_mode=args.wrist_mode)
    args.out.mkdir(parents=True, exist_ok=False)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    previous_signals = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    obs, actions, modes, ticks = (deque(maxlen=args.trace_capacity) for _ in range(4))
    event_index = 0
    peak_ms = 0.; infer_ms = deque(maxlen=3000); error = None; close_report = {}; input_meta = None
    ingress_peak_ms = 0.; ingress_overruns = ingress_samples = 0
    source = validator = receiver = service = None
    receiver_state = None
    now_ns = time.monotonic_ns(); start_ns = now_ns
    try:
        with ExitStack() as resources:
            log = resources.enter_context((args.out/'inference.jsonl').open('x', encoding='utf-8'))
            def write(value):
                log.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'); log.flush()
            def flush_flow_events():
                nonlocal event_index
                for event in flow.events[event_index:]: write(event)
                event_index = len(flow.events)
            def infer(now, feedback):
                nonlocal peak_ms
                begin = time.perf_counter()
                try: result = flow.step(now, feedback)
                finally: flush_flow_events()
                elapsed = (time.perf_counter()-begin)*1000; peak_ms = max(peak_ms, elapsed)
                row = dict(record_type='inference_tick', time_ns=int(now), mode=result['mode'],
                    inferred=result['inferred'], total_tick_ms=elapsed, feedback_source=args.feedback,
                    actions_sent=0, control_handles_created=0)
                if result['inferred']:
                    candidate = result['report']['raw_action']; observation = result['observation']
                    if not np.isfinite(candidate).all() or not np.isfinite(observation).all():
                        raise ValueError('Nonfinite inference candidate')
                    infer_ms.append(elapsed); actions.append(candidate); modes.append(result['mode']); ticks.append(now)
                    if args.record_observations: obs.append(observation.copy())
                    row.update(raw_action_abs_max=float(np.max(np.abs(candidate))), candidate_only=True)
                write(row)
                return result
            if args.mode == 'replay':
                body, stamps, received, input_meta = load_replay(args.input, args.sha256)
                script = []
                if args.trigger_script:
                    scenario = json.loads(args.trigger_script.read_text(encoding='utf-8'))
                    if scenario.get('schema') != 'pico_shadow_trigger_scenario_v1' or scenario.get('synthetic') is not True:
                        raise ValueError('Replay script must explicitly declare synthetic trigger events')
                    script = scenario['events']
                    times = [x['time_s'] for x in script]
                    if not np.isfinite(times).all() or any(t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
                        raise ValueError('Trigger scenario must have finite increasing nonnegative times')
                    for x in script:
                        if not np.isfinite([x['left'], x['right']]).all() or min(x['left'], x['right']) < 0 or max(x['left'], x['right']) > 1:
                            raise ValueError('Trigger values outside0..1')
                start_ns = int(received[0]); index = button_index = 0; left = right = 0.
                feedback = synthetic_feedback(flow)
                end_ns = int(received[-1])+1_500_000_000
                if args.seconds: end_ns = min(end_ns, start_ns+int(args.seconds*1e9))
                for now_ns in range(start_ns, end_ns+1, PERIOD_NS):
                    if stopped: break
                    while index < len(body) and int(received[index]) <= now_ns:
                        flow.push(body[index], int(stamps[index]), int(received[index]), feedback); index += 1
                    t = (now_ns-start_ns)*1e-9
                    while button_index < len(script) and script[button_index]['time_s'] <= t:
                        left, right = script[button_index]['left'], script[button_index]['right']; button_index += 1
                    # Synthetic controller packets are explicitly labelled in the report.
                    if args.trigger_script: flow.button((True, now_ns, 0., left, right), now_ns)
                    infer(now_ns, feedback)
                if sha(args.input) != args.sha256: raise ValueError('Replay input changed during validation')
            else:
                from deployment.vita_s100_aorta_30dof.pico_input_process import PicoInputProcess
                from deployment.vita_s100_aorta_30dof.pico_service import PicoService, library_directories
                service = PicoService(args.root, args.out/'pico_service.log', mode=args.pico_service,
                    executable=args.service_exe, arguments=args.service_arg, sdk_dir=args.sdk_dir,
                    port=args.service_port, ready_timeout=args.service_ready_timeout, stopped=lambda: stopped)
                resources.callback(service.close)
                service_info = service.start()
                write(dict(record_type='pico_service_ready', **service_info, actions_sent=0, control_handles_created=0))
                print(f'PICO_SERVICE_READY: TCP {args.service_port}; owned={service_info["owned"]}; connect Pico to this computer/board service.', flush=True)
                if args.feedback == 'body31':
                    if not os.environ.get('ZENOH_SESSION_CONFIG_URI'):
                        raise RuntimeError('body31 shadow requires the already configured ZENOH_SESSION_CONFIG_URI; no config is generated')
                    from deployment.vita_s100_aorta_30dof.actor_pico_live import Inputs, require_peer_profile
                    from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources
                    require_peer_profile(Path(os.environ['ZENOH_SESSION_CONFIG_URI']))
                    verify_vendor_sources()
                    from types import SimpleNamespace
                    from locomotion_aorta.pull import PullStateSource
                    source = PullStateSource(group=args.group, node_name='vita-'+route+'-shadow')
                    resources.callback(source.close)
                    # The SDK rejects native BodyState candidates until robot
                    # discovery binds this read-only source to its runtime.
                    source.get_robot_info(timeout=min(5.,args.startup_timeout))
                    validator = Inputs(SimpleNamespace(config={'pico_max_age_s': .6}, policy=flow.runtime.policy), None, None, time)
                    # Allow subscription startup within the explicit budget;
                    # subsequent reads retain the normal 20ms wait and 100ms
                    # freshness checks, including this first measured frame.
                    validator.validate_state(source.wait_state(timeout=args.startup_timeout))
                libraries = library_directories(args.root, service.report['executable'])
                receiver = PicoInputProcess(sdk_dir=args.sdk_dir,
                    dll_dirs=list(dict.fromkeys([*map(str,args.dll_dir), *(libraries if os.name=='nt' else [])])),
                    library_dirs=libraries)
                print('PICO_NO_ACTION: first2s of valid input must be a quiet neutral standing pose. Double triggers toggle reference following; no robot actions are sent.', flush=True)
                receiver.start(); resources.callback(receiver.close)
                start_ns = deadline_ns = time.monotonic_ns(); next_progress = start_ns+1_000_000_000
                feedback = synthetic_feedback(flow) if source is None else None
                while not stopped and (args.seconds == 0 or time.monotonic_ns()-start_ns < args.seconds*1e9):
                    # Windows monotonic_ns can have a coarse clock quantum.
                    # An early sleep wakeup must not execute a duplicate tick.
                    while not stopped and time.monotonic_ns() < deadline_ns:
                        wait = (deadline_ns-time.monotonic_ns())*1e-9
                        if wait > 0: time.sleep(min(wait, .02))
                    if stopped: break
                    now_ns = time.monotonic_ns()
                    service.check()
                    if source is not None:
                        state = source.wait_state(timeout=.02)
                        from deployment.vita_s100_aorta_30dof.pico_realtime import Feedback
                        feedback = Feedback(*validator.validate_state(state))
                    poll_ns = time.monotonic_ns()
                    try:
                        pose, buttons, receiver_state = receiver.poll(poll_ns)
                        now_ns = time.monotonic_ns()
                        if receiver_state.get('state') == 'failed':
                            raise RuntimeError('Pico receiver failed: '+receiver_state.get('error', 'unknown'))
                        if pose is not None:
                            if not 0 <= now_ns-pose[2] <= 100_000_000:
                                flow.lose(now_ns, 'Receiver sample is stale/future before ingestion')
                            else: flow.push(*pose, feedback)
                    finally:
                        ingress_ms=(time.monotonic_ns()-poll_ns)*1e-6
                        ingress_peak_ms=max(ingress_peak_ms,ingress_ms)
                        ingress_overruns+=ingress_ms>PERIOD_NS*1e-6;ingress_samples+=1
                        flush_flow_events()
                    # Ingestion can include a one-time neutral calibration.
                    # Infer from a newly acquired, normally validated Body31
                    # frame, never from its pre-ingestion feedback or stamps.
                    if source is not None:
                        state=source.wait_state(timeout=.02)
                        feedback=Feedback(*validator.validate_state(state))
                    now_ns = time.monotonic_ns()
                    buttons = (buttons[0], buttons[1], buttons[2]+(now_ns-poll_ns)*1e-9, buttons[3], buttons[4])
                    if source is not None: validator.validate_state(state)
                    flow.button(buttons, now_ns)
                    infer(now_ns, feedback)
                    if flow.calibration is None and now_ns-start_ns > args.startup_timeout*1e9:
                        raise TimeoutError('No complete valid2s neutral calibration before startup timeout')
                    if now_ns >= next_progress:
                        print(json.dumps(flow.summary(), ensure_ascii=False), flush=True); next_progress = now_ns+1_000_000_000
                    # Skip missed ticks; never burst old controller references.
                    deadline_ns = next_deadline_ns(deadline_ns, time.monotonic_ns())
                close_report = receiver.close(); receiver = None
                # ExitStack callback is idempotent; leave the process closed.
            if flow.calibration is None: raise RuntimeError('No complete valid neutral calibration; inference alone is not a pass')
            if args.require_tracking and flow.track_ticks == 0: raise RuntimeError('No following inference completed')
            flow.stop(now_ns)
            flush_flow_events()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        for s, handler in previous_signals.items(): signal.signal(s, handler)
        if receiver is not None and receiver.close_report is not None:
            close_report.update(receiver.close_report)
        report = flow.summary()
        report.update(status='PASS_NO_ACTION_INFERENCE' if error is None and flow.inferences else 'FAILED_NO_ACTION_VALIDATION',
            execution_mode=args.mode, feedback_source=args.feedback, error=error, stopped_by_signal=stopped,
            launcher_mode=requested_mode, pico_service=dict(service.report) if service is not None else None,
            current_live_pico_verified=args.mode == 'shadow' and error is None and flow.frames > 0 and
                receiver_state is not None and not receiver_state.get('binding_test_fixture', False),
            receiver_state=receiver_state,
            tracking_verified=flow.track_ticks > 0,
            robot_feedback_read=args.feedback == 'body31', robot_controller_created=False,
            sdk_math_mode='pinned_python_only_no_native_transport' if args.feedback == 'synthetic-stand' else 'actual_SDK',
            trace_steps=len(actions), trace_is_tail=flow.inferences > len(actions),
            hardware_command_envelope_verified=False,
            inference_tick_median_ms=float(np.median(infer_ms)) if infer_ms else None, inference_tick_peak_ms=peak_ms,
            timing_sample_steps=len(infer_ms),
            timing_scope='PicoFlow.step only; excludes SDK read, pose ingestion, calibration and transport',
            input_ingress_peak_ms=ingress_peak_ms,input_ingress_overrun_ticks=ingress_overruns,
            input_ingress_samples=ingress_samples,
            input_ingress_scope='receiver.poll + pose ingestion/calibration; excludes robot feedback reads and inference',
            model_artifacts=flow.runtime.config['artifacts'],
            input_path=str(args.input.resolve()) if args.input else None, input_sha256=args.sha256, input_metadata=input_meta,
            synthetic_triggers=args.trigger_script is not None,
            trigger_script_sha256=sha(args.trigger_script) if args.trigger_script else None,
            closed_loop=False, hardware_motion_verified=False, **close_report)
        if flow.calibration is not None:
            (args.out/'calibration.json').write_text(json.dumps(flow.calibration, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if actions:
            arrays = dict(time_monotonic_ns=np.asarray(ticks, np.int64), raw_action=np.asarray(actions, np.float32), mode=np.asarray(modes))
            if args.record_observations: arrays['observations'] = np.asarray(obs, np.float32)
            np.savez_compressed(args.out/'trace.npz', **arrays)
        (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if error is not None: raise RuntimeError(error)
    if not flow.inferences: raise RuntimeError('No inference completed')


if __name__ == '__main__': main()
