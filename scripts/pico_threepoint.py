#!/usr/bin/env python3
"""Pico-only capture/calibration/export. Never imports robot control or inference."""
import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def capture(sdk, args, front, np):
    """Record unique same-packet samples with a bounded device-only session."""
    if not callable(getattr(sdk, 'get_body_pose_snapshot', None)):
        raise RuntimeError('Need atomic get_body_pose_snapshot; separate pose/timestamp getters are not accepted')
    if not 1 <= args.seconds <= 120 or not 1 <= args.startup_timeout <= 60:
        raise ValueError('Capture duration must be1..120s and startup timeout1..60s')
    if args.mode == 'record' and (args.out is None or args.out.exists()):
        raise ValueError('record requires a new --out NPZ path')
    sdk.init(); frames = []; stamps = []; received = []; complete = False; error = None
    try:
        print('PICO_ONLY: no robot control. Waiting for5 advancing valid body frames.', flush=True)
        gate = front.TimingGate(); ready = 0; deadline = time.monotonic()+args.startup_timeout
        last_error = None
        while ready < 5:
            if time.monotonic() > deadline: raise TimeoutError('No usable continuous Pico stream; last error: '+str(last_error))
            available, stamp, body = sdk.get_body_pose_snapshot(); now = time.monotonic_ns()
            if available:
                try:
                    if gate.accept(stamp, body, now): ready += 1
                except ValueError as exc:
                    last_error = str(exc); gate = front.TimingGate(); ready = 0
            time.sleep(.002)
        previous_stamp = gate.last_stamp; gate = front.TimingGate()
        begin = None; first_phase = True
        print('Ready. First2seconds: hold a comfortable neutral pose; then move upper body only.', flush=True)
        while begin is None or (time.monotonic_ns()-begin)*1e-9 < args.seconds:
            available, stamp, body = sdk.get_body_pose_snapshot(); now = time.monotonic_ns()
            if begin is None and (not available or stamp == previous_stamp):
                if time.monotonic() > deadline: raise TimeoutError('Stream stopped before recording')
                time.sleep(.002); continue
            if not available:
                gate.check_age(now); time.sleep(.002); continue
            if gate.accept(stamp, body, now):
                if begin is None: begin = now
                frames.append(np.asarray(body, np.float32).copy()); stamps.append(int(stamp)); received.append(now)
                if first_phase and (now-begin)*1e-9 >= 2:
                    print('NEUTRAL_DONE: begin gentle upper-body movement; keep pelvis/feet near the start.', flush=True)
                    first_phase = False
            time.sleep(.002)
        complete = True
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        failures = []
        def close():
            try: sdk.close()
            except Exception as exc: failures.append(str(exc))
        thread = threading.Thread(target=close, daemon=True); thread.start(); thread.join(2)
        if thread.is_alive() or failures:
            complete = False; error = error or 'Pico SDK close did not finish cleanly'
    metadata = dict(schema='pico_body24_record_v1', complete=complete, error=error,
        input='atomic get_body_pose_snapshot; raw24joint SMPL-topology world XYZ metres and XYZW',
        receive_clock='Python first_visibility_monotonic_ns, not kernel/callback arrival or end-to-end sensor age',
        requested_seconds=args.seconds, neutral_seconds=2., data_origin='current_live_pico_capture',
        python=sys.executable, sdk_path=getattr(sdk, '__file__', None), actions_sent=0, robot_controllers_created=0)
    result = dict(status='PASS_PICO_RECEIVE' if complete else 'INCOMPLETE_PICO_RECEIVE', frames=len(frames),
        error=error, actions_sent=0, robot_controllers_created=0)
    if args.mode == 'record' and frames:
        path = args.out if complete else args.out.with_name(args.out.stem+'.partial.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            np.savez_compressed(f, body_poses=np.asarray(frames, np.float32), sample_times_ns=np.asarray(stamps, np.int64),
                receive_monotonic_ns=np.asarray(received, np.int64), metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
        result.update(path=str(path.resolve()), sha256=front.sha(path))
    if len(stamps) > 1:
        dt = np.diff(np.asarray(stamps, np.int64))*1e-9
        result.update(device_duration_s=(stamps[-1]-stamps[0])*1e-9, median_source_fps=float(1/np.median(dt)), max_gap_s=float(dt.max()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not complete: raise RuntimeError(error or 'Pico capture incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('doctor', 'probe', 'record', 'calibrate', 'convert'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--sdk-dir', type=Path)
    parser.add_argument('--dll-dir', type=Path, action='append', default=[])
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--startup-timeout', type=float, default=10)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--calibration-sha256')
    parser.add_argument('--standing-reference', type=Path)
    parser.add_argument('--site-interface', type=Path)
    parser.add_argument('--neutral-seconds', type=float, default=2)
    parser.add_argument('--scale', type=float)
    args = parser.parse_args()
    if args.sdk_dir:
        if not args.sdk_dir.is_dir(): parser.error('--sdk-dir must exist')
        sys.path.insert(0, str(args.sdk_dir.resolve()))
    with contextlib.ExitStack() as stack:
        if os.name == 'nt':
            for path in ([args.sdk_dir] if args.sdk_dir else [])+args.dll_dir:
                if not path.is_dir(): parser.error('DLL directory must exist')
                stack.enter_context(os.add_dll_directory(str(path.resolve())))
        if args.mode == 'doctor':
            numpy_spec = importlib.util.find_spec('numpy'); sdk_spec = importlib.util.find_spec('xrobotoolkit_sdk')
            report = dict(python=sys.executable, version=platform.python_version(), machine=platform.machine(),
                numpy_path=numpy_spec.origin if numpy_spec else None, sdk_path=sdk_spec.origin if sdk_spec else None,
                sdk_initialized=False, actions_sent=0, robot_controllers_created=0)
            if sdk_spec:
                try:
                    sdk = importlib.import_module('xrobotoolkit_sdk')
                    report.update(sdk_import_ok=True, atomic_body_snapshot=callable(getattr(sdk, 'get_body_pose_snapshot', None)))
                except Exception as exc: report.update(sdk_import_ok=False, error=str(exc))
            report['status'] = 'PASS_LOCAL_CAPTURE_ENV' if numpy_spec and report.get('sdk_import_ok') and report.get('atomic_body_snapshot') else 'MISSING_OR_INCOMPATIBLE_PICO_BINDING'
            print(json.dumps(report, ensure_ascii=False, indent=2)); return
        import numpy as np
        from deployment.vita_s100_aorta_30dof import pico_threepoint_frontend as front
        if args.mode in ('probe', 'record'):
            sdk = importlib.import_module('xrobotoolkit_sdk'); capture(sdk, args, front, np); return
        if args.input is None or not args.sha256 or args.out is None or args.out.exists():
            parser.error('calibrate/convert require --input, --sha256 and a new --out path')
        body, times, metadata = front.load_raw(args.input, args.sha256)
        if args.mode == 'calibrate':
            standing = args.standing_reference or args.root/'artifacts/actor_threepoint/standing_reference.npz'
            sites = args.site_interface or args.root/'artifacts/actor_threepoint/teleop_site_interface.json'
            result = front.fit_calibration(body, times, standing, sites, neutral_seconds=args.neutral_seconds, scale=args.scale)
            result.update(raw_calibration_source_sha256=args.sha256, data_origin=metadata.get('data_origin'))
            save_json(args.out, result)
        else:
            if args.calibration is None or not args.calibration_sha256 or front.sha(args.calibration) != args.calibration_sha256:
                parser.error('convert requires a hash-matched --calibration file')
            c = json.loads(args.calibration.read_text(encoding='utf-8'))
            result = front.convert_recording(body, times, c)
            result['metadata'] = np.array(json.dumps(dict(raw_source_sha256=args.sha256,
                calibration_sha256=args.calibration_sha256, data_origin=metadata.get('data_origin'),
                temporal_mode='offline bracketing linear+SLERP; no gaps interpolated', startup_s=.8, blend_s=.8)))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('xb') as f: np.savez_compressed(f, **result)
        print(json.dumps(dict(status='PASS_'+args.mode.upper(), path=str(args.out.resolve()), sha256=front.sha(args.out), actions_sent=0), indent=2))


if __name__ == '__main__': main()
