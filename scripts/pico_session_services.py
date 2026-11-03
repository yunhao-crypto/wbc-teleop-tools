#!/usr/bin/env python3
"""Own a camera relay alongside an actor, or a connection-only XRoboToolkit session."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.pico_camera_bridge import TOPICS


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.1):
            return True
    except OSError:
        return False


def stop_child(child, timeout=30.):
    """Let the actor release control and its own Pico worker before video stops."""
    if child is None:
        return
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait(timeout=3.)
    else:
        child.wait()


def supervise(command, *, camera_command=None, camera_env=None, camera_log=None,
              camera_ready=None, primary_log=None, ready_port=None, seconds=0.,
              startup_timeout=10., shutdown_timeout=30., report=print):
    """Only children created here are signalled; a video failure never stops an actor."""
    primary = camera = None
    files = []
    requested = 0
    camera_announced = camera_failed = ready_announced = False

    def stopped(number, _frame):
        nonlocal requested
        requested = requested or number

    previous = {s: signal.signal(s, stopped) for s in (signal.SIGINT, signal.SIGTERM)}
    start = time.monotonic()
    try:
        if camera_command:
            try:
                output = open(camera_log, 'a') if camera_log else subprocess.DEVNULL
                if camera_log:
                    files.append(output)
                camera = subprocess.Popen(camera_command, env=camera_env, stdout=output,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                report('PICO_VIDEO: starting; log='+str(camera_log), flush=True)
            except OSError as error:
                camera_failed = True
                report('PICO_VIDEO: unavailable: '+str(error)+'; primary session continues', flush=True)
        if requested:
            return 128+requested
        output = open(primary_log, 'a') if primary_log else None
        if primary_log:
            files.append(output)
        primary = subprocess.Popen(command, stdout=output,
            stderr=subprocess.STDOUT if primary_log else None,
            stdin=subprocess.DEVNULL if ready_port else None)
        signalled = False
        stop_deadline = None
        while True:
            if requested and not signalled:
                if primary.poll() is None:
                    primary.send_signal(requested)
                signalled = True
                stop_deadline = time.monotonic()+shutdown_timeout
            result = primary.poll()
            if result is not None:
                if ready_port and not ready_announced and not requested:
                    raise RuntimeError('XRoboToolkit service exited before listening; see '+str(primary_log))
                return 128+requested if requested else (result if result >= 0 else 128-result)
            if stop_deadline is not None and time.monotonic() >= stop_deadline:
                raise TimeoutError('Primary session did not finish its shutdown')
            if ready_port and not ready_announced and not requested:
                if listening(ready_port):
                    ready_announced = True
                    report(f'PICO_CONNECT: ready on TCP {ready_port}; connect the headset to this robot IP', flush=True)
                elif time.monotonic()-start >= startup_timeout:
                    raise TimeoutError('XRoboToolkit listener did not start; see '+str(primary_log))
            if camera is not None and not camera_failed:
                if camera.poll() is not None:
                    camera_failed = True
                    report('PICO_VIDEO: unavailable; see '+str(camera_log)+'; primary session continues', flush=True)
                elif not camera_announced and camera_ready and Path(camera_ready).is_file():
                    try:
                        info = json.loads(Path(camera_ready).read_text())
                        if info['pid'] == camera.pid:
                            report('PICO_VIDEO: ready on TCP '+str(info['address'][1])+
                                   '; topic='+info['topic'], flush=True)
                            camera_announced = True
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
            if seconds and time.monotonic()-start >= seconds:
                if ready_port and not ready_announced:
                    raise TimeoutError('Connection test ended before service readiness')
                return 0
            time.sleep(.05)
    finally:
        try:
            stop_child(primary, shutdown_timeout)
        finally:
            try:
                stop_child(camera, 5.)
            finally:
                if camera_ready and camera is not None:
                    Path(camera_ready).unlink(missing_ok=True)
                for stream in files:
                    stream.close()
                for number, handler in previous.items():
                    signal.signal(number, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--connect-only', action='store_true', help='Receive Pico and video only; no model or robot control')
    p.add_argument('--no-camera', action='store_true')
    p.add_argument('--camera-topic', choices=TOPICS, default=TOPICS[0])
    p.add_argument('--camera-port', type=int, default=13579)
    p.add_argument('--camera-session-profile', type=Path, default=Path('/app_param/zenoh/factory.json5'))
    p.add_argument('--seconds', type=float, default=0., help='Connection-only test duration; 0 waits for Ctrl-C')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0 or not 1 <= args.camera_port <= 65535 or args.camera_port == 63901:
        p.error('Invalid duration or video port (63901 is reserved for XRoboToolkit)')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (args.connect_only and command) or (not args.connect_only and (not command or args.seconds)):
        p.error('Use --connect-only [--seconds N], or -- <actor command>')
    root = args.root.resolve()
    output = root/'outputs'
    output.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    camera_log = output/f'pico_camera_{stamp}.log'
    camera_ready = output/f'pico_camera_{stamp}.ready.json'
    camera_command = None
    camera_env = dict(os.environ, ZENOH_SESSION_CONFIG_URI=str(args.camera_session_profile),
                      AORTA_GROUP='default', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if not args.no_camera:
        if args.camera_session_profile.is_file():
            camera_command = ['nice', '-n', '10', os.environ.get('VITA_CAMERA_PYTHON', sys.executable),
                str(root/'scripts/pico_camera_bridge.py'), '--port', str(args.camera_port),
                '--topic', args.camera_topic, '--stats', str(output/f'pico_camera_{stamp}.json'),
                '--ready-file', str(camera_ready)]
        else:
            print('PICO_VIDEO: profile missing: '+str(args.camera_session_profile)+
                  '; video disabled for this session', flush=True)
    primary_log = None
    if args.connect_only:
        if listening(63901):
            p.error('TCP 63901 is already owned by another service; stop its owner before starting connect mode')
        sdk = root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
        executable = sdk/'RoboticsServiceProcess'
        if not executable.is_file():
            p.error('XRoboToolkit runtime is absent; bootstrap this S100 deployment first')
        command = [str(executable)]
        primary_log = output/f'pico_pc_{stamp}.log'
        os.chdir(sdk)
        print('PICO_CONNECT: connection/video test; no policy or robot-control process is started', flush=True)
    return supervise(command, camera_command=camera_command, camera_env=camera_env,
        camera_log=camera_log, camera_ready=camera_ready, primary_log=primary_log,
        ready_port=63901 if args.connect_only else None, seconds=args.seconds)


if __name__ == '__main__':
    raise SystemExit(main())
