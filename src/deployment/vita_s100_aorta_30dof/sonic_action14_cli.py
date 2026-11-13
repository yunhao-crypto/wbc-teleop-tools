"""Explicit native SONIC launcher. --plan never creates endpoints or models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time

from .sonic_action14 import CONFIG, MODEL_SHA256, plan


READY_ACTIONS = {'FIXED_STAND', 'HUMANOID'}
# Non-interruptible startup curves on the A-sample profile. One-click waits
# the full curve before the single FIXED_STAND request, matching the EXTERNAL entry.
DAMPING_S = {'PASSIVE': 10., 'QUIET_PASSIVE': 4.}


def _fresh(status, limit_ms=1500.):
    age = status.get('status_age_ms')
    return type(age) in (int, float) and math.isfinite(age) and 0 <= age < limit_ms


def ensure_supported_posture(control, request, *, clock, sleep, timeout=25.):
    """Reach a fresh FIXED_STAND or HUMANOID before action14 activation.

    PASSIVE and QUIET_PASSIVE get one firmware request after their damping
    curve. Any other action is left untouched.
    """
    def snapshot():
        status = control.status()
        return status, status.get('current_action') or ''

    status, action = snapshot()
    if action in READY_ACTIONS and _fresh(status):
        return action
    if action == 'HUMANOID_TRACKING':
        raise RuntimeError('HUMANOID_TRACKING is already active and this process does not own it; '
                           'action14 will not cancel it or activate a second session')
    if action not in DAMPING_S and action not in READY_ACTIONS:
        raise RuntimeError('refusing to leave '+repr(action)+' ; action14 stays inactive')
    if action in DAMPING_S:
        initial, deadline = action, clock()+DAMPING_S[action]
        print('ACTION14_DAMPING: waiting %.0fs for non-interruptible %s before one FIXED_STAND request' % (
            DAMPING_S[action], initial), flush=True)
        while clock() < deadline:
            _, seen = snapshot()
            if seen not in (initial, ''):
                raise RuntimeError('action changed during damping: '+repr(seen))
            sleep(.05)
    status, action = snapshot()
    if not (action in READY_ACTIONS and _fresh(status)):
        print('ACTION14_FIXED_STAND: requesting one FIXED_STAND from '+repr(action), flush=True)
        request()
    end = clock()+timeout
    while clock() < end:
        status, action = snapshot()
        if action in READY_ACTIONS and _fresh(status):
            return action
        sleep(.05)
    raise TimeoutError('FIXED_STAND/HUMANOID was not confirmed; action14 was not activated')


def _firmware_cli(args):
    import os
    import subprocess
    cli = Path('/app/locomotion/humanoid_locomotion_cli')
    env = os.environ.copy()
    config = os.environ.get('VITA_ACTOR_CLI_SESSION_CONFIG')
    if config:
        path = Path(config)
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError('Firmware CLI session must be an existing private 0600 file')
        env['ZENOH_SESSION_CONFIG_URI'] = str(path)
    completed = subprocess.run(
        [str(cli), *args], capture_output=True, text=True, env=env, timeout=12, check=False,
        start_new_session=True)
    if completed.returncode != 0:
        detail = (completed.stdout or '')+(completed.stderr or '')
        raise RuntimeError('firmware CLI failed: '+detail[-1600:])


def request_fixed_stand():
    import uuid
    _firmware_cli(['request', '--mode', 'FIXED_STAND', '--req_id', 'action14-'+uuid.uuid4().hex,
                   '--timeout_ms', '8000'])


def request_passive():
    """One firmware cancel. mode 0 / target_state 0 is PASSIVE, not a stand lead-in."""
    import uuid
    _firmware_cli(['cancel', '--req_id', 'action14-'+uuid.uuid4().hex, '--timeout_ms', '8000'])


def confirm_passive(control, request, *, clock, sleep, timeout=15.):
    """Send one PASSIVE cancel even if status can no longer be read."""
    def snapshot():
        try:
            status = control.status()
        except Exception:
            return None
        return status

    status = snapshot()
    action = '' if status is None else (status.get('current_action') or '')
    if not (status is not None and action == 'PASSIVE' and _fresh(status)):
        request()
    end = clock()+timeout
    last = status
    while clock() < end:
        status = snapshot()
        if status is not None:
            last = status
            action = status.get('current_action') or ''
            if action == 'PASSIVE' and _fresh(status):
                return action
        sleep(.05)
    raise TimeoutError('PASSIVE was not confirmed: '+repr(last))


def verify_model_file(path):
    """File identity only: never construct ORT/native inference sessions."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    if h.hexdigest() != MODEL_SHA256:
        raise ValueError('robot actor SHA256 differs from approved SONIC9700')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reference-only Pico/GMR native action14')
    parser.add_argument('--plan', action='store_true', help='offline only: no model, endpoint or Pico startup')
    parser.add_argument('--confirm-motion', action='store_true')
    parser.add_argument('--fixed-neck', action='store_true', help='publish zero neck targets')
    parser.add_argument('--follow-head', action='store_true', help='publish Pico neck targets on the action14 reference')
    parser.add_argument('--gripper-config', type=Path, help='optional OP3 gripper JSON; no EXTERNAL request')
    parser.add_argument('--bridge', type=Path, help='local humanoid_tracking_client executable')
    parser.add_argument('--robot-model', type=Path,
                        default=Path('/app/locomotion/models/vita02a/actor_motor_theta.onnx'))
    parser.add_argument('--seconds', type=float, default=0., help='0: until Ctrl-C; exit returns to ONNX stand')
    args = parser.parse_args(argv)
    if args.plan:
        if args.confirm_motion or args.fixed_neck or args.follow_head:
            parser.error('--plan cannot be combined with motion, fixed-neck, or follow-head')
        payload = plan()
        if args.gripper_config is not None:
            from .sonic_action14_gripper import describe
            try:
                payload.update(describe(args.gripper_config))
            except (OSError, ValueError) as error:
                parser.error('gripper config: '+str(error))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not args.confirm_motion or args.bridge is None or args.fixed_neck == args.follow_head:
        parser.error('live requires --confirm-motion --bridge and exactly one of --fixed-neck or --follow-head')
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be finite and nonnegative')
    if not args.bridge.is_file() or not args.bridge.stat().st_mode & 0o111:
        parser.error('bridge executable missing')
    # The selected native profile names this exact local robot model. Arbitrary
    # --robot-model alternatives must not validate a file the daemon never uses.
    if args.robot_model.resolve() != Path('/app/locomotion/models/vita02a/actor_motor_theta.onnx').resolve():
        parser.error('robot model must be the fixed native profile model path')
    verify_model_file(args.robot_model)
    from .sonic_action14_source import ReferenceSourceProcess, load_source_config
    from .sonic_action14_bridge import JsonControlBridge
    from .sonic_action14_transport import NativeReferencePublisher
    from .sonic_action14_orchestrator import Action14Session
    root = CONFIG.parents[1]
    load_source_config(root)
    stop = threading.Event()
    passive_on_exit = threading.Event()
    previous = {}

    def on_stop(_signum, _frame):
        passive_on_exit.set()
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, on_stop)
    source = control = publisher = session = gripper = None
    outcome = 'inactive'
    try:
        control = JsonControlBridge(args.bridge)
        publisher = NativeReferencePublisher()
        head = None
        if args.follow_head:
            from .sonic_action14_head import make_head
            head = make_head(root, load_source_config(root))
        session = Action14Session(control, publisher, head=head)
        source = ReferenceSourceProcess(root, follow_head=args.follow_head)
        source.start()
        print('PICO_WAIT: receiver starting; action14 stand does not depend on receiving Pico. '
              'Connect the headset to this robot on TCP 63901 at any time while this script is running.',
              flush=True)
        if args.gripper_config is not None:
            from .sonic_action14_gripper import attach
            gripper = attach(args.gripper_config, publisher.transport, source.controls)
        # From PASSIVE or QUIET_PASSIVE, wait out the non-interruptible damping
        # curve and issue one FIXED_STAND request. No fault reset or motor enable.
        def pause(seconds):
            if stop.wait(seconds):
                raise InterruptedError('startup interrupted before action14 activation')
        posture = ensure_supported_posture(
            control, request_fixed_stand, clock=time.monotonic, sleep=pause)
        print('ACTION14_POSTURE: '+posture, flush=True)
        if not stop.is_set():
            session.initialize()
            if gripper is not None:
                gripper.start()
            started = time.monotonic()
            print('ACTION14_STAND: native policy active. A fresh dual-trigger engages only after the robot has accepted a reference. Ctrl-C requests PASSIVE; stiffness drops and a loaded limb may fall.', flush=True)
            announced = None
            while not stop.is_set() and (args.seconds == 0 or time.monotonic()-started < args.seconds):
                snapshot = source.latest()
                if snapshot is not None:
                    session.accept(snapshot)
                count, event_receipt, controller_receipt = source.trigger()
                if session.following and not 0 <= time.monotonic()-controller_receipt <= .6:
                    session.disengage()
                else:
                    decision = session.update_trigger(count, event_receipt)
                    if decision == 'refused':
                        print('ACTION14_ENGAGE_REFUSED: staying in stand. Trigger again after references are accepted; this attempt will not auto-engage. '+str(session.last_engage_refusal), flush=True)
                if session.last_posture == 'ambiguous' and announced != 'ambiguous':
                    print('ACTION14_POSTURE_AMBIGUOUS: return-to-stand was not confirmed. The native policy may still be active.', flush=True)
                    announced = 'ambiguous'
                session.tick()
                if not source.process.is_alive():
                    raise RuntimeError('reference worker exited')
                stop.wait(.002)
    except Exception:
        if not passive_on_exit.is_set():
            raise
    finally:
        # Ctrl-C/SIGTERM requests PASSIVE and waits for that status. A timed
        # --seconds exit still uses return_to_stand and leaves the policy active.
        try:
            if passive_on_exit.is_set() and control is not None:
                print('ACTION14_PASSIVE: requesting one PASSIVE cancel. Stiffness will drop.', flush=True)
                try:
                    if source is not None:
                        source.close()
                        source = None
                    confirm_passive(control, request_passive, clock=time.monotonic, sleep=time.sleep)
                    print('ACTION14_PASSIVE: status is PASSIVE', flush=True)
                    outcome = 'passive'
                except Exception as error:
                    print('ACTION14_PASSIVE_UNCONFIRMED: '+repr(error), flush=True)
                    outcome = 'passive_unconfirmed'
                if session is not None:
                    session.close(return_to_stand=False)
                elif publisher is not None:
                    publisher.close()
                    if control is not None:
                        control.close()
            elif session is not None:
                outcome = session.close()
            else:
                if publisher is not None:
                    publisher.close()
                if control is not None:
                    control.close()
        finally:
            if gripper is not None:
                gripper.close()
            if source is not None:
                source.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if outcome == 'passive_unconfirmed':
        return 3
    if outcome == 'ambiguous':
        print('ACTION14_POSTURE_AMBIGUOUS: return-to-stand was not confirmed. The native policy may still be active.', flush=True)
        return 3
    if outcome == 'lead_in_accepted':
        print('ACTION14_STAND_LEAD_IN_ACCEPTED: the business result started a stand lead-in. Stiffness remains and balance is not verified.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
