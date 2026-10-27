"""Explicit, attended fixed-stand and pose preparation before actor inference.

No nodes, services or motion are created on import. The firmware CLI is the
only action-request implementation; EXTERNAL remains owned by the real SDK.
"""
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import uuid

import numpy as np

from .actor_pico_live import phase, StatusGuard, status_row

POSE_SPEED = .15  # rad/s, peak of cosine position interpolation
POSE_MIN_SECONDS = 5.
POSE_MAX_SECONDS = 15.
POSE_ERROR = .10
SETTLE_ERROR = .04
SETTLE_SPEED = .05
SETTLE_SECONDS = .5
SETTLE_TIMEOUT = 5.
ONLINE_START_GYRO = .10


def vector(value, label):
    from .sdkmain_adapter import BODY_TO_LEGACY
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(31, float(a))
    elif a.shape == (42,):
        a = np.array([a[i] if i is not None else 0. for i in BODY_TO_LEGACY])
    if a.shape != (31,) or not np.isfinite(a).all():
        raise ValueError('Invalid full-body startup '+label)
    a = a.copy(); a[14] = 0.
    return a


class StartupPlan:
    def __init__(self, runtime, envelope, config_path, *, online=False):
        import yaml
        root = yaml.safe_load(Path(config_path).read_text())
        self.online = online
        action = root['actions']['FIXED_STAND']
        params = action['params']
        frames = params['keyframes']
        if (action['generator'] != 'builtin::keyframe' or len(frames) != 1
                or frames[0]['timestamp'] != 0):
            raise ValueError('One-click requires the reviewed single-frame FIXED_STAND configuration')
        frame = frames[0]
        self.fixed_q = vector(frame['positions'], 'positions')
        self.fixed_kp = vector(frame.get('kp', params.get('kp')), 'kp')
        self.fixed_kd = vector(frame.get('kd', params.get('kd')), 'kd')
        self.target, _ = runtime.policy.decode(np.zeros(28))
        runtime.policy.reset()
        self.fixed = replace(self.target, joints=tuple(replace(j, q=float(q), dq=0., tau=0.,
            kp=float(kp), kd=float(kd)) for j,q,kp,kd in
            zip(self.target.joints, self.fixed_q, self.fixed_kp, self.fixed_kd)))
        envelope.check(self.fixed)
        envelope.check(self.target)
        for i, bounds in enumerate(envelope.limits):
            if bounds is not None and (self.fixed_kp[i] <= 0 or self.fixed_kd[i] <= 0):
                raise ValueError('Startup FIXED_STAND must control each installed body joint')
        self.damping = {}
        for name in ('PASSIVE', 'QUIET_PASSIVE'):
            p = root['actions'][name]['params']
            frames = p['keyframes']
            stamps = np.asarray([f['timestamp'] for f in frames], dtype=float)
            if (not len(stamps) or not np.isfinite(stamps).all() or stamps[0] != 0
                    or np.any(np.diff(stamps) <= 0) or not 0 < stamps[-1] <= 90
                    or np.any(vector(frames[-1].get('kd', p.get('kd')), 'terminal damping') != 0)):
                raise ValueError('Unknown PASSIVE/QUIET_PASSIVE damping completion contract')
            self.damping[name] = float(stamps[-1])+2.  # Conservative full curve from first observation.
        self.review = dict(sequence=['wait_damping', 'FIXED_STAND',
                                     'EXTERNAL_policy_lead_in' if online else 'EXTERNAL_pose_prepare',
                                     'ONNX_stand', 'manual_Pico'],
            fixed_stand=[dict(q=j.q,kp=j.kp,kd=j.kd) for j in self.fixed.joints],
            pose_speed_rad_s=POSE_SPEED, pose_min_seconds=POSE_MIN_SECONDS,
            settle_error_rad=SETTLE_ERROR, settle_speed_rad_s=SETTLE_SPEED,
            damping_wait_seconds=self.damping,
            entry_policy='policy-online' if online else 'joint-ramp',
            fixed_stand_ready='active FIXED_STAND, healthy hardware-bounded feedback and stationary joints' if online
                else 'measured positions at FIXED_STAND target and stationary joints',
            online_start_gyro_limit_rad_s=ONLINE_START_GYRO if online else None,
            startup_exit='CANCEL to PASSIVE on an incomplete FIXED_STAND startup; loaded limbs may fall')

    def check_hardware_pose(self, envelope, state):
        envelope.check(replace(self.fixed, joints=tuple(replace(j,q=float(state.q[i]))
            if envelope.limits[i] is not None else j for i,j in enumerate(self.fixed.joints))))


class FirmwareActions:
    """Call the installed firmware CLI once; never retry rejected motion."""
    def __init__(self, stop, clock, monitor):
        self.stop, self.clock, self.monitor = stop, clock, monitor

    def __call__(self, action):
        cli = '/app/locomotion/humanoid_locomotion_cli'
        command = ([cli, 'request', '--mode', 'FIXED_STAND', '--req_id', 'actor-'+uuid.uuid4().hex]
                   if action == 'FIXED_STAND' else [cli, 'cancel'])
        process = subprocess.Popen(command+['--timeout_ms', '5000'], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            deadline = self.clock.monotonic()+6.
            while process.poll() is None:
                if self.stop.is_set() and action == 'FIXED_STAND':
                    raise InterruptedError('Startup interrupted during FIXED_STAND request; result may be unknown')
                self.monitor()
                if self.clock.monotonic() >= deadline:
                    raise TimeoutError('Firmware action request timed out; no retry')
                self.clock.sleep(.02)
            output = process.communicate()[0]
            if process.returncode != 0:
                # Preserve bounded real firmware reason, including non-interruptible.
                raise RuntimeError('Firmware '+action+' rejected/failed: '+output[-1600:])
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=1.)


def fixed_stand_startup(plan, envelope, runtime, args, controller, inputs, stop,
                        write, clock, request_action=None, poll_command=lambda:None):
    """After the single on-site confirmation, wait then issue one fixed request.

    SDK action_phase describes the EXTERNAL session, not damping completion.
    Wait a full configured damping curve on continuous fresh feedback before
    the single request; a late/rejected admission is a failure, not a retry.
    """
    allowed = ('PASSIVE','QUIET_PASSIVE','FIXED_STAND','')
    guard = StatusGuard(clock, initial_actions=allowed)
    baseline = None
    def monitor():
        nonlocal baseline
        if poll_command():
            print('AUTO_START: CONNECT/PICO rejected before ONNX stand; not queued',flush=True)
        state,q,dq,root,gyro = inputs.read(.1)
        status = guard.check(controller.status)
        counts = (status.reject_count,status.ingress_drop_count,status.generation,status.last_reject_code)
        if baseline is None: baseline = counts
        if counts != baseline or status.timeout_hold:
            raise RuntimeError('Startup reject/drop/fault/generation changed; no retry')
        plan.check_hardware_pose(envelope,state)
        return state,status,root,gyro
    request_action = request_action or FirmwareActions(stop,clock,monitor)
    state,status,root,gyro = monitor()
    initial = status.current_action
    if initial in plan.damping:
        started=clock.monotonic(); deadline=started+plan.damping[initial]; next_print=started
        retired = False
        while clock.monotonic() < deadline:
            if stop.is_set(): return False
            state,status,root,gyro=monitor()
            # TickActive retires a completed keyframe and clears its name.
            # Continue the conservative timed wait, accepting only this known
            # completion edge. A fresh action appearing afterward is a fault.
            if status.current_action == '':
                retired = True
            elif status.current_action != initial or retired:
                raise RuntimeError('Action changed while waiting for damping completion')
            if clock.monotonic() >= next_print:
                print(f'AUTO_START: waiting for {initial} damping, {deadline-clock.monotonic():.0f}s remaining',flush=True)
                next_print=clock.monotonic()+5.
            clock.sleep(.02)
    requested = False
    try:
        if stop.is_set(): return False
        # Treat an existing verified FIXED_STAND as already requested.
        requested = True
        if initial != 'FIXED_STAND':
            write(dict(record_type='fixed_stand_requested',actions_sent=0))
            request_action('FIXED_STAND')
        deadline=clock.monotonic()+args.startup_timeout; settled=None; next_print=clock.monotonic()
        while True:
            if stop.is_set(): raise InterruptedError('Startup interrupted after FIXED_STAND')
            state,status,root,gyro=monitor()
            tilt=math.acos(float(np.clip(runtime.math.rotation(root)[2,2],-1.,1.)))
            gyro_speed=float(np.linalg.norm(gyro))
            # The zero-time FIXED_STAND keyframe can finish between two status
            # publications. After the one accepted request, verify actual q/dq
            # continuously and require either that action or no active action.
            named_ready = status.current_action == 'FIXED_STAND' if plan.online else status.current_action in ('FIXED_STAND','')
            if named_ready:
                position_ready = plan.online or np.max(np.abs(state.q-plan.fixed_q)) <= SETTLE_ERROR
                angular_motion_ready = not plan.online or gyro_speed<=ONLINE_START_GYRO
                # Tilt validity is already checked by policy.state against the
                # reviewed tracking profile. Display it, but do not invent a
                # tighter angular admission threshold from a single sample.
                good=position_ready and angular_motion_ready and np.max(np.abs(state.dq)) <= SETTLE_SPEED
                settled = (settled if settled is not None else clock.monotonic()) if good else None
                if settled is not None and clock.monotonic()-settled >= SETTLE_SECONDS:
                    print('AUTO_START: FIXED_STAND feedback ready; entering EXTERNAL '+
                          ('policy-online lead-in' if plan.online else 'pose preparation'),flush=True)
                    write(dict(record_type='fixed_stand_confirmed',actions_sent=0,
                               entry_policy=plan.review['entry_policy'],q=state.q.tolist(),dq=state.dq.tolist(),
                               root_wxyz=root.tolist(),gyro=gyro.tolist(),tilt_deg=math.degrees(tilt)))
                    return True
            elif status.current_action != initial:
                raise RuntimeError('Unexpected action during FIXED_STAND activation')
            if clock.monotonic() >= deadline:
                i=int(np.argmax(np.abs(state.q-plan.fixed_q)));v=int(np.argmax(np.abs(state.dq)))
                raise TimeoutError(f'FIXED_STAND did not settle; no EXTERNAL request sent; '
                    f'action={status.current_action!r}, {envelope.body.joint_names[i]} '
                    f'q={state.q[i]:.5f}, fixed_target={plan.fixed_q[i]:.5f}; '
                    f'{envelope.body.joint_names[v]} dq={state.dq[v]:.5f}; '
                    f'IMU tilt={math.degrees(tilt):.2f} deg, gyro_norm={gyro_speed:.3f} rad/s')
            if clock.monotonic() >= next_print:
                i=int(np.argmax(np.abs(state.q-plan.fixed_q)));v=int(np.argmax(np.abs(state.dq)))
                print(f'AUTO_START: verifying FIXED_STAND, action={status.current_action!r}, '
                      f'max position error={abs(state.q[i]-plan.fixed_q[i]):.3f} rad ({envelope.body.joint_names[i]}), '
                      f'max speed={abs(state.dq[v]):.3f} rad/s ({envelope.body.joint_names[v]}), '
                      f'IMU tilt={math.degrees(tilt):.2f} deg',flush=True)
                write(dict(record_type='fixed_stand_feedback',current_action=status.current_action,
                    q=state.q.tolist(),dq=state.dq.tolist(),position_target=plan.fixed_q.tolist(),
                    root_wxyz=root.tolist(),gyro=gyro.tolist(),tilt_deg=math.degrees(tilt),
                    entry_policy=plan.review['entry_policy'],actions_sent=0))
                next_print=clock.monotonic()+1.
            clock.sleep(.02)
    except BaseException:
        if requested:
            # A failed/unknown request can still have activated. One cancellation,
            # followed by actual status evidence; never proceed or retry motion.
            cancel_startup(controller, request_action, clock, write, baseline[2])
        raise


def cancel_startup(controller, request_action, clock, write, generation):
    try:
        current=controller.status
        if (current is None or current.current_action not in ('PASSIVE','QUIET_PASSIVE','FIXED_STAND','')
                or phase(current) != 'IDLE' or current.generation != generation):
            raise RuntimeError('Startup ownership/state changed; no cancellation of an unknown session')
        request_action('CANCEL')
        end=clock.monotonic()+5.
        while clock.monotonic() < end:
            status=controller.status
            if status is not None and status.current_action=='PASSIVE' and phase(status)=='IDLE':
                write(dict(record_type='startup_cancelled',current_action='PASSIVE',actions_sent=0))
                return
            clock.sleep(.02)
        raise RuntimeError('Startup CANCEL did not confirm PASSIVE/IDLE')
    except BaseException as error:
        print(f'EXIT INCOMPLETE: {error}. Inspect actual state; use E-stop if safety is uncertain.',flush=True)
        try: actual=status_row(controller.status)
        except BaseException: actual=None
        write(dict(record_type='startup_exit_failure',error=repr(error),actual_status=actual))


class PosePreparation:
    """Hardware-bounded cosine position/gain ramp; no ONNX actions in this phase."""
    def __init__(self, plan, envelope, state):
        plan.check_hardware_pose(envelope,state)
        self.plan,self.envelope=plan,envelope
        self.q0=np.array(state.q,dtype=float)
        self.q1=np.array([j.q for j in plan.target.joints])
        delta=float(np.max(np.abs(self.q1-self.q0)))
        # The reviewed fixed stand must have settled before take-over.
        if np.max(np.abs(self.q0-plan.fixed_q)) > SETTLE_ERROR:
            raise ValueError('FIXED_STAND drifted before EXTERNAL preparation')
        self.duration=max(POSE_MIN_SECONDS,math.pi*delta/(2*POSE_SPEED))
        if self.duration > POSE_MAX_SECONDS:
            raise ValueError('Pose preparation exceeds reviewed motion budget')
        self.last_target=self.q0.copy();self.settled=None

    def step(self, elapsed, state):
        if np.max(np.abs(state.dq)) > POSE_SPEED+.10:
            raise RuntimeError('Pose preparation measured joint speed exceeds 0.25 rad/s')
        if np.max(np.abs(np.asarray(state.q)-self.last_target)) > POSE_ERROR:
            raise RuntimeError('Pose preparation tracking error exceeds 0.10 rad')
        fraction=min(1.,max(0.,elapsed/self.duration))
        alpha=.5-.5*math.cos(math.pi*fraction)
        q=self.q0+alpha*(self.q1-self.q0)
        joints=tuple(replace(j,q=float(q[i]),dq=0.,tau=0.,
            kp=float((1-alpha)*self.plan.fixed_kp[i]+alpha*j.kp),
            kd=float((1-alpha)*self.plan.fixed_kd[i]+alpha*j.kd))
            for i,j in enumerate(self.plan.target.joints))
        command=replace(self.plan.target,joints=joints)
        self.envelope.check(command);self.last_target=q
        good=(fraction==1 and np.max(np.abs(np.asarray(state.q)-self.q1)) <= SETTLE_ERROR
              and np.max(np.abs(state.dq)) <= SETTLE_SPEED)
        self.settled=(elapsed if self.settled is None else self.settled) if good else None
        complete=self.settled is not None and elapsed-self.settled >= SETTLE_SECONDS
        if elapsed > self.duration+SETTLE_TIMEOUT and not complete:
            raise TimeoutError('Model default pose did not settle; ONNX control was not enabled')
        return command,complete,dict(record_type='pose_preparation',pose_fraction=fraction,
            pose_duration_s=self.duration,hardware_ranges_checked=True,onnx_control_enabled=False)
