"""Supported actor/Pico EXTERNAL control using the unmodified SDK.

Importing this module creates no robot handles. Hardware setup belongs to the
explicit CLI live entry. Injectable I/O lets offline tests exercise failures.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import select
import sys
import time

import numpy as np

STATE_AGE = .1
STATUS_AGE = .5
TARGET_PERIOD = .020


def phase(status):
    return str(getattr(status.phase, 'value', status.phase))


def status_row(status):
    if status is None:
        return None
    return dict(current_action=status.current_action, phase=phase(status),
                generation=status.generation, reject_count=status.reject_count,
                ingress_drop_count=status.ingress_drop_count,
                timeout_hold=status.timeout_hold, status_seq=status.status_seq)


def require_peer_profile(path):
    """Inspect the top-level JSON5 mode without logging any profile content.

    Tokenize strings/comments before looking at braces so credentials, URLs or
    commented examples cannot masquerade as the effective session mode. The
    Aorta runtime still parses/validates the complete deployment profile.
    """
    path = Path(path)
    if not path.is_absolute() or path.stat().st_mode & 0o077:
        raise ValueError('Use an absolute private (0600) peer session profile')
    tokens = re.findall(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|//[^\n]*|/\*[\s\S]*?\*/|[{}\[\]:,]|[^\s{}\[\]:,"']+''', path.read_text())
    tokens = [t for t in tokens if not t.startswith(('//', '/*'))]
    depth, modes = 0, []
    for i, token in enumerate(tokens):
        if token in ('{', '['):
            depth += 1
        elif token in ('}', ']'):
            depth -= 1
        elif depth == 1 and token in ('mode', '"mode"', "'mode'") and tokens[i+1:i+2] == [':']:
            modes.append(tokens[i+2:i+3])
    if len(modes) != 1 or modes[0] not in (['"peer"'], ["'peer'"]):
        raise ValueError('Effective JSON5 profile must explicitly select exactly one top-level mode=peer')


def load_robot_config(path):
    """Read the deployment-selected effective YAML; never patch robot config.

    This is not discovery of the running process's selected configuration.
    The attendant must verify that selection in the interactive confirmation.
    """
    import yaml
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('--robot-config must be the absolute effective configuration path')
    payload = path.read_bytes()
    root = yaml.safe_load(payload)
    try:
        ankle = root['parallel_ankle']
        external = root['actions']['EXTERNAL']
        params = external['params']
        if external['generator'] != 'builtin::external':
            raise ValueError('EXTERNAL must use builtin::external')
        if ankle['pd']['strategy'] != 'virtual_pd' or params['ankle_virtual_pd_intent'] is not True:
            raise ValueError('Requires virtual_pd and EXTERNAL ankle_virtual_pd_intent=true; no config is changed')
        unknown = set(params)-{'command_timeout_ms', 'recover_valid_frames',
                              'interpolation', 'safe_hold', 'ankle_virtual_pd_intent'}
        if unknown:
            raise ValueError('Unsupported EXTERNAL params: '+', '.join(sorted(unknown)))
        timeout = params['command_timeout_ms']
        if type(timeout) is not int or timeout <= 0:
            raise ValueError('EXTERNAL command watchdog must be enabled')
        recover = params['recover_valid_frames']
        if type(recover) is not int or recover < 1:
            raise ValueError('EXTERNAL recover_valid_frames must be a positive integer')
        interpolation = params['interpolation']
        periods = [interpolation[key] for key in ('min_period_ms', 'max_period_ms')]
        if (set(interpolation) != {'min_period_ms', 'max_period_ms'}
                or any(type(value) is not int or value <= 0 for value in periods)
                or periods[0] > periods[1]):
            raise ValueError('Invalid EXTERNAL interpolation periods')
        hold = params['safe_hold']
        for key in ('kp', 'kd'):
            values = np.asarray(hold[key], dtype=float)
            if values.shape not in ((31,), (42,)) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Requires full A-sample safe_hold gains (31 body or 42 legacy slots)')
        if len(hold['kp']) != len(hold['kd']):
            raise ValueError('safe_hold gain lengths differ')
        # Check the model's recorded A-sample geometry; another mechanism must
        # have its own reviewed export/profile instead of reusing this actor.
        for key, value in dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235).items():
            if not math.isclose(ankle['geometry'][key], value, rel_tol=0, abs_tol=1e-8):
                raise ValueError('Model/robot ankle geometry differs: '+key)
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete effective A-sample robot configuration') from error
    return params, dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


class CommandEnvelope:
    """Use RobotInfo ranges, with the A-sample virtual ankle delegation contract.

    Current firmware rejects YAML command_limits. Its four parallel ankle axes
    advertise all five supported fields but only joint-domain q limits; their
    dq/tau/kp/kd validation belongs to the motor-space adapter, not zero-valued
    absent RobotInfo ranges. Every other populated axis needs all five ranges.
    """
    def __init__(self, runtime, reply, params):
        from .sdkmain_adapter import validate_robot_info, BODY_TO_LEGACY
        validate_robot_info(reply)
        self.body = reply.info.body
        if self.body.roster_version != 1:
            raise ValueError('Expected body roster_version=1')
        runtime.policy.bind_body_info(self.body)
        self.limits = []
        self.delegated_joints = []
        ankles = {'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                  'right_ankle_pitch_joint', 'right_ankle_roll_joint'}
        for index, (name, cap) in enumerate(zip(self.body.joint_names, self.body.joints)):
            if not cap.populated:
                self.limits.append(None)
                continue
            if cap.is_mock or cap.slot_role != 1:
                raise ValueError('Physical ACTIVE body actuator required: '+name)
            if cap.supported_fields != 31:
                raise ValueError('Required q/dq/tau/kp/kd support unavailable: '+name)
            delegated = name in ankles and cap.valid_limits == 1
            if cap.valid_limits != 31 and not delegated:
                raise ValueError('Advertised command limits unavailable: '+name)
            ranges = [(cap.q_min, cap.q_max), (cap.dq_min, cap.dq_max),
                      (cap.tau_min, cap.tau_max), (cap.kp_min, cap.kp_max),
                      (cap.kd_min, cap.kd_max)]
            if delegated:
                self.delegated_joints.append(name)
                ranges[1:] = [None]*4
            if any(not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
                   for bounds in ranges if bounds is not None):
                raise ValueError('Invalid advertised command ranges: '+name)
            for key, value_range in (('kp', ranges[3]), ('kd', ranges[4])):
                values = params['safe_hold'][key]
                slot = index if len(values) == 31 else BODY_TO_LEGACY[index]
                value = values[slot]
                if (not math.isfinite(value) or value <= 0 or (value_range is not None
                        and not value_range[0] <= value <= value_range[1])):
                    raise ValueError('safe_hold gain unavailable/out of bounds: '+name)
            self.limits.append(ranges)
        # A fixed model/gain profile, including both necks, must fit BEFORE entry.
        command, _ = runtime.policy.decode(np.zeros(28))
        self.check(command)
        self.profile_command = command
        runtime.policy.reset()

    def check_measured_pose(self, state):
        # Observations are measurements, not decoded policy targets. Match the
        # SDK example: straight knees may be below the model's output bounds.
        command = replace(self.profile_command, joints=tuple(
            replace(j, q=float(state.q[i])) if self.limits[i] else j
            for i, j in enumerate(self.profile_command.joints)))
        self.check(command)

    def check(self, command):
        if len(command.joints) != 31:
            raise ValueError('Command must contain exactly 31 slots')
        for name, joint, ranges in zip(self.body.joint_names, command.joints, self.limits):
            values = [getattr(joint, key) for key in ('q', 'dq', 'tau', 'kp', 'kd')]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite body command: '+name)
            if ranges is None:
                if any(values):
                    raise ValueError('Nonzero command for unpopulated slot: '+name)
                continue
            if joint.kp < 0 or joint.kd < 0:
                raise ValueError('Negative body command gain: '+name)
            for key, value, bounds in zip(('q', 'dq', 'tau', 'kp', 'kd'), values, ranges):
                if bounds is None:
                    continue  # Validated by the firmware's parallel ankle adapter.
                low, high = bounds
                if not low <= value <= high:
                    raise ValueError(f'{name}.{key}={value} outside [{low}, {high}]')


class Inputs:
    def __init__(self, runtime, controller, pico, clock):
        self.runtime, self.controller, self.pico, self.clock = runtime, controller, pico, clock
        self.previous_stamp = 0
        self.last_pico_received = None
        self.pico_max_age = runtime.config['pico_max_age_s']

    def validate_state(self, state):
        from .sdkmain_adapter import validate_body_arrays
        validate_body_arrays(state)
        if state.roster_version != 1:
            raise ValueError('Expected body roster_version=1')
        self.fresh(state)
        if state.ts_state_real_us < self.previous_stamp:
            raise ValueError('BodyState source clock regressed')
        self.previous_stamp = state.ts_state_real_us
        return self.runtime.policy.state(state)

    def fresh(self, state):
        now, wall = self.clock.monotonic(), self.clock.time_ns()/1e9
        age = now-state.received_monotonic_ns/1e9
        if not 0 <= age <= STATE_AGE:
            raise TimeoutError('BodyState local receipt is stale/future')
        for stamp in (state.ts_state_real_us, state.ts_state_pub_us):
            if stamp <= 0 or not -.005 <= wall-stamp/1e6 <= STATE_AGE:
                raise TimeoutError('BodyState source is stale or S100 clock contract differs')

    def read(self, timeout=0):
        state = self.controller.wait_state(timeout)
        q, dq, root, gyro = self.validate_state(state)
        sample = self.pico.step() if self.pico is not None else None
        if sample is not None:
            body, _, metadata = sample
            self.runtime.push_pico(body, int(metadata['timestamp_ns']),
                received_monotonic=float(metadata['timestamp_monotonic']), robot_root=root)
            self.last_pico_received = float(metadata['timestamp_monotonic'])
        return state, q, dq, root, gyro

    def require_pico(self):
        stamp = self.last_pico_received
        if stamp is None or not 0 <= self.clock.monotonic()-stamp <= self.pico_max_age:
            raise TimeoutError('Pico input is stale/future')


class StatusGuard:
    def __init__(self, clock, initial_actions=('PASSIVE',)):
        self.clock, self.identity, self.changed = clock, None, clock.monotonic()
        self.initial_actions = initial_actions

    def check(self, status, entered=None):
        if status is None:
            raise RuntimeError('ControlStatus unavailable')
        identity = (status.status_seq, status.stamp_monotonic_ns)
        if identity != self.identity:
            self.identity, self.changed = identity, self.clock.monotonic()
        if self.clock.monotonic()-self.changed > STATUS_AGE:
            raise TimeoutError('ControlStatus stopped advancing')
        if entered is None:
            if status.current_action == 'EXTERNAL':
                raise RuntimeError('Another EXTERNAL session is active')
            if status.current_action not in self.initial_actions or phase(status) != 'IDLE':
                raise RuntimeError('Supported trial must start from confirmed '+
                                   '/'.join(self.initial_actions)+'/IDLE')
        elif (status.current_action != 'EXTERNAL' or phase(status) != 'RUNNING'
              or status.timeout_hold or any(getattr(status, key) != getattr(entered, key)
                  for key in ('generation', 'client_id', 'lease_id', 'reject_count', 'ingress_drop_count'))):
            raise RuntimeError('EXTERNAL ownership/state or reject/drop counters changed')
        return status


class TerminalCommands:
    def __init__(self, robot_id, stop):
        self.robot_id, self.stop = robot_id, stop

    def poll(self):
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        line = sys.stdin.readline()
        command = line.strip().upper()
        if not line or command == 'STOP':
            self.stop.set()
            return None
        if command in ('CONNECT', 'CONNECT '+self.robot_id):
            return 'connect'
        if command in ('PICO', 'PICO '+self.robot_id):
            return 'track'
        print(f'Commands: CONNECT {self.robot_id}, PICO {self.robot_id}, STOP', flush=True)
        return None


def manual_reference(runtime, snapshot, now):
    """Validate an immutable worker snapshot without touching policy history."""
    if snapshot.get('state') != 'ready':
        raise ValueError('Pico reference not ready: '+str(snapshot.get('state')))
    age = now-float(snapshot['received_monotonic'])
    if not 0 <= age <= runtime.config['pico_max_age_s']:
        raise TimeoutError('Pico reference worker snapshot is stale/future')
    motion = runtime.math.Motion(snapshot['q'], snapshot['base'], snapshot['quat'])
    if len(motion.q) != runtime.reference.steps:
        raise ValueError('Pico worker reference window length differs from model')
    return motion, dict(pico_age_s=age, reference_delay_s=runtime.reference.delay,
        source_gap_s=float(snapshot.get('source_gap_s',0.)),
        retarget_ms=float(snapshot['retarget_ms']),
        gmr_task_error=float(snapshot['gmr_task_error']))


def next_policy_deadline(deadline, now, period):
    """Advance to a future nominal tick, skipping missed slots without bursts."""
    following = deadline+period
    missed = 0
    if now > following+1e-9:
        missed = int(math.ceil((now-following-1e-9)/period))
        following += missed*period
    return following, missed


def record_period_overrun(write, phase_name, elapsed, actions_sent, **details):
    """50 Hz is a scheduling target, not an extra firmware safety limit."""
    if elapsed > TARGET_PERIOD:
        write(dict(record_type='policy_period_overrun',phase=phase_name,
                   elapsed_ms=elapsed*1000,target_period_ms=TARGET_PERIOD*1000,
                   actions_sent=actions_sent,**details))


def run_live(runtime, args, write, controller, pico, stop, service_alive,
             clock=time, manual_pico=None, poll_command=lambda:None,
             request_action=None):
    """One trial only. Every attempted entry is followed by checked CANCEL.

    Fake I/O in unit tests exercises this same runner; actual transport and
    hardware performance remain separate acceptance checks.
    """
    if not math.isfinite(args.seconds) or args.seconds < 0:
        raise ValueError('Live duration must be finite and nonnegative; 0 means until STOP')
    params, receipt = load_robot_config(args.robot_config)
    envelope = CommandEnvelope(runtime, controller.get_robot_info(args.startup_timeout), params)
    pico_max_age = runtime.config['pico_max_age_s']
    auto = bool(getattr(args, 'auto_start', False))
    if getattr(args, 'entry_policy', 'policy-online') != 'policy-online':
        raise ValueError('Only policy-online startup is supported; joint-ramp was removed')
    wait_pico = bool(getattr(args, 'wait_pico', False))
    auto_pico = bool(getattr(args, 'auto_pico', False))
    startup = None
    if auto:
        from .actor_pico_startup import (StartupPlan, fixed_stand_startup,
                                         cancel_startup, FirmwareActions)
        startup = StartupPlan(runtime, envelope, args.robot_config)
    manual = bool(getattr(args, 'manual_pico', False))
    requires_pico = not args.stand_only and not manual
    if auto and (not manual or args.stand_only):
        raise ValueError('One-click startup requires manual Pico mode')
    # Firmware retires completed keyframes, leaving current_action empty. Auto
    # explicitly requests FIXED_STAND and verifies measured targets afterward;
    # an empty name alone is never treated as completed fixed stand.
    initial_actions = ('PASSIVE', 'QUIET_PASSIVE', 'FIXED_STAND', '') if auto else ('PASSIVE',)
    inputs = Inputs(runtime, controller, pico if requires_pico else None, clock)
    guard = StatusGuard(clock, initial_actions=initial_actions)
    if manual and manual_pico is None:
        raise ValueError('Manual Pico mode requires an idle capture worker')
    runtime.reset_inputs()
    # Warm actual inference before taking ownership; erase its action history.
    runtime.infer_motion(runtime.math.Motion.hold(runtime.model.contract.default),
        q=runtime.model.contract.default, dq=np.zeros(28), root=[1, 0, 0, 0], gyro=np.zeros(3))
    runtime.reset_inputs()
    preflight_deadline = clock.monotonic()+args.startup_timeout
    while not stop.is_set():
        if requires_pico and not service_alive():
            raise RuntimeError('Pico service exited during preflight')
        if clock.monotonic() >= preflight_deadline:
            raise TimeoutError('Preflight did not receive fresh status, Body31 and complete Pico window')
        try:
            state, q, dq, root, gyro = inputs.read(min(.1, max(.001, preflight_deadline-clock.monotonic())))
        except TimeoutError:
            clock.sleep(.02)
            continue
        status = controller.status
        if status is not None:
            guard.check(status)
        if (not requires_pico or runtime.reference.ready()) and status is not None:
            if requires_pico:
                inputs.require_pico()
            if auto:
                startup.check_hardware_pose(envelope, state)
            else:
                envelope.check_measured_pose(state)
            if auto:
                # The real first-command preflight belongs after FIXED_STAND.
                # A second policy rollout on the earlier PASSIVE pose is unused.
                break
            started = clock.monotonic()
            if requires_pico:
                command, obs, row = runtime.infer_command(q=q, dq=dq, root=root, gyro=gyro,
                    now_monotonic=started, maximum_pico_age=pico_max_age)
            else:
                motion = runtime.math.Motion.hold(runtime.model.contract.default)
                motion = motion.lead_in(q, root, runtime.config['body_profile']['lead_in'])
                command, obs, row = runtime.infer_motion_command(motion,
                    q=q, dq=dq, root=root, gyro=gyro, yaw_offset=runtime.math.yaw(root))
            envelope.check(command)
            record_period_overrun(write,'preflight',clock.monotonic()-started,0)
            inputs.fresh(state)
            write(dict(row, record_type='live_preflight', robot_id=args.robot_id,
                       robot_config=receipt, status=status_row(status)), obs)
            break
        clock.sleep(runtime.model.contract.period)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Display and record actual targets/gains; starting live requests motion
    # directly, without a second typed MOVE prompt.
    print('ACTOR_LIVE_REVIEW '+json.dumps(dict(robot_id=args.robot_id,
        model=runtime.config['artifacts']['model'], config=receipt,
        stand_only=args.stand_only, seconds=args.seconds,
        manual_pico=manual, auto_start=auto, entry_policy='policy-online',
        wait_pico=wait_pico, auto_pico=auto_pico,
        startup_plan=startup.review if auto else None,
        default_pose=dict(zip(runtime.model.contract.names, runtime.model.contract.default.tolist())),
        kp=dict(zip(runtime.model.contract.names, runtime.model.contract.kp.tolist())),
        kd=dict(zip(runtime.model.contract.names, runtime.model.contract.kd.tolist())),
        fixed_joints=runtime.config['body_profile']['fixed_joints'], reference_delay_s=runtime.reference.delay)), flush=True)
    print('ACTOR_START: motion requested; '+
          ('running until STOP/Ctrl-C or fault' if args.seconds == 0 else f'duration limit {args.seconds:g}s')+
          '; exit requests CANCEL -> PASSIVE. Keep support and E-stop available.',flush=True)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Refresh effective configuration before requesting any motion.
    if load_robot_config(args.robot_config)[1] != receipt:
        raise RuntimeError('Effective config file changed during preflight')
    if auto:
        if not fixed_stand_startup(startup, envelope, runtime, args, controller, inputs,
                                  stop, write, clock, request_action=request_action,
                                  poll_command=poll_command):
            return dict(mode='live', actions_sent=0, entered=False, stopped_by_signal=True)
    startup_generation = controller.status.generation if auto else None
    cancel_action = (request_action or FirmwareActions(stop, clock, lambda:inputs.read(.1))) if auto else None
    try:
        if auto and load_robot_config(args.robot_config)[1] != receipt:
            raise RuntimeError('Effective config file changed during startup')
        runtime.reset_inputs()
        state, q, dq, root, gyro = inputs.read(STATE_AGE)
        if auto:
            startup.check_hardware_pose(envelope, state)
        else:
            envelope.check_measured_pose(state)
        if requires_pico:
            inputs.require_pico()
        guard = StatusGuard(clock, initial_actions=('FIXED_STAND', '') if auto else ('PASSIVE',))
        guard.check(controller.status)
        if auto:
            # PASSIVE can have a very different pose. Validate the first policy
            # command on the actual prepared FIXED_STAND feedback, not on an
            # earlier passive snapshot, and still before requesting EXTERNAL.
            runtime.reset_inputs()
            begin = clock.monotonic()
            lead = runtime.math.Motion.hold(runtime.model.contract.default).lead_in(
                q, root, runtime.config['body_profile']['lead_in'])
            command, obs, row = runtime.infer_motion_command(lead,q=q,dq=dq,root=root,gyro=gyro,
                                                            yaw_offset=runtime.math.yaw(root))
            model_elapsed = clock.monotonic()-begin
            # Preserve the actual failing input before any guard can raise.
            # This is one pre-ownership record, not a streaming log on the
            # 50 Hz control path. It supports exact inference reproduction.
            write(dict(row,record_type='prepared_policy_candidate',actions_sent=0,
                body_targets=[dict(q=j.q,dq=j.dq,tau=j.tau,kp=j.kp,kd=j.kd) for j in command.joints],
                input_state=dict(q=q.tolist(),dq=dq.tolist(),root=root.tolist(),gyro=gyro.tolist(),
                                 body_q=state.q.tolist(),state_id=int(state.state_id)),
                reference=dict(q=lead.q.tolist(),base=lead.base.tolist(),quat=lead.quat.tolist(),
                               ref_time=0.,yaw_offset=runtime.math.yaw(root)),
                observation=obs.tolist()))
            envelope.check(command)
            record_period_overrun(write,'prepared_preflight',model_elapsed,0)
            inputs.fresh(state)
            write(dict(row,record_type='prepared_policy_preflight',actions_sent=0),obs)
            runtime.reset_inputs()
    except BaseException:
        if auto:
            cancel_startup(controller, cancel_action, clock, write, startup_generation)
        raise
    attempted, sent_count = False, 0
    report = dict(mode='live', actions_sent=0, control_handles_created=1, entered=False,
                  exit_confirmed=False, robot_id=args.robot_id, stand_only=args.stand_only,
                  auto_start=auto, onnx_control_enabled=False,
                  missed_policy_ticks=0, duration_limit_s=args.seconds)
    try:
        if stop.is_set():
            return report
        attempted = True
        entered = controller.enter_external(timeout=5.)
        if (entered.current_action != 'EXTERNAL' or phase(entered) != 'RUNNING'
                or entered.client_id != controller.client_id or not entered.lease_id
                or entered.generation <= 0):
            raise RuntimeError('Entry did not confirm owned EXTERNAL/RUNNING')
        report['entered'] = True
        write(dict(record_type='external_entered', status=status_row(entered), actions_sent=0))
        runtime.reset_inputs()
        state, q, dq, root, gyro = inputs.read(0)
        if requires_pico:
            inputs.require_pico()
        if auto:
            startup.check_hardware_pose(envelope,state)
        if not auto:
            envelope.check_measured_pose(state)
        lead = runtime.math.Motion.hold(runtime.model.contract.default).lead_in(
            q, root, runtime.config['body_profile']['lead_in'])
        offset = runtime.math.yaw(root)
        preparation = max(3., (len(lead.q)-1)*runtime.model.contract.period)
        start = deadline = clock.monotonic()
        end = start+args.seconds if args.seconds > 0 else math.inf
        policy_tick = 0
        stand_tick = 0
        tracking = False
        if auto:
            print('POLICY_LEAD_IN: ONNX runs from measured-pose reference toward model stand; awaiting Pico',flush=True)
        if manual:
            print('PICO_BUTTONS: release both triggers, then squeeze both past 0.7 to toggle '
                  'following/ONNX stand. STOP or Ctrl-C cancels.',flush=True)
        previous_pico_state = None
        previous_controller_state = None
        last_trigger_event = 0
        if manual and wait_pico:
            manual_pico.connect(root)
            print('PICO_WAIT: receiver starting; ONNX stand does not depend on receiving Pico',flush=True)
            write(dict(record_type='pico_connect_requested',automatic=True,actions_sent=0))
        while not stop.is_set() and clock.monotonic() < end:
            if requires_pico and not service_alive():
                raise RuntimeError('Pico service exited')
            clock.sleep(max(0., deadline-clock.monotonic()))
            if stop.is_set() or clock.monotonic() >= end:
                break
            cycle = clock.monotonic()
            cpu_clock = getattr(clock,'thread_time',None)
            cpu_start = cpu_clock() if cpu_clock is not None else None
            guard.check(controller.status, entered)
            status_end = clock.monotonic()
            state, q, dq, root, gyro = inputs.read(0)
            state_end = clock.monotonic()
            if requires_pico:
                inputs.require_pico()
            # Match SDK ReferenceSession: one reference frame per successful
            # inference/publication, independent of wall-clock scheduling jitter.
            elapsed = stand_tick*runtime.model.contract.period
            reference_frame = min(stand_tick,len(lead.q)-1)
            reference_time = reference_frame*runtime.model.contract.period
            snapshot = None
            if manual:
                request = poll_command()
                if request == 'connect':
                    if manual_pico.connect(root):
                        print('PICO_CONNECTING: ONNX stand continues; use the configured GMR reference stance.', flush=True)
                        write(dict(record_type='pico_connect_requested', actions_sent=sent_count))
                snapshot = manual_pico.snapshot()
                controller_age = clock.monotonic()-float(snapshot.get('controller_received_monotonic',-math.inf))
                controller_ready = 0 <= controller_age <= pico_max_age
                if controller_ready != previous_controller_state:
                    previous_controller_state = controller_ready
                    print('PICO_BUTTONS: '+('controller input ready; release then squeeze both triggers to toggle'
                        if controller_ready else 'waiting for fresh controller input'),flush=True)
                    write(dict(record_type='pico_controller_input',ready=controller_ready,actions_sent=sent_count))
                event = int(snapshot.get('trigger_event_seq',last_trigger_event))
                if event != last_trigger_event:
                    previous_event = last_trigger_event
                    last_trigger_event = event
                    event_age = clock.monotonic()-float(snapshot.get('trigger_event_monotonic',-math.inf))
                    controller_age = clock.monotonic()-float(snapshot.get('controller_received_monotonic',-math.inf))
                    if event == previous_event+1 and 0 <= event_age <= pico_max_age and 0 <= controller_age <= pico_max_age:
                        request = 'toggle'
                        write(dict(record_type='pico_trigger_toggle',event_seq=event,actions_sent=sent_count))
                    else:
                        write(dict(record_type='pico_trigger_rejected',reason='stale/missed trigger edge',
                                   event_seq=event,actions_sent=sent_count))
                state_name = snapshot.get('state')
                state_key = state_name
                if state_key != previous_pico_state:
                    previous_pico_state = state_key
                    detail = (': '+str(snapshot['error'])) if snapshot.get('error') else ''
                    print(f'PICO_INPUT: {state_name}{detail}; ONNX '+('tracking' if tracking else 'stand'), flush=True)
                    write(dict(record_type='pico_input_state', state=state_name,
                               error=snapshot.get('error'),
                               source_gap_s=snapshot.get('source_gap_s'),
                               receipt_gap_s=snapshot.get('receipt_gap_s'),
                               retarget_ms=snapshot.get('retarget_ms'),actions_sent=sent_count))
                if request == 'toggle' and tracking:
                    # An intentional standby transition is distinct from a
                    # source fault. Faulted following still cancels, never falls back.
                    manual_reference(runtime,snapshot,clock.monotonic())
                    tracking = False
                    auto_pico = False
                    lead = runtime.math.Motion.hold(runtime.model.contract.default).lead_in(
                        q,root,runtime.config['body_profile']['lead_in'])
                    offset = runtime.math.yaw(root)
                    stand_tick = 0
                    elapsed = reference_time = 0.
                    reference_frame = 0
                    preparation = max(.3,(len(lead.q)-1)*runtime.model.contract.period)
                    request = None
                    print('PICO_STANDBY: returning reference to ONNX stand; EXTERNAL session stays active.',flush=True)
                    write(dict(record_type='tracking_paused',actions_sent=sent_count,
                               transition_start_q=q.tolist(),policy_history_reset=False))
                automatic_switch = False
                if auto_pico and not tracking and elapsed >= preparation and snapshot.get('state') == 'ready':
                    try:
                        manual_reference(runtime,snapshot,clock.monotonic())
                        automatic_switch = True
                    except (ValueError,TimeoutError):
                        pass
                if (request in ('track','toggle') or automatic_switch) and not tracking:
                    try:
                        if elapsed < preparation:
                            raise ValueError('Stand preparation has not completed')
                        motion, _ = manual_reference(runtime, snapshot, clock.monotonic())
                    except (ValueError, TimeoutError) as error:
                        print(f'PICO_SWITCH_REJECTED: {error}; ONNX stand continues. No switch is queued.', flush=True)
                        write(dict(record_type='pico_switch_rejected', error=str(error), actions_sent=sent_count))
                    else:
                        tracking = True
                        # GMR keeps its world root. Align heading at each new
                        # engagement using the current measured robot heading.
                        tracking_yaw = runtime.math.yaw(root)-runtime.math.yaw(motion.quat[0])
                        print('PICO_TRACKING: same EXTERNAL session and policy history.', flush=True)
                        write(dict(record_type='tracking_started',actions_sent=sent_count,
                                   reference_yaw_offset=tracking_yaw))
            if manual and tracking:
                motion, diagnostics = manual_reference(runtime, snapshot, clock.monotonic())
                command, obs, row = runtime.infer_motion_command(motion, q=q, dq=dq,
                    root=root, gyro=gyro, diagnostics=diagnostics, yaw_offset=tracking_yaw)
                row['pico_reference'] = dict(q=motion.q.tolist(), root_wxyz=motion.quat.tolist(),
                    raw_body_poses=snapshot.get('raw_body_poses'),
                    source_log=snapshot.get('source_log'),
                    timestamp_ns=snapshot.get('pico_timestamp_ns'), yaw_offset=tracking_yaw)
            elif requires_pico and elapsed >= preparation:
                if not tracking:
                    if not runtime.reference.ready():
                        raise TimeoutError('Pico reference window not ready after preparation')
                    runtime.align_reference_yaw(root)
                    tracking = True
                    write(dict(record_type='tracking_started', actions_sent=sent_count))
                command, obs, row = runtime.infer_command(q=q, dq=dq, root=root, gyro=gyro,
                    now_monotonic=clock.monotonic(), maximum_pico_age=pico_max_age)
            else:
                command, obs, row = runtime.infer_motion_command(lead, q=q, dq=dq,
                    root=root, gyro=gyro, ref_time=reference_time, yaw_offset=offset)
            envelope.check(command)
            compute_end = clock.monotonic()
            record_period_overrun(write,'compute',compute_end-cycle,sent_count,
                status_ms=(status_end-cycle)*1000,state_ms=(state_end-status_end)*1000,
                policy_ms=row.get('total_ms'),
                thread_cpu_ms=(cpu_clock()-cpu_start)*1000 if cpu_start is not None else None)
            inputs.fresh(state)
            if requires_pico:
                inputs.require_pico()
            if manual and tracking:
                # Recheck source age immediately before publishing; do not
                # continue stand or reconnect automatically after tracking loss.
                age = clock.monotonic()-float(snapshot['received_monotonic'])
                if not 0 <= age <= pico_max_age:
                    raise TimeoutError('Pico reference expired before publication')
            guard.check(controller.status, entered)
            if stop.is_set():
                break
            command = replace(command, input_state_id=int(state.state_id),
                input_state_ts_real_us=int(state.ts_state_real_us),
                ts_client_created_us=clock.time_ns()//1000)
            before = clock.monotonic()
            sent = controller.publish(command)  # SDK alone owns session + sequence.
            sent_count += 1
            policy_tick += 1
            if not tracking:
                stand_tick += 1
            report['actions_sent'] = sent_count
            if auto and not report['onnx_control_enabled']:
                print('ONNX_READY: policy commands started; both Pico triggers toggle following',flush=True)
            report['onnx_control_enabled'] = True
            write(dict(row, actions_sent=sent_count, control_handles_created=1,
                stage='pico_tracking' if tracking else 'prepare_stand',
                policy_tick=policy_tick-1,
                reference_time_s=0. if tracking else reference_time,
                reference_frame=0 if tracking else reference_frame,
                state_id=int(state.state_id), source_state_us=int(state.ts_state_real_us),
                feedback=dict(q_body=state.q.tolist(),dq_body=state.dq.tolist(),
                    root_wxyz=root.tolist(),gyro=gyro.tolist(),
                    applied_session_id=int(state.applied_session_id),
                    applied_sequence=int(state.applied_sequence),
                    ts_first_output_us=int(state.ts_first_output_us)),
                session_id=sent.session_id, sequence=sent.sequence,
                publish_ms=(clock.monotonic()-before)*1000,
                compute_ms=(compute_end-cycle)*1000,
                sent_body_targets=[dict(q=j.q, dq=j.dq, tau=j.tau, kp=j.kp, kd=j.kd) for j in sent.joints]), obs)
            if sent_count % 50 == 0:
                stage = 'pico_tracking' if tracking else 'stand'
                print(f'ACTOR_LIVE steps={sent_count} stage={stage}', flush=True)
            complete = clock.monotonic()
            record_period_overrun(write,'complete_cycle',complete-cycle,sent_count)
            deadline,missed = next_policy_deadline(deadline,clock.monotonic(),runtime.model.contract.period)
            report['missed_policy_ticks'] += missed
            if missed:
                write(dict(record_type='policy_ticks_skipped',missed_ticks=missed,
                           total_missed_ticks=report['missed_policy_ticks'],actions_sent=sent_count))
        report['stopped_by_signal'] = stop.is_set()
        report['elapsed_s'] = clock.monotonic()-start
        report['stop_reason'] = 'operator_stop' if stop.is_set() else 'duration_elapsed'
        write(dict(record_type='stop_requested',reason=report['stop_reason'],
            elapsed_s=report['elapsed_s'],duration_limit_s=args.seconds,actions_sent=sent_count))
        print(f"ACTOR_STOP: {report['stop_reason']}, elapsed={report['elapsed_s']:.2f}s; CANCEL -> PASSIVE",flush=True)
        return report
    finally:
        if auto and not attempted:
            cancel_startup(controller, cancel_action, clock, write, startup_generation)
        if attempted:
            original = sys.exc_info()[1]
            try:
                result = controller.exit_external(timeout=5.)
                write(dict(record_type='external_exit', status=status_row(result), actions_sent=sent_count))
                if result.current_action != 'PASSIVE' or phase(result) != 'IDLE':
                    raise RuntimeError('CANCEL did not confirm PASSIVE/IDLE')
                report['exit_confirmed'] = True
            except BaseException as error:
                print(f'EXIT INCOMPLETE: {error}. Inspect actual state; use hardware E-stop if safety is uncertain.',
                      file=sys.stderr, flush=True)
                # Preserve the original fault while recording exit failure too.
                try:
                    write(dict(record_type='exit_failure', error=repr(error),
                               actual_status=status_row(controller.status), actions_sent=sent_count))
                except BaseException as log_error:
                    print(f'Cannot record exit failure: {log_error}', file=sys.stderr, flush=True)
                if original is None:
                    raise
