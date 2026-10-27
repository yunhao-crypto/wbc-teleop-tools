"""Bounded, supported actor/Pico EXTERNAL trial using the unmodified SDK.

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
PICO_AGE = .1
STATUS_AGE = .5
COMPUTE_BUDGET = .020
NEUTRAL_TOLERANCE = .10


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
        timeout = params['command_timeout_ms']
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('EXTERNAL command watchdog must be enabled')
        limits = params['command_limits']
        default = limits['default']
        for key in ('max_kp', 'max_kd', 'max_abs_dq_rad_s'):
            value = default[key]
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid EXTERNAL command limit: '+key)
        if 10 not in default['allowed_modes']:
            raise ValueError('EXTERNAL must allow position/PD mode 10')
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
    """Reject commands outside advertised hardware and effective limits; no clipping."""
    def __init__(self, runtime, reply, params):
        from .sdkmain_adapter import validate_robot_info, BODY_TO_LEGACY
        validate_robot_info(reply)
        self.body = reply.info.body
        if self.body.roster_version != 1:
            raise ValueError('Expected body roster_version=1')
        runtime.policy.bind_body_info(self.body)
        self.limits = []
        overrides = params['command_limits'].get('joint_overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError('joint_overrides must be a joint-name mapping')
        for index, (name, cap) in enumerate(zip(self.body.joint_names, self.body.joints)):
            if not cap.populated:
                self.limits.append(None)
                continue
            if cap.is_mock or cap.slot_role != 1:
                raise ValueError('Physical ACTIVE body actuator required: '+name)
            if not cap.valid_limits & 1:
                raise ValueError('Advertised position limits unavailable: '+name)
            limit = dict(params['command_limits']['default'])
            limit.update(overrides.get(name, {}))
            if 10 not in limit['allowed_modes']:
                raise ValueError('PD mode denied: '+name)
            ranges = [(cap.q_min, cap.q_max), (cap.dq_min, cap.dq_max),
                      (cap.tau_min, cap.tau_max), (cap.kp_min, cap.kp_max),
                      (cap.kd_min, cap.kd_max)]
            if not np.isfinite(ranges).all() or any(low > high for low, high in ranges):
                raise ValueError('Invalid advertised command ranges: '+name)
            ranges[0] = (max(cap.q_min, limit.get('q_lower_rad', cap.q_min)),
                         min(cap.q_max, limit.get('q_upper_rad', cap.q_max)))
            ranges[1] = (max(cap.dq_min, -limit['max_abs_dq_rad_s']),
                         min(cap.dq_max, limit['max_abs_dq_rad_s']))
            ranges[3] = (max(0., cap.kp_min), min(cap.kp_max, limit['max_kp']))
            ranges[4] = (max(0., cap.kd_min), min(cap.kd_max, limit['max_kd']))
            if not np.isfinite(ranges).all() or any(low > high for low, high in ranges):
                raise ValueError('Empty effective command range: '+name)
            for key, value_range in (('kp', ranges[3]), ('kd', ranges[4])):
                values = params['safe_hold'][key]
                slot = index if len(values) == 31 else BODY_TO_LEGACY[index]
                value = values[slot]
                if not value_range[0] <= value <= value_range[1] or value <= 0:
                    raise ValueError('safe_hold gain unavailable/out of bounds: '+name)
            self.limits.append(ranges)
        # A fixed model/gain profile, including both necks, must fit BEFORE entry.
        command, _ = runtime.policy.decode(np.zeros(28))
        self.check(command)
        runtime.policy.reset()

    def check_entry_pose(self, runtime, state, q):
        if np.any(q < runtime.policy.lower) or np.any(q > runtime.policy.upper):
            raise ValueError('Measured entry pose is outside actor limits; no clipping')
        command, _ = runtime.policy.decode(np.zeros(28))
        command = replace(command, joints=tuple(replace(j, q=float(state.q[i])) if self.limits[i] else j
                                               for i, j in enumerate(command.joints)))
        self.check(command)
        runtime.policy.reset()

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
            for key, value, (low, high) in zip(('q', 'dq', 'tau', 'kp', 'kd'), values, ranges):
                if not low <= value <= high:
                    raise ValueError(f'{name}.{key}={value} outside [{low}, {high}]')


class Inputs:
    def __init__(self, runtime, controller, pico, clock):
        self.runtime, self.controller, self.pico, self.clock = runtime, controller, pico, clock
        self.previous_stamp = 0
        self.last_pico_received = None

    def validate_state(self, state):
        from .sdkmain_adapter import raw_state_from_arrays
        raw_state_from_arrays(state, received_monotonic_ns=state.received_monotonic_ns)
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
        sample = self.pico.step()
        if sample is not None:
            body, _, metadata = sample
            self.runtime.push_pico(body, int(metadata['timestamp_ns']),
                received_monotonic=float(metadata['timestamp_monotonic']), robot_root=root)
            self.last_pico_received = float(metadata['timestamp_monotonic'])
        return state, q, dq, root, gyro

    def require_pico(self):
        stamp = self.last_pico_received
        if stamp is None or not 0 <= self.clock.monotonic()-stamp <= PICO_AGE:
            raise TimeoutError('Pico input is stale/future')


class StatusGuard:
    def __init__(self, clock):
        self.clock, self.identity, self.changed = clock, None, clock.monotonic()

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
            if status.current_action != 'PASSIVE' or phase(status) != 'IDLE':
                raise RuntimeError('Supported trial must start from confirmed PASSIVE/IDLE')
        elif (status.current_action != 'EXTERNAL' or phase(status) != 'RUNNING'
              or status.timeout_hold or any(getattr(status, key) != getattr(entered, key)
                  for key in ('generation', 'client_id', 'lease_id', 'reject_count', 'ingress_drop_count'))):
            raise RuntimeError('EXTERNAL ownership/state or reject/drop counters changed')
        return status


def onsite_confirmation(robot_id, stop):
    """Require a current terminal response; CLI flags never replace attendance."""
    if not sys.stdin.isatty():
        raise RuntimeError('live requires an attended interactive terminal')
    print(f'Robot: {robot_id}. This moves all body joints toward the actor stand, then follows Pico.\n'
          'Confirm THIS robot has reliable support, clear motion/pinch zones, an attendant and usable E-stop.\n'
          'Confirm --robot-config is the running effective full A-sample config, and model gains were reviewed.\n'
          'EXTERNAL has no automatic expiry. CANCEL returns PASSIVE; loaded limbs may fall.\n'
          f'Type MOVE {robot_id} within 60 seconds to authorize this supported trial:', flush=True)
    deadline = time.monotonic()+60
    while not stop.is_set() and time.monotonic() < deadline:
        if select.select([sys.stdin], [], [], .1)[0]:
            if sys.stdin.readline().strip() != 'MOVE '+robot_id:
                raise RuntimeError('On-site confirmation declined')
            return
    raise RuntimeError('On-site confirmation interrupted/timed out')


def run_live(runtime, args, write, controller, pico, stop, service_alive,
             confirm=onsite_confirmation, clock=time):
    """One trial only. Every attempted entry is followed by checked CANCEL.

    Fake I/O in unit tests exercises this same runner; actual transport and
    hardware performance remain separate acceptance checks.
    """
    params, receipt = load_robot_config(args.robot_config)
    envelope = CommandEnvelope(runtime, controller.get_robot_info(args.startup_timeout), params)
    inputs, guard = Inputs(runtime, controller, pico, clock), StatusGuard(clock)
    runtime.reset_inputs()
    # Warm actual inference before taking ownership; erase its action history.
    runtime.infer_motion(runtime.math.Motion.hold(runtime.model.contract.default),
        q=runtime.model.contract.default, dq=np.zeros(28), root=[1, 0, 0, 0], gyro=np.zeros(3))
    runtime.reset_inputs()
    preflight_deadline = clock.monotonic()+args.startup_timeout
    while not stop.is_set():
        if not service_alive():
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
        if runtime.reference.ready() and status is not None:
            inputs.require_pico()
            envelope.check_entry_pose(runtime, state, q)
            started = clock.monotonic()
            command, obs, row = runtime.infer_command(q=q, dq=dq, root=root, gyro=gyro,
                now_monotonic=started, maximum_pico_age=PICO_AGE)
            envelope.check(command)
            if clock.monotonic()-started > COMPUTE_BUDGET:
                raise TimeoutError('Preflight inference exceeds 20 ms budget')
            write(dict(row, record_type='live_preflight', robot_id=args.robot_id,
                       robot_config=receipt, status=status_row(status)), obs)
            break
        clock.sleep(runtime.model.contract.period)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Display actual named targets/gains before asking the attendant to move.
    print('ACTOR_LIVE_REVIEW '+json.dumps(dict(robot_id=args.robot_id,
        model=runtime.config['artifacts']['model'], config=receipt,
        stand_only=args.stand_only, seconds=args.seconds,
        default_pose=dict(zip(runtime.model.contract.names, runtime.model.contract.default.tolist())),
        kp=dict(zip(runtime.model.contract.names, runtime.model.contract.kp.tolist())),
        kd=dict(zip(runtime.model.contract.names, runtime.model.contract.kd.tolist())),
        fixed_joints=runtime.config['body_profile']['fixed_joints'], reference_delay_s=runtime.reference.delay)), flush=True)
    confirm(args.robot_id, stop)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Confirmation can take a minute: refresh inputs and config BEFORE PREPARE.
    if load_robot_config(args.robot_config)[1] != receipt:
        raise RuntimeError('Effective config file changed during confirmation')
    runtime.reset_inputs()
    state, q, _, _, _ = inputs.read(STATE_AGE)
    envelope.check_entry_pose(runtime, state, q)
    inputs.require_pico()
    guard = StatusGuard(clock)
    guard.check(controller.status)
    attempted, sent_count = False, 0
    report = dict(mode='live', actions_sent=0, control_handles_created=1, entered=False,
                  exit_confirmed=False, robot_id=args.robot_id, stand_only=args.stand_only)
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
        inputs.require_pico()
        envelope.check_entry_pose(runtime, state, q)
        lead = runtime.math.Motion.hold(runtime.model.contract.default).lead_in(
            q, root, runtime.config['body_profile']['lead_in'])
        offset = runtime.math.yaw(root)
        neck_start = {name:float(state.q[runtime.policy.order.index(name)])
                      for name in runtime.config['body_profile']['fixed_joints']}
        preparation = max(3., (len(lead.q)-1)*runtime.model.contract.period)
        start = deadline = clock.monotonic()
        tracking = False
        while not stop.is_set() and clock.monotonic()-start < args.seconds:
            if not service_alive():
                raise RuntimeError('Pico service exited')
            clock.sleep(max(0., deadline-clock.monotonic()))
            if stop.is_set() or clock.monotonic()-start >= args.seconds:
                break
            cycle = clock.monotonic()
            guard.check(controller.status, entered)
            state, q, dq, root, gyro = inputs.read(0)
            inputs.require_pico()
            elapsed = cycle-start
            if not args.stand_only and elapsed >= preparation:
                if not tracking:
                    if not runtime.reference.ready():
                        raise TimeoutError('Pico reference window not ready after preparation')
                    if np.max(np.abs(runtime.reference.motion().q-runtime.model.contract.default)) > NEUTRAL_TOLERANCE:
                        raise ValueError('Keep Pico wearer neutral through preparation; reference jump refused')
                    tracking = True
                    write(dict(record_type='tracking_started', actions_sent=sent_count))
                command, obs, row = runtime.infer_command(q=q, dq=dq, root=root, gyro=gyro,
                    now_monotonic=clock.monotonic(), maximum_pico_age=PICO_AGE)
            else:
                command, obs, row = runtime.infer_motion_command(lead, q=q, dq=dq,
                    root=root, gyro=gyro, ref_time=elapsed, yaw_offset=offset)
                # The policy omits neck actions. Ramp the fixed neck profile
                # from measured entry positions rather than jumping to zero.
                alpha = min(1., elapsed/preparation)
                joints = list(command.joints)
                for name, initial in neck_start.items():
                    index = runtime.policy.order.index(name)
                    joints[index] = replace(joints[index], q=(1-alpha)*initial+alpha*joints[index].q)
                command = replace(command, joints=tuple(joints))
            envelope.check(command)
            if clock.monotonic()-cycle > COMPUTE_BUDGET:
                raise TimeoutError('State/retarget/inference exceeds 20 ms; target not published')
            inputs.fresh(state)
            inputs.require_pico()
            guard.check(controller.status, entered)
            if stop.is_set():
                break
            command = replace(command, input_state_id=int(state.state_id),
                input_state_ts_real_us=int(state.ts_state_real_us),
                ts_client_created_us=clock.time_ns()//1000)
            before = clock.monotonic()
            sent = controller.publish(command)  # SDK alone owns session + sequence.
            sent_count += 1
            report['actions_sent'] = sent_count
            write(dict(row, actions_sent=sent_count, control_handles_created=1,
                stage='pico_tracking' if tracking else 'prepare_stand',
                state_id=int(state.state_id), source_state_us=int(state.ts_state_real_us),
                session_id=sent.session_id, sequence=sent.sequence,
                publish_ms=(clock.monotonic()-before)*1000,
                sent_body_targets=[dict(q=j.q, dq=j.dq, tau=j.tau, kp=j.kp, kd=j.kd) for j in sent.joints]), obs)
            deadline += runtime.model.contract.period
            if clock.monotonic() > deadline+1e-9:
                raise TimeoutError('Complete policy cycle including publish/log exceeded 20 ms; no catch-up/retry')
        report['stopped_by_signal'] = stop.is_set()
        return report
    finally:
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
