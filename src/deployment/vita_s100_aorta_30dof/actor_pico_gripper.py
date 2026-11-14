"""Opt-in PICO side-grip control of AgiLink OP3 grippers, off the body policy clock.

Left and right are independent. Holding a side grip closes that gripper to
close_m with the configured current limit. Releasing it, and the command
before any grip sample, opens to the maximum opening that gripper's Info
declares. Missing, stale, frozen or invalid input opens the same way. Ending
the session sends nothing more, so each gripper
keeps its last command (a held object stays held); only a side fault opens.
Control goes through the humanoid-sdk EndEffectorClient on
the same EXTERNAL connection as the body; an SDK without that client is refused
before any control handle is taken.
"""
import json
import math
import threading
import time
from collections import deque
from types import SimpleNamespace

SCHEMA = 'wtt_pico_gripper_v1'
SIDES = ('left', 'right')
CONTROL_MODES = ('POSITION', 'POSITION_TORQUE')
ON_FAULT = ('open_and_disable_side', 'stop_session')
GRASP_STATES = ('open', 'closing', 'grasped', 'closed_empty')
_TOP_KEYS = {'schema', 'device_model', 'control_mode', 'rate_hz', 'input_max_age_s',
             'close_threshold', 'open_threshold', 'close_hold_s', 'feedback_max_age_s',
             'startup_timeout_s', 'output_ready_timeout_s', 'on_fault', 'sides'}
_SIDE_KEYS = {'open_m', 'close_m', 'vel_limit_m_s', 'current_a', 'grasp_min_opening_m',
              'grasp_stall_m', 'grasp_settle_s', 'grasp_current_ratio'}


def _number(value, name, low=-math.inf, high=math.inf, *, low_open=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'gripper config {name} must be a finite number')
    if value < low or value > high or (low_open and value == low):
        raise ValueError(f'gripper config {name}={value} is outside its allowed range')
    return float(value)


def validate_config(config):
    """Every value is explicit; unknown keys are rejected so typos cannot fall back."""
    if not isinstance(config, dict) or set(config) != _TOP_KEYS:
        missing = sorted(_TOP_KEYS-set(config or {})) if isinstance(config, dict) else []
        extra = sorted(set(config)-_TOP_KEYS) if isinstance(config, dict) else []
        raise ValueError(f'gripper config keys differ: missing={missing} unknown={extra}')
    if config['schema'] != SCHEMA:
        raise ValueError(f'gripper config schema must be {SCHEMA}')
    if config['device_model'] != 'AGILINK_OP3':
        raise ValueError('only device_model AGILINK_OP3 is supported')
    if config['control_mode'] not in CONTROL_MODES:
        raise ValueError(f'control_mode must be one of {CONTROL_MODES}')
    if config['on_fault'] not in ON_FAULT:
        raise ValueError(f'on_fault must be one of {ON_FAULT}')
    out = dict(config)
    out['rate_hz'] = _number(config['rate_hz'], 'rate_hz', 10., 200.)
    out['input_max_age_s'] = _number(config['input_max_age_s'], 'input_max_age_s', 0., 1., low_open=True)
    out['open_threshold'] = _number(config['open_threshold'], 'open_threshold', 0., 1., low_open=True)
    out['close_threshold'] = _number(config['close_threshold'], 'close_threshold', 0., 1.)
    if not out['open_threshold'] < out['close_threshold'] < 1.:
        raise ValueError('gripper config requires 0 < open_threshold < close_threshold < 1')
    out['close_hold_s'] = _number(config['close_hold_s'], 'close_hold_s', 0., 1.)
    out['feedback_max_age_s'] = _number(config['feedback_max_age_s'], 'feedback_max_age_s', 0., 1., low_open=True)
    out['startup_timeout_s'] = _number(config['startup_timeout_s'], 'startup_timeout_s', 0., 30., low_open=True)
    out['output_ready_timeout_s'] = _number(config['output_ready_timeout_s'], 'output_ready_timeout_s',
                                            0., 30., low_open=True)
    sides = config['sides']
    if not isinstance(sides, dict) or not sides or not set(sides) <= set(SIDES):
        raise ValueError('gripper config sides must name left and/or right')
    out['sides'] = {}
    for side in SIDES:
        if side not in sides:
            continue
        spec = sides[side]
        if not isinstance(spec, dict) or set(spec) != _SIDE_KEYS:
            raise ValueError(f'{side} gripper keys differ: missing={sorted(_SIDE_KEYS-set(spec or {}))} '
                             f'unknown={sorted(set(spec or {})-_SIDE_KEYS)}')
        values = {key: _number(spec[key], f'{side}.{key}') for key in _SIDE_KEYS}
        if not values['open_m'] > values['close_m']:
            raise ValueError(f'{side}: open_m must exceed close_m')
        for key in ('vel_limit_m_s', 'current_a', 'grasp_min_opening_m', 'grasp_stall_m', 'grasp_settle_s'):
            _number(values[key], f'{side}.{key}', 0., low_open=True)
        _number(values['grasp_current_ratio'], f'{side}.grasp_current_ratio', 0., 1., low_open=True)
        out['sides'][side] = values
    return out


def load_config(path):
    with open(path) as stream:
        return validate_config(json.load(stream))


def load_sdk():
    """Import the EE client lazily; the pinned body-only SDK has no such module."""
    try:
        from locomotion_aorta.end_effector import EndEffectorClient
        from locomotion_aorta import _end_effector_codec as codec
        from locomotion_aorta import end_effector_types as types
    except Exception as error:
        raise RuntimeError(
            'OP3 gripper control requires a humanoid-sdk with EndEffectorClient '
            '(locomotion_aorta.end_effector and its generated EE bindings); '
            'the installed SDK does not provide it') from error
    return SimpleNamespace(
        EndEffectorClient=EndEffectorClient, codec=codec,
        ControlMode=types.ControlMode, DeviceModel=types.DeviceModel, Field=types.Field,
        PositionDomain=types.PositionDomain, SampleStatus=types.SampleStatus,
        InhibitReason=types.InhibitReason, SampleStream=types.SampleStream,
        EndEffectorSide=types.EndEffectorSide, EndEffectorTarget=types.EndEffectorTarget,
        STATUS_REQUIRED=(types._STATE_READY_VALID | types._STATE_OUTPUT_ALLOWED_VALID
                         | types._STATE_INHIBIT_REASON_VALID))


class SideGripGate:
    """One side: a release at or below open_threshold arms; a grip above
    close_threshold held for close_hold_s closes; the band between keeps the
    current command. Invalid, stale, frozen or regressed input opens and disarms,
    so a grip held through a reconnect cannot close until released once.
    """

    def __init__(self, config):
        self.open_threshold = config['open_threshold']
        self.close_threshold = config['close_threshold']
        self.hold = config['close_hold_s']
        self.max_age = config['input_max_age_s']
        self.last_stamp = 0.
        self.last_change = -math.inf
        self.valid = False
        self._reset()

    def _reset(self):
        self.armed = self.closed = False
        self.since = None

    def update(self, received, stamp, value, now):
        valid = (all(isinstance(x, (int, float)) and math.isfinite(x) for x in (received, stamp, value, now))
                 and stamp > 0 and received > 0 and 0 <= now-received <= self.max_age and 0 <= value <= 1)
        if valid and stamp > self.last_stamp:
            self.last_stamp, self.last_change = stamp, now
        elif valid and stamp < self.last_stamp:
            valid = False
        self.valid = valid and now-self.last_change <= self.max_age
        if not self.valid:
            self._reset()
            return False
        if value <= self.open_threshold:
            self.armed, self.closed, self.since = True, False, None
        elif value > self.close_threshold:
            if not self.closed and self.armed:
                self.since = now if self.since is None else self.since
                self.closed = now-self.since >= self.hold
        else:
            self.since = None
        return self.closed


class GraspMonitor:
    """Classify a close command from measured opening and current.

    grasped: opening stalled (or current near the limit) above
    close_m + grasp_min_opening_m after grasp_settle_s; closed_empty: stalled
    at the closed target. Classification only reports; it never commands.
    """

    def __init__(self, spec):
        self.spec = spec
        self.state = 'open'
        self.started = None
        self.history = deque()

    def update(self, closed, now, opening, current):
        if not closed:
            self.started = None
            self.history.clear()
            self.state = 'open'
            return self.state
        if self.started is None:
            self.started = now
            self.history.clear()
        if opening is not None:
            self.history.append((now, opening))
        settle = self.spec['grasp_settle_s']
        while len(self.history) > 1 and self.history[1][0] <= now-settle:
            self.history.popleft()
        if now-self.started < settle or opening is None or not self.history or self.history[0][0] > now-settle:
            self.state = 'closing'
            return self.state
        values = [x for _, x in self.history]
        stalled = max(values)-min(values) <= self.spec['grasp_stall_m']
        loaded = (current is not None and
                  abs(current) >= self.spec['grasp_current_ratio']*self.spec['current_a'])
        holding = opening >= self.spec['close_m']+self.spec['grasp_min_opening_m']
        if holding and (stalled or loaded):
            self.state = 'grasped'
        elif not holding and stalled:
            self.state = 'closed_empty'
        else:
            self.state = 'closing'
        return self.state


class GripperFault(RuntimeError):
    pass


class GripperRunner:
    """Owns one EndEffectorClient per configured side on the body controller.

    Construct before EXTERNAL entry (Info/State/output preparation), start()
    after entry, drain()/check() from the body loop, close() before exit.
    close() publishes nothing: OP3 holds its last target after the command
    stream stops, so a grasp survives the session exit.
    Haptics is an optional sink with pulse(side, event); the current PICO
    binding has no controller vibration call, so live passes none.
    """

    def __init__(self, config, shared, *, connection, sdk=None, haptics=None, clock=time.monotonic,
                 profiling=None):
        self.sdk = load_sdk() if sdk is None else sdk
        self.profiling = profiling
        self.feedback_received = {} if profiling is not None else None
        self.config, self.shared, self.haptics, self.clock = config, shared, haptics, clock
        self.mode = getattr(self.sdk.ControlMode, config['control_mode'])
        self.events = deque()
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.clients, self.infos, self.targets = {}, {}, {}
        self.gates = {side: SideGripGate(config) for side in config['sides']}
        self.monitors = {side: GraspMonitor(spec) for side, spec in config['sides'].items()}
        self.commanded = dict.fromkeys(config['sides'])
        self.disabled = set()
        self.output_seen = set()
        self.started_at = None
        self.publishes = dict.fromkeys(config['sides'], 0)
        timeout = config['startup_timeout_s']
        sdk, codec = self.sdk, self.sdk.codec
        try:
            for side, spec in config['sides'].items():
                side_id = sdk.EndEffectorSide.LEFT if side == 'left' else sdk.EndEffectorSide.RIGHT
                client = sdk.EndEffectorClient(
                    side, connection=connection,
                    observation_sources=((sdk.SampleStream.EE_STATE, side_id),))
                self.clients[side] = client
                info = client.wait_info(timeout=timeout)
                self._check_info(side, side_id, info)
                self.infos[side] = info
                fields = sdk.Field.POS | sdk.Field.VEL_LIMIT | sdk.Field.CURRENT
                # Open to the declared maximum. close_m stays the explicit close,
                # including 0, and is still checked against the same Info limits.
                open_m = float(info.pos_max[0])
                if not spec['close_m'] < open_m:
                    raise ValueError(f'{side}: close_m={spec["close_m"]} is not below the '
                                     f'OP3 maximum opening {open_m}')
                try:
                    self.targets[side] = {
                        closed: codec.prepare_target(info, sdk.EndEffectorTarget(
                            (self.mode,), (fields,),
                            pos=(spec['close_m'] if closed else open_m,),
                            vel_limit=(spec['vel_limit_m_s'],), current=(spec['current_a'],)))
                        for closed in (False, True)}
                except ValueError as error:
                    raise ValueError(f'{side}: configured OP3 target rejected by actual Info limits: {error}') from error
                try:
                    state = client.wait_state(timeout=timeout)
                except TimeoutError as error:
                    raise TimeoutError(f'{side}: initial OP3 State did not arrive within {timeout}s') from error
                codec.validate_state(info, state)
                client.wait_output_ready(timeout=timeout)
                self.events.append(dict(
                    record_type='gripper_ready', side=side, device_model=int(info.device_model),
                    firmware=info.firmware, serial=info.serial, axis=info.axis_names[0],
                    control_mode=config['control_mode'], pos_limits_m=[info.pos_min[0], info.pos_max[0]],
                    current_limits_a=[info.current_min[0], info.current_max[0]],
                    vel_limit_max_m_s=info.vel_limit_max[0], open_m=open_m, close_m=spec['close_m'],
                    current_a=spec['current_a'], initial_opening_m=self._opening(state)))
        except BaseException:
            for client in self.clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            raise

    def _check_info(self, side, side_id, info):
        sdk = self.sdk
        sdk.codec.validate_info(info)
        need = sdk.Field.POS | sdk.Field.VEL_LIMIT | sdk.Field.CURRENT
        problems = []
        if info.side != side_id:
            problems.append(f'side={info.side}')
        if info.device_model != sdk.DeviceModel.AGILINK_OP3:
            problems.append(f'device_model={info.device_model} (expected AGILINK_OP3)')
        if len(info.axis_names) != 1 or tuple(info.position_domain) != (sdk.PositionDomain.TOTAL_OPENING,):
            problems.append(f'axes={info.axis_names} position_domain={info.position_domain} '
                            '(expected one TOTAL_OPENING axis in metres)')
        else:
            if info.is_mock[0]:
                problems.append('axis is mock')
            if not info.supported_modes[0] & (1 << self.mode):
                problems.append(f'control_mode {self.config["control_mode"]} unsupported '
                                f'(supported_modes=0x{info.supported_modes[0]:x})')
            if info.supported_command_fields[0] & need != need or info.valid_limits[0] & need != need:
                problems.append('POS/VEL_LIMIT/CURRENT command fields or proven limits missing '
                                '(force limiting requires CURRENT)')
            if not info.supported_state_fields[0] & sdk.Field.POS:
                problems.append('State lacks measured opening')
        if problems:
            raise ValueError(f'{side}: endpoint is not a controllable OP3 gripper: '+'; '.join(problems))

    def _opening(self, state):
        if state is None or not state.valid_fields or not state.valid_fields[0] & self.sdk.Field.POS:
            return None
        return float(state.pos[0])

    def _current(self, state):
        if state is None or not state.valid_fields or not state.valid_fields[0] & self.sdk.Field.CURRENT:
            return None
        return float(state.current[0])

    def _allowed(self, state):
        return (state is not None and state.status_valid_mask & self.sdk.STATUS_REQUIRED == self.sdk.STATUS_REQUIRED
                and state.ready and state.output_allowed
                and state.inhibit_reason == self.sdk.InhibitReason.NONE)

    def _feedback(self, side, now):
        """Validated (latest, position) or None while output is not yet allowed."""
        sdk, client, info = self.sdk, self.clients[side], self.infos[side]
        latest, position, fault, received = client.motion_feedback()
        if self.profiling is not None:
            self.feedback_received[side] = received
        for state in (latest, position, fault):
            if state is not None:
                sdk.codec.validate_state(info, state)
                if any(mask & sdk.Field.ERROR_CODE and code
                       for mask, code in zip(state.valid_fields, state.error_code)):
                    raise GripperFault(f'{side}: OP3 reports error_code={list(state.error_code)}')
        if not self._allowed(latest):
            detail = 'no state' if latest is None else (
                f'ready={latest.ready} output_allowed={latest.output_allowed} '
                f'inhibit_reason={latest.inhibit_reason} status_valid_mask=0x{latest.status_valid_mask:x}')
            if side in self.output_seen or now-self.started_at > self.config['output_ready_timeout_s']:
                raise GripperFault(f'{side}: OP3 output not allowed; {detail}')
            return None
        self.output_seen.add(side)
        # Read the clock after the feedback snapshot: a State callback may land
        # between the tick start and the snapshot, which would give a negative age.
        age = None if received is None else self.clock()-received
        opening = self._opening(position)
        if (position is None or age is None or not 0 <= age <= self.config['feedback_max_age_s']
                or position.sample_status != sdk.SampleStatus.OK or opening is None
                or not info.pos_min[0]-1e-6 <= opening <= info.pos_max[0]+1e-6):
            raise GripperFault(f'{side}: stale or invalid OP3 opening feedback; age_s={age} opening_m={opening} '
                               f'sample_status={None if position is None else position.sample_status}')
        return latest, position

    def _step(self, side, closed, now, reason):
        feedback = self._feedback(side, now)
        if feedback is None:
            return
        latest, position = feedback
        if self.profiling is None:
            self.clients[side].publish(self.targets[side][closed], input_observations=(position.sample_ref,))
        else:
            self.profiling.publish_gripper(self.clients[side], self.targets[side][closed],
                latest, position, self.feedback_received[side],
                phase='wtt_close' if closed else 'wtt_open', rate=self.config['rate_hz'])
        self.publishes[side] += 1
        opening, current = self._opening(position), self._current(latest)
        if closed != self.commanded[side]:
            self.commanded[side] = closed
            command = 'close' if closed else 'open'
            self.events.append(dict(record_type='gripper_command', side=side, command=command, reason=reason,
                                    opening_m=opening, current_a=current))
            print(f'PICO_GRIPPER: {side} {command} ({reason})', flush=True)
        before = self.monitors[side].state
        after = self.monitors[side].update(closed, now, opening, current)
        if after != before:
            self.events.append(dict(record_type='gripper_grasp', side=side, state=after,
                                    previous=before, opening_m=opening, current_a=current))
            if after in ('grasped', 'closed_empty'):
                print(f'PICO_GRIPPER: {side} {after} opening={opening*1000:.1f}mm '
                      f'current={"n/a" if current is None else f"{current:.2f}A"}', flush=True)
            if after == 'grasped':
                self._pulse(side, 'grasped')

    def _pulse(self, side, event):
        if self.haptics is None:
            return
        try:
            self.haptics.pulse(side, event)
        except Exception as error:
            self.events.append(dict(record_type='gripper_haptics_failed', side=side, event=event, error=repr(error)))
            self.haptics = None

    def _fault(self, side, error):
        action = self.config['on_fault']
        self.disabled.add(side)
        opened = self._open(side)
        self.events.append(dict(record_type='gripper_fault', side=side, error=repr(error),
                                action=action, open_published=opened))
        print(f'PICO_GRIPPER FAULT: {side}: {error}; '
              f'{"stopping session" if action == "stop_session" else "side disabled, body continues"}',
              flush=True)
        self._pulse(side, 'fault')
        if action == 'stop_session' and self.error is None:
            self.error = error

    def _open(self, side):
        """Best effort only: an inhibit or fault cannot promise physical opening."""
        try:
            latest, position, _, _ = self.clients[side].motion_feedback()
            observations = () if position is None else (position.sample_ref,)
            if self.profiling is None:
                self.clients[side].publish(self.targets[side][False], input_observations=observations)
            else:
                self.profiling.publish_fault_open(self.clients[side], self.targets[side][False], position)
            return True
        except Exception:
            return False

    def start(self):
        self.started_at = self.clock()
        self.thread = threading.Thread(target=self.run, name='pico-op3-gripper', daemon=True)
        if self.profiling is not None:
            self.profiling.ee_owned_by_worker = True
        try:
            self.thread.start()
        except BaseException:
            if self.profiling is not None:
                self.profiling.ee_owned_by_worker = False
            raise

    def tick(self, now=None):
        now = self.clock() if now is None else now
        if self.started_at is None:
            self.started_at = now
        with self.shared.get_lock():
            received, stamp, left, right = self.shared[3:7]
        for side in self.config['sides']:
            if side in self.disabled:
                continue
            gate = self.gates[side]
            closed = gate.update(received, stamp, left if side == 'left' else right, now)
            reason = 'grip' if closed else 'release' if gate.valid else 'input_invalid'
            try:
                self._step(side, closed, now, reason)
            except Exception as error:
                self._fault(side, error)

    def run(self):
        period = 1./self.config['rate_hz']
        try:
            if self.profiling is not None:
                self.profiling.begin_grippers(self.infos)
            while not self.stop.is_set() and self.error is None:
                self.tick()
                self.stop.wait(period)
        except BaseException as error:
            self.error = error
        finally:
            if self.profiling is not None:
                try:
                    self.profiling.finish_grippers(self.error)
                except BaseException as error:
                    if self.error is None:
                        self.error = error

    def drain(self):
        rows = []
        while self.events:
            rows.append(self.events.popleft())
        return rows

    def check(self):
        if self.error is not None:
            raise RuntimeError('PICO OP3 gripper control failed') from self.error

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        for side, client in self.clients.items():
            try:
                client.close()
            except Exception as error:
                self.events.append(dict(record_type='gripper_close_failed', side=side, error=repr(error)))
        held = {side: ('closed' if self.commanded[side] else 'open' if self.commanded[side] is False else None)
                for side in self.config['sides'] if side not in self.disabled}
        self.events.append(dict(record_type='gripper_closed', publishes=dict(self.publishes),
                                disabled=sorted(self.disabled), held_last_command=held,
                                thread_alive=self.thread is not None and self.thread.is_alive()))
