"""Supported actor/Pico EXTERNAL control using the unmodified SDK.

Importing this module creates no robot handles. Hardware setup belongs to the
explicit CLI live entry. Injectable I/O lets offline tests exercise failures.
"""
from collections import deque
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

from .actor_parallel_ankle import command_space

STATE_AGE = .1
STATUS_AGE = .5
TARGET_PERIOD = .020
# Policy ticks skipped in a row on stale BodyState before cancelling. While a
# tick is skipped the firmware holds the last command until its watchdog
# expires (EXTERNAL watchdog 1000 ms); normal command interpolation is disabled.
MAX_STALE_TICKS = 2
# Firmware ingress_drop_count counts a valid command superseded by a newer one
# before its control tick consumed it (latest wins), plus full control-request
# queues. Occasional supersession is benign; a sustained rate is a fault.
INGRESS_DROP_WINDOW = 1.
MAX_INGRESS_DROPS = 10
# Measured positions can sit slightly past advertised ranges at hard stops or
# within calibration error. Commanded q keeps the exact advertised range.
MEASURED_Q_MARGIN = .03
BODY_FIELDS_ALL = 0x1f  # q|dq|tau|kp|kd, declared on every BodyCmd publish


class PicoWorkerError(RuntimeError):
    """The Pico input worker reported a terminal failure."""


class StaleStateError(TimeoutError):
    """BodyState too old (or from the future) for this policy tick."""


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


def load_robot_config(path, runtime=None):
    """Read the deployment-selected effective YAML; never patch robot config.

    This is not discovery of the running process's selected configuration.
    The attendant must verify that selection in the interactive confirmation.
    With a runtime, the EXTERNAL ankle mode must match its model: virtual
    pitch/roll actors need virtual PD. Motor-theta actors select either native
    motor commands or the legacy virtual-target motor-PD adapter. Match the
    command space before using its limits. Without a runtime, validate the YAML
    structure only (local start configuration).
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
        intent = params['ankle_virtual_pd_intent']
        motor_theta = runtime is not None and getattr(runtime.model.contract, 'ankle', None) is not None
        if type(intent) is not bool:
            raise ValueError('EXTERNAL ankle_virtual_pd_intent must be a boolean')
        space = params.get('ankle_command_space', 'virtual_pitch_roll')
        if space not in ('virtual_pitch_roll', 'motor_theta'):
            raise ValueError('Invalid EXTERNAL ankle_command_space')
        if runtime is not None and space != command_space(runtime.policy):
            raise ValueError('Model deployment/EXTERNAL ankle_command_space mismatch')
        if space == 'motor_theta' and (intent or (runtime is not None and not motor_theta)):
            raise ValueError('motor_theta requires a motor-angle policy and '
                             'ankle_virtual_pd_intent=false (motor PD)')
        raw_targets = params.get('allow_position_targets_outside_mechanical_range', False)
        if type(raw_targets) is not bool:
            raise ValueError('EXTERNAL allow_position_targets_outside_mechanical_range must be a boolean')
        if raw_targets and space != 'motor_theta':
            raise ValueError('Unclipped PD targets require motor_theta command space')
        margin = params.get('position_target_margin_rad')
        if raw_targets:
            if (type(margin) not in (int, float) or not math.isfinite(margin)
                    or not 0 < margin <= math.pi):
                raise ValueError('EXTERNAL allow_position_targets_outside_mechanical_range=true requires '
                                 'a finite position_target_margin_rad in (0, pi]')
        elif margin is not None:
            raise ValueError('EXTERNAL position_target_margin_rad requires '
                             'allow_position_targets_outside_mechanical_range=true')
        if (runtime is not None and space == 'motor_theta'
                and getattr(runtime.policy, 'clip_target_positions', True) is False
                and not raw_targets):
            raise ValueError('Affine native PD targets require EXTERNAL '
                             'allow_position_targets_outside_mechanical_range=true; '
                             'install matching firmware and configuration')
        if motor_theta:
            check_motor_theta_ankle(ankle, intent, runtime, native=space == 'motor_theta')
            if raw_targets and getattr(runtime.policy, 'clip_target_positions', True) is False:
                required = ankle_torque_equivalent_margin(ankle, runtime.model.contract)
                if margin < required:
                    raise ValueError(f'EXTERNAL position_target_margin_rad={margin} would change the '
                                     'trained unclipped ankle PD torque; use at least '
                                     f'{required:.4f} rad (motor PD saturation bound)')
        elif runtime is not None and (ankle['pd']['strategy'] != 'virtual_pd' or intent is not True):
            raise ValueError('Requires virtual_pd and EXTERNAL ankle_virtual_pd_intent=true; no config is changed')
        unknown = set(params)-{'command_timeout_ms', 'recover_valid_frames',
                              'interpolation', 'safe_hold', 'ankle_virtual_pd_intent', 'ankle_command_space',
                              'allow_position_targets_outside_mechanical_range',
                              'position_target_margin_rad'}
        if unknown:
            raise ValueError('Unsupported EXTERNAL params: '+', '.join(sorted(unknown)))
        timeout = params['command_timeout_ms']
        if type(timeout) is not int or timeout <= 0:
            raise ValueError('EXTERNAL command watchdog must be enabled')
        recover = params['recover_valid_frames']
        if type(recover) is not int or recover < 1:
            raise ValueError('EXTERNAL recover_valid_frames must be a positive integer')
        interpolation = params['interpolation']
        interpolation_enabled = interpolation.get('enabled')
        if type(interpolation_enabled) is not bool:
            raise ValueError('EXTERNAL interpolation.enabled must be a boolean and explicitly false')
        if interpolation_enabled:
            raise ValueError('WBC requires EXTERNAL interpolation.enabled=false; '
                             'install matching firmware and configuration')
        periods = [interpolation[key] for key in ('min_period_ms', 'max_period_ms')]
        if (set(interpolation)-{'enabled', 'min_period_ms', 'max_period_ms'}
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
    # Compare trial YAMLs without exposing their contents: only the wire-space
    # selectors may differ in a controlled ankle-path A/B trial. Raw-target
    # opt-in is native-only; legacy wire behavior remains part of that trial.
    comparable = json.loads(json.dumps(root))
    comparable['actions']['EXTERNAL']['params'].pop('ankle_command_space', None)
    comparable['actions']['EXTERNAL']['params'].pop('allow_position_targets_outside_mechanical_range', None)
    comparable['actions']['EXTERNAL']['params'].pop('position_target_margin_rad', None)
    return params, dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
        allow_position_targets_outside_mechanical_range=raw_targets,
        position_target_margin_rad=margin,
        command_interpolation_enabled=interpolation_enabled,
        ankle_path_comparison_sha256=hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest())


def ankle_torque_equivalent_margin(ankle, contract):
    """Smallest travel margin whose target clamp leaves the motor PD torque unchanged.

    With |dq| <= qd_motor_max, a target error of (tau_motor_max + kd*qd_motor_max)/kp
    already saturates the motor PD, so clamping farther from travel cannot change
    the commanded torque of an unclipped training target.
    """
    caps = ankle['caps']
    tau, speed = float(caps['tau_motor_max']), float(caps['qd_motor_max'])
    gains = [(float(contract.kp[i]), float(contract.kd[i])) for i in contract.ankle.slots]
    if not gains or any(not kp > 0 or kd < 0 for kp, kd in gains):
        raise ValueError('Motor-theta ankle gains must be positive')
    return max((tau+kd*speed)/kp for kp, kd in gains)


def check_motor_theta_ankle(ankle, intent, runtime, native=False):
    """Validate native motor caps, or the compatibility IK/fixed-gain contract."""
    contract, profile = runtime.model.contract, runtime.config['body_profile']
    if intent is not False:
        raise ValueError('motor_theta actor requires EXTERNAL ankle_virtual_pd_intent=false (motor PD); '
                         'no config is changed')
    slots = contract.ankle.slots
    gains = [(float(ankle['pd']['kp_motor']), float(ankle['pd']['kd_motor']))]
    if not native and {(float(contract.kp[i]), float(contract.kd[i])) for i in slots} != set(gains):
        raise ValueError(f'Robot ankle motor gains {gains[0]} differ from the trained motor PD')
    caps = (float(ankle['caps']['theta_min']), float(ankle['caps']['theta_max']))
    if not np.allclose(caps, contract.ankle.motor_limits, rtol=0, atol=1e-9):
        raise ValueError('Robot ankle motor limits differ from the model')
    for name, value in contract.ankle.geometry.items():
        if not math.isclose(ankle['geometry'][name], value, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Model/robot ankle geometry differs: '+name)
    if native:
        caps = ankle['caps']
        pd = ankle['pd']
        ranges = [tuple(contract.ankle.motor_limits),
                  (-float(caps['qd_motor_max']), float(caps['qd_motor_max'])),
                  (-float(caps['tau_motor_max']), float(caps['tau_motor_max'])),
                  (0., float(pd['kp_motor_max'])), (0., float(pd['kd_motor_max']))]
        if any(not np.isfinite(r).all() or r[0] >= r[1] for r in ranges):
            raise ValueError('Invalid native motor limits')
        if any(not 0 <= contract.kp[i] <= ranges[3][1] or
               not 0 <= contract.kd[i] <= ranges[4][1] for i in slots):
            raise ValueError('Model gains exceed native motor limits')
        if not 0 <= gains[0][0] <= ranges[3][1] or not 0 <= gains[0][1] <= ranges[4][1]:
            raise ValueError('Native motor hold gains exceed limits')
        # BodyCmd fields and firmware MotorAnkleRange are float32. Compare
        # against the same representable bounds, without a numerical margin.
        with np.errstate(over='ignore', invalid='ignore'):
            wire_ranges = np.asarray(ranges, dtype=np.float32)
        if (not np.isfinite(wire_ranges).all()
                or np.any(wire_ranges[:, 0] >= wire_ranges[:, 1])):
            raise ValueError('Native motor limits cannot be represented as float32')
        runtime.policy.native_motor_ranges = [tuple(map(float, row)) for row in wire_ranges]
        runtime.policy.native_hold_gains = gains[0]
        return
    # Validate the legacy virtual wire's geometry/range contract.
    for axis in ('pitch', 'roll'):
        box = [float(x) for x in ankle['virtual_limits'][axis]]
        for side in ('left', 'right'):
            if not np.allclose(profile['joint_limits'][f'{side}_ankle_{axis}_joint'], box, rtol=0, atol=1e-9):
                raise ValueError(f'Robot virtual ankle {axis} limits differ from the actor profile')


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
        # Policy setpoints encode PD effort and may exceed physical travel.
        # Position ranges still validate measured/startup poses; other command
        # fields retain their advertised limits during policy execution.
        self.unbounded_target_joints = (set(runtime.model.contract.names)
            if getattr(runtime.policy, 'clip_target_positions', True) is False else set())
        self.limits = []
        self.delegated_joints = []
        self.measured_q_ranges = []
        self.virtual_limits = []
        ankles = {'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                  'right_ankle_pitch_joint', 'right_ankle_roll_joint'}
        for index, (name, cap) in enumerate(zip(self.body.joint_names, self.body.joints)):
            self.measured_q_ranges.append((cap.q_min, cap.q_max) if not cap.is_mock else None)
            if cap.is_mock:
                self.limits.append(None)
                self.virtual_limits.append(None)
                continue
            if cap.slot_role != 1:
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
            self.virtual_limits.append(list(ranges))
            native = name in ankles and command_space(runtime.policy) == 'motor_theta'
            if native:
                if not hasattr(runtime.policy, 'native_motor_ranges'):
                    raise ValueError('Native motor mode requires effective robot config validation')
                ranges = runtime.policy.native_motor_ranges
            for key, value_range in (('kp', ranges[3]), ('kd', ranges[4])):
                values = params['safe_hold'][key]
                slot = index if len(values) == 31 else BODY_TO_LEGACY[index]
                value = (runtime.policy.native_hold_gains[0 if key == 'kp' else 1]
                         if native else values[slot])
                if (not math.isfinite(value) or value <= 0 or (value_range is not None
                        and not value_range[0] <= value <= value_range[1])):
                    raise ValueError('safe_hold gain unavailable/out of bounds: '+name)
            self.limits.append(ranges)
        self.target_names = tuple(name for name, cap in zip(self.body.joint_names, self.body.joints)
                                  if not cap.is_mock)
        if runtime.head is not None:
            runtime.head.bind_body_info(self.body)
        # A fixed model/gain profile, including both necks, must fit BEFORE entry.
        command, _ = runtime.policy.decode(np.zeros(28))
        self.check(command)
        self.profile_command = command
        runtime.policy.reset()

    def check_measured_pose(self, state):
        # Observations are measurements, not decoded policy targets. Match the
        # SDK example: straight knees may be below the model's output bounds.
        for i, bounds in enumerate(self.measured_q_ranges):
            if bounds is not None and (not math.isfinite(state.q[i]) or
                    not bounds[0]-MEASURED_Q_MARGIN <= state.q[i] <= bounds[1]+MEASURED_Q_MARGIN):
                raise ValueError('Measured virtual joint outside bounds: '+self.body.joint_names[i])

    def check(self, command, q_margin=0., *, virtual=False):
        if len(command.joints) != 31:
            raise ValueError('Command must contain exactly 31 slots')
        limits = self.virtual_limits if virtual else self.limits
        for name, joint, ranges in zip(self.body.joint_names, command.joints, limits):
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
                if key == 'q' and not virtual and name in self.unbounded_target_joints:
                    continue
                if bounds is None:
                    continue  # Validated by the firmware's parallel ankle adapter.
                low, high = bounds
                if key == 'q':
                    low, high = low-q_margin, high+q_margin
                if not low <= value <= high:
                    raise ValueError(f'{name}.{key}={value} outside [{low}, {high}]')


class Inputs:
    """Body31 feedback plus, when following automatically, the Pico worker snapshot.

    ``pico`` is a PicoReferenceProcess-like worker (connect/snapshot). GMR and
    the reference window run in that process; reading never retargets here.
    """
    def __init__(self, runtime, controller, pico, clock):
        self.runtime, self.controller, self.pico, self.clock = runtime, controller, pico, clock
        self.previous_stamp = 0
        self.last_pico_received = None
        self.snapshot = None
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

    def ages(self, state):
        now_ns, wall_ns = round(self.clock.monotonic()*1e9), self.clock.time_ns()
        return dict(receipt_age_ms=(now_ns-state.received_monotonic_ns)/1e6,
                    source_age_ms=(wall_ns-state.ts_state_real_us*1000)/1e6,
                    publication_age_ms=(wall_ns-state.ts_state_pub_us*1000)/1e6)

    def fresh(self, state):
        ages = self.ages(state)
        detail = ', '.join(f'{k}={v:.3f}' for k,v in ages.items())
        detail += f', state_id={state.state_id}, limit_ms={STATE_AGE*1000:g}'
        if state.ts_state_real_us <= 0 or state.ts_state_pub_us <= 0:
            raise ValueError('BodyState source/publication timestamps are missing: '+detail)
        if not 0 <= ages['receipt_age_ms'] <= STATE_AGE*1000:
            raise StaleStateError('BodyState local receipt is stale/future: '+detail)
        # As in the SDK WBC demo, gate on local receipt and producer publication
        # age. The sensor stamp age is reported in the detail only.
        if not -5 <= ages['publication_age_ms'] <= STATE_AGE*1000:
            raise StaleStateError('BodyState publication is stale or S100 clock contract differs: '+detail)

    def read(self, timeout=0):
        state = self.controller.wait_state(timeout)
        q, dq, root, gyro = self.validate_state(state)
        if self.pico is not None:
            self.pico.connect(root)  # Idempotent; the worker starts on first feedback.
            snapshot = self.pico.snapshot()
            if snapshot.get('state') == 'failed':
                raise PicoWorkerError('Pico input worker failed: '+str(snapshot.get('error')))
            if 'received_monotonic' in snapshot:
                self.last_pico_received = float(snapshot['received_monotonic'])
            self.snapshot = snapshot
        return state, q, dq, root, gyro

    def reference_ready(self):
        return self.snapshot is not None and self.snapshot.get('state') == 'ready'

    def require_pico(self):
        stamp = self.last_pico_received
        if stamp is None or not 0 <= self.clock.monotonic()-stamp <= self.pico_max_age:
            raise TimeoutError('Pico input is stale/future')


class StatusGuard:
    def __init__(self, clock, initial_actions=('PASSIVE',)):
        self.clock, self.identity, self.changed = clock, None, clock.monotonic()
        self.initial_actions = initial_actions
        self.drops = deque()

    def ingress_drops(self, status, entered):
        total = status.ingress_drop_count-entered.ingress_drop_count
        if total < 0:
            raise RuntimeError('EXTERNAL ownership/state or reject/drop counters changed')
        now = self.clock.monotonic()
        if not self.drops:
            # Include increments since entry, even before the first status poll.
            self.drops.append((now, 0))
        self.drops.append((now, total))
        while len(self.drops) > 1 and self.drops[1][0] <= now-INGRESS_DROP_WINDOW:
            self.drops.popleft()
        recent = total-self.drops[0][1]
        if recent > MAX_INGRESS_DROPS:
            raise RuntimeError(f'Firmware superseded {recent} EXTERNAL commands within '
                               f'{INGRESS_DROP_WINDOW:g} s (limit {MAX_INGRESS_DROPS})')
        return total

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
                  for key in ('runtime_id', 'generation', 'client_id', 'control_id', 'reject_count'))):
            raise RuntimeError('EXTERNAL ownership/state or reject/drop counters changed')
        else:
            self.ingress_drops(status, entered)
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
    playback = {}
    if snapshot.get('stream') is not None:
        # Resample the worker's samples on this tick's clock: the window
        # advances by exactly the elapsed control time between ticks.
        motion, playback = runtime.reference.playback(snapshot['stream'], now,
                                                      runtime.stream.playback_margin_s)
    else:
        motion = runtime.math.Motion(snapshot['q'], snapshot['base'], snapshot['quat'])
    if len(motion.q) != runtime.reference.steps:
        raise ValueError('Pico worker reference window length differs from model')
    return motion, dict(pico_age_s=age, reference_delay_s=runtime.reference.delay,
        source_gap_s=float(snapshot.get('source_gap_s',0.)),
        retarget_ms=float(snapshot['retarget_ms']),
        gmr_task_error=float(snapshot['gmr_task_error']),
        **({'retarget_cpu_ms': float(snapshot['retarget_cpu_ms'])} if 'retarget_cpu_ms' in snapshot else {}),
        **playback)


def reference_yaw(runtime, motion, root):
    """Heading offset aligning the GMR world root to the measured robot heading."""
    return runtime.math.yaw(root)-runtime.math.yaw(motion.quat[runtime.reference.current_index])


def pico_reference_command(runtime, snapshot, now, *, q, dq, root, gyro, yaw_offset=None):
    """Policy command on a worker snapshot; a late reference holds its current frame."""
    from .actor_teleop_modes import active_planner
    planner = active_planner(runtime)
    modes=getattr(runtime,'teleop_modes',None)
    if planner is not None:
        try:
            upper, diagnostics = manual_reference(runtime, snapshot, now)
            upper_q = upper.q if diagnostics['pico_age_s'] <= runtime.config['pico_hold_age_s'] else None
        except (ValueError, TimeoutError):
            upper_q, diagnostics = None, {}
        sticks=modes.stick_sample(snapshot,now) if modes is not None else snapshot.get('planner_sticks')
        motion, heading, extra = planner.tick(upper_q, sticks, now, root, robot_q=q,
            input_enabled=modes is None or modes.stage!='entry')
        if modes is not None:
            if extra['planner_state'] not in ('loading','waiting_previous'):
                motion,heading=modes.filter(motion,heading)
            motion=modes.follow_upper(motion,upper_q)
            extra.update(teleop_mode=modes.mode,teleop_mode_stage=modes.stage,
                stick_armed=planner.stick.armed,stick_needs_center=modes.needs_center,
                reference_limiter_active=modes.entry is not None)
        diagnostics.update(extra)
        command, obs, row = runtime.infer_motion_command(motion,q=q,dq=dq,root=root,gyro=gyro,
            diagnostics=diagnostics,yaw_offset=heading,ref_time=0.)
        row['pico_reference'] = dict(q=motion.q.tolist(),root_wxyz=motion.quat.tolist(),
            base=motion.base.tolist(),ref_time=0.,yaw_offset=heading,source='planner_legs_pico_arms',
            timestamp_ns=snapshot.get('pico_timestamp_ns'),sticks=snapshot.get('planner_sticks'))
        return command, obs, row
    motion, diagnostics = manual_reference(runtime, snapshot, now)
    if yaw_offset is None:
        yaw_offset = reference_yaw(runtime, motion, root)
    motion, diagnostics['reference_hold'] = runtime.hold_stale_reference(
        motion, diagnostics['pico_age_s'])
    if modes is not None:
        upper_q=motion.q.copy()
        motion,yaw_offset=modes.filter(motion,yaw_offset)
        motion=modes.follow_upper(motion,upper_q)
        diagnostics.update(teleop_mode=modes.mode,teleop_mode_stage=modes.stage)
    command, obs, row = runtime.infer_motion_command(motion, q=q, dq=dq,
        root=root, gyro=gyro, diagnostics=diagnostics, yaw_offset=yaw_offset,
        ref_time=runtime.reference.ref_time)
    row['pico_reference'] = dict(q=motion.q.tolist(), root_wxyz=motion.quat.tolist(),
        base=motion.base.tolist(), ref_time=runtime.reference.ref_time,
        raw_body_poses=snapshot.get('raw_body_poses'),
        source_log=snapshot.get('source_log'),
        timestamp_ns=snapshot.get('pico_timestamp_ns'), yaw_offset=yaw_offset)
    return command, obs, row


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
             request_action=None, motion_source=None):
    """One trial only. Every attempted entry is followed by checked CANCEL.

    Fake I/O in unit tests exercises this same runner; actual transport and
    hardware performance remain separate acceptance checks.
    """
    if not math.isfinite(args.seconds) or args.seconds < 0:
        raise ValueError('Live duration must be finite and nonnegative; 0 means until STOP')
    params, receipt = load_robot_config(args.robot_config, runtime)
    if motion_source is not None:
        write(dict(record_type='robot_configuration', robot_config=receipt, actions_sent=0))
    envelope = CommandEnvelope(runtime, controller.get_robot_info(args.startup_timeout), params)
    pico_max_age = runtime.config['pico_max_age_s']
    pico_loss_action = runtime.config.get('pico_loss_action', 'stand')
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
    if motion_source is not None and (manual or wait_pico or auto_pico or args.stand_only):
        raise ValueError('File motion cannot combine Pico following or stand-only mode')
    requires_pico = motion_source is None and not args.stand_only and not manual
    if auto and ((not manual and motion_source is None) or args.stand_only):
        raise ValueError('One-click startup requires manual Pico mode')
    # Firmware retires completed keyframes, leaving current_action empty. Auto
    # explicitly requests FIXED_STAND and verifies measured targets afterward;
    # an empty name alone is never treated as completed fixed stand.
    initial_actions = ('PASSIVE', 'QUIET_PASSIVE', 'FIXED_STAND', '') if auto else ('PASSIVE',)
    inputs = Inputs(runtime, controller, pico if requires_pico else None, clock)
    guard = StatusGuard(clock, initial_actions=initial_actions)
    if manual and manual_pico is None:
        raise ValueError('Manual Pico mode requires an idle capture worker')
    gripper_config = getattr(args, 'gripper_config', None)
    if gripper_config is not None:
        if not manual or args.stand_only or motion_source is not None:
            raise ValueError('PICO gripper control requires live manual Pico mode without stand-only or file motion')
        if len(getattr(manual_pico, 'controls', None) or ()) < 7:
            raise ValueError('PICO gripper control requires the 7-slot controller array (grip snapshots)')
        from .actor_pico_gripper import GripperRunner, load_config as load_gripper_config
        gripper_config = load_gripper_config(gripper_config)
    runtime.reset_inputs()
    # Warm actual inference before taking ownership; erase its action history.
    runtime.infer_motion(runtime.hold_motion(),
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
        if (not requires_pico or inputs.reference_ready()) and status is not None:
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
                command, obs, row = pico_reference_command(runtime, inputs.snapshot, started,
                    q=q, dq=dq, root=root, gyro=gyro)
            else:
                motion = runtime.hold_motion()
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
        policy=runtime.policy_name, model=runtime.config['artifacts']['model'], config=receipt,
        stand_only=args.stand_only, seconds=args.seconds,
        manual_pico=manual, auto_start=auto, entry_policy='policy-online',
        reference_source='npz' if motion_source is not None else 'pico',
        motion=motion_source.metadata if motion_source is not None else None,
        wait_pico=wait_pico, auto_pico=auto_pico,
        startup_plan=startup.review if auto else None,
        default_pose=dict(zip(runtime.model.contract.names, runtime.model.contract.default.tolist())),
        kp=dict(zip(runtime.model.contract.names, runtime.model.contract.kp.tolist())),
        kd=dict(zip(runtime.model.contract.names, runtime.model.contract.kd.tolist())),
        fixed_joints=runtime.config['body_profile']['fixed_joints'],
        reference_delay_s=runtime.reference.delay if motion_source is None else 0.,
        human_height_m=runtime.config['retargeting']['human_height_m'] if motion_source is None else None,
        gmr_calibration=runtime.gmr_calibration if motion_source is None else None,
        pico_loss_action=pico_loss_action if motion_source is None else None,
        pico_stream=runtime.config.get('pico_stream') if motion_source is None else None,
        pico_worker=runtime.config.get('pico_worker'),
        head_control=runtime.config.get('head_control') if runtime.head is not None else None)), flush=True)
    print('ACTOR_START: motion requested; '+
          ('running until STOP/Ctrl-C or fault' if args.seconds == 0 else f'duration limit {args.seconds:g}s')+
          '; exit requests CANCEL -> PASSIVE. Keep support and E-stop available.',flush=True)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Refresh effective configuration before requesting any motion.
    if load_robot_config(args.robot_config, runtime)[1] != receipt:
        raise RuntimeError('Effective config file changed during preflight')
    if auto:
        if not fixed_stand_startup(startup, envelope, runtime, args, controller, inputs,
                                  stop, write, clock, request_action=request_action,
                                  poll_command=poll_command):
            return dict(mode='live', actions_sent=0, entered=False, stopped_by_signal=True)
    startup_generation = controller.status.generation if auto else None
    cancel_action = (request_action or FirmwareActions(stop, clock, lambda:inputs.read(.1))) if auto else None
    try:
        if auto and load_robot_config(args.robot_config, runtime)[1] != receipt:
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
            lead = runtime.hold_motion().lead_in(
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
                  missed_policy_ticks=0, duration_limit_s=args.seconds, pico_losses=0)
    grippers = None

    def write_gripper_events():
        for row in grippers.drain():
            write(dict(row, actions_sent=sent_count))

    try:
        if stop.is_set():
            return report
        if gripper_config is not None:
            # Info, State and output preparation complete before entry; output
            # permission is only checked once the gripper thread starts.
            grippers = GripperRunner(gripper_config, manual_pico.controls, connection=controller)
            write_gripper_events()
        attempted = True
        entered = controller.enter_external(timeout=5.)
        if (entered.current_action != 'EXTERNAL' or phase(entered) != 'RUNNING'
                or entered.client_id != controller.client_id or not entered.control_id
                or entered.generation <= 0):
            raise RuntimeError('Entry did not confirm owned EXTERNAL/RUNNING')
        report['entered'] = True
        write(dict(record_type='external_entered', status=status_row(entered), actions_sent=0))
        if grippers is not None:
            grippers.start()
            print('PICO_GRIPPER: hold a side grip to close that gripper, release to open; '
                  'lost or stale input opens.', flush=True)
        runtime.reset_inputs()
        state, q, dq, root, gyro = inputs.read(0)
        if requires_pico:
            inputs.require_pico()
        if auto:
            startup.check_hardware_pose(envelope,state)
        if not auto:
            envelope.check_measured_pose(state)
        lead = runtime.hold_motion().lead_in(
            q, root, runtime.config['body_profile']['lead_in'])
        offset = runtime.math.yaw(root)
        preparation = max(3., (len(lead.q)-1)*runtime.model.contract.period)
        start = deadline = clock.monotonic()
        end = start+args.seconds if args.seconds > 0 else math.inf
        policy_tick = 0
        stand_tick = 0
        tracking = False
        pico_losses = 0
        pico_reengage_blocked = False
        motion_complete = False
        if runtime.head is not None and not args.stand_only and motion_source is None:
            print('PICO_HEAD: look straight ahead relative to your chest when enabling following; '
                  'this pose becomes head neutral.', flush=True)
        last_inferred_identity = None
        last_inferred_receipt_ns = None
        report['duplicate_body_ticks'] = 0
        if auto:
            print('POLICY_LEAD_IN: ONNX runs from measured-pose reference toward model stand; '+
                  ('then fixed NPZ playback' if motion_source is not None else 'awaiting Pico'),flush=True)
        if manual:
            print('PICO_BUTTONS: release both triggers, then squeeze both past 0.7 to toggle '
                  'following/ONNX stand. STOP or Ctrl-C cancels.',flush=True)
        previous_pico_state = None
        previous_controller_state = None
        last_trigger_event = 0
        modes=getattr(runtime,'teleop_modes',None)
        if modes is not None:
            print('TELEOP_MODE: full_body; center and click right stick to switch. Right Y=forward/back, X=turn in joystick mode.',flush=True)
        if manual and wait_pico:
            manual_pico.connect(root)
            print('PICO_WAIT: receiver starting; ONNX stand does not depend on receiving Pico',flush=True)
            write(dict(record_type='pico_connect_requested',automatic=True,actions_sent=0))
        stale_ticks = 0
        logged_drops = entered.ingress_drop_count

        def skip_stale(phase_name, error, **details):
            nonlocal stale_ticks
            stale_ticks += 1
            cancel = stale_ticks > MAX_STALE_TICKS
            write(dict(record_type='body_state_age_failure', phase=phase_name,
                       consecutive_ticks=stale_ticks, action='cancel' if cancel else 'skip_tick',
                       actions_sent=sent_count, error=str(error), **details))
            if cancel:
                raise StaleStateError(f'BodyState stale for {stale_ticks} consecutive policy ticks: {error}')

        def handle_pico_loss(error, q, root):
            """Pico lost while following: return to ONNX stand, same EXTERNAL session.

            Only an unusable Pico reference during tracking takes this path;
            robot-feedback, firmware and envelope faults still cancel. After a
            fault there is no automatic re-engagement: manual mode needs a fresh
            double-trigger (same as PICO_STANDBY), requires_pico mode keeps
            standing until STOP. q/root are the latest validated measurement.
            """
            nonlocal tracking, auto_pico, lead, offset, stand_tick, preparation
            nonlocal pico_losses, pico_reengage_blocked
            if not tracking or motion_source is not None or pico_loss_action != 'stand':
                return False
            tracking = False
            auto_pico = False
            if modes is not None:
                modes.cancel();modes.upper_follow=None
            pico_reengage_blocked = True
            # Required-Pico mode now stands using robot feedback only. A
            # terminal worker snapshot must not keep failing Inputs.read().
            # Manual re-engagement reads its separate manual_pico source.
            inputs.pico = None
            lead = runtime.hold_motion().lead_in(q, root, runtime.config['body_profile']['lead_in'])
            offset = runtime.math.yaw(root)
            stand_tick = 0
            preparation = max(.3, (len(lead.q)-1)*runtime.model.contract.period)
            pico_losses += 1
            report['pico_losses'] = pico_losses
            print('ACTOR_WARNING: Pico lost during following: '+str(error)+
                  '; returning to ONNX stand, EXTERNAL session stays active. '+
                  ('Re-engage with a fresh double-trigger once Pico recovers; '
                   if manual else 'No re-engagement this run; ')+
                  'STOP/Ctrl-C still cancels to PASSIVE.', file=sys.stderr, flush=True)
            write(dict(record_type='pico_loss_stand', error=str(error), loss_count=pico_losses,
                       transition_start_q=q.tolist(), policy_history_reset=False,
                       actions_sent=sent_count))
            return True

        def advance(current):
            following, missed = next_policy_deadline(current, clock.monotonic(), runtime.model.contract.period)
            report['missed_policy_ticks'] += missed
            if missed:
                write(dict(record_type='policy_ticks_skipped', missed_ticks=missed,
                           total_missed_ticks=report['missed_policy_ticks'], actions_sent=sent_count))
            return following

        while not stop.is_set() and clock.monotonic() < end:
            if motion_source is not None:
                poll_command()  # Terminal STOP/EOF sets the same cancellation event.
                if stop.is_set():
                    break
            if requires_pico and not service_alive() and not pico_reengage_blocked:
                error = RuntimeError('Pico service exited')
                if not handle_pico_loss(error, q, root):
                    raise error
                deadline = advance(deadline)
                continue
            clock.sleep(max(0., deadline-clock.monotonic()))
            if stop.is_set() or clock.monotonic() >= end:
                break
            cycle = clock.monotonic()
            cpu_clock = getattr(clock,'thread_time',None)
            cpu_start = cpu_clock() if cpu_clock is not None else None
            status = guard.check(controller.status, entered)
            if grippers is not None:
                write_gripper_events()
                grippers.check()
            if status.ingress_drop_count != logged_drops:
                write(dict(record_type='firmware_ingress_drop', previous=logged_drops,
                           count=status.ingress_drop_count, actions_sent=sent_count))
                logged_drops = status.ingress_drop_count
            status_end = clock.monotonic()
            try:
                state, q, dq, root, gyro = inputs.read(0)
            except StaleStateError as error:
                skip_stale('at_read', error)
                deadline = advance(deadline)
                continue
            except PicoWorkerError as error:
                # q/root are the previous tick's validated measurement here.
                if not handle_pico_loss(error, q, root):
                    raise
                deadline = advance(deadline)
                continue
            state_age_at_read = inputs.ages(state)
            state_end = clock.monotonic()
            identity = (int(state.state_id), int(state.ts_state_real_us))
            if identity == last_inferred_identity:
                # PullStateSource returns the cached snapshot when its keep-last
                # ring is empty. Do not advance model history/reference or issue
                # a new action from that same measurement. A republished heartbeat
                # may refresh receipt/publication clocks without new feedback;
                # bound it by the original receipt and the existing age budget.
                unchanged_ms = (round(clock.monotonic()*1e9)-last_inferred_receipt_ns)/1e6
                if unchanged_ms > STATE_AGE*1000:
                    skip_stale('at_read', StaleStateError(
                        f'BodyState identity did not advance: state_id={state.state_id}, '
                        f'unchanged_age_ms={unchanged_ms:.3f}, limit_ms={STATE_AGE*1000:g}'))
                else:
                    report['duplicate_body_ticks'] += 1
                    write(dict(record_type='body_state_not_advanced', state_id=int(state.state_id),
                        source_state_us=int(state.ts_state_real_us), ages=state_age_at_read,
                        unchanged_age_ms=unchanged_ms, action='skip_tick', actions_sent=sent_count))
                deadline = advance(deadline)
                continue
            if requires_pico and not pico_reengage_blocked:
                try:
                    inputs.require_pico()
                except TimeoutError as error:
                    if not handle_pico_loss(error, q, root):
                        raise
                    deadline = advance(deadline)
                    continue
            # Match SDK ReferenceSession: one reference frame per successful
            # inference/publication, independent of wall-clock scheduling jitter.
            elapsed = stand_tick*runtime.model.contract.period
            reference_frame = min(stand_tick,len(lead.q)-1)
            reference_time = reference_frame*runtime.model.contract.period
            if not tracking and getattr(runtime, 'planner', None) is not None and runtime.planner.active:
                runtime.planner.reset()
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
                if modes is not None:
                    # Pause has priority over a concurrent mode click. Consume
                    # rejected edges so held/cached buttons cannot re-engage.
                    if request=='toggle':
                        modes.cancel();modes.upper_follow=None
                    event=modes.request(snapshot,clock.monotonic(),tracking=tracking and request!='toggle',
                        q=q,dq=dq,root=root,gyro=gyro)
                    if event is not None:
                        write(dict(event,actions_sent=sent_count))
                        print('TELEOP_MODE: '+str(event),flush=True)
                        if event['record_type']=='teleop_mode_requested':request='toggle'
                    if not tracking and modes.pending is not None:
                        try:
                            candidate,diag=manual_reference(runtime,snapshot,clock.monotonic())
                            if diag['pico_age_s']>runtime.config['pico_hold_age_s']:candidate=None
                        except (ValueError,TimeoutError):
                            candidate=None;modes.cancel()
                        if modes.ready(snapshot,clock.monotonic(),q=q,dq=dq,root=root,gyro=gyro,
                                preparation_done=elapsed>=preparation,motion=candidate):
                            request='track'
                            print('TELEOP_MODE: '+modes.mode+'; reference handover, then center stick to arm.',flush=True)
                            write(dict(record_type='teleop_mode_selected',mode=modes.mode,actions_sent=sent_count))
                from .actor_teleop_modes import active_planner
                planner = active_planner(runtime)
                if planner is not None and tracking:
                    if request == 'toggle' and planner.stop_started is None:
                        planner.stop_started = clock.monotonic()
                        write(dict(record_type='planner_stop_requested', actions_sent=sent_count))
                    if planner.stop_started is not None:
                        # Let the generated gait decelerate before returning to stand.
                        legs = [i for i,n in enumerate(runtime.model.contract.names)
                                if any(k in n for k in ('hip','knee','ankle'))]
                        settled = (clock.monotonic()-planner.stop_started >= 2.
                                   and np.max(np.abs(dq[legs])) < .4)
                        if modes is not None:
                            quiet=(np.max(np.abs(dq[legs]))<.15 and np.max(np.abs(gyro))<.1
                                and planner.future is None and planner.clip is not None
                                and planner.cursor>=planner.origin+len(planner.clip[0])-1)
                            modes.quiet_since=(clock.monotonic() if modes.quiet_since is None else modes.quiet_since) if quiet else None
                            settled=settled and modes.quiet_since is not None and clock.monotonic()-modes.quiet_since>=.5
                        request = 'toggle' if settled else None
                if request == 'toggle' and tracking:
                    # An intentional standby transition; a source fault takes
                    # handle_pico_loss (pico_loss_action) instead of this path.
                    if planner is None:
                        manual_reference(runtime,snapshot,clock.monotonic())
                    tracking = False
                    auto_pico = False
                    transition_q=q if modes is None or modes.last_q is None else modes.last_q
                    transition_root=root if modes is None or modes.last_root is None else modes.last_root
                    stand_target=runtime.hold_motion()
                    if modes is not None and modes.pending is not None:
                        # R3 returns only the lower body to stand. Arm targets
                        # continue from Pico below, using the same filter state.
                        stand_target=runtime.math.Motion(stand_target.q.copy(),stand_target.base.copy(),stand_target.quat.copy())
                        slots=modes.upper_follow.slots
                        stand_target.q[:,slots]=transition_q[slots]
                    lead = stand_target.lead_in(
                        transition_q,transition_root,runtime.config['body_profile']['lead_in'])
                    offset = runtime.math.yaw(transition_root)
                    if modes is not None:modes.paused()
                    stand_tick = 0
                    elapsed = reference_time = 0.
                    reference_frame = 0
                    preparation = max(.3,(len(lead.q)-1)*runtime.model.contract.period)
                    request = None
                    print('PICO_STANDBY: returning reference to ONNX stand; EXTERNAL session stays active.',flush=True)
                    write(dict(record_type='tracking_paused',actions_sent=sent_count,
                               transition_start_q=transition_q.tolist(),policy_history_reset=False))
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
                        tracking_yaw = reference_yaw(runtime, motion, root)
                        if modes is not None:
                            modes.engage(lead.q[-1],runtime.math.multiply(runtime.math.yaw_quaternion(offset),lead.quat[-1]))
                        print('PICO_TRACKING: same EXTERNAL session and policy history.', flush=True)
                        write(dict(record_type='tracking_started',actions_sent=sent_count,
                                   reference_yaw_offset=tracking_yaw))
            published_action = runtime.policy.last_action.copy()
            try:
                if motion_source is not None and elapsed >= preparation:
                    if not tracking:
                        motion_source.begin(q, root)
                        tracking = True
                        write(dict(record_type='motion_started', actions_sent=sent_count,
                            motion=motion_source.metadata, lead_frames=motion_source.lead_frames,
                            total_ticks=motion_source.total_ticks, policy_history_reset=False))
                        print('MOTION_PLAYING: one pass, live robot feedback; STOP/Ctrl-C cancels.',flush=True)
                    if motion_source.finished:
                        motion_complete = True
                        break
                    reference_frame = motion_source.tick
                    reference_time = reference_frame*runtime.model.contract.period
                    command, obs, row = motion_source.infer(q=q, dq=dq, root=root, gyro=gyro)
                elif manual and tracking:
                    command, obs, row = pico_reference_command(runtime, snapshot, clock.monotonic(),
                        q=q, dq=dq, root=root, gyro=gyro, yaw_offset=tracking_yaw)
                elif requires_pico and not pico_reengage_blocked and elapsed >= preparation:
                    if not tracking:
                        if not inputs.reference_ready():
                            raise TimeoutError('Pico reference window not ready after preparation')
                        motion, _ = manual_reference(runtime, inputs.snapshot, clock.monotonic())
                        tracking_yaw = reference_yaw(runtime, motion, root)
                        tracking = True
                        write(dict(record_type='tracking_started', actions_sent=sent_count,
                                   reference_yaw_offset=tracking_yaw))
                    command, obs, row = pico_reference_command(runtime, inputs.snapshot, clock.monotonic(),
                        q=q, dq=dq, root=root, gyro=gyro, yaw_offset=tracking_yaw)
                else:
                    stand_motion=lead
                    if modes is not None and modes.upper_follow is not None:
                        try:
                            upper,diag=manual_reference(runtime,snapshot,clock.monotonic())
                            upper_q=upper.q if diag['pico_age_s']<=runtime.config['pico_hold_age_s'] else None
                        except (ValueError,TimeoutError):upper_q=None
                        # Infer a current-frame window so the persistent arm
                        # filter is evaluated at frame zero, not a lead-in offset.
                        start=min(round(reference_time/runtime.model.contract.period),len(lead.q)-1)
                        indices=np.minimum(start+np.arange(runtime.reference.steps),len(lead.q)-1)
                        stand_motion=runtime.math.Motion(lead.q[indices],lead.base[indices],lead.quat[indices])
                        stand_motion=modes.follow_upper(stand_motion,upper_q)
                    command, obs, row = runtime.infer_motion_command(stand_motion, q=q, dq=dq,
                        root=root, gyro=gyro, ref_time=0. if stand_motion is not lead else reference_time, yaw_offset=offset)
            except (ValueError, TimeoutError) as error:
                if not handle_pico_loss(error, q, root):
                    raise
                deadline = advance(deadline)
                continue
            last_inferred_identity = identity
            last_inferred_receipt_ns = int(state.received_monotonic_ns)
            # SDK decode updates last_action. Commit that candidate only after
            # publication succeeds; skipped ticks keep the last published action.
            pending_action = runtime.policy.last_action.copy()
            runtime.policy.last_action = published_action
            if modes is not None:
                row['upper_body_source']='pico' if modes.upper_follow is not None else 'stand'
            command, pending_head = runtime.apply_head(command, row, now=clock.monotonic(),
                measured_q=state.q, tracking=(tracking or (modes is not None and modes.upper_follow is not None)) and motion_source is None and (
                    getattr(runtime, 'planner', None) is None or (snapshot is not None and
                    ('head_snapshot' in snapshot or 'received_monotonic' in snapshot))),
                snapshot=(snapshot if manual else inputs.snapshot) if (tracking or (modes is not None and modes.upper_follow is not None)) else None)
            envelope.check(command)
            compute_end = clock.monotonic()
            record_period_overrun(write,'compute',compute_end-cycle,sent_count,
                status_ms=(status_end-cycle)*1000,state_ms=(state_end-status_end)*1000,
                policy_ms=row.get('total_ms'),
                thread_cpu_ms=(cpu_clock()-cpu_start)*1000 if cpu_start is not None else None)
            try:
                inputs.fresh(state)
            except StaleStateError as error:
                # Keep the measured observation; the pending action is not committed.
                skip_stale('before_publish', error, state_id=int(state.state_id),
                    at_read=state_age_at_read, at_failure=inputs.ages(state),
                    compute_ms=(compute_end-cycle)*1000, policy_ms=row.get('total_ms'))
                deadline = advance(deadline)
                continue
            if requires_pico and not pico_reengage_blocked:
                try:
                    inputs.require_pico()
                except TimeoutError as error:
                    if not handle_pico_loss(error, q, root):
                        raise
                    deadline = advance(deadline)
                    continue
            if manual and tracking and active_planner(runtime) is None:
                # Recheck source age immediately before publishing. On loss,
                # handle_pico_loss returns to stand without auto reconnect.
                age = clock.monotonic()-float(snapshot['received_monotonic'])
                if not 0 <= age <= pico_max_age:
                    error = TimeoutError('Pico reference expired before publication')
                    if not handle_pico_loss(error, q, root):
                        raise error
                    deadline = advance(deadline)
                    continue
            guard.check(controller.status, entered)
            if stop.is_set():
                break
            command = replace(command, input_state_id=int(state.state_id),
                input_state_ts_real_us=int(state.ts_state_real_us),
                ts_client_created_us=clock.time_ns()//1000)
            state_age_before_publish = inputs.ages(state)
            before = clock.monotonic()
            # SDK alone owns session + sequence, shared with end-effector output.
            sent = controller.publish(command, target_names=envelope.target_names,
                                      required_fields=BODY_FIELDS_ALL)
            runtime.policy.last_action = pending_action
            if runtime.head is not None:
                runtime.head.commit(pending_head)
            sent_count += 1
            stale_ticks = 0
            policy_tick += 1
            if not tracking:
                stand_tick += 1
            elif motion_source is not None:
                motion_source.advance()
            report['actions_sent'] = sent_count
            if auto and not report['onnx_control_enabled']:
                print('ONNX_READY: policy commands started; '+('NPZ playback follows stand preparation'
                      if motion_source is not None else 'both Pico triggers toggle following'),flush=True)
            report['onnx_control_enabled'] = True
            write(dict(row, actions_sent=sent_count, control_handles_created=1,
                stage=('motion_tracking' if motion_source is not None else 'pico_tracking') if tracking else 'prepare_stand',
                policy_tick=policy_tick-1,
                reference_time_s=0. if tracking and motion_source is None else reference_time,
                reference_frame=0 if tracking and motion_source is None else reference_frame,
                state_id=int(state.state_id), source_state_us=int(state.ts_state_real_us),
                feedback_timing=dict(at_read=state_age_at_read, before_publish=state_age_before_publish,
                    publication_us=int(state.ts_state_pub_us),
                    received_monotonic_ns=int(state.received_monotonic_ns),
                    pull_context_receive_stamp_ns=int(state.pull_context_receive_stamp_ns)),
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
                stage = ('motion_tracking' if motion_source is not None else 'pico_tracking') if tracking else 'stand'
                print(f'ACTOR_LIVE steps={sent_count} stage={stage}', flush=True)
            complete = clock.monotonic()
            record_period_overrun(write,'complete_cycle',complete-cycle,sent_count)
            deadline = advance(deadline)
        report['stopped_by_signal'] = stop.is_set()
        report['elapsed_s'] = clock.monotonic()-start
        report['stop_reason'] = 'operator_stop' if stop.is_set() else 'motion_complete' if motion_complete else 'duration_elapsed'
        if motion_source is not None:
            report['motion'] = motion_source.metadata
            report['motion_ticks'] = motion_source.tick
        write(dict(record_type='stop_requested',reason=report['stop_reason'],
            elapsed_s=report['elapsed_s'],duration_limit_s=args.seconds,actions_sent=sent_count))
        print(f"ACTOR_STOP: {report['stop_reason']}, elapsed={report['elapsed_s']:.2f}s; CANCEL -> PASSIVE",flush=True)
        return report
    finally:
        if grippers is not None:
            # Opens and releases EE clients while EXTERNAL is still owned.
            try:
                grippers.close()
                write_gripper_events()
            except BaseException as error:
                print(f'PICO_GRIPPER close: {error}', file=sys.stderr, flush=True)
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
