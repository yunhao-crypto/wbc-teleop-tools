"""MimicLite OP3 observations and raw affine motor targets on the Body31 wire.

The host importer preserves the original trained ONNX graph/normalizers and
embeds the validated motor_deployment YAML in a single-input wrapper. Runtime
has no dependency on the training repository, PyTorch or an additional ORT.
"""
from dataclasses import dataclass
import copy
import functools
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .actor_groot_ankle import GrootMotorThetaContract, serial_policy_class
from .actor_parallel_ankle import MotorThetaAnkle, decode_motor_theta, command_space
from .policy_contract import VITA_ACTION_JOINT_NAMES

CONTRACT_KEY = 'mimiclite.tracking_contract.v1'
POLICY_KIND = 'mimiclite_vita02a_tracking'
DEPLOYMENT_SCHEMA = 'mimiclite.vita02a.motor_theta.v1'
MODEL_SCHEMA = 'mimiclite.vita02a.native.v2'
MODEL_SHA256 = '2f015a57135ea0fd76e2f76f60633a3d55e835c6978432fb3c43b2de4fa78a75'
ACTUATOR_SCHEMA = 'mimiclite.vita02a.achilles_friction.v2'
GEOMETRY = dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235)
MOTOR_LIMITS = (-.95975, .61075)
ALIASES = {f'{s}_{b}_achilles_joint': f'{s}_ankle_{a}_joint'
           for s in ('left', 'right') for a, b in (('pitch', 'A'), ('roll', 'B'))}
COMMAND_KINDS = {'ref_body_pos_future_local': 3, 'ref_root_ori_future_b': 6,
                 'ref_joint_pos_future': 28}
HISTORY_KINDS = {'root_ang_vel_history': 3, 'projected_gravity_history': 3,
                 'joint_pos_history': 28, 'joint_vel_history': 28}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError('Invalid MimicLite '+name)
    return result


def _indices(value, maximum, name):
    if (not isinstance(value, list) or not value or len(value) > 64
            or any(type(v) is not int or not 0 <= v <= maximum for v in value)
            or len(set(value)) != len(value)):
        raise ValueError('Invalid MimicLite '+name)
    return value


def validate_deployment(block):
    """Only the reviewed OP3 student/ordinary PPO deploy ABI is accepted."""
    d = copy.deepcopy(block)
    if (not isinstance(d, dict) or d.get('schema') != DEPLOYMENT_SCHEMA
            or d.get('model_schema') != MODEL_SCHEMA or d.get('model_sha256') != MODEL_SHA256
            or d.get('actuator_schema') != ACTUATOR_SCHEMA
            or d.get('control_space') != 'motor_theta'
            or d.get('reference_space') != 'motor_theta'
            or d.get('reference_root_body') != 'base_link' or d.get('quaternion_order') != 'wxyz'
            or type(d.get('control_period_s')) not in (int, float)
            or not math.isclose(d['control_period_s'], .02, rel_tol=0, abs_tol=1e-9)
            or type(d.get('physics_period_s')) not in (int, float)
            or not math.isclose(d['physics_period_s'], .005, rel_tol=0, abs_tol=1e-9)
            or type(d.get('deployment_alpha')) not in (int, float) or d['deployment_alpha'] != 1.0):
        raise ValueError('Unsupported MimicLite OP3 deployment/model/actuator contract')
    names = d.get('joint_names', [])
    if (not isinstance(names, list) or len(names) != 28 or not all(isinstance(n, str) for n in names)
            or len(set(names)) != 28
            or set(ALIASES)-set(names)
            or {ALIASES.get(n, n) for n in names} != set(VITA_ACTION_JOINT_NAMES[:28])):
        raise ValueError('MimicLite must name exactly the 28 OP3 motor-space joints')
    ref = d.get('reference_joint_names', [])
    if (not isinstance(ref, list) or len(ref) != 28 or not all(isinstance(n, str) for n in ref)
            or set(ref) != set(names)):
        raise ValueError('Invalid MimicLite reference joint order')
    default = _vector(d.get('default_position'), 28, 'default position')
    scale = _vector(d.get('action_scale'), 28, 'action scale')
    if np.any(scale <= 0):
        raise ValueError('MimicLite action scales must be positive')
    joints = d.get('joints', {})
    if not isinstance(joints, dict) or set(joints) != set(names):
        raise ValueError('MimicLite gains must resolve every commanded joint')
    for name in names:
        p = joints[name]
        if not isinstance(p, dict):
            raise ValueError('Invalid MimicLite actuator parameters: '+name)
        values = [p.get(k) for k in ('kp', 'kd', 'effort', 'armature', 'frictionloss', 'viscous_damping')]
        if (any(type(v) not in (int, float) for v in values)
                or not np.isfinite(values).all() or min(values) < 0 or p['kp'] <= 0 or p['effort'] <= 0):
            raise ValueError('Invalid MimicLite actuator parameters: '+name)
        if name in ALIASES:
            if (p.get('kind') != 'motor' or type(p.get('dynamic_friction')) not in (int, float)
                    or not 0 <= p['dynamic_friction'] <= p['frictionloss']
                    or p.get('friction_transition_speed') != .01
                    or not math.isclose(p['effort'], 50., rel_tol=0, abs_tol=1e-9)):
                raise ValueError('Invalid MimicLite Achilles model: '+name)
        elif p.get('kind') != 'position':
            raise ValueError('MimicLite serial axes require position actuators')
    future = d.get('future_steps', [])
    if (not isinstance(future, list) or not future or len(future) > 64 or 0 not in future
            or any(type(v) is not int or not -64 <= v <= 64 for v in future)
            or future != sorted(set(future))):
        raise ValueError('Invalid MimicLite future offsets')
    observations = d.get('observations', {})
    if not isinstance(observations, dict) or set(observations) != {'command', 'policy'}:
        raise ValueError('MimicLite deploy must have command/policy inputs, not privileged teacher inputs')
    widths = {}
    for group, terms in observations.items():
        expected = set(COMMAND_KINDS) if group == 'command' else set(HISTORY_KINDS) | {'prev_actions'}
        if (not isinstance(terms, list) or len(terms) != len(expected)
                or any(not isinstance(t, dict) or not isinstance(t.get('kind'), str) for t in terms)
                or {t['kind'] for t in terms} != expected):
            raise ValueError('Unsupported MimicLite observation terms: '+group)
        width = 0
        for term in terms:
            kind = term['kind']
            if kind in COMMAND_KINDS:
                ids = _indices(term.get('future_indices'), len(future)-1, 'reference indices')
                size = len(ids)*COMMAND_KINDS[kind]
            elif kind in HISTORY_KINDS:
                offsets = _indices(term.get('history_steps'), 128, 'history offsets')
                if offsets != sorted(offsets) or offsets[0] != 0:
                    raise ValueError('MimicLite histories must start with the current frame')
                size = len(offsets)*HISTORY_KINDS[kind]
                if kind.startswith('joint_') and (not isinstance(term.get('joint_names'), list)
                        or len(term['joint_names']) != 28
                        or not all(isinstance(n, str) for n in term['joint_names'])
                        or set(term['joint_names']) != set(names)):
                    raise ValueError('Invalid MimicLite history joint order')
            else:
                steps = term.get('steps')
                if type(steps) is not int or not 1 <= steps <= 16:
                    raise ValueError('Invalid MimicLite previous-action history')
                size = 28*steps
            if type(term.get('size')) is not int or term['size'] != size:
                raise ValueError('MimicLite observation size disagrees with its semantics')
            width += size
        widths[group] = width
    return d, widths


def metadata_contract(deployment, source_onnx_sha256, source_yaml_sha256):
    d, widths = validate_deployment(deployment)
    if not _sha(source_onnx_sha256) or not _sha(source_yaml_sha256):
        raise ValueError('MimicLite source hashes must be SHA256')
    order = sorted(widths)
    return dict(schema_version=1, policy_kind=POLICY_KIND, action_dim=28,
        input_dim=sum(widths.values()), input_groups=order,
        input_widths=[widths[k] for k in order],
        input_layout='concatenate_groups_then_original_graph',
        action_semantics='default_plus_scale_times_raw_action_no_clipping',
        source_onnx_sha256=source_onnx_sha256, source_yaml_sha256=source_yaml_sha256,
        motor_deployment=d)


@dataclass(frozen=True)
class MimicLiteContract(GrootMotorThetaContract):
    deployment: dict = None
    motor_names: tuple = ()
    input_groups: tuple = ()
    input_widths: tuple = ()
    nominal_root_height: float = .891088
    past_frames: int = 0

    @classmethod
    def from_metadata(cls, math_api, metadata):
        if CONTRACT_KEY not in metadata:
            raise ValueError('model is missing '+CONTRACT_KEY+'; import the MimicLite ONNX + YAML first')
        raw = json.loads(metadata[CONTRACT_KEY])
        if not isinstance(raw, dict) or type(raw.get('schema_version')) is not int or raw['schema_version'] != 1:
            raise ValueError('Unsupported MimicLite wrapper version')
        expected = metadata_contract(raw.get('motor_deployment'), raw.get('source_onnx_sha256'), raw.get('source_yaml_sha256'))
        if raw != expected:
            raise ValueError('MimicLite wrapper dimensions or semantics disagree')
        d = expected['motor_deployment']
        names = tuple(ALIASES.get(n, n) for n in d['joint_names'])
        block = dict(schema_version=3, control_space='motor_theta',
            reference_conversion='virtual_pitch_roll_to_motor_theta', geometry=GEOMETRY,
            motor_limits=MOTOR_LIMITS, roll_sign=dict(left=1, right=-1),
            slots={f'{s}_ankle_{a}_joint': dict(side=s, motor=motor, joint=f'{s}_{b}_achilles_joint')
                for s in ('left', 'right') for a, b, motor in
                (('pitch', 'A', 'theta_top'), ('roll', 'B', 'theta_bottom'))})
        ankle = MotorThetaAnkle(names, block)
        default = np.asarray(d['default_position'], dtype=float)
        if np.any(default[ankle.slots] < MOTOR_LIMITS[0]) or np.any(default[ankle.slots] > MOTOR_LIMITS[1]):
            raise ValueError('MimicLite nominal motors exceed travel')
        virtual, residual = ankle.to_virtual(default)
        if residual > 1e-8:
            raise ValueError('MimicLite nominal ankle pose does not close')
        # Hardware/profile intersections and measured-state checks remain in the
        # SDK/WTT safety layer; these are not bounds on affine PD targets.
        lower, upper = np.full(28, -math.pi), np.full(28, math.pi)
        policy_lower, policy_upper = lower.copy(), upper.copy()
        policy_lower[ankle.slots], policy_upper[ankle.slots] = MOTOR_LIMITS
        max_history = max(max(t.get('history_steps', [0])) for t in d['observations']['policy'])+1
        future = d['future_steps']
        return cls(names=names, default=virtual, scale=np.asarray(d['action_scale'], dtype=float),
            kp=np.array([d['joints'][n]['kp'] for n in d['joint_names']]),
            kd=np.array([d['joints'][n]['kd'] for n in d['joint_names']]),
            lower=lower, upper=upper, action_clip=None, period=.02,
            future_frames=max(future), future_dt=.02, history_length=max_history,
            input_dim=expected['input_dim'], include_z=True, policy_default=default,
            policy_lower=policy_lower, policy_upper=policy_upper, ankle=ankle,
            policy_kind=POLICY_KIND, deployment=d, motor_names=tuple(d['joint_names']),
            input_groups=tuple(expected['input_groups']), input_widths=tuple(expected['input_widths']),
            past_frames=-min(future))


def is_mimiclite(contract):
    return isinstance(contract, MimicLiteContract)


def checked_reference(ankle, q):
    """Reject infeasible references; do not silently clip or project them."""
    motor = ankle.to_policy(q)
    if (not np.isfinite(motor).all() or np.any(motor[:, ankle.slots] < MOTOR_LIMITS[0]-1e-7)
            or np.any(motor[:, ankle.slots] > MOTOR_LIMITS[1]+1e-7)):
        raise ValueError('MimicLite reference exceeds coupled motor travel')
    # The shared analytic IK clamps its square root/asin. Independently check
    # rod closure so an unreachable virtual pose cannot masquerade as valid.
    from .actor_parallel_ankle import _foot, _links
    for row, mapped in zip(q, motor):
        for p, r, sign in ankle.legs:
            for (y, height), theta in zip(_links(GEOMETRY), mapped[[p, r]]):
                anchor = np.asarray(_foot(row[p], sign*row[r], -GEOMETRY['lx'], y, 0.))
                crank = np.array([-GEOMETRY['lx']*math.cos(theta), y, height+GEOMETRY['lx']*math.sin(theta)])
                if abs(np.linalg.norm(anchor-crank)-height) > 1e-7:
                    raise ValueError('MimicLite reference linkage does not close')
    return motor


@functools.cache
def policy_class(math_api):
    class MimicLitePolicy(serial_policy_class(math_api)):
        clip_actions = False

        def __init__(self, contract, profile):
            super().__init__(contract, profile)
            if profile.get('ankle_command_space') != 'motor_theta':
                raise ValueError('MimicLite requires native motor_theta command space')
            self.reset()

        def reset(self):
            super().reset()
            self.signal_history = {}
            self.action_steps = next(t['steps'] for t in self.contract.deployment['observations']['policy']
                                     if t['kind'] == 'prev_actions')
            self.previous_actions = np.zeros((self.action_steps, 28))

        def observe(self, motion, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            c, m = self.contract, math_api
            q, dq, gyro = m.finite(q, 'q'), m.finite(dq, 'dq'), m.finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('Invalid MimicLite state dimensions')
            quat = m.quaternion(quat)
            rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
            current = dict(root_ang_vel_history=gyro,
                projected_gravity_history=rotation.inv().apply([0, 0, -1]),
                joint_pos_history=c.ankle.to_policy(q), joint_vel_history=c.ankle.velocity_to_policy(q, dq))
            for kind, value in current.items():
                old = self.signal_history.get(kind)
                self.signal_history[kind] = (np.repeat(value[None], c.history_length, axis=0)
                    if old is None else np.concatenate((value[None], old[:-1])))
            center_frame = int(round(ref_time/c.period))
            ids = np.clip(center_frame+np.asarray(c.deployment['future_steps']), 0, len(motion.q)-1)
            offset = m.yaw_quaternion(yaw_offset)
            reference_rot = Rotation.from_quat(np.array([m.multiply(offset, motion.quat[i]) for i in ids])[:, [1, 2, 3, 0]])
            reference_pos = np.asarray(motion.base[ids]) @ m.rotation(offset).T
            center = c.deployment['future_steps'].index(0)
            matrix = reference_rot[center].as_matrix()
            heading = matrix[:2, 0] if np.linalg.norm(matrix[:2, 0]) > .1 else matrix[:2, 2]*(1 if matrix[2, 0] < 0 else -1)
            yaw = Rotation.from_euler('z', np.arctan2(heading[1], heading[0]))
            origin = reference_pos[center].copy(); origin[2] = 0.
            local = yaw.inv().apply(reference_pos-origin)
            orientation = (rotation.inv()*reference_rot).as_matrix()[:, :2, :]
            motors = checked_reference(c.ankle, motion.q[ids])
            ref_order = [c.motor_names.index(n) for n in c.deployment['reference_joint_names']]
            motors = motors[:, ref_order]
            groups = []
            for group in c.input_groups:
                parts = []
                for term in c.deployment['observations'][group]:
                    kind = term['kind']
                    if kind in current:
                        value = self.signal_history[kind][term['history_steps']]
                        if 'joint_names' in term:
                            value = value[:, [c.motor_names.index(n) for n in term['joint_names']]]
                    elif kind == 'prev_actions':
                        value = self.previous_actions[:term['steps']]
                    elif kind == 'ref_body_pos_future_local':
                        value = local[term['future_indices']]
                    elif kind == 'ref_root_ori_future_b':
                        value = orientation[term['future_indices']]
                    else:
                        value = motors[term['future_indices']]
                    parts.append(value.reshape(-1))
                groups.append(np.concatenate(parts))
            observation = np.concatenate(groups).astype(np.float32)
            if observation.shape != (c.input_dim,) or not np.isfinite(observation).all():
                raise ValueError('Invalid MimicLite observation')
            return observation

        def decode(self, raw_action, **timing):
            if command_space(self) != 'motor_theta':
                raise ValueError('MimicLite requires native motor_theta command space')
            command, row = decode_motor_theta(self, math_api, raw_action, **timing)
            self.previous_actions = np.concatenate((self.last_action.astype(float)[None], self.previous_actions[:-1]))
            return command, row

    return MimicLitePolicy


def create_reference(contract, math_api, config):
    from .actor_supermimic import SuperMimicReference
    from .actor_pico_stream import stream_settings
    if config.get('reference_window') != dict(past_frames=contract.past_frames, tail_frames=0):
        raise ValueError('MimicLite reference window must match the imported model contract')
    settings = stream_settings(config)
    history = 0. if settings is None else settings.playback_margin_s+config['pico_hold_age_s']+.1
    # This class supplies causal interpolation only, not another policy's goal
    # builder. Past/current/future indices come from the reviewed MimicLite YAML.
    return SuperMimicReference(contract, math_api, past_frames=contract.past_frames,
        tail_frames=0, history_s=history)
