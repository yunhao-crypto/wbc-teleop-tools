"""MimicLite v2 model contract and Pico reference producer for native locomotion."""

import copy
from dataclasses import dataclass
import json
import math

import numpy as np

from .actor_parallel_ankle import MotorThetaAnkle
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
OBSERVATION_SCHEMA_V2 = 'mimic_lite.motor_observation.v2'
COMMAND_KINDS = {'ref_body_pos_future_local': 3, 'ref_root_ori_future_b': 6,
                 'ref_joint_pos_future': 28, 'ref_joint_vel_future': 28}

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
    if future != [-8, -4, -2, 0, 1, 2, 3, 4]:
        raise ValueError('Invalid MimicLite future offsets')
    if d.get('observation_schema') != OBSERVATION_SCHEMA_V2:
        raise ValueError('MimicLite native tracking requires the v2 observation schema')
    observations = d.get('observations', {})
    if not isinstance(observations, dict) or set(observations) != {'command', 'policy'}:
        raise ValueError('MimicLite deploy must have command/policy inputs, not privileged teacher inputs')
    widths = {}
    for group, terms in observations.items():
        expected = list(COMMAND_KINDS) if group == 'command' else [*HISTORY_KINDS, 'prev_actions']
        if (not isinstance(terms, list) or len(terms) != len(expected)
                or any(not isinstance(t, dict) or not isinstance(t.get('kind'), str) for t in terms)
                or [t['kind'] for t in terms] != expected):
            raise ValueError('Unsupported MimicLite observation terms: '+group)
        width = 0
        for term in terms:
            kind = term['kind']
            if kind in COMMAND_KINDS:
                ids = _indices(term.get('future_indices'), len(future)-1, 'reference indices')
                if ids != list(range(8)):
                    raise ValueError('MimicLite v2 command terms require the reviewed reference order')
                size = len(ids)*COMMAND_KINDS[kind]
            elif kind in HISTORY_KINDS:
                offsets = _indices(term.get('history_steps'), 128, 'history offsets')
                if offsets != sorted(offsets) or offsets[0] != 0:
                    raise ValueError('MimicLite histories must start with the current frame')
                if offsets != list(range(10)):
                    raise ValueError('MimicLite v2 sensor histories must be [0..9]')
                if kind == 'joint_pos_history' and term.get('relative_to_default') is not True:
                    raise ValueError('MimicLite v2 joint_pos_history must set relative_to_default=true')
                size = len(offsets)*HISTORY_KINDS[kind]
                if kind.startswith('joint_') and (not isinstance(term.get('joint_names'), list)
                        or len(term['joint_names']) != 28
                        or term['joint_names'] != names):
                    raise ValueError('Invalid MimicLite history joint order')
            else:
                steps = term.get('steps')
                if type(steps) is not int or not 1 <= steps <= 16:
                    raise ValueError('Invalid MimicLite previous-action history')
                size = 28*steps
            if (type(term.get('size')) is not int or term['size'] != size
                    or ('width' in term and (type(term['width']) is not int or term['width'] != size))):
                raise ValueError('MimicLite observation size disagrees with its semantics')
            width += size
        widths[group] = width
    if widths != {'command': 520, 'policy': 704}:
        raise ValueError('MimicLite v2 dimensions must resolve to command=520, policy=704, total=1224')
    command_terms = {t['kind']: t for t in observations['command']}
    if command_terms['ref_joint_vel_future'].get('future_indices') != command_terms['ref_joint_pos_future'].get('future_indices'):
        raise ValueError('MimicLite v2 reference velocity indices must equal position indices')
    actions = next(t for t in observations['policy'] if t['kind'] == 'prev_actions')
    if actions.get('steps') != 3:
        raise ValueError('MimicLite v2 prev_actions must use steps=3')
    virtual_names = tuple(ALIASES.get(n, n) for n in names)
    ankle = MotorThetaAnkle(virtual_names, dict(schema_version=3,
        control_space='motor_theta', reference_conversion='virtual_pitch_roll_to_motor_theta',
        geometry=GEOMETRY, motor_limits=MOTOR_LIMITS, roll_sign=dict(left=1, right=-1),
        slots={f'{side}_ankle_{axis}_joint': dict(side=side, motor=motor)
               for side in ('left', 'right')
               for axis, motor in (('pitch', 'theta_top'), ('roll', 'theta_bottom'))}))
    if np.any(default[ankle.slots] < MOTOR_LIMITS[0]) or np.any(default[ankle.slots] > MOTOR_LIMITS[1]):
        raise ValueError('MimicLite nominal motors exceed travel')
    _, residual = ankle.to_virtual(default)
    if residual > 1e-8:
        raise ValueError('MimicLite nominal ankle pose does not close')
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
class MimicLiteReferenceContract:
    names: tuple[str, ...]
    period: float
    past_frames: int
    future_frames: int
    policy_kind: str = POLICY_KIND

    @classmethod
    def read(cls, path):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError('Invalid MimicLite ONNX contract')
        expected = metadata_contract(raw.get('motor_deployment'),
                                     raw.get('source_onnx_sha256'), raw.get('source_yaml_sha256'))
        if raw != expected:
            raise ValueError('MimicLite ONNX contract does not match the reviewed ABI')
        deployment = expected['motor_deployment']
        names = tuple(ALIASES.get(name, name) for name in deployment['joint_names'])
        return cls(names, deployment['control_period_s'],
                   -min(deployment['future_steps']), max(deployment['future_steps']))


def load_reference_config(root, model):
    from .actor_pico_runtime import digest

    config = json.loads((root/'config/mimiclite_native_reference.json').read_text())
    shared = json.loads((root/'config/actor_pico.json').read_text())
    for name, item in shared['artifacts'].items():
        if name.startswith('gmr_source_'):
            path = (root/item['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or digest(path) != item['sha256']:
                raise ValueError('Unreviewed Pico/GMR dependency: '+item['path'])
    config['sdk_sources'] = {name: shared['sdk_sources'][name]
                             for name in ('wbc_tracking_policy.py', '_policy_state.py')}
    config['artifacts'] = {'model': {'source_name': model.name, 'sha256': digest(model)}}
    return config


def create_reference(contract, math_api, config):
    from .actor_pico_stream import stream_settings
    from .actor_supermimic import SuperMimicReference

    settings = stream_settings(config)
    history = settings.playback_margin_s + config['pico_hold_age_s'] + .1
    return SuperMimicReference(contract, math_api, past_frames=contract.past_frames,
                               tail_frames=0, history_s=history)


def reference_anchor(state, math_api):
    """Validate fresh public HIP feedback before starting the Pico retargeter."""
    if len(state.q) != 31 or not state.attitude_valid:
        raise ValueError('Expected valid 31-slot public body feedback')
    indices = np.flatnonzero(state.imu_location == math_api.HIP)
    if len(indices) != 1:
        raise ValueError('Expected one HIP IMU for Pico reference alignment')
    index = indices[0]
    if state.imu_status[index] != math_api.SAMPLE_OK or not state.imu_state_present[index]:
        raise ValueError('HIP IMU is unavailable for Pico reference alignment')
    quat = math_api.quaternion(state.imu_values[index, :4])
    gravity_z = (math_api.rotation(quat).T @ [0., 0., -1.])[2]
    if math.acos(float(np.clip(-gravity_z, -1., 1.))) > .8:
        raise ValueError('Body tilt exceeds the Pico reference alignment limit')
    return quat
