"""Stationary-reference three-point encoder -> shared decoder -> Body31 DTOs.

All 28 actions are retained. No device transport, robot controller, GMR or
configuration update is performed. Inputs are already calibrated Vita site
poses in reference-base coordinates; raw Pico poses are not interchangeable.
"""
from dataclasses import dataclass, replace
import functools
import json
from pathlib import Path
import platform
import time

import numpy as np

from .actor_groot_ankle import contract_from_metadata, policy_class as motor_policy_class
from .actor_human_encoder import Network
from .actor_smpl import digest

POLICY_KIND = 'groot_vita02a_threepoint_stationary'
POINT_NAMES = ('left_op3_mount', 'right_op3_mount', 'head_capsule_center')
LOWER_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left', 'right')
                    for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
TERMS = [('motion_anchor_ori_b', [6]), ('command_multi_future_lower_body', [240]),
         ('vr_3point_local_target', [9]), ('vr_3point_local_orn_target', [12])]


def finite(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} requires finite {shape}')
    return value.copy()


def quaternions(value, shape, name):
    value = finite(value, shape, name)
    if not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=1e-3, rtol=0):
        raise ValueError(f'{name} requires unit WXYZ quaternions')
    return value  # No sign changes: quaternions are numerical network features.


@dataclass(frozen=True)
class ThreePointTarget:
    positions_base_m: np.ndarray
    quaternions_base_wxyz: np.ndarray
    sample_time_s: float
    coordinate_space: str = 'reference_base_link'
    point_names: tuple = POINT_NAMES

    def __post_init__(self):
        if self.coordinate_space != 'reference_base_link' or tuple(self.point_names) != POINT_NAMES:
            raise ValueError('Need calibrated reference-base Vita site poses in left/right/head order')
        if not np.isfinite(self.sample_time_s) or self.sample_time_s < 0:
            raise ValueError('Invalid three-point sample time')
        object.__setattr__(self, 'positions_base_m', finite(self.positions_base_m, (3, 3), 'positions_base_m'))
        object.__setattr__(self, 'quaternions_base_wxyz', quaternions(self.quaternions_base_wxyz, (3, 4), 'site orientations'))


class ThreePointReference:
    """50Hz calibrated site-pose file; no future device trajectory is required."""
    def __init__(self, times, positions, rotations, roots):
        self.times = np.asarray(times, dtype=float)
        n = len(self.times)
        if (self.times.shape != (n,) or not 1 <= n <= 180000 or not np.isfinite(self.times).all()
                or abs(self.times[0]) > 1e-8 or not np.allclose(np.diff(self.times), .02, atol=1e-7, rtol=0)):
            raise ValueError('Three-point reference requires a zero-origin uniform 50Hz time grid')
        self.positions = finite(positions, (n, 3, 3), 'site positions')
        self.rotations = quaternions(rotations, (n, 3, 4), 'site orientations')
        self.roots = quaternions(roots, (n, 4), 'reference roots')
        if not np.allclose(abs(self.roots@self.roots[0]), 1., atol=1e-5, rtol=0):
            raise ValueError('Stationary three-point reference cannot turn its root')

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as z:
            required = {'time_s', 'positions_base_m', 'quaternions_base_wxyz',
                        'reference_root_quat_world_wxyz', 'fps', 'point_names', 'coordinate_space'}
            if not required <= set(z.files):
                raise ValueError('Missing three-point file fields: '+', '.join(sorted(required-set(z.files))))
            if z['fps'].shape != () or float(z['fps']) != 50:
                raise ValueError('Three-point reference must declare 50Hz')
            if (z['point_names'].dtype.kind != 'U' or tuple(z['point_names']) != POINT_NAMES
                    or z['coordinate_space'].shape != () or str(z['coordinate_space'].item()) != 'reference_base_link'):
                raise ValueError('Three-point file needs explicit calibrated coordinate space and point order')
            return cls(z['time_s'], z['positions_base_m'], z['quaternions_base_wxyz'], z['reference_root_quat_world_wxyz'])

    def target(self, ref_time):
        if not np.isfinite(ref_time) or ref_time < 0:
            raise ValueError('Invalid three-point reference time')
        i = min(int(round(ref_time/.02)), len(self.times)-1)
        return ThreePointTarget(self.positions[i], self.rotations[i], float(self.times[i]))


def parse_contract(math_api, sidecar, source):
    if (sidecar.get('schema') != 'sonic_threepoint_contract_v1' or sidecar.get('policy_kind') != POLICY_KIND
            or sidecar.get('point_names') != list(POINT_NAMES) or sidecar.get('lower_joint_names') != list(LOWER_NAMES)
            or sidecar.get('input_dim') != 1167 or sidecar.get('encoder_input_dim') != 267
            or sidecar.get('action_dim') != 28 or sidecar.get('root_quaternion_order') != 'wxyz'
            or sidecar.get('site_coordinate_space') != 'reference_base_link'
            or [(t['name'], t['shape']) for t in sidecar.get('tokenizer_observations', [])] != TERMS):
        raise ValueError('Unsupported stationary three-point contract/layout')
    raw = sidecar['physical_contract']
    # SDK validates the original physical parser envelope. The new modality is
    # explicit; only observation-layout fields differ in the source contract.
    differences = {k for k in raw if source.get(k) != raw[k]}
    if (differences != {'encoder_name', 'input_dim', 'tokenizer_observations'}
            or source.get('encoder_name') != 'teleop' or source.get('input_dim') != 1167
            or source.get('tokenizer_observations') != sidecar['tokenizer_observations']):
        raise ValueError('Three-point source actor physical contract disagrees')
    base = contract_from_metadata(math_api, {'groot_wbc.tracking_contract.v1': json.dumps(raw)})
    if (base.input_dim != 1520 or base.period != .02 or base.future_frames != 10
            or base.future_dt != .02 or base.history_length != 10 or base.include_z or base.ankle is None):
        raise ValueError('Unsupported three-point physical timing/ankle contract')
    return replace(base, input_dim=1167, policy_kind=POLICY_KIND)


@functools.cache
def policy_class(math_api):
    class ThreePointMotorPolicy(motor_policy_class(math_api)):
        def observe_target(self, target, lower, lower_vel, reference_root, q, dq, quat, gyro):
            if not isinstance(target, ThreePointTarget):
                raise ValueError('Need calibrated three-point target')
            c, m = self.contract, math_api
            q, dq, gyro = m.finite(q, 'q'), m.finite(dq, 'dq'), m.finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('Three-point feedback dimensions disagree')
            quat = m.quaternion(quat); reference_root = m.quaternion(reference_root)
            root6 = (m.rotation(quat).T@m.rotation(reference_root))[:, :2].ravel()
            parts = [root6, lower.ravel(), lower_vel.ravel(), target.positions_base_m.ravel(), target.quaternions_base_wxyz.ravel()]
            terms = tuple(np.asarray(v, np.float32) for v in (
                gyro, c.ankle.to_policy(q)-c.policy_default, c.ankle.velocity_to_policy(q, dq),
                self.last_action.copy(), m.rotation(quat).T@[0., 0., -1.]))
            if not self.history: self.history.extend([terms]*c.history_length)
            else: self.history.append(terms)
            parts.extend(np.concatenate([h[k] for h in self.history]) for k in range(5))
            observation = np.concatenate(parts).astype(np.float32)
            if observation.shape != (1167,) or not np.isfinite(observation).all():
                raise ValueError('Invalid three-point observation')
            return observation
    return ThreePointMotorPolicy


def artifact(root, item):
    path = (root/item['path']).resolve()
    if (not path.is_relative_to(root) or not path.is_file() or digest(path) != item['sha256']
            or ('bytes' in item and path.stat().st_size != item['bytes'])):
        raise ValueError('Three-point artifact missing or hash/size mismatch: '+item['path'])
    return path


class ThreePointActorRuntime:
    """Python inference, stationary leg reference and unchanged full-body decode."""
    def __init__(self, root, *, backend='auto', threads=1):
        from .actor_pico_runtime import sdk_math
        self.root = Path(root).resolve()
        self.config = json.loads((self.root/'config/actor_threepoint.json').read_text())
        if (self.config.get('schema') != 'vita_actor_threepoint_v1'
                or self.config.get('policy_kind') != POLICY_KIND
                or set(self.config.get('artifacts', {})) != {'model', 'encoder', 'decoder', 'contract', 'source_contract', 'standing_reference'}):
            raise ValueError('Unsupported three-point actor configuration')
        paths = {k: artifact(self.root, v) for k, v in self.config['artifacts'].items()}
        self.sidecar = json.loads(paths['contract'].read_text())
        if self.sidecar.get('model_sha256') != digest(paths['model']):
            raise ValueError('Three-point actor provenance mismatch')
        self.math = sdk_math(self.root, self.config)
        self.contract = parse_contract(self.math, self.sidecar, json.loads(paths['source_contract'].read_text()))
        if backend == 'auto': backend = 'sdk' if platform.machine() == 'aarch64' else 'ort'
        self.backend = backend
        self.encoder = Network(paths['encoder'], self.math, backend, threads,
            input_name='teleop_features', input_dim=267, output_name='encoded_tokens', output_dim=64)
        self.decoder = Network(paths['decoder'], self.math, backend, threads,
            input_name='obs_dict', input_dim=964, output_name='action', output_dim=28)
        if (type(self.config.get('checkpoint_iteration')) is not int or self.config['checkpoint_iteration']<=0
                or self.encoder.metadata.get('checkpoint_iteration') != str(self.config['checkpoint_iteration'])
                or self.encoder.metadata.get('source_onnx_sha256') != self.sidecar['model_sha256']):
            raise ValueError('Three-point encoder source checkpoint differs')
        with np.load(paths['standing_reference'], allow_pickle=False) as z:
            self.lower = finite(z['lower_positions_policy'], (10, 12), 'standing leg reference')
            self.lower_vel = finite(z['lower_velocities_policy'], (10, 12), 'standing leg velocities')
            self.reference_root = quaternions(z['reference_root_quat_world_wxyz'], (4,), 'standing root')
        if not np.array_equal(self.lower, np.repeat(self.lower[:1], 10, axis=0)) or np.any(self.lower_vel):
            raise ValueError('Stationary mode requires fixed leg references and zero reference velocities')
        self.policy = policy_class(self.math)(self.contract, self.config['body_profile'])
        self.last_tokens = None; self.last_timing = None

    def reset(self):
        self.policy.reset(); self.last_tokens = None; self.last_timing = None

    def infer(self, target, *, q, dq, root, gyro):
        obs = self.policy.observe_target(target, self.lower, self.lower_vel, self.reference_root, q, dq, root, gyro)
        start = time.perf_counter()
        tokens = self.encoder.run(obs[:267][None]); middle = time.perf_counter()
        raw_action = self.decoder.run(np.concatenate((tokens, obs[267:][None]), axis=1))[0]
        end = time.perf_counter(); self.last_tokens = tokens[0].copy()
        self.last_timing = dict(encoder_ms=(middle-start)*1000, decoder_ms=(end-middle)*1000)
        command, decoded = self.policy.decode(raw_action)  # Retain every leg channel.
        return command, obs, dict(policy=POLICY_KIND, backend=self.backend, raw_action=raw_action.tolist(),
            decoded=decoded, sample_time_s=float(target.sample_time_s),
            all_28_actions_retained=True, actions_sent=0, control_handles_created=0)


def require_live_ready():
    raise RuntimeError('Three-point live unavailable: calibrated Pico time/pose input and hardware startup/exit '
                       'have not been validated. No controller is created; protections remain unchanged.')
