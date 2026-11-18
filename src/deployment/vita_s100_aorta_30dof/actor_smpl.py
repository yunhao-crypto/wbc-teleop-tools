"""SONIC SMPL human-reference input with the existing SDK/Body31 motor decoder.

No transport, GMR, controller or robot configuration is constructed here. The
metadata-free step27000 ONNX requires an explicit, hash-bound sidecar. It is
never interpreted as a G1 reference model and never falls back to another model.
"""
from dataclasses import replace
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from .actor_groot_ankle import contract_from_metadata, policy_class as motor_policy_class

POLICY_KIND = 'groot_vita02a_smpl_tracking'
SIDECAR_SCHEMA = 'sonic_smpl_contract_v1'
JOINT_NAMES = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
               'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
               'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand')
WRIST_NAMES = ('left_wrist_roll_joint', 'right_wrist_roll_joint',
               'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
               'left_wrist_pitch_joint', 'right_wrist_pitch_joint')
TERMS = [('smpl_joints_multi_future_local_nonflat', [10, 72]),
         ('smpl_root_ori_b_multi_future', [10, 6]),
         ('joint_pos_multi_future_wrist_for_smpl', [10, 6])]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest_stream(stream)


def _digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4*1024*1024), b''):
        h.update(block)
    return h.hexdigest()


def validate_model_binding(path, sidecar):
    if not isinstance(sidecar, dict) or sidecar.get('schema') != SIDECAR_SCHEMA:
        raise ValueError('SMPL requires a reviewed, hash-bound sidecar')
    expected = sidecar.get('model', {})
    if (expected.get('bytes') != Path(path).stat().st_size or expected.get('sha256') != digest(path)):
        raise ValueError('SMPL model size/SHA256 differs from the sidecar')


def parse_contract(math_api, sidecar, metadata):
    if sidecar.get('schema') != SIDECAR_SCHEMA or sidecar.get('policy_kind') != POLICY_KIND:
        raise ValueError('Unsupported SMPL sidecar')
    if (sidecar.get('encoder_name') != 'smpl' or sidecar.get('decoder_name') != 'g1_dyn'
            or sidecar.get('input_dim') != 1740 or sidecar.get('action_dim') != 28
            or [(t['name'], t['shape']) for t in sidecar.get('tokenizer_observations', [])] != TERMS
            or sidecar.get('human_joint_names') != list(JOINT_NAMES)
            or sidecar.get('wrist_joint_names') != list(WRIST_NAMES)
            or sidecar.get('human_joint_space') != 'pelvis_relative_world_z_up'
            or sidecar.get('root_quaternion_order') != 'wxyz'):
        raise ValueError('Unsupported SMPL observation layout or coordinates')
    # Do not silently override a conflicting embedded contract. This export has none.
    if metadata:
        raise ValueError('This SMPL binding expects a metadata-free ONNX export')
    raw = sidecar['physical_contract']
    base = contract_from_metadata(math_api, {'groot_wbc.tracking_contract.v1': json.dumps(raw)})
    if (getattr(base, 'ankle', None) is None or base.input_dim != 1520 or base.include_z
            or base.future_frames != 10 or base.future_dt != .02 or base.history_length != 10):
        raise ValueError('SMPL export needs the matching 28-DOF motor-theta 50Hz physical contract')
    return replace(base, input_dim=1740, policy_kind=POLICY_KIND)


def rotvec_quaternion(vectors):
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('Expected finite root axis-angle [N,3]')
    theta = np.linalg.norm(v, axis=1)
    factor = .5*np.sinc(theta/(2*np.pi))
    return np.column_stack((np.cos(theta/2), v*factor[:, None]))


def root_from_pose_aa(math_api, pose_aa, *, convention):
    """Explicit source convention, with no automatic Y-up/Z-up guessing.

    The supplied v4.4 release is already in world Z-up. The training map removes
    the SMPL base on the right: q_root = exp(pose_aa[:3]) * conj([.5]*4).
    A different source must first be converted and separately checked.
    """
    pose = np.asarray(pose_aa, dtype=float)
    if convention != 'z_up_smpl_base' or pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError('Need [N,72] pose_aa and explicit z_up_smpl_base convention')
    return np.array([math_api.multiply(q, [.5, -.5, -.5, -.5])
                     for q in rotvec_quaternion(pose[:, :3])])


class SmplReference:
    """50Hz human root-relative joints plus six explicit robot wrist references.

    pose_aa alone cannot supply the required SMPL joint positions without a body
    model. Missing wrists are an error, never a silent zero/GMR fallback.
    """
    def __init__(self, math_api, joints, quaternions, wrists):
        self.math = math_api
        self.joints = np.asarray(joints, dtype=float)
        self.quat = np.asarray(quaternions, dtype=float)
        self.wrists = np.asarray(wrists, dtype=float)
        n = len(self.joints)
        if (not 1 <= n <= 180000 or self.joints.shape != (n, 24, 3)
                or self.quat.shape != (n, 4) or self.wrists.shape != (n, 6)
                or not all(np.isfinite(x).all() for x in (self.joints, self.quat, self.wrists))):
            raise ValueError('Expected finite SMPL [N,24,3], root [N,4], wrist [N,6]')
        if np.max(np.abs(self.joints[:, 0])) > 1e-5:
            raise ValueError('SMPL joints must be pelvis-relative, in metres')
        norms = np.linalg.norm(self.quat, axis=1)
        if np.any(abs(norms-1) > .02):
            raise ValueError('SMPL root needs unit wxyz quaternions')
        self.quat = self.quat/norms[:, None]
        self.local = np.array([self.joints[i] @ math_api.rotation(q) for i, q in enumerate(self.quat)])

    @classmethod
    def from_npz(cls, math_api, path):
        with np.load(path, allow_pickle=False) as z:
            required = {'smpl_joints', 'root_quat_wxyz', 'wrist_joint_pos', 'fps',
                        'smpl_joint_names', 'wrist_joint_names', 'joint_space'}
            if not required <= set(z.files):
                raise ValueError('Missing human-reference fields: '+', '.join(sorted(required-set(z.files))))
            if (z['fps'].shape != () or float(z['fps']) != 50.
                    or z['joint_space'].shape != () or str(z['joint_space'].item()) != 'pelvis_relative_world_z_up'):
                raise ValueError('SMPL input requires 50Hz and explicit pelvis_relative_world_z_up')
            if z['smpl_joint_names'].dtype.kind not in 'US' or z['wrist_joint_names'].dtype.kind not in 'US':
                raise ValueError('SMPL/wrist joint names must be strings')
            decode = lambda a: [v.decode() if isinstance(v, bytes) else str(v) for v in a]
            if decode(z['smpl_joint_names']) != list(JOINT_NAMES) or decode(z['wrist_joint_names']) != list(WRIST_NAMES):
                raise ValueError('SMPL or wrist joint order differs from the route contract')
            return cls(math_api, z['smpl_joints'], z['root_quat_wxyz'], z['wrist_joint_pos'])

    def indices(self, ref_time, count=10, dt=.02):
        if not np.isfinite(ref_time) or ref_time < 0 or count != 10 or dt != .02:
            raise ValueError('Invalid SMPL reference time/window')
        return np.minimum(int(round(ref_time/.02))+np.arange(count), len(self.joints)-1)


@functools.cache
def policy_class(math_api):
    class SmplMotorThetaPolicy(motor_policy_class(math_api)):
        def observe(self, reference, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            if not isinstance(reference, SmplReference):
                raise ValueError('SMPL policy requires human references, not G1 joint motion')
            c, m = self.contract, math_api
            q, dq, gyro = m.finite(q, 'q'), m.finite(dq, 'dq'), m.finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('SMPL observation feedback dimensions disagree')
            quat = m.quaternion(quat)
            ix = reference.indices(ref_time, c.future_frames, c.future_dt)
            offset = m.yaw_quaternion(yaw_offset)
            ori = [m.rotation(m.multiply(m.conjugate(quat), m.multiply(offset, reference.quat[i])))[:, :2].ravel() for i in ix]
            parts = [reference.local[ix].ravel(), np.asarray(ori).ravel(), reference.wrists[ix].ravel()]
            terms = tuple(np.asarray(v, np.float32) for v in (
                gyro, c.ankle.to_policy(q)-c.policy_default, c.ankle.velocity_to_policy(q, dq),
                self.last_action.copy(), m.rotation(quat).T@[0., 0., -1.]))
            if not self.history:
                self.history.extend([terms]*c.history_length)
            else:
                self.history.append(terms)
            parts.extend(np.concatenate([h[k] for h in self.history]) for k in range(5))
            obs = np.concatenate(parts).astype(np.float32)
            if obs.shape != (1740,) or not np.isfinite(obs).all():
                raise ValueError('Invalid SMPL observation')
            return obs
    return SmplMotorThetaPolicy


def _artifact(root, item):
    path = (root/item['path']).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or digest(path) != item['sha256']
            or ('bytes' in item and path.stat().st_size != item['bytes'])):
        raise ValueError('SMPL artifact missing or hash/size mismatch: '+item['path'])
    return path


class SmplActorRuntime:
    """Python-only orchestration, supporting ORT offline and sealed SDK inference.

    This is deliberately independent of Pico/GMR and its standing/startup state
    machine. Hardware live entry stays unavailable until SMPL startup and exit
    have separate validation; importing this class creates inert DTOs only.
    """
    def __init__(self, root, *, backend='auto', threads=1):
        from .actor_pico_runtime import ActorModel, sdk_math
        self.root = Path(root).resolve()
        self.config = json.loads((self.root/'config/actor_smpl.json').read_text())
        pipeline = self.config.get('inference_pipeline', 'end_to_end')
        expected_artifacts = ({'model', 'smpl_contract'} if pipeline == 'end_to_end' else
                              {'model', 'smpl_contract', 'human_encoder', 'action_decoder'} if pipeline == 'human_encoder_decoder' else None)
        if (self.config.get('schema') != 'vita_actor_smpl_v1'
                or self.config.get('policy_kind') != POLICY_KIND
                or expected_artifacts is None or set(self.config.get('artifacts', {})) != expected_artifacts):
            raise ValueError('Unsupported SMPL actor configuration')
        paths = {name: _artifact(self.root, item) for name, item in self.config['artifacts'].items()}
        self.sidecar = json.loads(paths['smpl_contract'].read_text())
        validate_model_binding(paths['model'], self.sidecar)
        self.math = sdk_math(self.root, self.config)
        if pipeline == 'human_encoder_decoder':
            from .actor_human_encoder import HumanEncoderDecoder
            self.model = HumanEncoderDecoder(paths['human_encoder'], paths['action_decoder'], self.math, self.sidecar,
                                            backend=backend, threads=threads)
        else:
            self.model = ActorModel(paths['model'], self.math, backend=backend, threads=threads,
                                    policy_kind=POLICY_KIND, smpl_contract=self.sidecar)
        self.policy = policy_class(self.math)(self.model.contract, self.config['body_profile'])
        self.policy_name = 'sonic-smpl'

    def reset(self):
        self.policy.reset()

    def infer(self, reference, ref_time, *, q, dq, root, gyro, yaw_offset=0.):
        observation = self.policy.observe(reference, ref_time, q, dq, root, gyro, yaw_offset)
        action = self.model.infer(observation)
        command, decoded = self.policy.decode(action)
        return command, observation, dict(policy=self.policy_name, backend=self.model.backend,
            inference_pipeline=self.config.get('inference_pipeline', 'end_to_end'),
            encoded_tokens=(self.model.last_tokens.tolist() if getattr(self.model, 'last_tokens', None) is not None else None),
            inference_timing=getattr(self.model, 'last_timing', None),
            raw_action=action.tolist(), decoded=decoded, actions_sent=0, control_handles_created=0)


def require_live_ready():
    raise RuntimeError('SMPL live motion is unavailable: no validated SMPL standing/startup/exit path '
                       'or target RobotInfo/configuration evidence. No controller is created; '
                       'do not change robot configuration or weaken protection to bypass this gate.')
