"""Pico's post-conversion SMPL pose interface -> SONIC human reference.

Only NumPy FK from the existing Pico implementation is reused. Its 258-value
HoloMotion descriptor is NOT the input to the SONIC 840-value human encoder.
This module creates no transport, control handles, estimator or BVH parser.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .actor_smpl import SmplReference, WRIST_NAMES, digest, rotvec_quaternion
from .onboard_preprocessing import COMMAND, forward_kinematics, load_skeleton


def _float32(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite float32 {shape}')
    return array


def read_formal(path):
    """Read the existing formal_human_npz_v1 schema, already Z-up and 50Hz.

    File timing is target_fps in metadata. No implicit resampling, Y-up
    conversion, hand padding, or reinterpretation of legacy pose_aa is done.
    """
    required = {'human_pose_aa', 'human_root_trans', 'human_shape_beta',
                'human_root_height', 'human_gravity_projection', 'metadata'}
    with np.load(path, allow_pickle=False) as data:
        if not required <= set(data.files):
            raise ValueError('Missing formal SMPL fields: '+', '.join(sorted(required-set(data.files))))
        if set(data.files) & {'root_orient', 'pose_body', 'trans', 'betas', 'source_fps', 'target_fps'}:
            raise ValueError('Canonical fields must be converted to formal SMPL explicitly')
        raw = data['metadata']
        if raw.shape != () or raw.dtype.kind not in 'US':
            raise ValueError('metadata must be a JSON string scalar')
        meta = json.loads(raw.item())
        if not isinstance(meta, dict):
            raise ValueError('metadata must contain a JSON object')
        pose = data['human_pose_aa'].copy()
        if pose.ndim != 2 or pose.shape[1] != 72 or not 1 <= len(pose) <= 180000:
            raise ValueError('human_pose_aa must be [T,72], 1 <= T <= 180000')
        count = len(pose)
        pose = _float32(pose, (count, 72), 'human_pose_aa')
        trans = _float32(data['human_root_trans'], (count, 3), 'human_root_trans').copy()
        beta = data['human_shape_beta']
        if beta.ndim != 1 or not 1 <= len(beta) <= 300:
            raise ValueError('human_shape_beta must be a nonempty vector')
        _float32(beta, beta.shape, 'human_shape_beta')
        if np.any(beta != 0):
            raise ValueError('This Pico route uses a neutral zero-beta skeleton; shaped FK is not implemented')
        height = _float32(data['human_root_height'], (count, 1), 'human_root_height')
        gravity = _float32(data['human_gravity_projection'], (count, 3), 'human_gravity_projection')
        if not np.allclose(height, trans[:, 2:3], atol=1e-6, rtol=1e-6):
            raise ValueError('human_root_height differs from human_root_trans')
        # Same source-root gravity as holosmpl. No command-frame correction here.
        root = rotvec_quaternion(pose[:, :3])
        w, x, y, z = root.T
        expected = np.column_stack((2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1))
        if not np.allclose(gravity, expected, atol=1e-5, rtol=1e-5):
            raise ValueError('human_gravity_projection differs from SMPL root orientation')
        if (meta.get('schema_version') != 'formal_human_npz_v1'
                or meta.get('canonical_coordinate_system') != 'canonical_z_up_xyz_m'
                or meta.get('up_axis') != 'z' or meta.get('unit') != 'meter'
                or meta.get('slice_policy') != 'none'
                or meta.get('frame_count') != count or meta.get('target_fps') != 50
                or not meta.get('formal_pose_72_policy')):
            raise ValueError('Formal metadata must declare canonical Z-up metres, frame count, pose policy and target_fps=50')
    return pose, trans, meta


def read_wrists(path, *, expected_sha256, count):
    """Separate robot-reference sidecar: formal human files stay human-only."""
    if digest(path) != expected_sha256:
        raise ValueError('Wrist input SHA256 mismatch')
    with np.load(path, allow_pickle=False) as data:
        required = {'wrist_joint_pos', 'wrist_joint_names', 'fps'}
        if not required <= set(data.files):
            raise ValueError('Missing explicit wrist-reference fields')
        names = data['wrist_joint_names']
        if names.dtype.kind not in 'US' or names.shape != (6,):
            raise ValueError('wrist_joint_names must be six strings')
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in names]
        if names != list(WRIST_NAMES):
            raise ValueError('Wrist joint order differs from the model contract')
        if data['fps'].shape != () or float(data['fps']) != 50.:
            raise ValueError('Wrist reference requires 50Hz')
        wrists = _float32(data['wrist_joint_pos'], (count, 6), 'wrist_joint_pos').copy()
    if digest(path) != expected_sha256:
        raise ValueError('Wrist input changed during loading')
    return wrists


class PicoSmplAdapter:
    """Fixed neutral skeleton FK using the exact existing Pico conventions.

    The CLI supplies the repository's reviewed skeleton manifest. Supplying
    an unrelated skeleton/manifest pair does not satisfy that binding.
    """
    def __init__(self, math_api, *, skeleton_path, manifest_path):
        self.math = math_api
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if (manifest.get('schema') != 'vita_cpu_smpl_skeleton_v1'
                or manifest.get('rest_shape') != 'neutral zero-beta model'):
            raise ValueError('Need a reviewed neutral SMPL skeleton manifest')
        if not Path(skeleton_path).is_file():
            raise FileNotFoundError('Missing licensed preprocessing_skeleton.npz: SMPL rotations need rest joint offsets for FK')
        if digest(skeleton_path) != manifest.get('artifact_sha256'):
            raise ValueError('Skeleton differs from the reviewed Pico manifest')
        parents, offsets, metadata = load_skeleton(skeleton_path)
        calculated = hashlib.sha256(parents.astype(np.int64).tobytes()+offsets.tobytes()).hexdigest()
        if (calculated != manifest.get('skeleton_hash')
                or metadata.get('skeleton_hash') != calculated
                or metadata.get('source_pickle_sha256') != manifest.get('source_pickle_sha256')
                or np.any(offsets[0] != 0)):
            raise ValueError('Skeleton geometry or source differs from its reviewed manifest')
        self.parents, self.offsets = parents, offsets
        self.skeleton_sha256 = manifest['artifact_sha256']

    def convert_frame(self, human_pose_aa, human_root_trans):
        """Same pose/trans shapes as Pico store_smpl, without a live buffer.

        Global translation is returned for provenance/visualization. SONIC's
        840 human features use pelvis-relative positions, so it is not added
        to the joints and cannot become a second translation accidentally.
        """
        pose = np.asarray(human_pose_aa)
        trans = np.asarray(human_root_trans)
        if pose.shape == (1, 72):
            pose = pose[0]
        if trans.shape == (1, 3):
            trans = trans[0]
        pose = _float32(pose, (72,), 'human_pose_aa')
        trans = _float32(trans, (3,), 'human_root_trans')
        joints, rotations = forward_kinematics(pose, self.parents, self.offsets)
        # Exactly Pico's root_command = global root * inverse SMPL base.
        root = self.math.multiply(rotations[0], COMMAND)
        root = root/np.linalg.norm(root)
        return joints, root, trans.copy()

    def from_npz(self, path, *, wrist_path, wrist_sha256):
        pose, trans, meta = read_formal(path)
        wrists = read_wrists(wrist_path, expected_sha256=wrist_sha256, count=len(pose))
        joints = np.empty((len(pose), 24, 3), np.float32)
        roots = np.empty((len(pose), 4), np.float64)
        for i in range(len(pose)):
            joints[i], roots[i], _ = self.convert_frame(pose[i], trans[i])
        ref = SmplReference(self.math, joints, roots, wrists)
        ref.human_root_trans = trans
        ref.input_metadata = meta
        ref.skeleton_sha256 = self.skeleton_sha256
        return ref
