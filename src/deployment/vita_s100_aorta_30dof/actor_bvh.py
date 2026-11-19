"""Authenticated BONES BVH -> SMPL24 semantic references, matching the v4.4 map.

This is hierarchical FK, not a fitted SMPL body mesh. No robot transport, IK,
motion command, pose clipping or coordinate guessing is performed.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .actor_smpl import JOINT_NAMES, WRIST_NAMES, digest

BVH_JOINTS = ('Hips', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftShin', 'RightShin',
              'Spine2', 'LeftFoot', 'RightFoot', 'Chest', 'LeftToeBase', 'RightToeBase',
              'Neck1', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm',
              'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
              'LeftHandMiddle1', 'RightHandMiddle1')


def _require(test, message):
    if not test:
        raise ValueError(message)


@dataclass
class BvhMotion:
    bones: list
    names: dict
    dt: float
    values: np.ndarray


def read_bvh(raw):
    try:
        header, body = raw.decode('ascii').split('MOTION', 1)
    except (UnicodeError, ValueError) as error:
        raise ValueError('Expected ASCII BVH HIERARCHY and MOTION') from error
    _require(header.lstrip().startswith('HIERARCHY'), 'Missing BVH hierarchy')
    stack, bones, names, columns = [], [], {}, 0
    for line in header.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in ('ROOT', 'JOINT', 'End'):
            _require(len(tokens) == 2, 'Invalid BVH joint header')
            parent = stack[-1] if stack else -1
            _require(bool(stack) or not bones, 'BVH must have one root')
            name = tokens[1] if tokens[0] != 'End' else f'end_{len(bones)}'
            _require(name not in names, 'Duplicate BVH joint '+name)
            names[name] = len(bones); stack.append(len(bones))
            bones.append(dict(name=name, parent=parent, channels=()))
        elif tokens[0] == '}':
            _require(bool(stack), 'Unbalanced BVH braces'); stack.pop()
        elif tokens[0] == 'OFFSET':
            _require(bool(stack) and len(tokens) == 4, 'Invalid BVH offset')
            bones[stack[-1]]['offset'] = np.asarray(tokens[1:], dtype=float)
        elif tokens[0] == 'CHANNELS':
            _require(bool(stack) and len(tokens) >= 2, 'Invalid BVH channels')
            labels = tokens[2:]
            _require(len(labels) == int(tokens[1]) and len(labels) == len(set(labels)), 'Invalid BVH channel count')
            _require(all(v in {a+b for a in 'XYZ' for b in ('position', 'rotation')} for v in labels), 'Unsupported BVH channel')
            bones[stack[-1]]['channels'] = [(label, columns+i) for i, label in enumerate(labels)]
            columns += len(labels)
    lines = body.strip().splitlines()
    _require(not stack and bones and all('offset' in b for b in bones), 'Incomplete BVH hierarchy')
    _require(len(lines) >= 3 and lines[0].startswith('Frames:') and lines[1].startswith('Frame Time:'), 'Invalid BVH clock')
    count = int(lines[0].split(':')[1]); dt = float(lines[1].split(':')[1])
    _require(2 <= count <= 1000000 and np.isfinite(dt) and dt > 0 and columns > 0, 'Invalid BVH frame count/time')
    values = np.fromstring(' '.join(lines[2:]), sep=' ')
    _require(values.size == count*columns and np.isfinite(values).all(), 'BVH frame data size/nonfinite values')
    return BvhMotion(bones, names, dt, values.reshape(count, columns))


def semantic_geometry(raw, pairing, *, expected_sha256, clock='training_float32'):
    """Use the authenticated pairing basis and source crop on a 50Hz grid.

    Position channels replace OFFSET on their axis. Rotations preserve the
    declared intrinsic CHANNEL order, are SLERPed locally, then composed by FK.
    Do not mirror an `_M` file a second time.
    """
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'BVH SHA256 mismatch')
    _require(pairing.get('human', {}).get('sha256') == expected_sha256, 'BVH/pairing identity mismatch')
    _require(isinstance(pairing.get('training_motion_id'), str) and pairing['training_motion_id'].startswith('bones__'), 'Missing pairing motion identity')
    source = read_bvh(raw); timing = pairing['time']
    _require(len(source.values) == timing['source_frames'] and np.isclose(1/source.dt, timing['source_fps'], rtol=0, atol=1e-8), 'BVH/pairing source clock mismatch')
    start, end = timing['crop_start_frames'], timing['crop_end_frames']
    _require(type(start) is int and type(end) is int and start >= 0 and end >= 0 and start+end < len(source.values)-1, 'Invalid BVH crop')
    values = source.values[start:len(source.values)-end]
    _require(timing.get('source_frames_after_crop', len(values)) == len(values), 'BVH cropped frame count mismatch')
    duration = (len(values)-1)*source.dt
    count = int(np.ceil(duration*50))
    _require(count >= 2, 'BVH duration too short')
    if clock == 'training_float32':
        target = (np.arange(count, dtype=np.float64)/50).astype(np.float32).astype(np.float64)
    elif clock == 'exact_50hz':
        target = np.arange(count, dtype=np.float64)/50
    else:
        raise ValueError('Unknown BVH resampling clock')
    target = np.minimum(target, duration)
    times = np.arange(len(values))*source.dt
    _require(set(BVH_JOINTS) <= set(source.names), 'BVH missing required BONES semantic joints')
    selected = [source.names[n] for n in BVH_JOINTS]; needed = set(selected)
    for i in selected:
        parent = source.bones[i]['parent']
        while parent != -1:
            needed.add(parent); parent = source.bones[parent]['parent']
    positions, orientations = {}, {}
    for i in sorted(needed):
        bone = source.bones[i]; p = np.broadcast_to(bone['offset'], (count,3)).copy()
        rot_axes, rot_columns = [], []
        for label, col in bone['channels']:
            if label.endswith('position'):
                p[:, 'XYZ'.index(label[0])] = np.interp(target,times,values[:,col])
            else:
                rot_axes.append(label[0]); rot_columns.append(col)
        _require(len(rot_columns) == 3 and set(rot_axes) == set('XYZ'), 'Each selected BVH ancestor needs three rotation channels')
        rot = Slerp(times, Rotation.from_euler(''.join(rot_axes), values[:,rot_columns], degrees=True))(target)
        parent = bone['parent']
        if parent >= 0:
            positions[i] = positions[parent]+orientations[parent].apply(p)
            orientations[i] = orientations[parent]*rot
        else:
            positions[i], orientations[i] = p, rot
    z_up = Rotation.from_euler('x',90,degrees=True)
    world = np.stack([z_up.apply(positions[i])*.01 for i in selected],axis=1)
    pelvis = world[:,0].copy(); joints = world-pelvis[:,None]
    basis = np.asarray(pairing['basis_to_robot'],dtype=float)
    _require(basis.shape == (3,3) and np.isfinite(basis).all()
             and np.allclose(basis@basis.T,np.eye(3),rtol=0,atol=1e-8)
             and np.isclose(np.linalg.det(basis),1.,rtol=0,atol=1e-8), 'Pairing basis must be a proper rotation')
    physical = z_up*orientations[selected[0]]*Rotation.from_matrix(basis.T)
    base = Rotation.from_quat([.5,.5,.5,.5])
    pose = np.zeros((count,72),np.float32); pose[:,:3] = (physical*base).as_rotvec()
    # Match the original float32 root sidecar roundtrip, preserving its exact
    # source representation rather than silently changing precision/convention.
    decoded = Rotation.from_rotvec(pose[:,:3])*base.inv()
    local = np.einsum('tji,tkj->tki',decoded.as_matrix(),joints)
    separation = local[:,1,1]-local[:,2,1]
    _require(np.min(separation) > .03, 'Incorrect BVH left/right or root axes')
    return dict(smpl_joints=joints.astype(np.float32),pose_aa=pose,transl=pelvis.astype(np.float32),
                root_quat_wxyz=decoded.as_quat()[:,[3,0,1,2]],time_s=target), dict(
        motion_id=pairing['training_motion_id'],bvh_sha256=expected_sha256,source_frames=len(source.values),source_fps=1/source.dt,
        source_crop=[start,end],frames_50hz=count,clock=clock,output_fps=50,units='cm_to_m',source_up='Y',output_up='Z',
        semantic_bvh_names=list(BVH_JOINTS),semantic_smpl_names=list(JOINT_NAMES),basis_to_robot=basis.tolist(),
        min_hip_left_minus_right_m=float(np.min(separation)),extra_mirroring=False,smpl_mesh_fitting=False)


def paired_wrists(path, *, expected_sha256, count):
    _require(digest(path) == expected_sha256, 'Paired wrist source SHA256 mismatch')
    with np.load(path,allow_pickle=False) as z:
        names = [str(n) for n in z['joint_names']]
        _require(len(set(names)) == len(names) and set(WRIST_NAMES) <= set(names), 'Paired robot wrist names unavailable')
        q = np.asarray(z['dof'],dtype=float); times = np.asarray(z['time'],dtype=float)
        _require(q.shape == (len(times),len(names)) and len(times) >= 2 and np.all(np.diff(times)>0)
                 and np.isfinite(q).all() and np.isfinite(times).all(), 'Invalid paired wrist trajectory')
        query = np.arange(count)/50
        _require(query[-1] <= times[-1]+.02, 'BVH/wrist duration mismatch')
        return np.stack([np.interp(query,times,q[:,names.index(n)]) for n in WRIST_NAMES],axis=-1)


def save_reference(path, geometry, wrists):
    _require(wrists.shape == (len(geometry['smpl_joints']),6) and np.isfinite(wrists).all(), 'Six aligned robot wrist references required')
    np.savez_compressed(path,smpl_joints=geometry['smpl_joints'],root_quat_wxyz=geometry['root_quat_wxyz'],
        wrist_joint_pos=wrists,fps=np.array(50.),smpl_joint_names=np.array(JOINT_NAMES),wrist_joint_names=np.array(WRIST_NAMES),
        joint_space=np.array('pelvis_relative_world_z_up'))
