"""Named virtual-joint NPZ references, independent of Pico and recorded feedback.

Only the reference comes from disk. Observation history, ankle conversion and
goal features belong to the selected policy; live state always comes from SDK.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite motion '+name)
    return array


def load_motion(runtime, config_path):
    """Validate the entire clip before creating any live controller."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = runtime.root/config_path
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text())
    known = {'schema', 'path', 'sha256', 'root_body_index', 'end_hold_s', 'limit_projection_max_rad'}
    if (cfg.get('schema') != 'vita_actor_motion_v1' or set(cfg)-known
            or not isinstance(cfg.get('path'), str) or not cfg['path']):
        raise ValueError('Invalid fixed-motion configuration')
    root_index = cfg.get('root_body_index', 0)
    hold = cfg.get('end_hold_s', 1.)
    projection = cfg.get('limit_projection_max_rad', 1e-6)
    if (isinstance(projection, bool) or not isinstance(projection, (int, float))
            or not math.isfinite(projection) or not 0 <= projection <= 1e-4):
        raise ValueError('limit_projection_max_rad must be within [0, 0.0001]')
    if (type(root_index) is not int or root_index < 0 or isinstance(hold, bool)
            or not isinstance(hold, (int, float)) or not math.isfinite(hold) or not 0 <= hold <= 30):
        raise ValueError('Invalid root_body_index or end_hold_s')
    path = Path(cfg['path'])
    if not path.is_absolute():
        path = runtime.root/path
    path = path.resolve()
    sha = digest(path)
    if cfg.get('sha256') is not None and cfg['sha256'] != sha:
        raise ValueError('Fixed-motion SHA256 mismatch')
    c, m = runtime.model.contract, runtime.math
    with np.load(path, allow_pickle=False) as f:
        raw_names = np.asarray(f['joint_names'])
        if raw_names.ndim != 1 or raw_names.dtype.kind not in 'US':
            raise ValueError('joint_names must be a one-dimensional string array')
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw_names]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError('Duplicate/empty motion joint names')
        missing = set(c.names)-set(names)
        if missing:
            raise ValueError('Motion missing virtual joints: '+', '.join(sorted(missing)))
        if 'joint_space' in f and str(np.asarray(f['joint_space']).item()) != 'virtual_pitch_roll':
            raise ValueError('Motion must use virtual_pitch_roll, not motor A/B coordinates')
        source_control_space = str(np.asarray(f['control_space']).item()) if 'control_space' in f else None
        if source_control_space == 'motor_theta' and 'joint_space' not in f:
            motor_names = {f'{side}_{motor}_achilles_joint' for side in ('left','right') for motor in ('A','B')}
            if not motor_names <= set(names):
                raise ValueError('Ambiguous motor-theta NPZ: need separate named motor joints or explicit virtual joint_space')
        raw_q = _finite(f['joint_pos'], 'joint_pos')
        if raw_q.ndim != 2 or raw_q.shape[1] != len(names) or not 1 <= len(raw_q) <= 180000:
            raise ValueError('Expected joint_pos [N,J], 1 <= N <= 180000')
        for name in set(names)-set(c.names):
            if name == 'waist_roll_joint' and np.any(np.abs(raw_q[:, names.index(name)]) > 1e-6):
                raise ValueError('Nonzero waist_roll_joint cannot be represented by these 28-DOF policies')
        q = raw_q[:, [names.index(n) for n in c.names]]
        base_keys = {'base_pos_w', 'base_quat_w'}
        body_keys = {'body_pos_w', 'body_quat_w'}
        keys = set(f.files)
        if ((base_keys.intersection(keys) and not base_keys <= keys)
                or (body_keys.intersection(keys) and not body_keys <= keys)):
            raise ValueError('Incomplete root position/quaternion pair')
        if base_keys <= keys and not body_keys.intersection(keys):
            base, quat = _finite(f['base_pos_w'], 'base_pos_w'), _finite(f['base_quat_w'], 'base_quat_w')
            layout = 'base'
        elif body_keys <= keys:
            bp, bq = _finite(f['body_pos_w'], 'body_pos_w'), _finite(f['body_quat_w'], 'body_quat_w')
            if (bp.ndim != 3 or bq.ndim != 3 or bp.shape[:2] != bq.shape[:2]
                    or bp.shape[2] != 3 or bq.shape[2] != 4 or root_index >= bp.shape[1]):
                raise ValueError('Expected body_pos_w [N,B,3], body_quat_w [N,B,4] and a valid root_body_index')
            base, quat = bp[:, root_index], bq[:, root_index]
            layout = 'body'
            if 'body_names' in f or 'root_body' in f:
                if not {'body_names', 'root_body'} <= keys:
                    raise ValueError('body_names and root_body must be supplied together')
                body_names = np.asarray(f['body_names'])
                if (body_names.shape != (bp.shape[1],) or body_names.dtype.kind not in 'US'
                        or str(body_names[root_index]) != str(np.asarray(f['root_body']).item())):
                    raise ValueError('root_body_index disagrees with named root_body')
            if base_keys <= keys:
                if (not np.array_equal(_finite(f['base_pos_w'], 'base_pos_w'), base)
                        or not np.array_equal(_finite(f['base_quat_w'], 'base_quat_w'), quat)):
                    raise ValueError('Conflicting base/body root aliases')
                layout = 'body_with_verified_base_aliases'
        else:
            raise ValueError('Supply base_pos_w/base_quat_w or body_pos_w/body_quat_w')
        if base.shape != (len(q), 3) or quat.shape != (len(q), 4):
            raise ValueError('Root arrays must match the joint_pos frame count')
        norms = np.linalg.norm(quat, axis=1)
        if np.any(np.abs(norms-1.) > .02):
            raise ValueError('Root quaternions must be unit wxyz (within 2%)')
        rates = [_finite(f[key], key) for key in ('fps', 'framerate') if key in f]
        if not rates or any(x.shape != () for x in rates):
            raise ValueError('Motion needs scalar fps or framerate')
        fps = float(rates[0])
        if not 1 <= fps <= 1000 or any(not np.isclose(float(x), fps) for x in rates):
            raise ValueError('Invalid or inconsistent motion frame rate')
    # Compare virtual references to virtual profile limits, never motor-theta limits.
    limits = np.asarray([runtime.config['body_profile']['joint_limits'][n] for n in c.names])
    invalid = (q < limits[:, 0]-projection) | (q > limits[:, 1]+projection)
    if invalid.any():
        i, j = np.argwhere(invalid)[0]
        raise ValueError(f'Motion frame {i} {c.names[j]}={q[i,j]} outside virtual limits {limits[j].tolist()}')
    projected = np.clip(q, limits[:, 0], limits[:, 1])
    corrections = [dict(frame=int(i), joint=c.names[j], original_rad=float(q[i,j]),
                        projected_rad=float(projected[i,j]), delta_rad=float(projected[i,j]-q[i,j]))
                   for i,j in np.argwhere(projected != q)]
    q = projected  # Reference-only projection; command/firmware bounds remain unchanged.
    duration = (len(q)-1)/fps
    count = int(math.ceil(duration/c.period-1e-9))+1
    if count > 180000:
        raise ValueError('Resampled clip exceeds 180000 policy frames')
    times, target = np.arange(len(q))/fps, np.arange(count)*c.period
    src = m.Motion(q, base, quat)
    qp = np.stack([np.interp(target, times, q[:, j]) for j in range(28)], axis=1)
    bp = np.stack([np.interp(target, times, base[:, j]) for j in range(3)], axis=1)
    rotations = []
    for t in target:
        if len(q) == 1 or t >= duration:
            rotations.append(src.quat[-1])
        else:
            i = max(0, int(np.searchsorted(times, t, side='right'))-1)
            rotations.append(m.slerp(src.quat[i], src.quat[i+1], (t-times[i])*fps))
    bp[:, :2] -= bp[0, :2].copy()  # Remove the world origin; retain relative translation.
    motion = m.Motion(qp, bp, rotations)
    if digest(path) != sha:
        raise ValueError('Motion file changed while loading')
    return FixedMotion(runtime, motion, hold, dict(path=str(path), sha256=sha,
        config=str(config_path), config_sha256=digest(config_path), source_fps=fps,
        source_frames=len(q), policy_frames=count, period_s=c.period, root_layout=layout,
        root_body_index=root_index if layout.startswith('body') else None,
        limit_projection_max_rad=projection, limit_corrections=corrections,
        joint_space='virtual_pitch_roll', quaternion_order='wxyz',
        source_control_space=source_control_space,
        joint_names=list(c.names), ignored_joints=sorted(set(names)-set(c.names)),
        source_duration_s=duration, end_hold_s=hold))


class FixedMotion:
    def __init__(self, runtime, motion, end_hold_s, metadata):
        self.runtime, self.motion, self.end_hold_s, self.metadata = runtime, motion, end_hold_s, metadata
        self.prepared = None
        self.tick = 0

    def begin(self, q, root):
        r, c, m = self.runtime, self.runtime.model.contract, self.runtime.math
        prepared = self.motion.lead_in(q, root, r.config['body_profile']['lead_in'])
        self.lead_frames = len(prepared.q)-len(self.motion.q)
        # MorphMimic consumes absolute reference anchor height. Blend it from
        # the model's stand height, as well as blending measured joint angles.
        if hasattr(c, 'nominal_root_height') and self.lead_frames:
            alpha = .5-.5*np.cos(np.pi*np.arange(self.lead_frames)/self.lead_frames)
            prepared.base[:self.lead_frames, 2] = c.nominal_root_height+alpha*(prepared.base[self.lead_frames, 2]-c.nominal_root_height)
        hold_frames = int(math.ceil(self.end_hold_s/c.period))
        self.total_ticks = len(prepared.q)+hold_frames
        # Enough terminal frames for every derivative/future goal to see a
        # constant pose; these extra look-ahead frames do not prolong playback.
        padding = max(hold_frames, r.reference.steps+2)
        repeat = lambda a: np.concatenate((a, np.repeat(a[-1:], padding, axis=0)))
        self.prepared = m.Motion(repeat(prepared.q), repeat(prepared.base), repeat(prepared.quat))
        self.yaw_offset = m.yaw(root)
        self.tick = 0

    @property
    def finished(self):
        return self.prepared is not None and self.tick >= self.total_ticks

    def sample(self):
        if self.prepared is None or self.finished:
            raise RuntimeError('Motion is not active')
        r, p = self.runtime, self.prepared
        past = r.reference.current_index
        lo = max(0, self.tick-past)
        hi = min(len(p.q), self.tick+r.reference.steps-past)
        # Bounded windows keep MorphMimic's gradient cost independent of clip length.
        window = r.math.Motion(p.q[lo:hi], p.base[lo:hi], p.quat[lo:hi])
        return window, (self.tick-lo)*r.model.contract.period

    def infer(self, *, q, dq, root, gyro):
        window, ref_time = self.sample()
        command, obs, row = self.runtime.infer_motion_command(window, q=q, dq=dq,
            root=root, gyro=gyro, yaw_offset=self.yaw_offset, ref_time=ref_time,
            diagnostics=dict(reference_source='npz', motion_tick=self.tick,
                motion_time_s=self.tick*self.runtime.model.contract.period,
                motion_phase='lead_in' if self.tick < self.lead_frames else
                    'clip' if self.tick < self.lead_frames+len(self.motion.q) else 'end_hold'))
        row['motion_reference'] = dict(q=window.q.tolist(), base=window.base.tolist(),
            root_wxyz=window.quat.tolist(), ref_time=ref_time, yaw_offset=self.yaw_offset)
        return command, obs, row

    def advance(self):
        """Advance only after successful publication (or an explicit offline tick)."""
        if self.prepared is None or self.finished:
            raise RuntimeError('Cannot advance an inactive motion')
        self.tick += 1
