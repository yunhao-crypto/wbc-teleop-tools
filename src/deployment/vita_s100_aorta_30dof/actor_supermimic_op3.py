# Linkage portions: Copyright (c) 2024-2026 Ziqi Fan, Apache-2.0.
# NumPy adaptation; see third_party/supermimic/NOTICE.md and LICENSE.
"""Official SuperMimic GenCoord OP3 actor; NumPy-only deployment adapter.

Body31 state/reference slots are virtual pitch/roll. Actions are native A/B
motor PD targets. The distinct metadata key prevents loading trackerlab models.
"""
from dataclasses import dataclass
import functools
import json

import numpy as np

from .actor_parallel_ankle import MotorThetaAnkle, command_space, decode_motor_theta
from .actor_groot_ankle import serial_policy_class
from .actor_supermimic import SuperMimicContract, _finite

CONTRACT_KEY = 'supermimic.op3_contract.v1'
POLICY_KIND = 'supermimic_op3_tracking'
TERM_WIDTHS = (3, 3, 28, 28, 28)
OFFSETS = tuple(range(0, 46, 5))
GEOMETRY = dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235)


def native_name(name):
    for side in ('left', 'right'):
        for axis, label in (('pitch', 'A'), ('roll', 'B')):
            if name == f'{side}_ankle_{axis}_joint':
                return f'{side}_{label}_achilles_joint'
    return name


@dataclass(frozen=True)
class OP3Contract(SuperMimicContract):
    reference_names: tuple = ()
    native_names: tuple = ()
    policy_kind: str = POLICY_KIND

    @classmethod
    def from_metadata(cls, metadata):
        if CONTRACT_KEY not in metadata:
            raise ValueError('model is missing '+CONTRACT_KEY)
        raw = json.loads(metadata[CONTRACT_KEY])
        expected = dict(schema_version=1, policy_kind=POLICY_KIND, robot_type='vita02a_op3',
                        input_dim=1760, action_dim=28, history_length=10, policy_dt=.02,
                        observation_terms=['gravity', 'gyro', 'motor_pos_rel', 'motor_vel', 'last_action'],
                        history_order='term_major_oldest_first', reset='repeat_first_frame',
                        goal_layout='per_frame_q40_qd40_then_relative_orientation_6d',
                        frame_offsets=list(OFFSETS), rotation_6d='first_two_columns_row_major',
                        orientation_frame='actual_robot_root', beyond_reference_end='clamp_last_frame_with_velocity',
                        streaming_velocity='virtual_gradient_then_linkage_central_difference',
                        action_clip=None, command_target_clipping='none',
                        required_ankle_command_space='motor_theta', normalizers='baked_into_onnx')
        if any(raw.get(k, 'missing') != v for k, v in expected.items()) or type(raw['schema_version']) is not int:
            raise ValueError('unsupported official OP3 contract/layout')
        names, native, reference = (tuple(raw[k]) for k in ('joint_names', 'native_joint_names', 'reference_joint_names'))
        if (len(names) != 28 or len(set(names)) != 28 or native != tuple(map(native_name, names))
                or len(reference) != 40 or len(set(reference)) != 40 or not set(native) <= set(reference)):
            raise ValueError('invalid OP3 joint mapping')
        dependent = {f'{s}_{l}_achilles{suffix}_joint' for s in ('left', 'right')
                     for l in ('A', 'B') for suffix in ('', '_rod_pitch', '_rod_roll')}
        if set(reference) != set(names) | dependent:
            raise ValueError('reference must contain all 40 native OP3 joints')
        arrays = [_finite(raw[k], k) for k in ('default_joint_pos', 'action_scale', 'joint_stiffness',
                  'joint_damping', 'joint_pos_limit_lower', 'joint_pos_limit_upper')]
        if any(a.shape != (28,) for a in arrays):
            raise ValueError('expected 28 control parameters')
        default, scale, kp, kd, lower, upper = arrays
        if (np.any(scale <= 0) or np.any(kp < 0) or np.any(kd < 0) or np.any(lower >= upper)
                or np.any(default < lower) or np.any(default > upper)):
            raise ValueError('invalid OP3 control parameters')
        ankle = MotorThetaAnkle(names, raw['ankle'])
        if ankle.geometry != GEOMETRY or ankle.motor_limits != (-.95975, .61075):
            raise ValueError('OP3 linkage geometry/travel mismatch')
        if not (np.allclose(lower[ankle.slots], ankle.motor_limits[0], atol=1e-9, rtol=0)
                and np.allclose(upper[ankle.slots], ankle.motor_limits[1], atol=1e-9, rtol=0)):
            raise ValueError('OP3 motor travel mismatch')
        virtual, error = ankle.to_virtual(default)
        if error > 1e-8:
            raise ValueError('unreachable default OP3 pose')
        lo, hi = lower.copy(), upper.copy()
        lo[ankle.slots], hi[ankle.slots] = -np.pi/2, np.pi/2
        # future_frames is the reference BUFFER horizon, not the number of goal samples.
        return cls(names, virtual, scale, kp, kd, lo, hi, np.inf, .02, 45, .02,
                   10, 1760, 90, 86, 1., policy_default=default, policy_lower=lower,
                   policy_upper=upper, ankle=ankle, required_ankle_command_space='motor_theta',
                   reference_names=reference, native_names=native)


def _linkage(pitch, roll, sign):
    """Vectorized port of training linkage_joint_positions (Apache-2.0, Ziqi Fan)."""
    roll = sign*roll
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    out = {}
    lx, ly = GEOMETRY['lx'], GEOMETRY['ly']
    for label, y, height in (('A', ly, GEOMETRY['z_top']), ('B', -ly, GEOMETRY['z_bot'])):
        ax, ay, az = -lx*cp+sp*sr*y, cr*y, lx*sp+cp*sr*y
        a, b, dy = ax, az-height, ay-y
        c = -(height**2-lx**2-(a*a+b*b+dy*dy))/(2*lx)
        disc = b*b*c*c-(a*a+b*b)*(c*c-a*a)
        theta = np.arcsin(np.clip((b*c+np.sqrt(np.maximum(disc, 0)))/(a*a+b*b), -1, 1))
        x, z = ax+lx*np.cos(theta), az-height-lx*np.sin(theta)
        length = np.sqrt(x*x+dy*dy+z*z)
        # Reject impossible virtual references instead of silently using clamped IK.
        if np.any(np.abs(length-height) > 1e-6):
            raise ValueError('OP3 reference outside closed-linkage workspace')
        x, yy, z = x/length, dy/length, z/length
        rx, rz = np.cos(theta)*x-np.sin(theta)*z, np.sin(theta)*x+np.cos(theta)*z
        out[f'{label}_achilles_joint'] = theta
        out[f'{label}_achilles_rod_pitch_joint'] = np.arctan2(-rx, -rz)
        out[f'{label}_achilles_rod_roll_joint'] = sign*np.arcsin(np.clip(yy, -1, 1))
    return out


def native_reference(q, qd, contract):
    """Expand virtual 28 coordinates/rates to training's 40-coordinate order."""
    q, qd = _finite(q, 'reference q'), _finite(qd, 'reference qd')
    if q.shape != qd.shape or q.ndim != 2 or q.shape[1] != 28:
        raise ValueError('reference must have matching [N,28] positions and rates')
    pos = dict(zip(contract.names, q.T))
    vel = dict(zip(contract.names, qd.T))
    eps = 1e-4
    for side, sign in (('left', 1), ('right', -1)):
        p, r = (contract.names.index(f'{side}_ankle_{a}_joint') for a in ('pitch', 'roll'))
        now = _linkage(q[:, p], q[:, r], sign)
        ahead = _linkage(q[:, p]+eps*qd[:, p], q[:, r]+eps*qd[:, r], sign)
        behind = _linkage(q[:, p]-eps*qd[:, p], q[:, r]-eps*qd[:, r], sign)
        for key in now:
            pos[f'{side}_{key}'] = now[key]
            vel[f'{side}_{key}'] = (ahead[key]-behind[key])/(2*eps)
    return (np.stack([pos[n] for n in contract.reference_names], axis=-1),
            np.stack([vel[n] for n in contract.reference_names], axis=-1))


@functools.cache
def policy_class(math_api):
    class OfficialOP3Policy(serial_policy_class(math_api)):
        def __init__(self, contract, profile):
            super().__init__(contract, profile)
            self._check_wire()

        def _check_wire(self):
            if command_space(self) != 'motor_theta':
                raise ValueError('official OP3 requires native motor_theta command space')

        def observe(self, motion, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            c, m = self.contract, math_api
            q, dq, gyro = _finite(q, 'q'), _finite(dq, 'dq'), _finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('observation state dimensions disagree')
            if not np.isfinite([ref_time, yaw_offset]).all():
                raise ValueError('invalid reference time/yaw')
            robot = m.rotation(m.quaternion(quat))
            current = np.concatenate((-robot[2], gyro, c.ankle.to_policy(q)-c.policy_default,
                                      c.ankle.velocity_to_policy(q, dq), self.last_action)).astype(np.float32)
            if not self.history:
                self.history.extend([current]*c.history_length)
            else:
                self.history.append(current)
            history = np.stack(self.history)
            chunks = np.split(history, np.cumsum(TERM_WIDTHS)[:-1], axis=1)
            frame = int(np.clip(round(ref_time/c.period), 0, len(motion.q)-1))
            indices = np.minimum(frame+np.asarray(OFFSETS), len(motion.q)-1)
            if hasattr(motion, 'op3_joint_pos'):
                if tuple(motion.op3_joint_names) != c.reference_names:
                    raise ValueError('native reference joint order mismatch')
                qp, qv = motion.op3_joint_pos[indices], motion.op3_joint_vel[indices]
            else:
                # One past and one tail sample suffice for central gradients.
                qd = (np.gradient(motion.q, c.period, axis=0) if len(motion.q) > 1
                      else np.zeros_like(motion.q))
                qp, qv = native_reference(motion.q[indices], qd[indices], c)
            offset = m.yaw_quaternion(yaw_offset)
            orientation = np.stack([(robot.T @ m.rotation(m.multiply(offset, motion.quat[i])))[:, :2].ravel()
                                    for i in indices])
            obs = np.concatenate([*(x.ravel() for x in chunks),
                                  np.concatenate((qp, qv), axis=1).ravel(), orientation.ravel()]).astype(np.float32)
            if obs.shape != (c.input_dim,) or not np.isfinite(obs).all():
                raise ValueError('invalid official OP3 observation')
            return obs

        def decode(self, raw_action, **timing):
            self._check_wire()
            action = _finite(raw_action, 'action')
            if action.shape != (28,) or np.any(np.abs(action) > np.finfo(np.float32).max):
                raise ValueError('OP3 actions must be representable as 28 finite float32 values')
            return decode_motor_theta(self, math_api, raw_action, **timing)
    return OfficialOP3Policy


def load_native_motion(path, contract, math_api):
    """Load qualified SuperMimic NPZ without re-differentiating its 40-joint rates.

    The returned SDK Motion can be passed directly to policy.observe. SDK
    lead_in/normalized_heading return a new virtual Motion; they intentionally
    fall back to reference reconstruction instead of retaining stale native data.
    """
    with np.load(path, allow_pickle=False) as data:
        if float(data['fps']) != 50. or tuple(data['joint_names']) != contract.reference_names:
            raise ValueError('native reference must match the 50 Hz OP3 joint contract')
        qp, qv = (_finite(data[k], k).copy() for k in ('joint_pos', 'joint_vel'))
        if qp.ndim != 2 or qp.shape[1] != 40 or qv.shape != qp.shape:
            raise ValueError('native reference must contain [N,40] positions and velocities')
        virtual = qp[:, [contract.reference_names.index(n) for n in contract.names]]
        rebuilt, _ = native_reference(virtual, np.zeros_like(virtual), contract)
        if not np.allclose(rebuilt, qp, atol=2e-4, rtol=0):
            raise ValueError('native reference violates OP3 linkage closure')
        motion = math_api.Motion(virtual, data['root_pos_w'], data['root_quat_w'])
    motion.op3_joint_names = contract.reference_names
    motion.op3_joint_pos, motion.op3_joint_vel = qp, qv
    return motion
