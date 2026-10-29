"""SuperMimic Vita02A tracking actor on the SDK 0.2.1 Body31 interface.

The exported model states its observation/goal layout in
``supermimic.tracking_contract.v1`` metadata. The SDK's quaternion math,
feedback validation, Body31 mapping, joint limits and ankle feedforward are
reused unchanged; only the observation builder differs from the G1 actor.
"""
from dataclasses import dataclass
import functools
import importlib
import json
import math
from collections import deque

import numpy as np

CONTRACT_KEY = 'supermimic.tracking_contract.v1'
POLICY_KIND = 'supermimic_vita02a_tracking'
OBSERVATION_TERMS = [('joint_pos_rel_default', 28), ('joint_vel', 28), ('base_ang_vel_body', 3),
                     ('gravity_dir_body', 3), ('last_action', 28), ('reference_root_ori_rel_6d', 6)]
GOAL_TERMS = [('root_pos_delta_heading', 3), ('root_rot_heading_6d', 6), ('root_lin_vel_local', 3),
              ('root_ang_vel_local', 3), ('joint_pos', 28), ('joint_vel_scaled', 28)]
VELOCITY_RULES = dict(joint='forward_difference_last_repeats_penultimate',
                      root_linear='gradient_then_gaussian_sigma2_nearest',
                      root_angular='forward_quaternion_difference_world_then_gaussian_sigma2_nearest')
# Training references come from SONIC Humanoid_Batch:
# scipy gaussian_filter1d(sigma=2, mode='nearest', truncate=4).
SIGMA, RADIUS = 2., 8
KERNEL = np.exp(-.5*(np.arange(-RADIUS, RADIUS+1)/SIGMA)**2)
KERNEL /= KERNEL.sum()


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains NaN or infinity')
    return value


@dataclass(frozen=True)
class SuperMimicContract:
    names: tuple
    default: np.ndarray
    scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    action_clip: float
    period: float
    future_frames: int
    future_dt: float
    history_length: int
    input_dim: int
    frame_dim: int
    goal_frame_dim: int
    joint_velocity_scale: float
    include_z: bool = False
    policy_kind: str = POLICY_KIND

    @classmethod
    def from_metadata(cls, metadata):
        if CONTRACT_KEY not in metadata:
            raise ValueError(f'model is missing {CONTRACT_KEY}')
        raw = json.loads(metadata[CONTRACT_KEY])
        obs, goal = raw.get('observation', {}), raw.get('goal', {})
        if (raw.get('schema_version') != 1 or raw.get('policy_kind') != POLICY_KIND
                or raw.get('robot_type') != 'vita02a' or raw.get('action_dim') != 28
                or raw.get('input_layout') != 'observation_history_then_goal'
                or obs.get('history_order') != 'frame_major_oldest_first'
                or obs.get('reset') != 'repeat_first_frame'
                or obs.get('rotation_6d') != 'first_two_columns_row_major'
                or [(t['name'], t['dim']) for t in obs.get('terms', [])] != OBSERVATION_TERMS
                or [(t['name'], t['dim']) for t in goal.get('terms', [])] != GOAL_TERMS
                or goal.get('heading_frame') != 'current_reference_root_yaw'
                or goal.get('beyond_reference_end') != 'hold_last_frame_zero_velocity'
                or goal.get('velocity_rules') != VELOCITY_RULES):
            raise ValueError('unsupported SuperMimic tracking contract/layout')
        names = tuple(raw['joint_names'])
        if len(names) != 28 or len(set(names)) != 28 or not all(names):
            raise ValueError('model must name 28 distinct policy joints')
        arrays = [_finite(raw[k], k) for k in (
            'default_joint_pos', 'action_scale', 'joint_stiffness',
            'joint_damping', 'joint_pos_limit_lower', 'joint_pos_limit_upper')]
        if any(x.shape != (28,) for x in arrays):
            raise ValueError('model arrays must match the 28 named joints')
        default, scale, kp, kd, lower, upper = arrays
        if (np.any(scale <= 0) or np.any(kp < 0) or np.any(kd < 0) or np.any(lower >= upper)
                or np.any(default < lower) or np.any(default > upper)):
            raise ValueError('model has invalid scales, gains or limits')
        period, clip = float(raw['policy_dt']), float(raw['action_clip'])
        history, count = obs['history_length'], goal['num_frames']
        frame_dim, goal_dim = obs['frame_dim'], goal['frame_dim']
        if (not math.isclose(period, .02, abs_tol=1e-9) or not math.isclose(float(goal['dt']), period, abs_tol=1e-9)
                or not math.isfinite(clip) or clip <= 0
                or type(history) is not int or history <= 0 or type(count) is not int or count <= 0
                or goal['frame_offsets'] != list(range(1, count+1))
                or frame_dim != sum(d for _, d in OBSERVATION_TERMS)
                or goal_dim != sum(d for _, d in GOAL_TERMS)
                or raw['input_dim'] != history*frame_dim+count*goal_dim
                or float(goal['joint_velocity_scale']) != .1):
            raise ValueError('SuperMimic metadata timing or dimensions disagree')
        if raw.get('ankle', {}).get('control_space') != 'virtual_pitch_roll':
            raise ValueError('Body31 tracking requires virtual pitch/roll ankles')
        return cls(names, default, scale, kp, kd, lower, upper, clip, period, count, period,
                   history, raw['input_dim'], frame_dim, goal_dim, float(goal['joint_velocity_scale']))


def is_supermimic(contract):
    return getattr(contract, 'policy_kind', None) == POLICY_KIND


def gaussian_nearest(x):
    padded = np.pad(x, [(RADIUS, RADIUS)]+[(0, 0)]*(x.ndim-1), mode='edge')
    return sum(w*padded[i:i+len(x)] for i, w in enumerate(KERNEL))


def reference_velocities(math_api, base, quat, q, period):
    """Joint, world root linear and world root angular velocities of a window.

    The final sample has no successor. Its joint velocity copies the
    penultimate forward difference, q[n-2]-q[n-3], exactly as SONIC
    Humanoid_Batch does. Training zeroes the final angular difference before
    filtering, but inside a causal window the next sample is merely not yet
    known, so the previous difference is repeated instead.
    """
    n = len(q)
    if n < 2:
        return np.zeros_like(q), np.zeros((n, 3)), np.zeros((n, 3))
    qd = np.empty_like(q)
    qd[:-1] = (q[1:]-q[:-1])/period
    qd[-1] = qd[-3] if n >= 3 else qd[-2]
    v = gaussian_nearest(np.gradient(base, axis=0)/period)
    w = np.empty((n, 3))
    for t in range(n-1):
        d = math_api.multiply(quat[t+1], math_api.conjugate(quat[t]))
        if d[0] < 0:
            d = -d
        s = np.linalg.norm(d[1:])
        w[t] = (2*math.atan2(s, d[0])/s*d[1:] if s > 1e-12 else 2*d[1:])/period
    w[-1] = w[-2]
    return qd, v, gaussian_nearest(w)


def heading(R):
    """Yaw-only rotation from the projected x axis, as in training."""
    fx, fy = R[0, 0], R[1, 0]
    n = math.sqrt(fx*fx+fy*fy+1e-9)
    c, s = fx/n, fy/n
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def goal_features(base, R, v, w, q, qd, frame, count, joint_velocity_scale=.1):
    """Training MotionLib.goal for one frame of a [N] reference."""
    n = len(q)
    requested = frame+np.arange(1, count+1)
    future = np.minimum(requested, n-1)
    recorded = (requested <= n-1)[:, None]
    H = heading(R[frame])
    dp = (base[future]-base[frame]) @ H
    rot = np.einsum('ji,kjl->kil', H, R[future])[:, :, :2].reshape(count, 6)
    Rt = R[future].transpose(0, 2, 1)
    v_local = np.where(recorded, np.einsum('kij,kj->ki', Rt, v[future]), 0.)
    w_local = np.where(recorded, np.einsum('kij,kj->ki', Rt, w[future]), 0.)
    qd_future = np.where(recorded, qd[future], 0.)
    return np.concatenate((dp, rot, v_local, w_local, q[future],
                           qd_future*joint_velocity_scale), axis=1).ravel()


@functools.cache
def policy_class(math_api):
    class SuperMimicTrackingPolicy(math_api.WbcTrackingPolicy):
        """SDK policy with the SuperMimic observation; SDK state/decode unchanged."""

        def __init__(self, contract, profile, *, past_frames, tail_frames):
            super().__init__(contract, profile)
            if type(past_frames) is not int or type(tail_frames) is not int or past_frames < 0 or tail_frames < 0:
                raise ValueError('invalid SuperMimic reference window')
            self.past_frames, self.tail_frames = past_frames, tail_frames

        def goal(self, motion, frame):
            """Goal from a window [frame-past, frame+K+tail] of the reference."""
            c, m = self.contract, math_api
            n = len(motion.q)
            lo, hi = max(0, frame-self.past_frames), min(n-1, frame+c.future_frames+self.tail_frames)
            span = slice(lo, hi+1)
            quat = motion.quat[span]
            qd, v, w = reference_velocities(m, motion.base[span], quat, motion.q[span], c.period)
            R = np.stack([m.rotation(x) for x in quat])
            # Goals past the window end are held only when the motion itself ends.
            return goal_features(motion.base[span], R, v, w, motion.q[span], qd, frame-lo,
                                 c.future_frames, c.joint_velocity_scale)

        def observe(self, motion, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            c, m = self.contract, math_api
            q, dq, gyro = _finite(q, 'q'), _finite(dq, 'dq'), _finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('observation state dimensions disagree')
            frame = int(np.clip(round(ref_time/c.period), 0, len(motion.q)-1))
            robot = m.rotation(quat)
            reference = m.rotation(m.multiply(m.yaw_quaternion(yaw_offset), motion.quat[frame]))
            current = np.concatenate((q-c.default, dq, gyro, -robot[2], self.last_action,
                                      (robot.T @ reference)[:, :2].ravel())).astype(np.float32)
            if not self.history:
                self.history.extend([current]*c.history_length)
            else:
                self.history.append(current)
            obs = np.concatenate((*self.history, self.goal(motion, frame))).astype(np.float32)
            if obs.shape != (c.input_dim,) or not np.isfinite(obs).all():
                raise ValueError('constructed observation is invalid')
            return obs

    return SuperMimicTrackingPolicy


class NativeModel:
    """SDK 0.2.1 native ONNX session with SuperMimic metadata (mirrors SDK Model)."""

    def __init__(self, path, math_api, threads=1):
        session_type = importlib.import_module(math_api.__name__.rpartition('.')[0]+'._onnx').Session
        if threads < 1:
            raise ValueError('threads must be positive')
        self.session = session_type(str(path), intra_threads=threads, inter_threads=1)
        self.contract = SuperMimicContract.from_metadata(self.session.metadata)
        inputs, outputs = self.session.inputs, self.session.outputs
        if (len(inputs) != 1 or len(outputs) != 1
                or inputs[0].shape != [1, self.contract.input_dim] or outputs[0].shape != [1, 28]
                or inputs[0].dtype != 'tensor(float)' or outputs[0].dtype != 'tensor(float)'):
            raise ValueError('ONNX IO does not match its tracking metadata')
        self.input, self.output = inputs[0].name, outputs[0].name

    def infer(self, obs):
        obs = np.asarray(obs, np.float32)
        if obs.shape != (self.contract.input_dim,) or not np.isfinite(obs).all():
            raise ValueError('invalid model input')
        out = self.session.run(self.input, self.output, obs[None])
        if out.shape != (1, 28) or not np.isfinite(out).all():
            raise ValueError('invalid ONNX output')
        return out[0]


class SuperMimicReference:
    """Causal Pico window: past filter context, current frame, goals and a derivative tail.

    The current frame lags the newest sample by (K+tail) periods. Past frames
    only feed the reference velocity filters, so they add no latency.
    """
    def __init__(self, contract, math_api, *, past_frames, tail_frames, capacity=1024):
        if not is_supermimic(contract) or capacity < 3:
            raise ValueError('Invalid SuperMimic reference settings')
        self.contract, self.math = contract, math_api
        self.samples = deque(maxlen=capacity)
        self.last_stamp_ns = self.origin_ns = self.last_receive_monotonic = None
        self.current_index = past_frames
        self.steps = past_frames+contract.future_frames+tail_frames+1
        self.ref_time = past_frames*contract.period
        self.delay = (contract.future_frames+tail_frames)*contract.period
        self.span = (self.steps-1)*contract.period

    def append(self, stamp_ns, q, root, *, received_monotonic, base=None):
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, (int, np.integer)) or stamp_ns <= 0:
            raise ValueError('Pico timestamp must be positive integer nanoseconds')
        q = np.asarray(q, dtype=np.float64)
        base = np.asarray(base, dtype=np.float64) if base is not None else None
        root = self.math.quaternion(root)
        if (q.shape != (28,) or not np.isfinite(q).all() or base is None or base.shape != (3,)
                or not np.isfinite(base).all() or not math.isfinite(received_monotonic)):
            raise ValueError('Invalid retargeted sample; SuperMimic needs the GMR root position')
        if self.last_stamp_ns is not None and int(stamp_ns) <= self.last_stamp_ns:
            raise ValueError('Pico timestamps must strictly increase')
        if self.origin_ns is None:
            self.origin_ns = int(stamp_ns)
        self.samples.append(((int(stamp_ns)-self.origin_ns)*1e-9, q.copy(), base.copy(), root.copy()))
        # Keep one sample at or before the window start; older ones never
        # affect motion(), and dropping them keeps it O(window) on the worker.
        start = self.samples[-1][0]-self.span
        while len(self.samples) > 2 and self.samples[1][0] <= start:
            self.samples.popleft()
        self.last_stamp_ns = int(stamp_ns)
        self.last_receive_monotonic = float(received_monotonic)

    def ready(self):
        return len(self.samples) >= 2 and self.samples[-1][0]-self.samples[0][0] >= self.span-1e-9

    def motion(self):
        if not self.ready():
            raise RuntimeError('Waiting for complete causal actor reference window')
        times = np.array([s[0] for s in self.samples])
        query = times[-1]-self.span+np.arange(self.steps)*self.contract.period
        index = np.clip(np.searchsorted(times, query, side='right')-1, 0, len(times)-2)
        alpha = np.clip((query-times[index])/(times[index+1]-times[index]), 0., 1.)
        # Linear interpolation, clamped at both ends exactly like np.interp.
        def interp(k):
            values = np.stack([s[k] for s in self.samples])
            return values[index]+alpha[:, None]*(values[index+1]-values[index])
        roots = [self.math.slerp(self.samples[i][3], self.samples[i+1][3], float(a))
                 for i, a in zip(index, alpha)]
        return self.math.Motion(interp(1), interp(2), roots)


def reference_window(config):
    window = config.get('reference_window', {})
    past, tail = window.get('past_frames'), window.get('tail_frames')
    if type(past) is not int or type(tail) is not int or not 0 <= past <= 64 or not 0 <= tail <= 16:
        raise ValueError('SuperMimic configuration needs integer reference_window past/tail frames')
    return past, tail


def create_policy(contract, profile, math_api, config):
    if not is_supermimic(contract):
        return math_api.WbcTrackingPolicy(contract, profile)
    past, tail = reference_window(config)
    return policy_class(math_api)(contract, profile, past_frames=past, tail_frames=tail)


def create_reference(contract, math_api, config):
    if not is_supermimic(contract):
        from .actor_pico_reference import CausalActorReference
        return CausalActorReference(contract, math_api)
    past, tail = reference_window(config)
    return SuperMimicReference(contract, math_api, past_frames=past, tail_frames=tail)
