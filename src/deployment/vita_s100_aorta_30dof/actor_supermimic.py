"""SuperMimic Vita02A tracking actor on the SDK 0.2.1 Body31 interface.

The exported model states its observation/goal layout in
``supermimic.tracking_contract.v1`` metadata. The SDK's quaternion math,
feedback validation, Body31 mapping and joint limits are reused unchanged;
the observation builder differs from the G1 actor.

Two ankle contracts exist. ``virtual_pitch_roll`` (serial-ankle plant) reuses
the SDK decode, including its ankle clamp feedforward. ``motor_theta``
(closed-loop plant) observes and commands the Achilles motor angles in the
ankle slots: measured state and references are mapped to motor space, and the
motor-angle targets are mapped back to virtual pitch/roll through the forward
kinematics, because Body31 EXTERNAL takes only virtual targets. The robot's
motor PD (EXTERNAL ``ankle_virtual_pd_intent=false``) then re-solves the same
IK, which reproduces the trained motor PD inside the virtual ankle box. Outside
it the robot clamps the virtual target first; with ``ankle_clamp_feedforward``
the motor PD torque that clamp removes is sent back as virtual ankle torque.
"""
from dataclasses import dataclass
import functools
import importlib
import json
import math
from collections import deque

import numpy as np

from .actor_parallel_ankle import MotorThetaAnkle

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
    # Policy joint space. Equal to default/lower/upper for virtual ankles;
    # motor_theta contracts hold motor angles and motor limits in the ankle
    # slots, while default/lower/upper above stay virtual pitch/roll.
    policy_default: np.ndarray = None
    policy_lower: np.ndarray = None
    policy_upper: np.ndarray = None
    ankle: MotorThetaAnkle = None

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
        space = raw.get('ankle', {}).get('control_space')
        if space == 'virtual_pitch_roll':
            return cls(names, default, scale, kp, kd, lower, upper, clip, period, count, period,
                       history, raw['input_dim'], frame_dim, goal_dim, float(goal['joint_velocity_scale']),
                       policy_default=default, policy_lower=lower, policy_upper=upper)
        if space != 'motor_theta':
            raise ValueError('Body31 tracking requires virtual pitch/roll or motor_theta ankles')
        ankle = MotorThetaAnkle(names, raw['ankle'])
        slots = ankle.slots
        if (not np.allclose(lower[slots], ankle.motor_limits[0], rtol=0, atol=1e-9)
                or not np.allclose(upper[slots], ankle.motor_limits[1], rtol=0, atol=1e-9)):
            raise ValueError('motor_theta ankle slot limits must equal the motor limits')
        virtual_default, residual = ankle.to_virtual(default)
        if residual > 1e-8:
            raise ValueError('motor_theta default ankle pose has no forward-kinematics solution')
        # Virtual ankle bounds come from the robot profile's joint_limits.
        virtual_lower, virtual_upper = lower.copy(), upper.copy()
        virtual_lower[slots], virtual_upper[slots] = -math.pi/2, math.pi/2
        return cls(names, virtual_default, scale, kp, kd, virtual_lower, virtual_upper, clip, period, count,
                   period, history, raw['input_dim'], frame_dim, goal_dim, float(goal['joint_velocity_scale']),
                   policy_default=default, policy_lower=lower, policy_upper=upper, ankle=ankle)


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


def goal_features(base, R, v, w, q, qd, frame, count, joint_velocity_scale=.1, ankle=None):
    """Training MotionLib.goal for one frame of a [N] reference.

    With a motor_theta ``ankle`` the joint columns go to policy space as in
    training's PolicyJointGoals (rates through the Jacobian at the same frame).
    """
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
    q_future, qd_future = q[future], np.where(recorded, qd[future], 0.)*joint_velocity_scale
    if ankle is not None:
        q_future, qd_future = ankle.to_policy(q_future), ankle.velocity_to_policy(q_future, qd_future)
    return np.concatenate((dp, rot, v_local, w_local, q_future, qd_future), axis=1).ravel()


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
                                 c.future_frames, c.joint_velocity_scale, c.ankle)

        def observe(self, motion, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            c, m = self.contract, math_api
            q, dq, gyro = _finite(q, 'q'), _finite(dq, 'dq'), _finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('observation state dimensions disagree')
            frame = int(np.clip(round(ref_time/c.period), 0, len(motion.q)-1))
            robot = m.rotation(quat)
            reference = m.rotation(m.multiply(m.yaw_quaternion(yaw_offset), motion.quat[frame]))
            if c.ankle is not None:
                q, dq = c.ankle.to_policy(q), c.ankle.velocity_to_policy(q, dq)
            current = np.concatenate((q-c.policy_default, dq, gyro, -robot[2], self.last_action,
                                      (robot.T @ reference)[:, :2].ravel())).astype(np.float32)
            if not self.history:
                self.history.extend([current]*c.history_length)
            else:
                self.history.append(current)
            obs = np.concatenate((*self.history, self.goal(motion, frame))).astype(np.float32)
            if obs.shape != (c.input_dim,) or not np.isfinite(obs).all():
                raise ValueError('constructed observation is invalid')
            return obs

        def decode(self, raw_action, **timing):
            c = self.contract
            if c.ankle is None:
                return super().decode(raw_action, **timing)
            action = _finite(raw_action, 'model action')
            if action.shape != (28,):
                raise ValueError('model output must have 28 actions')
            self.last_action = np.clip(action, -c.action_clip, c.action_clip).astype(np.float32)
            raw_q = c.policy_default+c.scale*self.last_action.astype(np.float64)
            # Motor angles stop at the motor travel, as in the vita-robot
            # passthrough; the robot clamps the virtual pose to its ankle box.
            motor_q = raw_q.copy()
            slots = c.ankle.slots
            motor_q[slots] = np.clip(raw_q[slots], c.policy_lower[slots], c.policy_upper[slots])
            virtual_q, residual = c.ankle.to_virtual(motor_q)
            target = np.clip(virtual_q, self.lower, self.upper)
            tau = np.zeros(28)
            if self.profile['ankle_clamp_feedforward']:
                # Motor angles the robot will track: IK of the clamped virtual
                # target, then its motor-limit clamp (vita-robot PD-A).
                tracked = np.clip(c.ankle.to_policy(target)[slots], *c.ankle.motor_limits)
                missing = np.zeros(28)
                missing[slots] = c.kp[slots]*(motor_q[slots]-tracked)
                tau[slots] = c.ankle.torque_to_virtual(target, missing)[slots]
            motors = [math_api.BodyJointCmd() for _ in range(31)]
            for p, slot in enumerate(self.mapping):
                motors[slot] = math_api.BodyJointCmd(float(np.float32(target[p])), 0., float(np.float32(tau[p])),
                                                     float(np.float32(c.kp[p])), float(np.float32(c.kd[p])))
            for name, fixed in self.profile['fixed_joints'].items():
                motors[self.order.index(name)] = math_api.BodyJointCmd(fixed['q'], 0., 0., fixed['kp'], fixed['kd'])
            clamped = target != virtual_q
            clamped[slots] |= motor_q[slots] != raw_q[slots]
            return math_api.BodyCmd(joints=tuple(motors), **timing), {
                'action': self.last_action.tolist(), 'raw_target': raw_q.tolist(),
                'motor_target': motor_q[slots].tolist(), 'ankle_fk_residual': residual,
                'target': target.tolist(), 'tau_ff': tau.tolist(), 'clamped': clamped.tolist(),
            }

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


def window_motion(math_api, times, q, base, quat, query):
    """Linear/slerp resampling at ``query``, clamped at both ends exactly like np.interp."""
    times = np.asarray(times, dtype=np.float64)
    index = np.clip(np.searchsorted(times, query, side='right')-1, 0, len(times)-2)
    alpha = np.clip((query-times[index])/(times[index+1]-times[index]), 0., 1.)
    def interp(values):
        values = np.asarray(values, dtype=np.float64)
        return values[index]+alpha[:, None]*(values[index+1]-values[index])
    roots = [math_api.slerp(quat[i], quat[i+1], float(a)) for i, a in zip(index, alpha)]
    return math_api.Motion(interp(q), interp(base), roots)


class SuperMimicReference:
    """Causal Pico window: past filter context, current frame, goals and a derivative tail.

    The current frame lags the window end by (K+tail) periods. Past frames
    only feed the reference velocity filters, so they add no latency.
    ``motion()`` ends the window at the newest sample; ``playback()`` ends it
    at a control-clock time, which needs ``history_s`` of extra samples.
    """
    def __init__(self, contract, math_api, *, past_frames, tail_frames, capacity=1024, history_s=0.):
        if not is_supermimic(contract) or capacity < 3 or not 0 <= history_s <= 2:
            raise ValueError('Invalid SuperMimic reference settings')
        self.contract, self.math = contract, math_api
        self.samples = deque(maxlen=capacity)
        self.last_stamp_ns = self.origin_ns = self.last_receive_monotonic = None
        self.current_index = past_frames
        self.steps = past_frames+contract.future_frames+tail_frames+1
        self.ref_time = past_frames*contract.period
        self.delay = (contract.future_frames+tail_frames)*contract.period
        self.span = (self.steps-1)*contract.period
        self.history = self.span+history_s

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
        start = self.samples[-1][0]-self.history
        while len(self.samples) > 2 and self.samples[1][0] <= start:
            self.samples.popleft()
        self.last_stamp_ns = int(stamp_ns)
        self.last_receive_monotonic = float(received_monotonic)

    def ready(self):
        return len(self.samples) >= 2 and self.samples[-1][0]-self.samples[0][0] >= self.span-1e-9

    def export(self):
        """Raw samples (source seconds since the first sample) for the control loop."""
        return dict(t=np.array([s[0] for s in self.samples]), q=np.stack([s[1] for s in self.samples]),
                    base=np.stack([s[2] for s in self.samples]), quat=np.stack([s[3] for s in self.samples]))

    def query(self, end):
        return end-self.span+np.arange(self.steps)*self.contract.period

    def motion(self):
        if not self.ready():
            raise RuntimeError('Waiting for complete causal actor reference window')
        s = self.export()
        return window_motion(self.math, s['t'], s['q'], s['base'], s['quat'], self.query(s['t'][-1]))

    def playback(self, stream, now, margin):
        """Window ending at the source time usable ``margin`` before local ``now``.

        The window advances with the control clock, not with sample arrival.
        Past the newest sample it holds that sample (``playback_lead_s`` < 0).
        """
        t = np.asarray(stream['t'], dtype=np.float64)
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError('Pico stream needs at least two increasing samples')
        q, base, quat = (np.asarray(stream[k], dtype=np.float64) for k in ('q', 'base', 'quat'))
        if q.shape != (len(t), 28) or base.shape != (len(t), 3) or quat.shape != (len(t), 4):
            raise ValueError('Pico stream sample dimensions disagree')
        offset = float(stream['clock_offset_s'])
        end = now-offset-margin
        if not math.isfinite(end):
            raise ValueError('Pico stream clock offset is invalid')
        motion = window_motion(self.math, t, q, base, quat, self.query(end))
        return motion, dict(playback_lead_s=float(t[-1]-end), playback_time_s=end-self.delay,
                            clock_offset_s=offset)


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
    from .actor_pico_stream import stream_settings
    past, tail = reference_window(config)
    settings = stream_settings(config)
    # Playback lags the newest sample by the margin and, before the hold
    # rule takes over, by up to pico_hold_age_s; keep samples for both.
    history = 0. if settings is None else settings.playback_margin_s+config['pico_hold_age_s']+.1
    return SuperMimicReference(contract, math_api, past_frames=past, tail_frames=tail, history_s=history)
