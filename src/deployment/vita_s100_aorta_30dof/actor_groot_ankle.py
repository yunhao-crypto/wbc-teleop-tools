"""GR00T/SONIC Vita02A actor with either ankle control space.

The ONNX ``groot_wbc.tracking_contract.v1`` metadata names the ankle control
space; nothing else selects it. ``virtual_pitch_roll`` (serial-ankle plant)
uses the SDK ``Contract`` and observation builder with unclipped PD targets.
``motor_theta`` (closed-loop plant) keeps the SDK layout checks and term-major
observation, but the ankle slots of the proprioception history and of the
future joint references hold the two Achilles motor angles, as in training
(``_policy_dof_pos``/``_policy_dof_vel``: IK of the reference pose, rates
through the Jacobian at that pose). The shared ``decode_motor_theta`` decoder
preserves SONIC's affine targets without position clipping. The EXTERNAL ankle mode is
never changed here; the live runner refuses a mismatched one.
"""
from dataclasses import dataclass, fields
import functools
import json
import math

import numpy as np

from .actor_parallel_ankle import MotorThetaAnkle, decode_motor_theta

CONTRACT_KEY = 'groot_wbc.tracking_contract.v1'
POLICY_KIND = 'groot_vita02a_tracking'


@dataclass(frozen=True)
class GrootMotorThetaContract:
    """SDK ``Contract`` fields with virtual default/limits plus the policy (motor) space."""
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
    include_z: bool
    policy_default: np.ndarray
    policy_lower: np.ndarray
    policy_upper: np.ndarray
    ankle: MotorThetaAnkle
    policy_kind: str = POLICY_KIND


def ankle_control_space(metadata):
    if CONTRACT_KEY not in metadata:
        raise ValueError(f'model is missing {CONTRACT_KEY}')
    return json.loads(metadata[CONTRACT_KEY]).get('ankle', {}).get('control_space')


def contract_from_metadata(math_api, metadata):
    """SDK contract for virtual ankles; ``GrootMotorThetaContract`` for motor_theta."""
    if ankle_control_space(metadata) != 'motor_theta':
        return math_api.Contract.from_metadata(metadata)
    raw = json.loads(metadata[CONTRACT_KEY])
    ankle = MotorThetaAnkle(raw['joint_names'], raw['ankle'])
    # Every other layout, timing and array check is the SDK's own.
    serial = dict(raw, ankle=dict(control_space='virtual_pitch_roll'))
    base = math_api.Contract.from_metadata({CONTRACT_KEY: json.dumps(serial)})
    slots = ankle.slots
    low, high = ankle.motor_limits
    # Exports may carry soft limits inside motor travel. They describe the
    # training penalty region, not a bound on commanded PD setpoints.
    if np.any(base.lower[slots] < low-1e-9) or np.any(base.upper[slots] > high+1e-9):
        raise ValueError('motor_theta ankle slot limits must lie within the motor limits')
    virtual_default, residual = ankle.to_virtual(base.default)
    if residual > 1e-8:
        raise ValueError('motor_theta default ankle pose has no forward-kinematics solution')
    # Virtual ankle bounds come from the robot profile's joint_limits.
    virtual_lower, virtual_upper = base.lower.copy(), base.upper.copy()
    virtual_lower[slots], virtual_upper[slots] = -math.pi/2, math.pi/2
    values = {f.name: getattr(base, f.name) for f in fields(base)}
    values.update(default=virtual_default, lower=virtual_lower, upper=virtual_upper,
                  policy_default=base.default, policy_lower=base.lower, policy_upper=base.upper,
                  ankle=ankle)
    return GrootMotorThetaContract(**values)


def is_groot_motor_theta(contract):
    return isinstance(contract, GrootMotorThetaContract)


@functools.cache
def serial_policy_class(math_api):
    class GrootPositionPolicy(math_api.WbcTrackingPolicy):
        """Action clipping followed by affine PD setpoints without position clipping."""

        clip_target_positions = False

        def decode(self, raw_action, **timing):
            c = self.contract
            action = np.asarray(raw_action, dtype=np.float64)
            if action.shape != (28,) or not np.isfinite(action).all():
                raise ValueError('model output must have 28 finite actions')
            self.last_action = np.clip(action, -c.action_clip, c.action_clip).astype(np.float32)
            target = c.default+c.scale*self.last_action.astype(np.float64)
            motors = [math_api.BodyJointCmd() for _ in range(31)]
            for p, slot in enumerate(self.mapping):
                motors[slot] = math_api.BodyJointCmd(float(np.float32(target[p])), 0., 0.,
                                                     float(np.float32(c.kp[p])), float(np.float32(c.kd[p])))
            for name, fixed in self.profile['fixed_joints'].items():
                motors[self.order.index(name)] = math_api.BodyJointCmd(
                    fixed['q'], 0., 0., fixed['kp'], fixed['kd'])
            return math_api.BodyCmd(joints=tuple(motors), **timing), {
                'action': self.last_action.tolist(), 'raw_target': target.tolist(),
                'target': target.tolist(), 'tau_ff': np.zeros(28).tolist(),
                'clamped': np.zeros(28, dtype=bool).tolist(),
            }

    return GrootPositionPolicy


@functools.cache
def policy_class(math_api):
    class GrootMotorThetaPolicy(serial_policy_class(math_api)):
        """SDK G1 observation with motor-angle ankle slots; motor_theta decode."""

        def observe(self, motion, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            c, m = self.contract, math_api
            q, dq, gyro = m.finite(q, 'q'), m.finite(dq, 'dq'), m.finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('observation state dimensions disagree')
            quat = m.quaternion(quat)
            indices = motion.indices(ref_time, c.future_frames, c.future_dt)
            reference = motion.q[indices]
            parts = [c.ankle.to_policy(reference).ravel(),
                     c.ankle.velocity_to_policy(reference, motion.velocities(indices)).ravel()]
            if c.include_z:
                parts.append(motion.base[indices, 2])
            offset = m.yaw_quaternion(yaw_offset)
            ori = [m.rotation(m.multiply(m.conjugate(quat), m.multiply(offset, motion.quat[i])))[:, :2].ravel()
                   for i in indices]
            parts.append(np.asarray(ori).ravel())
            terms = tuple(np.asarray(x, np.float32) for x in (
                gyro, c.ankle.to_policy(q)-c.policy_default, c.ankle.velocity_to_policy(q, dq),
                self.last_action.copy(), m.rotation(quat).T @ [0., 0., -1.]))
            if not self.history:
                self.history.extend([terms]*c.history_length)
            else:
                self.history.append(terms)
            parts.extend(np.concatenate([h[k] for h in self.history]) for k in range(5))
            obs = np.concatenate(parts).astype(np.float32)
            if obs.shape != (c.input_dim,) or not np.isfinite(obs).all():
                raise ValueError('constructed observation is invalid')
            return obs

        def decode(self, raw_action, **timing):
            return decode_motor_theta(self, math_api, raw_action, **timing)

    return GrootMotorThetaPolicy


def create_policy(contract, profile, math_api):
    if is_groot_motor_theta(contract):
        return policy_class(math_api)(contract, profile)
    return serial_policy_class(math_api)(contract, profile)
