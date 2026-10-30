"""GR00T/SONIC Vita02A actor with either ankle control space.

The ONNX ``groot_wbc.tracking_contract.v1`` metadata names the ankle control
space; nothing else selects it. ``virtual_pitch_roll`` (serial-ankle plant)
returns the SDK ``Contract`` and ``WbcTrackingPolicy`` unchanged.
``motor_theta`` (closed-loop plant) keeps the SDK layout checks and term-major
observation, but the ankle slots of the proprioception history and of the
future joint references hold the two Achilles motor angles, as in training
(``_policy_dof_pos``/``_policy_dof_vel``: IK of the reference pose, rates
through the Jacobian at that pose). Targets are decoded like SuperMimic's
motor_theta actor (``decode_motor_theta``). The robot's EXTERNAL ankle mode is
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
    # Exports may carry soft limits inside the motor travel; decode clamps
    # motor targets to them, the robot to its own motor caps.
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
def policy_class(math_api):
    class GrootMotorThetaPolicy(math_api.WbcTrackingPolicy):
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
    return math_api.WbcTrackingPolicy(contract, profile)
