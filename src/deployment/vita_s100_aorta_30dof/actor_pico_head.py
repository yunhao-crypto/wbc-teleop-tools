"""Pico head articulation composed with a 28-action policy's Body31 command.

No transport or GMR changes. The caller commits state only after publication;
discarded candidates must not advance calibration, filtering or slew limits.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial.transform import Rotation


NECK_NAMES = ('neck_yaw_joint', 'neck_pitch_joint')


@dataclass(frozen=True)
class HeadSettings:
    limits: tuple
    max_velocity: float
    filter_tau: float


def head_settings(config):
    raw = config.get('head_control')
    if raw is None:
        return None
    keys = {'enabled', 'joint_limits', 'max_velocity_rad_s', 'filter_time_constant_s'}
    if (not isinstance(raw, dict) or set(raw) != keys
            or type(raw['enabled']) is not bool):
        raise ValueError('Invalid head_control settings')
    limits = raw['joint_limits']
    if not isinstance(limits, dict) or set(limits) != set(NECK_NAMES):
        raise ValueError('head_control.joint_limits must name both neck joints')
    pairs = []
    for name in NECK_NAMES:
        pair = np.asarray(limits[name], dtype=float)
        if (pair.shape != (2,) or not np.isfinite(pair).all()
                or not -math.pi < pair[0] < 0 < pair[1] < math.pi):
            raise ValueError('Invalid head_control joint limits: '+name)
        pairs.append(tuple(pair))
    values = [raw[k] for k in ('max_velocity_rad_s', 'filter_time_constant_s')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Head velocity and filter time constant must be positive')
    return HeadSettings(tuple(pairs), *map(float, values)) if raw['enabled'] else None


@dataclass(frozen=True)
class HeadState:
    q: np.ndarray
    filtered: np.ndarray
    angles: np.ndarray
    neutral: np.ndarray | None
    tracking: bool
    stamp: int | None
    now: float


class PicoHeadControl:
    def __init__(self, settings, chest_offset_wxyz, *, period, hold_age, maximum_age):
        self.settings = settings
        # GMR uses R_robot_chest = C_unity_to_world R_pico_chest O.
        # The common world conversion C cancels in the head/chest relative pose.
        # O maps robot-local axes into the Pico skeleton's local frame.
        q = np.asarray(chest_offset_wxyz, dtype=float)
        self.offset = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        self.period, self.hold_age, self.maximum_age = period, hold_age, maximum_age
        self.hardware_limits = None
        self.reset()

    def reset(self):
        self.state = None

    def bind_body_info(self, body):
        limits = []
        for name, configured in zip(NECK_NAMES, self.settings.limits):
            cap = body.joints[body.joint_names.index(name)]
            if not cap.populated or cap.is_mock or cap.valid_limits != 31:
                raise ValueError('Head control needs physical neck command limits: '+name)
            pair = (max(configured[0], cap.q_min), min(configured[1], cap.q_max))
            if not pair[0] < 0 < pair[1]:
                raise ValueError('Head limits must include neutral: '+name)
            limits.append((cap.q_min, cap.q_max))
        self.hardware_limits = np.asarray(limits, dtype=float)

    @staticmethod
    def relative_rotation(body):
        body = np.asarray(body, dtype=float)
        if body.shape != (24, 7) or not np.isfinite(body).all():
            raise ValueError('Head control requires finite Pico [24,7] body poses')
        quats = body[[9, 15], 3:7]  # Spine3 and Head, SDK xyzw.
        norms = np.linalg.norm(quats, axis=1)
        if np.any(norms < .5) or np.any(norms > 1.5):
            raise ValueError('Invalid Pico chest/head quaternion')
        chest, head = Rotation.from_quat(quats).as_matrix()
        return chest.T @ head

    def preview(self, command, order, *, now, measured_q=None, snapshot=None, tracking=False):
        """Return a new command, diagnostics and candidate state without mutation."""
        if not math.isfinite(now):
            raise ValueError('Invalid head control clock')
        slots = [order.index(n) for n in NECK_NAMES]
        previous = self.state
        if previous is not None and now < previous.now:
            raise ValueError('Head control clock regressed')
        dt = self.period if previous is None else min(now-previous.now, self.period)
        if previous is not None:
            q = previous.q.copy()
        elif measured_q is None:
            q = np.array([command.joints[i].q for i in slots])
        else:
            q = np.asarray(measured_q, dtype=float)[slots]
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError('Invalid measured neck pose')
        if self.hardware_limits is not None:
            q = np.clip(q, self.hardware_limits[:, 0], self.hardware_limits[:, 1])
        limits = np.asarray(self.settings.limits)
        if self.hardware_limits is not None:
            limits = np.stack((np.maximum(limits[:, 0], self.hardware_limits[:, 0]),
                               np.minimum(limits[:, 1], self.hardware_limits[:, 1])), axis=1)
        filtered = q.copy() if previous is None else previous.filtered.copy()
        neutral = previous.neutral if previous is not None and previous.tracking and tracking else None
        angles = previous.angles.copy() if previous is not None and neutral is not None else np.zeros(2)
        stamp = previous.stamp if previous is not None and previous.tracking and tracking else None
        desired = np.zeros(2)
        age, calibrated = None, False
        mode = 'return_to_neutral'
        if tracking:
            if snapshot is None:
                raise ValueError('Head tracking requires a Pico snapshot')
            age = now-float(snapshot['received_monotonic'])
            if not 0 <= age <= self.maximum_age:
                raise TimeoutError('Pico head input is stale/future')
            current_stamp = snapshot.get('pico_timestamp_ns')
            if type(current_stamp) is not int or current_stamp <= 0:
                raise ValueError('Head tracking requires a positive Pico timestamp')
            if stamp is not None and current_stamp < stamp:
                raise ValueError('Pico head timestamp regressed')
            relative = self.relative_rotation(snapshot.get('raw_body_poses'))
            stamp = current_stamp
            if age > self.hold_age:
                # Freeze the emitted pose as well as the filter; do not keep
                # chasing the last target after the body reference starts hold.
                mode = 'hold'
                filtered = q.copy()
                desired = q.copy()
            else:
                if neutral is None:
                    neutral = relative.copy()
                    calibrated = True
                delta = self.offset.T @ relative @ neutral.T @ self.offset
                forward = delta[:, 0]
                horizontal = math.hypot(forward[0], forward[1])
                yaw = math.atan2(forward[1], forward[0]) if horizontal > 1e-6 else angles[0]
                # Keep saturation on the same side across the +/-pi seam.
                yaw = angles[0]+math.remainder(yaw-angles[0], 2*math.pi)
                angles = np.array([yaw, math.atan2(-forward[2], horizontal)])
                desired = np.clip(angles, limits[:, 0], limits[:, 1])
                mode = 'tracking'
        if mode != 'hold':
            alpha = -math.expm1(-dt/self.settings.filter_tau)
            filtered = filtered+alpha*(desired-filtered)
            step = self.settings.max_velocity*dt
            q = q+np.clip(filtered-q, -step, step)
        joints = list(command.joints)
        for slot, value in zip(slots, q):
            # Preserve profile PD gains; the head has no feedforward torque.
            joints[slot] = replace(joints[slot], q=float(value), dq=0., tau=0.)
        candidate = HeadState(q, filtered, angles, neutral, tracking, stamp, now)
        diagnostics = dict(mode=mode, calibrated=calibrated, pico_age_s=age,
                           pico_timestamp_ns=stamp, joint_names=list(NECK_NAMES),
                           target_q=q.tolist(), desired_q=desired.tolist(),
                           effective_limits=limits.tolist())
        return replace(command, joints=tuple(joints)), diagnostics, candidate

    def commit(self, candidate):
        self.state = candidate
