"""MorphMimic motion interface: canonical 29 slots, waist anchor, Body31 output.

Normalizers and the 29 -> 28 output selection are baked into the ONNX graph.
Feedback and references remain virtual joint coordinates, as in the SDK.
"""
from dataclasses import dataclass, fields
import functools
import json

import numpy as np

from .actor_groot_ankle import (GrootMotorThetaContract, contract_from_metadata as control_contract,
                              serial_policy_class)
from .actor_parallel_ankle import decode_motor_theta

CONTRACT_KEY = 'morphmimic.tracking_contract.v1'
POLICY_KIND = 'morphmimic_vita02a_tracking'


@dataclass(frozen=True)
class MorphMimicContract(GrootMotorThetaContract):
    canonical_indices: tuple = ()
    anchor_chain: tuple = ()
    nominal_root_height: float = .891088
    reference_feasibility_guard: bool = False
    policy_kind: str = POLICY_KIND

    @classmethod
    def from_metadata(cls, math_api, metadata):
        raw = json.loads(metadata[CONTRACT_KEY])
        guard = raw.get('reference_feasibility_guard')
        if guard == 'native_serial_and_coupled_motor_workspace_v2':
            from .actor_morphmimic_sonic import contract_from_metadata
            return contract_from_metadata(math_api, metadata)
        if guard not in (None, 'virtual_box_and_coupled_motor_v1'):
            raise ValueError('Unsupported MorphMimic reference feasibility guard')
        if (raw.get('schema_version') != 1 or raw.get('policy_kind') != POLICY_KIND
                or raw.get('goal_anchor') != 'reference' or raw.get('history_length') != 10
                or raw.get('input_dim') != 1600 or raw.get('future_frames') != 10
                or raw.get('velocity_rule') != 'virtual_gradient_then_motor_jacobian'
                or raw.get('canonical_indices') != [i for i in range(29) if i != 13]):
            raise ValueError('Unsupported MorphMimic layout or reference contract')
        height = float(raw['nominal_root_height'])
        if not np.isfinite(height) or not 0 < height < 2:
            raise ValueError('Invalid MorphMimic nominal root height')
        base = control_contract(math_api, {'groot_wbc.tracking_contract.v1': json.dumps(raw['control'])})
        if getattr(base, 'ankle', None) is None:
            raise ValueError('MorphMimic Vita02A export must use motor_theta')
        values = {f.name: getattr(base, f.name) for f in fields(base)}
        values.update(input_dim=1600, history_length=10, future_frames=10, include_z=False,
                      canonical_indices=tuple(raw['canonical_indices']),
                      anchor_chain=tuple(raw['anchor_chain']), nominal_root_height=height, policy_kind=POLICY_KIND,
                      reference_feasibility_guard=guard is not None)
        if [x['joint'] for x in raw['anchor_chain']] != ['waist_yaw_joint', 'waist_pitch_joint']:
            raise ValueError('Unsupported waist anchor chain')
        return cls(**values)


def axis_rotation(axis, theta):
    axis = np.asarray(axis, dtype=float)
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)


def anchor_pose(c, m, q, root_pos, root_quat):
    p, R = np.asarray(root_pos, dtype=float).copy(), m.rotation(root_quat)
    for joint in c.anchor_chain:
        p += R @ np.asarray(joint['pos'])
        R = R @ m.rotation(joint['quat'])
        pivot = np.asarray(joint['pivot'])
        dR = axis_rotation(joint['axis'], q[c.names.index(joint['joint'])])
        p += R @ (pivot-dR @ pivot)
        R = R @ dR
    return p, R


@functools.cache
def policy_class(math_api):
    class MorphMimicPolicy(serial_policy_class(math_api)):
        def canonical(self, values):
            result = np.zeros((*np.asarray(values).shape[:-1], 29))
            result[..., self.contract.canonical_indices] = values
            return result

        def observe(self, motion, ref_time, q, dq, quat, gyro, yaw_offset=0.):
            c, m = self.contract, math_api
            q, dq, gyro = m.finite(q, 'q'), m.finite(dq, 'dq'), m.finite(gyro, 'gyro')
            if q.shape != (28,) or dq.shape != (28,) or gyro.shape != (3,):
                raise ValueError('Invalid MorphMimic state dimensions')
            quat = m.quaternion(quat)
            current = np.concatenate((self.canonical(c.ankle.to_policy(q)-c.policy_default),
                self.canonical(c.ankle.velocity_to_policy(q, dq)), gyro,
                -m.rotation(quat)[2], self.canonical(self.last_action))).astype(np.float32)
            if not self.history:
                self.history.extend([current.copy() for _ in range(c.history_length)])
            else:
                self.history.append(current)
            frame = int(np.clip(round(ref_time/c.period), 0, len(motion.q)-1))
            ids = np.minimum(frame+np.arange(1, c.future_frames+1), len(motion.q)-1)
            reference_q = motion.q
            self.reference_ankle_projected_frames = 0
            if c.reference_feasibility_guard:
                reference_q, changed = c.ankle.project_reference(motion.q)
                self.reference_ankle_projected_frames = int(np.count_nonzero(changed))
            qd_ref = (np.gradient(reference_q, c.period, axis=0) if len(reference_q)>1
                      else np.zeros_like(reference_q))
            future_q = reference_q[ids]
            offset = m.yaw_quaternion(yaw_offset)
            offset_R = m.rotation(offset)
            roots = [m.multiply(offset, motion.quat[i]) for i in ids]
            bases = np.asarray(motion.base[ids]) @ offset_R.T
            heading = m.rotation(m.yaw_quaternion(m.yaw(roots[0])))
            origin = np.array([bases[0, 0], bases[0, 1], 0.])
            robot_R = anchor_pose(c, m, q, np.zeros(3), quat)[1]
            anchors = [anchor_pose(c, m, pose, base, root)
                       for pose, base, root in zip(future_q, bases, roots)]
            goal = np.concatenate((self.canonical(c.ankle.to_policy(future_q)),
                self.canonical(c.ankle.velocity_to_policy(future_q, qd_ref[ids])),
                np.stack([heading.T @ (p-origin) for p, _ in anchors]),
                np.stack([(robot_R.T @ R)[:, :2].ravel() for _, R in anchors])), axis=1)
            obs = np.concatenate((*self.history, goal.ravel())).astype(np.float32)
            if obs.shape != (c.input_dim,) or not np.isfinite(obs).all():
                raise ValueError('Invalid MorphMimic observation')
            return obs

        def decode(self, raw_action, **timing):
            command, row = decode_motor_theta(self, math_api, raw_action, **timing)
            row['reference_ankle_projected_frames'] = getattr(self, 'reference_ankle_projected_frames', 0)
            return command, row

    return MorphMimicPolicy
