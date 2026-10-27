"""Calibrated orientation IK and causal references for actor_g1 shadow inference.

This is a new kinematic retargeter, not the trained HoloMotion SMPL frontend.
It matches segment rotations, not contacts, balance or Cartesian foot targets.
Only NumPy and the existing Pico coordinate converter are required on S100.
"""
from collections import deque
import math
import xml.etree.ElementTree as ET

import numpy as np

from .onboard_preprocessing import CPUPicoToSmplConverter, COMMAND, _axis_quat, _mul


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross)


def matrix_quaternion(r):
    """SO(3) matrix to WXYZ, including rotations close to pi."""
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(1+tr)*2
        q = [s/4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
             (r[1, 0]-r[0, 1])/s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(max(1+r[i, i]-r[j, j]-r[k, k], 0.))*2
        q = np.empty(4)
        q[0] = (r[k, j]-r[j, k])/s
        q[i+1] = s/4
        q[j+1] = (r[j, i]+r[i, j])/s
        q[k+1] = (r[k, i]+r[i, k])/s
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def rotation_vector(r):
    q = matrix_quaternion(r)
    if q[0] < 0:
        q = -q
    mag = np.linalg.norm(q[1:])
    return q[1:]*(2 if mag < 1e-10 else 2*math.atan2(mag, q[0])/mag)


class OrientationKinematics:
    """URDF rotations and spatial joint axes; mesh files are not loaded."""
    def __init__(self, path, names):
        self.names = tuple(names)
        index = {n: i for i, n in enumerate(names)}
        document = ET.parse(path).getroot()
        links = {x.attrib['name'] for x in document.findall('link')}
        self.joints = []
        children = set()
        for joint in document.findall('joint'):
            kind = joint.attrib['type']
            if kind not in ('fixed', 'revolute', 'continuous'):
                raise ValueError('Unsupported retarget URDF joint: '+kind)
            parent = joint.find('parent').attrib['link']
            child = joint.find('child').attrib['link']
            if child in children:
                raise ValueError('URDF child has multiple parents')
            children.add(child)
            origin = joint.find('origin')
            rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            if rpy.shape != (3,) or not np.isfinite(rpy).all():
                raise ValueError('Invalid URDF RPY')
            r = (axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1])
                 @ axis_rotation([1, 0, 0], rpy[0]))
            node = joint.find('axis')
            axis = np.fromstring(node.attrib['xyz'] if node is not None else '1 0 0', sep=' ')
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
                raise ValueError('Invalid URDF axis')
            axis /= np.linalg.norm(axis)
            self.joints.append((parent, child, index.get(joint.attrib['name']), r, axis))
        roots = links-children
        if len(roots) != 1:
            raise ValueError('Expected a single-root URDF')
        self.root = roots.pop()
        if not set(names) <= {j.attrib['name'] for j in document.findall('joint')}:
            raise ValueError('URDF does not contain every actor joint')
        pending, ordered, known = self.joints, [], {self.root}
        while pending:
            ready = [j for j in pending if j[0] in known]
            if not ready:
                raise ValueError('Disconnected/cyclic URDF')
            ordered.extend(ready)
            known.update(j[1] for j in ready)
            pending = [j for j in pending if j[1] not in known]
        self.joints = ordered
        self.links = links
        self.constants = []
        for parent, child, col, origin, axis in ordered:
            x, y, z = axis
            cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            self.constants.append((parent, child, col, origin, origin @ axis,
                                   origin @ cross, origin @ (cross @ cross)))

    def forward(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError('Invalid retarget joint vector')
        rotations = {self.root: np.eye(3)}
        jacobians = {self.root: np.zeros((3, len(q)))}
        sin_q, one_minus_cos = np.sin(q), 1-np.cos(q)
        for parent, child, col, origin, axis, cross, cross2 in self.constants:
            parent_r = rotations[parent]
            jac = jacobians[parent].copy()
            if col is not None:
                jac[:, col] = parent_r @ axis
                r = parent_r @ (origin + sin_q[col]*cross + one_minus_cos[col]*cross2)
            else:
                r = parent_r @ origin
            rotations[child], jacobians[child] = r, jac
        return rotations, jacobians


class PicoOrientationRetargeter:
    """First valid frame calibrates segment offsets against actor default pose."""
    def __init__(self, *, urdf, contract, mapping, math_api, iterations=12):
        self.math = math_api
        self.contract = contract
        self.kinematics = OrientationKinematics(urdf, contract.names)
        self.mapping = tuple((x['robot_link'], int(x['pico_joint'])) for x in mapping)
        if (len({r for r, _ in self.mapping}) != len(self.mapping)
                or any(r not in self.kinematics.links or not 0 <= i < 24 for r, i in self.mapping)):
            raise ValueError('Invalid Pico/robot segment mapping')
        # Every action coordinate must affect at least one measured segment.
        _, jac = self.kinematics.forward(contract.default)
        stacked = np.concatenate([jac[r] for r, _ in self.mapping])
        if np.any(np.linalg.norm(stacked, axis=0) < 1e-8):
            raise ValueError('Retarget mapping leaves an actor joint unobserved')
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError('Invalid IK iteration budget')
        self.converter = CPUPicoToSmplConverter()
        self.q = contract.default.copy()
        self.offsets = self.root_offset = None
        self.last_iterations = 0

    def human_rotations(self, body):
        from .onboard_preprocessing import PICO_PARENTS
        pose, _ = self.converter.to_device(body)
        local = _axis_quat(pose.reshape(24, 3))
        global_q = np.empty((24, 4), np.float32)
        global_q[0] = local[0]
        for i in range(1, 24):
            global_q[i] = _mul(global_q[PICO_PARENTS[i]], local[i])
        root = self.math.quaternion(_mul(global_q[0], COMMAND))
        root_inv = self.math.rotation(root).T
        relative = [root_inv @ self.math.rotation(q) for q in global_q]
        return relative, root

    def solve(self, body, *, robot_root=(1., 0., 0., 0.)):
        human, root = self.human_rotations(body)
        if self.offsets is None:
            robot, _ = self.kinematics.forward(self.contract.default)
            self.offsets = [human[i].T @ robot[r] for r, i in self.mapping]
            self.root_offset = self.math.multiply(self.math.quaternion(robot_root),
                                                 self.math.conjugate(root))
        targets = [human[i] @ offset for (_, i), offset in zip(self.mapping, self.offsets)]
        q = self.q.copy()
        for iteration in range(self.iterations):
            self.last_iterations = iteration+1
            rotations, jacobians = self.kinematics.forward(q)
            error = np.concatenate([rotation_vector(t @ rotations[r].T)
                                    for (r, _), t in zip(self.mapping, targets)])
            if np.max(np.abs(error)) < 1e-5:
                break
            jac = np.concatenate([jacobians[r] for r, _ in self.mapping])
            delta = np.linalg.solve(jac.T @ jac + 0.01*np.eye(len(q)), jac.T @ error)
            if np.max(np.abs(delta)) < 1e-5:
                break
            q = np.clip(q+np.clip(delta, -.2, .2), self.contract.lower, self.contract.upper)
        rotations, _ = self.kinematics.forward(q)
        residuals = [np.linalg.norm(rotation_vector(t @ rotations[r].T))
                     for (r, _), t in zip(self.mapping, targets)]
        worst = int(np.argmax(residuals))
        residual = residuals[worst]
        if not np.isfinite(residual):
            raise ValueError('Nonfinite Pico retarget residual')
        self.q = q
        root = self.math.multiply(self.root_offset, root)
        return q.copy(), self.math.quaternion(root), float(residual)


class CausalActorReference:
    """Delay real references until the complete future/derivative window exists.

    Timing comes from the loaded model. Ten references at 20 ms need a
    180 ms horizon plus a 20 ms derivative tail (200 ms total); the legacy
    100 ms spacing needs 920 ms. Neither value is measured end-to-end latency.
    No prediction, repeated terminal pose, or invented future velocity is used.
    """
    def __init__(self, contract, math_api, *, capacity=1024):
        if contract.include_z:
            raise ValueError('This Pico retargeter does not supply calibrated robot root height')
        self.contract, self.math = contract, math_api
        if capacity < 3:
            raise ValueError('Invalid reference buffering settings')
        self.samples = deque(maxlen=capacity)
        self.last_stamp_ns = None
        self.origin_ns = None
        self.stride = int(round(contract.future_dt/contract.period))
        self.steps = (contract.future_frames-1)*self.stride+2
        self.delay = (self.steps-1)*contract.period
        self.last_receive_monotonic = None

    def append(self, stamp_ns, q, root, *, received_monotonic):
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, (int, np.integer)) or stamp_ns <= 0:
            raise ValueError('Pico timestamp must be positive integer nanoseconds')
        q = np.asarray(q, dtype=np.float64)
        root = self.math.quaternion(root)
        if q.shape != (28,) or not np.isfinite(q).all() or not math.isfinite(received_monotonic):
            raise ValueError('Invalid retargeted sample')
        if self.last_stamp_ns is not None:
            gap = (int(stamp_ns)-self.last_stamp_ns)*1e-9
            if gap <= 0:
                raise ValueError('Pico timestamps must strictly increase')
        if self.origin_ns is None:
            self.origin_ns = int(stamp_ns)
        self.samples.append(((int(stamp_ns)-self.origin_ns)*1e-9, q.copy(), root.copy()))
        self.last_stamp_ns = int(stamp_ns)
        self.last_receive_monotonic = float(received_monotonic)

    def ready(self):
        return len(self.samples) >= 2 and self.samples[-1][0]-self.samples[0][0] >= self.delay-1e-9

    def motion(self):
        if not self.ready():
            raise RuntimeError('Waiting for complete causal actor reference window')
        times = np.array([s[0] for s in self.samples])
        query = times[-1]-self.delay+np.arange(self.steps)*self.contract.period
        positions = np.stack([s[1] for s in self.samples])
        q = np.stack([np.interp(query, times, positions[:, j]) for j in range(28)], axis=1)
        roots = []
        for t in query:
            i = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, len(times)-2))
            alpha = float(np.clip((t-times[i])/(times[i+1]-times[i]), 0., 1.))
            roots.append(self.math.slerp(self.samples[i][2], self.samples[i+1][2], alpha))
        return self.math.Motion(q, np.zeros((self.steps, 3)), roots)
