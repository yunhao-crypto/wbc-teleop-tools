"""NumPy-only raw Pico SMPL24 -> calibrated stationary three-point targets.

No SMPL body model, robot SDK, controller, IK or ONNX is loaded here. The
pelvis-relative upright anatomical frame removes operator translation/yaw.
Robot leg references and all policy protection remain outside this front end.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'pico24_threepoint_delta_calibration_v1'
POINTS = ('left_op3_mount', 'right_op3_mount', 'head_capsule_center')
INDICES = (20, 21, 15)
AXES = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def qmatrix(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=-1)
    if not np.isfinite(q).all() or np.any(abs(norms-1) > .02):
        raise ValueError('Expected finite near-unit WXYZ quaternions')
    q = q/norms[..., None]
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def matrixq(matrix):
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.allclose(r.T@r, np.eye(3), atol=1e-6) or np.linalg.det(r) < .99999:
        raise ValueError('Expected a proper rotation matrix')
    trace = np.trace(r)
    if trace > 0:
        s = 2*np.sqrt(trace+1)
        q = np.array([s/4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(r))); j = (i+1)%3; k = (i+2)%3
        s = 2*np.sqrt(1+r[i, i]-r[j, j]-r[k, k]); q = np.empty(4)
        q[0] = (r[k, j]-r[j, k])/s; q[i+1] = s/4
        q[j+1] = (r[j, i]+r[i, j])/s; q[k+1] = (r[k, i]+r[i, k])/s
    return q/np.linalg.norm(q)


def slerp(a, b, fraction):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a/np.linalg.norm(a, axis=-1, keepdims=True); b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a*b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b); dot = np.clip(abs(dot), 0, 1)
    angle = np.arccos(dot); sin = np.sin(angle); f = float(fraction)
    first = np.divide(np.sin((1-f)*angle), sin, out=np.full_like(sin, 1-f), where=sin > 1e-7)
    second = np.divide(np.sin(f*angle), sin, out=np.full_like(sin, f), where=sin > 1e-7)
    q = first*a+second*b
    return q/np.linalg.norm(q, axis=-1, keepdims=True)


def anatomical_sites(body):
    """Raw world XYZ + XYZW -> pelvis-relative forward/left/up positions/rotations.

    All positions must really exist. Degenerate orientation-only fixtures are
    rejected rather than silently synthesizing a human skeleton.
    """
    b = np.asarray(body, dtype=float)
    if b.shape != (24, 7) or not np.isfinite(b).all():
        raise ValueError('Pico body requires finite [24,7] XYZ+XYZW')
    rotations = AXES@qmatrix(b[:, [6, 3, 4, 5]])
    positions = b[:, :3]@AXES.T
    y = positions[1]-positions[2]; y[2] = 0
    width = np.linalg.norm(y)
    if not .05 <= width <= .5:
        raise ValueError('Missing/degenerate hip positions: cannot use orientation-only Pico data as three-point XYZ')
    y /= width; z = np.array([0., 0., 1.]); x = np.cross(y, z)
    heading = np.column_stack((x, y, z))
    local = (positions[list(INDICES)]-positions[0])@heading
    if not .25 <= local[2, 2] <= 1.2:
        raise ValueError('Head/pelvis geometry is invalid for upright neutral human tracking')
    return local, heading.T@rotations[list(INDICES)]


class TimingGate:
    """Device and receive clocks are separate. Duplicate polling never refreshes age."""
    def __init__(self, max_gap_s=.1, stale_s=.6, clock_error_s=.25):
        self.max_gap_s = float(max_gap_s); self.stale_s = float(stale_s); self.clock_error_s = float(clock_error_s)
        if not 0 < self.max_gap_s <= self.stale_s or self.clock_error_s <= 0:
            raise ValueError('Invalid Pico timing budgets')
        self.last_stamp = self.last_rx = self.origin_stamp = self.origin_rx = None
        self.last_body = None; self.duplicates = 0

    def accept(self, stamp_ns, body, receive_ns):
        if (isinstance(stamp_ns, (bool, np.bool_)) or not isinstance(stamp_ns, (int, np.integer))
                or not 0 < int(stamp_ns) <= np.iinfo(np.int64).max):
            raise ValueError('Device timestamp must be positive int64 nanoseconds')
        if not isinstance(receive_ns, (int, np.integer)) or receive_ns <= 0:
            raise ValueError('Receive clock must be monotonic integer nanoseconds')
        stamp_ns, receive_ns = int(stamp_ns), int(receive_ns)
        if self.last_stamp is not None:
            if stamp_ns == self.last_stamp:
                if not np.array_equal(body, self.last_body):
                    raise ValueError('Body changed without an advancing same-packet timestamp')
                self.duplicates += 1; self.check_age(receive_ns, self.stale_s)
                return False
            dt = (stamp_ns-self.last_stamp)*1e-9; dr = (receive_ns-self.last_rx)*1e-9
            if dt <= 0 or dr <= 0: raise ValueError('Pico device/receive clock moved backwards or repeated')
            if dt > self.max_gap_s or dr > self.max_gap_s:
                raise ValueError('Pico frame gap exceeds100ms; do not interpolate or compress the gap')
            drift = (stamp_ns-self.origin_stamp)*1e-9-(receive_ns-self.origin_rx)*1e-9
            if abs(drift) > self.clock_error_s: raise ValueError('Device/receive clock progression disagrees')
        anatomical_sites(body)
        if self.origin_stamp is None: self.origin_stamp, self.origin_rx = stamp_ns, receive_ns
        self.last_stamp, self.last_rx = stamp_ns, receive_ns
        self.last_body = np.array(body, copy=True)
        return True

    def check_age(self, now_ns, maximum_s=None):
        if self.last_rx is None: raise ValueError('No Pico frame has been received')
        age = (int(now_ns)-self.last_rx)*1e-9
        if age < 0 or age > (self.stale_s if maximum_s is None else maximum_s):
            raise ValueError('Pico data is stale; repeated polling does not make it fresh')
        return age


def validate_raw(body, stamps, received):
    body = np.asarray(body); stamps = np.asarray(stamps); received = np.asarray(received)
    n = len(body)
    if body.shape != (n, 24, 7) or not 2 <= n <= 180000 or stamps.shape != (n,) or received.shape != (n,):
        raise ValueError('Invalid Pico recording arrays')
    if stamps.dtype.kind not in 'iu' or received.dtype.kind not in 'iu':
        raise ValueError('Raw timestamps must be integer nanoseconds')
    gate = TimingGate()
    for b, t, r in zip(body, stamps, received):
        if not gate.accept(int(t), b, int(r)): raise ValueError('Raw recording must contain unique frames')
    return (stamps.astype(np.int64)-int(stamps[0]))*1e-9


def load_raw(path, expected_sha256):
    if sha(path) != expected_sha256: raise ValueError('Raw recording SHA256 mismatch')
    with np.load(path, allow_pickle=False) as z:
        b, t, r = (z[k].copy() for k in ('body_poses', 'sample_times_ns', 'receive_monotonic_ns'))
        meta = json.loads(z['metadata'].item())
    if meta.get('schema') != 'pico_body24_record_v1' or not meta.get('complete'):
        raise ValueError('Need a complete raw Pico recording with provenance')
    seconds = validate_raw(b, t, r)
    return b, seconds, meta


def fit_calibration(body, times, standing_path, sites_path, *, neutral_seconds=2., scale=None):
    times = np.asarray(times, dtype=float)
    if neutral_seconds < .5 or len(times) != len(body) or times[-1] < neutral_seconds:
        raise ValueError('Need at least0.5s of explicit neutral recording')
    chosen = np.flatnonzero(times <= neutral_seconds)
    if len(chosen) < 15: raise ValueError('Too few neutral frames')
    values = [anatomical_sites(body[i]) for i in chosen]
    positions = np.array([v[0] for v in values]); rotations = np.array([v[1] for v in values])
    center = np.median(positions, axis=0)
    mean_rotations = []
    for point in range(3):
        u, _, v = np.linalg.svd(rotations[:, point].sum(axis=0)); r = u@v
        if np.linalg.det(r) < 0: u[:, -1] *= -1; r = u@v
        mean_rotations.append(r)
    mean_rotations = np.asarray(mean_rotations)
    position_p95 = np.quantile(np.linalg.norm(positions-center, axis=-1), .95, axis=0)
    rel = np.swapaxes(mean_rotations, -1, -2)@rotations
    angle = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1)-1)/2, -1, 1))
    angle_p95 = np.quantile(angle, .95, axis=0)
    if np.any(position_p95 > .02) or np.any(angle_p95 > np.radians(5)):
        raise ValueError('Neutral calibration was not still: require position p95<=2cm and orientation p95<=5deg')
    with np.load(standing_path, allow_pickle=False) as z:
        target_pos = z['positions_base_m'].copy(); target_q = z['quaternions_base_wxyz'].copy()
        root = z['reference_root_quat_world_wxyz'].copy()
    used_scale = float(target_pos[2, 2]/center[2, 2] if scale is None else scale)
    if not .25 <= used_scale <= 2.: raise ValueError('Human-to-robot scale outside reviewed geometric range')
    sites = json.loads(Path(sites_path).read_text())['sites']
    return dict(schema=SCHEMA, coordinate_space='reference_base_link', point_names=list(POINTS),
        raw_coordinate_convention='pico_world_xyz_m_xyzw', anatomical_frame='upright_hip_left_forward_zup_pelvis_relative',
        mode='neutral_delta_no_operator_translation_or_heading_command', scale=used_scale,
        scale_policy='neutral_head_height_ratio' if scale is None else 'explicit',
        human_neutral_positions=center.tolist(), human_neutral_rotations=mean_rotations.tolist(),
        robot_neutral_positions=target_pos.tolist(), robot_neutral_quaternions=target_q.tolist(),
        reference_root_wxyz=root.tolist(), wrist_mount_sites=sites[:2],
        head_policy='track_head_origin_delta_to_capsule_center_no_waist_lever_arm',
        neutral_seconds=neutral_seconds, neutral_frames=len(chosen),
        neutral_position_p95_m=position_p95.tolist(), neutral_orientation_p95_deg=np.degrees(angle_p95).tolist(),
        standing_reference_sha256=sha(standing_path), site_interface_sha256=sha(sites_path))


def rpy_matrix(v):
    x, y, z = map(float, v); cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


class CalibratedPicoThreePoint:
    def __init__(self, calibration):
        c = calibration
        if (c.get('schema') != SCHEMA or c.get('point_names') != list(POINTS)
                or c.get('anatomical_frame') != 'upright_hip_left_forward_zup_pelvis_relative'
                or c.get('mode') != 'neutral_delta_no_operator_translation_or_heading_command'):
            raise ValueError('Unsupported Pico three-point calibration')
        self.calibration = c; self.scale = float(c['scale'])
        if not .25 <= self.scale <= 2.: raise ValueError('Invalid calibrated scale')
        self.human_p = np.asarray(c['human_neutral_positions'], dtype=float)
        self.human_r = np.asarray(c['human_neutral_rotations'], dtype=float)
        self.robot_p = np.asarray(c['robot_neutral_positions'], dtype=float)
        self.robot_q = np.asarray(c['robot_neutral_quaternions'], dtype=float)
        self.robot_r = qmatrix(self.robot_q)
        if (self.human_p.shape != (3, 3) or self.human_r.shape != (3, 3, 3)
                or self.robot_p.shape != (3, 3) or self.robot_q.shape != (3, 4)
                or not all(np.isfinite(x).all() for x in (self.human_p, self.human_r, self.robot_p))):
            raise ValueError('Invalid calibration geometry')
        for r in self.human_r: matrixq(r)
        self.previous = self.robot_q.copy()

    def map(self, body):
        p, rotations = anatomical_sites(body)
        delta = rotations@np.swapaxes(self.human_r, -1, -2)
        desired_r = delta@self.robot_r
        desired_p = self.robot_p+self.scale*(p-self.human_p)
        # Wrist samples are joint origins; account for the target OP3 mount's
        # rotation-induced translation. Head is already a head origin, not waist.
        for i, mount in enumerate(self.calibration['wrist_mount_sites']):
            offset = np.asarray(mount['offset_m']); rm = rpy_matrix(mount['mount_rpy_rad'])
            desired_p[i] += (desired_r[i]@rm.T-self.robot_r[i]@rm.T)@offset
        result_q = np.asarray([matrixq(r) for r in desired_r])
        result_q = np.where(np.sum(result_q*self.previous, axis=1)[:, None] < 0, -result_q, result_q)
        self.previous = result_q.copy()
        return desired_p.astype(np.float32), result_q.astype(np.float32)


def convert_recording(body, times, calibration, *, startup_s=.8, blend_s=.8):
    """Offline interpolation only; never bridges missing/bad clock intervals."""
    t = np.asarray(times, dtype=float)
    if len(t) != len(body) or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .1:
        raise ValueError('Cannot convert nonmonotonic or gapped raw Pico timing')
    if startup_s < 0 or blend_s < .1: raise ValueError('Invalid offline reference entry blend')
    mapper = CalibratedPicoThreePoint(calibration)
    mapped = [mapper.map(b) for b in body]
    p = np.asarray([x[0] for x in mapped]); q = np.asarray([x[1] for x in mapped])
    sample_times = np.arange(int(np.floor(t[-1]/.02))+1)*.02
    if len(sample_times) < 2: raise ValueError('Pico recording too short for50Hz playback')
    interpolated_p = []; interpolated_q = []
    for x in sample_times:
        i = min(max(int(np.searchsorted(t, x, side='right'))-1, 0), len(t)-2)
        f = float((x-t[i])/(t[i+1]-t[i]))
        interpolated_p.append((1-f)*p[i]+f*p[i+1]); interpolated_q.append(slerp(q[i], q[i+1], f))
    p = np.asarray(interpolated_p); q = np.asarray(interpolated_q)
    warm = int(round(startup_s/.02)); blend = int(round(blend_s/.02))
    pos = [mapper.robot_p.copy() for _ in range(warm)]; rot = [mapper.robot_q.copy() for _ in range(warm)]
    for i in range(blend):
        a = (i+1)/blend; a = a*a*(3-2*a)
        pos.append((1-a)*mapper.robot_p+a*p[0]); rot.append(slerp(mapper.robot_q, q[0], a))
    pos.extend(p); rot.extend(q)
    count = len(pos)
    return dict(time_s=np.arange(count)*.02, positions_base_m=np.asarray(pos, np.float32),
        quaternions_base_wxyz=np.asarray(rot, np.float32),
        reference_root_quat_world_wxyz=np.repeat(np.asarray(calibration['reference_root_wxyz'], np.float32)[None], count, axis=0),
        fps=np.array(50.), point_names=np.array(POINTS), coordinate_space=np.array('reference_base_link'))


class LivePicoThreePoint:
    """Device-only adapter; output is calibrated data, never robot commands."""
    def __init__(self, calibration):
        self.mapper = CalibratedPicoThreePoint(calibration); self.timing = TimingGate(); self.latest = None

    def push_snapshot(self, available, stamp_ns, body, receive_ns):
        if not available: return False
        if not self.timing.accept(stamp_ns, body, receive_ns): return False
        p, q = self.mapper.map(body)
        self.latest = dict(positions_base_m=p, quaternions_base_wxyz=q,
            sample_time_s=(int(stamp_ns)-self.timing.origin_stamp)*1e-9, coordinate_space='reference_base_link', point_names=POINTS)
        return True

    def target(self, now_ns):
        self.timing.check_age(now_ns, maximum_s=.1)
        return self.latest
