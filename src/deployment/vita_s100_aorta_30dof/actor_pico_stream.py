"""Pico/GMR stream conditioning for the SuperMimic reference (worker side).

Settings come from the optional ``pico_stream`` block of the actor
configuration. Without it the worker keeps the original behaviour: the
control loop receives a reference window anchored to the newest sample.

With it, the worker ships the raw retargeted samples plus a source-clock to
local-clock offset, and the control loop resamples them at ``now - delay`` so
the reference advances by exactly the elapsed control time every tick,
however irregularly Pico frames or GMR solves arrive.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OneEuroSettings:
    min_cutoff_hz: float
    beta: float
    d_cutoff_hz: float


@dataclass(frozen=True)
class StreamSettings:
    ik_max_iter: int
    playback_margin_s: float
    joint_filter: OneEuroSettings = None
    root_filter: OneEuroSettings = None
    root_max_tilt_rad: float = None
    worker_cpus: tuple = ()


def _euro(raw, name):
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {'min_cutoff_hz', 'beta', 'd_cutoff_hz'}:
        raise ValueError(f'pico_stream.{name} needs min_cutoff_hz, beta and d_cutoff_hz')
    values = [raw[k] for k in ('min_cutoff_hz', 'beta', 'd_cutoff_hz')]
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
            or values[0] <= 0 or values[1] < 0 or values[2] <= 0):
        raise ValueError(f'Invalid pico_stream.{name} One-Euro parameters')
    return OneEuroSettings(*(float(v) for v in values))


def stream_settings(config):
    """Validated ``pico_stream`` settings, or None for the original stream."""
    raw = config.get('pico_stream')
    if raw is None:
        return None
    known = {'ik_max_iter', 'playback_margin_s', 'joint_filter', 'root_filter',
             'root_max_tilt_deg', 'worker_cpus', 'note'}
    if not isinstance(raw, dict):
        raise ValueError('pico_stream must be an object')
    if set(raw)-known or not {'ik_max_iter', 'playback_margin_s'} <= set(raw):
        raise ValueError('pico_stream needs ik_max_iter and playback_margin_s; unknown keys: '
                         +', '.join(sorted(set(raw)-known)))
    iterations, margin = raw['ik_max_iter'], raw['playback_margin_s']
    if type(iterations) is not int or not 0 <= iterations <= 10:
        raise ValueError('pico_stream.ik_max_iter must be an integer in [0, 10]')
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= .3:
        raise ValueError('pico_stream.playback_margin_s must lie in [0, 0.3] s')
    tilt = raw.get('root_max_tilt_deg')
    if tilt is not None and (isinstance(tilt, bool) or not isinstance(tilt, (int, float))
                             or not 0 < tilt <= 90):
        raise ValueError('pico_stream.root_max_tilt_deg must lie in (0, 90]')
    cpus = raw.get('worker_cpus', [])
    if (not isinstance(cpus, list) or any(type(c) is not int or c < 0 for c in cpus)
            or len(set(cpus)) != len(cpus)):
        raise ValueError('pico_stream.worker_cpus must list distinct CPU indices')
    return StreamSettings(iterations, float(margin), _euro(raw.get('joint_filter'), 'joint_filter'),
                          _euro(raw.get('root_filter'), 'root_filter'),
                          None if tilt is None else math.radians(tilt), tuple(cpus))


def _smoothing(cutoff, dt):
    return 1./(1.+1./(2*math.pi*cutoff*dt))


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) on a vector, with per-element cutoff."""
    def __init__(self, settings):
        self.settings = settings
        self.x = self.dx = None

    def reset(self):
        self.x = self.dx = None

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x.copy()
        s = self.settings
        self.dx = self.dx+_smoothing(s.d_cutoff_hz, dt)*((x-self.x)/dt-self.dx)
        cutoff = s.min_cutoff_hz+s.beta*np.abs(self.dx)
        self.x = self.x+(1./(1.+1./(2*np.pi*cutoff*dt)))*(x-self.x)
        return self.x.copy()


def _multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def _conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _log(q):
    """Rotation vector of a unit wxyz quaternion (shortest arc)."""
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    return 2*math.atan2(s, q[0])/s*q[1:] if s > 1e-12 else 2*q[1:]


def _exp(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1., *(v/2)])
    return np.array([math.cos(angle/2), *(math.sin(angle/2)/angle*v)])


class QuaternionOneEuro:
    """One-Euro on SO(3): the filtered rotation moves along the geodesic."""
    def __init__(self, settings):
        self.settings = settings
        self.q = self.speed = None

    def reset(self):
        self.q = self.speed = None

    def __call__(self, q, dt):
        q = np.asarray(q, dtype=np.float64)
        q = q/np.linalg.norm(q)
        if self.q is None:
            self.q, self.speed = q.copy(), 0.
            return q.copy()
        s = self.settings
        delta = _log(_multiply(_conjugate(self.q), q))
        self.speed += _smoothing(s.d_cutoff_hz, dt)*(np.linalg.norm(delta)/dt-self.speed)
        alpha = _smoothing(s.min_cutoff_hz+s.beta*self.speed, dt)
        self.q = _multiply(self.q, _exp(alpha*delta))
        self.q /= np.linalg.norm(self.q)
        return self.q.copy()


def limit_tilt(q, max_tilt):
    """Limit the angle between the root z axis and world z; keep the heading twist."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q/np.linalg.norm(q)
    up = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    tilt = math.acos(min(1., max(-1., up[2])))
    horizontal = math.hypot(up[0], up[1])
    if tilt <= max_tilt or horizontal < 1e-12:
        return np.array([w, x, y, z])
    axis = np.array([-up[1], up[0], 0.])/horizontal
    swing = _exp(axis*tilt)
    twist = _multiply(_conjugate(swing), np.array([w, x, y, z]))
    return _multiply(_exp(axis*max_tilt), twist)


class StreamFilter:
    """Causal per-sample conditioning of GMR q, root position and root rotation."""
    # A longer source gap starts a new segment rather than smoothing across it.
    RESET_GAP_S = .5

    def __init__(self, settings):
        self.settings = settings
        self.joint = OneEuro(settings.joint_filter) if settings.joint_filter else None
        self.base = OneEuro(settings.root_filter) if settings.root_filter else None
        self.root = QuaternionOneEuro(settings.root_filter) if settings.root_filter else None
        self.last_time = None

    def __call__(self, time_s, q, base, root):
        dt = None if self.last_time is None else time_s-self.last_time
        if dt is not None and not 0 < dt <= self.RESET_GAP_S:
            for f in (self.joint, self.base, self.root):
                if f is not None:
                    f.reset()
            dt = None
        self.last_time = time_s
        if self.settings.root_max_tilt_rad is not None:
            root = limit_tilt(root, self.settings.root_max_tilt_rad)
        step = dt if dt is not None else 1.
        q = self.joint(q, step) if self.joint else np.asarray(q, dtype=np.float64)
        base = self.base(base, step) if self.base else np.asarray(base, dtype=np.float64)
        root = self.root(root, step) if self.root else np.asarray(root, dtype=np.float64)
        return q, base, root


class SourceClock:
    """Offset mapping Pico source seconds to the local time a sample is usable.

    Tracks the lower envelope of (availability - source time): it falls toward
    faster samples and rises only slowly, so playback stays monotonic and
    nearly uniform. Only a sample earlier than the envelope by more than
    ``RESET_S`` (a forward source clock step) re-anchors. A late sample
    (worker stall, queueing) never re-anchors: raising the offset at once
    would move playback backwards by the same amount.
    """
    RISE, FALL, RESET_S = .02, .05, .5

    def __init__(self):
        self.offset = self.last_source = None

    def update(self, source_s, available_s):
        observed = available_s-source_s
        if self.offset is None or observed < self.offset-self.RESET_S:
            self.offset = observed
        else:
            # Capped so one update raises the offset by at most RISE*RESET_S
            # (10 ms), less than a control period.
            dt = min(max(0., source_s-self.last_source), self.RESET_S)
            if observed < self.offset:
                self.offset = max(observed, self.offset-self.FALL*dt)
            else:
                self.offset = min(observed, self.offset+self.RISE*dt)
        self.last_source = source_s
        return self.offset
