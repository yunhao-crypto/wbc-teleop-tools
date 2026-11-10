"""Causal motion windows for the Vita02A G1-reference actor.

Retargeting is provided by actor_pico_gmr using the unchanged bridge solver.
"""
from collections import deque
import math
import numpy as np


class CausalActorReference:
    """Delay real references until the complete future/derivative window exists.

    Timing comes from the loaded model. Ten references at 20 ms need a
    180 ms horizon plus a 20 ms derivative tail (200 ms total); the legacy
    100 ms spacing needs 920 ms. Neither value is measured end-to-end latency.
    No prediction, repeated terminal pose, or invented future velocity is used.

    ``motion()`` ends the window at the newest sample. ``playback()`` ends it
    at a control-clock time (``pico_stream``), which needs ``history_s`` of
    samples beyond the window.
    """
    # The current reference frame is the first window sample.
    current_index = 0
    ref_time = 0.

    def __init__(self, contract, math_api, *, capacity=1024, history_s=0.):
        if contract.include_z:
            raise ValueError('This Pico retargeter does not supply calibrated robot root height')
        self.contract, self.math = contract, math_api
        if capacity < 3 or not 0 <= history_s <= 2:
            raise ValueError('Invalid reference buffering settings')
        self.samples = deque(maxlen=capacity)
        self.last_stamp_ns = None
        self.origin_ns = None
        self.stride = int(round(contract.future_dt/contract.period))
        self.steps = (contract.future_frames-1)*self.stride+2
        self.delay = (self.steps-1)*contract.period
        self.history = self.delay+history_s
        self.last_receive_monotonic = None

    def append(self, stamp_ns, q, root, *, received_monotonic, base=None):
        # base (GMR root position) is unused: this actor has no root-position input.
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
        # Keep one sample at or before the oldest time any window can query;
        # older ones never change the interpolation and would only be shipped.
        start = self.samples[-1][0]-self.history
        while len(self.samples) > 2 and self.samples[1][0] <= start:
            self.samples.popleft()
        self.last_stamp_ns = int(stamp_ns)
        self.last_receive_monotonic = float(received_monotonic)

    def ready(self):
        return len(self.samples) >= 2 and self.samples[-1][0]-self.samples[0][0] >= self.delay-1e-9

    def query(self, end):
        return end-self.delay+np.arange(self.steps)*self.contract.period

    def window(self, times, positions, quats, query):
        """Linear joint / slerp root resampling at ``query``, clamped at both ends like np.interp."""
        q = np.stack([np.interp(query, times, positions[:, j]) for j in range(28)], axis=1)
        roots = []
        for t in query:
            i = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, len(times)-2))
            alpha = float(np.clip((t-times[i])/(times[i+1]-times[i]), 0., 1.))
            roots.append(self.math.slerp(quats[i], quats[i+1], alpha))
        return self.math.Motion(q, np.zeros((self.steps, 3)), roots)

    def export(self):
        """Raw samples (source seconds since the first sample) for the control loop."""
        return dict(t=np.array([s[0] for s in self.samples]), q=np.stack([s[1] for s in self.samples]),
                    quat=np.stack([s[2] for s in self.samples]))

    def motion(self):
        if not self.ready():
            raise RuntimeError('Waiting for complete causal actor reference window')
        s = self.export()
        return self.window(s['t'], s['q'], s['quat'], self.query(s['t'][-1]))

    def playback(self, stream, now, margin):
        """Window ending at the source time usable ``margin`` before local ``now``.

        The window advances with the control clock, not with sample arrival.
        Past the newest sample it holds that sample (``playback_lead_s`` < 0).
        """
        t = np.asarray(stream['t'], dtype=np.float64)
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError('Pico stream needs at least two increasing samples')
        q, quat = np.asarray(stream['q'], dtype=np.float64), np.asarray(stream['quat'], dtype=np.float64)
        if q.shape != (len(t), 28) or quat.shape != (len(t), 4):
            raise ValueError('Pico stream sample dimensions disagree')
        offset = float(stream['clock_offset_s'])
        end = now-offset-margin
        if not math.isfinite(end):
            raise ValueError('Pico stream clock offset is invalid')
        motion = self.window(t, q, quat, self.query(end))
        return motion, dict(playback_lead_s=float(t[-1]-end), playback_time_s=end-self.delay,
                            clock_offset_s=offset)
