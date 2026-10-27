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
