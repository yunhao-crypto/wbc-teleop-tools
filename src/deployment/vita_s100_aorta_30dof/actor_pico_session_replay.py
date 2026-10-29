"""Offline replay of a frozen on-robot Pico session through the deployed worker code.

``scripts/freeze_pico_session.py`` writes the fixture. ``SessionReplay`` feeds
its Pico frames, in their recorded order, through the real ``ReferenceStream``
(GMR, stream conditioning, causal window) on a virtual monotonic clock that
reproduces the recorded receipt times and on-robot GMR durations, then yields
the snapshot a 50 Hz control loop would have held at each tick. No robot,
transport or Pico SDK is involved; callers add their own plant (e.g. MuJoCo).
"""
import copy
import json

import numpy as np

from .actor_pico_worker import ReferenceStream


class SessionFixture:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {k: data[k] for k in data.files}
        self.meta = json.loads(str(self.arrays['meta']))
        if self.meta.get('schema') != 'vita_pico_session_fixture_v1':
            raise ValueError('Unsupported Pico session fixture')
        for key, value in self.arrays.items():
            if key != 'meta':
                setattr(self, key, value)

    @property
    def human_height_m(self):
        return float(self.meta['retargeting']['human_height_m'])

    def following_segments(self):
        """[(first, last)] recorded control-tick monotonic times of each following segment."""
        stage, times = self.tick_stage, self.tick_monotonic
        segments, start = [], None
        for i, s in enumerate(stage):
            if s == 1 and start is None:
                start = i
            if start is not None and (s != 1 or i == len(stage)-1):
                last = i if s == 1 else i-1
                window = times[start:last+1]
                window = window[np.isfinite(window)]
                if len(window):
                    segments.append((float(window[0]), float(window[-1])))
                start = None
        return segments

    def gmr_durations(self, mode='robot'):
        """Per-frame GMR seconds: 'robot' uses the on-robot measurement where a
        tick logged it and the median elsewhere; a number is a constant in ms."""
        if mode == 'robot':
            ms = self.frame_robot_retarget_ms.astype(np.float64)
            known = np.isfinite(ms)
            ms = np.where(known, ms, np.median(ms[known]) if known.any() else 0.)
        else:
            ms = np.full(len(self.frame_stamp_ns), float(mode))
        return ms*1e-3


class _Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class SessionReplay:
    """Recorded Pico frames -> worker snapshots, on the recorded timeline.

    The worker handles one frame at a time: a frame starts at its receipt or
    when the previous solve finished, whichever is later, and is published
    when its solve finishes. A control tick sees the latest published snapshot.
    """
    def __init__(self, runtime, fixture, *, gmr='robot', config=None):
        self.runtime, self.fixture = runtime, fixture
        self.clock = _Clock()
        config = copy.deepcopy(runtime.config if config is None else config)
        config['retargeting']['human_height_m'] = fixture.human_height_m
        self.config = config
        self.stream = ReferenceStream(runtime.root, config, runtime.model.contract, runtime.math,
                                      clock=self.clock)
        received = fixture.frame_received_monotonic
        done, previous = np.empty(len(received)), -np.inf
        for i, (r, d) in enumerate(zip(received, fixture.gmr_durations(gmr))):
            previous = done[i] = max(r, previous)+d
        self.published = done
        self.next_frame = 0
        self.snapshot = dict(state='idle')
        self.anchor = (1., 0., 0., 0.)

    def advance(self, now):
        """Process every frame published by ``now``; return the latest snapshot."""
        f = self.fixture
        while self.next_frame < len(self.published) and self.published[self.next_frame] <= now:
            i = self.next_frame
            self.clock.now = float(self.published[i])
            body = f.frame_body[i].astype(np.float64)
            self.snapshot = self.stream.push(body, dict(timestamp_ns=int(f.frame_stamp_ns[i]),
                timestamp_monotonic=float(f.frame_received_monotonic[i])), self.anchor)
            self.next_frame += 1
        return dict(self.snapshot)

    def ticks(self, start, end, period=.02):
        """Yield (now, snapshot) on a uniform control grid over [start, end]."""
        for k in range(int(np.floor((end-start)/period))+1):
            now = start+k*period
            yield now, self.advance(now)

    def current_source_time(self, snapshot, now):
        """Source time (s since the first frame) of the policy's current reference frame."""
        reference = self.stream.reference
        if snapshot.get('stream') is not None:
            margin = self.config['pico_stream']['playback_margin_s']
            return now-snapshot['stream']['clock_offset_s']-margin-reference.delay
        return (snapshot['pico_timestamp_ns']-reference.origin_ns)*1e-9-reference.delay


def advance_statistics(times, period=.02):
    """How the current reference frame moves between consecutive control ticks."""
    step = np.diff(np.asarray(times))
    return dict(ticks=len(step), repeated=float(np.mean(np.abs(step) < 1e-9)),
                nominal=float(np.mean(np.abs(step-period) <= .002)),
                backwards=int(np.sum(step < -1e-9)),
                step_p50_ms=float(np.median(step)*1e3), step_p95_ms=float(np.percentile(step, 95)*1e3),
                step_max_ms=float(step.max()*1e3))

def export_windows(runtime, fixture, *, config, gmr='robot', segments=None, stand=2., seconds=0.):
    """Per control tick, the reference window the live loop would feed the policy.

    Covers ``stand`` seconds before and the whole of each following segment.
    The window is manual_reference()'s output before the stale-hold rule,
    which depends only on ``pico_age_s`` and is applied by the consumer. The
    reference never depends on robot state, so a plant can replay it exactly.
    """
    from .actor_pico_live import manual_reference
    replay = SessionReplay(runtime, fixture, gmr=gmr, config=config)
    rows, period = [], runtime.model.contract.period
    all_segments = fixture.following_segments()
    for index in (range(len(all_segments)) if segments is None else segments):
        start, end = all_segments[index]
        if seconds:
            end = min(end, start+seconds)
        for k in range(int(round((end-start+stand)/period))):
            now = start-stand+k*period
            snapshot = replay.advance(now)
            row = dict(segment=index, time=now, following=now >= start, valid=False)
            try:
                motion, diagnostics = manual_reference(runtime, snapshot, now)
            except (ValueError, TimeoutError):
                pass
            else:
                row.update(valid=True, pico_age_s=diagnostics['pico_age_s'], q=motion.q,
                           base=motion.base, quat=np.asarray(motion.quat),
                           current_source_s=replay.current_source_time(snapshot, now))
            rows.append(row)
    steps = runtime.reference.steps
    pick = lambda key, shape: np.stack([r[key] if r['valid'] else np.full(shape, np.nan) for r in rows])
    return dict(segment=np.array([r['segment'] for r in rows], dtype=np.int16),
                time=np.array([r['time'] for r in rows]),
                following=np.array([r['following'] for r in rows]),
                valid=np.array([r['valid'] for r in rows]),
                pico_age_s=np.array([r.get('pico_age_s', np.nan) for r in rows]),
                current_source_s=np.array([r.get('current_source_s', np.nan) for r in rows]),
                q=pick('q', (steps, 28)), base=pick('base', (steps, 3)), quat=pick('quat', (steps, 4)))
