"""Local Pico source for the released pure-SMPL policy path."""

from __future__ import annotations
from collections import deque
from threading import Lock
import time

from humanoid_policy.pico_body_pose_reader import PicoBodyPoseReader


class PicoFrameTimeline:
    """Select real reader samples on a 50 Hz source clock with bounded catch-up."""

    period_ns = 20_000_000
    max_gap_ns = 50_000_000
    max_lag_ns = 50_000_000

    def __init__(self):
        self.frames = deque(maxlen=16)
        self.lock = Lock()
        self.target_ns = None
        self.last_selected_ns = None
        self.last_input_ns = None
        self.clock_reversed = False

    def append(self, sample):
        stamp = sample['timestamp_ns']
        with self.lock:
            if self.last_input_ns is not None:
                if stamp <= self.last_input_ns:
                    self.clock_reversed = True
                if stamp - self.last_input_ns > self.max_gap_ns:
                    self.frames.clear()
                    self.target_ns = None
            self.last_input_ns = stamp
            self.frames.append(sample)

    def take(self, now_ns=None):
        with self.lock:
            if now_ns is None:
                now_ns = time.monotonic_ns()
            if self.clock_reversed:
                raise ValueError('Pico source clock reversed; restart Pico acquisition')
            while self.frames and now_ns - self.frames[0]['reader_monotonic_ns'] > self.max_lag_ns:
                self.frames.popleft()
            if not self.frames:
                return None
            newest = self.frames[-1]['timestamp_ns']
            target = self.target_ns
            if target is None:
                target = newest
            if newest < target:
                return None
            # Drop overdue grid slots without changing the phase. Limit lag
            # in source time too: SDK samples may become visible in bursts.
            while len(self.frames) > 1 and newest-self.frames[0]['timestamp_ns'] > self.max_lag_ns:
                self.frames.popleft()
            # Prefer a recent pose. After a stall, keep one intermediate
            # sample when jumping straight to the newest would break continuity.
            selected = self.frames[-1]
            if self.last_selected_ns is not None and newest-self.last_selected_ns > self.max_gap_ns:
                for frame in reversed(self.frames):
                    if frame['timestamp_ns']-self.last_selected_ns <= self.max_gap_ns:
                        selected = frame
                        break
            while self.frames.popleft() is not selected:
                pass
            stamp = selected['timestamp_ns']
            self.last_selected_ns = stamp
            self.target_ns = target + (max(0, stamp-target)//self.period_ns + 1)*self.period_ns
            return selected


class LocalPicoSmplSource:
    """Expose an atomic Pico pose without running robot IK."""

    def __init__(
        self,
        *,
        logger,
        max_source_age: float = 0.6,
        sdk=None,
        duplicate_poll_interval_s: float = 0.0001,
        on_frame=None,
        timeline_sampling=False,
    ) -> None:
        self.logger = logger
        self.timeline = PicoFrameTimeline() if timeline_sampling else None
        def publish(sample):
            self.timeline.append(sample)
            if on_frame is not None:
                on_frame(sample)
        self.reader = PicoBodyPoseReader(logger=logger, sdk=sdk,
                                        duplicate_poll_interval_s=duplicate_poll_interval_s,
                                        on_frame=publish if self.timeline is not None else on_frame)
        self.max_source_age = float(max_source_age)
        self.frame_index = 0
        self.last_sample = None
        self._last_timestamp_ns = None
        self._stale_count = 0

    def start(self) -> None:
        self.reader.start()
        self.logger.info("[SMPL] local Pico -> canonical SMPL source ready")

    def step(self):
        sample = (self.timeline.take() if self.timeline is not None
                  else self.reader.latest())
        if sample is None:
            return None
        timestamp_ns = int(sample["timestamp_ns"])
        if timestamp_ns == self._last_timestamp_ns:
            return None
        age = time.monotonic() - float(sample["timestamp_monotonic"])
        if age > self.max_source_age:
            self._stale_count += 1
            if self._stale_count == 1 or self._stale_count % 50 == 0:
                self.logger.warn(
                    f"[Pico] source stale: age={age * 1000.0:.1f}ms > "
                    f"{self.max_source_age * 1000.0:.1f}ms"
                )
            return None
        self._stale_count = 0
        result = (sample["body_poses"], self.frame_index, sample)
        self._last_timestamp_ns = timestamp_ns
        self.frame_index += 1
        self.last_sample = sample
        return result

    def stop(self) -> None:
        self.reader.stop()


__all__ = ["LocalPicoSmplSource"]
