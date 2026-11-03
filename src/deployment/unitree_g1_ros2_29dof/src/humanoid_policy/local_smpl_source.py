"""Local Pico source for the released pure-SMPL policy path."""

from __future__ import annotations
import time

from humanoid_policy.pico_body_pose_reader import PicoBodyPoseReader


class LocalPicoSmplSource:
    """Expose the newest Pico frame without running robot IK."""

    def __init__(
        self,
        *,
        logger,
        max_source_age: float = 0.6,
        sdk=None,
        duplicate_poll_interval_s: float = 0.0001,
    ) -> None:
        self.logger = logger
        self.reader = PicoBodyPoseReader(logger=logger, sdk=sdk,
                                        duplicate_poll_interval_s=duplicate_poll_interval_s)
        self.max_source_age = float(max_source_age)
        self.frame_index = 0
        self.last_sample = None
        self._last_timestamp_ns = None
        self._stale_count = 0

    def start(self) -> None:
        self.reader.start()
        self.logger.info("[SMPL] local Pico -> canonical SMPL source ready")

    def step(self):
        sample = self.reader.latest()
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
