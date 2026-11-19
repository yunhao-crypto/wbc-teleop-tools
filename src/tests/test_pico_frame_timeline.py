from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'deployment/unitree_g1_ros2_29dof/src'))
from humanoid_policy.local_smpl_source import PicoFrameTimeline


class PicoFrameTimelineTests(unittest.TestCase):
    def setUp(self):
        self.timeline = PicoFrameTimeline()

    def add(self, stamp, arrival=None):
        sample = dict(timestamp_ns=stamp*1_000_000,
                      reader_monotonic_ns=(stamp if arrival is None else arrival)*1_000_000)
        self.timeline.append(sample)
        return sample

    def take(self, now):
        return self.timeline.take(now*1_000_000)

    def test_recovers_real_frame_after_stall_then_catches_up(self):
        initial = self.add(10)
        self.assertIs(self.take(10), initial)
        self.add(20)
        self.assertIsNone(self.take(20))
        poses = [self.add(t) for t in (30, 40, 50, 60, 70, 80)]
        self.assertIs(self.take(80), poses[3])
        self.assertIs(self.take(81), poses[-1])
        self.assertIsNone(self.take(82))

    def test_real_gap_remains_and_expired_frames_are_never_reissued(self):
        self.add(10)
        self.take(10)
        latest = self.add(100)
        self.assertIs(self.take(100), latest)
        self.add(120)
        self.assertIsNone(self.take(171))

    def test_reverse_clock_remains_visible_after_queue_overflow(self):
        self.add(10)
        self.take(10)
        self.add(5, 11)
        for t in range(12, 100):
            self.add(t)
        with self.assertRaisesRegex(ValueError, 'clock reversed'):
            self.take(100)
        self.assertLessEqual(len(self.timeline.frames),16)
