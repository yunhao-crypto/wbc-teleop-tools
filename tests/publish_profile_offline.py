"""Publisher instrumentation invariants using a fake synchronous sink only."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deployment.vita_s100_aorta_30dof.onboard_publish_profile import (
    ProfiledCommandPublisher,
)


class FakePublisher:
    def __init__(self):
        self.sequence = 0
        self.seen = []
        self.closed = False
        self.fail = False

    def publish_typed(self, fill, *, initial_size):
        seq = self.sequence
        self.sequence += 1
        self.seen.append(("capacity", initial_size))
        self.publish_bytes(fill("builder", seq))

    def publish_bytes(self, payload):
        if self.fail:
            raise RuntimeError("publication failed")
        self.seen.append(("published", payload))

    def close(self):
        self.closed = True


class ProfileTest(unittest.TestCase):
    def test_sync_call_order_and_timings(self):
        p = FakePublisher()
        wrapper = ProfiledCommandPublisher(p)
        for i in range(4):
            wrapper.publish_typed(lambda b, h: bytes([h]))
            self.assertEqual(p.seen[-1], ("published", bytes([i])))
            self.assertLessEqual(
                wrapper.last["bytes_started_wall_ns"],
                wrapper.last["bytes_returned_wall_ns"],
            )
            self.assertGreaterEqual(
                wrapper.last["typed_wall_ms"], wrapper.last["bytes_wall_ms"]
            )
            self.assertEqual(wrapper.last["payload_bytes"], 1)
        self.assertEqual(p.sequence, 4)
        wrapper.close()
        self.assertTrue(p.closed)

    def test_errors_propagate(self):
        p = FakePublisher()
        wrapper = ProfiledCommandPublisher(p)
        p.fail = True
        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            wrapper.publish_typed(lambda b, h: b"test")
        self.assertIn("bytes_cpu_ms", wrapper.last)
        self.assertFalse(any(x[0] == "published" for x in p.seen))

    def test_fill_error_never_publishes(self):
        p = FakePublisher()
        wrapper = ProfiledCommandPublisher(p)

        def fail(*args):
            raise ValueError("invalid body")

        with self.assertRaises(ValueError):
            wrapper.publish_typed(fail)
        self.assertNotIn("bytes_wall_ms", wrapper.last)
        self.assertFalse(any(x[0] == "published" for x in p.seen))


if __name__ == "__main__":
    unittest.main(verbosity=2)
