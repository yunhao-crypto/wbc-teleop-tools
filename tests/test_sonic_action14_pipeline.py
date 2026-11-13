"""Synthetic-only tests; no sockets, Pico, hardware, ONNX or inference."""
import unittest
from types import SimpleNamespace
from deployment.vita_s100_aorta_30dof.sonic_action14_resample import SourceGrid
from deployment.vita_s100_aorta_30dof.sonic_action14_transport import NativeReferencePublisher


class FakeTransport:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.prepared = False
    def _prepare_body_output(self):
        self.prepared = True
    def get_robot_info(self, timeout):
        return SimpleNamespace(runtime_id=42, info=object())
    def _publish_body(self, command, guard):
        guard(lambda: self.commands.append(command))
    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def test_causal_grid_no_replay(self):
        grid = SourceGrid()
        q = [0.] * 28
        self.assertEqual(grid.push(1_000_000_000, 1., q, (1., 0., 0., 0.)), [])
        first = grid.push(1_050_000_000, 1.05, [1.]*28, (1., 0., 0., 0.))
        self.assertEqual([x['grid_sequence'] for x in first], [1, 2])
        self.assertEqual([x['source_ns'] for x in first], [1_020_000_000, 1_040_000_000])
        self.assertAlmostEqual(first[0]['q'][0], .4)
        self.assertAlmostEqual(first[0]['received_monotonic'], 1.02)
        self.assertLess(first[0]['received_monotonic'], 1.05)
        self.assertEqual(grid.push(1_055_000_000, 1.055, [1.]*28, (1.,0.,0.,0.)), [])
        with self.assertRaises(ValueError):
            grid.push(1_055_000_000, 1.06, q, (1.,0.,0.,0.))
        filled = grid.push(1_200_000_000, 1.2, q, (1.,0.,0.,0.))
        self.assertGreater(len(filled), 0)
        self.assertLess(filled[0]['received_monotonic'], 1.2)
        with self.assertRaises(ValueError):
            grid.push(1_200_000_000 + 600_000_001, 1.9, q, (1.,0.,0.,0.))

    def test_generated_publisher_gate(self):
        fake = FakeTransport()
        publisher = NativeReferencePublisher(transport=fake)
        self.assertTrue(fake.prepared)
        self.assertEqual(publisher.robot_info().runtime_id, 42)
        command = SimpleNamespace(action=14, control_id=7, input_state_id=0,
                                  input_state_ts_real_us=0)
        with self.assertRaises(RuntimeError):
            publisher.publish(command, authorized=lambda: False)
        self.assertEqual(fake.commands, [])
        publisher.publish(command, authorized=lambda: True)
        self.assertEqual(fake.commands, [command])
        publisher.close()
        self.assertTrue(fake.closed)


if __name__ == '__main__':
    unittest.main()
