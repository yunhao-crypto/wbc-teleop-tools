"""Pure synthetic data only; no transports, model sessions, Pico or actions."""
import unittest
from unittest.mock import patch
from deployment.vita_s100_aorta_30dof.sonic_action14 import ReferenceEncoder, contract, plan


class ContractTests(unittest.TestCase):
    def test_offline_plan(self):
        with patch('socket.socket', side_effect=AssertionError('network')):
            self.assertEqual(plan()['reference_joints'], 28)

    def test_encode_and_reject_stale(self):
        slots, _ = contract()
        names = set(slots) - {'waist_roll_joint', 'neck_pitch_joint', 'neck_yaw_joint'}
        encoder = ReferenceEncoder(123, 7, session_id=789)
        ref = encoder.encode({n: 0.25 for n in names}, (1, 0, 0, 0),
                             source_ns=123456, received_monotonic=10, now_monotonic=10.02)
        self.assertEqual((ref.action, ref.protocol_version, ref.session_id, ref.sequence), (14, 3, 789, 1))
        self.assertEqual((ref.runtime_id, ref.control_id, ref.input_state_id, ref.input_state_ts_real_us), (123, 7, 0, 0))
        self.assertEqual(encoder.last_source_ns, 123456)
        self.assertEqual(len(ref.joints), 31)
        self.assertEqual(ref.joints[14].q, 0)
        self.assertEqual(ref.joints[slots.index('neck_pitch_joint')].q, 0)
        self.assertEqual(ref.joints[slots.index('neck_yaw_joint')].q, 0)
        followed_encoder = ReferenceEncoder(123, 7, session_id=790)
        followed = followed_encoder.encode({n: 0.25 for n in names}, (1, 0, 0, 0),
                                           source_ns=123500, received_monotonic=10.1, now_monotonic=10.12,
                                           neck_q={'neck_yaw_joint': 0.2, 'neck_pitch_joint': -0.1})
        self.assertAlmostEqual(followed.joints[slots.index('neck_yaw_joint')].q, 0.2)
        self.assertAlmostEqual(followed.joints[slots.index('neck_pitch_joint')].q, -0.1)
        with self.assertRaises(ValueError):
            followed_encoder.encode({n: 0.25 for n in names}, (1, 0, 0, 0),
                                    source_ns=123600, received_monotonic=10.2, now_monotonic=10.22,
                                    neck_q={'neck_yaw_joint': 2.0, 'neck_pitch_joint': 0.0})
        with self.assertRaises(ValueError):
            encoder.encode({n: 0.25 for n in names}, (1, 0, 0, 0),
                           source_ns=123456, received_monotonic=10, now_monotonic=10.02)
        with self.assertRaises(ValueError):
            encoder.encode({n: 0.25 for n in names}, (1, 0, 0, 0),
                           source_ns=123457, received_monotonic=10, now_monotonic=11)
        self.assertEqual(encoder.sequence, 1)
        with self.assertRaises(ValueError):
            ReferenceEncoder(123, 0)


if __name__ == '__main__':
    unittest.main()
