"""Head and gripper migration with fakes. No Pico service, EE client, or robot."""
import unittest
from pathlib import Path
from unittest.mock import patch

from deployment.vita_s100_aorta_30dof.sonic_action14_gripper import attach, describe
from deployment.vita_s100_aorta_30dof.sonic_action14_head import Action14Head


EXAMPLE = Path(__file__).resolve().parents[1] / 'config' / 'pico_op3_gripper.example.json'


class _Control:
    def __init__(self):
        self.state = None
        self.calls = 0

    def preview(self, command, order, *, now, snapshot=None, tracking=False):
        self.calls += 1
        self.last = dict(now=now, tracking=tracking, poses=snapshot['raw_body_poses'])
        return command, dict(target_q=(0.2, -0.1)), ('candidate', now)

    def commit(self, candidate):
        self.state = candidate


class HeadTests(unittest.TestCase):
    def test_preview_does_not_commit_until_asked(self):
        control = _Control()
        head = Action14Head(control)
        targets, pending = head.preview_targets(
            dict(raw_body_poses=[[0.]], received_monotonic=1.0, pico_timestamp_ns=10), 1.05)
        self.assertIsNone(control.state)
        self.assertEqual(targets, {'neck_yaw_joint': 0.2, 'neck_pitch_joint': -0.1})
        self.assertEqual(control.last['tracking'], True)
        head.commit(pending)
        self.assertEqual(control.state, ('candidate', 1.05))


class GripperTests(unittest.TestCase):
    def test_describe_does_not_open_a_client(self):
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.load_sdk',
                   side_effect=AssertionError('sdk')), \
             patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.GripperRunner',
                   side_effect=AssertionError('runner')):
            described = describe(EXAMPLE)
        self.assertFalse(described['gripper_opened'])
        self.assertEqual(described['gripper_sides'], ['left', 'right'])

    def test_attach_requires_a_connection_and_uses_the_injected_sdk(self):
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.GripperRunner',
                   side_effect=AssertionError('runner')):
            with self.assertRaises(ValueError):
                attach(EXAMPLE, None, object())
        seen = {}

        class Runner:
            def __init__(self, config, shared, *, connection, sdk=None, **kwargs):
                seen.update(connection=connection, sdk=sdk, schema=config['schema'])

        with patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.GripperRunner', Runner):
            attach(EXAMPLE, 'action14-transport', object(), sdk='injected')
        self.assertEqual(seen['connection'], 'action14-transport')
        self.assertEqual(seen['sdk'], 'injected')
        self.assertEqual(seen['schema'], 'wtt_pico_gripper_v1')


if __name__ == '__main__':
    unittest.main()
