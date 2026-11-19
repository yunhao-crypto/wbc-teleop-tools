"""Head and gripper migration with fakes. No Pico service, EE client, or robot."""
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from deployment.vita_s100_aorta_30dof.actor_pico_native_accessories import (
    EVENT, NativePicoAccessories, load_native_gripper_config, require_accessories,
)
from deployment.vita_s100_aorta_30dof.sonic_action14_head import Action14Head


EXAMPLE = Path(__file__).resolve().parents[1] / 'config' / 'pico_op3_gripper.example.json'


class _Control:
    def __init__(self):
        self.state = None
        self.calls = 0
        self.hold_age = .25

    def reset(self):
        self.state = None

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
    def test_config_and_capability_do_not_construct_an_sdk_client(self):
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.load_sdk',
                   side_effect=AssertionError('sdk')):
            config = load_native_gripper_config(EXAMPLE)
        capabilities = dict(native_pico_accessories_v1=True, native_pico_op3_sides=3)
        require_accessories(capabilities, gripper=config)
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            require_accessories(dict(capabilities, native_pico_op3_sides=1), gripper=config)

    def test_fresh_side_inputs_arm_independently_and_reengagement_requires_release(self):
        class Controls(list):
            def get_lock(self):
                return nullcontext()

        controls = Controls([0.] * 7)
        sent = []
        clock = [5.]
        head = Action14Head(_Control())
        runner = NativePicoAccessories(lambda name, payload: sent.append((name, payload)),
                                      controls, head=head, gripper=load_native_gripper_config(EXAMPLE),
                                      clock=lambda: clock[0])
        runner.tick()
        self.assertEqual(sent, [])
        runner.update(True, dict(raw_body_poses=[[0.]], received_monotonic=5.0))
        controls[3:7] = [5., 1., .0, .9]
        runner.tick()
        name, payload = sent[-1]
        self.assertEqual(name, EVENT)
        self.assertEqual(payload['neck_yaw_rad'], .2)
        self.assertFalse(payload['left_closed'])
        self.assertNotIn('right_closed', payload)
        controls[3:7] = [5.01, 2., .9, .9]
        clock[0] = 5.01
        runner.tick()
        clock[0] = 5.08
        runner.tick()
        self.assertTrue(sent[-1][1]['left_closed'])
        self.assertNotIn('right_closed', sent[-1][1])
        clock[0] = 5.4
        runner.tick()
        self.assertEqual(set(sent[-1][1]), {'capture_ns'})
        runner.update(False)
        runner.update(True)
        controls[3:7] = [5.41, 3., .9, .9]
        clock[0] = 5.41
        runner.tick()
        self.assertNotIn('left_closed', sent[-1][1])


if __name__ == '__main__':
    unittest.main()
