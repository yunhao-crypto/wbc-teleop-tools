"""Head articulation, transactional state and Body31 composition; no hardware."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_head import head_settings, NECK_NAMES
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort')

    def setUp(self):
        r = self.runtime
        r.reset_inputs()
        self.head = copy.deepcopy(r.head)
        self.command, _ = r.policy.decode(np.zeros(28))
        self.order = r.policy.order
        self.now = 10.
        self.stamp = 1_000_000_000
        self.slots = [self.order.index(n) for n in NECK_NAMES]

    def snapshot(self, yaw=0., pitch=0., roll=0., chest=None, neutral=None, age=0.):
        chest = np.eye(3) if chest is None else chest
        neutral = np.eye(3) if neutral is None else neutral
        rotation = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
        relative = self.head.offset @ rotation @ self.head.offset.T @ neutral
        body = np.zeros((24, 7)); body[:, 6] = 1
        body[9, 3:] = Rotation.from_matrix(chest).as_quat()
        body[15, 3:] = Rotation.from_matrix(chest @ relative).as_quat()
        return dict(raw_body_poses=body, received_monotonic=self.now-age,
                    pico_timestamp_ns=self.stamp)

    def tick(self, snapshot=None, tracking=True, measured=None, commit=True):
        command, diag, candidate = self.head.preview(self.command, self.order,
            now=self.now, snapshot=snapshot, tracking=tracking, measured_q=measured)
        if commit:
            self.head.commit(candidate)
        self.now += .02
        self.stamp += 20_000_000
        return command, diag, candidate

    def test_head_yaw_pitch_and_roll_projection_in_robot_frame(self):
        self.tick(self.snapshot())
        command, diag, _ = self.tick(self.snapshot(yaw=.4, pitch=-.2, roll=.7))
        np.testing.assert_allclose(diag['desired_q'], [.4, -.2], atol=1e-12)
        for _ in range(200):
            command, _, _ = self.tick(self.snapshot(yaw=.4, pitch=-.2, roll=.7))
        np.testing.assert_allclose([command.joints[i].q for i in self.slots], [.4, -.2], atol=1e-10)

    def test_runtime_uses_independent_head_pose_without_changing_body_reference(self):
        body_snapshot=self.snapshot()
        with patch.object(self.runtime,'head',self.head):
            for yaw in (0.,.4):
                fresh=self.snapshot(yaw=yaw)
                fresh.update(source='pico_reader',reader_frame_sequence=self.stamp,
                             reader_monotonic_ns=round(self.now*1e9))
                combined=dict(body_snapshot,head_snapshot=fresh)
                row={}
                _,candidate=self.runtime.apply_head(self.command,row,now=self.now,
                    snapshot=combined,tracking=True)
                self.head.commit(candidate)
                self.assertEqual(row['head_control']['pico_timestamp_ns'],self.stamp)
                self.assertEqual(row['head_control']['source'],'pico_reader')
                np.testing.assert_allclose(row['head_control']['desired_q'],[yaw,0.],atol=1e-12)
                self.assertEqual(combined['pico_timestamp_ns'],1_000_000_000)
                self.now+=.02;self.stamp+=20_000_000

    def test_stale_independent_head_holds_neck_without_changing_body_targets(self):
        with patch.object(self.runtime,'head',self.head):
            combined=dict(self.snapshot(),head_snapshot=self.snapshot(age=self.head.maximum_age+.01))
            measured = np.zeros(31); measured[self.slots] = [.2, -.1]
            row = {}
            command, candidate = self.runtime.apply_head(self.command, row, now=self.now,
                measured_q=measured, snapshot=combined, tracking=True)
            self.assertIsNone(self.head.state)  # Preview remains transactional.
            self.assertEqual(row['head_control']['mode'], 'hold')
            self.assertTrue(row['head_control']['input_expired'])
            np.testing.assert_array_equal(candidate.q, [.2, -.1])
            for i, joint in enumerate(command.joints):
                if i not in self.slots:
                    self.assertEqual(joint, self.command.joints[i])

    def test_world_turn_cancels_and_each_engagement_calibrates_current_pose(self):
        neutral = Rotation.from_euler('XYZ', [.1, -.4, .2]).as_matrix()
        _, diag, _ = self.tick(self.snapshot(neutral=neutral))
        self.assertTrue(diag['calibrated'])
        np.testing.assert_allclose(diag['desired_q'], 0., atol=1e-12)
        world = Rotation.from_euler('XYZ', [.3, -.2, 1.2]).as_matrix()
        _, diag, _ = self.tick(self.snapshot(yaw=.5, pitch=.2, chest=world, neutral=neutral))
        np.testing.assert_allclose(diag['desired_q'], [.5, .2], atol=1e-12)
        self.tick(tracking=False)
        _, diag, _ = self.tick(self.snapshot(yaw=.5, chest=world))
        self.assertTrue(diag['calibrated'])
        np.testing.assert_allclose(diag['desired_q'], 0., atol=1e-12)

    def test_rate_limits_first_measured_pose_and_standby_return(self):
        self.head.settings = replace(self.head.settings, max_velocity=.8, filter_tau=.12)
        measured = np.zeros(31)
        measured[self.slots] = [.5, -.3]
        _, _, state = self.tick(tracking=False, measured=measured)
        self.assertTrue(np.all(np.abs(state.q-measured[self.slots]) <= .8*.02+1e-12))
        last = state.q
        for _ in range(120):
            _, _, state = self.tick(tracking=False)
            self.assertLessEqual(np.max(np.abs(state.q-last)), .8*.02+1e-12)
            last = state.q
        np.testing.assert_allclose(state.q, 0., atol=1e-8)

    def test_unlimited_profile_follows_filtered_angle_without_legacy_slew_cap(self):
        self.assertIsNone(self.head.settings.max_velocity)
        self.assertEqual(self.head.settings.filter_tau, .04)
        self.tick(self.snapshot())
        command, _, state = self.tick(self.snapshot(yaw=.8, pitch=-.3))
        expected = -np.expm1(-.02/.04)*np.array([.8, -.3])
        np.testing.assert_allclose(state.q, expected, atol=1e-12)
        np.testing.assert_allclose(state.filtered, state.q, atol=1e-12)
        np.testing.assert_allclose([command.joints[i].q for i in self.slots], expected, atol=1e-12)
        self.assertGreater(state.q[0], .8*.02)

    def test_unlimited_standby_filter_starts_from_measured_neck_pose(self):
        measured = np.zeros(31); measured[self.slots] = [.5, -.3]
        _, _, state = self.tick(tracking=False, measured=measured)
        np.testing.assert_allclose(state.q, np.exp(-.02/.04)*measured[self.slots], atol=1e-12)

    def test_unlimited_discarded_candidate_does_not_advance_filter(self):
        self.tick(self.snapshot())
        previous = self.head.state
        _, _, first = self.tick(self.snapshot(yaw=.7), commit=False)
        self.now += 1.
        _, _, second = self.tick(self.snapshot(yaw=.7), commit=False)
        self.assertIs(self.head.state, previous)
        np.testing.assert_allclose(first.q, second.q, atol=1e-12)

    def test_head_does_not_change_policy_targets_or_profile_gains(self):
        self.tick(self.snapshot())
        command, _, _ = self.tick(self.snapshot(yaw=.6, pitch=.3))
        for i, (old, new) in enumerate(zip(self.command.joints, command.joints)):
            if i in self.slots:
                self.assertEqual((new.kp, new.kd, new.dq, new.tau), (old.kp, old.kd, 0., 0.))
            else:
                self.assertEqual(old, new)

    def test_stale_input_freezes_position_and_filter_then_resumes(self):
        self.tick(self.snapshot())
        _, _, before = self.tick(self.snapshot(yaw=.8))
        for _ in range(10):
            _, diag, state = self.tick(self.snapshot(yaw=.8, age=.3))
            self.assertEqual(diag['mode'], 'hold')
            np.testing.assert_array_equal(state.q, before.q)
        _, diag, state = self.tick(self.snapshot(yaw=.8))
        self.assertEqual(diag['mode'], 'tracking')
        self.assertGreater(state.q[0], before.q[0])
        _, diag, expired = self.tick(self.snapshot(age=.61))
        self.assertEqual(diag['mode'], 'hold')
        self.assertTrue(diag['input_expired'])
        np.testing.assert_array_equal(expired.q, state.q)
        _, diag, resumed = self.tick(self.snapshot(yaw=.8))
        self.assertEqual(diag['mode'], 'tracking')
        self.assertGreater(resumed.q[0], expired.q[0])

    def test_future_head_receipt_holds_without_advancing_filter_or_neutral(self):
        self.tick(self.snapshot())
        _, _, previous = self.tick(self.snapshot(yaw=.5))
        for age in (-1., .61, 10.):
            _, diag, held = self.tick(self.snapshot(yaw=-.5, age=age))
            self.assertEqual(diag['mode'], 'hold')
            np.testing.assert_array_equal(held.q, previous.q)
            np.testing.assert_array_equal(held.filtered, previous.q)
            np.testing.assert_array_equal(held.neutral, previous.neutral)
            self.assertEqual(held.stamp, previous.stamp)

    def test_invalid_missing_future_and_regressed_input_fail_without_state_change(self):
        self.tick(self.snapshot())
        previous = self.head.state
        invalid = []
        for field, value in [('raw_body_poses', None), ('pico_timestamp_ns', None),
                             ('pico_timestamp_ns', 1)]:
            bad = self.snapshot(); bad[field] = value; invalid.append(bad)
        for component, value in [(3, float('nan')), (6, 0.)]:
            bad = self.snapshot(); bad['raw_body_poses'][15, component] = value; invalid.append(bad)
        for bad in invalid:
            with self.subTest(bad=str(bad)[:60]), self.assertRaises((ValueError, TimeoutError)):
                self.tick(bad)
            self.assertIs(self.head.state, previous)

    def test_discarded_tick_does_not_advance_calibration_or_slew(self):
        self.head.settings = replace(self.head.settings, max_velocity=.8, filter_tau=.12)
        self.tick(self.snapshot())
        previous = self.head.state
        self.tick(self.snapshot(yaw=.7), commit=False)
        self.assertIs(self.head.state, previous)
        self.now += 1.  # A missed deadline still permits only a nominal tick's step.
        _, _, candidate = self.tick(self.snapshot(yaw=.7), commit=False)
        self.assertLessEqual(abs(candidate.q[0]-previous.q[0]), .8*.02+1e-12)

    def test_config_and_robot_limits_intersect_and_yaw_does_not_flip_at_pi(self):
        caps = [NS(populated=True, is_mock=False, valid_limits=31, q_min=-.2, q_max=.3)
                for _ in self.order]
        self.head.bind_body_info(NS(joint_names=self.order, joints=caps))
        self.tick(self.snapshot())
        _, diag, _ = self.tick(self.snapshot(yaw=3.12, pitch=-.5))
        np.testing.assert_allclose(diag['desired_q'], [.3, -.2])
        _, diag, _ = self.tick(self.snapshot(yaw=-3.12, pitch=-.5))
        np.testing.assert_allclose(diag['desired_q'], [.3, -.2])
        for _ in range(100):
            _, _, state = self.tick(self.snapshot(yaw=-3.12, pitch=-.5))
        self.assertTrue(np.all(state.q >= -.2))
        self.assertTrue(np.all(state.q <= .3))

    def test_disabled_head_keeps_fixed_neck_and_malformed_settings_fail(self):
        config = json.loads((ROOT/'config/actor_pico.json').read_text())
        self.assertEqual(head_settings(config), self.head.settings)
        config['head_control']['enabled'] = False
        self.assertIsNone(head_settings(config))
        self.assertIsNone(head_settings({}))
        row = {}
        with patch.object(self.runtime, 'head', None):
            command, candidate = self.runtime.apply_head(self.command, row, now=self.now,
                tracking=True, snapshot={'raw_body_poses': None})
        self.assertIs(command, self.command)
        self.assertIsNone(candidate)
        self.assertEqual(row, {})
        for key, value in [('enabled', 1), ('max_velocity_rad_s', 0.),
                           ('max_velocity_rad_s', False), ('max_velocity_rad_s', float('inf')),
                           ('max_velocity_rad_s', float('nan')), ('max_velocity_rad_s', -1.),
                           ('filter_time_constant_s', None), ('filter_time_constant_s', 0.),
                           ('filter_time_constant_s', float('nan')), ('unexpected', 1)]:
            bad = copy.deepcopy(config); bad['head_control'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                head_settings(bad)

    def test_frozen_robot42_session_produces_bounded_nonzero_head_targets(self):
        self.head.settings = replace(self.head.settings, max_velocity=.8, filter_tau=.12)
        with np.load(ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz') as data:
            frames = data['frame_body'].copy()
            stamps = data['frame_stamp_ns'].copy()
            times = data['frame_received_monotonic'].copy()
        targets = []
        previous = None
        for body, stamp, now in zip(frames, stamps, times):
            self.now = float(now)
            self.stamp = int(stamp)
            _, _, state = self.tick(dict(raw_body_poses=body, pico_timestamp_ns=self.stamp,
                                         received_monotonic=self.now))
            if previous is not None:
                dt = min(float(now)-previous.now, .02)
                self.assertLessEqual(np.max(np.abs(state.q-previous.q)), .8*dt+1e-12)
            previous = state
            targets.append(state.q)
        targets = np.asarray(targets)
        limits = np.asarray(self.head.settings.limits)
        self.assertGreater(np.max(np.abs(targets)), .2)
        self.assertTrue(np.all(targets >= limits[:, 0]))
        self.assertTrue(np.all(targets <= limits[:, 1]))


if __name__ == '__main__':
    unittest.main()
