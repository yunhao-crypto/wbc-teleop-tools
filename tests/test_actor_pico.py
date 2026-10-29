"""Numerical/reference regressions; no robot, network or control handles."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import (
    ActorPicoRuntime, load_actor_config, sdk_math,
)
from deployment.vita_s100_aorta_30dof.actor_pico_reference import (
    CausalActorReference,
)


class ActorPicoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='auto')
        cls.math = cls.runtime.math
        cls.c = cls.runtime.model.contract
        cls.config = cls.runtime.config

    def setUp(self):
        self.policy = self.math.WbcTrackingPolicy(self.c, self.config['body_profile'])

    def retarget(self, human_height=None):
        from deployment.vita_s100_aorta_30dof.actor_pico_gmr import create_retargeter
        config = dict(self.config, retargeting=dict(self.config['retargeting']))
        if human_height is not None:
            config['retargeting']['human_height_m'] = human_height
        return create_retargeter(ROOT, config, self.c, self.math)

    def body(self):
        b = np.zeros((24, 7), np.float32)
        b[:, 6] = 1
        return b

    def test_cpp_two_tick_observation_golden(self):
        # Independently generated C++ oracle uses 20 ms reference spacing.
        self.policy.contract = replace(self.c, future_dt=.02)
        data = json.loads((ROOT/'tests/fixtures/actor_pico/groot_observation_golden.json').read_text())
        motion = self.math.Motion(np.reshape(data['joint_pos'], (-1, 28)),
                                  np.reshape(data['base_pos'], (-1, 3)),
                                  np.reshape(data['base_quat'], (-1, 4)))
        for tick in data['ticks']:
            self.policy.last_action = np.array(tick['last_action'], np.float32)
            obs = self.policy.observe(motion, tick['ref_t'], tick['q'], tick['qd'],
                                      tick['robot_quat'], tick['base_ang_vel_body'],
                                      tick['yaw_offset_rad'])
            np.testing.assert_allclose(obs, tick['obs'], rtol=0, atol=1e-7)

    def test_body31_mapping_preserves_ankle_compensation(self):
        action = np.linspace(-30, 30, 28)
        command, decoded = self.policy.decode(action)
        clipped = np.clip(action, -20, 20).astype(np.float32)
        raw = self.c.default+self.c.scale*clipped.astype(np.float64)
        safe = np.clip(raw, self.policy.lower, self.policy.upper)
        self.assertEqual(len(command.joints), 31)
        for i, name in enumerate(self.c.names):
            j = command.joints[self.policy.order.index(name)]
            self.assertEqual(j.q, float(np.float32(safe[i])))
            self.assertEqual(j.kp, float(np.float32(self.c.kp[i])))
            tau = self.c.kp[i]*(raw[i]-safe[i]) if 'ankle_' in name else 0.
            self.assertEqual(j.tau, float(np.float32(tau)))
        self.assertTrue(any(decoded['tau_ff']))
        for name in ['neck_pitch_joint', 'neck_yaw_joint']:
            j = command.joints[self.policy.order.index(name)]
            self.assertEqual((j.q, j.kp, j.kd), (0., 20., 1.))
        hole = command.joints[self.policy.order.index('waist_roll_joint')]
        self.assertEqual((hole.q, hole.kp, hole.kd, hole.tau), (0., 0., 0., 0.))

    def test_causal_window_has_real_future_and_forward_velocities(self):
        # Keep both ABIs covered: frame count alone does not determine timing.
        for dt, delay in ((.02, .20), (.10, .92)):
            with self.subTest(future_dt=dt):
                c = replace(self.c, future_dt=dt)
                policy = self.math.WbcTrackingPolicy(c, self.config['body_profile'])
                ref = CausalActorReference(c, self.math)
                for i in range(ref.steps-1):
                    ref.append(1_000_000_000+i*20_000_000, np.full(28, i*.02), [1, 0, 0, 0],
                               received_monotonic=i*.02)
                self.assertFalse(ref.ready())
                ref.append(1_000_000_000+round(delay*1e9), np.full(28, delay), [1, 0, 0, 0],
                           received_monotonic=delay)
                self.assertTrue(ref.ready())
                obs = policy.observe(ref.motion(), 0., c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3))
                np.testing.assert_allclose(obs[:280].reshape(10, 28)[:, 0], np.arange(10)*dt, atol=1e-7)
                np.testing.assert_allclose(obs[280:560], 1., atol=1e-7)
                self.assertAlmostEqual(ref.delay, delay)

    def test_reference_rejects_invalid_frames_but_accepts_sparse_timestamps(self):
        ref = CausalActorReference(self.c, self.math)
        ref.append(1_000_000_000, self.c.default, [1, 0, 0, 0], received_monotonic=0.)
        for stamp in [1_000_000_000, 999_000_000]:
            with self.assertRaises(ValueError):
                ref.append(stamp, self.c.default, [1, 0, 0, 0], received_monotonic=.02)
        with self.assertRaises(ValueError):
            ref.append(1_020_000_000, np.full(28, np.nan), [1, 0, 0, 0], received_monotonic=.02)
        self.assertEqual(len(ref.samples), 1)
        ref.append(1_200_000_000,self.c.default,[1,0,0,0],received_monotonic=.2)
        self.assertTrue(ref.ready())
        self.assertEqual(len(ref.samples),2)

    def test_irregular_timestamps_are_interpolated_without_extrapolation(self):
        ref = CausalActorReference(self.c, self.math)
        for i in range(71):
            t = i*.015
            ref.append(1_000_000_000+i*15_000_000, np.full(28, 2*t),
                       self.math.yaw_quaternion(.2*t), received_monotonic=t)
        m = ref.motion()
        np.testing.assert_allclose(m.q[:, 0], 2*(1.05-ref.delay+np.arange(ref.steps)*.02), atol=1e-12)
        self.assertAlmostEqual(self.math.yaw(m.quat[-1]), .21, places=8)

    def test_gmr_matches_original_bridge_qpos_and_named_joint_order(self):
        rt = self.retarget(human_height=1.6)
        fixture = np.load(ROOT/'tests/fixtures/actor_pico/bridge_gmr_golden.npz')
        for body, expected in zip(fixture['body_poses'], fixture['qpos']):
            q, root, error = rt.solve(body)
            np.testing.assert_allclose(rt.last_qpos, expected, rtol=0, atol=1e-8)
            np.testing.assert_allclose(q, expected[rt.indices], rtol=0, atol=1e-8)
            np.testing.assert_allclose(root, self.math.quaternion(expected[3:7]), rtol=0, atol=1e-8)
            self.assertTrue(np.isfinite(error))
        # Model order is interleaved left/right; bridge wire wrists are ordered
        # roll/pitch/yaw. Verify named selection against explicit qpos addresses.
        self.assertNotEqual(rt.indices, sorted(rt.indices))

    def test_accelerated_gmr_is_bit_identical_to_the_pinned_engine(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_gmr import create_retargeter
        replay = np.load(ROOT/'tests/fixtures/actor_pico/pico_replay.npz')['body_poses']
        golden = np.load(ROOT/'tests/fixtures/actor_pico/bridge_gmr_golden.npz')['body_poses']
        for height, bodies in ((None, replay), (1.6, golden), (None, [self.body()]*5)):
            config = dict(self.config, retargeting=dict(self.config['retargeting']))
            if height is not None:
                config['retargeting']['human_height_m'] = height
            fast = create_retargeter(ROOT, config, self.c, self.math)
            pinned = create_retargeter(ROOT, config, self.c, self.math, accelerated=False)
            self.assertTrue(fast.accelerated)
            self.assertFalse(pinned.accelerated)
            self.assertEqual(type(pinned.engine).__name__, 'GeneralMotionRetargeting')
            for body in bodies:
                self.assertEqual(fast.processed_body(body), pinned.processed_body(body))
                a, b = fast.solve(body), pinned.solve(body)
                np.testing.assert_array_equal(fast.last_qpos, pinned.last_qpos)
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
                self.assertEqual(a[2], b[2])
                self.assertEqual(fast.last_error, max(pinned.engine.error1(), pinned.engine.error2()))

    def test_accelerated_gmr_refuses_other_dependencies_instead_of_falling_back(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_gmr
        from deployment.vita_s100_aorta_30dof.actor_pico_gmr import create_retargeter
        with patch.dict(actor_pico_gmr._ACCELERATED_VERSIONS, mink='0.0.0'):
            with self.assertRaisesRegex(RuntimeError, 'requires mink 0.0.0.*installed mink 1.3.0'):
                self.retarget()
            # The pinned engine is available only when requested explicitly.
            rt = create_retargeter(ROOT, self.config, self.c, self.math, accelerated=False)
        self.assertFalse(rt.accelerated)
        self.assertEqual(type(rt.engine).__name__, 'GeneralMotionRetargeting')
        rt.solve(self.body())
        self.assertTrue(np.isfinite(rt.last_qpos).all())

    def test_human_height_default_override_and_invalid_values(self):
        self.assertEqual(self.config['retargeting']['human_height_m'], 1.75)
        default = self.retarget()
        shorter = self.retarget(human_height=1.6)
        key = default.engine.human_root_name
        self.assertAlmostEqual(default.engine.human_scale_table[key] /
                               shorter.engine.human_scale_table[key], 1.75/1.6)
        for height in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.retarget(human_height=height)

    def test_gmr_sources_and_vita02a_settings_are_unmodified(self):
        import hashlib
        root = ROOT/'third_party/motion_stream_bridge'
        source = json.loads((root/'SOURCE.json').read_text())
        self.assertEqual(source['commit'], self.config['retargeting']['source_commit'])
        for item in source['files']:
            self.assertEqual(hashlib.sha256((root/item['path']).read_bytes()).hexdigest(), item['sha256'])
        rt = self.retarget()
        for name, default in zip(self.c.names, self.c.default):
            expected = rt.engine._default_qpos[rt.indices[list(self.c.names).index(name)]]
            self.assertAlmostEqual(expected, default, places=6)

    def test_invalid_pico_quaternion_and_nan_are_rejected(self):
        for body in [np.zeros((24, 7)), np.full((24, 7), np.nan), np.zeros((23, 7))]:
            with self.assertRaises(ValueError):
                self.retarget().solve(body)

    def test_incompatible_model_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            self.math.Contract.from_metadata({})

    def test_real_actor_inference_produces_finite_body31(self):
        r = self.runtime
        r.policy.reset()
        obs, row = r.infer_motion(self.math.Motion.hold(self.c.default), q=self.c.default,
                                  dq=np.zeros(28), root=[1, 0, 0, 0], gyro=np.zeros(3))
        self.assertEqual(obs.shape, (1520,))
        self.assertEqual(len(row['raw_action']), 28)
        self.assertEqual(len(row['body_targets']), 31)
        self.assertEqual(row['actions_sent'], 0)
        self.assertTrue(np.isfinite(row['raw_action']).all())
        r.policy.reset()

    def test_delivered_actor_is_pinned_low_latency_g1(self):
        raw = json.loads((ROOT/'tests/fixtures/actor_pico/actor_g1_contract.json').read_text())
        self.assertEqual(raw['encoder_name'], 'g1')
        self.assertEqual((self.c.future_frames, self.c.future_dt, self.c.period), (10, .02, .02))
        self.assertAlmostEqual(CausalActorReference(self.c, self.math).delay, .20)

    def test_stale_source_cannot_infer(self):
        with self.assertRaises(TimeoutError):
            self.runtime.infer(q=self.c.default, dq=np.zeros(28), root=[1, 0, 0, 0],
                               gyro=np.zeros(3), now_monotonic=100.)

    def test_entire_pico_to_actor_path_without_transport(self):
        import aorta
        import locomotion_aorta
        with (patch.object(aorta.Node, '__init__', side_effect=AssertionError('Node forbidden')),
              patch.object(locomotion_aorta.PullController, '__init__', side_effect=AssertionError('Control forbidden'))):
            runtime = ActorPicoRuntime(ROOT, backend='auto')
            for i in range(48):
                runtime.push_pico(self.body(), 1_000_000_000+i*20_000_000,
                                  received_monotonic=i*.02, robot_root=[1, 0, 0, 0])
            obs, row = runtime.infer(q=self.c.default, dq=np.zeros(28),
                                     root=[1, 0, 0, 0], gyro=np.zeros(3), now_monotonic=47*.02)
            self.assertEqual(obs.shape, (1520,))
            self.assertEqual(row['actions_sent'], 0)
            self.assertEqual(len(row['body_targets']), 31)
            with self.assertRaises(TimeoutError):
                runtime.infer(q=self.c.default, dq=np.zeros(28), root=[1, 0, 0, 0],
                              gyro=np.zeros(3), now_monotonic=1.55)


if __name__ == '__main__':
    unittest.main()
