"""SuperMimic actor regressions against training-code fixtures; no robot or network."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime, load_actor_config
from deployment.vita_s100_aorta_30dof import actor_supermimic as sm
import test_actor_pico_live

FIXTURE = ROOT/'tests/fixtures/actor_pico/supermimic_fixture.npz'


class SuperMimicTests(unittest.TestCase):
    """Fixture: trackerlab export_vita_deployment.py, VitaMDP.obs and MotionLib.goal."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='auto', policy='supermimic')
        cls.math, cls.c = cls.runtime.math, cls.runtime.model.contract
        with np.load(FIXTURE, allow_pickle=False) as data:
            cls.fx = {k: data[k] for k in data.files}
        cls.motion = cls.math.Motion(cls.fx['q'], cls.fx['base'], cls.fx['quat_wxyz'])

    def setUp(self):
        self.policy = sm.create_policy(self.c, self.runtime.config['body_profile'],
                                       self.math, self.runtime.config)

    def test_fixture_is_from_the_configured_model(self):
        self.assertEqual(str(self.fx['onnx_sha256']), self.runtime.config['artifacts']['model']['sha256'])
        self.assertEqual(list(self.fx['joint_names']), list(self.c.names))
        self.assertEqual((self.c.input_dim, self.c.history_length, self.c.future_frames), (1670, 10, 10))

    def test_configs_share_robot_profile_retargeting_and_sdk(self):
        sonic, ours = load_actor_config(ROOT), load_actor_config(ROOT, 'supermimic')
        for key in ('body_profile', 'retargeting', 'sdk_sources', 'sdk_version', 'sdk_commit', 'pico_max_age_s'):
            self.assertEqual(sonic[key], ours[key], key)
        self.assertEqual({k: v for k, v in sonic['artifacts'].items() if k != 'model'},
                         {k: v for k, v in ours['artifacts'].items() if k != 'model'})
        with self.assertRaises(ValueError):
            sm.SuperMimicContract.from_metadata({})

    def test_goal_formula_matches_training_with_training_velocities(self):
        f = self.fx
        for i, frame in enumerate(f['goal_frames']):
            goal = sm.goal_features(*(f[k].astype(np.float64) for k in ('base', 'root_R', 'root_v', 'root_w', 'q', 'qd')),
                                    int(frame), self.c.future_frames)
            np.testing.assert_allclose(goal, f['goal_expected'][i], rtol=0, atol=2e-6)

    def test_windowed_reference_velocities_match_training(self):
        # Inside the clip the window is exact for joints and linear velocity;
        # angular labels carry the training formula's float32 noise.
        for i, frame in enumerate(self.fx['goal_frames']):
            goal = self.policy.goal(self.motion, int(frame)).reshape(10, 71)
            expected = self.fx['goal_expected'][i].reshape(10, 71)
            error = np.abs(goal-expected)
            np.testing.assert_allclose(goal[:, :9], expected[:, :9], rtol=0, atol=2e-6)
            np.testing.assert_allclose(goal[:, 15:], expected[:, 15:], rtol=0, atol=2e-6)
            self.assertLess(error[:, 9:12].max(), 2e-4)
            self.assertLess(error[:, 12:15].max(), .04)

    def test_observation_history_and_actions_match_training(self):
        f = self.fx
        self.policy.reset()
        for i, frame in enumerate(f['tick_frames']):
            self.policy.last_action = f['tick_last_action'][i].astype(np.float32)
            obs = self.policy.observe(self.motion, frame*self.c.period, f['tick_q'][i], f['tick_dq'][i],
                                      f['tick_quat_wxyz'][i], f['tick_gyro'][i])
            np.testing.assert_allclose(obs[:960], f['tick_input'][i][:960], rtol=0, atol=1e-6)
            goal, expected = obs[960:].reshape(10, 71), f['tick_input'][i][960:].reshape(10, 71)
            np.testing.assert_allclose(goal[:, :9], expected[:, :9], rtol=0, atol=2e-6)
            np.testing.assert_allclose(goal[:, 15:], expected[:, 15:], rtol=0, atol=2e-6)
            np.testing.assert_allclose(self.runtime.model.infer(f['tick_input'][i]), f['tick_action'][i],
                                       rtol=0, atol=2e-5)
            np.testing.assert_allclose(self.runtime.model.infer(obs), f['tick_action'][i], rtol=0, atol=.02)

    def test_history_repeats_first_frame_and_rolls_oldest_first(self):
        self.policy.reset()
        hold = self.math.Motion.hold(self.c.default)
        first = self.policy.observe(hold, 0., self.c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3))
        frames = first[:960].reshape(10, 96)
        np.testing.assert_array_equal(frames, np.repeat(frames[:1], 10, axis=0))
        np.testing.assert_allclose(frames[0, 59:62], [0, 0, -1])
        np.testing.assert_allclose(frames[0, 90:], [1, 0, 0, 1, 0, 0])
        goal = first[960:].reshape(10, 71)
        np.testing.assert_allclose(goal[:, :3], 0)
        np.testing.assert_allclose(goal[:, 9:15], 0)
        np.testing.assert_allclose(goal[:, 15:43], np.repeat(self.c.default[None], 10, axis=0))
        second = self.policy.observe(hold, 0., self.c.default, np.full(28, .5), [1, 0, 0, 0], np.zeros(3))
        np.testing.assert_array_equal(second[:864], first[96:960])
        np.testing.assert_allclose(second[864+28:864+56], .5)

    def test_reference_yaw_offset_rotates_only_the_orientation_observation(self):
        turn = self.math.yaw_quaternion(.7)
        yawed = self.math.Motion(self.motion.q, self.motion.base @ self.math.rotation(turn).T,
                                 [self.math.multiply(turn, q) for q in self.motion.quat])
        a = self.policy.observe(self.motion, .4, self.c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3), .7)
        self.policy.reset()
        b = self.policy.observe(yawed, .4, self.c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3))
        np.testing.assert_allclose(a[:960], b[:960], atol=1e-6)
        # Goals are in the reference's own heading frame: invariant to world yaw.
        np.testing.assert_allclose(a[960:], b[960:], atol=1e-5)

    def test_decode_reuses_sdk_body31_mapping_by_joint_name(self):
        action = np.linspace(-30, 30, 28)
        command, decoded = self.policy.decode(action)
        raw = self.c.default+self.c.scale*np.clip(action, -20, 20).astype(np.float32).astype(np.float64)
        safe = np.clip(raw, self.policy.lower, self.policy.upper)
        for i, name in enumerate(self.c.names):
            joint = command.joints[self.policy.order.index(name)]
            self.assertEqual(joint.q, float(np.float32(safe[i])))
            self.assertEqual((joint.kp, joint.kd), (float(np.float32(self.c.kp[i])), float(np.float32(self.c.kd[i]))))
            tau = self.c.kp[i]*(raw[i]-safe[i]) if 'ankle_' in name else 0.
            self.assertEqual(joint.tau, float(np.float32(tau)))
        self.assertTrue(any(decoded['tau_ff']))

    def test_causal_window_streams_the_clip_like_training(self):
        """Stream the clip as retargeted samples; compare goals at the current frame."""
        ref = sm.create_reference(self.c, self.math, self.runtime.config)
        past, tail = sm.reference_window(self.runtime.config)
        self.assertEqual((ref.steps, ref.current_index), (past+10+tail+1, past))
        self.assertAlmostEqual(ref.delay, .02*(10+tail))
        f, worst = self.fx, np.zeros(6)
        for i in range(len(f['q'])):
            ref.append(1_000_000_000+i*20_000_000, f['q'][i], f['quat_wxyz'][i],
                       base=f['base'][i], received_monotonic=i*.02)
            if not ref.ready():
                self.assertLess(i, ref.steps-1)
                continue
            current = i-10-tail
            motion = ref.motion()
            np.testing.assert_allclose(motion.q[past], f['q'][current], atol=1e-6)
            np.testing.assert_allclose(motion.base[past], f['base'][current], atol=1e-6)
            goal = self.policy.goal(motion, past).reshape(10, 71)
            expected = sm.goal_features(*(f[k].astype(np.float64) for k in ('base', 'root_R', 'root_v', 'root_w', 'q', 'qd')),
                                        current, 10).reshape(10, 71)
            error = np.abs(goal-expected)
            worst = np.maximum(worst, [error[:, s].max() for s in (slice(0, 3), slice(3, 9), slice(9, 12),
                                                                   slice(12, 15), slice(15, 43), slice(43, 71))])
        np.testing.assert_allclose(worst[[0, 1, 4]], 0, atol=2e-6)
        self.assertLess(worst[5], 2e-6)    # joint velocity: exact with one tail frame
        self.assertLess(worst[2], .05)     # root linear: filter truncated at the window end
        self.assertLess(worst[3], .2)      # root angular: also includes training label noise

    def test_jittered_window_matches_full_history_interpolation_and_stays_bounded(self):
        ref = sm.create_reference(self.c, self.math, self.runtime.config)
        f, rng = self.fx, np.random.default_rng(0)
        stamps = 1_000_000_000+np.cumsum(rng.integers(8_000_000, 35_000_000, len(f['q'])))
        for i, stamp in enumerate(stamps):
            ref.append(int(stamp), f['q'][i], f['quat_wxyz'][i], base=f['base'][i], received_monotonic=i*.02)
            self.assertLessEqual(len(ref.samples), ref.steps*3)
            if not ref.ready():
                continue
            times = (stamps[:i+1]-stamps[0])*1e-9
            query = times[-1]-ref.span+np.arange(ref.steps)*self.c.period
            motion = ref.motion()
            for name, values in (('q', f['q']), ('base', f['base'])):
                full = np.stack([np.interp(query, times, values[:i+1, j]) for j in range(values.shape[1])], axis=1)
                np.testing.assert_allclose(getattr(motion, name), full, rtol=0, atol=1e-12)
            for t, quat in zip(query, motion.quat):
                k = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, i-1))
                a = float(np.clip((t-times[k])/(times[k+1]-times[k]), 0., 1.))
                np.testing.assert_allclose(quat, self.math.slerp(f['quat_wxyz'][k], f['quat_wxyz'][k+1], a),
                                           rtol=0, atol=1e-12)

    def test_reference_requires_root_position_and_increasing_stamps(self):
        ref = sm.create_reference(self.c, self.math, self.runtime.config)
        with self.assertRaises(ValueError):
            ref.append(1_000_000_000, self.c.default, [1, 0, 0, 0], received_monotonic=0.)
        ref.append(1_000_000_000, self.c.default, [1, 0, 0, 0], base=[0, 0, .8], received_monotonic=0.)
        with self.assertRaises(ValueError):
            ref.append(1_000_000_000, self.c.default, [1, 0, 0, 0], base=[0, 0, .8], received_monotonic=.02)
        with self.assertRaises(RuntimeError):
            ref.motion()

    def test_entire_pico_to_actor_path_without_transport(self):
        runtime = ActorPicoRuntime(ROOT, backend='auto', policy='supermimic')
        body = np.zeros((24, 7), np.float32)
        body[:, 6] = 1
        steps = runtime.reference.steps
        for i in range(steps+5):
            runtime.push_pico(body, 1_000_000_000+i*20_000_000, received_monotonic=i*.02,
                              robot_root=[1, 0, 0, 0])
        motion = runtime.reference.motion()
        # The newest window sample carries the GMR root position.
        np.testing.assert_allclose(motion.base[-1], runtime.retarget.last_qpos[:3], atol=1e-12)
        obs, row = runtime.infer(q=self.c.default, dq=np.zeros(28), root=[1, 0, 0, 0],
                                 gyro=np.zeros(3), now_monotonic=(steps+4)*.02)
        self.assertEqual(obs.shape, (1670,))
        self.assertEqual((row['actions_sent'], len(row['body_targets'])), (0, 31))
        self.assertTrue(np.isfinite(row['raw_action']).all())


class SuperMimicLiveTests(test_actor_pico_live.LiveTests):
    """The full simulated live runner, unchanged, driving the SuperMimic runtime."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort', policy='supermimic')


if __name__ == '__main__':
    unittest.main()
