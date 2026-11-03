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
from deployment.vita_s100_aorta_30dof import actor_parallel_ankle as pa
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
        # Closed-loop checkpoint: ankle slots hold motor angles under motor PD.
        self.assertIsNotNone(self.c.ankle)
        np.testing.assert_array_equal(self.c.kp[self.c.ankle.slots], 20)
        np.testing.assert_array_equal(self.c.kd[self.c.ankle.slots], 1)

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
                                    int(frame), self.c.future_frames, ankle=self.c.ankle)
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
        np.testing.assert_allclose(goal[:, 15:43], np.repeat(self.c.policy_default[None], 10, axis=0), atol=1e-9)
        np.testing.assert_allclose(frames[0, :28], 0, atol=1e-9)
        second = self.policy.observe(hold, 0., self.c.default, np.full(28, .5), [1, 0, 0, 0], np.zeros(3))
        np.testing.assert_array_equal(second[:864], first[96:960])
        rates = self.c.ankle.velocity_to_policy(self.c.default, np.full(28, .5))
        np.testing.assert_allclose(second[864+28:864+56], rates.astype(np.float32))
        self.assertFalse(np.allclose(rates[self.c.ankle.slots], .5))

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
        raw = self.c.policy_default+self.c.scale*np.clip(action, -20, 20).astype(np.float32).astype(np.float64)
        slots = self.c.ankle.slots
        motor = raw.copy()
        motor[slots] = np.clip(raw[slots], *self.c.ankle.motor_limits)
        safe = np.clip(self.c.ankle.to_virtual(motor)[0], self.policy.lower, self.policy.upper)
        for i, name in enumerate(self.c.names):
            joint = command.joints[self.policy.order.index(name)]
            self.assertEqual(joint.q, float(np.float32(safe[i])))
            self.assertEqual((joint.kp, joint.kd), (float(np.float32(self.c.kp[i])), float(np.float32(self.c.kd[i]))))
            self.assertEqual(joint.dq, 0.)
            self.assertEqual(joint.tau, float(np.float32(decoded['tau_ff'][i])))
            if 'ankle_' not in name:
                self.assertEqual(joint.tau, 0.)
        self.assertTrue(any(decoded['tau_ff']))
        np.testing.assert_array_equal(decoded['motor_target'], motor[slots])

    def test_decode_inside_the_ankle_box_reproduces_the_motor_target(self):
        """The robot's PD-A re-solves IK on the virtual target: same motor angles."""
        slots = self.c.ankle.slots
        rng, checked = np.random.default_rng(3), 0
        for _ in range(20):
            _, decoded = self.policy.decode(rng.uniform(-.7, .7, 28))
            if any(np.asarray(decoded['clamped'])[slots]):
                continue
            target, motor = np.asarray(decoded['target']), np.asarray(decoded['motor_target'])
            np.testing.assert_allclose(self.c.ankle.to_policy(target)[slots], motor, atol=1e-9)
            self.assertLess(decoded['ankle_fk_residual'], 1e-9)
            checked += 1
        self.assertGreater(checked, 10)

    def test_clamp_feedforward_restores_the_trained_motor_pd_torque(self):
        """Robot PD-A: kp_m*(IK(clamped target)-theta) + J^-T tau. Training: kp_m*(motor target-theta)."""
        slots, legs, g = self.c.ankle.slots, self.c.ankle.legs, self.c.ankle.geometry
        rng = np.random.default_rng(5)
        for _ in range(20):
            _, decoded = self.policy.decode(rng.uniform(-4, 4, 28))
            target, tau = np.asarray(decoded['target']), np.asarray(decoded['tau_ff'])
            theta = rng.uniform(-.5, .3, 28)
            for p, r, sign in legs:
                pitch, roll = target[p], sign*target[r]
                tracked = np.clip(pa.motor_angles(pitch, roll, g), *self.c.ankle.motor_limits)
                feedforward = np.linalg.solve(np.array(pa.jacobian(pitch, roll, g)).T, [sign*tau[r], tau[p]])
                robot = 20*(tracked-theta[[p, r]])+feedforward
                trained = 20*(np.asarray(decoded['motor_target'])[[list(slots).index(p), list(slots).index(r)]]
                              - theta[[p, r]])
                np.testing.assert_allclose(robot, trained, atol=1e-6)

    def test_clamp_feedforward_can_be_disabled(self):
        profile = dict(self.runtime.config['body_profile'], ankle_clamp_feedforward=False)
        policy = sm.create_policy(self.c, profile, self.math, self.runtime.config)
        command, decoded = policy.decode(np.full(28, 20.))
        self.assertFalse(any(decoded['tau_ff']))
        self.assertFalse(any(joint.tau for joint in command.joints))

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
                                        current, 10, ankle=self.c.ankle).reshape(10, 71)
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


class ParallelAnkleTests(unittest.TestCase):
    """NumPy port of the training map; FK inverts IK; the Jacobian is its derivative."""

    @classmethod
    def setUpClass(cls):
        cls.c = ActorPicoRuntime(ROOT, backend='auto', policy='supermimic').model.contract
        cls.g = cls.c.ankle.geometry

    def test_forward_inverts_motor_angles_over_the_virtual_box(self):
        for pitch in np.linspace(-.751, .414, 9):
            for roll in np.linspace(-.43625, .43625, 9):
                theta = pa.motor_angles(pitch, roll, self.g)
                p, r, residual = pa.forward(*theta, self.g)
                self.assertLess(residual, 1e-10)
                np.testing.assert_allclose((p, r), (pitch, roll), atol=1e-8)

    def test_jacobian_matches_finite_differences(self):
        h = 1e-6
        for pitch, roll in ((0., 0.), (-.5, .3), (.3, -.4)):
            J = np.array(pa.jacobian(pitch, roll, self.g))
            d_roll = (np.subtract(pa.motor_angles(pitch, roll+h, self.g), pa.motor_angles(pitch, roll-h, self.g)))/(2*h)
            d_pitch = (np.subtract(pa.motor_angles(pitch+h, roll, self.g), pa.motor_angles(pitch-h, roll, self.g)))/(2*h)
            np.testing.assert_allclose(J, np.stack((d_roll, d_pitch), axis=1), atol=1e-6)

    def test_roll_sign_mirrors_the_right_leg(self):
        q = np.zeros(28)
        names = list(self.c.names)
        for side in ('left', 'right'):
            q[names.index(f'{side}_ankle_pitch_joint')] = -.2
        q[names.index('left_ankle_roll_joint')], q[names.index('right_ankle_roll_joint')] = .1, -.1
        motor = self.c.ankle.to_policy(q)
        left = motor[[names.index('left_ankle_pitch_joint'), names.index('left_ankle_roll_joint')]]
        right = motor[[names.index('right_ankle_pitch_joint'), names.index('right_ankle_roll_joint')]]
        np.testing.assert_allclose(left, right, atol=1e-12)
        back, residual = self.c.ankle.to_virtual(motor)
        self.assertLess(residual, 1e-10)
        np.testing.assert_allclose(back, q, atol=1e-8)

    def test_contract_default_maps_to_the_trained_motor_default(self):
        np.testing.assert_allclose(self.c.ankle.to_policy(self.c.default), self.c.policy_default, atol=1e-9)
        lower, upper = self.c.ankle.motor_limits
        np.testing.assert_allclose(self.c.policy_lower[self.c.ankle.slots], lower)
        np.testing.assert_allclose(self.c.policy_upper[self.c.ankle.slots], upper)


class SuperMimicLiveTests(test_actor_pico_live.LiveTests):
    """The full simulated live runner, unchanged, driving the SuperMimic runtime."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort', policy='supermimic')

    def test_motor_theta_robot_config_must_match_the_trained_ankle(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import load_robot_config
        self.assertFalse(load_robot_config(self.config, self.runtime)[0]['ankle_virtual_pd_intent'])
        # Local start configuration has no model and takes either mode.
        load_robot_config(self.config)
        for section, key, value, message in (
                ('pd', 'kp_motor', 40., 'motor gains'), ('pd', 'kd_motor', 2., 'motor gains'),
                ('caps', 'theta_max', .7, 'motor limits'), ('geometry', 'lx', .051, 'geometry'),
                ('virtual_limits', 'pitch', [-.95, .6], 'virtual ankle pitch')):
            config = test_actor_pico_live.robot_config(self.runtime)
            config['parallel_ankle'][section][key] = value
            self.config.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, message):
                load_robot_config(self.config, self.runtime)
        config = test_actor_pico_live.robot_config(self.runtime)
        config['parallel_ankle']['pd']['kp_motor'] = 40.
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'motor gains'):
            self.run_trial()
        self.assertEqual((self.transport.requests, self.transport.commands), ([], []))


if __name__ == '__main__':
    unittest.main()
