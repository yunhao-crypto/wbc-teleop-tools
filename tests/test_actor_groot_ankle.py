"""GR00T/SONIC ankle control space chosen from ONNX metadata; no robot or network."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorModel, ActorPicoRuntime
from deployment.vita_s100_aorta_30dof import actor_groot_ankle as ga
from deployment.vita_s100_aorta_30dof import actor_parallel_ankle as pa
import test_actor_pico_live

# Ankle block of GR00T's vita02a_closed_loop export (paired fine-tune, step 100).
ANKLE = {
    'schema_version': 3, 'control_space': 'motor_theta', 'robot_type': 'vita02a_closed_loop',
    'reference_conversion': 'virtual_pitch_roll_to_motor_theta',
    'geometry': dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235),
    'motor_limits': [-.95975, .61075], 'roll_sign': dict(left=1, right=-1),
    'slots': {f'{side}_ankle_{axis}_joint': dict(joint=f'{side}_{label}_achilles_joint', motor=motor, side=side)
              for side in ('left', 'right')
              for axis, label, motor in (('pitch', 'A', 'theta_top'), ('roll', 'B', 'theta_bottom'))},
}
# The export's 0.9 soft joint limits inside the motor travel.
SOFT_LIMITS = (-.8812225, .5322225)
MOTOR_SCALE = .43857731389829635


def serial_contract():
    return json.loads((ROOT/'tests/fixtures/actor_pico/actor_g1_contract.json').read_text())


def motor_theta_contract(raw=None, ankle=ANKLE, limits=SOFT_LIMITS):
    """A closed-loop export of the delivered G1 layout: motor angles in the ankle slots."""
    raw = copy.deepcopy(raw or serial_contract())
    slots = pa.MotorThetaAnkle(raw['joint_names'], ANKLE).slots
    default = pa.MotorThetaAnkle(raw['joint_names'], ANKLE).to_policy(raw['default_joint_pos'])
    for key, value in (('default_joint_pos', default[slots]), ('joint_pos_limit_lower', limits[0]),
                       ('joint_pos_limit_upper', limits[1]), ('joint_stiffness', 20.),
                       ('joint_damping', 1.), ('action_scale', MOTOR_SCALE)):
        array = np.array(raw[key], dtype=np.float64)
        array[slots] = value
        raw[key] = array.tolist()
    raw.update(robot_type='vita02a_closed_loop', ankle=copy.deepcopy(ankle))
    return raw


def metadata(raw):
    return {ga.CONTRACT_KEY: json.dumps(raw)}


def install_motor_theta(runtime, command_space='virtual_pitch_roll'):
    """Point a runtime at a motor_theta contract on the delivered network weights."""
    runtime.model.contract = ga.contract_from_metadata(runtime.math, metadata(motor_theta_contract()))
    runtime.config['body_profile']['ankle_command_space']=command_space
    runtime.policy = ga.create_policy(runtime.model.contract, runtime.config['body_profile'], runtime.math)
    runtime.reset_inputs()


class GrootAnkleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='auto')
        cls.math, cls.profile = cls.runtime.math, dict(cls.runtime.config['body_profile'],
                                                     ankle_command_space='virtual_pitch_roll')
        cls.raw = motor_theta_contract()
        cls.c = ga.contract_from_metadata(cls.math, metadata(cls.raw))
        # The same export read as if every slot were a serial joint: the policy-space reference.
        cls.serial = cls.math.Contract.from_metadata(metadata(dict(cls.raw, ankle=dict(control_space='virtual_pitch_roll'))))

    def motion(self, n=30):
        rng = np.random.default_rng(3)
        q = self.c.default+.05*np.cumsum(rng.standard_normal((n, 28)), axis=0)/math.sqrt(n)
        yaw = np.linspace(0, .3, n)
        quat = np.stack([np.cos(yaw/2), np.zeros(n), np.zeros(n), np.sin(yaw/2)], axis=1)
        return self.math.Motion(q, np.cumsum(np.full((n, 3), .01), axis=0), quat)

    def test_default_sonic_is_pinned_9700_native_motor_policy(self):
        model = self.runtime.config['artifacts']['model']
        self.assertEqual(model['sha256'], '2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d')
        self.assertEqual(model['source_name'], 'model_step_009700_longrun_closed_loop_g1.onnx')
        self.assertEqual(pa.command_space(self.runtime.policy), 'motor_theta')
        self.assertEqual(self.runtime.model.contract.action_clip, 20.)
        self.assertFalse(self.runtime.policy.clip_target_positions)
        self.assertFalse(self.runtime.config['body_profile']['ankle_clamp_feedforward'])

    def test_virtual_contract_keeps_sdk_layout_with_affine_pd_policy(self):
        # Keep this independent of the locally deployed (motor-theta) model.
        contract = ga.contract_from_metadata(self.math, metadata(serial_contract()))
        self.assertIs(type(contract), self.math.Contract)
        policy = ga.create_policy(contract, self.profile, self.math)
        self.assertIsInstance(policy, self.math.WbcTrackingPolicy)
        self.assertFalse(policy.clip_target_positions)

    def test_virtual_policy_does_not_require_clamp_compensation(self):
        contract = ga.contract_from_metadata(self.math, metadata(serial_contract()))
        for value in (False, None, True):
            policy = ga.create_policy(contract, dict(self.profile, ankle_clamp_feedforward=value), self.math)
            self.assertFalse(policy.clip_target_positions)
        ga.create_policy(self.c, dict(self.profile, ankle_clamp_feedforward=False), self.math)

    def test_virtual_affine_targets_preserve_policy_pd_torque(self):
        contract = ga.contract_from_metadata(self.math, metadata(serial_contract()))
        policy = ga.create_policy(contract, dict(self.profile, ankle_clamp_feedforward=True), self.math)
        rng = np.random.default_rng(57400)
        for sign in (-1., 1.):
            action = np.zeros(28)
            action[policy.ankles] = sign * contract.action_clip
            command, decoded = policy.decode(action)
            q = rng.uniform(policy.lower, policy.upper)
            dq = rng.normal(size=28)
            raw_target = contract.default + contract.scale * np.float32(action).astype(float)
            for i in policy.ankles:
                joint = command.joints[policy.mapping[i]]
                self.assertAlmostEqual(joint.q, raw_target[i], delta=1e-6)
                self.assertEqual(joint.tau, 0.)
                actual = joint.kp * (joint.q - q[i]) - joint.kd * dq[i]
                expected = contract.kp[i] * (raw_target[i] - q[i]) - contract.kd[i] * dq[i]
                self.assertAlmostEqual(actual, expected, delta=5e-5)
            np.testing.assert_array_equal(decoded['tau_ff'], np.zeros(28))

    def test_motor_theta_metadata_selects_motor_space(self):
        c, slots = self.c, self.c.ankle.slots
        self.assertTrue(ga.is_groot_motor_theta(c))
        np.testing.assert_allclose(c.ankle.to_policy(c.default), c.policy_default, atol=1e-9)
        np.testing.assert_allclose(c.policy_lower[slots], SOFT_LIMITS[0])
        np.testing.assert_allclose(c.policy_upper[slots], SOFT_LIMITS[1])
        np.testing.assert_array_equal(c.lower[slots], -math.pi/2)
        np.testing.assert_array_equal(c.upper[slots], math.pi/2)
        others = np.setdiff1d(np.arange(28), slots)
        np.testing.assert_array_equal(c.default[others], self.serial.default[others])
        self.assertIsInstance(ga.create_policy(c, self.profile, self.math), self.math.WbcTrackingPolicy)

    def test_unsupported_or_inconsistent_ankle_metadata_is_rejected(self):
        for raw in (dict(self.raw, ankle=dict(ANKLE, control_space='motor_torque')),
                    dict(self.raw, ankle=dict(ANKLE, schema_version=2)),
                    dict(self.raw, ankle=dict(ANKLE, roll_sign=dict(left=1, right=1))),
                    motor_theta_contract(limits=(-1., .5)),
                    dict(self.raw, input_dim=1519)):
            with self.assertRaises(ValueError):
                ga.contract_from_metadata(self.math, metadata(raw))
        with self.assertRaises(ValueError):
            ga.contract_from_metadata(self.math, {})

    def test_observation_is_the_sdk_layout_in_policy_space(self):
        """Training PolicyJoint terms: IK of measured/reference poses, rates through the Jacobian."""
        c, motion = self.c, self.motion()
        ours = ga.create_policy(c, self.profile, self.math)
        sdk = self.math.WbcTrackingPolicy(self.serial, self.profile)
        motor_motion = self.math.Motion(c.ankle.to_policy(motion.q), motion.base, motion.quat)
        rng = np.random.default_rng(5)
        for tick, ref_time in enumerate((.1, .12, .44)):
            q = motion.q[tick]+.01*rng.standard_normal(28)
            dq, gyro = rng.standard_normal(28), .1*rng.standard_normal(3)
            quat = self.math.multiply(self.math.yaw_quaternion(.05), [math.cos(.05), math.sin(.05), 0., 0.])
            action = rng.uniform(-1, 1, 28).astype(np.float32)
            ours.last_action, sdk.last_action = action.copy(), action.copy()
            obs = ours.observe(motion, ref_time, q, dq, quat, gyro, .2)
            expected = sdk.observe(motor_motion, ref_time, c.ankle.to_policy(q), c.ankle.velocity_to_policy(q, dq),
                                   quat, gyro, .2)
            indices = motion.indices(ref_time, c.future_frames, c.future_dt)
            rates = c.ankle.velocity_to_policy(motion.q[indices], motion.velocities(indices))
            count = c.future_frames
            block = expected[28*count:56*count].reshape(count, 28)
            block[:, c.ankle.slots] = rates[:, c.ankle.slots]
            np.testing.assert_allclose(obs, expected, rtol=0, atol=2e-6)

    def test_decode_commands_the_virtual_pose_of_the_motor_targets(self):
        c = self.c
        ours = ga.create_policy(c, self.profile, self.math)
        sdk = self.math.WbcTrackingPolicy(self.serial, self.profile)
        action = np.random.default_rng(7).uniform(-.3, .3, 28)
        command, decoded = ours.decode(action)
        _, serial = sdk.decode(action)
        slots, others = c.ankle.slots, np.setdiff1d(np.arange(28), c.ankle.slots)
        motor = c.policy_default+c.scale*np.float32(action).astype(np.float64)
        np.testing.assert_allclose(decoded['motor_target'], motor[slots], atol=1e-12)
        target = np.array(decoded['target'])
        np.testing.assert_allclose(c.ankle.to_policy(target)[slots], motor[slots], atol=1e-8)
        np.testing.assert_allclose(target[others], np.array(serial['target'])[others], atol=1e-12)
        # Inside the virtual box nothing is clamped, so no feedforward beyond FK round-off.
        np.testing.assert_allclose(decoded['tau_ff'], 0., atol=1e-6)
        for p, slot in enumerate(ours.mapping):
            self.assertAlmostEqual(command.joints[slot].q, target[p], places=6)
            self.assertEqual((command.joints[slot].kp, command.joints[slot].kd), (float(c.kp[p]), float(c.kd[p])))
        # A virtual wire cannot represent arbitrary motor PD setpoints. It
        # must reject an unreachable request instead of silently projecting it.
        with self.assertRaisesRegex(ValueError, 'motor_theta command space'):
            ours.decode(np.full(28, 20.))

    def test_serial_decode_preserves_pd_targets_beyond_position_limits(self):
        policy = ga.create_policy(self.serial, self.profile, self.math)
        for sign in (-1., 1.):
            command, row = policy.decode(np.full(28, sign*100.))
            expected = self.serial.default+self.serial.scale*(sign*self.serial.action_clip)
            np.testing.assert_array_equal(policy.last_action, sign*self.serial.action_clip)
            np.testing.assert_allclose(row['target'], expected)
            self.assertFalse(any(row['clamped']))
            self.assertFalse(any(row['tau_ff']))
            self.assertTrue(np.any((expected < policy.lower) | (expected > policy.upper)))
            for i, slot in enumerate(policy.mapping):
                self.assertEqual(command.joints[slot].q, float(np.float32(expected[i])))

    def test_motor_decode_ignores_soft_hard_and_profile_target_bounds(self):
        profile = dict(self.profile, ankle_command_space='motor_theta')
        policy = ga.create_policy(self.c, profile, self.math)
        for sign in (-1., 1.):
            command, row = policy.decode(np.full(28, sign*100.))
            expected = self.c.policy_default+self.c.scale*(sign*self.c.action_clip)
            np.testing.assert_allclose(row['target'], expected)
            np.testing.assert_allclose(row['motor_target'], expected[self.c.ankle.slots])
            self.assertFalse(any(row['clamped']))
            self.assertFalse(any(row['tau_ff']))
            for i, slot in enumerate(policy.mapping):
                self.assertEqual(command.joints[slot].q, float(np.float32(expected[i])))

    def test_virtual_wire_preserves_representable_motor_targets_above_soft_limits(self):
        policy = ga.create_policy(self.c, self.profile, self.math)
        action = np.zeros(28)
        slots = self.c.ankle.slots
        action[slots] = (.6-self.c.policy_default[slots])/self.c.scale[slots]
        _, row = policy.decode(action)
        np.testing.assert_allclose(row['motor_target'], .6, atol=1e-7)
        np.testing.assert_allclose(self.c.ankle.to_policy(row['target'])[slots], .6, atol=1e-7)
        self.assertFalse(any(row['clamped']))
        self.assertFalse(any(row['tau_ff']))

    @unittest.skipUnless(__import__('importlib').util.find_spec('onnx'), 'onnx is needed to rewrite metadata')
    def test_runtime_model_picks_the_control_space_from_the_onnx(self):
        import onnx
        model = onnx.load(str(ROOT/self.runtime.config['artifacts']['model']['path']))
        for prop in model.metadata_props:
            if prop.key == ga.CONTRACT_KEY:
                prop.value = json.dumps(self.raw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'closed_loop.onnx'
            onnx.save(model, str(path))
            actor = ActorModel(path, self.math, backend='ort')
        self.assertTrue(ga.is_groot_motor_theta(actor.contract))
        policy = ga.create_policy(actor.contract, self.profile, self.math)
        self.assertIsNot(type(policy), self.math.WbcTrackingPolicy)
        obs = policy.observe(self.math.Motion.hold(actor.contract.default), 0., actor.contract.default,
                             np.zeros(28), [1, 0, 0, 0], np.zeros(3))
        self.assertTrue(np.isfinite(actor.infer(obs)).all())
        # The standing default observes zero joint offsets in policy space too.
        start = 10*(56+6)+10*3
        np.testing.assert_allclose(obs[start:start+280], 0., atol=1e-6)


class AnkleTypeReportTests(unittest.TestCase):
    def test_check_reports_the_ankle_type_and_required_robot_mode(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('actor_pico_entry', ROOT/'scripts/actor_pico.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        runtime = ActorPicoRuntime(ROOT, backend='ort')
        runtime.model.contract = ga.contract_from_metadata(runtime.math, metadata(serial_contract()))
        runtime.policy = ga.create_policy(runtime.model.contract, runtime.config['body_profile'], runtime.math)
        self.assertEqual(entry.ankle_type(runtime), dict(ankle_control_space='virtual_pitch_roll',
                                                         required_ankle_command_space='virtual_pitch_roll',
                                                         required_ankle_virtual_pd_intent=True))
        install_motor_theta(runtime)
        self.assertEqual(entry.ankle_type(runtime), dict(ankle_control_space='motor_theta',
                                                         required_ankle_command_space='virtual_pitch_roll',
                                                         required_ankle_virtual_pd_intent=False))


class GrootMotorThetaLiveTests(test_actor_pico_live.LiveTests):
    """The full simulated live runner driving a motor_theta GR00T contract."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort')
        install_motor_theta(cls.runtime, command_space='motor_theta')

    def test_complete_candidate_replay_and_feedback_evidence(self):
        """In-process replay: replay_actor_candidate.py rebuilds the pinned (virtual) model."""
        self.online_setup(); self.args.seconds = 1.
        self.auto_trial()
        candidate = next(r for r in self.rows if r.get('record_type') == 'prepared_policy_candidate')
        runtime = ActorPicoRuntime(ROOT, backend='ort')
        install_motor_theta(runtime, command_space='motor_theta')
        state, ref = candidate['input_state'], candidate['reference']
        command, obs, row = runtime.infer_motion_command(
            runtime.math.Motion(ref['q'], ref['base'], ref['quat']), q=state['q'], dq=state['dq'],
            root=state['root'], gyro=state['gyro'], ref_time=ref['ref_time'], yaw_offset=ref['yaw_offset'])
        np.testing.assert_allclose(obs, candidate['observation'], rtol=0, atol=1e-5)
        np.testing.assert_allclose(row['raw_action'], candidate['raw_action'], rtol=0, atol=1e-4)
        np.testing.assert_allclose([[getattr(j, k) for k in ('q', 'dq', 'tau', 'kp', 'kd')] for j in command.joints],
                                   [[j[k] for k in ('q', 'dq', 'tau', 'kp', 'kd')] for j in candidate['body_targets']],
                                   rtol=0, atol=1e-4)

    def test_motor_theta_needs_motor_pd_and_the_robot_mode_is_never_changed(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import load_robot_config
        self.assertFalse(load_robot_config(self.config, self.runtime)[0]['ankle_virtual_pd_intent'])
        config = test_actor_pico_live.robot_config(self.runtime)
        config['actions']['EXTERNAL']['params']['ankle_virtual_pd_intent'] = True
        self.config.write_text(json.dumps(config))
        before = self.config.read_bytes()
        with self.assertRaisesRegex(ValueError, 'ankle_virtual_pd_intent=false'):
            load_robot_config(self.config, self.runtime)
        with self.assertRaisesRegex(ValueError, 'ankle_virtual_pd_intent'):
            self.run_trial()
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual((self.transport.requests, self.transport.commands), ([], []))


if __name__ == '__main__':
    unittest.main()
