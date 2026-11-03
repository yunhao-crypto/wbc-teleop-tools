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


def install_motor_theta(runtime):
    """Point a runtime at a motor_theta contract on the delivered network weights."""
    runtime.model.contract = ga.contract_from_metadata(runtime.math, metadata(motor_theta_contract()))
    runtime.policy = ga.create_policy(runtime.model.contract, runtime.config['body_profile'], runtime.math)
    runtime.reset_inputs()


class GrootAnkleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='auto')
        cls.math, cls.profile = cls.runtime.math, cls.runtime.config['body_profile']
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

    def test_delivered_virtual_model_keeps_the_sdk_contract_and_policy(self):
        self.assertIs(type(self.runtime.model.contract), self.math.Contract)
        self.assertIs(type(self.runtime.policy), self.math.WbcTrackingPolicy)
        self.assertIs(type(ga.contract_from_metadata(self.math, metadata(serial_contract()))), self.math.Contract)

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
        # Motor targets stop at the exported soft limits.
        _, decoded = ours.decode(np.full(28, 20.))
        np.testing.assert_allclose(decoded['motor_target'], SOFT_LIMITS[1])

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
        self.assertEqual(entry.ankle_type(runtime), dict(ankle_control_space='virtual_pitch_roll',
                                                         required_ankle_virtual_pd_intent=True))
        install_motor_theta(runtime)
        self.assertEqual(entry.ankle_type(runtime), dict(ankle_control_space='motor_theta',
                                                         required_ankle_virtual_pd_intent=False))


class GrootMotorThetaLiveTests(test_actor_pico_live.LiveTests):
    """The full simulated live runner driving a motor_theta GR00T contract."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort')
        install_motor_theta(cls.runtime)

    def test_complete_candidate_replay_and_feedback_evidence(self):
        """In-process replay: replay_actor_candidate.py rebuilds the pinned (virtual) model."""
        self.online_setup(); self.args.seconds = 1.
        self.auto_trial()
        candidate = next(r for r in self.rows if r.get('record_type') == 'prepared_policy_candidate')
        runtime = ActorPicoRuntime(ROOT, backend='ort')
        install_motor_theta(runtime)
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
