"""Official OP3 protocol regressions, including history and stream timing."""
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import sdk_math
from deployment.vita_s100_aorta_30dof import actor_supermimic_op3 as op3
from deployment.vita_s100_aorta_30dof.actor_supermimic import create_policy, create_reference


class OP3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config/actor_pico_supermimic_op3.json').read_text())
        cls.raw = json.loads((ROOT/'artifacts/supermimic_op3/actor.contract.json').read_text())
        cls.math = sdk_math(ROOT, cls.config)
        cls.c = op3.OP3Contract.from_metadata({op3.CONTRACT_KEY: json.dumps(cls.raw)})

    def setUp(self):
        self.policy = create_policy(self.c, copy.deepcopy(self.config['body_profile']), self.math, self.config)

    def obs(self, motion=None, **kw):
        values = dict(ref_time=0., q=self.c.default, dq=np.zeros(28), quat=[1,0,0,0], gyro=np.zeros(3))
        values.update(kw)
        return self.policy.observe(motion or self.math.Motion.hold(self.c.default), **values)

    def test_history_term_order_and_reset(self):
        first = self.obs(gyro=[1,2,3])
        np.testing.assert_array_equal(first[:30].reshape(10,3), [[0,0,-1]]*10)
        np.testing.assert_array_equal(first[30:60].reshape(10,3), [[1,2,3]]*10)
        second = self.obs(gyro=[4,5,6])
        np.testing.assert_array_equal(second[30:60].reshape(10,3), [[1,2,3]]*9+[[4,5,6]])
        self.policy.reset()
        third = self.obs(gyro=[7,8,9])
        np.testing.assert_array_equal(third[30:60].reshape(10,3), [[7,8,9]]*10)
        self.assertEqual(first.shape, (1760,))

    def test_native_goal_offsets_end_clamping_and_actual_root_yaw(self):
        m = self.math.Motion(np.repeat(self.c.default[None], 55, axis=0), np.zeros((55,3)), [[1,0,0,0]]*55)
        m.op3_joint_names = self.c.reference_names
        m.op3_joint_pos = np.repeat(np.arange(55)[:,None],40,axis=1).astype(float)
        m.op3_joint_vel = 2*m.op3_joint_pos+1
        obs = self.obs(m, ref_time=.4, quat=self.math.yaw_quaternion(.3), yaw_offset=.1)
        indices = np.minimum(20+np.array(op3.OFFSETS),54)
        expected = np.concatenate((m.op3_joint_pos[indices], m.op3_joint_vel[indices]),axis=1)
        np.testing.assert_array_equal(obs[900:1700].reshape(10,80), expected)
        rot = self.math.rotation(self.math.yaw_quaternion(-.2))[:, :2].ravel()
        np.testing.assert_allclose(obs[1700:].reshape(10,6), [rot]*10, atol=1e-7)
        self.assertEqual(obs[1699],109)  # clamped final velocity is retained, not zeroed

    def test_native_motor_targets_are_unclipped_and_not_double_converted(self):
        action = np.linspace(-25,25,28,dtype=np.float32)
        command, row = self.policy.decode(action)
        expected = self.c.policy_default+self.c.scale*action
        np.testing.assert_allclose(row['target'],expected)
        np.testing.assert_array_equal(self.policy.last_action,action)
        for i, slot in enumerate(self.policy.mapping):
            self.assertEqual(command.joints[slot].q,float(np.float32(expected[i])))
        self.policy.profile['ankle_command_space']='virtual_pitch_roll'
        with self.assertRaisesRegex(ValueError,'native motor_theta'):
            self.policy.decode(action)

    def test_reference_waits_for_whole_920ms_future_and_derivative_tail(self):
        ref = create_reference(self.c,self.math,self.config)
        self.assertAlmostEqual(ref.delay,.92)
        self.assertEqual(ref.steps,48)
        for i in range(47):
            ref.append(1_000_000_000+i*20_000_000,self.c.default,[1,0,0,0],
                       received_monotonic=i*.02,base=[0,0,.9])
        self.assertFalse(ref.ready())
        ref.append(1_940_000_000,self.c.default,[1,0,0,0],received_monotonic=.94,base=[0,0,.9])
        self.assertTrue(ref.ready())
        self.assertEqual(len(ref.motion().q),48)

    def test_strict_contract_rejects_old_layout_and_wire(self):
        for key,value in [('input_dim',1670),('history_order','frame_major_oldest_first'),
                          ('frame_offsets',list(range(10))),('required_ankle_command_space','virtual_pitch_roll'),
                          ('action_clip',20),('schema_version',True),('orientation_frame','reference_yaw')]:
            raw = dict(self.raw, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                op3.OP3Contract.from_metadata({op3.CONTRACT_KEY:json.dumps(raw)})
        with self.assertRaises(ValueError):
            op3.OP3Contract.from_metadata({'supermimic.tracking_contract.v1':json.dumps(self.raw)})

    def test_native_file_preserves_rates_and_rejects_broken_closure(self):
        q = np.repeat(self.c.default[None], 3, axis=0)
        qp, qv = op3.native_reference(q, np.full_like(q, .1), self.c)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'motion.npz'
            fields = dict(joint_pos=qp, joint_vel=qv, joint_names=self.c.reference_names,
                          fps=50., root_pos_w=np.zeros((3,3)), root_quat_w=[[1,0,0,0]]*3)
            np.savez(path, **fields)
            motion = op3.load_native_motion(path,self.c,self.math)
            np.testing.assert_array_equal(motion.op3_joint_vel,qv)
            fields['joint_pos'] = qp.copy()
            fields['joint_pos'][0,self.c.reference_names.index('left_A_achilles_joint')] += .1
            np.savez(path, **fields)
            with self.assertRaisesRegex(ValueError, 'closure'):
                op3.load_native_motion(path,self.c,self.math)

    def test_nonfinite_state_rejected_and_single_hold_is_stationary(self):
        obs = self.obs()
        np.testing.assert_array_equal(obs[900:1700].reshape(10,80)[:,40:],0)
        with self.assertRaises(ValueError):
            self.obs(gyro=[0,np.nan,0])
        with self.assertRaises(ValueError):
            self.obs(quat=[0,0,0,0])
        with self.assertRaises(ValueError):
            self.policy.decode(np.full(28,np.inf))
        with self.assertRaises(ValueError):
            self.policy.decode(np.full(28,1e300))


if __name__ == '__main__':
    unittest.main()
