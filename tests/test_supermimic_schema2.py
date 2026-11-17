"""Schema-2 ONNX contracts preserve unclipped native A/B PD setpoints offline."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import locomotion_aorta.wbc_tracking_policy as sdk

from deployment.vita_s100_aorta_30dof import actor_supermimic as sm

ROOT = Path(__file__).resolve().parents[1]


class Schema2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = json.loads((ROOT/'tests/fixtures/actor_pico/supermimic_contract_v1.json').read_text())
        cls.raw = dict(cls.legacy, schema_version=2, command_target_clipping='none',
                       joint_pos_limit_usage='reference_only_not_PD_target_clipping',
                       required_ankle_command_space='motor_theta')
        cls.profile = json.loads((ROOT/'config/actor_pico_supermimic.json').read_text())['body_profile']

    def contract(self, raw=None):
        return sm.SuperMimicContract.from_metadata({sm.CONTRACT_KEY: json.dumps(self.raw if raw is None else raw)})

    def test_legacy_schema_still_loads_and_schema2_declares_native_requirement(self):
        self.assertIsNone(self.contract(self.legacy).required_ankle_command_space)
        self.assertEqual(self.contract().required_ankle_command_space, 'motor_theta')

    def test_incomplete_or_conflicting_schema2_is_rejected(self):
        for key, value in [('command_target_clipping', 'deployment'),
                           ('joint_pos_limit_usage', 'PD_target_clipping'),
                           ('required_ankle_command_space', 'virtual_pitch_roll')]:
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    raw = copy.deepcopy(self.raw)
                    if missing:
                        raw.pop(key)
                    else:
                        raw[key] = value
                    with self.assertRaises(ValueError):
                        self.contract(raw)
        for version in (True, 2., 3):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.contract(dict(self.raw, schema_version=version))

    def test_native_decode_preserves_out_of_range_targets_and_gains(self):
        c = self.contract()
        for space in ('policy', 'motor_theta'):
            profile = dict(self.profile, ankle_command_space=space)
            policy = sm.policy_class(sdk)(c, profile, past_frames=8, tail_frames=1)
            for sign in (-1., 1.):
                with self.subTest(space=space, sign=sign), patch.object(
                        c.ankle, 'to_virtual', side_effect=AssertionError('native decoder ran FK')):
                    command, row = policy.decode(np.full(28, sign*100.))
                expected = c.policy_default+c.scale*(sign*c.action_clip)
                np.testing.assert_allclose(row['target'], expected)
                self.assertFalse(any(row['clamped']))
                for i, slot in enumerate(policy.mapping):
                    self.assertEqual(command.joints[slot].q, float(np.float32(expected[i])))
                    self.assertEqual(command.joints[slot].kp, float(np.float32(c.kp[i])))
                    self.assertEqual(command.joints[slot].kd, float(np.float32(c.kd[i])))
                    self.assertEqual(command.joints[slot].tau, 0.)

    def test_virtual_wire_rejected_before_startup_and_after_profile_mutation(self):
        c = self.contract()
        with self.assertRaisesRegex(ValueError, 'native motor_theta'):
            sm.policy_class(sdk)(c, dict(self.profile, ankle_command_space='virtual_pitch_roll'),
                                 past_frames=8, tail_frames=1)
        policy = sm.policy_class(sdk)(c, dict(self.profile), past_frames=8, tail_frames=1)
        policy.profile['ankle_command_space'] = 'virtual_pitch_roll'
        with self.assertRaisesRegex(ValueError, 'native motor_theta'):
            policy.decode(np.zeros(28))


if __name__ == '__main__':
    unittest.main()
