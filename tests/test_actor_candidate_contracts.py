"""SuperMimic contract regression; MimicLite has its own import/parity tests."""
import copy
import json
from pathlib import Path
import unittest

from deployment.vita_s100_aorta_30dof.actor_supermimic import SuperMimicContract

ROOT = Path(__file__).resolve().parents[1]


class CandidateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((ROOT/'tests/fixtures/actor_pico/supermimic_v2_contract.json').read_text())

    def contract(self, raw=None):
        return SuperMimicContract.from_metadata({
            'supermimic.tracking_contract.v1': json.dumps(self.raw if raw is None else raw)})

    def test_supermimic_v2_loads_explicit_affine_contract(self):
        c = self.contract()
        self.assertEqual(c.input_dim, 1670)
        self.assertIsNotNone(c.ankle)
        self.assertEqual(c.kp[c.names.index('waist_pitch_joint')], 90.)
        self.assertEqual(c.kd[c.names.index('waist_pitch_joint')], 5.)

    def test_supermimic_v2_rejects_different_execution_and_reference_semantics(self):
        for key, value in (
            ('schema_version', 3), ('schema_version', True),
            ('command_target_clipping', 'joint_range'),
            ('joint_pos_limit_usage', 'PD_target_clipping'),
            ('joint_target_formula', 'clip(default+scale*action)'),
            ('required_ankle_command_space', 'virtual_pitch_roll')):
            raw = copy.deepcopy(self.raw); raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.contract(raw)
        for key in ('command_target_clipping', 'joint_pos_limit_usage', 'joint_target_formula'):
            raw = copy.deepcopy(self.raw); del raw[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.contract(raw)
        raw = copy.deepcopy(self.raw); raw['goal']['zero_dp_v'] = True
        with self.assertRaises(ValueError): self.contract(raw)

    def test_supermimic_v2_rejects_overrides_that_disagree_with_exported_pd(self):
        for key, value in (('kp', 120.), ('kd', float('nan')), ('action_scale', True)):
            raw = copy.deepcopy(self.raw)
            raw['actuation_overrides']['waist_pitch_joint'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.contract(raw)


if __name__ == '__main__':
    unittest.main()
