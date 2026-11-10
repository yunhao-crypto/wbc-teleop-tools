"""Versioned model contracts: reject incompatible semantics before robot IO."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from deployment.vita_s100_aorta_30dof.actor_pico_runtime import sdk_math
from deployment.vita_s100_aorta_30dof.actor_morphmimic import MorphMimicContract, policy_class
from deployment.vita_s100_aorta_30dof.actor_supermimic import SuperMimicContract

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/actor_pico'


class CandidateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config/actor_pico.json').read_text())
        cls.math = sdk_math(ROOT, cls.config)
        cls.morph = json.loads((FIXTURES/'morphmimic_sonic_137670_contract.json').read_text())
        cls.supermimic = json.loads((FIXTURES/'supermimic_v2_contract.json').read_text())

    def morph_contract(self, raw=None):
        return MorphMimicContract.from_metadata(self.math, {
            'morphmimic.tracking_contract.v1': json.dumps(self.morph if raw is None else raw)})

    def super_contract(self, raw=None):
        return SuperMimicContract.from_metadata({
            'supermimic.tracking_contract.v1': json.dumps(self.supermimic if raw is None else raw)})

    def test_morph_candidate_routes_to_native_workspace_without_changing_pd(self):
        c = self.morph_contract()
        self.assertEqual((c.input_dim, c.history_length, c.future_frames), (1600, 10, 10))
        self.assertTrue(c.reference_feasibility_guard)
        np.testing.assert_array_equal(c.kp[c.ankle.slots], 20.)
        np.testing.assert_array_equal(c.kd[c.ankle.slots], 1.)
        i = c.names.index('waist_pitch_joint')
        self.assertEqual((c.kp[i], c.kd[i]), (90., 5.))
        profile = copy.deepcopy(self.config['body_profile'])
        profile['ankle_command_space'] = 'motor_theta'
        policy = policy_class(self.math)(c, profile)
        command, decoded = policy.decode(np.full(28, 30.))
        expected = c.policy_default+20*c.scale
        np.testing.assert_allclose([command.joints[i].q for i in policy.mapping], expected, rtol=1e-6)
        self.assertTrue(np.any(expected > c.policy_upper))
        self.assertFalse(any(decoded['clamped']))

    def test_morph_transfer_provenance_cannot_be_dropped_or_relabelled(self):
        for section, key, value in (
            ('execution_transfer', 'source_checkpoint_sha256', '0'*64),
            ('execution_transfer', 'actor_weights_unchanged', False),
            ('training_contract', 'version', 4),
            ('execution_contract', 'version', 3),
            (None, 'pd_target_semantics', 'bounded'),
            (None, 'robot_profile', 'unknown')):
            with self.subTest(section=section, key=key):
                raw = copy.deepcopy(self.morph)
                (raw if section is None else raw[section])[key] = value
                with self.assertRaises(ValueError):
                    self.morph_contract(raw)
        raw = copy.deepcopy(self.morph)
        del raw['execution_contract']
        with self.assertRaises(ValueError):
            self.morph_contract(raw)

    def test_native_workspace_preserves_valid_reference_outside_legacy_box(self):
        c = self.morph_contract()
        q = np.tile(c.default, (3, 1))
        for p, r, sign in c.ankle.legs:
            q[:, p], q[:, r] = -.8, 0.
        out, changed = c.ankle.project_reference(q)
        np.testing.assert_array_equal(out, q)
        self.assertFalse(changed.any())
        motors = c.ankle.to_policy(out)[:, c.ankle.slots]
        self.assertTrue(np.all(motors >= c.ankle.motor_limits[0]))
        self.assertTrue(np.all(motors <= c.ankle.motor_limits[1]))

    def test_native_workspace_rechecks_mutated_reference_and_projects_coupled_limit(self):
        c = self.morph_contract()
        q = np.tile(c.default, (3, 1))
        c.ankle.project_reference(q)
        for p, r, sign in c.ankle.legs:
            q[:, p], q[:, r] = -.95, sign*.43
        before = q.copy()
        out, changed = c.ankle.project_reference(q)
        self.assertTrue(changed.all())
        np.testing.assert_array_equal(q, before)
        motors = c.ankle.to_policy(out)
        self.assertTrue(np.all(motors[:, c.ankle.slots] >= c.ankle.motor_limits[0]-1e-7))
        self.assertTrue(np.all(motors[:, c.ankle.slots] <= c.ankle.motor_limits[1]+1e-7))
        for pose, motor in zip(out, motors):
            recovered, residual = c.ankle.to_virtual(motor)
            self.assertLess(residual, 1e-7)
            np.testing.assert_allclose(recovered[c.ankle.slots], pose[c.ankle.slots], atol=1e-7)

    def test_supermimic_v2_loads_explicit_affine_contract(self):
        c = self.super_contract()
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
            raw = copy.deepcopy(self.supermimic)
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.super_contract(raw)
        for missing in ('command_target_clipping', 'joint_pos_limit_usage', 'joint_target_formula'):
            raw = copy.deepcopy(self.supermimic)
            del raw[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.super_contract(raw)
        raw = copy.deepcopy(self.supermimic)
        raw['goal']['zero_dp_v'] = True
        with self.assertRaises(ValueError):
            self.super_contract(raw)

    def test_supermimic_v2_rejects_overrides_that_disagree_with_exported_pd(self):
        for key, value in (('kp', 120.), ('kd', float('nan')), ('action_scale', True)):
            raw = copy.deepcopy(self.supermimic)
            raw['actuation_overrides']['waist_pitch_joint'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.super_contract(raw)


if __name__ == '__main__':
    unittest.main()
