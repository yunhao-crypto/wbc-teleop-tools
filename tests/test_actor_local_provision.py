"""Local provisioning boundary tests: no network, SDK, services or robot motion."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import actor_pico_local as local
from test_actor_pico_live import robot_config


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.machine = self.root/'machine-id'; self.machine.write_text('this-board')
        self.selector = self.root/'end_effector'; self.selector.write_text('none')
        self.config = self.root/'running.yaml'; self.config.write_text(json.dumps(robot_config()))
        self.profile = self.root/'outputs/actor_pico_local.json'
        self.arguments = dict(robot_id='192.0.2.15', robot_config=self.config,
            machine_id_path=self.machine, selector_path=self.selector, addresses=['192.0.2.15'])
        self.platform = patch.multiple(local.platform, system=lambda:'Linux', machine=lambda:'aarch64')
        self.platform.start(); self.addCleanup(self.platform.stop)

    def configure(self, **values):
        return local.configure_local_start(self.root, **dict(self.arguments, **values))

    def test_generated_profile_roundtrips_without_motion_authorization(self):
        result = self.configure()
        self.assertTrue(result['changed'])
        self.assertEqual(self.profile.stat().st_mode & 0o777, 0o600)
        saved = json.loads(self.profile.read_text())
        self.assertNotIn('confirm_motion', saved)
        self.assertEqual(saved['pico_handoff'], 'manual')
        args = local.load_local_start(self.root, machine_id_path=self.machine,
            selector_path=self.selector, addresses=['192.0.2.15'])
        self.assertEqual(args['robot_config'], self.config)
        self.assertEqual(args['seconds'], 0)
        self.assertFalse(args['auto_pico'])

    def test_identical_configuration_is_idempotent(self):
        self.configure()
        before = self.profile.read_bytes()
        self.assertFalse(self.configure()['changed'])
        self.assertEqual(self.profile.read_bytes(), before)

    def test_different_existing_configuration_is_preserved(self):
        self.configure()
        before = self.profile.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Preserving existing'):
            self.configure(seconds=5)
        self.assertEqual(self.profile.read_bytes(), before)

    def test_explicit_replace_keeps_private_backup(self):
        self.configure()
        before = self.profile.read_bytes()
        result = self.configure(seconds=5, replace=True)
        backup = Path(result['backup'])
        self.assertEqual(backup.read_bytes(), before)
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.profile.read_text())['seconds'], 5)

    def test_malformed_existing_profile_can_be_replaced_with_backup(self):
        self.profile.parent.mkdir(); self.profile.write_text('broken profile')
        result = self.configure(replace=True)
        self.assertEqual(Path(result['backup']).read_text(), 'broken profile')

    def test_other_host_or_invalid_config_cannot_write_profile(self):
        with self.assertRaisesRegex(ValueError, 'not a local'):
            self.configure(addresses=['192.0.2.16'])
        self.config.write_text('{}')
        with self.assertRaises(ValueError):self.configure()
        self.assertFalse(self.profile.exists())

    def test_symlink_is_not_overwritten(self):
        self.profile.parent.mkdir()
        sentinel = self.root/'sentinel'; sentinel.write_text('preserve')
        self.profile.symlink_to(sentinel)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.configure(replace=True)
        self.assertEqual(sentinel.read_text(), 'preserve')

    def test_empty_machine_id_and_invalid_duration_are_rejected(self):
        for seconds in (-1, 1.5, True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.configure(seconds=seconds)
        self.machine.write_text('')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.configure()
        self.assertFalse(self.profile.exists())


if __name__ == '__main__':
    unittest.main()
