"""Pinned client preparation preserves unrelated edits and is repeatable."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('viewer_prepare', ROOT/'scripts/prepare_pico_camera_viewer.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        ui = self.project/'Assets/Scripts/UI/UIOperate.cs'
        ui.parent.mkdir(parents=True)
        ui.write_text('// unrelated user comment\nif (AcontrolerTog != null && AcontrolerTog.isOn) {}\n')
        toggle = self.project/'Assets/Scripts/Li/toggle-objects-script.cs'
        toggle.parent.mkdir(parents=True)
        toggle.write_text('void Update()\n    {\n    }\n')
        actions = self.project/'Assets/InteractionTools/PICO Input Actions.inputactions'
        actions.parent.mkdir(parents=True)
        head = self.project/'Assets/InteractionTools/Scripts/HeadDevice.cs'
        head.parent.mkdir(parents=True)
        head.write_text('state.confirm = Input.GetKey(KeyCode.JoystickButton0);\n'
                        'state.cancel = Input.GetButton("Cancel");\n')
        actions.write_text(json.dumps({'maps': [{'bindings': [
            {'action': 'Quit', 'path': '<PXR_Controller>/secondaryButton'},
            {'action': 'SendData', 'path': '<PXR_Controller>{RightHand}/primaryButton'},
            {'action': 'Confirm', 'path': '<PXR_Controller>{RightHand}/triggerPressed'}]}]}))
        self.git = patch.object(prepare.subprocess, 'check_output', return_value=prepare.BASE_COMMIT+'\n')
        self.git.start()
        self.addCleanup(self.git.stop)

    def test_preparation_can_repeat_and_only_reserves_face_button_shortcuts(self):
        first = prepare.prepare(self.project, '10.0.0.2')
        second = prepare.prepare(self.project, '10.0.0.2')
        self.assertEqual(first, second)
        ui = (self.project/'Assets/Scripts/UI/UIOperate.cs').read_text()
        self.assertIn('// unrelated user comment', ui)
        self.assertEqual(ui.count('!VitaCameraView.ReserveFaceButtons'), 1)
        toggle = (self.project/'Assets/Scripts/Li/toggle-objects-script.cs').read_text()
        self.assertEqual(toggle.count('if (VitaCameraView.ReserveFaceButtons) return;'), 1)
        profile = json.loads((self.project/'Assets/Resources/vita_camera.json').read_text())
        self.assertEqual(profile['robot_host'], '10.0.0.2')
        self.assertEqual(first['package'], 'com.xrobotoolkit.client.vitacamera')
        head = (self.project/'Assets/InteractionTools/Scripts/HeadDevice.cs').read_text()
        self.assertEqual(head.count('!VitaCameraView.ReserveFaceButtons'), 2)
        actions = json.loads((self.project/'Assets/InteractionTools/PICO Input Actions.inputactions').read_text())
        self.assertEqual(actions['maps'][0]['bindings'],
            [{'action': 'Confirm', 'path': '<PXR_Controller>{RightHand}/triggerPressed'}])

    def test_user_changes_to_generated_overlay_are_preserved(self):
        prepare.prepare(self.project)
        target = self.project/'Assets/VitaCamera/FourButtonChord.cs'
        original = target.read_text()+'\n// user customization\n'
        target.write_text(original)
        with self.assertRaisesRegex(RuntimeError, 'preserve/review local edits'):
            prepare.prepare(self.project)
        self.assertEqual(target.read_text(), original)

    def test_wrong_upstream_commit_fails_before_creating_overlay(self):
        with patch.object(prepare.subprocess, 'check_output', return_value='wrong\n'):
            with self.assertRaisesRegex(ValueError, 'Expected XRoboToolkit'):
                prepare.prepare(self.project)
        self.assertFalse((self.project/'Assets/VitaCamera').exists())

    def test_generated_private_profile_can_be_rebuilt_as_portable(self):
        prepare.prepare(self.project, '10.0.0.2')
        receipt = prepare.prepare(self.project)
        self.assertEqual(receipt['profile']['robot_host'], '')
        self.assertEqual(json.loads((self.project/'Assets/Resources/vita_camera.json').read_text())['robot_host'], '')

    def test_manually_modified_device_profile_is_preserved(self):
        prepare.prepare(self.project, '10.0.0.2')
        profile = self.project/'Assets/Resources/vita_camera.json'
        custom = json.loads(profile.read_text()); custom['robot_host'] = '10.0.0.9'
        profile.write_text(json.dumps(custom))
        with self.assertRaisesRegex(RuntimeError, 'local edits'):
            prepare.prepare(self.project)
        self.assertEqual(json.loads(profile.read_text()), custom)

    def test_invalid_robot_address_fails_before_creating_overlay(self):
        for address in ('127.0.0.1', '0.0.0.0', '224.0.0.1', 'not-an-ip'):
            with self.subTest(address=address), self.assertRaises(ValueError):
                prepare.prepare(self.project, address)
        self.assertFalse((self.project/'Assets/VitaCamera').exists())


class ReleaseTests(unittest.TestCase):
    def test_shipped_apk_matches_its_manifest_portable_profile_and_overlay_sources(self):
        release = ROOT/'deploy/pico_viewer/releases'
        manifest = json.loads((release/'manifest.json').read_text())
        apk = release/manifest['file']
        self.assertEqual(apk.stat().st_size, manifest['bytes'])
        self.assertEqual(hashlib.sha256(apk.read_bytes()).hexdigest(), manifest['sha256'])
        self.assertEqual((release/'SHA256SUMS').read_text(), manifest['sha256']+'  '+apk.name+'\n')
        self.assertEqual(manifest['upstream_commit'], prepare.BASE_COMMIT)
        self.assertEqual(manifest['default_profile']['robot_host'], '')
        self.assertEqual(len(manifest['certificate_sha256']), 64)
        for item in manifest['overlay_files']:
            source = ROOT/'deploy/pico_viewer'/item['path']
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), item['sha256'], item['path'])
        with zipfile.ZipFile(apk) as archive:
            names = archive.namelist()
            self.assertIn('lib/arm64-v8a/libil2cpp.so', names)
            dex = [name for name in names if name.endswith('.dex')]
            self.assertTrue(any(b'Lcom/vita/xr/camera/VitaCameraReceiver;' in archive.read(name) for name in dex))
            profile = json.dumps(manifest['default_profile'], indent=2).encode()
            self.assertTrue(any(profile in archive.read(name) for name in names if name.startswith('assets/')))


if __name__ == '__main__':
    unittest.main()
