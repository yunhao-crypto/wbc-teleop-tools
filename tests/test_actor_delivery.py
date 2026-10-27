"""Installer regressions: integrity, archive boundaries and existing-file safety."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('actor_bootstrap_tests', ROOT/'scripts/bootstrap_actor_s100.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def archive(root, entries):
    p = root/'runtime.tar.gz'
    with tarfile.open(p, 'w:gz') as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return p


class InstallerTests(unittest.TestCase):
    def test_environment_uses_isolated_python_and_offline_pip(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, PYTHONHOME='/wrong/python', PYTHONPATH='/unrelated/modules',
                            PIP_INDEX_URL='https://example.invalid/'):
                env = bootstrap.environment(root)
            self.assertNotIn('PYTHONHOME', env)
            self.assertNotIn('/unrelated/modules', env['PYTHONPATH'])
            self.assertEqual(env['PYTHONNOUSERSITE'], '1')
            self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)
            self.assertNotIn('PIP_INDEX_URL', env)

    def model_manifest(self, root):
        item = dict(path='artifacts/actor_g1/actor_g1.onnx', bytes=4,
                    sha256=hashlib.sha256(b'good').hexdigest())
        (root/'config').mkdir()
        (root/'config/actor_pico.json').write_text(json.dumps(dict(artifacts=dict(model=item))))
        return dict(external_resources=[item])

    def test_missing_external_model_explains_how_to_supply_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.model_manifest(root)
            with self.assertRaisesRegex(RuntimeError, 'Supply --model'):
                bootstrap.provision_model(root, manifest)
            self.assertFalse((root/'artifacts').exists())

    def test_wrong_external_model_is_rejected_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.model_manifest(root)
            source = root/'external.onnx'; source.write_bytes(b'evil')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 does not match'):
                bootstrap.provision_model(root, manifest, source)
            self.assertFalse((root/'artifacts').exists())

    def test_verified_model_copy_and_existing_model_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.model_manifest(root)
            source = root/'external.onnx'; source.write_bytes(b'good')
            bootstrap.provision_model(root, manifest, source)
            destination = root/manifest['external_resources'][0]['path']
            self.assertEqual(destination.read_bytes(), b'good')
            bootstrap.provision_model(root, manifest)
            destination.write_bytes(b'old!')
            with self.assertRaisesRegex(RuntimeError, 'Preserving different existing model'):
                bootstrap.provision_model(root, manifest, source)
            self.assertEqual(destination.read_bytes(), b'old!')

    def test_changed_input_rejected_before_unpack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delivery = root/'deploy/actor_s100'
            delivery.mkdir(parents=True)
            (root/'required').write_bytes(b'changed')
            data = dict(schema='vita_actor_s100_delivery_v1', archives=[], required_inputs=['required'],
                        files=[dict(path='required', bytes=4, sha256=hashlib.sha256(b'good').hexdigest())])
            (delivery/'manifest.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'changed delivery input'):
                bootstrap.verify(root)
            self.assertFalse((root/'vendor_packages').exists())

    def test_parent_path_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'escape').symlink_to(root.parent, target_is_directory=True)
            for name in ['../outside', '/tmp/outside', 'escape/outside']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    bootstrap.inside(root, name)

    def test_archive_symlink_is_rejected_during_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delivery = root/'deploy/actor_s100'; delivery.mkdir(parents=True)
            p = root/'bad.tar.gz'
            with tarfile.open(p, 'w:gz') as tar:
                link = tarfile.TarInfo('linked'); link.type = tarfile.SYMTYPE; link.linkname = '/etc/passwd'
                tar.addfile(link)
            data = dict(schema='vita_actor_s100_delivery_v1', required_inputs=['bad.tar.gz'],
                        files=[dict(path='bad.tar.gz', bytes=p.stat().st_size, sha256=bootstrap.sha(p))],
                        archives=[dict(path='bad.tar.gz', required_members=[])])
            (delivery/'manifest.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'link, special file'):
                bootstrap.verify(root)

    def test_different_library_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive(root, [('library.so', b'new-library')])
            target = root/'runtime'; target.mkdir(); (target/'library.so').write_bytes(b'existing-library')
            manifest = dict(archives=[dict(role='pico_service', path='runtime.tar.gz', destination='runtime')])
            with self.assertRaisesRegex(RuntimeError, 'Preserving different'):
                bootstrap.unpack(root, manifest)
            self.assertEqual((target/'library.so').read_bytes(), b'existing-library')

    def test_device_settings_and_database_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive(root, [('setting.ini', b'factory-settings'), ('epDBBackup.s3db', b'factory-db')])
            target = root/'runtime'; target.mkdir()
            for name in ['setting.ini', 'epDBBackup.s3db']:
                (target/name).write_bytes(b'existing-device-data')
            bootstrap.unpack(root, dict(archives=[dict(role='pico_service', path='runtime.tar.gz', destination='runtime')]))
            for name in ['setting.ini', 'epDBBackup.s3db']:
                self.assertEqual((target/name).read_bytes(), b'existing-device-data')

    def test_unrelated_venv_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'venv').mkdir(); sentinel = root/'venv/user-data'; sentinel.write_text('preserve')
            with (patch.object(bootstrap.platform, 'machine', return_value='aarch64'),
                  patch.object(bootstrap.platform, 'libc_ver', return_value=('glibc', '2.35')),
                  patch.object(bootstrap.sys, 'platform', 'linux'),
                  patch.object(bootstrap.sys, 'version_info', (3, 10, 12))):
                with self.assertRaisesRegex(RuntimeError, 'Existing venv is preserved'):
                    bootstrap.install(root, {})
            self.assertEqual(sentinel.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
