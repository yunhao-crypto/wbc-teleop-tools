"""Exercise real offline ARM wheel resolution and extracted package boundaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_actor_delivery import build, extract_bundle, resolve_wheels
from bootstrap_actor_s100 import DELIVERY, provision_planner_assets, verify


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='actor-package-tests-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.built, cls.packages = {}, {}
        # The verifier's isolated source export has no .git directory.
        source = json.loads(os.environ['ACTOR_DELIVERY_TEST_SOURCE']) if 'ACTOR_DELIVERY_TEST_SOURCE' in os.environ else None
        for variant in ('base', 'planner'):
            cls.built[variant] = build(ROOT, cls.work/'archives', variant, allow_dirty=True, source=source)
            cls.packages[variant] = extract_bundle(Path(cls.built[variant]['archive']), cls.work/variant)

    def test_variants_resolve_independent_complete_wheel_closures(self):
        for variant, expected in (('base', 21), ('planner', 28)):
            with self.subTest(variant=variant):
                package = self.packages[variant]
                manifest = verify(package)
                wheels = list((package/DELIVERY/'wheelhouse').glob('*.whl'))
                self.assertEqual(len(wheels), expected)
                self.assertEqual(any(p.name.startswith('onnxruntime-') for p in wheels), variant == 'planner')
                self.assertEqual((package/'config/actor_sonic_planner.json').exists(), variant == 'planner')
                self.assertEqual(manifest['source'], self.built[variant]['source'])
                self.assertEqual(manifest['target']['architecture'], 'aarch64')
                self.assertEqual({i['role'] for i in manifest['external_resources']},
                    {'sonic_actor', 'supermimic_actor', 'morphmimic_actor'} |
                    ({'planner', 'bridge'} if variant == 'planner' else set()))

    def test_archive_contains_runtime_only_and_one_manifest(self):
        for package in self.packages.values():
            files = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
            self.assertEqual([p for p in files if p.endswith('/manifest.json')],
                             ['deploy/actor_s100/manifest.json'])
            self.assertFalse(any(p.startswith(('tests/', 'outputs/', 'venv/', 'reference/')) for p in files))
            self.assertFalse(any(p.endswith(('.md', '.onnx', '.pyc')) for p in files))
            self.assertIn('scripts/start_actor_pico.sh', files)
            self.assertIn('scripts/configure_actor_s100.py', files)

    def test_documentation_and_tests_do_not_change_payload_or_archive(self):
        package = self.packages['base']
        (package/'README.md').write_text('Developer documentation changed.\n')
        (package/'tests').mkdir(exist_ok=True)
        (package/'tests/test_unused.py').write_text('raise RuntimeError("must not ship")\n')
        try:
            repeated = build(package, self.work/'repeat', 'base', allow_dirty=True,
                             source=self.built['base']['source'])
            self.assertEqual(repeated['payload_sha256'], self.built['base']['payload_sha256'])
            self.assertEqual(repeated['sha256'], self.built['base']['sha256'])
        finally:
            (package/'README.md').unlink()
            (package/'tests/test_unused.py').unlink()
            (package/'tests').rmdir()

    def test_modified_and_missing_runtime_files_are_rejected(self):
        package = self.packages['base']
        path = package/'scripts/start_actor_pico.sh'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n# changed\n')
            with self.assertRaisesRegex(RuntimeError, 'changed delivery input'):
                verify(package)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'changed delivery input'):
                verify(package)
        finally:
            path.write_bytes(original)
            path.chmod(0o755)

    def test_base_rejects_planner_assets_before_copying(self):
        package = self.packages['base']
        with self.assertRaisesRegex(RuntimeError, 'planner delivery package'):
            provision_planner_assets(package, verify(package), planner=Path('/not-a-model.onnx'))
        self.assertFalse((package/'artifacts/sonic').exists())

    def test_wrong_target_and_unsealed_requirements_are_rejected(self):
        package = self.packages['base']
        path = package/DELIVERY/'manifest.json'
        original = path.read_text()
        try:
            manifest = json.loads(original)
            manifest['target']['architecture'] = 'x86_64'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Delivery target'):
                verify(package)
            manifest = json.loads(original)
            manifest['requirements'] = 'unsealed.txt'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, 'unsealed package requirements'):
                verify(package)
        finally:
            path.write_text(original)

    def test_missing_wheel_fails_offline_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            delivery = root/DELIVERY
            wheels = delivery/'wheelhouse'
            wheels.mkdir(parents=True)
            for name in ('requirements.txt', 'requirements-planner.txt'):
                shutil.copyfile(ROOT/DELIVERY/name, delivery/name)
            for wheel in (ROOT/DELIVERY/'wheelhouse').glob('*.whl'):
                if not wheel.name.startswith('onnxruntime-'):
                    shutil.copyfile(wheel, wheels/wheel.name)
            with self.assertRaisesRegex(RuntimeError, 'Offline ARM64 wheel resolution failed'):
                resolve_wheels(root, DELIVERY/'requirements-planner.txt', root/'resolved')

    def test_packages_verify_and_unpack_without_git_sdk_or_models(self):
        for variant, package in self.packages.items():
            with self.subTest(variant=variant):
                # Use a separate extraction so the reproducibility fixture stays pristine.
                fresh = extract_bundle(Path(self.built[variant]['archive']), self.work/('unpack-'+variant))
                for mode in ('--verify-only', '--unpack-only'):
                    result = subprocess.run([sys.executable, 'scripts/bootstrap_actor_s100.py', mode],
                        cwd=fresh, check=True, text=True, stdout=subprocess.PIPE)
                    self.assertEqual(json.loads(result.stdout)['actions_sent'], 0)
                self.assertTrue((fresh/'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so').is_file())
                self.assertFalse((fresh/'venv').exists())


if __name__ == '__main__':
    unittest.main()
