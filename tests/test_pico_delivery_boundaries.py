"""Real manifest/entry boundaries with opaque fixture bytes; no model execution."""
import hashlib
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
import bootstrap_actor_s100 as actor
import bootstrap_pico_inputs_s100 as pico


class DeliveryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, data):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())

    def seal(self, release=None):
        names = sorted(p for p in self.root.rglob('*') if p.is_file()
                       and p != self.root/actor.DELIVERY/'manifest.json')
        files = [dict(path=p.relative_to(self.root).as_posix(), bytes=p.stat().st_size,
                      sha256=actor.sha(p), mode=oct(p.stat().st_mode & 0o777)) for p in names]
        manifest = dict(schema=actor.SCHEMA, variant='base', target=actor.TARGET,
            source=dict(commit='a'*40, dirty=False), requirements=(actor.DELIVERY/'requirements.txt').as_posix(),
            files=files, payload_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            archives=[], external_resources=[], pico_inputs_release=release or {})
        self.put(actor.DELIVERY/'manifest.json', manifest)
        return manifest

    def isolated(self):
        self.put(actor.DELIVERY/'requirements.txt', (ROOT/'deploy/pico_27000/requirements.txt').read_bytes())
        self.put('config/sdk021-wheel-guard.json', (ROOT/'config/pico_27000_sdk_guard.json').read_bytes())
        entries = ('scripts/pico_actor_aligned.py', 'scripts/check_pico_aligned.py',
                   'src/deployment/vita_s100_aorta_30dof/pico_aligned_runtime.py')
        for name in entries:
            self.put(name, (ROOT/name).read_bytes())
        self.put('fixtures/opaque.dat', b'opaque integrity fixture, not a model')
        item = dict(path='fixtures/opaque.dat', sha256=actor.sha(self.root/'fixtures/opaque.dat'))
        for name in ('actor_smpl', 'actor_threepoint'):
            self.put('config/'+name+'.json', dict(artifacts=dict(model=item)))
        bindings = {name: actor.sha(self.root/name) for name in (*entries, 'config/sdk021-wheel-guard.json')}
        for route in actor.PICO_POLICIES:
            self.put('config/'+route+'_live.json', dict(bindings=bindings))
        return dict(isolated_sdk=True, unified_environment=False, allowed_policies=list(actor.PICO_POLICIES),
            schema='vita_pico_inputs_s100_physical_v1', physical_workflow=True, motion_enabled=True,
            body_cmd_protocol_version=4, primary_route='pico_smpl', backend='ort',
            offline_check=dict(input=item, triggers=item))

    def test_real_isolated_verify_without_legacy_profiles_retains_integrity_and_binding_gates(self):
        release = self.isolated()
        self.seal(release)
        self.assertFalse(any((self.root/name).exists() for name in actor.POLICY_CONFIGS.values()))
        manifest = pico.verify_inputs(self.root)  # Calls real actor.verify(), with no mocked gate.
        self.assertTrue(actor.is_pico_package(manifest))
        guard_path = self.root/'config/sdk021-wheel-guard.json'
        original = guard_path.read_bytes()
        guard_path.write_bytes(original+b' ')
        with self.assertRaisesRegex(RuntimeError, 'Missing or changed delivery input'):
            actor.verify(self.root)
        self.seal(release)
        with self.assertRaisesRegex(ValueError, 'Unsealed Pico runtime binding'):
            pico.verify_inputs(self.root)
        guard_path.write_bytes(original)
        release['body_cmd_protocol_version'] = 3
        self.seal(release)
        with self.assertRaisesRegex(ValueError, 'Not a sealed'):
            pico.verify_inputs(self.root)

    def test_real_normal_verify_still_requires_matching_actor_pins_and_all_profiles(self):
        self.put(actor.DELIVERY/'requirements.txt', (ROOT/actor.DELIVERY/'requirements.txt').read_bytes())
        self.put('config/sdk021-wheel-guard.json', (ROOT/'config/sdk021-wheel-guard.json').read_bytes())
        for name in actor.POLICY_CONFIGS.values():
            self.put(name, (ROOT/name).read_bytes())
        self.seal()
        actor.verify(self.root)
        path = self.root/actor.POLICY_CONFIGS['sonic']
        original = path.read_bytes()
        config = json.loads(original); config['sdk_version'] = 'wrong'
        self.put(path.relative_to(self.root), config)
        self.seal()
        with self.assertRaisesRegex(RuntimeError, 'differs from the sealed ARM64 wheel guard'):
            actor.verify(self.root)
        path.unlink()
        self.seal()
        with self.assertRaises(FileNotFoundError):
            actor.verify(self.root)

    def entry(self, policy, blocked):
        entry = self.root/'scripts'
        entry.mkdir(exist_ok=True)
        for name in ('actor_pico.py', 'pico_actor_aligned.py'):
            shutil.copy2(ROOT/'scripts'/name, entry/name)
        if not (self.root/'src').exists():
            (self.root/'src').symlink_to(ROOT/'src', target_is_directory=True)
        self.put('config/actor_smpl.json', {})
        blocker = self.root/'blocker'
        blocker.mkdir(exist_ok=True)
        (blocker/'sitecustomize.py').write_text(
            'import importlib.abc, os, sys\n'
            'class Block(importlib.abc.MetaPathFinder):\n'
            ' def find_spec(self, fullname, path=None, target=None):\n'
            '  prefix=os.environ["BLOCK_SDK_PREFIX"]\n'
            '  if fullname==prefix or fullname.startswith(prefix+"."):\n'
            '   raise ModuleNotFoundError("SDK_BLOCKED:"+fullname, name=fullname)\n'
            'sys.meta_path.insert(0, Block())\n')
        env = dict(os.environ, PYTHONPATH=str(blocker), BLOCK_SDK_PREFIX=blocked,
                   PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, str(entry/'actor_pico.py'), 'check',
            '--policy', policy, '--help'], env=env, text=True, capture_output=True)

    def test_real_pico_entry_dispatch_does_not_require_sdk_or_trace(self):
        for blocked in ('locomotion_aorta.trace', 'locomotion_aorta'):
            for policy in actor.PICO_POLICIES:
                with self.subTest(blocked=blocked, policy=policy):
                    result = self.entry(policy, blocked)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('Selected-checkpoint Pico modalities', result.stdout)

    def test_real_isolated_legacy_rejection_precedes_sdk_imports(self):
        for blocked in ('locomotion_aorta.trace', 'locomotion_aorta'):
            with self.subTest(blocked=blocked):
                result = self.entry('sonic', blocked)
                self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
                self.assertIn('This isolated Pico delivery only supports', result.stderr)
                self.assertNotIn('SDK_BLOCKED', result.stderr)
