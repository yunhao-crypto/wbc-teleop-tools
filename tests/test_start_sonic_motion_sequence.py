"""Exercise the launcher against a fake actor entry; never open robot transport."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).parents[1]/'scripts/start_sonic_motion_sequence.sh'

class SequenceLauncherTests(unittest.TestCase):
    def fixture(self,d):
        (d/'scripts').mkdir();(d/'config').mkdir();f=d/'reference/sequences/sonic_max10';f.mkdir(parents=True)
        shutil.copy2(SCRIPT,d/'scripts'/SCRIPT.name)
        (d/'scripts/start_actor_motion.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        (d/'config/actor_pico.json').write_text('{}');(f/'sequence.npz').write_bytes(b'fixture')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        m=dict(policy_config_sha256=sha(d/'config/actor_pico.json'),sha256=sha(f/'sequence.npz'),duration_s=126.92)
        (f/'sequence_manifest.json').write_text(json.dumps(m))
        (f/'continuous_sim_validation.json').write_text(json.dumps(dict(completed=True,adapted_sha256=m['sha256'])))
        (d/'config/actor_motion_sequence_sonic.json').write_text(json.dumps(dict(sha256=m['sha256'],path='reference/sequences/sonic_max10/sequence.npz')))
        return f
    def run_launcher(self,d,*args):
        return subprocess.run(['bash',str(d/'scripts'/SCRIPT.name),*args],env=dict(os.environ,VITA_ACTOR_PYTHON=sys.executable),text=True,capture_output=True)
    def test_check_and_live_dispatch(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);self.fixture(d)
            r=self.run_launcher(d,'check');self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('sonic\ncheck\n',r.stdout);self.assertNotIn('--record-observations',r.stdout)
            r=self.run_launcher(d,'live');self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('sonic\nlive\n',r.stdout);self.assertIn('--record-observations',r.stdout)
            self.assertIn('--seconds\n0',r.stdout)
    def test_reject_configuration_drift_and_unvalidated_reference(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);f=self.fixture(d)
            (d/'config/actor_pico.json').write_text('{"changed":true}')
            r=self.run_launcher(d,'check');self.assertNotEqual(r.returncode,0);self.assertNotIn('sonic\ncheck',r.stdout)
            (d/'config/actor_pico.json').write_text('{}')
            v=json.loads((f/'continuous_sim_validation.json').read_text());v['completed']=False;(f/'continuous_sim_validation.json').write_text(json.dumps(v))
            r=self.run_launcher(d,'live');self.assertNotEqual(r.returncode,0);self.assertNotIn('sonic\nlive',r.stdout)
    def test_reject_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);self.fixture(d)
            self.assertNotEqual(self.run_launcher(d,'live','--policy','supermimic').returncode,0)

if __name__=='__main__':unittest.main()
