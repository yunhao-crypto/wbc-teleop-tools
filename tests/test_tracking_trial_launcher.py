"""Use a fake actor entry to verify trial isolation and fail-closed validation."""
from pathlib import Path
import importlib.util,hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('verify_trial',ROOT/'scripts/verify_tracking_trial.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class TrialLauncherTests(unittest.TestCase):
 def fixture(self,d,label='A31000'):
  (d/'scripts').mkdir();(d/'venv/bin').mkdir(parents=True);(d/'venv/bin/python').symlink_to(sys.executable);(d/'outputs').mkdir()
  for n in ['run_tracking_trial.sh','verify_tracking_trial.py','test_supermimic_A31000.sh','test_sonic_16000.sh','test_morphmimic_72000.sh']:shutil.copy2(ROOT/'scripts'/n,d/'scripts'/n)
  (d/'scripts/start_actor_motion.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
  trial=d/'trials/tracking_20261106'/label;(trial/'config').mkdir(parents=True);(trial/'outputs').mkdir();(trial/'model.onnx').write_bytes(b'fixture-model');(trial/'motion.npz').write_bytes(b'fixture-motion');(d/'robot.yaml').write_text('fixture')
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=dict(label=label,target_robot='10.100.23.42',files={},config_name='actor_pico_supermimic.json',model_sha256=sha(trial/'model.onnx'),reference_sha256=sha(trial/'motion.npz'))
  (trial/'trial_manifest.json').write_text(json.dumps(m));(trial/'config/actor_pico_supermimic.json').write_text(json.dumps(dict(artifacts=dict(model=dict(path='model.onnx',sha256=m['model_sha256'])))));(trial/'config/actor_motion_test.json').write_text(json.dumps(dict(path='motion.npz',sha256=m['reference_sha256'])))
  (trial/'continuous_sim_validation.json').write_text(json.dumps(dict(model_sha256=m['model_sha256'],adapted_sha256=m['reference_sha256'],completed=True,diagnostic_only=False)))
  (trial/'outputs/actor_pico_local.json').write_text(json.dumps(dict(robot_id='10.100.23.42',robot_config=str(d/'robot.yaml'),robot_config_sha256=sha(d/'robot.yaml'))))
  return trial
 def test_live_uses_isolated_root_and_records_observations(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);trial=self.fixture(d);r=subprocess.run(['bash',str(d/'scripts/test_supermimic_A31000.sh')],text=True,capture_output=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('supermimic\nlive\n--root\n'+str(trial),r.stdout);self.assertIn('--record-observations',r.stdout);self.assertIn('tracking_20261106_A31000_',r.stdout)
 def test_reject_changed_model_and_nonpassing_sim(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);trial=self.fixture(d);p=trial/'continuous_sim_validation.json';m=json.loads(p.read_text());m['completed']=False;p.write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'simulation did not pass'):mod.verify(d,'A31000')
   mod.verify(d,'A31000',require_sim=False)
   (trial/'model.onnx').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'model/config'):mod.verify(d,'A31000',require_sim=False)
 def test_sonic16000_routes_to_sonic_with_independent_root(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);trial=self.fixture(d,label='Sonic16000');r=subprocess.run(['bash',str(d/'scripts/test_sonic_16000.sh')],text=True,capture_output=True)
   self.assertEqual(r.returncode,0,r.stderr);self.assertIn('sonic\nlive\n--root\n'+str(trial),r.stdout);self.assertIn('--record-observations',r.stdout)
 def test_morphmimic72000_routes_to_its_own_policy_and_root(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);trial=self.fixture(d,label='MorphMimic72000');r=subprocess.run(['bash',str(d/'scripts/test_morphmimic_72000.sh')],text=True,capture_output=True)
   self.assertEqual(r.returncode,0,r.stderr);self.assertIn('morphmimic\nlive\n--root\n'+str(trial),r.stdout);self.assertIn('--record-observations',r.stdout)
 def test_reject_wrong_robot_and_argument_overrides(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);trial=self.fixture(d);p=trial/'outputs/actor_pico_local.json';m=json.loads(p.read_text());m['robot_id']='10.100.23.173';p.write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'profile changed'):mod.verify(d,'A31000')
   r=subprocess.run(['bash',str(d/'scripts/test_supermimic_A31000.sh'),'live','--root','/wrong'],capture_output=True);self.assertNotEqual(r.returncode,0)
if __name__=='__main__':unittest.main()
