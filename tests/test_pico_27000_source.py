"""Portable source boundary checks; no weights, vendor SDK or hardware required."""
import ast,hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
class SourceTests(unittest.TestCase):
    def test_shared_lifecycle_function_matches_git_runner(self):
        module=ROOT/'src/deployment/vita_s100_aorta_30dof'
        original=(module/'actor_pico_live.py').read_text(encoding='utf-8')
        aligned=(module/'pico_aligned_lifecycle.py').read_text(encoding='utf-8')
        self.assertEqual(original[original.index('def run_live('):],aligned[aligned.index('def run_live('):])
    def test_legacy_policies_and_default_sdk_are_retained(self):
        shell=(ROOT/'scripts/start_actor_pico.sh').read_text()
        for policy in ('sonic','supermimic','morphmimic','pico_smpl','pico_teleop'):self.assertIn(policy,shell)
        self.assertIn('ACTOR_ENTRY="$ROOT/scripts/actor_pico.py"',shell)
        self.assertIn('ACTOR_ENTRY="$ROOT/scripts/pico_actor_aligned.py"',shell)
        self.assertIn('locomotion-aorta==0.3.0',(ROOT/'deploy/actor_s100/requirements.txt').read_text())
        self.assertIn('bodycmdv4.6863ad7',(ROOT/'deploy/pico_27000/requirements.txt').read_text())
    def test_delivery_paths_unique_and_no_private_asset_bytes_published(self):
        data=json.loads((ROOT/'deploy/pico_27000/delivery_inputs.json').read_text())
        paths=[x['path'] for x in data['source_files']+data['external_files']]
        self.assertEqual(len(paths),len(set(paths)))
        for item in data['source_files']:self.assertTrue((ROOT/item['source']).is_file(),item)
        external={x['path'] for x in data['external_files']}
        self.assertTrue(any(x.endswith('.onnx') for x in external))
        self.assertTrue(any('bodycmdv4' in x for x in external))
        self.assertFalse(any('host_validation' in x for x in external))
    def test_python_sources_support_target_python310_syntax(self):
        spec=json.loads((ROOT/'deploy/pico_27000/delivery_inputs.json').read_text())
        for item in spec['source_files']:
            if item['source'].endswith('.py'):
                ast.parse((ROOT/item['source']).read_text(encoding='utf-8-sig'),feature_version=(3,10))
    def test_model_and_command_contracts_explicit(self):
        for name,dimension in (('sonic_smpl_27000_contract.json',1740),('sonic_threepoint_27000_contract.json',1167)):
            data=json.loads((ROOT/'config'/name).read_text());self.assertEqual(data['input_dim'],dimension)
            self.assertEqual(data['action_dim'],28)
        for route in ('pico_smpl','pico_teleop'):
            settings=json.loads((ROOT/f'config/{route}_live.json').read_text())
            self.assertEqual(settings['body_cmd_protocol_version'],4)
            self.assertEqual(settings['stand_preparation_seconds'],3.)
    def test_host_live_rejected_before_transport(self):
        if sys.platform=='linux':self.skipTest('host rejection test is for a non-Linux build host')
        for route in ('pico_smpl','pico_teleop'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts/pico_actor_aligned.py'),'auto','--policy',route,
                '--robot-id','NO_ROBOT','--robot-config',str(ROOT/'unused.yaml')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Install only on Linux aarch64',result.stderr)
    def test_asset_builder_rejects_missing_assets(self):
        import tempfile
        from build_pico_27000_delivery import stage_delivery
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'separately supplied asset'):
                stage_delivery(ROOT,Path(tmp)/'absent',Path(tmp)/'stage',allow_dirty=True)
if __name__=='__main__':unittest.main()
