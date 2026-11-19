"""Shared bootstrap dispatch and ownership protections, without hardware."""
import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import bootstrap_actor_s100 as actor
import bootstrap_pico_inputs_s100 as pico
class UnifiedTests(unittest.TestCase):
    def install(self,policy):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'build').mkdir()
            manifest={'pico_inputs_release':{'unified_environment':True}}
            with patch.object(actor.sys,'platform','linux'),patch.object(actor.sys,'version_info',(3,10)), \
                 patch.object(actor.platform,'machine',return_value='aarch64'), \
                 patch.object(actor.platform,'libc_ver',return_value=('glibc','2.35')), \
                 patch.object(pico,'install',return_value={'actions_sent':0}) as shared, \
                 patch.object(actor.subprocess,'run') as run:
                proof=actor.install(root,manifest,policy=policy)
                shared.assert_called_once_with(root,manifest)
                self.assertEqual(proof['environment'],'shared_actor_venv')
                self.assertEqual(json.loads((root/'build/actor_s100_install.json').read_text())['policy'],policy)
                return run.call_args_list
    def test_both_modalities_use_shared_bootstrap(self):
        for policy in ('pico_smpl','pico_teleop'):self.assertEqual(self.install(policy),[])
    def test_legacy_policies_use_same_environment_and_original_model_check(self):
        for policy in actor.POLICY_CONFIGS:
            calls=self.install(policy);self.assertEqual(len(calls),1)
            self.assertEqual(calls[0].args[0][-2:],['--policy',policy])
            self.assertTrue(calls[0].args[0][1].endswith('actor_pico.py'))
    def test_unowned_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'venv').mkdir()
            with patch.object(pico,'require_target'),patch.object(pico,'verify_inputs',return_value={'payload_sha256':'same'}), \
                 patch.object(pico.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'unowned venv'):pico.install(root,{'payload_sha256':'same'})
                run.assert_not_called()
    def test_single_public_python_and_shell_entry(self):
        shell=(ROOT/'scripts/start_actor_pico.sh').read_text()
        self.assertNotIn('ACTOR_ENTRY="$ROOT/scripts/pico_actor_aligned.py"',shell)
        self.assertIn('from pico_actor_aligned import main',(ROOT/'scripts/actor_pico.py').read_text())
    def test_target_sdk_unchanged_math_and_dual_protocol_declarations(self):
        legacy=json.loads((ROOT/'config/sdk021-wheel-guard.json').read_text())
        shared=json.loads((ROOT/'config/pico_27000_sdk_guard.json').read_text())
        for name in ('wbc_tracking_policy.py','_policy_state.py'):
            self.assertEqual(legacy['source_sha256'][name],shared['source_sha256'][name])
@unittest.skipUnless(os.environ.get('PICO_27000_ASSETS'),'Supply pinned assets for full staging checks')
class UnifiedPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from build_pico_27000_delivery import stage_delivery
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)/'stage'
        cls.manifest=stage_delivery(ROOT,Path(os.environ['PICO_27000_ASSETS']),cls.root,allow_dirty=True)
    def test_both_verifiers_accept_same_manifest(self):
        self.assertEqual(actor.verify(self.root)['payload_sha256'],pico.verify_inputs(self.root)['payload_sha256'])
        self.assertEqual(len(self.manifest['external_resources']),2)
        self.assertTrue((self.root/'artifacts/actor_g1/model_step_035000_g1.onnx').is_file())
    def test_one_sdk_and_matching_config_pins_for_all_routes(self):
        wheels=list((self.root/'deploy/actor_s100/wheelhouse').glob('locomotion_aorta-*.whl'))
        self.assertEqual(len(wheels),1);self.assertIn('bodycmdv4',wheels[0].name)
        requirements=(self.root/'deploy/actor_s100/requirements.txt').read_text()
        self.assertIn('locomotion-aorta==1.0.0rc1+bodycmdv4',requirements)
        guard=json.loads((self.root/'config/sdk021-wheel-guard.json').read_text())
        for name in ('actor_pico','actor_pico_supermimic','actor_pico_morphmimic','actor_smpl','actor_threepoint'):
            settings=json.loads((self.root/f'config/{name}.json').read_text())
            for path,digest in settings['sdk_sources'].items():self.assertEqual(digest,guard['source_sha256'][path])
if __name__=='__main__':unittest.main()
