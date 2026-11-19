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
            root=Path(temp);(root/'build').mkdir();(root/'venv').mkdir()
            (root/'venv/.actor_bootstrap_owned').write_text('owned')
            release={}
            if policy in actor.PICO_POLICIES:
                (root/'venv/.pico_bodycmd4_owned').write_text('isolated')
                release=dict(isolated_sdk=True,unified_environment=False,allowed_policies=list(actor.PICO_POLICIES))
            manifest=dict(pico_inputs_release=release,source={'commit':'a'*40,'dirty':False},
                variant='base',payload_sha256='sealed',requirements='requirements.txt')
            proof=dict(actions_sent=0,policy=policy,environment='isolated_pico_venv')
            with patch.object(actor.sys,'platform','linux'),patch.object(actor.sys,'version_info',(3,10)), \
                 patch.object(actor.platform,'machine',return_value='aarch64'), \
                 patch.object(actor.platform,'libc_ver',return_value=('glibc','2.35')), \
                 patch.object(pico,'check_installation',return_value=proof) as checked, \
                 patch.object(actor,'unpack') as unpack,patch.object(actor,'sha',return_value='digest'), \
                 patch.object(actor,'configured_model',return_value={'path':'model.onnx'}), \
                 patch.object(actor.subprocess,'run',return_value=__import__('types').SimpleNamespace(returncode=0,stdout='')) as run:
                result=actor.install(root,manifest,policy=policy)
                unpack.assert_called_once_with(root,manifest,use_prebuilt=True)
                self.assertEqual(json.loads((root/'build/actor_s100_install.json').read_text())['policy'],policy)
                if policy in actor.PICO_POLICIES:
                    checked.assert_called_once();self.assertEqual(result['environment'],'isolated_pico_venv')
                    self.assertFalse(any('actor_pico.py' in str(call.args[0]) for call in run.call_args_list))
                else:
                    checked.assert_not_called()
                    self.assertEqual(run.call_args.args[0][-2:],['--policy',policy])
                    self.assertEqual(result['status'],'PASS_S100_OFFLINE_INSTALL_AND_MODEL_CHECK')
                self.assertEqual(sum('pip' in call.args[0] for call in run.call_args_list),2)
    def test_both_modalities_extend_original_install(self):
        for policy in actor.PICO_POLICIES:self.install(policy)

    def test_fresh_venv_gets_r4_marker_only_for_pico(self):
        import venv
        for policy in ('pico_smpl','sonic'):
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'build').mkdir()
                manifest=dict(variant='base',requirements='requirements.txt',source={},payload_sha256='sealed')
                if policy in actor.PICO_POLICIES:
                    manifest['pico_inputs_release']=dict(isolated_sdk=True,unified_environment=False,
                        allowed_policies=list(actor.PICO_POLICIES))
                def create(path):path.mkdir()
                with patch.object(actor.sys,'platform','linux'),patch.object(actor.sys,'version_info',(3,10)), \
                        patch.object(actor.platform,'machine',return_value='aarch64'), \
                        patch.object(actor.platform,'libc_ver',return_value=('glibc','2.35')), \
                        patch.object(venv,'EnvBuilder') as builder,patch.object(actor,'unpack'), \
                        patch.object(actor,'sha',return_value='digest'), \
                        patch.object(actor,'configured_model',return_value={'path':'model.onnx'}), \
                        patch.object(pico,'check_installation',return_value={'policy':policy}), \
                        patch.object(actor.subprocess,'run',return_value=__import__('types').SimpleNamespace(returncode=0,stdout='')):
                    builder.return_value.create.side_effect=create
                    actor.install(root,manifest,policy=policy)
                self.assertTrue((root/'venv/.actor_bootstrap_owned').is_file())
                self.assertEqual((root/'venv/.pico_bodycmd4_owned').exists(),policy in actor.PICO_POLICIES)
    def test_legacy_policy_uses_only_its_original_install(self):
        for policy in actor.POLICY_CONFIGS:self.install(policy)
    def test_legacy_verify_does_not_require_pico_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest={'variant':'base','files':[]}
            with patch.object(actor,'verify',return_value=manifest),patch.object(pico,'verify_inputs') as checked:
                actor.main(['--root',temp,'--policy','sonic','--verify-only'])
                checked.assert_not_called()
    def test_receipt_invalidation_is_limited_to_opt_in_feature_bundle(self):
        for feature in (False,True):
            with self.subTest(feature=feature),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'build').mkdir();receipt=root/'build/actor_s100_install.json'
                receipt.write_text('previous receipt');manifest={'variant':'base'}
                if feature:manifest['pico_inputs_release']=dict(isolated_sdk=True,unified_environment=False,allowed_policies=list(actor.PICO_POLICIES))
                policy='pico_smpl' if feature else 'sonic'
                with patch.object(actor,'verify',return_value=manifest),patch.object(pico,'verify_inputs'),patch.object(actor,'provision_model'), \
                     patch.object(actor,'provision_planner_assets'),patch.object(actor,'install',side_effect=RuntimeError('injected failure')):
                    with self.assertRaisesRegex(RuntimeError,'injected failure'):
                        actor.main(['--root',temp,'--policy',policy])
                self.assertEqual(receipt.exists(),not feature)
    def test_unowned_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'venv').mkdir()
            with patch.object(actor.sys,'platform','linux'),patch.object(actor.sys,'version_info',(3,10)), \
                 patch.object(actor.platform,'machine',return_value='aarch64'), \
                 patch.object(actor.platform,'libc_ver',return_value=('glibc','2.35')),patch.object(actor.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'Existing legacy or unowned venv is preserved'):
                    actor.install(root,{'pico_inputs_release':dict(isolated_sdk=True,unified_environment=False,
                        allowed_policies=list(actor.PICO_POLICIES))},policy='pico_smpl')
                run.assert_not_called()
    def test_pico_no_action_install_checks_only_current_models_and_codec(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'build').mkdir();manifest=dict(payload_sha256='sealed',source={})
            def execute(command,**kwargs):
                if command[1].endswith('verify_bodycmd4_codec.py'):
                    path=root/'build/bodycmd4_codec_check.json';proof={'status':'PASS_NATIVE_BODYCMD4_SERIALIZATION'}
                else:
                    self.assertTrue(command[1].endswith('check_pico_aligned.py'))
                    path=Path(command[command.index('--out')+1]);proof=dict(status='PASS_ALIGNED_OFFLINE_CHECK',actions_sent=0,control_handles_created=0)
                path.write_text(json.dumps(proof))
            with patch.object(pico,'verify_inputs',return_value=manifest),patch.object(pico.subprocess,'run',side_effect=execute) as run:
                receipt=pico.check_installation(root,manifest,root/'venv/bin/python',{},'pico_smpl')
            self.assertEqual(run.call_count,5)
            self.assertEqual(set(receipt['aligned_model_checks']),set(actor.PICO_POLICIES))
            for value in receipt['aligned_model_checks'].values():self.assertEqual(set(value),{'sdk','ort'})
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
        self.assertEqual(self.manifest['external_resources'],[])
        release=self.manifest['pico_inputs_release']
        self.assertTrue(release['isolated_sdk'])
        self.assertFalse(release['unified_environment'])
        self.assertEqual(release['allowed_policies'],['pico_smpl','pico_teleop'])
        self.assertEqual(release['delivery_version'],'35000-pico-only-r4')
        self.assertFalse((self.root/'artifacts/actor_g1/model_step_035000_g1.onnx').exists())
        for name in actor.POLICY_CONFIGS.values():
            self.assertFalse((self.root/name).exists(),name+' must stay in the original SDK environment')
    def test_only_the_two_pico_routes_use_the_isolated_r4_sdk(self):
        wheels=list((self.root/'deploy/actor_s100/wheelhouse').glob('locomotion_aorta-*.whl'))
        self.assertEqual(len(wheels),1);self.assertIn('bodycmdv4',wheels[0].name)
        requirements=(self.root/'deploy/actor_s100/requirements.txt').read_text()
        self.assertIn('locomotion-aorta==1.0.0rc1+bodycmdv4',requirements)
        guard=json.loads((self.root/'config/sdk021-wheel-guard.json').read_text())
        for name in ('actor_smpl','actor_threepoint'):
            settings=json.loads((self.root/f'config/{name}.json').read_text())
            for path,digest in settings['sdk_sources'].items():self.assertEqual(digest,guard['source_sha256'][path])
    def test_pico_delivery_rejects_legacy_runtime_selection(self):
        import subprocess
        for policy in actor.POLICY_CONFIGS:
            result=subprocess.run([sys.executable,'-X','utf8',str(self.root/'scripts/actor_pico.py'),
                'check','--policy',policy],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertIn('isolated Pico delivery only supports',result.stderr)
    def test_both_new_models_run_from_the_sealed_stage(self):
        # Run in child interpreters to isolate the host-only SDK math namespace.
        # This catches missing transitive files in the actual delivery boundary.
        import importlib.util,subprocess,zipfile
        if importlib.util.find_spec('onnxruntime') is None:self.skipTest('Host ONNX Runtime required')
        from bootstrap_actor_s100 import inside
        site=self.root/'venv/lib/python3.10/site-packages'
        for wheel in (self.root/'deploy/actor_s100/wheelhouse').glob('*.whl'):
            with zipfile.ZipFile(wheel) as archive:
                for name in archive.namelist():
                    if name.endswith('.py') and name.startswith(('locomotion_aorta/','lowlevel/','aorta/services/')):
                        target=inside(site,name);target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(archive.read(name))
        for route in ('pico_smpl','pico_teleop'):
            output=self.root/'outputs'/(route+'_model_check.json')
            result=subprocess.run([sys.executable,'-X','utf8',str(self.root/'scripts/check_pico_aligned.py'),
                '--route',route,'--backend','ort','--out',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['checkpoint_iteration'],35000)
            self.assertEqual(report['full_onnx_parity_abs_max'],0.)
            self.assertEqual(report['actions_sent'],0)
            self.assertFalse(report['live_lifecycle_exercised'])
if __name__=='__main__':unittest.main()
