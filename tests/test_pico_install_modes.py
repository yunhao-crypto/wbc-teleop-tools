"""Shared install ownership and public non-actuating output paths, no assets needed."""
import argparse,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import bootstrap_actor_s100 as actor
import bootstrap_pico_inputs_s100 as pico
import pico_aligned_modes as modes


class InstallerOwnershipTests(unittest.TestCase):
    def test_live_lock_blocks_both_install_and_unpack_entries(self):
        # Separate processes prove OS lock contention rather than a mock flock.
        # Linux takes the exact flock used by live; Windows uses its native
        # byte lock to exercise the architecture-independent unpack tooling.
        child='''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import bootstrap_actor_s100 as actor
import bootstrap_pico_inputs_s100 as pico
root=Path(sys.argv[2]);entry=sys.argv[3];mode=sys.argv[4]
manifest={'variant':'base','pico_inputs_release':{'isolated_sdk':True,'unified_environment':False,
    'allowed_policies':['pico_smpl','pico_teleop']}}
with patch.object(actor,'verify',return_value=manifest),patch.object(pico,'verify_inputs',return_value=manifest), \\
     patch.object(actor,'install') as install,patch.object(actor,'unpack') as unpack, \\
     patch.object(actor,'provision_model') as model,patch.object(actor,'provision_planner_assets') as planner:
    try:
        (actor if entry=='actor' else pico).main(['--root',str(root),
            *(['--policy','pico_smpl'] if entry=='actor' else []),*(['--unpack-only'] if mode=='unpack' else [])])
    except RuntimeError as exc:
        assert 'Another operator' in str(exc),str(exc)
    else:raise AssertionError('Concurrent mutation was admitted')
    assert not any(x.called for x in (install,unpack,model,planner))
'''
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'build').mkdir()
            receipt=root/'build/actor_s100_install.json';receipt.write_text('previous successful receipt')
            with actor.operator_lock(root):
                for entry in ('actor','pico'):
                    for mode in ('install','unpack'):
                        with self.subTest(entry=entry,mode=mode):
                            result=subprocess.run([sys.executable,'-c',child,str(ROOT/'scripts'),str(root),entry,mode],
                                capture_output=True,text=True)
                            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                            self.assertEqual(receipt.read_text(),'previous successful receipt')
            # Lock release must admit a subsequent transaction.
            with actor.operator_lock(root),pico.installation_transaction(root):self.assertFalse(receipt.exists())

    def test_failed_install_invalidates_old_and_partial_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'build').mkdir()
            paths=[root/'build'/name for name in ('actor_s100_install.json','pico_inputs_s100_install.json')]
            for path in paths:path.write_text('old')
            with self.assertRaisesRegex(RuntimeError,'installation failed'):
                with actor.operator_lock(root),pico.installation_transaction(root):
                    self.assertFalse(any(path.exists() for path in paths))
                    pico.write_install_receipt(root,{'status':'premature'})
                    raise RuntimeError('installation failed')
            self.assertFalse(any(path.exists() for path in paths))

    def test_compatibility_entry_delegates_to_original_transaction(self):
        with patch.object(actor,'main') as main:
            pico.main(['--root','some-root','--unpack-only'])
            main.assert_called_once_with(['--policy','pico_smpl','--root','some-root','--unpack-only'])

    def test_isolation_rejections_preserve_environment_and_receipts(self):
        isolated={'variant':'base','pico_inputs_release':dict(isolated_sdk=True,
            unified_environment=False,allowed_policies=list(actor.PICO_POLICIES))}
        cases=[(policy,isolated,markers) for policy in actor.PICO_POLICIES for markers in
            ((),('.actor_bootstrap_owned',),('.pico_bodycmd4_owned',))]
        cases += [(policy,isolated,('.actor_bootstrap_owned','.pico_bodycmd4_owned'))
            for policy in actor.POLICY_CONFIGS]
        cases += [(policy,{'variant':'base'},('.actor_bootstrap_owned','.pico_bodycmd4_owned'))
            for policy in actor.POLICY_CONFIGS]
        cases += [('pico_smpl',{'variant':'base'},('.actor_bootstrap_owned',))]
        for policy,manifest,markers in cases:
            for unpack in (False,True):
                with self.subTest(policy=policy,markers=markers,unpack=unpack),tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);(root/'venv').mkdir();(root/'build').mkdir()
                    saved={root/'build/actor_s100_install.json':b'old receipt',
                        root/'build/pico_inputs_s100_install.json':b'old pico receipt',
                        root/'venv/preserved.bin':b'original environment'}
                    saved.update({root/'venv'/name:b'original '+name.encode() for name in markers})
                    for path,content in saved.items():path.write_bytes(content)
                    with patch.object(actor,'verify',return_value=manifest),patch.object(pico,'verify_inputs'), \
                            patch.object(actor,'provision_model') as provision,patch.object(actor,'unpack') as unpacker, \
                            patch.object(actor.subprocess,'run') as run:
                        with self.assertRaises(RuntimeError):
                            actor.main(['--root',str(root),'--policy',policy,*(['--unpack-only'] if unpack else [])])
                        provision.assert_not_called();unpacker.assert_not_called();run.assert_not_called()
                    for path,content in saved.items():self.assertEqual(path.read_bytes(),content)

    def test_direct_install_also_rejects_cross_sdk_reuse_before_pip(self):
        isolated={'pico_inputs_release':dict(isolated_sdk=True,unified_environment=False,
            allowed_policies=list(actor.PICO_POLICIES))}
        for policy,manifest,marker in [('pico_smpl',isolated,'.actor_bootstrap_owned'),
                ('sonic',isolated,'.pico_bodycmd4_owned'),('sonic',{},'.pico_bodycmd4_owned')]:
            with self.subTest(policy=policy,manifest=manifest),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'venv').mkdir();(root/'venv'/marker).write_bytes(b'preserve marker')
                with patch.object(actor.subprocess,'run') as run:
                    with self.assertRaises(RuntimeError):actor.install(root,manifest,policy=policy)
                    run.assert_not_called()
                self.assertEqual((root/'venv'/marker).read_bytes(),b'preserve marker')


    def test_default_package_rejects_pico_without_importing_optional_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'venv').mkdir();(root/'build').mkdir()
            marker=root/'venv/.actor_bootstrap_owned';marker.write_bytes(b'legacy marker')
            receipt=root/'build/actor_s100_install.json';receipt.write_bytes(b'legacy receipt')
            with patch.object(actor,'verify',return_value={'variant':'base'}), \
                    patch.dict(sys.modules,{'bootstrap_pico_inputs_s100':None}), \
                    patch.object(actor.subprocess,'run') as run:
                for invoke in (lambda:actor.main(['--root',temp,'--policy','pico_smpl']),
                        lambda:actor.install(root,{'variant':'base'},policy='pico_teleop')):
                    with self.assertRaisesRegex(RuntimeError,'separate --pico-assets'):invoke()
                run.assert_not_called()
            self.assertEqual(marker.read_bytes(),b'legacy marker');self.assertEqual(receipt.read_bytes(),b'legacy receipt')


class PublicOutputTests(unittest.TestCase):
    def command(self,root,route,mode,out=None,*,sdk_dir=None,startup_timeout=30.):
        args=SimpleNamespace(root=root,policy=route,mode=mode,backend='ort',threads=1,seconds=None,
            pico_service='external',group='default',out=out,record_observations=False,
            sdk_dir=sdk_dir,startup_timeout=startup_timeout)
        fixture={key:{'path':key+'.json','sha256':'0'*64} for key in ('input','triggers')}
        manifest={'pico_inputs_release':{'offline_check':fixture}}
        path=root/'deploy/actor_s100/manifest.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(manifest))
        with patch.object(pico,'verify_inputs'),patch.object(modes.os,'execv') as execute:
            modes.run_mode(args,argparse.ArgumentParser())
            return execute.call_args.args[1]

    def test_shadow_replay_both_routes_get_unique_new_output_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);outputs=[]
            for route in ('pico_smpl','pico_teleop'):
                for mode in ('shadow','replay'):
                    for _ in range(2):
                        command=self.command(root,route,mode)
                        output=Path(command[command.index('--out')+1]);outputs.append(output)
                        self.assertEqual(output.parent,root/'outputs')
                        self.assertFalse(output.exists(),'pico_inputs requires a nonexistent output directory')
            self.assertEqual(len(set(outputs)),8)

    def test_explicit_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'explicit'
            command=self.command(root,'pico_smpl','replay',out)
            self.assertEqual(command[command.index('--out')+1],str(out))

    def test_shadow_forwards_explicit_sdk_and_calibration_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sdk=root/'custom-sdk'
            for route in ('pico_smpl','pico_teleop'):
                command=self.command(root,route,'shadow',sdk_dir=sdk,startup_timeout=90.)
                self.assertEqual(command[command.index('--sdk-dir')+1],str(sdk))
                self.assertEqual(command[command.index('--startup-timeout')+1],'90.0')

    def test_replay_does_not_receive_device_startup_options(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            command=self.command(root,'pico_smpl','replay',sdk_dir=root/'custom-sdk',startup_timeout=90.)
            self.assertNotIn('--sdk-dir',command)
            self.assertNotIn('--startup-timeout',command)


class LiveAdmissionOwnershipTests(unittest.TestCase):
    def test_live_rejects_busy_environment_before_receipt_or_sdk_reads(self):
        child='''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import bootstrap_pico_inputs_s100 as pico
import pico_actor_aligned as entry
root=Path(sys.argv[2])
argv=['pico_actor_aligned.py','auto','--policy','pico_teleop','--root',str(root),
      '--robot-id','IN_MEMORY_ONLY','--robot-config',str(root/'effective.yaml')]
with patch.object(sys,'argv',argv),patch.object(pico,'require_target'), \\
     patch.object(pico,'verify_inputs',side_effect=AssertionError('receipt/SDK bytes read before lock')) as verify:
    try:entry.main()
    except RuntimeError as exc:assert 'Another operator' in str(exc),str(exc)
    else:raise AssertionError('Live was admitted during installation')
    verify.assert_not_called()
'''
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with actor.operator_lock(root):
                result=subprocess.run([sys.executable,'-c',child,str(ROOT/'scripts'),str(root)],
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_loading_failure_releases_live_operator_lock(self):
        import pico_actor_aligned as entry
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            argv=['pico_actor_aligned.py','auto','--policy','pico_teleop','--root',str(root),
                '--robot-id','IN_MEMORY_ONLY','--robot-config',str(root/'effective.yaml')]
            with patch.object(sys,'argv',argv),patch.object(pico,'require_target'), \
                    patch.object(entry,'run_live_entry',side_effect=RuntimeError('model load failed')) as load:
                with self.assertRaisesRegex(RuntimeError,'model load failed'):entry.main()
                load.assert_called_once()
            with actor.operator_lock(root):pass


class PublicCheckThreadTests(unittest.TestCase):
    def test_public_check_forwards_threads_for_both_routes(self):
        import check_pico_aligned as checker
        for route in actor.PICO_POLICIES:
            args=SimpleNamespace(root=ROOT,policy=route,mode='check',backend='ort',steps=240,threads=4,out=None)
            with patch.object(pico,'verify_inputs'),patch.object(checker,'check',return_value={}) as check:
                modes.run_mode(args,argparse.ArgumentParser())
                check.assert_called_once_with(ROOT,route,'ort',240,threads=4)

    def test_public_check_rejects_invalid_threads_before_inference(self):
        import check_pico_aligned as checker
        from contextlib import redirect_stderr
        import io
        for threads in (0,17):
            args=SimpleNamespace(root=ROOT,policy='pico_smpl',mode='check',backend='ort',steps=240,threads=threads,out=None)
            with patch.object(pico,'verify_inputs'),patch.object(checker,'check') as check,redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):modes.run_mode(args,argparse.ArgumentParser())
                check.assert_not_called()

    def test_check_uses_selected_threads_in_split_and_full_networks(self):
        import numpy as np
        from contextlib import nullcontext
        import check_pico_aligned as checker
        from deployment.vita_s100_aorta_30dof import pico_aligned_runtime as runtime_module
        from deployment.vita_s100_aorta_30dof import actor_human_encoder as encoder_module
        from deployment.vita_s100_aorta_30dof import pico_shadow_math as shadow
        runtime=SimpleNamespace(model=SimpleNamespace(contract=SimpleNamespace(default=np.zeros(28))),math=object(),
            runtime=SimpleNamespace(config={'artifacts':{'model':{'path':'model.onnx'}}}))
        for backend in ('ort','sdk'):
            with self.subTest(backend=backend),patch.object(shadow,'prepare'), \
                    patch.object(runtime_module,'AlignedRuntime',return_value=runtime) as create, \
                    patch.object(runtime_module,'ManualInput'), \
                    patch.object(np,'load',return_value=nullcontext({'body_poses':np.zeros((1,24,7))})), \
                    patch.object(encoder_module,'Network',side_effect=RuntimeError('stop before inference')) as network:
                with self.assertRaisesRegex(RuntimeError,'stop before inference'):
                    checker.check(ROOT,'pico_smpl',backend,240,threads=4)
                create.assert_called_once_with(ROOT,'pico_smpl',backend=backend,threads=4)
                self.assertEqual(network.call_args.args[2:4],(backend,4))

    def test_standalone_cli_accepts_threads_and_enforces_the_same_limits(self):
        import check_pico_aligned as checker
        from contextlib import redirect_stderr
        import io
        for threads in (0,17,16):
            argv=['check_pico_aligned.py','--route','pico_teleop','--backend','ort',
                '--out',str(ROOT/'unused-report.json'),'--threads',str(threads)]
            with patch.object(sys,'argv',argv),patch.object(checker,'check',side_effect=RuntimeError('inference boundary')) as check,redirect_stderr(io.StringIO()):
                if threads==16:
                    with self.assertRaisesRegex(RuntimeError,'inference boundary'):checker.main()
                    check.assert_called_once_with(ROOT,'pico_teleop','ort',240,threads=16)
                else:
                    with self.assertRaises(SystemExit):checker.main()
                    check.assert_not_called()


class InteractivePolicyAdmissionTests(unittest.TestCase):
    def select(self,isolated,answer):
        bash=os.environ.get('PICO_BASH_EXE') or shutil.which('bash')
        if not bash:self.skipTest('Bash required for actual shell menu selection regression')
        shell=(ROOT/'scripts/start_actor_pico.sh').read_text()
        start=shell.index('choose_policy() {');end=shell.index('\n}\n',start)+2
        menu=shell.index('case "$answer" in');menu_end=shell.index('\n    esac',menu)+len('\n    esac')
        # Execute the real validator and menu statement with simulated terminal
        # input. No environment setup, service or actor command is included.
        code='PICO_ONLY="$1"; answer="$2"; USAGE=test; POLICY=\n'+shell[start:end]+'\n'+shell[menu:menu_end]+'\nprintf "ADMITTED:%s\\n" "$POLICY"\n'
        return subprocess.run([bash,'--noprofile','--norc','-c',code,'menu-test',str(int(isolated)),answer],
            capture_output=True,text=True)
    def test_hidden_legacy_choices_are_rejected_in_pico_menu(self):
        for answer in ('1','sonic','2','supermimic','3','morphmimic'):
            with self.subTest(answer=answer):
                result=self.select(True,answer)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr)
                self.assertNotIn('ADMITTED:',result.stdout)
                self.assertIn('isolated Pico R4 delivery supports',result.stderr)
    def test_original_menu_choices_still_select_the_same_policy(self):
        for answer,policy in (('1','sonic'),('sonic','sonic'),('2','supermimic'),('supermimic','supermimic'),('3','morphmimic'),('morphmimic','morphmimic')):
            with self.subTest(answer=answer):
                result=self.select(False,answer)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(result.stdout.strip(),'ADMITTED:'+policy)
    def test_both_pico_choices_work_in_the_pico_menu(self):
        for answer,policy in (('4','pico_smpl'),('pico_smpl','pico_smpl'),('5','pico_teleop'),('pico_teleop','pico_teleop')):
            with self.subTest(answer=answer):
                result=self.select(True,answer)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(result.stdout.strip(),'ADMITTED:'+policy)


if __name__=='__main__':unittest.main()
