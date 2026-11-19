"""Shared install ownership and public non-actuating output paths, no assets needed."""
import argparse,json,os,subprocess,sys,tempfile,unittest
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
manifest={'variant':'base','pico_inputs_release':{'unified_environment':True}}
with patch.object(actor,'verify',return_value=manifest),patch.object(pico,'verify_inputs',return_value=manifest), \\
     patch.object(actor,'install') as install,patch.object(actor,'unpack') as unpack, \\
     patch.object(actor,'provision_model') as model,patch.object(actor,'provision_planner_assets') as planner:
    try:
        (actor if entry=='actor' else pico).main(['--root',str(root),*(['--unpack-only'] if mode=='unpack' else [])])
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
            with actor.installation_transaction(root):self.assertFalse(receipt.exists())

    def test_failed_install_invalidates_old_and_partial_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'build').mkdir()
            paths=[root/'build'/name for name in ('actor_s100_install.json','pico_inputs_s100_install.json')]
            for path in paths:path.write_text('old')
            with self.assertRaisesRegex(RuntimeError,'installation failed'):
                with actor.installation_transaction(root):
                    self.assertFalse(any(path.exists() for path in paths))
                    actor.write_install_receipt(root,paths[0].name,{'status':'premature'})
                    raise RuntimeError('installation failed')
            self.assertFalse(any(path.exists() for path in paths))

    def test_compatibility_entry_delegates_to_original_transaction(self):
        with patch.object(actor,'main') as main:
            pico.main(['--root','some-root','--unpack-only'])
            main.assert_called_once_with(['--policy','pico_smpl','--root','some-root','--unpack-only'])


class PublicOutputTests(unittest.TestCase):
    def command(self,root,route,mode,out=None):
        args=SimpleNamespace(root=root,policy=route,mode=mode,backend='ort',threads=1,seconds=None,
            pico_service='external',group='default',out=out,record_observations=False)
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


if __name__=='__main__':unittest.main()
