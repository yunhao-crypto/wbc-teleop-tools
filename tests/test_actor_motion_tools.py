"""Dataset import and controlled ankle-trial comparison; no hardware."""
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
imp = runpy.run_path(str(ROOT/'scripts/import_actor_motion.py'))
compare = runpy.run_path(str(ROOT/'scripts/compare_actor_motion.py'))
prepare = runpy.run_path(str(ROOT/'scripts/prepare_external_ankle_config.py'))['prepare']


class MotionToolsTests(unittest.TestCase):
    def test_import_verified_original_and_refuse_bad_hash_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); archive=root/'data.tar.gz'; blob=b'npz-test-payload'
            def pack(sha):
                with tarfile.open(archive,'w:gz') as t:
                    for name,data in [('pack/SHA256SUMS',(sha+'  L1/clip.npz\n').encode()),('pack/L1/clip.npz',blob)]:
                        info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
            pack(hashlib.sha256(blob).hexdigest())
            result=imp['install'](archive,'L1/clip.npz',root)
            self.assertEqual(Path(result['path']).read_bytes(),blob)
            with self.assertRaises(FileExistsError):imp['install'](archive,'L1/clip.npz',root)
            with self.assertRaises(ValueError):imp['selected_bytes'](archive,'../L1/clip.npz')
            pack('0'*64)
            with self.assertRaisesRegex(ValueError,'SHA256'):imp['install'](archive,'L1/clip.npz',root,True)
            self.assertEqual(Path(result['path']).read_bytes(),blob)

    def test_prepare_sets_native_opt_in_and_clears_it_for_legacy(self):
        original=dict(parallel_ankle={'geometry':{}},actions={'EXTERNAL':{'generator':'builtin::external',
                      'params':{'ankle_virtual_pd_intent':False,'ankle_command_space':'motor_theta','command_timeout_ms':1000,'interpolation':{'min_period_ms':2,'max_period_ms':100}}}})
        source=yaml.safe_dump(original,sort_keys=False)
        a=yaml.safe_load(prepare(source,'motor_theta',False))
        b=yaml.safe_load(prepare(source,'virtual_pitch_roll',False))
        self.assertTrue(a['actions']['EXTERNAL']['params']['allow_position_targets_outside_mechanical_range'])
        self.assertFalse(b['actions']['EXTERNAL']['params']['allow_position_targets_outside_mechanical_range'])
        self.assertFalse(b['actions']['EXTERNAL']['params']['ankle_virtual_pd_intent'])
        self.assertEqual(a['actions']['EXTERNAL']['params']['position_target_margin_rad'],3.14159)
        self.assertNotIn('position_target_margin_rad',b['actions']['EXTERNAL']['params'])
        b['actions']['EXTERNAL']['params']['ankle_command_space']='motor_theta'
        b['actions']['EXTERNAL']['params']['allow_position_targets_outside_mechanical_range']=True
        b['actions']['EXTERNAL']['params']['position_target_margin_rad']=3.14159
        self.assertEqual(a,b)
        original['actions']['EXTERNAL']['params']['allow_position_targets_outside_mechanical_range']=True
        original['actions']['EXTERNAL']['params']['position_target_margin_rad']=3.14159
        original['actions']['EXTERNAL']['params']['interpolation']['enabled']=False
        self.assertEqual(a,original)
        for enabled in (True, False):
            original['actions']['EXTERNAL']['params']['interpolation']['enabled']=enabled
            direct=yaml.safe_load(prepare(yaml.safe_dump(original,sort_keys=False),'motor_theta'))
            self.assertIs(direct['actions']['EXTERNAL']['params']['interpolation']['enabled'],False)
        # Source already opted in: a virtual-wire rewrite must reset the flag.
        rewritten=prepare(yaml.safe_dump(original,sort_keys=False),'virtual_pitch_roll',False)
        self.assertFalse(yaml.safe_load(rewritten)['actions']['EXTERNAL']['params']['allow_position_targets_outside_mechanical_range'])
        self.assertNotIn('position_target_margin_rad',yaml.safe_load(rewritten)['actions']['EXTERNAL']['params'])
        with self.assertRaises(ValueError):prepare(source,'motor_theta',True)
        for margin in (0,-1.,3.2,float('nan'),True):
            with self.subTest(margin=margin),self.assertRaises(ValueError):prepare(source,'motor_theta',False,margin)
        custom=yaml.safe_load(prepare(source,'motor_theta',False,3.12))
        self.assertEqual(custom['actions']['EXTERNAL']['params']['position_target_margin_rad'],3.12)

    def trial(self,space):
        header=dict(reference_source='npz',policy='morphmimic',model={'sha256':'model'},comparison_config_sha256='cfg',
            motion=dict(sha256='clip',limit_corrections=[],end_hold_s=1.,root_body_index=0,
                        joint_names=['left_ankle_roll_joint'],policy_frames=2,period_s=.02),
            contract=dict(required_ankle_command_space=space,ankle_control_space='motor_theta',required_ankle_virtual_pd_intent=False))
        frames={i:dict(feedback=dict(q_body=[.1],root_wxyz=[1.,0.,0.,0.]),body_order=['left_ankle_roll_joint'],
                       motion_reference=dict(ref_time=0.,q=[[0.]]),compute_ms=2.) for i in range(2)}
        return dict(path=space,header=header,robot_hash='robot-cfg',frames=frames,
                    summary=dict(exit_confirmed=True,stop_reason='motion_complete'))

    def test_compare_matching_clip_frames_and_reject_confounds(self):
        a,b=self.trial('motor_theta'),self.trial('virtual_pitch_roll')
        result=compare['compare'](a,b)
        self.assertTrue(result['both_clips_complete']);self.assertEqual(result['common_clip_frames'],2)
        self.assertAlmostEqual(result['trials'][0]['ankle_rmse_deg'],5.7295779513)
        for field in ('robot_hash','model','projection','space','reference','semantics'):
            bad=copy.deepcopy(b)
            if field=='robot_hash':bad['robot_hash']='other'
            elif field=='model':bad['header']['model']['sha256']='other'
            elif field=='projection':bad['header']['motion']['limit_corrections']=[{'delta_rad':.1}]
            elif field=='space':bad['header']['contract']['required_ankle_command_space']='motor_theta'
            elif field=='semantics':bad['header']['contract']['position_target_semantics']='affine_unclipped'
            else:bad['frames'][0]['motion_reference']['q']=[[.2]]
            with self.subTest(field=field),self.assertRaises(ValueError):compare['compare'](a,bad)
        del b['frames'][1]
        self.assertFalse(compare['compare'](a,b)['both_clips_complete'])

    def test_read_trial_matches_real_log_record_structure(self):
        t=self.trial('motor_theta')
        rows=[dict(t['header'],record_type='header'),
              dict(record_type='robot_configuration',robot_config={'ankle_path_comparison_sha256':'robot-cfg'}),
              dict(record_type='motion_started',lead_frames=3),dict(t['summary'],record_type='summary')]
        rows.extend(dict(r,stage='motion_tracking',motion_phase='clip',motion_tick=i+3) for i,r in t['frames'].items())
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'trial.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=compare['read_trial'](p)
            self.assertEqual(set(result['frames']),{0,1});self.assertEqual(result['robot_hash'],'robot-cfg')


if __name__=='__main__':unittest.main()
