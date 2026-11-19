"""Two new checkpoints without replacing an existing policy or default build."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
class CheckpointTests(unittest.TestCase):
    def test_new_routes_select_35000_and_identical_physical_contract(self):
        for filename in ('actor_smpl','actor_threepoint'):
            settings=json.loads((ROOT/f'config/{filename}.json').read_text())
            self.assertEqual(settings['checkpoint_iteration'],35000)
            self.assertIn('035000',settings['artifacts']['model']['path'])
        smpl=json.loads((ROOT/'config/sonic_smpl_27000_contract.json').read_text())
        teleop=json.loads((ROOT/'config/sonic_threepoint_27000_contract.json').read_text())
        self.assertEqual(smpl['physical_contract'],teleop['physical_contract'])
        self.assertEqual(smpl['checkpoint_iteration'],35000)
        self.assertEqual(teleop['checkpoint_iteration'],35000)
    def test_original_default_policy_and_sdk_are_retained(self):
        settings=json.loads((ROOT/'config/actor_pico.json').read_text())
        self.assertEqual(settings['artifacts']['model']['source_name'],'model_step_009700_longrun_closed_loop_g1.onnx')
        self.assertEqual(settings['artifacts']['model']['sha256'],'2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d')
        self.assertEqual(settings['sdk_version'],'0.3.0')
        self.assertEqual(settings['sdk_commit'],'a0b50cef329181b138239f5576238fcb9e0c02ca')
    def test_default_build_has_only_the_original_policy_models(self):
        from build_actor_delivery import runtime_inputs
        for variant in ('base','planner'):
            with self.subTest(variant=variant):
                names,external=runtime_inputs(ROOT,variant)
                self.assertEqual([x['role'] for x in external[:3]],['sonic_actor','supermimic_actor','morphmimic_actor'])
                self.assertFalse(any('035000' in x['path'] for x in external))
                self.assertNotIn('config/actor_smpl.json',names)
                self.assertNotIn('config/actor_threepoint.json',names)
                self.assertNotIn('config/pico_27000_sdk_guard.json',names)
                self.assertFalse(any('bodycmdv4' in name for name in names))
                for filename in ('actor_smpl.py','actor_threepoint.py','actor_human_encoder.py'):
                    self.assertNotIn('src/deployment/vita_s100_aorta_30dof/'+filename,names)
if __name__=='__main__':unittest.main()
