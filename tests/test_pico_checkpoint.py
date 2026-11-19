"""Checkpoint consistency and bundled-model provisioning, with no inference I/O."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from bootstrap_actor_s100 import provision_external
class CheckpointTests(unittest.TestCase):
    def test_all_three_configs_select_35000(self):
        for filename in ('actor_pico','actor_smpl','actor_threepoint'):
            settings=json.loads((ROOT/f'config/{filename}.json').read_text())
            self.assertEqual(settings['checkpoint_iteration'],35000)
            self.assertIn('035000',settings['artifacts']['model']['path'])
        smpl=json.loads((ROOT/'config/sonic_smpl_27000_contract.json').read_text())
        teleop=json.loads((ROOT/'config/sonic_threepoint_27000_contract.json').read_text())
        self.assertEqual(smpl['physical_contract'],teleop['physical_contract'])
        self.assertEqual(smpl['checkpoint_iteration'],35000)
        self.assertEqual(teleop['checkpoint_iteration'],35000)
    def test_bundled_model_requires_no_separate_external_declaration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'model.onnx').write_bytes(b'pinned-test-bytes')
            item=dict(path='model.onnx',bytes=17,sha256=hashlib.sha256(b'pinned-test-bytes').hexdigest())
            provision_external(root,{'files':[item],'external_resources':[]},item)
            (root/'model.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Bundled model changed'):
                provision_external(root,{'files':[item],'external_resources':[]},item)
    def test_bundled_model_override_must_match_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'model.onnx').write_bytes(b'pinned');(root/'other.onnx').write_bytes(b'other')
            item=dict(path='model.onnx',bytes=6,sha256=hashlib.sha256(b'pinned').hexdigest())
            with self.assertRaisesRegex(RuntimeError,'override disagrees'):
                provision_external(root,{'files':[item],'external_resources':[]},item,root/'other.onnx')
if __name__=='__main__':unittest.main()
