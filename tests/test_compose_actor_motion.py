import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

spec=importlib.util.spec_from_file_location('compose',Path(__file__).parents[1]/'scripts/compose_actor_motion.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ComposeMotionTests(unittest.TestCase):
    def test_time_ramp_preserves_endpoints_and_interior_speed(self):
        t=m.source_times(8,50,.6)
        self.assertEqual(t[0],0);self.assertEqual(t[-1],8)
        self.assertTrue(np.all(np.diff(t)>=0))
        self.assertLess((t[1]-t[0])*50,.002)
        self.assertLess((t[-1]-t[-2])*50,.002)
        np.testing.assert_allclose(np.diff(t[40:-40])*50,1,atol=1e-10)

    def fixture(self,d):
        clips=[]
        for i in range(2):
            t=np.linspace(0,2,241);q=(i*.4+.1*t)[:,None]
            b=np.c_[i*10+.1*t,np.full(len(t),i*3.),np.full(len(t),.9)]
            quat=Rotation.from_euler('z',np.full(len(t),i*np.pi/2)).as_quat()[:,[3,0,1,2]]
            path=d/f'{i}.npz'
            np.savez(path,joint_names=['joint'],joint_pos=q,base_pos_w=b,base_quat_w=quat,fps=120.,joint_space='virtual_pitch_roll')
            clips.append(dict(id=i+1,path=path.name,sha256=m.digest(path)))
        s=dict(schema='vita_actor_sequence_v1',clips=clips)
        (d/'spec.json').write_text(json.dumps(s));(d/'policy.json').write_text(json.dumps(dict(body_profile=dict(joint_limits=dict(joint=[-1,1])))))
        return s

    def test_root_alignment_transition_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);self.fixture(d)
            receipt=m.compose(d/'spec.json',d/'policy.json',d/'out')
            z=np.load(d/'out/sequence.npz');a,tr,b=receipt['timeline']
            lo,hi=tr['start_frame'],tr['end_frame']
            q=z['joint_pos'][:,0]
            np.testing.assert_allclose(q[[lo,hi]],[.2,.4])
            self.assertTrue(np.all(np.diff(q[lo:hi+1])>=0))
            self.assertLessEqual(np.max(abs(np.diff(q[lo:hi+1])))*50,1)
            np.testing.assert_allclose(z['base_pos_w'][lo],z['base_pos_w'][hi],atol=1e-12)
            np.testing.assert_allclose(z['base_quat_w'][lo],z['base_quat_w'][hi],atol=1e-12)
            np.testing.assert_allclose(z['base_pos_w'][-1]-z['base_pos_w'][hi],[0,-.2,0],atol=1e-12)
            self.assertEqual(z['source_clip_id'][lo],1);self.assertEqual(z['source_clip_id'][hi],2)
            self.assertTrue(np.all(z['source_clip_id'][lo+1:hi]==-1))
            self.assertEqual(receipt['sha256'],m.digest(d/'out/sequence.npz'))
            self.assertEqual(receipt['frames'],len(q))

    def test_reject_hash_and_limit_violation(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);s=self.fixture(d);s['clips'][0]['sha256']='bad';(d/'spec.json').write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'SHA256'):m.compose(d/'spec.json',d/'policy.json',d/'bad')
            self.fixture(d);(d/'policy.json').write_text(json.dumps(dict(body_profile=dict(joint_limits=dict(joint=[-.1,.1])))))
            with self.assertRaisesRegex(ValueError,'reference limits'):m.compose(d/'spec.json',d/'policy.json',d/'bad')
            self.assertFalse((d/'bad').exists())

if __name__=='__main__':unittest.main()
