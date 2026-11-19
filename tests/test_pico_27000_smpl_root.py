"""Root-only Pico conversion parity with the original complete SMPL converter.

Load only hash-pinned pure SDK quaternion functions; no native initializer,
controller, service or robot connection is imported by these tests.
"""
import ast,hashlib,json,math,os,sys,unittest,zipfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import pico_threepoint_frontend as front
from deployment.vita_s100_aorta_30dof.pico_realtime import SmplWindow,PicoFlow,Feedback
from deployment.vita_s100_aorta_30dof.actor_pico_buttons import DualTriggerToggle
from deployment.vita_s100_aorta_30dof.onboard_preprocessing import CPUPicoToSmplConverter
from deployment.vita_s100_aorta_30dof.actor_smpl import root_from_pose_aa,WRIST_NAMES


def sdk_math():
    guard=json.loads((ROOT/'config/sdk021-wheel-guard.json').read_text())
    with zipfile.ZipFile(ROOT/'deploy/actor_s100/wheelhouse'/guard['wheel']['filename']) as wheel:
        source=wheel.read('locomotion_aorta/wbc_tracking_policy.py')
    pico_guard=json.loads((ROOT/'config/pico_27000_sdk_guard.json').read_text())
    assert hashlib.sha256(source).hexdigest()==pico_guard['source_sha256']['wbc_tracking_policy.py']
    names={'finite','quaternion','multiply','rotation','yaw','yaw_quaternion'}
    functions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions}==names
    scope={'np':np,'math':math}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned SDK quaternion math>','exec'),scope)
    return SimpleNamespace(**{name:scope[name] for name in names})


class OriginalWindow(SmplWindow):
    def frame(self,body):
        front.anatomical_sites(body)
        pose,_=CPUPicoToSmplConverter().to_device(body)
        root=root_from_pose_aa(self.math,pose,convention='z_up_smpl_base')[0]
        joints=(np.asarray(body[:,:3],dtype=float)-body[0,:3])@front.AXES.T
        return joints@self.yaw_rotation.T,self.math.multiply(self.yaw_quat,root)


def body_with_root(q):
    body=np.zeros((24,7),np.float32);body[:,6]=1.
    body[1,2]=-.1;body[2,2]=.1;body[15,1]=.7
    body[20,:3]=[.2,.2,-.3];body[21,:3]=[.2,.2,.3]
    body[0,3:7]=np.asarray(q)[[1,2,3,0]]
    return body


class RootConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.math=sdk_math()

    def compare(self,bodies,*,align=False):
        actual=SmplWindow(self.math,np.zeros(6));original=OriginalWindow(self.math,np.zeros(6))
        if align:
            heading=self.math.yaw_quaternion(.63)
            actual.align(bodies[0],heading);original.align(bodies[0],heading)
        else:
            # Exercise the complete XYZ/root frame under an existing heading.
            for window in (actual,original):
                window.yaw_quat=self.math.yaw_quaternion(-.41)
                window.yaw_rotation=self.math.rotation(window.yaw_quat)
        maximum=0.
        for body in bodies:
            a_xyz,a_q=actual.frame(body);b_xyz,b_q=original.frame(body)
            np.testing.assert_allclose(a_xyz,b_xyz,atol=2e-6,rtol=0)
            delta=np.max(np.abs(self.math.rotation(a_q)-self.math.rotation(b_q)))
            maximum=max(maximum,float(delta))
            self.assertLessEqual(delta,2e-6)
        return maximum

    def test_near_zero_and_pi_in_original_converted_root_coordinates(self):
        c=np.array([math.sqrt(.5),math.sqrt(.5),0.,0.]);b=np.array([0.,0.,1.,0.])
        conjugate=lambda q:q*np.array([1.,-1.,-1.,-1.])
        bodies=[]
        for axis in np.eye(3):
            for angle in (0.,1e-10,1e-7,math.pi-1e-7,math.pi,math.pi+1e-7):
                target=np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
                q=self.math.multiply(self.math.multiply(conjugate(c),target),conjugate(b))
                bodies.extend([body_with_root(q),body_with_root(-q)])
        self.compare(bodies)

    def test_random_positive_and_negative_quaternions(self):
        rng=np.random.default_rng(35000);values=rng.normal(size=(1024,4))
        values/=np.linalg.norm(values,axis=1,keepdims=True)
        self.compare([body_with_root(sign*q) for q in values for sign in (1.,-1.)])

    def test_first_heading_alignment_and_xyz_use_the_same_coordinates(self):
        rng=np.random.default_rng(35);values=rng.normal(size=(128,4))
        values/=np.linalg.norm(values,axis=1,keepdims=True)
        self.compare([body_with_root(q) for q in values],align=True)

    def test_frame_does_not_call_the_full_body_converter(self):
        with patch.object(CPUPicoToSmplConverter,'to_device',side_effect=AssertionError('full body conversion is unnecessary')):
            SmplWindow(self.math,np.zeros(6)).frame(body_with_root([1.,0.,0.,0.]))

    def test_calibration_reuses_frames_and_preserves_tail_and_clock_offset(self):
        # Exercise the real Flow/TimingGate and calibration assembly, replacing
        # only the asset-dependent neutral-fit approval. A 19ms input cadence
        # crosses 2s on a final packet that must seed playback but not neutral.
        flow=PicoFlow.__new__(PicoFlow)
        flow.route='pico_smpl';flow.math=self.math
        flow.contract=SimpleNamespace(names=list(WRIST_NAMES),default=np.zeros(6))
        flow.gate=front.TimingGate();flow.toggle=DualTriggerToggle(.6)
        flow.raw=deque(maxlen=512);flow.window=None;flow.calibration=None
        flow.mode='calibrating';flow.frames=flow.losses=0;flow.events=[]
        flow.last_frame_ns=None;flow.button_not_before_ns=0;flow.source_fault=None
        flow.standing_path=Path('unused_standing.npz');flow.sites_path=Path('unused_sites.json')
        feedback=Feedback(np.zeros(28),np.zeros(28),self.math.yaw_quaternion(.63),np.zeros(3))
        count=107;stamps=10_000_000_000+np.arange(count,dtype=np.int64)*19_000_000
        received=stamps+123_456_789
        bodies=[]
        for i in range(count):
            body=body_with_root([1.,0.,0.,0.]);body[20,0]+=.02*i;bodies.append(body)
        original_frame=SmplWindow.frame
        with patch.object(front,'fit_calibration',return_value={}), \
             patch.object(SmplWindow,'frame',autospec=True,side_effect=original_frame) as converted:
            for body,stamp,receipt in zip(bodies,stamps,received):
                self.assertTrue(flow.push(body,int(stamp),int(receipt),feedback))
        self.assertEqual(converted.call_count,count+1,'one conversion per packet plus heading alignment')
        self.assertEqual(flow.window.samples.maxlen,512)
        self.assertEqual(flow.window.offset_ns,123_456_789)
        self.assertEqual([row[0] for row in flow.window.samples],stamps.tolist())
        original=OriginalWindow(self.math,np.zeros(6));original.align(bodies[0],feedback.root)
        for body,stamp,receipt in zip(bodies,stamps,received):original.append(body,int(stamp),int(receipt))
        for actual,expected in zip(flow.window.samples,original.samples):
            self.assertEqual(actual[0],expected[0])
            np.testing.assert_allclose(actual[1],expected[1],atol=2e-6,rtol=0)
            np.testing.assert_allclose(self.math.rotation(actual[2]),self.math.rotation(expected[2]),atol=2e-6,rtol=0)
        neutral=[row[1] for row,t in zip(original.samples,stamps-stamps[0]) if t<=2_000_000_000]
        np.testing.assert_allclose(flow.standing.joints[0],np.median(neutral,axis=0),atol=2e-6,rtol=0)
        self.assertGreater(stamps[-1]-stamps[0],2_000_000_000)
        query=int(received[-1])+20_000_000
        actual=flow.window.reference(query);expected=original.reference(query)
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(actual.joints,expected.joints,atol=2e-6,rtol=0)
        for a,b in zip(actual.quat,expected.quat):
            np.testing.assert_allclose(self.math.rotation(a),self.math.rotation(b),atol=2e-6,rtol=0)

    @unittest.skipUnless(os.environ.get('PICO_27000_ASSETS'),'Supply the pinned historical 75-frame asset')
    def test_historical_75_frames_and_heading_alignment(self):
        path=Path(os.environ['PICO_27000_ASSETS'])/'checks/pico_historical_75frames.npz'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            '1d744df8f8376b2c6d2a6e9e1454d6667dd00369e8f1b0a5680c0ecd8c644b11')
        with np.load(path,allow_pickle=False) as data:bodies=data['body_poses']
        self.assertEqual(len(bodies),75)
        self.compare(bodies,align=True)


if __name__=='__main__':unittest.main()
