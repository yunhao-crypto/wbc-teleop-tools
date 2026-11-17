"""Exercise fixed-file playback with official OP3, without GMR or hardware."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_actor_motion
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_supermimic_op3 import native_reference


class OP3MotionTests(test_actor_motion.MotionTests):
    @classmethod
    def setUpClass(cls):
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_runtime.create_retargeter',
                   side_effect=AssertionError('File source must not construct GMR')):
            # The inherited fixture calls its default runtime 'sonic'; its model
            # is explicitly OP3, and every inherited lifecycle assertion runs.
            cls.runtimes={'sonic':ActorPicoRuntime(ROOT,policy='supermimic_op3',backend='ort',reference_source='npz')}

    def test_native_rates_survive_lead_in_and_bounded_windows(self):
        c=self.runtime.model.contract
        q=np.tile(c.default,(5,1));qd=np.zeros_like(q)
        qd[:,c.names.index('left_elbow_joint')]=.123
        qp,qv=native_reference(q,qd,c)
        self.data=dict(joint_names=np.array(c.reference_names),joint_pos=qp,joint_vel=qv,
                       root_pos_w=np.tile([0,0,.9],(5,1)),root_quat_w=np.tile([1,0,0,0],(5,1)),fps=50.)
        source=self.source()
        self.assertTrue(source.metadata['native_rates_preserved'])
        source.begin(c.default,[1,0,0,0])
        source.tick=source.lead_frames+2
        window,t=source.sample()
        frame=round(t/c.period)
        np.testing.assert_array_equal(window.op3_joint_vel[frame],qv[2])
        _,obs,_=source.infer(q=c.default,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))
        np.testing.assert_allclose(obs[940:980],qv[2],atol=1e-7)
