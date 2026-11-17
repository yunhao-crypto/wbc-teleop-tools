"""Optional dense-rate capture uses actual FixedMotion sampling, with inert I/O."""
from types import SimpleNamespace
import numpy as np
from deployment.vita_s100_aorta_30dof.actor_motion import FixedMotion


class Motion:
    def __init__(self,q,base,quat):self.q=q;self.base=base;self.quat=quat


def setup(velocity=True):
    motion=Motion(np.arange(20*28,dtype=float).reshape(20,28)*1e-4,np.zeros((20,3)),np.tile([1.,0,0,0],(20,1)))
    if velocity:motion.joint_vel=np.arange(20*28,dtype=float).reshape(20,28)*.17
    sentinel=object()
    runtime=SimpleNamespace(reference=SimpleNamespace(current_index=8,steps=13),math=SimpleNamespace(Motion=Motion),model=SimpleNamespace(contract=SimpleNamespace(period=.02)),
                            infer_motion_command=lambda *a,**kw:(sentinel,np.zeros(1224),dict(record_type='actor_inference')))
    f=FixedMotion(runtime,motion,0.,{});f.prepared=motion;f.tick=8;f.lead_frames=0;f.total_ticks=20;f.yaw_offset=.03
    return f,runtime,sentinel


def test_optional_dense_velocity_capture_does_not_modify_inputs_or_command():
    f,r,s=setup();original=f.prepared.joint_vel.copy()
    kwargs=dict(q=np.zeros(28),dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))
    off=f.infer(**kwargs);r._control_timeline=object();on=f.infer(**kwargs)
    assert off[0] is on[0] is s
    np.testing.assert_array_equal(off[1],on[1])
    assert 'joint_velocity' not in off[2]['motion_reference']
    np.testing.assert_array_equal(on[2]['motion_reference']['joint_velocity'],original[:13])
    np.testing.assert_array_equal(f.prepared.joint_vel,original)
    assert on[2]['motion_reference']['velocity_space']=='virtual_pitch_roll_rad_s'


def test_legacy_missing_velocity_not_fabricated():
    f,r,_=setup(False);r._control_timeline=object()
    _,_,row=f.infer(q=np.zeros(28),dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))
    assert 'joint_velocity' not in row['motion_reference']
