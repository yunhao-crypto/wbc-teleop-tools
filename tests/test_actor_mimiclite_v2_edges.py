"""Regression cases discovered by real hist10 checkpoint feature replay."""
import importlib.util
import json
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_actor_mimiclite_v2 import ROOT, v2_contract, motion, policy_sdk
from deployment.vita_s100_aorta_30dof import actor_mimiclite as ml


def oracle_module():
    path=ROOT.parent/'MimicLite/mimic-lite/mimic_lite/vita02a_deploy.py'
    if not path.exists(): pytest.skip('Training-side checkout not available')
    spec=importlib.util.spec_from_file_location('_hist10_edges_oracle',path)
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try: spec.loader.exec_module(module)
    finally: sys.path.pop(0)
    return module


def test_quantized_reference_heading_and_named_relative_histories(policy_sdk):
    c=v2_contract(); d=c.deployment
    profile=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p=ml.policy_class(policy_sdk)(c,profile)
    oracle=oracle_module().MotorAngleInterface(d)
    reference=motion(policy_sdk,c)
    # Reproduce fp16 dataset roundoff: scipy normalization is NOT what training
    # projected_yaw_quat does to a reference's horizontal axis projection.
    raw_quat=Rotation.from_euler('xyz',[.4,.8,1.2]).as_quat()[[3,0,1,2]].astype(np.float16).astype(float)
    ids=np.clip(10+np.array(d['future_steps']),0,len(reference.q)-1)
    positions=np.stack([np.linspace(-.3,.4,8),np.linspace(.2,-.2,8),np.ones(8)*.9],axis=1)
    quats=np.tile(raw_quat,(8,1))
    refq=c.ankle.to_policy(reference.q[ids]);refvel=c.ankle.velocity_to_policy(reference.q[ids],reference.joint_vel[ids])
    ref=dict(root_position=positions,root_quaternion=quats,joint_position=refq,joint_velocity=refvel)
    for tick in range(24):
        if tick==12: p.reset();oracle.reset()
        q=c.default.copy();q[c.names.index('left_elbow_joint')]+=.02*tick
        dq=np.ones(28)*.02;gyro=np.array([.01*tick,0,0])
        got=p.observe(reference,.20,q,dq,[1,0,0,0],gyro,reference=ref)
        expected=oracle.observe(c.ankle.to_policy(q),c.ankle.velocity_to_policy(q,dq),[1,0,0,0],gyro,
                                dict(ref,joint_names=list(c.motor_names)))
        np.testing.assert_allclose(got,np.concatenate([expected[k][0] for k in c.input_groups]),atol=2e-6,rtol=1e-6)
        action=np.linspace(-1,1,28).astype(np.float32)*tick
        p.decode(action);oracle.decode(action)


def test_explicit_named_reference_is_reordered(policy_sdk):
    c=v2_contract()
    profile=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p=ml.policy_class(policy_sdk)(c,profile)
    ref=motion(policy_sdk,c)
    ids=np.clip(10+np.array(c.deployment['future_steps']),0,len(ref.q)-1)
    mq=c.ankle.to_policy(ref.q[ids]);mv=c.ankle.velocity_to_policy(ref.q[ids],ref.joint_vel[ids])
    args=(ref,.20,c.default,np.zeros(28),[1,0,0,0],np.zeros(3))
    direct=p.observe(*args,reference={'joint_position':mq,'joint_velocity':mv})
    p.reset()
    named=p.observe(*args,reference={'joint_position':mq[:,::-1],'joint_velocity':mv[:,::-1],
                                    'joint_names':list(reversed(c.motor_names))})
    np.testing.assert_array_equal(direct,named)
    with pytest.raises(ValueError,match='joint_names'):
        p.observe(*args,reference={'joint_position':mq,'joint_velocity':mv,'joint_names':['bad']*28})


def test_explicit_reference_motor_travel_is_not_bypassed(policy_sdk):
    c=v2_contract()
    profile=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p=ml.policy_class(policy_sdk)(c,profile)
    ref=motion(policy_sdk,c)
    q=np.tile(c.policy_default,(8,1));q[:,c.ankle.slots[0]]=10.
    with pytest.raises(ValueError,match='motor travel'):
        p.observe(ref,.20,c.default,np.zeros(28),[1,0,0,0],np.zeros(3),
                  reference_motor_q=q,reference_joint_velocity=np.zeros((8,28)))
