#!/usr/bin/env python3
"""Compare WTT inputs, ORT actions and Body31 targets against training code.

CPU only, no transport. Requires SuperMimic dependencies and onnxruntime.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--supermimic-repo',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'src'),str(args.supermimic_repo.resolve()/'src')]
    import numpy as np
    import torch
    from supermimic.assets.vita02a_op3.robot import close_linkage
    from supermimic.representations import _assemble_goal
    from supermimic.models import MultiEncoderAE
    from supermimic.tasks.task_spec import TaskSpec
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.exporter import _SingleEncoderOnnx
    from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorModel, load_actor_config, sdk_math
    from deployment.vita_s100_aorta_30dof.actor_supermimic import create_policy
    from deployment.vita_s100_aorta_30dof.actor_supermimic_op3 import native_reference, OFFSETS
    torch.set_num_threads(1)
    config=load_actor_config(root,'supermimic_op3'); m=sdk_math(root,config)
    model=ActorModel(root/config['artifacts']['model']['path'],m,backend='ort',policy_kind=config['policy_kind'])
    c=model.contract; policy=create_policy(c,config['body_profile'],m,config)
    state=torch.load(args.checkpoint/'rank_0000.pt',map_location='cpu',weights_only=False,mmap=True)
    cfg=load_runner_cfg({'runner_cfg':state['signature']['runner']})
    actor=MultiEncoderAE(90,28,[TaskSpec('motion',860,lambda *_:None),TaskSpec('velocity',3,lambda *_:None)],
                        cfg.slot_mix,cfg.latent_dim,cfg.actor_hidden_dims,cfg.latent_bc_weight,cfg.history_length).eval()
    actor.load_state_dict(state['actor']); wrapper=_SingleEncoderOnnx(actor,'motion',[3,3,28,28,28]).eval()
    rng=np.random.default_rng(24)
    q=c.default+rng.normal(0,.025,(100,28)); qd=rng.normal(0,.1,(100,28))
    source=np.zeros((100,40)); speed=np.zeros((100,40))
    virtual_ids=[c.reference_names.index(n) for n in c.names]
    motor_ids=[c.reference_names.index(n) for n in c.native_names]
    source[:,virtual_ids],speed[:,virtual_ids]=q,qd
    expected_q,expected_v=close_linkage(torch.tensor(source),torch.tensor(speed),list(c.reference_names))
    nq,nv=native_reference(q,qd,c)
    np.testing.assert_allclose(nq,expected_q.numpy(),atol=1e-12,rtol=0)
    np.testing.assert_allclose(nv,expected_v.numpy(),atol=2e-9,rtol=0)
    quat=rng.normal(size=(100,4));quat/=np.linalg.norm(quat,axis=1,keepdims=True)
    motion=m.Motion(q,np.zeros((100,3)),quat)
    motion.op3_joint_names=c.reference_names
    motion.op3_joint_pos, motion.op3_joint_vel=nq,nv
    history=[];max_input=max_action=max_target=0.
    for frame in list(range(15))+[80,99]:
        root_quat=rng.normal(size=4);root_quat/=np.linalg.norm(root_quat);gyro=rng.normal(size=3); yaw=.23
        actual=policy.observe(motion,frame*.02,q[frame],qd[frame],root_quat,gyro,yaw)
        proprio=np.concatenate((-m.rotation(root_quat)[2],gyro,expected_q[frame,motor_ids].numpy()-c.policy_default,
                                expected_v[frame,motor_ids].numpy(),policy.last_action)).astype(np.float32)
        if not history: history=[proprio]*10
        else: history=(history+[proprio])[-10:]
        h=np.stack(history)
        indices=np.minimum(frame+np.array(OFFSETS),99)
        target_quat=np.array([m.multiply(m.yaw_quaternion(yaw),quat[i]) for i in indices])
        content=torch.tensor(np.concatenate((nq[indices],nv[indices]),axis=1).reshape(1,-1),dtype=torch.float32)
        _,goal=_assemble_goal(content,torch.zeros(1,3),torch.tensor(root_quat[None],dtype=torch.float32),
                              torch.zeros(1,10,3),torch.tensor(target_quat[None],dtype=torch.float32))
        expected=np.concatenate([*(x.ravel() for x in np.split(h,[3,6,34,62],axis=1)),goal.numpy()[0]])
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
        max_input=max(max_input,float(np.max(abs(actual-expected))))
        with torch.inference_mode(): action=wrapper(torch.from_numpy(expected[None])).numpy()[0]
        out=model.infer(actual)
        np.testing.assert_allclose(out,action,atol=3e-5,rtol=3e-5)
        max_action=max(max_action,float(np.max(abs(out-action))))
        cmd,row=policy.decode(out)
        target=c.policy_default+c.scale*action
        sent=np.array([cmd.joints[i].q for i in policy.mapping])
        np.testing.assert_allclose(sent,target,atol=2e-5,rtol=2e-5)
        max_target=max(max_target,float(np.max(abs(sent-target))))
    report=dict(linkage_cases=100,observation_action_target_cases=17,max_input_error=max_input,
                max_action_error=max_action,max_target_error=max_target,hardware_tested=False)
    (root/'artifacts/supermimic_op3/input_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
