#!/usr/bin/env python3
"""Replay a qualified native reference through WTT/ORT/Body31 in MJLab.

No SDK transport is created. Requires the training Python environment, an
exported actor, and a 50 Hz NPZ from SuperMimic's convert_op3_reference.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--supermimic-repo',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'src'),str(args.supermimic_repo.resolve()/'src')]
    import numpy as np
    import torch
    from supermimic.scripts.train_utils import create_env, setup_device
    from supermimic.tasks.vita02a_op3 import OP3MotionCommand, vita02a_op3_env_cfg
    from deployment.vita_s100_aorta_30dof.actor_pico_runtime import load_actor_config,sdk_math,ActorModel
    from deployment.vita_s100_aorta_30dof.actor_supermimic import create_policy
    from deployment.vita_s100_aorta_30dof.actor_supermimic_op3 import load_native_motion
    torch.set_num_threads(1);torch.compiler.set_stance('force_eager')
    config=load_actor_config(root,'supermimic_op3');m=sdk_math(root,config)
    model=ActorModel(root/config['artifacts']['model']['path'],m,backend='ort',policy_kind=config['policy_kind'])
    policy=create_policy(model.contract,config['body_profile'],m,config)
    motion=load_native_motion(args.motion,model.contract,m)
    cfg=vita02a_op3_env_cfg(play=False)
    cfg.seed=20261117;cfg.auto_reset=False
    cfg.events['foot_friction'].params['ranges']=(1.,1.)
    cfg.events['base_com'].params['ranges']={i:(0.,0.) for i in range(3)}
    cfg.events['encoder_bias'].params['bias_range']=(0.,0.)
    cfg.episode_length_s=(len(motion.q)+10)*.02
    for group in cfg.observations.values():group.enable_corruption=False
    mc=cfg.commands['motion'];mc.motion_file=str(args.motion.resolve())
    mc.pose_range={};mc.velocity_range={};mc.resampling_time_range=(1e9,1e9);mc.sampling_mode='start'
    def reset(command,ids):
        t=torch.zeros(len(ids),device=command._env.device,dtype=torch.long)
        command.set_time_steps(ids,t)
        data=command.motion
        command._write_reference_state_to_sim(ids,data.root_pos_w[t]+command._env.scene.env_origins[ids],
            data.root_quat_w[t],data.root_lin_vel_w[t],data.root_ang_vel_w[t],data.joint_pos[t],data.joint_vel[t])
        command._pending_forward=True
    device=setup_device(args.device)
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
    with patch.object(OP3MotionCommand,'_resample_command',reset):
        wrapped=create_env(cfg,1,device)
        env=wrapped.unwrapped
        env.sim.mj_model.geom_friction[:,0]=1.;env.sim.model.geom_friction[...,0]=1.
        robot=env.scene['robot'];cmd=env.command_manager.get_term('motion')
        ids=[list(robot.joint_names).index(n) for n in model.contract.names]
        returns=0.;failed=False;steps=0;target_error=0.
        with torch.inference_mode():
            obs,_=env.reset()
            for _ in range(min(args.steps,len(motion.q)-1)):
                q=robot.data.joint_pos[0,ids].cpu().numpy()
                qd=robot.data.joint_vel[0,ids].cpu().numpy()
                quat=robot.data.root_link_quat_w[0].cpu().numpy()
                gyro=obs['actor'][0,3:6].cpu().numpy()
                x=policy.observe(motion,int(cmd.time_steps[0])*.02,q,qd,quat,gyro)
                action=model.infer(x);command,row=policy.decode(action)
                target=np.array([command.joints[i].q for i in policy.mapping])
                expected=model.contract.policy_default+model.contract.scale*action
                target_error=max(target_error,float(abs(target-expected).max()))
                # Feed the actual decoded Body31 setpoints back into the simulator's action interface.
                decoded=(target-model.contract.policy_default)/model.contract.scale
                obs,reward,terminated,truncated,_=env.step(torch.tensor(decoded[None],dtype=torch.float32,device=device))
                returns+=float(reward[0]);steps+=1
                if bool(terminated[0]) or bool(truncated[0]):
                    failed=bool(terminated[0]);break
        env.close()
    report=dict(steps=steps,requested_steps=args.steps,failed=failed,return_sum=returns,
                max_body31_target_rounding_error=target_error,onnx_sha256=config['artifacts']['model']['sha256'],
                motion=args.motion.name,motion_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
                hardware_tested=False,
                scope='single-reference MJLab replay through WTT observation, ORT and native Body31 decode')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if failed:raise SystemExit('Policy terminated during deployment-chain simulation; inspect report')


if __name__=='__main__':main()
