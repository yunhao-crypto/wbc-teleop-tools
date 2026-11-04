#!/usr/bin/env python3
"""Compare deployment inputs to the training implementation on converted motion.

Requires MorphMimic's Python, SDK math on PYTHONPATH, and an exported model.
No transport is constructed; no robot or GPU is used.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--morph-repo',type=Path,required=True)
    p.add_argument('--clip',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'src'),str(args.morph_repo/'src'),str(args.morph_repo)]
    import numpy as np
    import torch
    import onnxruntime as ort
    import mujoco
    import locomotion_aorta.wbc_tracking_policy as math
    from morphmimic.robots.vita02a import vita02a_spec, serial_to_native, TRACKED
    from morphmimic.robots.vita02a_model import make_spec
    from morphmimic.model.state import StateParts,join_state
    from morphmimic.model.rewards_torch import analytic_obs,goal_features
    from deployment.vita_s100_aorta_30dof.actor_morphmimic import MorphMimicContract,policy_class
    cfg=json.loads((root/'config/actor_pico_morphmimic.json').read_text())
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/cfg['artifacts']['model']['path']),opts,providers=['CPUExecutionProvider'])
    c=MorphMimicContract.from_metadata(math,session.get_modelmeta().custom_metadata_map)
    policy=policy_class(math)(c,cfg['body_profile'])
    spec=vita02a_spec(); f=np.load(args.clip)
    names=list(f['joint_names']);q=f['joint_pos'].astype(float);qd=f['joint_vel'].astype(float)
    serial=q[:,[names.index(n) for n in c.names]]
    motion=math.Motion(serial,f['body_pos_w'][:,0],f['body_quat_w'][:,0])
    def state(q,qd,pos,quat,lin,ang):
        q=torch.tensor(q);qd=torch.tensor(qd)
        return join_state(StateParts(spec.translator.to_canonical(q),spec.translator.to_canonical_delta(qd),
            torch.tensor(pos),torch.tensor(quat),torch.tensor(lin),torch.tensor(ang)))
    reference=state(q,qd,f['body_pos_w'].astype(float),f['body_quat_w'].astype(float),
                    f['body_lin_vel_w'].astype(float),f['body_ang_vel_w'].astype(float))
    default=policy.canonical(c.policy_default)
    max_obs=max_goal=0.;latencies=[]
    model=make_spec(visuals=False).compile();d=mujoco.MjData(model)
    for frame in (2,5,10,20):
        # Deliberately change waist yaw/pitch and base attitude: the goal must
        # use the robot's waist anchor, not its base or the reference anchor.
        robot_q=serial[frame].copy()
        robot_q[c.names.index('waist_yaw_joint')]=.17
        robot_q[c.names.index('waist_pitch_joint')]=-.12
        robot_root=math.multiply(math.yaw_quaternion(.3),motion.quat[frame])
        full,vel=serial_to_native(torch.tensor(robot_q[None]),torch.tensor(np.gradient(serial,.02,axis=0)[frame:frame+1]),tuple(c.names))
        d.qpos[:3]=motion.base[frame];d.qpos[3:7]=robot_root
        d.qpos[7:]=full[0].numpy();mujoco.mj_forward(model,d)
        bid=[model.body(n).id for n in TRACKED]
        gyro=np.array([.11,-.07,.13]);ang=np.zeros((14,3));ang[0]=math.rotation(robot_root)@gyro
        robot=state(full[0].numpy(),vel[0].numpy(),d.xpos[bid].copy(),d.xquat[bid].copy(),np.zeros((14,3)),ang)
        prev=np.linspace(-.3,.2,28);policy.reset();policy.last_action[:]=prev
        expected_obs=analytic_obs(robot,torch.zeros(29),torch.tensor(default),torch.tensor(policy.canonical(prev))).numpy()
        expected_goal=goal_features(robot,reference[frame+1:frame+11]).numpy()
        observation=policy.observe(motion,frame*.02,robot_q,np.gradient(serial,.02,axis=0)[frame],robot_root,gyro)
        np.testing.assert_allclose(observation[:930],np.tile(expected_obs,10),atol=2e-5,rtol=1e-5)
        np.testing.assert_allclose(observation[930:],expected_goal,atol=2e-4,rtol=2e-4)
        max_obs=max(max_obs,float(np.max(np.abs(observation[:93]-expected_obs))))
        max_goal=max(max_goal,float(np.max(np.abs(observation[930:]-expected_goal))))
        for _ in range(6):
            begin=time.perf_counter();action=session.run(None,{'observation':observation[None]})[0][0]
            command,row=policy.decode(action);latencies.append((time.perf_counter()-begin)*1000)
        assert row['ankle_command_space']=='motor_theta'
        for i in c.ankle.slots:
            np.testing.assert_allclose(command.joints[policy.mapping[i]].q,row['motor_target'][list(c.ankle.slots).index(i)],atol=1e-7)
    report=dict(clip=str(args.clip),frames=4,max_obs_error=max_obs,max_goal_error=max_goal,
        onnx_decode_median_ms=float(np.median(latencies)),onnx_decode_max_ms=float(np.max(latencies)))
    output=root/'artifacts/morphmimic/input_parity.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
