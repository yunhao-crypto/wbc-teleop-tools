#!/usr/bin/env python3
"""Read-only synthetic/recorded Pico replay against the existing CUDA pipeline.

Never imports Aorta or starts Pico/robot SDKs. Raw input+CPU/GPU observations are
saved for independent full-policy action comparisons by the deployment harness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--skeleton',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--raw-input',help='NPZ with body_poses[N,24,7], sample_times[N]')
    ap.add_argument('--frames',type=int,default=384)
    ap.add_argument('--seed',type=int,default=20260929)
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(root/'src/deployment/unitree_g1_ros2_29dof/src'))
    os.environ['HOLOMOTION_SMPL_MODEL_PATH']=a.model
    import warp as wp
    from humanoid_policy.smpl_reference import GpuOnlineSmplReference
    from humanoid_policy.smpl_motion_observation import GpuSmplMotionObservationBuilder
    from deployment.vita_s100_aorta_30dof.onboard_preprocessing import (
        CPUOnlineSmplReference,CPUSmplMotionObservationBuilder,_axis_quat)
    rng=np.random.default_rng(a.seed)
    n=a.frames
    if a.raw_input:
        with np.load(a.raw_input,allow_pickle=False) as data:
            bodies=data['body_poses'].astype(np.float32);stamps=data['sample_times'].astype(np.float64)
        n=len(bodies);kind='recorded_raw_pico'
    else:
        t=np.arange(n,dtype=np.float32)*np.float32(.02)
        amplitudes=rng.uniform(.03,.5,(24,3)).astype(np.float32)
        phases=rng.uniform(-np.pi,np.pi,(24,3)).astype(np.float32)
        aa=amplitudes[None]*np.sin(t[:,None,None]*np.float32(.8)+phases[None])
        q=_axis_quat(aa)
        # Pico global xyzw input, smooth rotations with deliberate sign flips.
        bodies=np.zeros((n,24,7),dtype=np.float32)
        bodies[:,:,3:]=q[:,:,[1,2,3,0]]
        bodies[::17,:,3:]*=np.float32(-1)
        bodies[:,:,0]=np.float32(.1)*np.sin(t)[:,None]
        bodies[:,:,1]=np.float32(1)+np.float32(.04)*np.cos(t)[:,None]
        bodies[:,:,2]=np.float32(.06)*np.cos(t*np.float32(.4))[:,None]
        # Slightly irregular sampling exercises timestamp-based derivatives.
        stamps=1700000000.+np.cumsum(.02+rng.uniform(-.001,.001,n))
        kind='synthetic_raw_pico_not_real_data'
    if n<64: raise ValueError('Need at least64 raw frames')
    reset_frames={90,151,250,333}
    defaults=np.linspace(-.1,.1,30,dtype=np.float32)
    cpu=CPUOnlineSmplReference(n_future_frames=10,policy_hz=50,skeleton_path=a.skeleton)
    gpu=GpuOnlineSmplReference(n_future_frames=10,policy_hz=50)
    cb=CPUSmplMotionObservationBuilder(n_future_frames=10,policy_hz=50,default_dof_pos=defaults)
    gb=GpuSmplMotionObservationBuilder(n_future_frames=10,policy_hz=50,default_dof_pos=defaults)
    cpu_obs=[];gpu_obs=[];cpu_root=[];gpu_root=[];cpu_body=[];gpu_body=[];sample_indices=[]
    positions=[];velocities=[];actions=[];robot_quats=[];angvels=[];alignments=[];cpu_elapsed=[]
    max_error={};worst={};count=0
    def compare(label,x,y,index):
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError(f'{label} shape/nonfinite mismatch')
        d=np.abs(x.astype(np.float64)-y.astype(np.float64))
        m=float(d.max())
        if m>max_error.get(label,-1):
            max_error[label]=m;worst[label]={'raw_frame':index,'coordinate':list(np.unravel_index(int(d.argmax()),d.shape))}
    for i in range(n):
        begin=time.perf_counter()
        accepted=cpu.store_pico(bodies[i],sample_time=stamps[i],current_time=100+i*.02,frame_index=i)
        cpu_store=time.perf_counter()-begin
        accepted_gpu=gpu.store_pico(bodies[i],sample_time=stamps[i],current_time=100+i*.02,frame_index=i)
        assert accepted==accepted_gpu
        if i in (31,89,177):
            assert not cpu.store_pico(bodies[i],sample_time=stamps[i],current_time=999)
            assert not gpu.store_pico(bodies[i],sample_time=stamps[i],current_time=999)
        if i in reset_frames:
            cpu.reset_root_anchor();gpu.reset_root_anchor()
        assert cpu.has_future_sequence()==gpu.has_future_sequence()
        # Compare conversion/storage stages even before ring readiness.
        slot=(cpu._write_slot-1)%cpu.capacity
        compare('body_ring',cpu._body_ring[slot],gpu._body_ring.numpy()[slot],i)
        compare('root_quat_ring',cpu._root_quat_ring[slot],gpu._root_quat_ring.numpy()[slot],i)
        compare('foot_z',cpu._foot_z_ring[slot],gpu._foot_z_ring.numpy()[slot],i)
        if not cpu.has_future_sequence(): continue
        croot,ct,cbod=cpu.device_observation_sequence()
        groot,gt,gbod=gpu.device_observation_sequence()
        groot,gt,gbod=groot.numpy(),gt.numpy(),gbod.numpy()
        compare('root_sequence',croot,groot,i);compare('sample_times',ct,gt,i);compare('body_sequence',cbod,gbod,i)
        compare('current_root_rot',cpu.current_root_rot(),gpu.current_root_rot(),i)
        # Deterministic independent robot state, not reference values reused as robot state.
        rq=_axis_quat(np.array([.02*np.sin(i*.03),.04*np.cos(i*.01),.3*np.sin(i*.02)],dtype=np.float32))
        align=_axis_quat(np.array([0,0,.25*np.sin(i*.013)],dtype=np.float32))
        pos=defaults+np.float32(.05)*np.sin(np.arange(30,dtype=np.float32)+np.float32(i*.03))
        vel=np.float32(.07)*np.cos(np.arange(30,dtype=np.float32)+np.float32(i*.02))
        act=np.float32(.03)*np.sin(np.arange(30,dtype=np.float32)+np.float32(i*.01))
        av=np.array([.01,-.03,.02],dtype=np.float32)
        kwargs=dict(robot_root_quat_wxyz=rq,robot_root_angvel_local=av,robot_dof_pos=pos,
                    robot_dof_vel=vel,last_action=act,reference_yaw_alignment_wxyz=align)
        begin=time.perf_counter();co=cb.build(smpl_reference=cpu,**kwargs).copy();cpu_elapsed.append((time.perf_counter()-begin+cpu_store)*1000)
        go=gb.build(smpl_reference=gpu,**kwargs).array.numpy()
        compare('observation',co,go,i)
        # Repeated gather/build must not silently re-anchor or mutate the sequence.
        again=cb.build(smpl_reference=cpu,**kwargs).copy()
        assert np.array_equal(co,again)
        cpu_obs.append(co[0]);gpu_obs.append(go[0]);cpu_root.append(croot[0].copy());gpu_root.append(groot[0])
        cpu_body.append(cbod[0].copy());gpu_body.append(gbod[0]);sample_indices.append(i)
        positions.append(pos);velocities.append(vel);actions.append(act);robot_quats.append(rq);angvels.append(av);alignments.append(align)
        count+=1
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output.with_suffix('.npz'),body_poses=bodies,sample_times=stamps,raw_frame_indices=np.array(sample_indices),
        reset_raw_frames=np.array(sorted(reset_frames)),default_dof_pos=defaults,
        cpu_observations=np.array(cpu_obs),gpu_observations=np.array(gpu_obs),
        cpu_root=np.array(cpu_root),gpu_root=np.array(gpu_root),cpu_body=np.array(cpu_body),gpu_body=np.array(gpu_body),
        robot_dof_pos=np.array(positions),robot_dof_vel=np.array(velocities),last_action=np.array(actions),
        robot_root_quat=np.array(robot_quats),robot_angvel=np.array(angvels),yaw_alignment=np.array(alignments))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={'scope':'CPU preprocessing vs unchanged existing CUDA preprocessing; no action/model inference or robot IO',
        'input_kind':kind,'raw_frames':n,'observation_frames':count,'observations_shape':[count,3126],
        'max_abs_errors':max_error,'worst':worst,'duplicate_timestamps_rejected':True,
        'anchor_reset_raw_frames':sorted(reset_frames),'ring_wrap_exercised':n>124,
        'sample_times_relative_float32':True,'cpu_total_preprocess_ms_host_diagnostic':{'mean':float(np.mean(cpu_elapsed)),'max':float(max(cpu_elapsed))},
        'action_acceptance':'pending independent original-model vs candidate end-to-end action comparison; intermediate agreement alone is not acceptance',
        'artifact_npz_sha256':digest(output.with_suffix('.npz')),'skeleton_sha256':digest(a.skeleton),
        'gpu_oracle_source_sha256':{p.name:digest(p) for p in [root/'src/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/smpl_reference.py',root/'src/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/smpl_motion_observation.py']}}
    # numpy coordinates must be JSON ints.
    report['worst']={k:{'raw_frame':v['raw_frame'],'coordinate':[int(x) for x in v['coordinate']]} for k,v in worst.items()}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
