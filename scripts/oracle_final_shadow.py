#!/usr/bin/env python3
"""Replay captured live shadow input through original GPU preprocessing/policy.

Consumes saved raw LowState payloads, Pico frames (including receipt time), and
controller samples only. Original history/last_action/KV advance independently;
board actions, observations and targets are comparison data, never oracle input.
"""
from __future__ import annotations
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
from oracle_policy_replay import digest

class ForbidRobotIO(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*_):
        if fullname=='aorta.sys' or fullname.startswith('aorta.sys.'):return None
        blocked=('aorta','locomotion_aorta','xrobotoolkit_sdk','deployment.vita_s100_aorta_30dof.aorta_adapter','deployment.vita_s100_aorta_30dof.real_teleop','deployment.vita_s100_aorta_30dof.cli')
        if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
            raise RuntimeError('Offline oracle forbids robot/input transport import: '+fullname)
        return None

def read_payloads(path):
    with path.open('rb') as f:
        while True:
            header=f.read(4)
            if not header:return
            if len(header)!=4:raise ValueError('Truncated LowState length')
            length=int.from_bytes(header,'little')
            if not 1<=length<=1024*1024:raise ValueError('Invalid LowState payload length')
            payload=f.read(length)
            if len(payload)!=length:raise ValueError('Truncated LowState payload')
            yield payload

def transition_record(t):
    if t is None:return None
    return dict(previous=t.previous.value,current=t.current.value,event=t.event.value,accepted=bool(t.accepted),changed=bool(t.changed),reason=t.reason)

def compare_vector(actual,reference,*,threshold=1e-4):
    x=np.asarray(actual,dtype=np.float32).reshape(-1);y=np.asarray(reference,dtype=np.float32).reshape(-1)
    if x.shape!=y.shape:raise ValueError(f'Comparison shape mismatch {x.shape}/{y.shape}')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite comparison')
    d=abs(x.astype(np.float64)-y);i=int(d.argmax())
    return dict(max_abs=float(d[i]),worst_coordinate=i,actual=float(x[i]),reference=float(y[i]),coordinates_exceeding_threshold=int((d>threshold).sum()),coordinates=len(d))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--lowstate',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--sdk-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--original-source',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True)
    sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(a.sdk_source));sys.meta_path.insert(0,ForbidRobotIO())
    # Generated LowState imports aorta.sys.AortaHeader. Supply only its data-schema
    # namespace, never execute aorta/__init__.py or load the Node/native transport.
    schema_namespace=types.ModuleType('aorta');schema_namespace.__path__=[str(a.sdk_source/'aorta')];sys.modules['aorta']=schema_namespace
    from deployment.common.controller_input import Button,ControllerSample,VelocityLimits
    from deployment.common.model_contract import load_model_bundle
    from deployment.vita_s100_aorta_30dof.runtime import VitaDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.onboard_shadow import OnboardShadowFlow
    from deployment.vita_s100_aorta_30dof.state_adapter import decode_humanoid_lowstate_buffer
    from deployment.vita_s100_aorta_30dof.live_shadow import live_alignment_quaternion,policy_state_from_lowstate
    from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState,VITA_CONTROLLER_JOINT_NAMES
    alignment_path=root/'src/deployment/vita_s100_aorta_30dof/config/vita02a_imu_alignment.json'
    alignment=live_alignment_quaternion(json.loads(alignment_path.read_text()))
    rows=[json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    rawfile=a.lowstate or a.input.with_suffix('.lowstate.bin');payloads=list(read_payloads(rawfile))
    if not rows or len(payloads)!=len(rows):raise ValueError('Raw payload and tick counts differ or empty')
    # First independently reconstruct every policy state from the raw payload.
    states=[];state_errors=[];initial_q=None
    for i,(row,payload) in enumerate(zip(rows,payloads)):
        if row['index']!=i or row.get('action_sent') is not False:raise ValueError('Noncontiguous or non-shadow trace')
        decoded=decode_humanoid_lowstate_buffer(payload,received_monotonic_ns=int(row['received_monotonic_ns']))
        if initial_q is None:initial_q=np.asarray(decoded.q_by_slot[:42],np.float64)
        v=policy_state_from_lowstate(decoded,base_from_hip_quaternion_wxyz=alignment)
        state=VitaRobotState(VITA_CONTROLLER_JOINT_NAMES,v.robot_dof_pos,v.robot_dof_vel,v.robot_root_quat_wxyz,v.robot_root_angvel_local,state_id=v.state_id,received_monotonic_ns=int(row['received_monotonic_ns']),attitude_valid=decoded.attitude_valid)
        if int(state.state_id)!=int(row['state_id']):raise ValueError(f'LowState ID mismatch frame{i}')
        for name in ('joint_pos','joint_vel','root_quat_wxyz','root_ang_vel_b'):
            c=compare_vector(getattr(state,name),row['state'][name],threshold=0)
            if c['max_abs']!=0:state_errors.append(dict(frame=i,field=name,**c))
        states.append(state)
    if state_errors:
        (a.output/'raw_state_differences.json').write_text(json.dumps(state_errors,indent=2))
    bundle=load_model_bundle(a.bundle)
    expected_hashes={'motion':'881d936f560e22e81302483b394cd6a89ba5648e6c2893fa153172dec38033a7','velocity':'f78fd39eb3f420675af63782d12ce2e6434f62e3159e1fd5b17b65b4fd6d7126'}
    for name,expected in expected_hashes.items():
        if bundle.artifacts[name].sha256!=expected:raise ValueError('Unexpected original model identity')
    os.environ['HOLOMOTION_SMPL_MODEL_PATH']=str(bundle.artifacts['smpl'].path)
    runtime=VitaDualPolicyRuntime.from_bundle(bundle,require_trained=True,providers=[('CUDAExecutionProvider',{'device_id':0,'use_tf32':0}),'CPUExecutionProvider'],require_cuda=True,velocity_providers=['CPUExecutionProvider'],velocity_require_cuda=False)
    assert runtime.motion_runner.session.get_provider_options()['CUDAExecutionProvider']['use_tf32']=='0'
    from oracle_final_output import OriginalOutputAudit
    output_audit=OriginalOutputAudit(runtime,initial_q,original_source=a.original_source,input_file=a.input,rows=rows,output=a.output)
    flow=OnboardShadowFlow(runtime,limits=VelocityLimits(forward=.90,backward=.70,lateral=0.,yaw=.80))
    count=len(rows);actions=np.zeros((count,30),np.float32);observations=np.zeros((count,3126),np.float32);targets=np.zeros((count,30),np.float32);active=np.zeros(count,np.uint8);widths=np.zeros(count,np.int32);obswidths=np.zeros(count,np.int32);steps=np.zeros(count,np.int64);comparisons=[];semantics=[];begin=time.monotonic()
    with (a.output/'comparison.jsonl').open('w') as out:
        for i,(row,state) in enumerate(zip(rows,states)):
            c=row['controller'];control=ControllerSample(sequence=int(c['sequence']),monotonic_time=float(c['monotonic_time']),buttons=frozenset(Button(x) for x in c['buttons']),axes=c.get('axes',{}))
            raw_pico=row.get('pico');pico=None
            if raw_pico is not None:
                if len(raw_pico) not in (3,4):raise ValueError('Pico tuple must have3 or4 components')
                pico=(np.asarray(raw_pico[0],np.float32),float(raw_pico[1]),int(raw_pico[2]))
                if len(raw_pico)==4:pico+= (float(raw_pico[3]),)
            step,info=flow.tick(state=state,controller=control,pico=pico,wall_time=float(row['tick_wall_time']),robot_ready=bool(row.get('robot_ready',True)))
            output_audit.tick(i,step,row)
            steps[i]=int(runtime.motion_runner.step_index[0]);record=dict(frame=i,mode=info['mode'],motion_step_next=int(steps[i]),action_sent=False)
            for key,value in [('mode',info['mode']),('reference_ready',info['reference_ready']),('pico_stored',info['pico_stored']),('transition',transition_record(info['transition']))]:
                board_value=row.get(key)
                if key=='transition' and board_value is not None:
                    board_value=dict(board_value);board_value.setdefault('changed',bool(board_value['previous']!=board_value['current']))
                if board_value!=value:semantics.append(dict(frame=i,field=key,board=board_value,original=value))
            if (step is not None)!=('raw_action' in row):semantics.append(dict(frame=i,field='active_output',board='raw_action' in row,original=step is not None))
            if step is not None:
                raw=np.asarray(runtime.last_raw_action,np.float32).reshape(-1);obs=np.asarray(runtime.last_observation,np.float32).reshape(-1);target=np.asarray(step.target_joint_pos,np.float32).reshape(-1)
                active[i]=1;widths[i]=len(raw);obswidths[i]=len(obs);actions[i,:len(raw)]=raw;observations[i,:len(obs)]=obs;targets[i]=target
                if 'raw_action' in row:
                    record.update(raw_action=compare_vector(row['raw_action'],raw),observation=compare_vector(row['observation'],obs),target_joint_pos=compare_vector(row['target_joint_pos'],target))
            comparisons.append(record);out.write(json.dumps(record,separators=(',',':'))+'\n')
            if (i+1)%100==0:print(json.dumps(dict(completed=i+1,total=count,mode=info['mode'])),flush=True)
    np.savez(a.output/'original_gpu_expected.npz',active=active,action_widths=widths,observation_widths=obswidths,raw_actions=actions,observations=observations,target_joint_pos=targets,motion_step_next=steps)
    def aggregate(field,mode=None):
        selected=[x for x in comparisons if field in x and (mode is None or x['mode']==mode)]
        if not selected:return None
        worst=max(selected,key=lambda x:x[field]['max_abs']);return dict(frames=len(selected),coordinates=sum(x[field]['coordinates'] for x in selected),max_abs=worst[field]['max_abs'],worst_frame=worst['frame'],worst_coordinate=worst[field]['worst_coordinate'],board_value=worst[field]['actual'],original_value=worst[field]['reference'],coordinates_exceeding_1e4=sum(x[field]['coordinates_exceeding_threshold'] for x in selected))
    output_report=output_audit.finish()
    rawsummary=aggregate('raw_action');targetsummary=aggregate('target_joint_pos');report=dict(kind='Actual live shadow capture replay through original GPU preprocessing and full FP32 policies; no robot I/O',frames=count,active_frames=int(active.sum()),input=str(a.input),input_sha256=digest(a.input),raw_lowstate_file=str(rawfile),raw_lowstate_sha256=digest(rawfile),raw_state_reconstruction=dict(method='IndependentrawFlatBuffers+alignment onoriginalhost; every field compared withboard capture',differing_field_samples=len(state_errors),max_abs=max((x['max_abs'] for x in state_errors),default=0.0),difference_file='raw_state_differences.json' if state_errors else None),alignment_sha256=digest(alignment_path),models=expected_hashes,motion_provider=runtime.motion_runner.session.get_providers(),motion_use_tf32=False,velocity_provider=runtime.velocity_runner.session.get_providers(),policy_hz=50,raw_action=rawsummary,raw_action_by_mode={mode:aggregate('raw_action',mode) for mode in sorted(set(x['mode'] for x in comparisons))},observation=aggregate('observation'),target_joint_pos=targetsummary,semantic_mismatches=semantics,passed=bool(rawsummary and rawsummary['coordinates_exceeding_1e4']==0 and not semantics),state_semantics='Original reference owns Velocityhistory/gait, bothlastactions, MotionKV/step, referencewindow/anchor; no boardcomputedobservation/action/cache injected',pico_clock='Captured sampletime and optionalreceiptwalltime passedverbatim; tickwalltime replayed forfreshness; no crossdeviceclocklatencyclaim',input_provenance='Real robot LowState plus host-generated syntheticPico through actualnativeSDK, controller source fromcapture summary; not realPico hardware or closed-loop robotactuation',original_expected_sha256=digest(a.output/'original_gpu_expected.npz'),comparison_sha256=digest(a.output/'comparison.jsonl'),wall_seconds=time.monotonic()-begin,action_sent=False,robot_io_imported=False,source_sha256={str(x.relative_to(root)):digest(x) for x in [root/'src/deployment/vita_s100_aorta_30dof/runtime.py',root/'src/deployment/vita_s100_aorta_30dof/onboard_shadow.py',root/'src/deployment/vita_s100_aorta_30dof/state_adapter.py',root/'src/deployment/common/policy_runtime.py']})
    report['output_audit']=output_report
    report['passed']=report['passed'] and output_report['passed']
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
