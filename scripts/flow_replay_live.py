#!/usr/bin/env python3
"""Replay captured raw LowState/Pico/controller through onboard flow, inert only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from flow_common import digest, stats, transition_record
from flow_board_replay import _ForbidControl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--raw-state',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preprocess-library',type=Path)
    p.add_argument('--state-library',type=Path)
    p.add_argument('--velocity-backend',choices=['ort','hbm'],default='ort')
    p.add_argument('--warmup-frames',type=int,default=90)
    p.add_argument('--no-cadence',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.root/'src'))
    # Generated LowState refers to aorta.sys.AortaHeader; expose data modules
    # without executing aorta/__init__.py or creating any transport capability.
    import importlib.machinery, types
    schema_spec=importlib.machinery.PathFinder.find_spec('aorta',sys.path)
    if schema_spec is None or not schema_spec.submodule_search_locations:
        raise RuntimeError('Missing generated aorta data schema')
    schema_namespace=types.ModuleType('aorta')
    schema_namespace.__path__=list(schema_spec.submodule_search_locations)
    sys.modules['aorta']=schema_namespace
    sys.meta_path.insert(0,_ForbidControl())
    if a.preprocess_library:os.environ['VITA_PREPROCESS_LIBRARY']=str(a.preprocess_library.resolve())
    if a.state_library:os.environ['VITA_STATE_LIBRARY']=str(a.state_library.resolve())
    from deployment.common.controller_input import Button,ControllerSample,VelocityLimits
    from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState,VITA_CONTROLLER_JOINT_NAMES
    from deployment.vita_s100_aorta_30dof.live_shadow import live_alignment_quaternion
    from deployment.vita_s100_aorta_30dof.onboard_state import decode_policy_lowstate_buffer
    from deployment.vita_s100_aorta_30dof.onboard_backend import NativeBackend,NativeMotionRunner,NativeVelocityRunner
    from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.onboard_shadow import OnboardShadowFlow
    rows=[json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    manifest=json.loads((a.oracle/'summary.json').read_text())
    alignment_path=a.root/'src/deployment/vita_s100_aorta_30dof/config/vita02a_imu_alignment.json'
    for path,key in [(a.input,'input_sha256'),(a.raw_state,'raw_lowstate_sha256'),
                     (alignment_path,'alignment_sha256'),(a.oracle/'original_gpu_expected.npz','original_expected_sha256')]:
        if digest(path)!=manifest[key]:raise ValueError(f'Oracle provenance mismatch: {key}')
    if manifest['semantic_mismatches']:raise ValueError('Oracle semantics disagree with capture')
    alignment=live_alignment_quaternion(json.loads(alignment_path.read_text()))
    with np.load(a.oracle/'original_gpu_expected.npz',allow_pickle=False) as archive:
        expected={key:archive[key] for key in archive.files}
    raw=a.raw_state.read_bytes();payloads=[];offset=0
    while offset<len(raw):
        if offset+4>len(raw):raise ValueError('Truncated raw-state record size')
        size=int.from_bytes(raw[offset:offset+4],'little');offset+=4
        if not size or offset+size>len(raw):raise ValueError('Truncated raw-state record')
        payloads.append(raw[offset:offset+size]);offset+=size
    count=len(rows)
    if not count or len(payloads)!=count or len(expected['active'])!=count:raise ValueError('Frame count mismatch')
    if not 0<=a.warmup_frames<=count:raise ValueError('Invalid warmup count')
    picos=[]
    for row in rows:
        pico=row['pico']
        picos.append(None if pico is None else (np.asarray(pico[0],np.float32),*pico[1:]))
    def arguments(i):
        row=rows[i]
        values=decode_policy_lowstate_buffer(payloads[i],base_from_hip_quaternion_wxyz=alignment,
            received_monotonic_ns=row['received_monotonic_ns'])
        state=VitaRobotState(VITA_CONTROLLER_JOINT_NAMES,values.robot_dof_pos,values.robot_dof_vel,
            values.robot_root_quat_wxyz,values.robot_root_angvel_local,state_id=values.state_id,
            received_monotonic_ns=values.received_monotonic_ns)
        c=row['controller']
        controller=ControllerSample(sequence=c['sequence'],monotonic_time=c['monotonic_time'],
            buttons=frozenset(Button(name) for name in c['buttons']),axes=c['axes'])
        return dict(state=state,controller=controller,pico=picos[i],wall_time=row['tick_wall_time'],robot_ready=True)
    actions=np.zeros((count,30),np.float32);observations=np.zeros((count,3126),np.float32)
    targets=np.zeros((count,30),np.float32);cycle=np.zeros(count);decode_ms=np.zeros(count)
    starts=np.zeros(count,np.int64);checks=np.zeros(count);slip=np.zeros(count)
    errors=[];semantic=[];records=[];worst=dict(error=0.,frame=None,coordinate=None)
    obs_error=0.;target_error=0.
    backend=NativeBackend(a.root/'build/libvita_backend.so',a.root/'outputs/model_runtime_manifest.json',velocity_backend=a.velocity_backend)
    try:
        runtime=OnboardDualPolicyRuntime(contracts_path=a.root/'artifacts/policy_contracts.json',
            skeleton_path=a.root/'artifacts/preprocessing_skeleton.npz',
            motion_runner=NativeMotionRunner(backend),velocity_runner=NativeVelocityRunner(backend))
        limits=VelocityLimits(forward=.90,backward=.70,lateral=0.,yaw=.80)
        flow=OnboardShadowFlow(runtime,limits=limits)
        for i in range(a.warmup_frames):flow.tick(**arguments(i))
        runtime.reset();flow=OnboardShadowFlow(runtime,limits=limits)
        begin=time.monotonic_ns();deadline=begin;period=20_000_000
        for i in range(count):
            if not a.no_cadence:
                wait=deadline-time.monotonic_ns()
                if wait>0:time.sleep(wait/1e9)
            starts[i]=time.monotonic_ns();slip[i]=max(0,starts[i]-(begin+i*period))/1e6
            args=arguments(i);decoded=time.monotonic_ns();decode_ms[i]=(decoded-starts[i])/1e6
            step,info=flow.tick(**args);ended=time.monotonic_ns();cycle[i]=(ended-starts[i])/1e6
            fields=[];transition=transition_record(info['transition']);native_state=backend.state()
            for key in ('mode','reference_ready','pico_stored'):
                if info[key]!=rows[i][key]:fields.append(key)
            expected_transition=rows[i].get('transition')
            comparable=None if transition is None else {k:v for k,v in transition.items() if k!='changed'}
            if comparable!=expected_transition:fields.append('transition')
            if bool(step is not None)!=bool(expected['active'][i]):fields.append('action_presence')
            if native_state['motion_step']!=int(expected['motion_step_next'][i]):fields.append('motion_step_next')
            frame_error=None
            if step is not None:
                raw_action=np.asarray(runtime.last_raw_action).reshape(-1)
                obs=np.asarray(runtime.last_observation).reshape(-1)
                width=int(expected['action_widths'][i]);ow=int(expected['observation_widths'][i])
                if raw_action.size!=width or obs.size!=ow:fields.append('shape')
                else:
                    if not np.isfinite(raw_action).all() or not np.isfinite(obs).all():raise ValueError(f'Nonfinite output at {i}')
                    actions[i,:width]=raw_action;observations[i,:ow]=obs;targets[i]=step.target_joint_pos
                    delta=np.abs(raw_action.astype(np.float64)-expected['raw_actions'][i,:width].astype(np.float64))
                    frame_error=float(delta.max());bad=np.flatnonzero(delta>1e-4)
                    if frame_error>worst['error']:worst=dict(error=frame_error,frame=i,coordinate=int(delta.argmax()),mode=info['mode'])
                    if len(bad):errors.append(dict(frame=i,coordinates=bad.tolist(),max_abs=frame_error))
                    obs_error=max(obs_error,float(np.max(np.abs(obs.astype(np.float64)-expected['observations'][i,:ow].astype(np.float64)))))
                    target_error=max(target_error,float(np.max(np.abs(step.target_joint_pos.astype(np.float64)-expected['target_joint_pos'][i].astype(np.float64)))))
            if fields:semantic.append(dict(frame=i,fields=fields))
            records.append(dict(frame=i,mode=info['mode'],transition=transition,reference_ready=info['reference_ready'],
                pico_stored=info['pico_stored'],timing_ms=info['timing_ms'],full_cycle_ms=cycle[i],
                state_controller_reconstruct_ms=decode_ms[i],action_max_abs_error=frame_error,motion_step_next=native_state['motion_step'],action_sent=False))
            checks[i]=(time.monotonic_ns()-ended)/1e6
            if not a.no_cadence:deadline=max(deadline+period,time.monotonic_ns())
        wall=(time.monotonic_ns()-begin)/1e9;final_state=backend.state()
    finally:backend.close()
    np.savez(a.output/'candidate.npz',raw_actions=actions,observations=observations,target_joint_pos=targets,
        full_cycle_ms=cycle,state_controller_reconstruct_ms=decode_ms,start_monotonic_ns=starts,validation_staging_ms=checks,
        cumulative_schedule_slip_ms=slip)
    (a.output/'frames.json').write_text(json.dumps(records,indent=2)+'\n')
    report=dict(passed=not errors and not semantic and target_error<=1e-4,frames=count,active_frames=int(expected['active'].sum()),
        worst_action_error=worst,action_bad_steps=len(errors),action_bad_coordinates=sum(len(r['coordinates']) for r in errors),
        action_failures=errors,semantic_mismatches=semantic,max_observation_absolute_error_diagnostic=obs_error,
        max_mapped_target_absolute_error=target_error,full_cycle_ms=stats(cycle),state_controller_reconstruct_ms=stats(decode_ms),
        by_mode_ms={mode:stats(cycle[[r['mode']==mode for r in records]]) for mode in sorted({r['mode'] for r in records})},
        cycles_over_20ms=int((cycle>20).sum()),validation_staging_ms=stats(checks),elapsed_wall_seconds=wall,
        requested_pacing_hz=None if a.no_cadence else 50,actual_start_rate_hz=float((count-1)*1e9/(starts[-1]-starts[0])) if count>1 else None,
        interarrival_ms=stats(np.diff(starts)/1e6),interarrival_over_20ms=int((np.diff(starts)>period).sum()),
        interarrival_over_21ms=int((np.diff(starts)>period+1_000_000).sum()),max_cumulative_schedule_slip_ms=float(slip.max()),
        native_final_state=final_state,velocity_backend=a.velocity_backend,warmup_frames=a.warmup_frames,
        input_sha256=digest(a.input),raw_state_sha256=digest(a.raw_state),oracle_manifest_sha256=digest(a.oracle/'summary.json'),
        preprocess_library_sha256=digest(a.preprocess_library) if a.preprocess_library else None,
        state_library_sha256=digest(a.state_library) if a.state_library else None,
        backend_library_sha256=digest(a.root/'build/libvita_backend.so'),
        runtime_manifest_sha256=digest(a.root/'outputs/model_runtime_manifest.json'),
        source_sha256={str(path.relative_to(a.root)):digest(path) for path in [
            Path(__file__),a.root/'src/deployment/vita_s100_aorta_30dof/onboard_state.py',
            a.root/'src/deployment/vita_s100_aorta_30dof/onboard_runtime.py',
            a.root/'src/deployment/vita_s100_aorta_30dof/onboard_shadow.py',
            a.root/'src/deployment/vita_s100_aorta_30dof/onboard_preprocessing.py']},
        timing_boundary='Raw LowState decode, alignment, controller reconstruction, complete Pico/FSM/policy/mapping; excludes actual SDK transport, startup, and postcycle validation/staging. Validation affects measured interarrival cadence.',
        state_semantics='Candidate owns KV/lastactions/history/reference; no expected observations/actions/cache injected. Recorded Pico sample/receipt and tick wall times preserve freshness.',
        provenance='Real recorded robot LowState and host synthetic Pico delivered via nativeSDK during capture; this replay performs no live transport or actuation.',action_sent=False,robot_sdk_imported=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='action_failures'}),flush=True)
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
