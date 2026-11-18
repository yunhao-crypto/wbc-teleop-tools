#!/usr/bin/env python3
"""445700 lifecycle with 27000 SMPL inference; this release NEVER moves a robot."""
import argparse
from collections import deque
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','state-check','auto','live'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--no-action',action='store_true')
    parser.add_argument('--robot-id')
    parser.add_argument('--robot-config',type=Path)
    parser.add_argument('--human-height',type=float,default=1.70,
        help='Operator record only; raw SMPL XYZ has measured scale, no GMR height scaling')
    parser.add_argument('--backend',choices=('ort','sdk'),default='ort')
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--startup-timeout',type=float,default=30.)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--record-observations',action='store_true')
    parser.add_argument('--sdk-dir',type=Path)
    parser.add_argument('--pico-service',choices=('auto','external'),default='auto')
    parser.add_argument('--service-exe',type=Path)
    parser.add_argument('--service-arg',action='append',default=[])
    parser.add_argument('--service-port',type=int,default=63901)
    parser.add_argument('--group',default='default')
    args=parser.parse_args()
    # Refuse BEFORE any SDK/inference/service imports or network handles.
    if args.mode=='live' or (args.mode=='auto' and not args.no_action):
        parser.error('This release only supports auto --no-action; physical FIXED_STAND/EXTERNAL disabled')
    if (not 0<=args.seconds<=3600 or not 0<args.startup_timeout<=180 or not 1<=args.threads<=16
            or not 1.0<=args.human_height<=2.3 or not 1<=args.service_port<=65535): parser.error('Invalid duration/thread/height budget')
    if args.mode!='check' and (args.robot_config is None or not args.robot_id):
        parser.error('state-check/auto require --robot-id and absolute --robot-config')
    if args.robot_config is not None and not args.robot_config.is_absolute(): parser.error('Absolute robot config required')
    args.root=args.root.resolve()
    out=args.out or args.root/'outputs'/('smpl_workflow_'+args.mode+'_'+str(time.time_ns()))
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    if args.mode=='check':
        from deployment.vita_s100_aorta_30dof.pico_shadow_math import prepare
        prepare(args.root,'pico_smpl')
    from deployment.vita_s100_aorta_30dof.pico_smpl_workflow import (
        SmplWorkflow,robot_reply,replay_workflow,validate_snapshot_admission,prepare_guard_imports)
    prepare_guard_imports(args.root)
    from deployment.vita_s100_aorta_30dof.actor_pico_live import StatusGuard,Inputs,require_peer_profile
    from types import SimpleNamespace as NS
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    workflow=source=receiver=service=None; error=None; exit_error=None; admission=None
    trace=deque(maxlen=3000); fullbody=0; receiver_state=None; source_guard=None; meta=None; actual_baseline=None
    fixture=json.loads((args.root/'checks/workflow/fixture.json').read_text()) if args.mode=='check' else None
    report=dict(actions_sent=0,control_handles_created=0,robot_configuration_modified=False,protection_relaxed=False,
        physical_actions_enabled=False,robot_id=args.robot_id,operator_height_record_m=args.human_height,
        firmware_wire_protocol_verified=False,hardware_startup_exit_validated=False,closed_loop=False)
    try:
        with ExitStack() as resources:
            log=resources.enter_context((out/'workflow.jsonl').open('x',encoding='utf-8'))
            def write(row): log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            def cleanup_workflow():
                nonlocal exit_error
                if workflow is None or workflow.stage=='CLOSED':return
                try:workflow.close()
                except Exception as exc:exit_error=f'{type(exc).__name__}: {exc}'
            # Registered after the log: status/exit records remain writable on
            # every exception path. Errors never become a clean-exit claim.
            resources.callback(cleanup_workflow)
            if args.mode=='check':
                from deployment.vita_s100_aorta_30dof.actor_smpl import digest
                for item in fixture['files'].values():
                    if digest(args.root/item['path'])!=item['sha256']: raise ValueError('Workflow fixture hash mismatch')
                snapshot=json.loads((args.root/fixture['files']['robot_snapshot']['path']).read_text())
                reply=robot_reply(snapshot['robot_info'])
                config=args.root/fixture['files']['robot_config']['path']
                try: validate_snapshot_admission(snapshot['control_status'],time)
                except RuntimeError as exc: admission=dict(status='BLOCKED_AS_EXPECTED',reason=str(exc),action=snapshot['control_status']['current_action'])
                else: raise RuntimeError('Expected stored HUMANOID_TRACKING snapshot to be refused')
                workflow=SmplWorkflow(args.root,reply,config,backend=args.backend,threads=args.threads,write=write)
            else:
                # The source has publishers=False. No PullController or firmware
                # action interface is constructed, including during startup.
                profile=os.environ.get('ZENOH_SESSION_CONFIG_URI')
                if not profile: raise RuntimeError('Set existing private ZENOH_SESSION_CONFIG_URI; no profile is generated')
                require_peer_profile(Path(profile))
                from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources
                verify_vendor_sources()
                from locomotion_aorta.pull import PullStateSource
                source=PullStateSource(group=args.group,node_name='pico-smpl-workflow-readonly')
                resources.callback(source.close)
                reply=source.get_robot_info(timeout=5.)
                workflow=SmplWorkflow(args.root,reply,args.robot_config,backend=args.backend,threads=args.threads,clock=time,write=write)
                validator=Inputs(workflow.guard_runtime,source,None,time)
                source_guard=StatusGuard(time,initial_actions=('PASSIVE','QUIET_PASSIVE','FIXED_STAND',''))
                def readonly_feedback():
                    state=source.wait_state(timeout=.5)
                    validator.validate_state(state);workflow.envelope.check_measured_pose(state)
                    if state.runtime_id != reply.runtime_id:raise RuntimeError('Actual feedback/RobotInfo runtime changed')
                    from deployment.vita_s100_aorta_30dof.actor_smpl import digest
                    if digest(args.robot_config)!=workflow.config_receipt['sha256']:
                        raise RuntimeError('Actual robot configuration changed during read-only validation')
                    status=source_guard.check(source.control_status)
                    identity=tuple(getattr(status,k) for k in ('runtime_id','generation','reject_count','ingress_drop_count','last_reject_code'))
                    if status.timeout_hold or (actual_baseline is not None and identity!=actual_baseline):
                        raise RuntimeError('Actual read-only status changed; no action or cancellation sent')
                    return identity
                actual_baseline=readonly_feedback()
                admission=dict(status='PASS_READONLY_ADMISSION',current_action=source.control_status.current_action)
                report['hardware_inputs_checked']=True
                if args.mode=='state-check':
                    end=time.monotonic()+5.;samples=0
                    while time.monotonic()<end and not stopped: readonly_feedback();samples+=1
                    report.update(status='PASS_READONLY_SMPL_WORKFLOW_PREFLIGHT',sampled_fresh_frames=samples)
                    # No virtual or physical FIXED_STAND request on state-check.
                    workflow.close()
            def record(result,now):
                row=dict(time_ns=now,reference_mode=result['mode'],candidate_only=True,
                    observation=result['observation'].copy() if args.record_observations else None,
                    raw_action=result['report']['raw_action'])
                trace.append(row)
                write({k:v for k,v in row.items() if k not in ('observation','raw_action')})
            if args.mode=='check':
                workflow.start(args.startup_timeout)
                meta=replay_workflow(workflow,args.root/fixture['files']['input']['path'],
                    fixture['files']['input']['sha256'],fixture['synthetic_trigger_events'],on_result=record)
                if workflow.flow.track_ticks==0 or workflow.flow.losses==0:
                    raise RuntimeError('Workflow fixture did not exercise following and disconnection')
                workflow.close()
                report['status']='PASS_SMPL_NO_ACTION_WORKFLOW_REPLAY'
            elif args.mode=='auto':
                # Monitor real robot admission DURING virtual damping wait as
                # well. This proxy observes status but never sends any action.
                class MonitoredClock:
                    monotonic=staticmethod(time.monotonic)
                    monotonic_ns=staticmethod(time.monotonic_ns)
                    time_ns=staticmethod(time.time_ns)
                    @staticmethod
                    def sleep(seconds):
                        if stopped: workflow.stop.set();raise InterruptedError('Stopped')
                        readonly_feedback();time.sleep(seconds)
                workflow.clock=MonitoredClock()
                workflow.inputs.clock=workflow.guard.clock=workflow.driver.clock=workflow.clock
                print('NO_ACTION_WORKFLOW: FIXED_STAND / EXTERNAL are VIRTUAL. Robot will not stand or move.',flush=True)
                workflow.start(args.startup_timeout)
                from deployment.vita_s100_aorta_30dof.pico_service import PicoService,default_sdk_directory,library_directories
                from deployment.vita_s100_aorta_30dof.pico_input_process import PicoInputProcess
                sdk=args.sdk_dir or default_sdk_directory(args.root)
                service=PicoService(args.root,out/'pico_service.log',mode=args.pico_service,
                    executable=args.service_exe,arguments=args.service_arg,sdk_dir=sdk,stopped=lambda:stopped,
                    port=args.service_port)
                resources.callback(service.close);info=service.start()
                receiver=PicoInputProcess(sdk_dir=sdk,library_dirs=library_directories(args.root,info['executable']))
                receiver.start();resources.callback(receiver.close)
                print(f'PICO_SERVICE_READY: TCP {args.service_port}. First2s neutral; double triggers follow/pause. No robot commands.',flush=True)
                start=time.monotonic();deadline=time.monotonic_ns();next_print=start
                from deployment.vita_s100_aorta_30dof.pico_realtime import next_deadline_ns
                while not stopped and (args.seconds==0 or time.monotonic()-start<args.seconds):
                    # An early wake on a coarse platform clock must not execute
                    # a second policy step on the same timestamp or burst ticks.
                    if time.monotonic_ns()<deadline:
                        time.sleep(max(0.,(deadline-time.monotonic_ns())*1e-9));continue
                    readonly_feedback();service.check()
                    now=time.monotonic_ns();pose,buttons,receiver_state=receiver.poll(now)
                    if receiver_state.get('state')=='failed': raise RuntimeError(receiver_state['error'])
                    if pose is not None: fullbody+=1
                    result=workflow.candidate(pose=pose,buttons=buttons);record(result,now)
                    if time.monotonic()>=next_print:
                        print(json.dumps(dict(lifecycle_stage=workflow.stage,pico_frames=workflow.flow.frames,
                            inference_steps=workflow.flow.inferences,tracking_steps=workflow.flow.track_ticks,
                            actions_sent=0,control_handles_created=0,virtual_firmware=True)),flush=True)
                        next_print=time.monotonic()+1.
                    deadline=next_deadline_ns(deadline,time.monotonic_ns())
                    time.sleep(max(0.,(deadline-time.monotonic_ns())*1e-9))
                workflow.close()
                report['status']='PASS_DEVICE_NO_ACTION_WORKFLOW' if workflow.flow.calibration else 'STAND_INFERENCE_ONLY_NO_PICO_CALIBRATION'
    except Exception as exc: error=f'{type(exc).__name__}: {exc}'
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)
        if workflow is not None: report.update(workflow.summary())
        report.update(mode=args.mode,robot_admission=admission,input_metadata=meta,
            feedback_source='synthetic_neutral_not_robot_dynamics',error=error,exit_error=exit_error,
            receiver_state=receiver_state,received_live_body_packets=fullbody,
            current_pico_tracking_verified=bool(args.mode=='auto' and workflow is not None and workflow.flow.track_ticks>0
                and receiver_state is not None and not receiver_state.get('binding_test_fixture',False)),
            pico_service=service.report if service is not None else None,
            receiver_cleanup=receiver.close_report if receiver is not None else None)
        if error or exit_error: report['status']='FAILED_NO_ACTION_WORKFLOW'
        if trace:
            arrays=dict(time_ns=np.array([r['time_ns'] for r in trace],np.int64),
                raw_action=np.array([r['raw_action'] for r in trace],np.float32),mode=np.array([r['reference_mode'] for r in trace]))
            if args.record_observations: arrays['observations']=np.array([r['observation'] for r in trace],np.float32)
            np.savez_compressed(out/'trace.npz',**arrays)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    if error or exit_error: raise SystemExit(1)


if __name__=='__main__':main()
