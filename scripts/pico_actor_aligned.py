#!/usr/bin/env python3
"""Selected-checkpoint Pico modalities operator. auto requests real motion."""
import argparse
from collections import deque
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('live','auto','check','state-check','shadow','replay'))
    parser.add_argument('--policy',choices=('pico_smpl','pico_teleop'),required=True)
    parser.add_argument('--backend',choices=('auto','sdk','ort'),default='auto')
    parser.add_argument('--confirm-motion',action='store_true')
    parser.add_argument('--auto-start',action='store_true')
    parser.add_argument('--manual-pico',action='store_true')
    parser.add_argument('--wait-pico',action='store_true')
    parser.add_argument('--local-start',action='store_true')
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--robot-id')
    parser.add_argument('--robot-config',type=Path)
    parser.add_argument('--human-height',type=float,default=1.70)
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--startup-timeout',type=float,default=30.)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--group',default='default')
    parser.add_argument('--record-observations',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--sdk-dir',type=Path)
    parser.add_argument('--pico-service',choices=('auto','external'),default='auto')
    args=parser.parse_args()
    if args.mode in ('check','shadow','replay','state-check'):
        from pico_aligned_modes import run_mode
        return run_mode(args,parser)
    if args.local_start:
        if args.robot_id is not None or args.robot_config is not None:
            parser.error('--local-start cannot combine explicit robot target/config')
        from deployment.vita_s100_aorta_30dof.actor_pico_local import load_local_start
        profile=load_local_start(args.root)
        args.robot_id=profile['robot_id'];args.robot_config=profile['robot_config']
        if args.seconds is None:args.seconds=profile['seconds']
    if args.seconds is None:args.seconds=0.
    if args.mode=='auto':args.confirm_motion=True
    if args.backend=='ort':parser.error('live requires the native SDK inference backend')
    if (not args.robot_id or args.robot_config is None or not args.robot_config.is_absolute()
            or not args.confirm_motion or not 0<=args.seconds<=3600
            or not 0<args.startup_timeout<=180 or not 1<=args.threads<=16
            or not 1.0<=args.human_height<=2.3):
        parser.error('Use the Git-style auto launcher or explicit --confirm-motion, robot/config and finite budgets')
    # Same production platform/backend gate as the original 445700 entry.
    sys.path.insert(0,str(ROOT/'scripts'))
    from bootstrap_pico_inputs_s100 import require_target,verify_inputs,sha
    require_target()
    root=args.root.resolve();manifest=verify_inputs(root)
    receipt=json.loads((root/'build/actor_s100_install.json').read_text())
    if (manifest['pico_inputs_release'].get('physical_workflow') is not True
            or receipt.get('payload_sha256')!=manifest['payload_sha256']
            or receipt.get('sdk_workflow_check',{}).get('status')!='PASS_SMPL_NO_ACTION_WORKFLOW_REPLAY'):
        raise RuntimeError('This independent physical package must pass its SDK no-action install check first')
    if (manifest['pico_inputs_release'].get('body_cmd_protocol_version')!=4
            or receipt.get('bodycmd4_codec_check',{}).get('status')!='PASS_NATIVE_BODYCMD4_SERIALIZATION'):
        raise RuntimeError('This native AB entry requires a successful SDK BodyCmd4 bytes-only self-check')
    if receipt.get('aligned_model_checks',{}).get(args.policy,{}).get('sdk',{}).get('status')!='PASS_ALIGNED_OFFLINE_CHECK':
        raise RuntimeError('Run the aligned package offline installation checks before live')
    from deployment.vita_s100_aorta_30dof.actor_pico_live import require_peer_profile
    profile=os.environ.get('ZENOH_SESSION_CONFIG_URI')
    if not profile:raise RuntimeError('Existing private ZENOH_SESSION_CONFIG_URI required')
    require_peer_profile(profile)
    from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources
    verify_vendor_sources()
    from deployment.vita_s100_aorta_30dof.sdkmain_adapter import verify_state_decoder,RealtimeBodyTransport
    verify_state_decoder()
    from deployment.vita_s100_aorta_30dof.pico_aligned_runtime import AlignedRuntime,run_aligned
    from deployment.vita_s100_aorta_30dof.actor_pico_live import load_robot_config,StatusGuard,TerminalCommands
    runtime=AlignedRuntime(root,args.policy,backend='sdk',threads=args.threads,human_height=args.human_height)
    load_robot_config(args.robot_config,runtime)  # Before any action/control/Pico handle.
    out=args.out or root/'outputs'/(args.policy+'_physical_'+str(time.time_ns()))
    out.mkdir(parents=True,exist_ok=False)
    stop=threading.Event();previous={s:signal.signal(s,lambda *_:stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    report=None;error=None;controller=None;trace=deque(maxlen=3000);rows=0
    import numpy as np
    try:
        with ExitStack() as resources:
            log=resources.enter_context((out/'control.jsonl').open('x',encoding='utf-8'))
            counts=dict(actions_sent=0,control_handles_created=0)
            lifecycle=dict(exit_confirmed=False,startup_fixed_stand_confirmed=False)
            def write(row):
                nonlocal rows
                rows+=1
                for k in counts:
                    if k in row:counts[k]=row[k]
                if row.get('record_type')=='external_exit' and row.get('status',{}).get('current_action')=='PASSIVE' and row.get('status',{}).get('phase')=='IDLE':lifecycle['exit_confirmed']=True
                if row.get('record_type')=='fixed_stand_confirmed':lifecycle['startup_fixed_stand_confirmed']=True
                if row.get('record_type')=='exit_failure':lifecycle['exit_error']=row.get('error')
                log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            def record(result,stamp):
                if args.record_observations:trace.append((stamp,result['observation'].copy(),result['report']['raw_action']))
            # Separate package operator lock; the Pico service also holds the
            # existing global per-port lease across all worktrees/operators.
            import fcntl
            lock_path=root/'build/operator.lock'
            try:
                inherited=os.fstat(9);target=lock_path.stat()
                descriptor=os.dup(9) if (inherited.st_dev,inherited.st_ino)==(target.st_dev,target.st_ino) else None
            except OSError:descriptor=None
            if descriptor is None:descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR,0o600)
            resources.callback(os.close,descriptor)
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Check current action with a publishers=False source before starting
            # Pico or constructing a controller. Never cancel an unknown action.
            from locomotion_aorta.pull import PullStateSource,PullController
            source=PullStateSource(group=args.group,node_name='teleop-physical-readonly-admission')
            try:
                source.get_robot_info(timeout=5.);source.wait_state(timeout=5.)
                StatusGuard(time,initial_actions=('PASSIVE','QUIET_PASSIVE','FIXED_STAND','')).check(source.control_status)
            finally:source.close()
            from deployment.vita_s100_aorta_30dof.pico_service import PicoService,default_sdk_directory,library_directories
            from deployment.vita_s100_aorta_30dof.pico_input_process import PicoInputProcess
            sdk=args.sdk_dir or default_sdk_directory(root)
            service=PicoService(root,out/'pico_service.log',mode=args.pico_service,sdk_dir=sdk,stopped=stop.is_set)
            resources.callback(service.close);service_info=service.start()
            receiver=PicoInputProcess(sdk_dir=sdk,library_dirs=library_directories(root,service_info['executable']))
            receiver.start();resources.callback(receiver.close)
            # Children start as ordinary-priority input workers; only this
            # original control process enters the reviewed scheduler policy.
            from deployment.vita_s100_aorta_30dof.actor_pico_scheduling import control_scheduler
            write(dict(record_type='control_scheduler',**control_scheduler()))
            from deployment.vita_s100_aorta_30dof.actor_pico_scheduling import control_affinity
            write(dict(record_type='control_affinity',cpus=control_affinity([2,3,4])))
            transport=RealtimeBodyTransport(group=args.group,node_name='vita-pico27000-git-aligned')
            resources.callback(transport.close)
            controller=PullController(group=args.group,node_name='vita-pico27000-git-aligned',_transport=transport)
            resources.callback(controller.close)
            write(dict(record_type='control_handle_created',actions_sent=0,control_handles_created=1))
            report=run_aligned(runtime,args,write,controller,receiver,stop,service_alive=service.check,on_result=record,poll_command=TerminalCommands(args.robot_id,stop).poll)
            report.update(status='COMPLETED_PHYSICAL_ALIGNED_SESSION',pico_service=service.report,
                operator_height_record_m=args.human_height,recorded_observations=len(trace))
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        for s,handler in previous.items():signal.signal(s,handler)
        if report is None:
            report=dict(status='FAILED_PHYSICAL_ALIGNED_SESSION',actions_sent=counts['actions_sent'] if 'counts' in locals() else 0,
                control_handles_created=counts['control_handles_created'] if 'counts' in locals() else 0,
                robot_configuration_modified=False,protection_relaxed=False,exit_confirmed=False)
            if 'lifecycle' in locals():report.update(lifecycle)
        if error:report.update(status='FAILED_PHYSICAL_ALIGNED_SESSION',error=error)
        if trace:
            np.savez_compressed(out/'trace.npz',time_ns=np.array([x[0] for x in trace],np.int64),
                observations=np.array([x[1] for x in trace],np.float32),raw_action=np.array([x[2] for x in trace],np.float32))
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if error:raise SystemExit(1)


if __name__=='__main__':main()
