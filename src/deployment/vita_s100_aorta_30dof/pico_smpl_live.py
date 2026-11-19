"""Actual SDK session/Body31 output for the explicitly selected 27000 SMPL.

I/O is injected only for protocol tests. Production entry supplies the original
PullController/RealtimeBodyTransport and firmware CLI, never a virtual driver.
Robot configuration, native ankle adapter and all original guards are read-only.
"""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

import numpy as np

from .actor_smpl import SmplReference, digest
from .pico_realtime import PicoFlow, Feedback
from .actor_pico_live import (load_robot_config,CommandEnvelope,Inputs,StatusGuard,phase,status_row,
    StaleStateError,MAX_STALE_TICKS,BODY_FIELDS_ALL,next_policy_deadline,record_period_overrun)
from .actor_pico_startup import StartupPlan,fixed_stand_startup,cancel_startup,FirmwareActions


def now_ns(clock):
    return clock.monotonic_ns() if hasattr(clock,'monotonic_ns') else round(clock.monotonic()*1e9)


class SmplLiveRuntime:
    def __init__(self,root,*,backend='sdk',threads=1):
        self.root=Path(root).resolve()
        self.settings=json.loads((self.root/'config/pico_smpl_live.json').read_text())
        if (self.settings.get('schema')!='vita_pico_smpl_physical_workflow_v1'
                or self.settings.get('route')!='pico_smpl'
                or self.settings.get('policy_step')!=json.loads((self.root/'config/actor_smpl.json').read_text())['checkpoint_iteration']
                or self.settings.get('entry_policy')!='policy-online'
                or self.settings.get('body_cmd_protocol_version')!=4
                or self.settings.get('body_cmd_coordinate_contract')!='native_motor_ab_ankles_virtual_pr_feedback'
                or self.settings.get('stand_preparation_seconds')!=3.):
            raise ValueError('Unsupported physical SMPL workflow contract')
        for path,expected in self.settings['bindings'].items():
            if digest(self.root/path)!=expected:raise ValueError('Physical SMPL binding changed: '+path)
        self.flow=PicoFlow(self.root,'pico_smpl',backend=backend,threads=threads,wrist_mode='stand')
        self.runtime=self.flow.runtime
        self.model,self.policy,self.math=self.runtime.model,self.runtime.policy,self.runtime.math
        self.config={**self.runtime.config,'pico_max_age_s':.6};self.head=None
        from locomotion_aorta.body_types import BODY_CMD_MOTOR_THETA_PROTOCOL_VERSION
        from .actor_parallel_ankle import command_space
        if BODY_CMD_MOTOR_THETA_PROTOCOL_VERSION!=4 or command_space(self.policy)!='motor_theta':
            raise ValueError('This physical entry requires SDK BodyCmd4 and native motor AB targets')
        self.command_protocol_version=BODY_CMD_MOTOR_THETA_PROTOCOL_VERSION
        spec=json.loads((self.root/'config/pico_smpl_workflow.json').read_text())
        for path,expected in spec['unchanged_guards'].items():
            if digest(self.root/path)!=expected:raise ValueError('Original physical guard changed: '+path)
        item=spec['standing_reference'];path=(self.root/item['path']).resolve()
        if not path.is_relative_to(self.root) or digest(path)!=item['sha256']:raise ValueError('Standing reference mismatch')
        self.asset=SmplReference.from_npz(self.math,path);self.standing=None

    def align(self,root):
        m=self.math;offset=m.yaw_quaternion(m.yaw(root)-m.yaw(self.asset.quat[0]))
        self.standing=SmplReference(m,self.asset.joints@m.rotation(offset).T,
            np.array([m.multiply(offset,q) for q in self.asset.quat]),self.asset.wrists.copy())
        self.flow.standing=self.standing

    def push(self,pose,feedback):
        try:self.flow.push(*pose,feedback)
        finally:self.flow.standing=self.standing


def run_live_smpl(runtime,args,write,controller,receiver,stop,*,service_alive=lambda:None,clock=time,
                  request_action=None,on_result=lambda result,stamp:None):
    """The same runner is used by production and actual-SDK loopback tests.

    No fallback model, fake state, output clamp, target ramp or firmware config
    patch is permitted. A failed guard cancels the owned SDK session once.
    """
    flow=runtime.flow
    params,receipt=load_robot_config(args.robot_config,runtime)
    reply=controller.get_robot_info(args.startup_timeout)
    envelope=CommandEnvelope(runtime,reply,params)
    plan=StartupPlan(runtime,envelope,args.robot_config)
    inputs=Inputs(runtime,controller,None,clock)
    guard=StatusGuard(clock,initial_actions=('PASSIVE','QUIET_PASSIVE','FIXED_STAND',''))
    entered=None;attempted=False;fixed_ready=False;startup_generation=None;sent_count=0;event_index=0
    report=dict(actions_sent=0,control_handles_created=1,policy_step=runtime.settings['policy_step'],route='pico_smpl',
        robot_configuration_modified=False,protection_relaxed=False,exit_confirmed=False,
        startup_fixed_stand_confirmed=False,onnx_control_enabled=False,tracking_steps=0,
        physical_backend='original_sdk_pull_controller',feedback_source='actual_body31',
        robot_config=receipt,standing_reference_physical_stability_validated=False)

    def emit(kind,**details):write(dict(record_type=kind,actions_sent=sent_count,control_handles_created=1,**details))
    def config_unchanged():
        if digest(args.robot_config)!=receipt['sha256']:raise RuntimeError('Effective config changed; no configuration is written')
    def read():
        state,q,dq,root,gyro=inputs.read(.1)
        if state.runtime_id!=reply.runtime_id:raise RuntimeError('BodyState/RobotInfo runtime_id mismatch')
        envelope.check_measured_pose(state)
        actual=controller.status
        try:status=guard.check(actual,entered)
        except Exception as exc:
            fields=('runtime_id','client_id','control_id','generation','last_reject_code','reject_count',
                    'ingress_drop_count','last_accepted_cmd_id','timeout_hold')
            emit('control_guard_failure',error=str(exc),status=dict(status_row(actual) or {},
                **{key:getattr(actual,key,None) for key in fields}))
            raise
        if status.runtime_id!=reply.runtime_id:raise RuntimeError('ControlStatus/RobotInfo runtime_id mismatch')
        config_unchanged()
        return state,Feedback(q,dq,root,gyro)
    def service_check():
        result=service_alive()
        if result is False:raise RuntimeError('Pico service exited')
    def flush_events():
        nonlocal event_index
        for row in flow.events[event_index:]:write(dict(row,actions_sent=sent_count,control_handles_created=1))
        event_index=len(flow.events)
    def cancel_monitor():
        # FirmwareActions only monitors read-only fresh input during CANCEL.
        return inputs.read(.1)
    cancel=request_action or FirmwareActions(stop,clock,cancel_monitor)

    try:
        service_check();state,feedback=read();runtime.align(feedback.root)
        emit('physical_start_review',robot_id=args.robot_id,robot_config=receipt,startup_plan=plan.review,
            model=runtime.runtime.config['artifacts']['model'],human_reference='pinned_SMPL_neutral',
            wrist_mode='explicit_contract_stand',stand_preparation_seconds=3.)
        emit('body_command_contract',protocol_version=runtime.command_protocol_version,
            ankle_targets='motor_AB',feedback='virtual_pitch_roll')
        print('SMPL_ACTOR_START: requesting actual FIXED_STAND and EXTERNAL; 27000 SMPL, live Body31 feedback.',flush=True)
        runtime.runtime.infer(runtime.standing,0.,q=runtime.model.contract.default,dq=np.zeros(28),
            root=feedback.root,gyro=np.zeros(3))
        runtime.policy.reset()  # Warm inference without output, before firmware startup.
        config_unchanged()
        if not fixed_stand_startup(plan,envelope,runtime,args,controller,inputs,stop,
                lambda row:write(dict(row,control_handles_created=1)),clock,request_action=request_action):
            report['stop_reason']='stopped_before_fixed_stand';return report
        fixed_ready=True;startup_generation=controller.status.generation
        report['startup_fixed_stand_confirmed']=True
        guard=StatusGuard(clock,initial_actions=('FIXED_STAND',''))
        state,feedback=read();runtime.align(feedback.root);runtime.policy.reset()
        # Actual prepared feedback, original neural decoder, envelope check,
        # and freshness BEFORE asking the SDK for any EXTERNAL ownership.
        command,observation,diag=runtime.runtime.infer(runtime.standing,0.,q=feedback.q,dq=feedback.dq,
            root=feedback.root,gyro=feedback.gyro)
        emit('prepared_smpl_candidate',observation=observation.tolist(),raw_action=diag['raw_action'],
            body_targets=[dict(q=j.q,dq=j.dq,tau=j.tau,kp=j.kp,kd=j.kd) for j in command.joints],
            state_id=int(state.state_id),feedback=dict(q=feedback.q.tolist(),dq=feedback.dq.tolist(),
                root=feedback.root.tolist(),gyro=feedback.gyro.tolist()))
        envelope.check(command);inputs.fresh(state);config_unchanged();guard.check(controller.status)
        if stop.is_set():report['stop_reason']='stopped_before_external';return report
        runtime.policy.reset()
        attempted=True
        entered=controller.enter_external(timeout=5.)
        if (entered.current_action!='EXTERNAL' or phase(entered)!='RUNNING'
                or entered.client_id!=controller.client_id or not entered.control_id or entered.generation<=0
                or entered.runtime_id!=reply.runtime_id):raise RuntimeError('Entry did not confirm owned EXTERNAL/RUNNING')
        guard.check(controller.status,entered)
        emit('external_entered',status=status_row(entered))
        start=deadline=clock.monotonic();ready_at=start+3.
        end=start+args.seconds if args.seconds>0 else float('inf')
        last_identity=None;last_receipt=None;stale_ticks=0;ready_printed=False
        report['duplicate_body_ticks']=0
        while not stop.is_set() and clock.monotonic()<end:
            if clock.monotonic()<deadline:
                clock.sleep(max(0.,deadline-clock.monotonic()));continue
            cycle=clock.monotonic();service_check()
            try:state,feedback=read()
            except StaleStateError as exc:
                stale_ticks+=1
                emit('body_state_age_failure',consecutive_ticks=stale_ticks,error=str(exc))
                if stale_ticks>MAX_STALE_TICKS:raise
                deadline,_=next_policy_deadline(deadline,clock.monotonic(),.02);continue
            identity=(state.state_id,state.ts_state_real_us,state.ts_state_pub_us)
            if identity==last_identity or (last_receipt is not None and state.received_monotonic_ns<=last_receipt):
                report['duplicate_body_ticks']+=1
                deadline,_=next_policy_deadline(deadline,clock.monotonic(),.02);continue
            stamp=now_ns(clock)
            if flow.last_step_ns is not None and stamp<=flow.last_step_ns:
                deadline,_=next_policy_deadline(deadline,clock.monotonic(),.02);continue
            pose,buttons,receiver_state=receiver.poll(stamp)
            if receiver_state.get('state')=='failed':raise RuntimeError('Pico receiver failed: '+str(receiver_state.get('error')))
            if pose is not None:runtime.push(pose,feedback)
            if clock.monotonic()>=ready_at:
                if not ready_printed:
                    print('SMPL_ONNX_READY: release both triggers then squeeze both to follow; release/re-squeeze to pause.',flush=True)
                    ready_printed=True
                flow.button(buttons,stamp)
            else:flow.toggle.armed=False  # Never queue a pre-ready held press.
            published_action=runtime.policy.last_action.copy()
            result=flow.step(stamp,feedback)
            pending_action=runtime.policy.last_action.copy();runtime.policy.last_action=published_action
            if not result['inferred']:raise RuntimeError('SMPL standing reference unavailable')
            envelope.check(result['command'])
            try:inputs.fresh(state)
            except StaleStateError as exc:
                stale_ticks+=1;emit('body_state_age_failure',consecutive_ticks=stale_ticks,error=str(exc))
                if stale_ticks>MAX_STALE_TICKS:raise
                deadline,_=next_policy_deadline(deadline,clock.monotonic(),.02);continue
            guard.check(controller.status,entered);config_unchanged()
            if flow.mode=='tracking' and not flow.reference_ready(now_ns(clock)):
                flow.lose(now_ns(clock),'Pico reference expired before command publication')
                flush_events();deadline,_=next_policy_deadline(deadline,clock.monotonic(),.02);continue
            if stop.is_set():break
            command=replace(result['command'],protocol_version=runtime.command_protocol_version,
                input_state_id=int(state.state_id),
                input_state_ts_real_us=int(state.ts_state_real_us),ts_client_created_us=clock.time_ns()//1000)
            sent=controller.publish(command,target_names=envelope.target_names,required_fields=BODY_FIELDS_ALL)
            runtime.policy.last_action=pending_action;sent_count+=1;stale_ticks=0
            last_identity=identity;last_receipt=state.received_monotonic_ns
            report['actions_sent']=sent_count;report['onnx_control_enabled']=True
            if result['mode']=='tracking':report['tracking_steps']+=1
            flush_events();on_result(result,stamp)
            emit('smpl_command_sent',stage=result['mode'],state_id=int(state.state_id),
                protocol_version=sent.protocol_version,
                session_id=sent.session_id,sequence=sent.sequence,raw_action=result['report']['raw_action'],
                targets=[dict(q=j.q,dq=j.dq,tau=j.tau,kp=j.kp,kd=j.kd) for j in sent.joints])
            record_period_overrun(write,'smpl_complete_cycle',clock.monotonic()-cycle,sent_count)
            if sent_count%50==0:print(f'SMPL_LIVE steps={sent_count} stage={result["mode"]} pico_frames={flow.frames}',flush=True)
            deadline,_=next_policy_deadline(deadline,clock.monotonic(),.02)
        report['stop_reason']='operator_stop' if stop.is_set() else 'duration_elapsed'
        return report
    finally:
        primary_error=sys.exc_info()[1]
        report.update(pico_frames=flow.frames,pico_losses=flow.losses,tracking_entries=flow.started_following,
            model_artifacts=runtime.runtime.config['artifacts'])
        if attempted:
            try:
                status=controller.exit_external(timeout=5.)
                if status.current_action!='PASSIVE' or phase(status)!='IDLE':raise RuntimeError('CANCEL did not confirm PASSIVE/IDLE')
                report['exit_confirmed']=True;emit('external_exit_confirmed',status=status_row(status))
            except BaseException as exc:
                try:actual=status_row(controller.status)
                except BaseException:actual=None
                emit('exit_failure',error=repr(exc),actual_status=actual)
                # Preserve the primary failure, but never silently report a
                # clean exit if only the CANCEL operation failed.
                report['exit_error']=str(exc)
                if primary_error is None:raise
        elif fixed_ready:
            failures=[]
            def cancelled(row):
                write(dict(row,control_handles_created=1))
                if row['record_type']=='startup_exit_failure':failures.append(row)
            cancel_startup(controller,cancel,clock,cancelled,startup_generation)
            if failures:
                report['exit_error']=failures[-1]['error']
                if primary_error is None:raise RuntimeError('Startup CANCEL did not confirm exit')
