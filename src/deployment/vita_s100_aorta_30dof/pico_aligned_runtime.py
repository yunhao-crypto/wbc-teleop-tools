"""Typed human/three-point references for the unchanged Git live state machine.

Only reference acquisition and encoder input differ from the Git joint-reference
actor. No controller, middleware node or service is constructed here.
"""
from dataclasses import dataclass,replace
from types import SimpleNamespace as NS
import time
import numpy as np
from .actor_pico_buttons import DualTriggerToggle
from .pico_realtime import Feedback
from .actor_pico_live import Inputs as GitInputs,MEASURED_Q_MARGIN


class GuardedInputs(GitInputs):
    """Preserve the prior per-frame hardware-range/runtime-id protection.

    The fetched runner checks measured ranges during entry. The existing SMPL
    deployment additionally checks them on every frame; keep that protection
    while retaining the Git state-machine function itself unchanged.
    """
    def __init__(self,runtime,controller,pico,clock):
        super().__init__(runtime,controller,pico,clock)
        reply=controller.get_robot_info(5.)
        self.robot_runtime_id=reply.runtime_id;self.body_info=reply.info.body

    def validate_state(self,state):
        feedback=super().validate_state(state)
        if state.runtime_id!=self.robot_runtime_id:raise ValueError('BodyState runtime_id mismatch')
        for i,cap in enumerate(self.body_info.joints):
            if not cap.is_mock and not cap.q_min-MEASURED_Q_MARGIN<=state.q[i]<=cap.q_max+MEASURED_Q_MARGIN:
                raise ValueError('Measured virtual joint outside bounds: '+self.body_info.joint_names[i])
        return feedback


@dataclass
class TaskReference:
    value: object


class StandReference:
    """Neutral encoder reference with the Git runner's preparation-plan interface.

    q/base/quat describe the model's neutral ROBOT pose for review only. They
    are never used as measured feedback or supplied to either human encoder.
    These modalities cannot invert measured robot joints to human XYZ/site
    targets. lead_in therefore blends encoded task references, never outputs.
    The actual encoded reference is logged separately on every inference.
    """
    def __init__(self,runtime):
        self.owner=runtime;self.source=runtime.last_task
        self.q=np.repeat(runtime.model.contract.default[None],51,axis=0)
        self.base=np.zeros((51,3));self.quat=np.tile([1.,0.,0.,0.],(51,1))

    def lead_in(self,q,root,profile):
        # q is already validated by the Git startup/envelope. Do not claim an
        # inverse human skeleton or replace the model by joint interpolation.
        self.owner.base.align(root)
        return self

    def target(self,seconds):
        target=self.owner.base.standing
        if self.source is None:return target
        a=float(np.clip(seconds,0.,1.));a=a*a*(3-2*a)
        return self.owner.flow.blend(self.source,target,a)


class AlignedRuntime:
    def __init__(self,root,route,*,backend='sdk',threads=1,human_height=1.70):
        if route=='pico_smpl':
            from .pico_smpl_live import SmplLiveRuntime
            self.base=SmplLiveRuntime(root,backend=backend,threads=threads)
        elif route=='pico_teleop':
            from .pico_teleop_live import TeleopLiveRuntime
            self.base=TeleopLiveRuntime(root,backend=backend,threads=threads)
        else:raise ValueError('Explicit pico_smpl or pico_teleop required')
        self.root=self.base.root;self.route=route;self.flow=self.base.flow
        self.runtime=self.base.runtime;self.policy=self.base.policy;self.math=self.base.math
        self.model=NS(contract=self.base.model.contract,backend=backend)
        self.config={**self.base.config,'pico_max_age_s':.1,'pico_hold_age_s':.1,'pico_loss_action':'stand',
            'retargeting':dict(human_height_m=human_height)}
        self.policy_name=route;self.head=None;self.gmr_calibration=None
        self.reference=NS(steps=10,current_index=0,delay=.28 if route=='pico_smpl' else 0.,ref_time=0.)
        self.last_task=None;self.last_feedback=None;self.track_since=None;self.track_from=None
        self.clock=time
        # A real encoder reference is available before current Pico calibration.
        self.base.align(np.array([1.,0.,0.,0.]))

    def reset_inputs(self):
        self.policy.reset();self.last_task=None;self.track_since=None;self.track_from=None

    def hold_motion(self):return StandReference(self)

    def on_lifecycle(self,row):
        kind=row.get('record_type')
        if kind=='tracking_started':
            self.track_since=self.clock.monotonic();self.track_from=self.last_task
        elif kind in ('tracking_paused','pico_loss_stand'):
            self.track_since=None;self.track_from=None

    def infer_motion(self,motion,**feedback):
        _,observation,row=self.infer_motion_command(motion,**feedback)
        return observation,row

    def infer_motion_command(self,motion,*,q,dq,root,gyro,diagnostics=None,ref_time=0.,yaw_offset=0.):
        self.last_feedback=Feedback(np.array(q),np.array(dq),np.array(root),np.array(gyro))
        self.last_feedback.validate()
        begin=time.perf_counter()
        if isinstance(motion,StandReference):
            reference=motion.target(ref_time)
        elif isinstance(motion,TaskReference):
            reference=motion.value
            if self.track_since is not None and self.track_from is not None:
                a=float(np.clip(self.clock.monotonic()-self.track_since,0.,1.));a=a*a*(3-2*a)
                reference=self.flow.blend(self.track_from,reference,a)
        else:raise ValueError('An explicit modality reference is required')
        if self.route=='pico_smpl':
            command,observation,report=self.runtime.infer(reference,0.,q=q,dq=dq,root=root,gyro=gyro)
            encoded=dict(kind='human_local_XYZ_root_standing_wrists',joints=reference.joints.tolist(),
                root_wxyz=reference.quat.tolist(),wrists=reference.wrists.tolist())
        else:
            from .actor_threepoint import ThreePointTarget
            target=ThreePointTarget(*reference,max(0.,self.clock.monotonic()))
            command,observation,report=self.runtime.infer(target,q=q,dq=dq,root=root,gyro=gyro)
            encoded=dict(kind='calibrated_threepoint_with_fixed_lower_reference',positions=reference[0].tolist(),
                orientations_wxyz=reference[1].tolist(),lower=self.runtime.lower.tolist(),
                reference_root_wxyz=self.runtime.reference_root.tolist())
        self.last_task=reference
        command=replace(command,protocol_version=self.base.command_protocol_version)
        return command,observation,dict(report,record_type='actor_inference',policy=self.route,
            observation_dim=len(observation),action_dim=28,total_ms=(time.perf_counter()-begin)*1000,
            encoded_reference=encoded,preparation_reference_semantics='neutral task reference, not robot joint lead-in',
            **(diagnostics or {}))

    def apply_head(self,command,row,**kwargs):return command,None


class ManualInput:
    """The Git manual worker snapshot contract, backed by raw atomic Pico input."""
    def __init__(self,runtime,receiver,*,clock=time,service_check=lambda:True,write=lambda row:None):
        self.runtime=runtime;self.receiver=receiver;self.clock=clock;runtime.clock=clock
        self.service_check=service_check;self.write=write;self.connected=False
        self.toggle=DualTriggerToggle(.6);self.event_seq=0;self.event_time=float('-inf');self.event_index=0
        self.last_loss=0

    def connect(self,root):
        if self.connected:return False
        self.connected=True;return True

    def snapshot(self):
        now=self.clock.monotonic();stamp=round(now*1e9);r=self.runtime;flow=r.flow
        try:
            if self.service_check() is False:raise RuntimeError('Pico service exited')
            pose,buttons,state=self.receiver.poll(stamp)
            if state.get('state')=='failed':raise RuntimeError('Pico receiver failed: '+str(state.get('error')))
            if pose is not None:
                if r.last_feedback is None:raise ValueError('No measured robot feedback before Pico calibration')
                r.base.push(pose,r.last_feedback)
            if flow.last_frame_ns is not None and stamp-flow.last_frame_ns>100_000_000:
                flow.lose(stamp,'No fresh Pico reference within100ms')
            if flow.losses!=self.last_loss:
                self.toggle=DualTriggerToggle(.6);self.last_loss=flow.losses
            if 'trigger_event_seq' in state:
                # Receiver latches at 2ms, independently of inference. The Git
                # runner consumes its monotonic event counter, not a late poll
                # of a potentially already released trigger pulse.
                self.event_seq=int(state['trigger_event_seq'])
                self.event_time=int(state['trigger_event_monotonic_ns'])/1_000_000_000
            elif self.toggle.update(buttons,now):self.event_seq+=1;self.event_time=now
            for row in flow.events[self.event_index:]:self.write(row)
            self.event_index=len(flow.events)
            available,_,age,_,_=buttons
            controller_receipt=now-age if available and np.isfinite(age) and 0<=age<=.6 else float('-inf')
            ready=flow.reference_ready(stamp)
            value=(flow.window.reference(stamp) if r.route=='pico_smpl' else flow.latest) if ready else None
            result=dict(state='ready' if ready else 'waiting_calibration' if flow.calibration is None else 'waiting_body',
                received_monotonic=flow.last_frame_ns/1_000_000_000 if flow.last_frame_ns is not None else float('-inf'),
                controller_received_monotonic=controller_receipt,trigger_event_seq=self.event_seq,
                trigger_event_monotonic=self.event_time,task_reference=TaskReference(value) if ready else None,
                source_gap_s=0.,retarget_ms=0.,gmr_task_error=0.)
        except Exception as exc:
            self.toggle=DualTriggerToggle(.6)
            flow.lose(stamp,str(exc))
            result=dict(state='failed',error=str(exc),trigger_event_seq=self.event_seq)
        for row in flow.events[self.event_index:]:self.write(row)
        self.event_index=len(flow.events)
        return result


def manual_reference(runtime,snapshot,now):
    if snapshot.get('state')!='ready' or not isinstance(snapshot.get('task_reference'),TaskReference):
        raise ValueError('Pico reference not ready: '+str(snapshot.get('state')))
    age=now-float(snapshot['received_monotonic'])
    if not 0<=age<=.1 or not runtime.flow.reference_ready(round(now*1e9)):
        raise TimeoutError('Pico reference expired within100ms budget')
    return snapshot['task_reference'],dict(pico_age_s=age,reference_delay_s=runtime.reference.delay)


def reference_yaw(runtime,motion,root):
    if runtime.route=='pico_smpl':
        # Match Git's engagement-time heading alignment using the human root,
        # without pretending that human XYZ is a robot joint-reference Motion.
        return runtime.math.yaw(root)-runtime.math.yaw(motion.value.quat[0])
    # Teleop sites are reference-base targets with a fixed standing heading.
    return 0.


def align_smpl_task(runtime,reference,offset):
    from .actor_smpl import SmplReference
    m=runtime.math;rotation=m.yaw_quaternion(offset)
    return SmplReference(m,reference.joints@m.rotation(rotation).T,
        np.array([m.multiply(rotation,q) for q in reference.quat]),reference.wrists.copy())


def pico_reference_command(runtime,snapshot,now,*,q,dq,root,gyro,yaw_offset=None):
    motion,diagnostics=manual_reference(runtime,snapshot,now)
    if runtime.route=='pico_smpl':
        if yaw_offset is None:yaw_offset=reference_yaw(runtime,motion,root)
        motion=TaskReference(align_smpl_task(runtime,motion.value,yaw_offset))
    return runtime.infer_motion_command(motion,q=q,dq=dq,root=root,gyro=gyro,diagnostics=diagnostics)


def run_aligned(runtime,args,write,controller,receiver,stop,*,service_alive=lambda:True,clock=time,
                request_action=None,on_result=lambda result,stamp:None,poll_command=lambda:None):
    """Call the source-identical Git runner with real or test-injected I/O."""
    from .pico_aligned_lifecycle import run_live
    runtime.clock=clock
    options=NS(**vars(args));options.auto_start=True;options.manual_pico=True;options.wait_pico=True
    options.auto_pico=False;options.stand_only=False;options.entry_policy='policy-online';options.gripper_config=None
    counts=dict(actions_sent=0,tracking_steps=0,tracking_entries=0,startup_fixed_stand_confirmed=False)
    def record(row,observation=None):
        row=dict(row);runtime.on_lifecycle(row)
        if row.get('record_type')=='tracking_started':counts['tracking_entries']+=1
        if row.get('record_type')=='fixed_stand_confirmed':counts['startup_fixed_stand_confirmed']=True
        sent=int(row.get('actions_sent',counts['actions_sent']))
        if observation is not None and sent>counts['actions_sent']:
            tracking=row.get('stage')=='pico_tracking';counts['tracking_steps']+=tracking
            on_result(dict(observation=observation,report=row,mode='tracking' if tracking else 'stand'),round(clock.monotonic()*1e9))
        counts['actions_sent']=max(counts['actions_sent'],sent)
        write(dict(row,route=runtime.route,actions_sent=counts['actions_sent'],control_handles_created=1))
    manual=ManualInput(runtime,receiver,clock=clock,service_check=service_alive,write=record)
    result=run_live(runtime,options,record,controller,None,stop,service_alive,clock=clock,
        manual_pico=manual,poll_command=poll_command,request_action=request_action)
    result.update(counts,route=runtime.route,policy_step=runtime.runtime.config['checkpoint_iteration'],pico_frames=runtime.flow.frames,
        original_git_commit='ca0aa6ec8496d0c65932350b38eba5103f570140',
        robot_configuration_modified=False,protection_relaxed=False,physical_stability_verified=False,
        firmware_v4_acceptance_verified=False)
    return result
