"""Actual aligned live call chain, deterministic in-memory I/O, no model assets.

The network and controller are fakes. The shared control loop, startup,
reference adapter, input freshness, command envelope and ownership guards run
unchanged. This verifies lifecycle faults, not training observations or firmware.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import actor_pico_live as live
from deployment.vita_s100_aorta_30dof import pico_aligned_runtime as aligned


@dataclass
class Joint:
    q: float=0.
    dq: float=0.
    tau: float=0.
    kp: float=10.
    kd: float=1.


@dataclass
class Command:
    joints: tuple
    protocol_version: int=4
    input_state_id: int=0
    input_state_ts_real_us: int=0
    ts_client_created_us: int=0
    session_id: int=0
    sequence: int=0


@contextmanager
def python_boundary():
    """Import real array/roster validators without the ARM transport initializer.

    Only SDK import-time DTO/transport names are placeholders; the two actual
    validators from sdkmain_adapter.py execute on every synthetic frame.
    """
    modules={}
    def module(name,**values):
        value=ModuleType(name);value.__dict__.update(values);modules[name]=value
    module('locomotion_aorta',BodyCmd=Command,BodyJointCmd=Joint,JointError=NS(NONE=0))
    module('locomotion_aorta._body_codec',decode_body_state=None)
    module('locomotion_aorta.transport',_AortaTransport=object,BODY_CMD_TOPIC='test-only')
    module('locomotion_aorta.pull',PullTransport=object)
    module('aorta.services.common.ServiceStatus',ServiceStatus=NS(SUCCESS=0,PARTIAL_SUCCESS=1))
    name='deployment.vita_s100_aorta_30dof.sdkmain_adapter'
    spec=importlib.util.spec_from_file_location(name,ROOT/'src/deployment/vita_s100_aorta_30dof/sdkmain_adapter.py')
    adapter=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,modules):
        spec.loader.exec_module(adapter)
        with patch.dict(sys.modules,{name:adapter}):yield adapter


class Clock:
    now=100.
    def monotonic(self):return round(self.now,9)
    def time_ns(self):return 1_800_000_000_000_000_000+round(self.now*1e9)
    def sleep(self,seconds):self.now+=max(0.,seconds)


class Flow:
    """Input fixture: a body sample becomes a ready modality reference."""
    def __init__(self):
        self.last_frame_ns=None;self.frames=0;self.losses=0;self.source_fault=None
        self.button_not_before_ns=0;self.events=[];self.calibration=True
        self.latest=(np.zeros((3,3)),np.tile([1.,0.,0.,0.],(3,1)))
        self.window=NS(reference=lambda stamp:self.latest)
    def reference_ready(self,stamp):
        return self.last_frame_ns is not None and 0<=stamp-self.last_frame_ns<=100_000_000
    def lose(self,stamp,reason):
        if self.source_fault is None:
            self.source_fault=reason;self.losses+=1;self.button_not_before_ns=stamp
            self.last_frame_ns=None


class Runtime:
    """Deterministic neutral model; no weights or model-semantics claim."""
    def __init__(self,root,names,route='pico_teleop'):
        config=json.loads((ROOT/'config/actor_threepoint.json').read_text())
        config.update(pico_max_age_s=.1,pico_hold_age_s=.1,pico_loss_action='stand',
                      retargeting=dict(human_height_m=1.7))
        self.root=root;self.route=route;self.policy_name=route;self.config=config
        self.model=NS(contract=NS(default=np.zeros(28),names=names[:28],period=.02,
            kp=np.full(28,10.),kd=np.ones(28)),backend='fake')
        self.head=None;self.gmr_calibration=None;self.flow=Flow();self.last_feedback=None
        self.reference=NS(delay=0.,steps=10);self.runtime=NS(config=dict(checkpoint_iteration=35000))
        self.math=NS(yaw=lambda root:0.,rotation=lambda root:np.eye(3),
            yaw_quaternion=lambda angle:np.array([1.,0.,0.,0.]),multiply=lambda a,b:np.asarray(b))
        if route=='pico_smpl':
            from deployment.vita_s100_aorta_30dof.actor_smpl import SmplReference
            self.flow.latest=SmplReference(self.math,np.zeros((10,24,3)),
                np.tile([1.,0.,0.,0.],(10,1)),np.zeros((10,6)))
        self.policy=NS(profile=dict(ankle_command_space='virtual_pitch_roll'),last_action=np.zeros(28),order=names,bind_body_info=lambda body:None,
            decode=lambda action:(self.command(),{}),reset=lambda:None,state=self.state)
        self.base=NS(push=self.push)
    def command(self):return Command(tuple(Joint(kp=0.,kd=0.) if i==14 else Joint() for i in range(31)))
    def push(self,pose,feedback):
        body,stamp,received=pose
        if self.flow.source_fault is not None:
            self.flow.source_fault=None;self.flow.button_not_before_ns=received
        self.flow.last_frame_ns=received;self.flow.frames+=1
    def state(self,state):return state.q[:28],state.dq[:28],state.imu_values[0,:4],state.imu_values[0,4:7]
    def reset_inputs(self):self.policy.last_action=np.zeros(28)
    def hold_motion(self):
        return NS(q=np.zeros((2,28)),base=np.zeros((2,3)),quat=np.tile([1.,0.,0.,0.],(2,1)),lead_in=lambda *args:self.hold_motion())
    def infer_motion(self,*args,**kwargs):return self.infer_motion_command(*args,**kwargs)[1:]
    def infer_motion_command(self,motion,**feedback):
        self.last_feedback=NS(**feedback)
        return self.command(),np.zeros(1740 if self.route=='pico_smpl' else 1167),dict(record_type='actor_inference',raw_action=[0.]*28,total_ms=0.)
    def apply_head(self,command,row,**kwargs):return command,None
    def on_lifecycle(self,row):pass


class Controller:
    """Synchronous mock owner and publication boundary, never opens a transport."""
    client_id=11
    def __init__(self,runtime,clock,names):
        self.runtime=runtime;self.clock=clock;self.commands=[];self.operations=[]
        self.after_publish=lambda command:None;self.state_fault=lambda state:state
        self.exit_fault=False;self.publish_fault=False;self.entry_fault=False
        self.value=NS(current_action='PASSIVE',phase='IDLE',status_seq=0,stamp_monotonic_ns=0,
            runtime_id=100,generation=0,client_id=0,control_id=0,reject_count=0,
            ingress_drop_count=0,timeout_hold=False,last_reject_code=0)
        caps=tuple(NS(body_index=i,is_mock=i==14,slot_role=1,supported_fields=31,valid_limits=31,
            q_min=-4.,q_max=4.,dq_min=-25.,dq_max=25.,tau_min=-100.,tau_max=100.,
            kp_min=0.,kp_max=500.,kd_min=0.,kd_max=50.) for i in range(31))
        self.reply=NS(status=0,runtime_id=100,info=NS(protocol_version=2,
            body=NS(slot_count=31,roster_version=1,joint_names=names,joints=caps)))
    @property
    def status(self):
        self.value.status_seq+=1;self.value.stamp_monotonic_ns=round(self.clock.monotonic()*1e9)
        return NS(**vars(self.value))
    def get_robot_info(self,timeout):return self.reply
    def wait_state(self,timeout=0):
        stamp=self.clock.time_ns()//1000
        imu=np.zeros((1,13),np.float32);imu[0,0]=1
        value=NS(q=np.zeros(31,np.float32),dq=np.zeros(31,np.float32),error_code=np.zeros(31,np.uint32),
            joint_stamps_us=np.full((31,3),stamp,np.uint64),imu_values=imu,
            imu_stamps_us=np.full((1,4),stamp,np.uint64),imu_location=np.array([0]),
            imu_status=np.array([1]),imu_state_present=np.array([True]),protocol_version=3,
            attitude_valid=True,roster_version=1,runtime_id=100,state_id=stamp,
            ts_state_real_us=stamp,ts_state_pub_us=stamp,received_monotonic_ns=round(self.clock.monotonic()*1e9),
            pull_context_receive_stamp_ns=0,applied_session_id=7,applied_sequence=len(self.commands),ts_first_output_us=0)
        return self.state_fault(value)
    def firmware(self,action):
        self.operations.append(action);self.value.current_action='FIXED_STAND' if action=='FIXED_STAND' else 'PASSIVE'
    def enter_external(self,timeout):
        self.operations.extend(['PREPARE','COMMIT'])
        if self.entry_fault:raise RuntimeError('injected entry failure')
        self.value.current_action='EXTERNAL';self.value.phase='RUNNING';self.value.generation=7
        self.value.client_id=self.client_id;self.value.control_id=22
        return self.status
    def publish(self,command,**kwargs):
        if self.publish_fault:raise RuntimeError('injected publication failure')
        command=replace(command,session_id=7,sequence=len(self.commands)+1)
        self.commands.append(command);self.after_publish(command);return command
    def exit_external(self,timeout):
        self.operations.append('CANCEL')
        if self.exit_fault:raise RuntimeError('injected cancellation failure')
        self.value.current_action='PASSIVE';self.value.phase='IDLE';return self.status


class Receiver:
    def __init__(self):self.start=None;self.silent=False;self.failed=False;self.buttons=lambda t:False
    def poll(self,stamp):
        if self.start is None:self.start=stamp
        both=float(self.buttons((stamp-self.start)*1e-9))
        return (None if self.silent else (None,stamp,stamp)),(True,stamp,0.,both,both),dict(
            state='failed' if self.failed else 'receiving',error='injected worker failure')


def effective_config():
    return dict(parallel_ankle=dict(pd=dict(strategy='virtual_pd'),
        geometry=dict(lx=.05,ly=.02175,z_top=.31,z_bot=.235)),actions=dict(
        EXTERNAL=dict(generator='builtin::external',params=dict(ankle_command_space='virtual_pitch_roll',
            ankle_virtual_pd_intent=True,command_timeout_ms=1000,recover_valid_frames=1,
            interpolation=dict(enabled=False,min_period_ms=2,max_period_ms=100),
            safe_hold=dict(kp=[10.]*31,kd=[1.]*31))),
        FIXED_STAND=dict(generator='builtin::keyframe',params=dict(
            keyframes=[dict(timestamp=0,positions=[0.]*31,kp=[10.]*31,kd=[1.]*31)])),
        **{name:dict(params=dict(kp=0.,keyframes=[dict(timestamp=0,kd=5.),dict(timestamp=.02,kd=0.)]))
           for name in ('PASSIVE','QUIET_PASSIVE')}))


class AlignedLiveTests(unittest.TestCase):
    route='pico_teleop'
    def context(self,manager):
        value=manager.__enter__();self.addCleanup(manager.__exit__,None,None,None)
        return value
    def setUp(self):
        self.boundary=self.context(python_boundary())
        self.temp=self.context(tempfile.TemporaryDirectory())
        self.clock=Clock();self.stop=threading.Event();self.rows=[]
        self.runtime=Runtime(Path(self.temp),self.boundary.BODY_NAMES,self.route)
        self.controller=Controller(self.runtime,self.clock,self.boundary.BODY_NAMES)
        self.receiver=Receiver()
        path=Path(self.temp)/'effective.json';path.write_text(json.dumps(effective_config()))
        self.args=NS(robot_config=path,robot_id='IN_MEMORY_ONLY',startup_timeout=3.,seconds=5.)
    def run_session(self,**options):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            return aligned.run_aligned(self.runtime,self.args,self.rows.append,self.controller,
                self.receiver,self.stop,clock=self.clock,request_action=self.controller.firmware,**options)
    def assert_cancelled(self):
        self.assertIn('CANCEL',self.controller.operations)
        self.assertEqual(self.controller.value.current_action,'PASSIVE')
    def test_wait_follow_pause_and_single_control_session(self):
        self.receiver.buttons=lambda t:3.3<t<3.5 or 4.2<t<4.4
        result=self.run_session()
        self.assertTrue(result['exit_confirmed']);self.assertEqual(result['tracking_entries'],1)
        self.assertGreater(result['tracking_steps'],0)
        self.assertEqual(self.controller.operations,['FIXED_STAND','PREPARE','COMMIT','CANCEL'])
        records=[row['record_type'] for row in self.rows]
        self.assertIn('tracking_started',records);self.assertIn('tracking_paused',records)
        self.assertEqual([c.sequence for c in self.controller.commands],list(range(1,result['actions_sent']+1)))
    def test_aligned_entry_calls_the_single_shared_runner(self):
        self.args.seconds=.1
        with patch.object(live,'run_live',wraps=live.run_live) as shared:
            self.run_session()
        self.assertEqual(shared.call_count,1)
        self.assertIsInstance(shared.call_args.kwargs['reference_adapter'],aligned.TaskReferences)
    def test_receiver_event_requires_release_after_body_recovery(self):
        self.args.seconds=5.6
        start=None
        def poll(stamp):
            nonlocal start
            if start is None:start=stamp
            t=(stamp-start)*1e-9
            sequence=0;released=pressed=0
            # The second press is armed only by a release BEFORE the body
            # outage. It must be consumed but never forwarded for engagement.
            for i,(release,press) in enumerate(((3.,3.3),(3.8,4.4),(5.,5.2)),1):
                if t>=press:sequence=i;released=start+round(release*1e9);pressed=start+round(press*1e9)
            pose=None if 4.<t<4.3 else (None,stamp,stamp)
            return pose,(True,stamp,0.,0.,0.),dict(state='receiving',trigger_event_seq=sequence,
                trigger_event_monotonic_ns=pressed,trigger_event_release_monotonic_ns=released)
        self.receiver.poll=poll
        result=self.run_session()
        self.assertEqual(result['tracking_entries'],2)
        self.assertEqual(result['pico_losses'],1)
        events=[r for r in self.rows if r['record_type']=='pico_trigger_toggle']
        self.assertEqual(len(events),2)
        # Had the fault-before-release press escaped, this would resume near
        # 4.4 seconds (220 publications), instead of the fresh 5.2s press.
        self.assertGreaterEqual(events[-1]['actions_sent'],259)
        self.assert_cancelled()
    def test_missing_pico_keeps_stand_without_tracking(self):
        self.receiver.silent=True
        result=self.run_session()
        self.assertEqual(result['tracking_entries'],0);self.assertGreater(result['actions_sent'],200)
        self.assert_cancelled()
    def test_preheld_triggers_never_engage(self):
        self.receiver.buttons=lambda t:True
        self.assertEqual(self.run_session()['tracking_entries'],0)
    def test_worker_failure_keeps_stand(self):
        self.receiver.failed=True
        result=self.run_session();self.assertEqual(result['tracking_entries'],0);self.assert_cancelled()
    def test_stale_feedback_stops_publication_and_cancels(self):
        def fail(command):
            def stale(state):state.received_monotonic_ns-=200_000_000;return state
            self.controller.state_fault=stale
        self.controller.after_publish=fail
        with self.assertRaises(live.StaleStateError):self.run_session()
        self.assertEqual(len(self.controller.commands),1);self.assert_cancelled()
    def test_ownership_change_stops_publication(self):
        self.controller.after_publish=lambda command:setattr(self.controller.value,'client_id',999)
        with self.assertRaisesRegex(RuntimeError,'ownership'):self.run_session()
        self.assertEqual(len(self.controller.commands),1);self.assert_cancelled()
    def test_joint_bounds_are_checked_on_every_frame(self):
        def fail(command):
            def outside(state):state.q[0]=10.;return state
            self.controller.state_fault=outside
        self.controller.after_publish=fail
        with self.assertRaisesRegex(ValueError,'outside bounds'):self.run_session()
        self.assertEqual(len(self.controller.commands),1);self.assert_cancelled()
    def test_publication_failure_cancels(self):
        self.controller.publish_fault=True
        with self.assertRaisesRegex(RuntimeError,'publication'):self.run_session()
        self.assertEqual(self.controller.commands,[]);self.assert_cancelled()
    def test_unknown_initial_owner_never_requests_motion(self):
        self.controller.value.current_action='OTHER_OWNER'
        with self.assertRaisesRegex(RuntimeError,'must start'):self.run_session()
        self.assertEqual(self.controller.operations,[]);self.assertEqual(self.controller.commands,[])
    def test_failed_entry_still_attempts_checked_cancel(self):
        self.controller.entry_fault=True
        with self.assertRaisesRegex(RuntimeError,'entry failure'):self.run_session()
        self.assertEqual(self.controller.commands,[]);self.assert_cancelled()
    def test_prepared_policy_failure_cancels_before_external(self):
        infer=self.runtime.infer_motion_command;calls=0
        def fail(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError('injected prepared policy failure')
            return infer(*args,**kwargs)
        self.runtime.infer_motion_command=fail
        with self.assertRaisesRegex(ValueError,'prepared policy'):self.run_session()
        self.assertEqual(self.controller.operations,['FIXED_STAND','CANCEL'])
        self.assertEqual(self.controller.commands,[])
    def test_feedback_runtime_id_change_cancels(self):
        def fail(command):
            def changed(state):state.runtime_id=999;return state
            self.controller.state_fault=changed
        self.controller.after_publish=fail
        with self.assertRaisesRegex(ValueError,'runtime_id mismatch'):self.run_session()
        self.assertEqual(len(self.controller.commands),1);self.assert_cancelled()
    def test_cancellation_failure_is_not_reported_as_success(self):
        self.controller.exit_fault=True;self.args.seconds=.1
        with self.assertRaisesRegex(RuntimeError,'cancellation'):self.run_session()
        self.assertTrue(any(row['record_type']=='exit_failure' for row in self.rows))
    def test_operator_stop_uses_same_checked_exit(self):
        self.controller.after_publish=lambda command:self.stop.set()
        result=self.run_session();self.assertTrue(result['exit_confirmed'])
        self.assertEqual(result['stop_reason'],'operator_stop');self.assert_cancelled()


class DefaultReferenceTests(unittest.TestCase):
    def test_warmup_hook_config_change_rejects_before_control(self):
        # The original actor's infer_motion hook must still run during warmup.
        # Exercise default JointReferences, not the Pico task-reference adapter.
        for policy in ('sonic','supermimic','morphmimic'):
            with self.subTest(policy=policy),python_boundary() as boundary,tempfile.TemporaryDirectory() as temp:
                root=Path(temp);clock=Clock();stop=threading.Event()
                runtime=Runtime(root,boundary.BODY_NAMES);runtime.policy_name=policy
                controller=Controller(runtime,clock,boundary.BODY_NAMES)
                config=root/'effective.json';config.write_text(json.dumps(effective_config()))
                args=NS(robot_config=config,robot_id='IN_MEMORY_ONLY',startup_timeout=3.,seconds=.1,
                    stand_only=True)
                original=runtime.infer_motion
                def changed(*values,**feedback):
                    result=original(*values,**feedback)
                    settings=json.loads(config.read_text())
                    settings['actions']['EXTERNAL']['params']['ankle_virtual_pd_intent']=False
                    config.write_text(json.dumps(settings))
                    return result
                with patch.object(runtime,'infer_motion',side_effect=changed) as hook, \
                     redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(ValueError,'ankle_virtual_pd_intent'):
                        live.run_live(runtime,args,lambda *row:None,controller,None,stop,lambda:True,clock=clock)
                hook.assert_called_once()
                self.assertEqual(controller.operations,[])
                self.assertEqual(controller.commands,[])


class ReferenceHeadingTests(unittest.TestCase):
    def test_teleop_heading_survives_pause_and_loss_stand_references(self):
        # Exercise the actual stand-reference seam, without loading any model.
        from deployment.vita_s100_aorta_30dof.pico_aligned_runtime import TaskInputs,StandReference
        from deployment.vita_s100_aorta_30dof.pico_threepoint_frontend import qmatrix
        yaw=lambda angle:np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])
        base=TaskInputs.__new__(TaskInputs);base.route='pico_teleop'
        base.asset_root=yaw(0.);base.runtime=NS(reference_root=base.asset_root.copy())
        base.math=NS(yaw=lambda q:2*np.arctan2(q[3],q[0]),yaw_quaternion=yaw,
            multiply=lambda a,b:a.copy())  # asset_root is the identity quaternion.
        base.standing=(np.zeros((3,3)),np.tile(yaw(0.),(3,1)))
        owner=NS(base=base,last_task=None,model=NS(contract=NS(default=np.zeros(28))))
        StandReference(owner).lead_in(np.zeros(28),yaw(.4),{})
        startup=base.runtime.reference_root.copy()
        np.testing.assert_allclose(startup,yaw(.4))
        for measured_yaw in (.8,-.2):
            StandReference(owner).lead_in(np.zeros(28),yaw(measured_yaw),{})
            np.testing.assert_array_equal(base.runtime.reference_root,startup)
            # A yaw deviation remains visible to the actual root6 feature;
            # returning to stand must not redefine that error as zero.
            root6=(qmatrix(yaw(measured_yaw)).T@qmatrix(base.runtime.reference_root))[:,:2]
            self.assertFalse(np.allclose(root6,np.eye(3)[:,:2]))


class SmplAlignedLiveTests(AlignedLiveTests):
    route='pico_smpl'


if __name__=='__main__':unittest.main()
