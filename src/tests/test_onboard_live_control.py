"""Whole live-control loop wiring without SDK transport, robot, GPU or models.

The FSM, stand interpolation, cycle clock, command adapter and fail-closed
command guard are real. Only external IO and numerical policy leaves are fakes.
"""
from contextlib import nullcontext, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from deployment.common.controller_input import Button, ControllerSample
from deployment.common.policy_runtime import PolicyStep
from deployment.vita_s100_aorta_30dof import real_teleop as teleop
from deployment.vita_s100_aorta_30dof.config import load_vita_config
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_ACTION_TO_COMMAND_SLOTS
from deployment.vita_s100_aorta_30dof.aorta_adapter import VITA_CONTROLLER_JOINT_NAMES


@pytest.mark.parametrize('scenario', ['flow','back_active','reference_loss','stale_entry','missing_entry','ownership_loss','ownership_loss_entry','stop_during_enter','link_loss_entry','back_during_enter','idle_during_enter','neutral_motion','neutral_velocity','neutral_pico_loss'])
def test_native_runtime_whole_control_wiring(tmp_path,monkeypatch,scenario):
    clock=SimpleNamespace(now=10.)
    fake_time=SimpleNamespace(monotonic=lambda:clock.now,monotonic_ns=lambda:round(clock.now*1e9),
                             thread_time_ns=lambda:round(clock.now*1e9),
                             time=lambda:clock.now,time_ns=lambda:round(clock.now*1e9),sleep=lambda dt:setattr(clock,'now',clock.now+dt))
    monkeypatch.setattr(teleop,'time',fake_time)
    config=replace(load_vita_config(Path(teleop.__file__).parent/'config/vita02a_s100.yaml'),
                   logs=tmp_path/'logs',stand_transition_seconds=.06)
    status=SimpleNamespace(current_action='QUIET_PASSIVE',phase='IDLE',client_id='mock',lease_id='lease',generation=1,reject_count=0)
    published=[];events=[];reads=[];calls=[];controller_count=[0];closed=[];lifecycle=[]
    target=PolicyStep(mode='stand',joint_names=VITA_CONTROLLER_JOINT_NAMES,raw_action=np.zeros(30),
        target_joint_pos=np.full(30,.1),joint_stiffness=np.full(30,10.),joint_damping=np.ones(30))
    class Reference:
        def has_future_sequence(self,*a,**kw):
            return not ((scenario=='reference_loss' and controller_count[0]>=11)
                        or (scenario=='neutral_pico_loss' and controller_count[0]>=13))
        def data_age(self,now):return 0.
        def reset_root_anchor(self):calls.append(('anchor_reset',))
        def current_root_rot(self):return np.array([1.,0.,0.,0.])
    reference=Reference()
    class Runtime:
        contracts=SimpleNamespace(velocity=target,motion=target)
        bundle=SimpleNamespace(manifest_path=Path('mock-manifest'),artifacts={n:SimpleNamespace(path=Path(n),sha256='mock') for n in ('velocity','motion')})
        last_timing_ms={}
        def warmup(self):calls.append(('warmup',));return {'mock':True}
        def create_online_reference(self):return reference
        def command_envelope(self,envelope,name):return envelope
        def stand_step(self):return target
        def reset(self):calls.append(('reset',))
        def step_velocity(self,state,command):calls.append(('velocity',tuple(command)));return replace(target,mode='velocity')
        def step_motion(self,state,ref,**kw):
            assert ref is reference
            np.testing.assert_allclose(kw['reference_yaw_alignment_wxyz'],[1.,0.,0.,0.])
            calls.append(('motion',));return replace(target,mode='motion')
    class Session:
        def __init__(self,**kw):self.status=status
        def wait_ready(self,timeout):return status
        def require_available(self,value):assert value.current_action=='QUIET_PASSIVE'
        def enter(self,timeout,**kw):
            assert kw['lease_duration_s']==config.sdk_lease_duration_seconds
            calls.append(('enter',));clock.now+=.25;status.current_action='EXTERNAL';status.phase='RUNNING'
            if scenario=='stop_during_enter':teleop.signal.getsignal(teleop.signal.SIGINT)(teleop.signal.SIGINT,None)
            return SimpleNamespace(**vars(status))
        def publish(self,q,kp,kd):
            assert status.current_action=='EXTERNAL'
            published.append((np.copy(q),np.copy(kp),np.copy(kd),clock.now))
            return SimpleNamespace(command_id=len(published),published_monotonic_ns=round(clock.now*1e9),period_s=.02,
                command_build_started_wall_ns=round(clock.now*1e9),
                publish_started_wall_ns=round(clock.now*1e9),published_wall_ns=round(clock.now*1e9))
        def exit(self,timeout):
            if scenario.startswith('neutral_'):
                assert controller_count[0]==20, 'Neutral link-loss input cancelled ownership before explicit BACK'
                if status.current_action=='EXTERNAL':calls.append(('cancel_active',controller_count[0]))
            calls.append(('exit',));lifecycle.append('cancel');status.current_action='QUIET_PASSIVE';status.phase='IDLE'
        def close(self):closed.append('session')
    class Source:
        group=config.aorta_group
        def read(self,timeout):
            after=status.current_action=='EXTERNAL'
            if after and scenario=='missing_entry':raise TimeoutError('missing post-entry LowState')
            clock.now+=.002
            q=np.zeros(44);q[0]=.01 if after else 0.
            raw=SimpleNamespace(q_by_slot=q,state_id=len(reads)+1,attitude_valid=True,
                    received_monotonic_ns=round((clock.now-(1. if after and scenario=='stale_entry' else 0.))*1e9))
            if (scenario=='ownership_loss' and len(published)>=3) or (scenario=='ownership_loss_entry' and after):status.lease_id='different-owner'
            reads.append(raw);return raw
    class Controller:
        def sample(self,timeout):
            controller_count[0]+=1;i=controller_count[0]
            if scenario=='link_loss_entry' and status.current_action=='EXTERNAL':raise RuntimeError('controller heartbeat lost')
            assert i<=20,'Mock script failed to terminate loop'
            button={1:Button.START,8:Button.A,10:Button.B,12:Button.Y,14:Button.X,16:Button.BACK}.get(i)
            if scenario=='back_active' and i==14:button=Button.BACK
            if scenario=='back_during_enter' and i==2:button=Button.BACK
            if scenario=='idle_during_enter' and i==2:button=Button.X
            axes={'left_y':-.4} if i==9 else {}
            if scenario.startswith('neutral_'):
                # Resilient input adapters emit fresh neutral samples on link
                # loss. Exercise that adapter contract through the real loop,
                # including its extra sample immediately before publication.
                button={1:Button.START,8:Button.A,16:Button.A,20:Button.BACK}.get(i)
                if scenario!='neutral_velocity' and i==10:button=Button.B
                axes={'left_y':-.4} if i in (9,10) and scenario=='neutral_velocity' else {}
                if 11<=i<=15:
                    assert button is None and not axes

            return ControllerSample(sequence=i,monotonic_time=clock.now,buttons=frozenset([button]) if button else frozenset(),axes=axes)
    class Pico:
        def start(self):calls.append(('pico_start',))
        def stop(self):closed.append('pico')
    def convert(raw,**kw):return SimpleNamespace(joint_pos=raw.q_by_slot[list(VITA_ACTION_TO_COMMAND_SLOTS)],
        joint_vel=np.zeros(30),attitude_valid=True,root_quat_wxyz=np.array([1.,0.,0.,0.]),root_ang_vel_b=np.zeros(3))
    @contextmanager
    def input_context(resource,label):
        try:yield resource
        finally:
            lifecycle.append(label)
            assert status.current_action=='QUIET_PASSIVE', 'Input teardown preceded ownership cancellation'
    monkeypatch.setattr(teleop,'VitaExternalSession',Session)
    monkeypatch.setattr(teleop,'VitaRawStateSource',lambda **kw:input_context(Source(),'state_close'))
    monkeypatch.setattr(teleop,'LocalXboxController',lambda *a:input_context(Controller(),'controller_close'))
    monkeypatch.setattr(teleop,'ControlProcessLock',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(teleop,'JsonlSessionRecorder',lambda *a,**kw:nullcontext(SimpleNamespace(write=events.append)))
    monkeypatch.setattr(teleop,'_wait_for_neutral_controller',lambda *a,**kw:None)
    monkeypatch.setattr(teleop,'robot_state_from_raw',convert)
    monkeypatch.setattr(teleop,'create_pico_source',lambda **kw:Pico())
    monkeypatch.setattr(teleop,'store_latest_pico',lambda *a:None)
    if scenario in ('stale_entry','missing_entry','ownership_loss','ownership_loss_entry','link_loss_entry'):
        with pytest.raises((RuntimeError,TimeoutError),match='LowState|guard rejected|heartbeat'):
            teleop.run_real_teleop(config=config,runtime=Runtime(),suspended_validation=True)
        assert len(published)==(3 if scenario=='ownership_loss' else 0)
    elif scenario in ('stop_during_enter','back_during_enter','idle_during_enter'):
        teleop.run_real_teleop(config=config,runtime=Runtime(),suspended_validation=True)
        assert not published
    elif scenario.startswith('neutral_'):
        teleop.run_real_teleop(config=config,runtime=Runtime(),suspended_validation=True)
        controls=[e for e in events if e.get('record_type')=='control']
        by_sequence={e['controller']['sequence']:e for e in controls}
        # Publication continues on every lost-link sample and after reconnect;
        # BACK is consumed by the pre-publication check at sequence 20.
        assert set(range(11,19)) <= set(by_sequence)
        assert all(b[3]-a[3]>=.019999 for a,b in zip(published,published[1:]))
        # Cleanup may repeat exit; actual ownership is cancelled only by BACK.
        assert [c for c in calls if c[0]=='cancel_active']==[('cancel_active',20)]
        assert all(not by_sequence[i]['controller']['buttons'] and
                   not by_sequence[i]['controller']['axes'] for i in range(11,16))
        if scenario=='neutral_motion':
            assert all(by_sequence[i]['mode']=='motion' and by_sequence[i]['pico_ready']
                       for i in range(11,19))
            # Reconnected A is a real new edge, but cannot change Motion.
            assert by_sequence[16]['controller']['buttons']==[Button.A.value]
            assert by_sequence[16]['mode']=='motion'
        elif scenario=='neutral_velocity':
            assert any(e['mode']=='velocity' and e['velocity_command'][0]>0 for e in controls)
            assert all(by_sequence[i]['mode']=='velocity' and
                       tuple(by_sequence[i]['velocity_command'])==(0.,0.,0.)
                       for i in range(11,19))
        else:
            assert by_sequence[11]['mode']=='motion' and by_sequence[11]['pico_ready']
            assert all(by_sequence[i]['mode']=='velocity' and not by_sequence[i]['pico_ready'] and
                       tuple(by_sequence[i]['velocity_command'])==(0.,0.,0.)
                       for i in range(14,19))
        assert any(e.get('event')=='external_exited' and e.get('reason')=='operator_stop'
                   for e in events)
    else:
        teleop.run_real_teleop(config=config,runtime=Runtime(),suspended_validation=True)
        assert published[0][0][0]==pytest.approx(.01) # Post-entry feedback, not pre-START zero.
        assert np.count_nonzero(published[0][1])==30 # Existing full model-gain hold behavior.
        assert all(b[3]-a[3]>=.019999 for a,b in zip(published,published[1:]))
        assert ('motion',) in calls and ('anchor_reset',) in calls
        velocity=[v[1] for v in calls if v[0]=='velocity']
        assert (0.,0.,0.) in velocity and any(v[0]>0 for v in velocity)
        controls=[e for e in events if e.get('record_type')=='control']
        assert {'stand','velocity','motion'} <= {e['mode'] for e in controls}
        assert all(e['controller']['sequence']<14 for e in controls) # X stops publishing immediately.
        assert any(e.get('event')=='external_exited' and e.get('reason')==('operator_stop' if scenario=='back_active' else 'operator_idle') for e in events)
        entry=next(e for e in events if e.get('event')=='external_running')
        assert entry['state_id']>entry['pre_entry_state_id']
        assert controls[0]['state_id']>entry['state_id']
        assert len([v for v in calls if v[0]=='reset'])==3 # START preparation, B entry, Y/reference-loss return.
    assert status.current_action=='QUIET_PASSIVE'
    assert closed==['session','pico']
    assert lifecycle.index('cancel')<lifecycle.index('state_close')<lifecycle.index('controller_close')
    assert any(v[0]=='exit' for v in calls)
