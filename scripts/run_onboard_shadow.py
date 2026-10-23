#!/usr/bin/env python3
"""50 Hz local state/Pico/controller -> policies -> inert trace. NO ACTION API."""
import argparse
import dataclasses
import gc
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np

from deployment.common.controller_input import Button, ControllerSample, VelocityLimits
from deployment.common.xbox_joystick import LocalXboxController
from deployment.vita_s100_aorta_30dof.onboard_backend import NativeBackend, NativeMotionRunner, NativeVelocityRunner
from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime
from deployment.vita_s100_aorta_30dof.onboard_shadow import OnboardShadowFlow, ReadOnlyStateSource
from deployment.vita_s100_aorta_30dof.onboard_state import decode_policy_lowstate_buffer
from deployment.vita_s100_aorta_30dof.state_adapter import decode_humanoid_lowstate_buffer
from deployment.vita_s100_aorta_30dof.onboard_output import InertCommandSink, load_output_envelopes
from deployment.vita_s100_aorta_30dof.live_shadow import live_alignment_quaternion
from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState, VITA_CONTROLLER_JOINT_NAMES


def warm_native_policies(backend, root, iterations=20):
    """Touch both resident inference paths before any live subscription."""
    started=time.monotonic_ns()
    motion=np.asarray(np.load(root/'reference/policy_motion_recorded_8192/observations.npy',
        mmap_mode='r',allow_pickle=False)[:iterations],dtype=np.float32).copy()
    velocity=np.asarray(np.load(root/'reference/policy_velocity_recorded_1536/observations.npy',
        mmap_mode='r',allow_pickle=False)[:iterations],dtype=np.float32).copy()
    if len(motion)!=iterations or len(velocity)!=iterations:
        raise ValueError('Insufficient recorded observations for native warmup')
    for i in range(iterations):
        backend.run_velocity(velocity[i])
        backend.run_motion(motion[i])
    return dict(iterations_per_policy=iterations,elapsed_ms=(time.monotonic_ns()-started)/1e6,
        scope='Original recorded observations; outputs discarded; all sequence state reset before live inputs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--velocity-backend',choices=['ort','hbm'],default='ort')
    p.add_argument('--controller',choices=['scripted','joystick'],default='scripted')
    p.add_argument('--joystick',type=Path,default=Path('/dev/input/js0'))
    p.add_argument('--pico-mode',choices=['none','live'],default='live')
    a=p.parse_args();assert 1<=a.seconds<=600
    backend=None;source=None;pico=None;controller=None
    rows=[];raw_payloads=[];command_payloads=[];failure=None
    gc_was_enabled=gc.isenabled();warmup=None;gc_disabled_for_loop=False
    try:
        cfg=a.root/'src/deployment/vita_s100_aorta_30dof/config'
        alignment=live_alignment_quaternion(json.loads((cfg/'vita02a_imu_alignment.json').read_text()))
        backend=NativeBackend(a.root/'build/libvita_backend.so',a.root/'outputs/model_runtime_manifest.json',
                              velocity_backend=a.velocity_backend)
        runtime=OnboardDualPolicyRuntime(contracts_path=a.root/'artifacts/policy_contracts.json',
                skeleton_path=a.root/'artifacts/preprocessing_skeleton.npz',
                motion_runner=NativeMotionRunner(backend),velocity_runner=NativeVelocityRunner(backend))
        warmup=warm_native_policies(backend,a.root)
        runtime.reset()
        flow=OnboardShadowFlow(runtime,limits=VelocityLimits(forward=.90,backward=.70,lateral=0.,yaw=.80))
        envelopes=load_output_envelopes(a.root/'artifacts/output_envelopes.json',
            expected_sha256='595b0d1fb26739e8fd3b0874dbc70bdd89ad8d772b5103adce199433dcaf03e0')
        sink=InertCommandSink(envelopes['velocity'])
        if a.pico_mode!='none':
            sys.path.insert(0,str(a.root/'src/deployment/unitree_g1_ros2_29dof/src'))
            from humanoid_policy.local_smpl_source import LocalPicoSmplSource
            pico=LocalPicoSmplSource(logger=logging.getLogger('pico'));pico.start()
        if a.controller=='joystick':
            controller=LocalXboxController(a.joystick);controller.__enter__()
        source=ReadOnlyStateSource()
        timeout=time.monotonic()+10
        while source.latest is None:
            if time.monotonic()>timeout:raise TimeoutError('No local LowState')
            time.sleep(.01)
        initial_payload,initial_received=source.latest
        initial_raw=decode_humanoid_lowstate_buffer(initial_payload,received_monotonic_ns=initial_received)
        hold_q=np.asarray(initial_raw.q_by_slot[:42],np.float64)
        initial_guard=sink.arm(hold_q)
        if not initial_guard.allowed:raise RuntimeError('Inert initial guard: '+','.join(initial_guard.reasons))
        gc.collect()
        gc.disable();gc_disabled_for_loop=True
        started=time.monotonic();deadline=started;index=0
        while time.monotonic()-started<a.seconds:
            wait=deadline-time.monotonic()
            if wait>0:time.sleep(wait)
            begin=time.monotonic_ns();wall=time.time()
            payload,received=source.latest
            if (begin-received)>100_000_000:raise RuntimeError('LowState older than100ms')
            values=decode_policy_lowstate_buffer(payload,received_monotonic_ns=received,
                    base_from_hip_quaternion_wxyz=alignment)
            state=VitaRobotState(VITA_CONTROLLER_JOINT_NAMES,values.robot_dof_pos,values.robot_dof_vel,
                    values.robot_root_quat_wxyz,values.robot_root_angvel_local,
                    state_id=values.state_id,received_monotonic_ns=received,attitude_valid=True)
            state_done=time.monotonic_ns()
            pico_input=None
            if pico is not None:
                sample=pico.step()
                if sample is not None:
                    body,frame,metadata=sample
                    pico_input=(body,float(metadata['timestamp_ns'])*1e-9,frame,
                                float(metadata['timestamp_realtime']))
            if controller is not None:
                control=controller.sample(timeout=0.)
            else:
                # Deliberately synthetic button events; do not claim physical controller validation.
                button={0:Button.START,2:Button.A}.get(index)
                if index>=100 and index%100==0:button=Button.B
                if index>=150 and index%100==50:button=Button.Y
                control=ControllerSample(index,time.monotonic(),frozenset([button]) if button else frozenset())
            inputs_done=time.monotonic_ns()
            step,info=flow.tick(state=state,controller=control,pico=pico_input,wall_time=wall)
            inert=None
            if step is not None:
                inert=sink.process(step,hold_q=hold_q,period_s=.02,
                    now_monotonic_ns=time.monotonic_ns(),lowstate_received_monotonic_ns=received,
                    state_id=state.state_id,attitude_valid=True,
                    envelope=envelopes['motion' if info['mode']=='motion' else 'velocity'])
                info['timing_ms'].update(inert.timing_ms)
            finished=time.monotonic_ns()
            transition=info.pop('transition')
            row=dict(index=index,tick_wall_time=wall,received_monotonic_ns=received,state_id=state.state_id,
                     scheduled_monotonic=deadline,started_monotonic_ns=begin,
                     callback_to_output_ms=(finished-received)/1e6,
                     state_decode_ms=(state_done-begin)/1e6,
                     input_poll_ms=(inputs_done-state_done)/1e6,
                     compute_ms=(finished-begin)/1e6,
                     scheduling_lateness_ms=max(0.,begin/1e6-deadline*1000),
                     **info)
            row['transition']=None if transition is None else dataclasses.asdict(transition)
            row['controller']=dataclasses.asdict(control)
            row['controller']['buttons']=[x.value for x in control.buttons]
            row['state']={k:getattr(state,k).tolist() for k in ('joint_pos','joint_vel','root_quat_wxyz','root_ang_vel_b')}
            row['pico']=None if pico_input is None else [pico_input[0].tolist(),*pico_input[1:]]
            if step is not None:
                row['raw_action']=runtime.last_raw_action.tolist()
                row['observation']=runtime.last_observation.reshape(-1).tolist()
                row['target_joint_pos']=step.target_joint_pos.tolist()
                row['inert_output']={'guard_allowed':inert.decision.allowed,
                    'guard_reasons':inert.decision.reasons,'ownership_scope':inert.ownership_scope,
                    'target_q42':inert.target_q.tolist(),'kp42':inert.kp.tolist(),'kd42':inert.kd.tolist(),
                    'conditioning':inert.conditioning,'payload_bytes':0 if inert.payload is None else len(inert.payload)}
            rows.append(row);raw_payloads.append(payload)
            command_payloads.append(b'' if inert is None or inert.payload is None else inert.payload)
            if inert is not None and not inert.decision.allowed:
                raise RuntimeError('Inert output guard: '+','.join(inert.decision.reasons))
            index+=1;deadline+=.02
            # Skip missed periods rather than catching up with action-rate bursts.
            while deadline<time.monotonic():deadline+=.02
    except Exception as exc:
        failure=repr(exc)
    finally:
        if gc_was_enabled:gc.enable()
        else:gc.disable()
        final_state=None
        cleanup_errors=[]
        if backend is not None:
            try:final_state=backend.state()
            except Exception as exc:cleanup_errors.append(repr(exc))
        for resource,method in ((backend,'close'),(source,'close'),(pico,'stop'),(controller,'close')):
            if resource is not None:
                try:getattr(resource,method)()
                except Exception as exc:cleanup_errors.append(repr(exc))
        if cleanup_errors:failure=(failure or '')+'; cleanup: '+'; '.join(cleanup_errors)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    with a.out.with_suffix('.lowstate.bin').open('wb') as f:
        for raw in raw_payloads:f.write(len(raw).to_bytes(4,'little'));f.write(raw)
    with a.out.with_suffix('.inert_lowcmd.bin').open('wb') as f:
        for raw in command_payloads:f.write(len(raw).to_bytes(4,'little'));f.write(raw)
    def summary(field):
        x=np.array([r[field] for r in rows])
        return None if not len(x) else dict(mean=float(x.mean()),p99=float(np.percentile(x,99)),max=float(x.max()),over20ms=int((x>20).sum()))
    starts=np.asarray([r['started_monotonic_ns'] for r in rows],np.float64)
    intervals=np.diff(starts)/1e6
    cadence=None if not len(intervals) else dict(
            measured_hz=float(1e9*(len(starts)-1)/(starts[-1]-starts[0])),
            interval_mean_ms=float(intervals.mean()),interval_p99_ms=float(np.percentile(intervals,99)),
            interval_max_ms=float(intervals.max()),intervals_over21ms=int((intervals>21).sum()))
    result=dict(action_sent=False,frames=len(rows),failure=failure,backend_state=final_state,
                pico_source=a.pico_mode,
                controller_source=a.controller,
                modes={mode:sum(r['mode']==mode for r in rows) for mode in {r['mode'] for r in rows}},
                state_callback_frames=0 if source is None else source.received,
                compute_ms=summary('compute_ms'),callback_to_output_ms=summary('callback_to_output_ms'),
                scheduling_lateness_ms=summary('scheduling_lateness_ms'),
                requested_policy_hz=50.,actual_cadence=cadence,
                startup_warmup=warmup,cyclic_gc_disabled_during_timed_loop=gc_disabled_for_loop,
                inert_serialized_frames=sum(bool(x) for x in command_payloads),
                output_scope='42-slot conditioning and in-memory LowCmd bytes; simulated ownership only, no lease/header/publication',
                timing_scope='Local callback receipt to inert output; source timestamp clock and physical sampling latency not verified. File serialization excluded; in-memory trace bookkeeping affects scheduling.')
    a.out.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if failure is None else 2


if __name__=='__main__':raise SystemExit(main())
