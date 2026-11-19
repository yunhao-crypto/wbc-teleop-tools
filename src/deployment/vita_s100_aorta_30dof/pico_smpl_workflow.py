"""27000 SMPL policy-online lifecycle with an exclusively no-action driver.

Reuse the reviewed 445700 guard/startup code. No Aorta controller, publisher,
firmware CLI, configuration write or physical-action backend exists here.
Virtual acknowledgements test application sequencing, not SDK wire protocols
or firmware execution. Synthetic feedback is never a physical simulation.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np

from .actor_smpl import SmplReference, digest
from .pico_realtime import PicoFlow, Feedback


def prepare_guard_imports(root):
    """Expose original inert DTOs on the explicit host math-only namespace."""
    site = str(Path(root).resolve()/'venv/lib/python3.10/site-packages')
    if site not in sys.path: sys.path.append(site)
    import locomotion_aorta as sdk
    from locomotion_aorta import body_types
    # Host prepare() intentionally skips __init__ (ARM transport). These are
    # original SDK data classes, not guard replacements or simulated transport.
    for name in ('BodyCmd', 'BodyJointCmd', 'JointError'):
        if not hasattr(sdk, name): setattr(sdk, name, getattr(body_types, name))


def robot_reply(data):
    """JSON snapshot for no-action fixtures; never considered fresh telemetry."""
    def node(value):
        if isinstance(value, dict): return NS(**{k:node(v) for k,v in value.items()})
        if isinstance(value, list): return tuple(node(x) for x in value)
        return value
    return node(data)


class VirtualClock:
    def __init__(self): self.now = 100.
    def monotonic(self): return round(self.now, 9)
    def monotonic_ns(self): return round(self.now*1e9)
    def time_ns(self): return 1_800_000_000_000_000_000+round(self.now*1e9)
    def sleep(self, seconds): self.now += max(0., seconds)


class NoActionDriver:
    """Application-level virtual firmware. It has no network/robot handles.

    Feedback holds the model's virtual-joint neutral pose. It NEVER follows
    predicted PD targets: they may exceed travel by the original PD contract.
    Consequently no physical stability or tracking accuracy is inferred.
    """
    def __init__(self, runtime, reply, clock):
        from locomotion_aorta.types import ControlStatus, ExternalPhase
        self.runtime, self.reply, self.clock = runtime, reply, clock
        self.value = ControlStatus(1, 'PASSIVE', ExternalPhase.IDLE, runtime_id=reply.runtime_id)
        self.requests = []; self.sessions = 0; self.candidates = 0
        self.status_fault = lambda value:value
        self.state_fault = lambda value:value
        self.entry_fault = False; self.exit_fault = False

    @property
    def status(self):
        self.value = replace(self.value, status_seq=self.value.status_seq+1,
            stamp_monotonic_ns=round(self.clock.monotonic()*1e9))
        return self.status_fault(self.value)

    def request(self, action):
        from locomotion_aorta.types import ExternalPhase
        if action not in ('FIXED_STAND', 'CANCEL'): raise ValueError('Unknown virtual action')
        self.requests.append(action)
        self.value = replace(self.value, current_action='FIXED_STAND' if action == 'FIXED_STAND' else 'PASSIVE',
            phase=ExternalPhase.IDLE)

    def wait_state(self, timeout=0):
        from locomotion_aorta.state_arrays import BodyStateArrays
        stamp = self.clock.time_ns()//1000
        q = np.zeros(31, np.float32); q[self.runtime.policy.mapping] = self.runtime.model.contract.default
        imu = np.zeros((1, 13), np.float32); imu[0, 0] = 1
        state = BodyStateArrays(state_id=stamp, q=q, dq=np.zeros(31,np.float32), tau_est=np.zeros(31,np.float32),
            joint_stamps_us=np.full((31,3),stamp,np.uint64), error_code=np.zeros(31,np.uint32),
            imu_values=imu, imu_stamps_us=np.full((1,4),stamp,np.uint64), imu_location=np.array([0]),
            imu_status=np.array([1]), imu_state_present=np.array([True]), protocol_version=3, roster_version=1,
            ts_state_real_us=stamp, ts_state_pub_us=stamp, ts_consumed_us=stamp, attitude_valid=True,
            applied_session_id=0, applied_sequence=0, ts_first_output_us=0, runtime_id=self.reply.runtime_id,
            ts_control_us=stamp, applied_key_valid=False, association_source_type=0,
            received_monotonic_ns=round(self.clock.monotonic()*1e9), received_realtime_ns=self.clock.time_ns())
        return self.state_fault(state)

    def enter_external(self):
        from locomotion_aorta.types import ExternalPhase
        self.sessions += 1
        if not self.entry_fault:
            self.value = replace(self.value, current_action='EXTERNAL', phase=ExternalPhase.RUNNING,
                generation=self.value.generation+1, client_id=11, control_id=22)
        return self.status

    def publish_candidate(self, command): self.candidates += 1  # Never a publisher.

    def exit_external(self, entered):
        from locomotion_aorta.types import ExternalPhase
        if not self.exit_fault:
            self.value = replace(self.value, current_action='PASSIVE', phase=ExternalPhase.IDLE)
        return self.status


class SmplWorkflow:
    def __init__(self, root, reply, robot_config, *, backend='ort', threads=1, clock=None, driver=None, write=lambda row:None):
        prepare_guard_imports(root)
        from .actor_pico_live import load_robot_config, CommandEnvelope, Inputs, StatusGuard
        from .actor_pico_startup import StartupPlan
        self.root = Path(root).resolve(); self.clock = clock or VirtualClock()
        self.write = lambda row:write(dict(row,virtual_firmware=True))
        settings = json.loads((self.root/'config/pico_smpl_workflow.json').read_text())
        if (settings.get('schema') != 'vita_pico_smpl_workflow_no_action_v1'
                or settings.get('motion_enabled') is not False or settings.get('route') != 'pico_smpl'
                or settings.get('entry_policy') != 'policy-online' or settings.get('wrist_mode') != 'stand'
                or settings.get('lead_in_seconds') != .3):
            raise ValueError('Unsupported SMPL lifecycle settings; motion gate cannot be overridden')
        for path, expected in settings['reviewed_guards'].items():
            if digest(self.root/path) != expected: raise ValueError('Original guard changed: '+path)
        self.settings = settings
        self.flow = PicoFlow(self.root,'pico_smpl',backend=backend,threads=threads,wrist_mode='stand')
        self.runtime = self.flow.runtime
        # The original guard interface; do not modify the sealed model config.
        self.guard_runtime = NS(model=self.runtime.model, policy=self.runtime.policy, math=self.runtime.math,
            config={**self.runtime.config, 'pico_max_age_s':.6}, head=None)
        params, self.config_receipt = load_robot_config(Path(robot_config).resolve(), self.guard_runtime)
        self.envelope = CommandEnvelope(self.guard_runtime,reply,params)
        self.robot_config=Path(robot_config).resolve()
        self.plan = StartupPlan(self.guard_runtime,self.envelope,robot_config)
        self.driver = driver or NoActionDriver(self.guard_runtime,reply,self.clock)
        if type(self.driver) is not NoActionDriver:
            raise ValueError('This delivery permits only the no-action driver')
        self.inputs = Inputs(self.guard_runtime,self.driver,None,self.clock)
        self.guard = StatusGuard(self.clock, initial_actions=('PASSIVE','QUIET_PASSIVE','FIXED_STAND',''))
        item = settings['standing_reference']; path = (self.root/item['path']).resolve()
        if not path.is_relative_to(self.root) or digest(path) != item['sha256']:
            raise ValueError('SMPL standing reference hash mismatch')
        self.standing_asset = SmplReference.from_npz(self.runtime.math,path)
        self.locked_standing = None; self.entered = None; self.enter_attempted = False
        self.stage = 'CREATED'; self.stop = threading.Event(); self.events=[]; self.steps=0; self.flow_event_index=0
        self.event('created',state_machine_backend='no_action_virtual_firmware')

    def event(self, kind, **details):
        row = dict(record_type=kind, lifecycle_stage=self.stage, time_ns=round(self.clock.monotonic()*1e9),
            actions_sent=0,control_handles_created=0,**details)
        self.events.append(row); self.write(row)

    def read(self):
        state,q,dq,root,gyro = self.inputs.read(.1)
        if state.runtime_id != self.driver.reply.runtime_id:
            raise RuntimeError('Feedback runtime_id differs from RobotInfo')
        if digest(self.robot_config) != self.config_receipt['sha256']:
            raise RuntimeError('Effective configuration changed; no configuration is written')
        self.envelope.check_measured_pose(state)
        self.guard.check(self.driver.status,self.entered)
        return state,Feedback(q,dq,root,gyro)

    def align_standing(self, root):
        from .actor_smpl import SmplReference
        m = self.runtime.math
        offset = m.yaw_quaternion(m.yaw(root)-m.yaw(self.standing_asset.quat[0]))
        self.locked_standing = SmplReference(m,self.standing_asset.joints@m.rotation(offset).T,
            np.array([m.multiply(offset,q) for q in self.standing_asset.quat]), self.standing_asset.wrists.copy())
        self.flow.standing = self.locked_standing

    def start(self, startup_timeout=30.):
        from .actor_pico_startup import fixed_stand_startup
        if self.stage != 'CREATED': raise RuntimeError('Workflow cannot be restarted')
        # Admission guard runs BEFORE any action intent, including in fixtures.
        _, feedback = self.read()
        self.align_standing(feedback.root)
        self.stage='WAIT_DAMPING'; self.event('startup_begin',plan=self.plan.review)
        ready = fixed_stand_startup(self.plan,self.envelope,self.guard_runtime,
            NS(startup_timeout=startup_timeout),self.driver,self.inputs,self.stop,self.write,self.clock,
            request_action=self.driver.request)
        if not ready: raise InterruptedError('Startup stopped before EXTERNAL')
        self.stage='FIXED_STAND_CONFIRMED'; self.event('virtual_fixed_stand_confirmed')
        from .actor_pico_live import StatusGuard
        self.guard=StatusGuard(self.clock,initial_actions=('FIXED_STAND',''))
        self.runtime.reset()
        # A real SMPL candidate, not a G1 joint reference or zero neural action.
        self.candidate(publish=False)
        self.runtime.reset()  # Prepared preflight must not seed the running policy's history.
        previous=self.driver.status
        self.enter_attempted=True
        ack = self.driver.enter_external()
        from .actor_pico_live import phase
        if (ack.current_action != 'EXTERNAL' or phase(ack) != 'RUNNING' or ack.timeout_hold
                or not ack.generation or not ack.client_id or not ack.control_id
                or ack.generation != previous.generation+1
                or ack.reject_count != previous.reject_count
                or ack.ingress_drop_count != previous.ingress_drop_count
                or ack.last_reject_code != previous.last_reject_code
                or ack.runtime_id != self.driver.reply.runtime_id):
            raise RuntimeError('EXTERNAL acknowledgement/ownership missing; no candidate publish')
        self.entered=ack; self.guard.check(ack,ack)
        self.stage='EXTERNAL_POLICY_LEAD_IN'; self.event('virtual_external_confirmed')
        until=self.clock.monotonic()+self.settings['lead_in_seconds']
        while self.clock.monotonic() < until:
            self.clock.sleep(.02); self.candidate()
        self.stage='ONNX_STAND_WAIT_PICO'; self.event('smpl_stand_ready_before_pico')

    def push(self, body, stamp_ns, received_ns, feedback):
        # Current operator calibration provides the tracking transform, but
        # cannot replace the reviewed stand reference used before connection.
        try: return self.flow.push(body,stamp_ns,received_ns,feedback)
        finally: self.flow.standing=self.locked_standing

    def candidate(self, *, publish=True, pose=None, buttons=None):
        if self.stop.is_set(): raise InterruptedError('Workflow stopped')
        state,feedback=self.read()
        now_ns=self.clock.monotonic_ns()
        if pose is not None: self.push(*pose,feedback)
        if buttons is not None: self.flow.button(buttons,now_ns)
        result=self.flow.step(now_ns,feedback)
        for row in self.flow.events[self.flow_event_index:]:self.write(row)
        self.flow_event_index=len(self.flow.events)
        if not result['inferred']: raise RuntimeError('Standing reference absent')
        self.envelope.check(result['command'])
        self.inputs.fresh(state)
        self.guard.check(self.driver.status,self.entered)
        if publish:
            if self.entered is None: raise RuntimeError('EXTERNAL required before candidate sink')
            self.driver.publish_candidate(result['command'])
        if self.stage in ('ONNX_STAND_WAIT_PICO','PICO_TRACKING','PICO_RETURNING_STAND'):
            next_stage = {'tracking':'PICO_TRACKING','returning_stand':'PICO_RETURNING_STAND'}.get(
                self.flow.mode,'ONNX_STAND_WAIT_PICO')
            if next_stage != self.stage:
                self.stage=next_stage; self.event('policy_reference_mode_changed',reference_mode=self.flow.mode)
        self.steps+=1
        return result

    def close(self):
        from .actor_pico_live import phase
        if self.stage == 'CLOSED': return
        if self.entered is not None:
            # Never cancel an unknown or changed session.
            self.guard.check(self.driver.status,self.entered)
            value=self.driver.exit_external(self.entered)
            if value.current_action != 'PASSIVE' or phase(value) != 'IDLE':
                self.stage='EXIT_FAILED'; self.event('virtual_exit_unconfirmed')
                raise RuntimeError('EXTERNAL exit did not confirm PASSIVE/IDLE')
            self.event('virtual_external_exit_confirmed',current_action='PASSIVE')
        elif self.enter_attempted:
            # An ambiguous entry must not be treated as a clean exit.
            self.stage='EXIT_FAILED'; self.event('virtual_entry_result_unknown')
            raise RuntimeError('EXTERNAL entry result unknown; exit not verified')
        elif self.driver.requests:
            from .actor_pico_startup import cancel_startup
            failures=[]
            def log(row):
                self.write(row)
                if row['record_type']=='startup_exit_failure': failures.append(row)
            cancel_startup(self.driver,self.driver.request,self.clock,log,self.driver.status.generation)
            if failures: raise RuntimeError('Startup cancellation failed')
        self.flow.stop(round(self.clock.monotonic()*1e9)); self.stage='CLOSED'; self.event('closed')

    def summary(self):
        return dict(**self.flow.summary(), lifecycle_stage=self.stage,
            lifecycle_backend='application_virtual_firmware_no_robot_transport',
            simulated_action_requests=self.driver.requests, simulated_external_sessions=self.driver.sessions,
            candidate_sink_steps=self.driver.candidates, workflow_inference_steps=self.steps,
            robot_config_readonly=self.config_receipt, real_robot_actions_sent=0,
            motion_enabled=False, closed_loop=False, firmware_wire_protocol_verified=False,
            physical_standing_validated=False, standing_reference=self.settings['standing_reference'])


def validate_snapshot_admission(snapshot, clock):
    """Inspect a stored status only; do not fabricate fresh live feedback."""
    from .actor_pico_live import StatusGuard
    return StatusGuard(clock,initial_actions=('PASSIVE','QUIET_PASSIVE','FIXED_STAND','')).check(robot_reply(snapshot))


def replay_workflow(workflow, path, expected, events, *, tail_seconds=1.5, on_result=lambda result,now:None):
    """Historical poses on their ORIGINAL device/receipt clock intervals.

    A single common shift maps both clocks onto the virtual lifecycle clock;
    delays, gaps and regressions remain intact. Buttons are explicitly synthetic.
    """
    if digest(path) != expected: raise ValueError('Workflow replay hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        body=z['body_poses'].copy(); stamps=z['sample_times_ns'].copy(); received=z['receive_monotonic_ns'].copy()
        metadata=json.loads(z['metadata'].item())
    n=len(body)
    if (body.shape != (n,24,7) or n < 2 or stamps.shape != (n,) or received.shape != (n,)
            or stamps.dtype != np.int64 or received.dtype != np.int64 or not np.isfinite(body).all()
            or np.any(np.diff(received)<=0) or metadata.get('schema') != 'pico_body24_record_v1'
            or metadata.get('complete') is not True): raise ValueError('Invalid workflow replay')
    times=[e['time_s'] for e in events]
    if not np.isfinite(times).all() or any(t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Invalid synthetic button scenario times')
    for e in events:
        if not np.isfinite([e['left'],e['right']]).all() or not 0 <= e['left'] <= 1 or not 0 <= e['right'] <= 1:
            raise ValueError('Invalid synthetic trigger values')
    workflow.clock.sleep(.02)
    start=round(workflow.clock.monotonic()*1e9)
    shift=start-int(received[0]); index=button_index=0; left=right=0.
    end=int(received[-1])+shift+int(tail_seconds*1e9)
    for now in range(start,end+1,20_000_000):
        workflow.clock.now=now*1e-9
        _, feedback=workflow.read()
        while index<n and int(received[index])+shift<=now:
            workflow.push(body[index],int(stamps[index])+shift,int(received[index])+shift,feedback); index+=1
        elapsed=(now-start)*1e-9
        while button_index<len(events) and events[button_index]['time_s']<=elapsed:
            left,right=events[button_index]['left'],events[button_index]['right']; button_index+=1
        result=workflow.candidate(buttons=(True,now,0.,left,right))
        on_result(result,now)
    if digest(path) != expected: raise ValueError('Workflow replay changed during validation')
    return metadata
