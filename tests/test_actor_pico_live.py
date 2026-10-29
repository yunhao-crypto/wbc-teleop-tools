"""Real SDK session/math with simulated robot/Pico I/O; no network or motors."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_pico_live import (
    run_live, load_robot_config, require_peer_profile, CommandEnvelope, StatusGuard,
    next_policy_deadline,
)


class Clock:
    def __init__(self): self.now = 100.
    def monotonic(self): return round(self.now, 9)
    def time_ns(self): return 1_800_000_000_000_000_000+round(self.now*1e9)
    def sleep(self, seconds): self.now += max(0., seconds)


def robot_config():
    config = dict(parallel_ankle=dict(pd=dict(strategy='virtual_pd'),
                geometry=dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235)),
        actions=dict(EXTERNAL=dict(generator='builtin::external', params=dict(
            ankle_virtual_pd_intent=True, command_timeout_ms=1000,
            recover_valid_frames=1, interpolation=dict(min_period_ms=2, max_period_ms=100),
            safe_hold=dict(kp=[10.]*42, kd=[1.]*42)))))
    config['actions']['FIXED_STAND'] = dict(generator='builtin::keyframe', params=dict(
        keyframes=[dict(timestamp=0,positions=[0.]*42,kp=[10.]*42,kd=[1.]*42)]))
    for name in ('PASSIVE','QUIET_PASSIVE'):
        config['actions'][name] = dict(generator='builtin::keyframe', params=dict(
            kp=0.,keyframes=[dict(timestamp=0,positions=[0.]*42,kd=5.),
                            dict(timestamp=2,positions=[0.]*42,kd=0.)]))
    return config


class Pico:
    """In-process PicoReferenceProcess(buttons=False): the real GMR and reference window."""
    def __init__(self, runtime, clock):
        self.runtime, self.clock, self.fault, self.last_stamp = runtime, clock, None, None
        self.stream, self.anchor, self.previous = None, None, dict(state='idle')

    def connect(self, root):
        if self.stream is not None: return False
        from deployment.vita_s100_aorta_30dof.actor_pico_worker import ReferenceStream
        r = self.runtime
        self.stream = ReferenceStream(r.root, r.config, r.model.contract, r.math)
        self.anchor, self.previous = tuple(root), dict(state='waiting_pico')
        return True

    def snapshot(self):
        sample = self.step() if self.stream is not None else None
        if sample is not None:
            body, _, metadata = sample
            self.previous = self.stream.push(body, metadata, self.anchor)
        return dict(self.previous)

    def step(self):
        if self.fault: return self.fault()
        stamp = round(self.clock.monotonic()*1e9)
        if stamp == self.last_stamp: return None
        self.last_stamp = stamp
        body = np.zeros((24, 7), np.float32); body[:, 6] = 1
        return body, 1, dict(timestamp_ns=stamp,
                            timestamp_monotonic=self.clock.monotonic())


class ManualPico:
    def __init__(self, runtime, clock):
        self.runtime, self.clock = runtime, clock
        self.connected = False
        self.fault = None

    def connect(self, root):
        if self.connected: return False
        self.connected = True
        return True

    def snapshot(self):
        if self.fault: return self.fault()
        if not self.connected: return dict(state='idle')
        r = self.runtime
        n = r.reference.steps
        return dict(state='ready', received_monotonic=self.clock.monotonic(),
            q=np.repeat(r.model.contract.default[None], n, axis=0),
            base=np.zeros((n, 3)), quat=np.tile([1., 0., 0., 0.], (n, 1)),
            retarget_ms=.1, gmr_task_error=0.)


class Loopback:
    """Feed PREPARE/COMMIT/CANCEL ACKs to the shipped PullController.

    This is a mock server, not evidence of firmware or physical safety.
    """
    def __init__(self, runtime, clock):
        from locomotion_aorta import ControlStatus, ExternalPhase
        from locomotion_aorta.body_types import (BodyJointCapability, BodyInfo, RobotInfo,
                                                 GetRobotInfoResponse)
        self.runtime, self.clock = runtime, clock
        self.callbacks, self.requests, self.commands = {}, [], []
        self.after_publish = lambda command: None
        self.state_fault, self.exit_fault, self.entry_fault = None, False, False
        self.exit_action = 'PASSIVE'
        self.status_fault = lambda s:s
        self.closed = False
        self.value = ControlStatus(1, 'PASSIVE', ExternalPhase.IDLE)
        caps = tuple(BodyJointCapability(i, 1, False, i != 14, 31, 31,
                -4., 4., -25., 25., -100., 100., 0., 500., 0., 50.) for i in range(31))
        body = BodyInfo(1, 31, tuple(runtime.policy.order), caps, ())
        self.reply = GetRobotInfoResponse(0, RobotInfo(1, 1, body, ()), '')

    def subscribe(self, topic, callback):
        self.callbacks.setdefault(topic, []).append(callback)
        return NS(close=lambda:None)

    def get_robot_info(self, timeout): return self.reply

    def poll_control_status(self):
        from locomotion_aorta.transport import CONTROL_STATUS_TOPIC
        self.value = replace(self.value, status_seq=self.value.status_seq+1,
                             stamp_monotonic_ns=round(self.clock.monotonic()*1e9))
        value = self.status_fault(self.value)
        for callback in self.callbacks.get(CONTROL_STATUS_TOPIC, []): callback(value)
        return value

    def wait_state(self, timeout=0):
        from locomotion_aorta.state_arrays import BodyStateArrays
        stamp = self.clock.time_ns()//1000
        q = np.zeros(31, np.float32)
        q[self.runtime.policy.mapping] = self.runtime.model.contract.default
        # Simulated robot follows targets only to exercise feedback plumbing.
        if self.commands: q[:] = [j.q for j in self.commands[-1].joints]
        imu = np.zeros((1, 13), np.float32); imu[0, 0] = 1
        state = BodyStateArrays(state_id=stamp, q=q, dq=np.zeros(31, np.float32),
            tau_est=np.zeros(31, np.float32), joint_stamps_us=np.full((31, 3), stamp, np.uint64),
            error_code=np.zeros(31, np.uint32), imu_values=imu,
            imu_stamps_us=np.full((1, 4), stamp, np.uint64), imu_location=np.array([0]),
            imu_status=np.array([1]), imu_state_present=np.array([True]), protocol_version=2,
            roster_version=1, ts_state_real_us=stamp, ts_state_pub_us=stamp,
            ts_consumed_us=stamp, attitude_valid=True, applied_session_id=0,
            applied_sequence=0, ts_first_output_us=0,
            received_monotonic_ns=round(self.clock.monotonic()*1e9), received_realtime_ns=self.clock.time_ns())
        return self.state_fault(state) if self.state_fault else state

    def publish(self, topic, message):
        from locomotion_aorta import ControlAck, ControlOperation, ExternalPhase
        from locomotion_aorta.transport import CONTROL_REQUEST_TOPIC, CONTROL_ACK_TOPIC, BODY_CMD_TOPIC
        if topic == BODY_CMD_TOPIC:
            self.commands.append(message)
            self.after_publish(message)
            return
        if topic != CONTROL_REQUEST_TOPIC: raise AssertionError(topic)
        self.requests.append(message)
        lease = 'mock-lease'
        op = message.operation
        if op == ControlOperation.COMMIT:
            if self.entry_fault: raise RuntimeError('injected entry failure')
            self.value = replace(self.value, current_action='EXTERNAL', phase=ExternalPhase.RUNNING,
                                 lease_id=lease, client_id=message.client_id, generation=7)
        if op == ControlOperation.CANCEL:
            if self.exit_fault: raise RuntimeError('injected exit failure')
            self.value = replace(self.value, current_action=self.exit_action, phase=ExternalPhase.IDLE)
        ack = ControlAck(message.request_id, op, True, lease_id=lease)
        for callback in self.callbacks.get(CONTROL_ACK_TOPIC, []): callback(ack)
        self.poll_control_status()

    def close(self): self.closed = True


class LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort')

    def setUp(self):
        from locomotion_aorta.pull import PullController
        self.runtime.reset_inputs()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.config = Path(self.tmp.name)/'locomotion.yaml'
        self.config.write_text(json.dumps(robot_config()))  # JSON is YAML, no parser stubs.
        self.clock, self.stop, self.rows = Clock(), threading.Event(), []
        self.transport = Loopback(self.runtime, self.clock)
        self.controller = PullController(group='default', client_id='mock-client', _transport=self.transport)
        self.addCleanup(self.controller.close)
        self.pico = Pico(self.runtime, self.clock)
        self.args = NS(robot_config=self.config, robot_id='TEST-ONLY', startup_timeout=5.,
                       seconds=3.12, stand_only=False)

    def run_trial(self, **kwargs):
        write = kwargs.pop('write', lambda row, obs=None:self.rows.append(row))
        with redirect_stdout(io.StringIO()):
            return run_live(self.runtime, self.args, write,
                self.controller, self.pico, self.stop, service_alive=lambda:True,
                clock=self.clock, **kwargs)

    def assert_cancelled(self):
        self.assertEqual(self.controller.status.current_action, 'PASSIVE')
        self.assertIn('CANCEL', [r.operation.value for r in self.transport.requests])

    def test_real_sdk_protocol_and_full_commands_through_prepare_and_tracking(self):
        # GMR belongs to the worker; the control loop only consumes snapshots.
        with patch.object(self.runtime, 'push_pico', side_effect=AssertionError('control-side GMR')):
            result = self.run_trial()
        self.assertTrue(result['exit_confirmed'])
        self.assertGreater(result['actions_sent'], 150)
        self.assertEqual([r.operation.value for r in self.transport.requests], ['PREPARE', 'COMMIT', 'CANCEL'])
        self.assertTrue(all(r.requested_lease_ms == 0 for r in self.transport.requests))
        self.assertEqual([c.sequence for c in self.transport.commands], list(range(1, result['actions_sent']+1)))
        self.assertTrue(all(c.session_id == 7 and c.input_state_id > 0 for c in self.transport.commands))
        self.assertTrue(any(r.get('stage') == 'pico_tracking' for r in self.rows))
        self.assertTrue(all(len(c.joints) == 31 for c in self.transport.commands))
        self.assert_cancelled()

    def test_stand_only_never_switches_to_pico_tracking(self):
        self.args.stand_only = True
        self.pico.step = lambda:(_ for _ in ()).throw(AssertionError('Pico must not be polled'))
        result = self.run_trial()
        self.assertTrue(result['exit_confirmed'])
        self.assertFalse(any(r.get('stage') == 'pico_tracking' for r in self.rows))

    def manual_trial(self, request_after=155):
        self.args.manual_pico = True
        self.args.seconds = 3.4
        worker = ManualPico(self.runtime, self.clock)
        def command():
            n = len(self.transport.commands)
            if n == 50: return 'connect'
            if n == request_after: return 'track'
            return None
        return worker, command

    def test_manual_mode_enters_stand_without_any_pico(self):
        worker, _ = self.manual_trial()
        result = self.run_trial(manual_pico=worker)
        self.assertTrue(result['exit_confirmed'])
        self.assertGreater(result['actions_sent'], 150)
        self.assertFalse(worker.connected)
        self.assertFalse(any(r.get('stage') == 'pico_tracking' for r in self.rows))

    def test_manual_connect_does_not_switch_until_operator_requests(self):
        worker, command = self.manual_trial(request_after=9999)
        self.run_trial(manual_pico=worker, poll_command=command)
        self.assertTrue(worker.connected)
        self.assertFalse(any(r.get('stage') == 'pico_tracking' for r in self.rows))

    def test_gmr_heading_is_aligned_on_engagement_and_reference_is_recorded(self):
        worker, command = self.manual_trial()
        original_snapshot = worker.snapshot
        def snapshot():
            row = original_snapshot()
            if row['state'] == 'ready':
                row['quat'] = np.tile(self.runtime.math.yaw_quaternion(.8),
                                      (self.runtime.reference.steps, 1))
            return row
        worker.snapshot = snapshot
        self.run_trial(manual_pico=worker, poll_command=command)
        event = next(r for r in self.rows if r.get('record_type') == 'tracking_started')
        self.assertAlmostEqual(event['reference_yaw_offset'], -.8, places=6)
        row = next(r for r in self.rows if r.get('stage') == 'pico_tracking')
        self.assertAlmostEqual(row['pico_reference']['yaw_offset'], -.8, places=6)
        self.assertEqual(len(row['pico_reference']['q']), self.runtime.reference.steps)

    def test_manual_switch_keeps_one_session_and_policy_history(self):
        worker, command = self.manual_trial()
        # A reset on switching would clear this deque and cause this audit to fail.
        lengths=[]
        original=self.runtime.policy.observe
        def observe(*a, **kw):
            lengths.append(len(self.runtime.policy.history))
            return original(*a, **kw)
        with patch.object(self.runtime.policy, 'observe', observe):
            result=self.run_trial(manual_pico=worker, poll_command=command)
        self.assertTrue(result['exit_confirmed'])
        self.assertTrue(any(r.get('stage') == 'pico_tracking' for r in self.rows))
        self.assertTrue(all(n == self.runtime.model.contract.history_length for n in lengths[-20:]))
        self.assertEqual([r.operation.value for r in self.transport.requests], ['PREPARE','COMMIT','CANCEL'])
        self.assertEqual([c.sequence for c in self.transport.commands], list(range(1,result['actions_sent']+1)))

    def test_early_manual_switch_is_rejected_and_not_queued(self):
        worker, command=self.manual_trial(request_after=100)
        self.run_trial(manual_pico=worker, poll_command=command)
        self.assertTrue(any(r.get('record_type')=='pico_switch_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

    def test_pico_loss_after_manual_switch_cancels_instead_of_returning_to_stand(self):
        worker, command=self.manual_trial()
        def fault_after_publish(c):
            if len(self.transport.commands)>160:
                worker.fault=lambda:dict(state='failed',error='injected disconnect')
        self.transport.after_publish=fault_after_publish
        with self.assertRaisesRegex(ValueError,'not ready'):
            self.run_trial(manual_pico=worker,poll_command=command)
        self.assert_cancelled()
        stages=[r['stage'] for r in self.rows if 'stage' in r]
        self.assertEqual(stages[-1],'pico_tracking')

    def test_holomotion_600ms_budget_accepts_300ms_old_reference_and_sparse_frames(self):
        worker,command=self.manual_trial()
        original=worker.snapshot
        def snapshot():
            value=original()
            if value['state']=='ready':
                value['received_monotonic']-=.3
                value['source_gap_s']=.122
            return value
        with patch.object(worker,'snapshot',snapshot):
            self.run_trial(manual_pico=worker,poll_command=command)
        tracked=[r for r in self.rows if r.get('stage')=='pico_tracking']
        self.assertTrue(tracked)
        self.assertAlmostEqual(tracked[0]['pico_age_s'],.3)
        self.assertAlmostEqual(tracked[0]['source_gap_s'],.122)
        # Older than pico_hold_age_s: the current frame is held with zero velocity.
        self.assertTrue(all(r['reference_hold'] for r in tracked))
        q=np.asarray(tracked[0]['pico_reference']['q'])
        np.testing.assert_array_equal(q,np.repeat(q[self.runtime.reference.current_index][None],len(q),0))
        self.assert_cancelled()

    def test_reference_within_hold_age_is_not_held(self):
        worker,command=self.manual_trial()
        original=worker.snapshot
        def snapshot():
            value=original()
            if value['state']=='ready':
                value['received_monotonic']-=.2
                value['q']=value['q']+np.linspace(0.,.1,len(value['q']))[:,None]
            return value
        with patch.object(worker,'snapshot',snapshot):
            self.run_trial(manual_pico=worker,poll_command=command)
        tracked=[r for r in self.rows if r.get('stage')=='pico_tracking']
        self.assertTrue(tracked)
        self.assertFalse(any(r['reference_hold'] for r in tracked))
        q=np.asarray(tracked[0]['pico_reference']['q'])
        self.assertGreater(np.ptp(q[:,0]),.09)

    def test_hold_age_must_lie_below_pico_maximum_age(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_runtime as module
        config=json.loads((ROOT/module.POLICY_CONFIGS['sonic']).read_text())
        for value in (0.,.6,.7,float('nan'),True,None):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/'config').mkdir()
                (Path(tmp)/'config/actor_pico.json').write_text(json.dumps(dict(config,pico_hold_age_s=value)))
                with self.assertRaisesRegex(ValueError,'hold age'):module.load_actor_config(tmp)

    def test_input_age_settings_match_original_holomotion_deployment(self):
        import yaml
        from deployment.vita_s100_aorta_30dof.actor_pico_live import STATE_AGE
        original=yaml.safe_load((ROOT/'src/deployment/vita_s100_aorta_30dof/config/vita02a_s100.yaml').read_text())
        self.assertEqual(self.runtime.config['pico_max_age_s'],original['runtime']['pico_max_age'])
        self.assertEqual(STATE_AGE,original['runtime']['state_max_age'])


    def test_initial_incomplete_reference_rejects_early_follow_request_without_queuing(self):
        worker,command=self.manual_trial()
        original=worker.snapshot
        def snapshot():
            value=original();count=len(self.transport.commands)
            if count>=150:
                if count<160:value['state']='collecting'
            return value
        with patch.object(worker,'snapshot',snapshot):
            self.run_trial(manual_pico=worker,poll_command=command)
        self.assertTrue(any(r.get('record_type')=='pico_switch_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))
        self.assert_cancelled()

    def capture_fixture(self, stamps, bodies=None):
        import queue
        from unittest.mock import Mock
        from deployment.vita_s100_aorta_30dof import actor_pico_worker as worker,pico_input
        commands=queue.Queue();commands.put((1.,0.,0.,0.));output=queue.Queue(maxsize=128)
        (ROOT/'outputs').mkdir(exist_ok=True)  # Same precondition as the CLI log setup.
        stop=threading.Event();body=np.zeros((24,7),np.float32);body[:,6]=1.
        samples=iter(stamps)
        poses=iter(bodies) if bodies is not None else None
        def step():
            try:stamp=next(samples)
            except StopIteration:stop.set();return None
            return body if poses is None else next(poses),0,dict(timestamp_ns=stamp,timestamp_monotonic=stamp*1e-9)
        pico=Mock();pico.step.side_effect=step
        service=Mock();service.poll.return_value=None
        with (patch.object(worker.os,'setsid'),patch.object(worker.subprocess,'Popen',return_value=service),
              patch.object(pico_input,'create_pico_source',return_value=pico)):
            worker._capture(str(ROOT),self.runtime.config,self.runtime.model.contract,
                self.runtime.policy.lower,self.runtime.policy.upper,commands,output,stop)
        rows=[]
        while not output.empty():rows.append(output.get_nowait())
        pico.stop.assert_called_once();service.terminate.assert_called_once()
        return rows

    def test_capture_source_gap_does_not_clear_ready_reference(self):
        stamps=[1_000_000_000+i*20_000_000 for i in range(11)]
        stamps += [1_500_000_000+i*20_000_000 for i in range(11)]
        rows=self.capture_fixture(stamps)
        self.assertFalse(any(r['state']=='failed' for r in rows))
        ready=next(i for i,r in enumerate(rows) if r['state']=='ready')
        self.assertTrue(all(r['state']=='ready' for r in rows[ready:]))
        self.assertTrue(any(abs(r.get('source_gap_s',0)-.3)<1e-6 for r in rows))

    def test_capture_clock_regression_remains_terminal_and_is_not_resynced(self):
        rows=self.capture_fixture([1_000_000_000,1_020_000_000,1_010_000_000])
        self.assertEqual(rows[-1]['state'],'failed')
        self.assertIn('strictly increase',rows[-1]['error'])

    def test_capture_large_fit_error_remains_diagnostic_without_rebuilding(self):
        neutral=np.zeros((24,7),np.float32);neutral[:,6]=1.
        bad=neutral.copy();bad[18,3:7]=[np.sin(.7),0.,0.,np.cos(.7)]
        stamps=[1_000_000_000+i*20_000_000 for i in range(23)]
        rows=self.capture_fixture(stamps,[neutral]*11+[bad]+[neutral]*11)
        self.assertFalse(any(r['state'] in ('rejected','failed') for r in rows))
        self.assertTrue(any(r.get('gmr_task_error',0)>.6 for r in rows))
        ready=next(i for i,r in enumerate(rows) if r['state']=='ready')
        self.assertTrue(all(r['state']=='ready' for r in rows[ready:]))

    def test_dual_trigger_toggles_follow_stand_follow_without_resetting_session_or_history(self):
        self.online_setup();self.args.wait_pico=True;self.args.seconds=5.1
        original=self.worker.snapshot
        edges=[155,175,205,225]
        event_times={}
        def snapshot():
            value=original();count=len(self.transport.commands)
            event=sum(count>=edge for edge in edges)
            if event not in event_times:event_times[event]=self.clock.monotonic()
            value.update(trigger_event_seq=event,trigger_event_monotonic=event_times[event],
                         controller_received_monotonic=self.clock.monotonic())
            return value
        before=[];observe=self.runtime.policy.observe
        def observed(*a,**kw):
            if self.transport.commands:before.append(len(self.runtime.policy.history))
            return observe(*a,**kw)
        with patch.object(self.worker,'snapshot',snapshot),patch.object(self.runtime.policy,'observe',observed):
            report=self.auto_trial()
        self.assertEqual(sum(r.get('record_type')=='tracking_started' for r in self.rows),2)
        self.assertEqual(sum(r.get('record_type')=='tracking_paused' for r in self.rows),2)
        self.assertTrue(all(n==10 for n in before))
        self.assertEqual([r.operation.value for r in self.transport.requests],['PREPARE','COMMIT','CANCEL'])
        self.assertEqual([c.sequence for c in self.transport.commands],list(range(1,report['actions_sent']+1)))
        for index,row in enumerate(self.rows):
            if row.get('record_type')=='tracking_paused':
                next_command=next(r for r in self.rows[index+1:] if 'sent_body_targets' in r)
                self.assertEqual((next_command['stage'],next_command['reference_frame']),('prepare_stand',0))
        self.assert_cancelled()

    def test_stale_trigger_edge_cannot_be_queued_for_later_engagement(self):
        self.online_setup();self.args.wait_pico=True
        original=self.worker.snapshot
        def snapshot():
            value=original()
            value.update(trigger_event_seq=1,trigger_event_monotonic=self.clock.monotonic()-.601,
                         controller_received_monotonic=self.clock.monotonic())
            return value
        with patch.object(self.worker,'snapshot',snapshot):self.auto_trial()
        self.assertTrue(any(r.get('record_type')=='pico_trigger_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))
        self.assert_cancelled()

    def test_real_capture_process_can_start_and_stop_without_pico(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_worker import PicoReferenceProcess
        worker=PicoReferenceProcess(self.runtime)
        worker.start()
        try:
            self.assertEqual(worker.snapshot()['state'],'idle')
            self.assertFalse(worker.connected)
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_manual_stale_reference_rejects_switch_and_keeps_stand(self):
        worker, command=self.manual_trial()
        worker.fault=lambda:dict(state='ready',received_monotonic=self.clock.monotonic()-.601)
        result=self.run_trial(manual_pico=worker,poll_command=command)
        self.assertTrue(result['exit_confirmed'])
        self.assertTrue(any(r.get('record_type')=='pico_switch_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

    def test_manual_non_neutral_reference_can_enter_tracking(self):
        worker,command=self.manual_trial()
        original=worker.snapshot
        def non_neutral():
            snapshot=original()
            if snapshot['state']=='ready':snapshot['q']=snapshot['q']+.2
            return snapshot
        with patch.object(worker,'snapshot',non_neutral):
            self.run_trial(manual_pico=worker,poll_command=command)
        self.assertTrue(any(r.get('stage')=='pico_tracking' for r in self.rows))
        self.assert_cancelled()

    def test_stand_cli_initializes_no_pico_service_or_reader(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_live,sdk021_guard,sdkmain_adapter,pico_input
        spec=importlib.util.spec_from_file_location('actor_cli_no_pico_test',ROOT/'scripts/actor_pico.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        profile=Path(self.tmp.name)/'peer.json5';profile.touch(mode=0o600);profile.write_text('{mode:"peer"}')
        runtime=NS(root=Path(self.tmp.name),model=NS(backend='sdk'))
        self.args.stand_only=True;self.args.manual_pico=False;self.args.group='default'
        with (patch.dict(os.environ,ZENOH_SESSION_CONFIG_URI=str(profile)),
              patch.object(cli.platform,'machine',return_value='aarch64'),
              patch.object(sdk021_guard,'verify_vendor_sources',return_value={}),
              patch.object(sdkmain_adapter,'verify_state_decoder'),
              patch('deployment.vita_s100_aorta_30dof.actor_pico_scheduling.control_scheduler',return_value={'policy':'SCHED_RR','priority':1}),
              patch.object(sdkmain_adapter,'RealtimeBodyTransport',return_value=self.transport),
              patch('locomotion_aorta.pull.PullController',return_value=self.controller),
              patch.object(pico_input,'create_pico_source',side_effect=AssertionError('Pico reader forbidden')),
              patch.object(cli.subprocess,'Popen',side_effect=AssertionError('Pico service forbidden')),
              patch.object(actor_pico_live,'run_live',return_value={'actions_sent':0}) as runner):
            result=cli.live(runtime,self.args,lambda *_:None)
        self.assertEqual(result['actions_sent'],0)
        self.assertIsNone(runner.call_args.args[4])
        self.assertIsNone(runner.call_args.kwargs['manual_pico'])

    def test_automatic_cli_runs_pico_and_gmr_in_worker_without_buttons(self):
        from unittest.mock import Mock
        from deployment.vita_s100_aorta_30dof import (actor_pico_live,actor_pico_worker,
                                                      sdk021_guard,sdkmain_adapter,pico_input)
        spec=importlib.util.spec_from_file_location('actor_cli_worker_test',ROOT/'scripts/actor_pico.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        profile=Path(self.tmp.name)/'peer.json5';profile.touch(mode=0o600);profile.write_text('{mode:"peer"}')
        runtime=NS(root=Path(self.tmp.name),model=NS(backend='sdk'))
        self.args.stand_only=False;self.args.manual_pico=False;self.args.group='default'
        worker=Mock()
        with (patch.dict(os.environ,ZENOH_SESSION_CONFIG_URI=str(profile)),
              patch.object(cli.platform,'machine',return_value='aarch64'),
              patch.object(sdk021_guard,'verify_vendor_sources',return_value={}),
              patch.object(sdkmain_adapter,'verify_state_decoder'),
              patch('deployment.vita_s100_aorta_30dof.actor_pico_scheduling.control_scheduler',return_value={'policy':'SCHED_RR','priority':1}),
              patch.object(sdkmain_adapter,'RealtimeBodyTransport',return_value=self.transport),
              patch('locomotion_aorta.pull.PullController',return_value=self.controller),
              patch.object(actor_pico_worker,'PicoReferenceProcess',return_value=worker) as process,
              patch.object(pico_input,'create_pico_source',side_effect=AssertionError('control-side Pico reader')),
              patch.object(cli.subprocess,'Popen',side_effect=AssertionError('control-side Pico service')),
              patch.object(actor_pico_live,'run_live',return_value={'actions_sent':0}) as runner):
            cli.live(runtime,self.args,lambda *_:None)
        process.assert_called_once_with(runtime,buttons=False)
        worker.start.assert_called_once();worker.close.assert_called_once()
        self.assertIs(runner.call_args.args[4],worker)
        self.assertIsNone(runner.call_args.kwargs['manual_pico'])
        self.assertIs(runner.call_args.kwargs['service_alive'],worker.process.is_alive)

    def test_buttonless_worker_snapshot_has_no_trigger_fields(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_worker import PicoReferenceProcess
        worker=PicoReferenceProcess(self.runtime,buttons=False)
        worker.start()
        try:
            self.assertIsNone(worker.controls)
            self.assertEqual(worker.snapshot(),dict(state='idle'))
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_live_starts_without_reading_a_move_prompt(self):
        self.args.stand_only=True
        self.transport.after_publish=lambda *_:self.stop.set()
        with patch('sys.stdin',io.StringIO('')),patch('select.select',side_effect=AssertionError('MOVE prompt forbidden')):
            self.assertEqual(self.run_trial()['actions_sent'],1)
        self.assert_cancelled()

    def test_interrupt_before_preflight_never_enters(self):
        self.stop.set()
        self.assertFalse(self.run_trial()['entered'])
        self.assertEqual(self.transport.requests, [])

    def test_continuous_mode_runs_beyond_120_seconds_until_operator_stop(self):
        self.args.stand_only=True;self.args.seconds=0.
        command,_=self.runtime.policy.decode(np.zeros(28))
        self.transport.after_publish=lambda c:self.stop.set() if c.sequence==6002 else None
        with patch.object(self.runtime,'infer_motion_command',return_value=(command,np.zeros(1520),{})):
            report=self.run_trial()
        self.assertGreater(report['elapsed_s'],120.)
        self.assertEqual(report['duration_limit_s'],0.)
        self.assertEqual(report['stop_reason'],'operator_stop')
        self.assertEqual(report['actions_sent'],6002)
        self.assert_cancelled()

    def test_explicit_duration_reports_expiry_and_confirms_passive(self):
        self.args.stand_only=True;self.args.seconds=.06
        report=self.run_trial()
        self.assertEqual(report['stop_reason'],'duration_elapsed')
        self.assertFalse(report['stopped_by_signal'])
        self.assertAlmostEqual(report['elapsed_s'],.06)
        self.assert_cancelled()

    def test_invalid_duration_never_requests_control(self):
        for value in (-1.,float('nan'),float('inf')):
            self.args.seconds=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'duration'):
                self.run_trial()
        self.assertEqual((self.transport.requests,self.transport.commands),([],[]))

    def test_interrupt_after_publish_cancels_once_without_extra_command(self):
        self.transport.after_publish = lambda c:self.stop.set()
        self.assertEqual(self.run_trial()['actions_sent'], 1)
        self.assert_cancelled()

    def test_persistent_stale_feedback_cancels_after_skipped_ticks(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import MAX_STALE_TICKS
        self.transport.after_publish = lambda c:setattr(self.transport, 'state_fault',
            lambda s:replace(s, ts_state_pub_us=s.ts_state_pub_us-200_000))
        with self.assertRaisesRegex(TimeoutError, 'consecutive policy ticks.*publication is stale'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
        failures = [r for r in self.rows if r.get('record_type')=='body_state_age_failure']
        self.assertEqual([r['action'] for r in failures], ['skip_tick']*MAX_STALE_TICKS+['cancel'])
        self.assert_cancelled()

    def test_transient_stale_feedback_skips_one_tick_and_continues(self):
        self.args.stand_only = True
        def once(command):
            if command.sequence == 3:
                self.transport.state_fault = lambda s:(setattr(self.transport,'state_fault',None),
                    replace(s, ts_state_pub_us=s.ts_state_pub_us-200_000))[1]
            if command.sequence == 10:
                self.stop.set()
        self.transport.after_publish = once
        report = self.run_trial()
        self.assertEqual(report['actions_sent'], 10)
        failures = [r for r in self.rows if r.get('record_type')=='body_state_age_failure']
        self.assertEqual([(r['phase'], r['action']) for r in failures], [('at_read', 'skip_tick')])
        self.assertEqual([c.sequence for c in self.transport.commands], list(range(1, 11)))
        self.assert_cancelled()

    def test_feedback_that_expires_during_inference_is_not_published(self):
        self.args.stand_only = True
        original = self.runtime.model.infer
        def delayed(observation):
            action = original(observation)
            if self.transport.commands:
                self.clock.sleep(.085)
            return action
        self.transport.state_fault = lambda state: replace(state,
            ts_state_pub_us=state.ts_state_pub_us-20_000) if self.transport.commands else state
        with patch.object(self.runtime.model,'infer',delayed):
            with self.assertRaisesRegex(TimeoutError,'publication_age_ms=105.000'):
                self.run_trial()
        self.assertEqual(len(self.transport.commands),1)
        failure = next(r for r in self.rows if r.get('record_type')=='body_state_age_failure')
        self.assertEqual((failure['phase'], failure['action']), ('before_publish', 'skip_tick'))
        self.assertEqual(failure['at_read']['publication_age_ms'],20.)
        self.assertEqual(failure['at_failure']['publication_age_ms'],105.)
        self.assertEqual(failure['at_failure']['receipt_age_ms'],85.)
        self.assert_cancelled()

    def test_expired_compute_keeps_last_published_action_on_recovery(self):
        self.args.stand_only = True
        self.args.seconds = .5
        published_action = np.full(28, .02, np.float32)
        skipped_action = np.full(28, .08, np.float32)
        recovered_action = np.full(28, .04, np.float32)
        observed_last_actions = []
        expired = False

        def infer(observation):
            nonlocal expired
            if not self.transport.commands:
                return published_action.copy()
            observed_last_actions.append(self.runtime.policy.last_action.copy())
            if not expired:
                expired = True
                self.clock.sleep(.11)
                return skipped_action.copy()
            return recovered_action.copy()

        self.transport.after_publish = lambda command: self.stop.set() if command.sequence == 2 else None
        with patch.object(self.runtime.model, 'infer', infer):
            report = self.run_trial()
        self.assertEqual(report['actions_sent'], 2)
        self.assertEqual(len(observed_last_actions), 2)
        for action in observed_last_actions:
            np.testing.assert_array_equal(action, published_action)
        np.testing.assert_array_equal(self.runtime.policy.last_action, recovered_action)
        failures = [r for r in self.rows if r.get('record_type') == 'body_state_age_failure']
        self.assertEqual([(r['phase'], r['action']) for r in failures],
                         [('before_publish', 'skip_tick')])
        self.assertEqual([command.sequence for command in self.transport.commands], [1, 2])
        self.assert_cancelled()

    def test_feedback_future_clock_error_reports_signed_age(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import Inputs
        inputs = Inputs(self.runtime,self.controller,None,self.clock)
        state = self.transport.wait_state()
        with self.assertRaisesRegex(TimeoutError,'publication_age_ms=-6.000'):
            inputs.fresh(replace(state,ts_state_pub_us=state.ts_state_pub_us+6000))
        # Sensor stamp age is diagnostic only, as in the SDK WBC demo.
        inputs.fresh(replace(state,ts_state_real_us=state.ts_state_real_us-500_000))
        with self.assertRaisesRegex(ValueError,'missing'):
            inputs.fresh(replace(state,ts_state_real_us=0))
        self.assertEqual(self.transport.commands,[])

    def test_stale_pico_after_entry_cancels_without_retry(self):
        self.transport.after_publish = lambda c:setattr(self.pico, 'fault', lambda:None)
        with self.assertRaisesRegex(TimeoutError, 'Pico input'): self.run_trial()
        self.assertLessEqual(len(self.transport.commands), 35)
        self.assert_cancelled()

    def test_failed_pico_worker_after_entry_cancels(self):
        self.transport.after_publish = lambda c:setattr(self.pico, 'snapshot',
            lambda:dict(state='failed', error='Owned Pico service exited'))
        with self.assertRaisesRegex(RuntimeError, 'Pico input worker failed'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
        self.assert_cancelled()

    def test_automatic_tracking_matches_reference_snapshot(self):
        self.run_trial()
        start = next(r for r in self.rows if r.get('record_type') == 'tracking_started')
        tracked = [r for r in self.rows if r.get('stage') == 'pico_tracking']
        self.assertTrue(tracked)
        self.assertTrue(all(r['pico_reference']['yaw_offset'] == start['reference_yaw_offset'] for r in tracked))
        self.assertEqual(len(tracked[0]['pico_reference']['q']), self.runtime.reference.steps)
        self.assertTrue(all(0 <= r['pico_age_s'] <= self.runtime.config['pico_max_age_s'] for r in tracked))

    def test_reject_drop_timeout_and_ownership_changes_stop_publication(self):
        for change in (dict(reject_count=1), dict(timeout_hold=True),
                       dict(client_id='other'), dict(generation=8)):
            with self.subTest(change=change):
                self.setUp()
                self.transport.after_publish = lambda c:setattr(self.transport, 'status_fault',
                    lambda s:replace(s, **change))
                with self.assertRaisesRegex(RuntimeError, 'ownership/state'): self.run_trial()
                self.assertEqual(len(self.transport.commands), 1)
                self.assertIn('CANCEL', [r.operation.value for r in self.transport.requests])

    def test_occasional_superseded_commands_are_logged_not_fatal(self):
        self.args.stand_only = True
        def drop(c):
            if c.sequence in (3, 30):
                self.transport.status_fault = (lambda n:lambda s:replace(s, ingress_drop_count=n))(c.sequence//3)
            if c.sequence == 40: self.stop.set()
        self.transport.after_publish = drop
        self.assertEqual(self.run_trial()['actions_sent'], 40)
        drops = [r['count'] for r in self.rows if r.get('record_type')=='firmware_ingress_drop']
        self.assertEqual(drops, [1, 10])
        self.assert_cancelled()

    def test_sustained_superseded_commands_cancel(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import MAX_INGRESS_DROPS
        self.args.stand_only = True
        self.transport.after_publish = lambda c:setattr(self.transport, 'status_fault',
            lambda s:replace(s, ingress_drop_count=c.sequence))
        with self.assertRaisesRegex(RuntimeError, 'superseded'): self.run_trial()
        self.assertEqual(len(self.transport.commands), MAX_INGRESS_DROPS+1)
        self.assert_cancelled()

    def test_first_ingress_drop_poll_counts_from_entry(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import MAX_INGRESS_DROPS
        guard = StatusGuard(self.clock)
        entered = NS(ingress_drop_count=17)
        status = NS(ingress_drop_count=17+MAX_INGRESS_DROPS+1)
        with self.assertRaisesRegex(RuntimeError, 'superseded'):
            guard.ingress_drops(status, entered)

    def test_ingress_drop_window_expires_without_losing_entry_burst(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import (
            INGRESS_DROP_WINDOW, MAX_INGRESS_DROPS)
        guard = StatusGuard(self.clock)
        entered = NS(ingress_drop_count=17)
        guard.ingress_drops(NS(ingress_drop_count=17+MAX_INGRESS_DROPS), entered)
        self.clock.sleep(INGRESS_DROP_WINDOW+.01)
        guard.ingress_drops(NS(ingress_drop_count=18+MAX_INGRESS_DROPS), entered)
        with self.assertRaisesRegex(RuntimeError, 'superseded'):
            guard.ingress_drops(NS(ingress_drop_count=18+2*MAX_INGRESS_DROPS), entered)

    def test_one_period_inference_overrun_is_logged_and_skips_missed_ticks(self):
        self.args.stand_only = True
        original = self.runtime.infer_motion_command
        def expensive(*a, **kw):
            result = original(*a, **kw)
            if self.transport.commands: self.clock.sleep(.021)
            return result
        self.transport.after_publish = lambda c:self.stop.set() if len(self.transport.commands) == 3 else None
        with patch.object(self.runtime, 'infer_motion_command', expensive):
            report = self.run_trial()
        self.assertEqual([c.sequence for c in self.transport.commands], [1,2,3])
        self.assertEqual(report['missed_policy_ticks'], 2)
        self.assertTrue(any(r.get('record_type') == 'policy_period_overrun' and
                            r.get('phase') == 'compute' for r in self.rows))
        self.assert_cancelled()

    def test_inference_overrun_cannot_publish_expired_feedback(self):
        self.args.stand_only = True
        original = self.runtime.infer_motion_command
        def expensive(*a, **kw):
            result = original(*a, **kw)
            if self.transport.commands: self.clock.sleep(.101)
            return result
        with patch.object(self.runtime, 'infer_motion_command', expensive):
            with self.assertRaisesRegex(TimeoutError, 'stale'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
        self.assert_cancelled()

    def test_nominal_scheduler_skips_late_slots_without_catchup_bursts(self):
        for now, expected, missed in [(100.,100.02,0),(100.02,100.02,0),
                                      (100.025,100.04,1),(100.04,100.04,1),
                                      (100.045,100.06,2)]:
            with self.subTest(now=now):
                following, count = next_policy_deadline(100.,now,.02)
                self.assertAlmostEqual(following,expected)
                self.assertEqual(count,missed)

    def test_scheduler_delay_uses_new_fresh_state_without_extra_publication_gap_rule(self):
        self.args.stand_only=True
        original=self.clock.sleep
        def delayed(seconds):
            original(seconds)
            if self.transport.commands:original(.101)
        self.transport.after_publish=lambda c:self.stop.set() if c.sequence==3 else None
        with patch.object(self.clock,'sleep',delayed):report=self.run_trial()
        self.assertEqual(report['actions_sent'],3)
        self.assert_cancelled()

    def test_entry_failure_still_attempts_cancel(self):
        self.transport.entry_fault = True
        with self.assertRaisesRegex(RuntimeError, 'entry failure'): self.run_trial()
        self.assertEqual(self.transport.commands, [])
        self.assert_cancelled()

    def test_exit_failure_is_reported_without_claiming_passive(self):
        # Avoid waiting real SDK deadlines for this injected exit fault.
        self.transport.after_publish = lambda c:self.stop.set()
        with patch.object(self.controller, 'exit_external', side_effect=TimeoutError('exit incomplete')):
            with self.assertRaisesRegex(TimeoutError, 'exit incomplete'): self.run_trial()
        self.assertTrue(any(r.get('record_type') == 'exit_failure' for r in self.rows))
        self.assertEqual(self.controller.status.current_action, 'EXTERNAL')

    def test_mock_actuator_or_missing_roster_rejected_before_prepare(self):
        body = self.transport.reply.info.body
        caps = list(body.joints); caps[0] = replace(caps[0], is_mock=True)
        self.transport.reply = replace(self.transport.reply,
            info=replace(self.transport.reply.info, body=replace(body, joints=tuple(caps))))
        with self.assertRaisesRegex(ValueError, 'Physical ACTIVE'): self.run_trial()
        self.assertEqual(self.transport.requests, [])

    def test_config_changed_during_preflight_refuses_entry(self):
        original=self.runtime.infer_motion
        def changed(*a,**kw):
            result=original(*a,**kw)
            root = robot_config(); root['parallel_ankle']['pd']['strategy'] = 'position_motor_gains'
            self.config.write_text(json.dumps(root))
            return result
        with patch.object(self.runtime,'infer_motion',changed):
            with self.assertRaisesRegex(ValueError, 'virtual_pd'): self.run_trial()
        self.assertEqual(self.transport.requests, [])

    def test_full_body_compensation_command_is_preserved(self):
        index = self.runtime.policy.order.index('left_ankle_pitch_joint')
        original = self.runtime.infer_motion_command
        def compensated(*a, **kw):
            cmd, obs, row = original(*a, **kw)
            joints = list(cmd.joints); joints[index] = replace(joints[index], tau=2.5, dq=.1)
            return replace(cmd, joints=tuple(joints)), obs, row
        self.transport.after_publish = lambda c:self.stop.set()
        with patch.object(self.runtime, 'infer_motion_command', compensated): self.run_trial()
        sent = self.transport.commands[0]
        self.assertEqual((sent.joints[index].tau, sent.joints[index].dq), (2.5, .1))
        self.assert_cancelled()

    def test_slow_logging_skips_ticks_without_burst_publication(self):
        self.args.stand_only = True
        published_at = []
        def after_publish(command):
            published_at.append(self.clock.monotonic())
            if len(published_at) == 3: self.stop.set()
        def write(row, obs=None):
            self.rows.append(row)
            if 'sent_body_targets' in row: self.clock.sleep(.021)
        self.transport.after_publish = after_publish
        report = self.run_trial(write=write)
        self.assertEqual(report['actions_sent'],3)
        self.assertEqual(report['missed_policy_ticks'],3)
        self.assertTrue(np.allclose(np.diff(published_at),[.04,.04]))
        self.assert_cancelled()

    def test_publish_time_is_recorded_without_an_extra_100ms_deadline(self):
        self.args.stand_only=True
        def published(c):self.clock.sleep(.101);self.stop.set()
        self.transport.after_publish=published
        report=self.run_trial()
        self.assertEqual(report['actions_sent'],1)
        row=next(r for r in self.rows if 'sent_body_targets' in r)
        self.assertAlmostEqual(row['publish_ms'],101.)
        self.assert_cancelled()

    def test_logging_delay_does_not_reject_fresh_next_feedback(self):
        self.args.stand_only=True
        def write(row,obs=None):
            self.rows.append(row)
            if 'sent_body_targets' in row:self.clock.sleep(.101)
        self.transport.after_publish=lambda c:self.stop.set() if c.sequence==3 else None
        report=self.run_trial(write=write)
        self.assertEqual(report['actions_sent'],3)
        self.assert_cancelled()

    def test_exit_to_other_action_is_not_reported_as_passive(self):
        self.transport.exit_action = 'FIXED_STAND'
        self.transport.after_publish = lambda c:self.stop.set()
        with self.assertRaisesRegex(RuntimeError, 'PASSIVE/IDLE'): self.run_trial()
        self.assertTrue(any(r.get('record_type') == 'exit_failure' for r in self.rows))
        self.assertEqual(self.controller.status.current_action, 'FIXED_STAND')

    def test_disabled_virtual_pd_refuses_before_any_request(self):
        config = robot_config()
        config['actions']['EXTERNAL']['params']['ankle_virtual_pd_intent'] = False
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'virtual_pd'): self.run_trial()
        self.assertEqual((self.transport.requests, self.transport.commands), ([], []))

    def test_measured_pose_has_small_margin_but_commands_use_exact_ranges(self):
        params, _ = load_robot_config(self.config)
        env = CommandEnvelope(self.runtime, self.transport.reply, params)
        state = self.transport.wait_state()
        q = state.q.copy(); q[0] = 4.02
        env.check_measured_pose(replace(state, q=q))
        q[0] = 4.05
        with self.assertRaisesRegex(ValueError, 'outside'): env.check_measured_pose(replace(state, q=q))
        cmd, _ = self.runtime.policy.decode(np.zeros(28))
        joints = list(cmd.joints); joints[0] = replace(joints[0], q=4.01)
        with self.assertRaisesRegex(ValueError, 'outside'): env.check(replace(cmd, joints=tuple(joints)))

    def test_command_outside_effective_hardware_limits_is_rejected(self):
        params, _ = load_robot_config(self.config)
        env = CommandEnvelope(self.runtime, self.transport.reply, params)
        cmd, _ = self.runtime.policy.decode(np.zeros(28))
        for key, value in [('q', 99.), ('dq', 99.), ('tau', 999.), ('kp', 999.), ('kd', 999.)]:
            joints = list(cmd.joints); joints[0] = replace(joints[0], **{key:value})
            with self.assertRaisesRegex(ValueError, 'outside'): env.check(replace(cmd, joints=tuple(joints)))

    def capability_reply(self, index, **changes):
        reply = self.transport.reply
        caps = list(reply.info.body.joints)
        caps[index] = replace(caps[index], **changes)
        return replace(reply, info=replace(reply.info, body=replace(reply.info.body, joints=tuple(caps))))

    def test_current_firmware_config_needs_no_yaml_command_limits(self):
        params, _ = load_robot_config(self.config)
        self.assertNotIn('command_limits', params)
        CommandEnvelope(self.runtime, self.transport.reply, params)

    def test_obsolete_yaml_limits_and_invalid_timing_refuse_before_control(self):
        for changes in (dict(command_limits={}), dict(command_timeout_ms=True),
                        dict(recover_valid_frames=0),
                        dict(interpolation=dict(min_period_ms=100, max_period_ms=2))):
            config = robot_config()
            config['actions']['EXTERNAL']['params'].update(changes)
            self.config.write_text(json.dumps(config))
            with self.assertRaises(ValueError): self.run_trial()
        self.assertEqual((self.transport.requests, self.transport.commands), ([], []))

    def test_missing_or_malformed_linear_ranges_and_support_are_rejected(self):
        params, _ = load_robot_config(self.config)
        for changes in (dict(valid_limits=1), dict(supported_fields=1),
                        dict(kp_max=0.), dict(tau_min=float('nan'))):
            with self.assertRaises(ValueError):
                CommandEnvelope(self.runtime, self.capability_reply(0, **changes), params)

    def test_model_and_safe_hold_gains_must_fit_robot_info(self):
        params, _ = load_robot_config(self.config)
        with self.assertRaisesRegex(ValueError, 'safe_hold'):
            CommandEnvelope(self.runtime, self.capability_reply(0, kp_max=5.), params)
        with self.assertRaisesRegex(ValueError, 'outside'):
            CommandEnvelope(self.runtime, self.capability_reply(0, kp_max=11.), params)

    def test_parallel_ankle_absent_ranges_are_delegated_not_zero_limits(self):
        params, _ = load_robot_config(self.config)
        reply = self.transport.reply
        caps = list(reply.info.body.joints)
        for i in (4, 5, 10, 11):
            caps[i] = replace(caps[i], valid_limits=1, dq_min=0., dq_max=0.,
                tau_min=0., tau_max=0., kp_min=0., kp_max=0., kd_min=0., kd_max=0.)
        reply = replace(reply, info=replace(reply.info, body=replace(reply.info.body, joints=tuple(caps))))
        env = CommandEnvelope(self.runtime, reply, params)
        self.assertEqual(len(env.delegated_joints), 4)
        command, _ = self.runtime.policy.decode(np.zeros(28))
        self.assertGreater(command.joints[4].kp, 0.)
        env.check(command)
        for change in (dict(q=9.), dict(tau=float('nan')), dict(kp=-1.)):
            joints = list(command.joints); joints[4] = replace(joints[4], **change)
            with self.assertRaises(ValueError): env.check(replace(command, joints=tuple(joints)))

    def test_reference_advances_one_frame_per_published_step_despite_wallclock_jitter(self):
        self.online_setup()
        times=[]
        original=self.runtime.infer_motion_command
        def infer(*a,**kw):
            if self.transport.requests: times.append(kw.get('ref_time',0.))
            return original(*a,**kw)
        def published(command):
            self.clock.sleep(.021)
            if command.sequence==5:self.stop.set()
        self.transport.after_publish=published
        with patch.object(self.runtime,'infer_motion_command',infer):report=self.auto_trial()
        np.testing.assert_allclose(times,np.arange(5)*.02,atol=1e-10)
        self.assertEqual(report['missed_policy_ticks'],5)
        self.assert_cancelled()

    def test_straight_knees_are_valid_observations_for_direct_stand_entry(self):
        self.args.stand_only=True
        self.transport.state_fault=lambda s:replace(s,q=np.zeros(31,np.float32))
        self.transport.after_publish=lambda *_:self.stop.set()
        self.run_trial()
        self.assertEqual(len(self.transport.commands),1)
        self.assert_cancelled()

    def test_actor_does_not_construct_legacy44_state_on_body31_path(self):
        from deployment.vita_s100_aorta_30dof import sdkmain_adapter as adapter
        self.args.stand_only=True
        self.transport.after_publish=lambda *_:self.stop.set()
        with patch.object(adapter,'VitaRawLowState',side_effect=AssertionError('unused legacy conversion')):
            self.run_trial()
        self.assertEqual(len(self.transport.commands),1)
        self.assert_cancelled()

    def test_native_body_validation_preserves_fault_and_finite_guards(self):
        from deployment.vita_s100_aorta_30dof.sdkmain_adapter import validate_body_arrays
        state=self.transport.wait_state()
        validate_body_arrays(state)
        for name,index,value in [('error_code',0,1),('q',29,np.nan),('dq',30,np.inf),
                                 ('imu_values',(0,7),np.nan)]:
            data=getattr(state,name).copy();data[index]=value
            with self.subTest(name=name),self.assertRaises(ValueError):
                validate_body_arrays(replace(state,**{name:data}))

    def test_joint_ramp_option_fails_before_any_motion(self):
        self.online_setup();self.args.entry_policy='joint-ramp'
        with self.assertRaisesRegex(ValueError,'removed'):self.auto_trial()
        self.assertEqual((self.transport.requests,self.transport.commands,self.startup_calls),([],[],[]))

    def auto_setup(self, initial='PASSIVE'):
        self.args.auto_start = self.args.manual_pico = True
        self.args.seconds = 12.
        self.transport.value = replace(self.transport.value,current_action=initial)
        self.startup_calls = []
        self.transport.state_fault = lambda s:replace(s,q=np.zeros(31,np.float32)) if not self.transport.commands else s
        self.worker = ManualPico(self.runtime,self.clock)
        def action(name):
            self.startup_calls.append((name,self.clock.monotonic()))
            self.transport.value = replace(self.transport.value,
                current_action='FIXED_STAND' if name=='FIXED_STAND' else 'PASSIVE')
        self.request_action = action

    def auto_trial(self, **kwargs):
        return self.run_trial(manual_pico=self.worker,request_action=self.request_action,**kwargs)

    def test_auto_waits_full_damping_once_then_fixed_and_continuous_onnx_without_pico(self):
        self.auto_setup()
        result = self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND'])
        self.assertGreaterEqual(self.startup_calls[0][1],104.)
        stages = [r['stage'] for r in self.rows if 'stage' in r]
        self.assertNotIn('pose_prepare',stages);self.assertIn('prepare_stand',stages)
        self.assertNotIn('pico_tracking',stages)
        self.assertFalse(self.worker.connected)
        self.assertEqual([r.operation.value for r in self.transport.requests],['PREPARE','COMMIT','CANCEL'])
        self.assertTrue(result['exit_confirmed'])
        self.assertTrue(result['onnx_control_enabled'])
        self.assertEqual(self.rows[0]['record_type'],'fixed_stand_requested')
        self.assertFalse(any(r.get('record_type')=='live_preflight' for r in self.rows))
        self.assert_cancelled()

    def test_auto_existing_fixed_stand_does_not_cycle_through_passive(self):
        self.auto_setup('FIXED_STAND')
        self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assert_cancelled()

    def test_auto_stop_before_preflight_has_no_fixed_or_external_requests(self):
        self.auto_setup()
        self.stop.set()
        self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])

    def test_auto_hardware_outside_pose_refuses_before_fixed_stand(self):
        self.auto_setup()
        def bad(s):
            q=s.q.copy();q[3]=99.;return replace(s,q=q)
        self.transport.state_fault=bad
        with self.assertRaisesRegex(ValueError,'outside'):self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])

    def test_auto_fixed_rejection_cancels_once_without_external_retry(self):
        self.auto_setup()
        original=self.request_action
        def reject(name):
            original(name)
            if name=='FIXED_STAND':raise RuntimeError('firmware rejected')
        self.request_action=reject
        with self.assertRaisesRegex(RuntimeError,'firmware rejected'):self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND','CANCEL'])
        self.assertEqual(self.transport.requests,[])


    def test_auto_wait_interrupt_does_not_send_any_motion_request(self):
        self.auto_setup()
        original=self.transport.state_fault
        def interrupt(s):
            if self.clock.monotonic()>101.:self.stop.set()
            return original(s)
        self.transport.state_fault=interrupt
        self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])

    def test_auto_other_external_owner_is_not_cancelled(self):
        self.auto_setup('FIXED_STAND')
        def takeover(name):
            self.startup_calls.append((name,self.clock.monotonic()))
            self.transport.value=replace(self.transport.value,current_action='EXTERNAL',generation=99)
        self.transport.value=replace(self.transport.value,current_action='PASSIVE')
        self.request_action=takeover
        with self.assertRaisesRegex(RuntimeError,'Another EXTERNAL'):self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND'])
        self.assertEqual(self.transport.requests,[])

    def test_auto_pico_handoff_stays_in_one_external_session(self):
        self.auto_setup('FIXED_STAND')
        self.args.seconds=14.
        def command():
            n=len(self.transport.commands)
            if n==450:return 'connect'
            if n==500:return 'track'
        self.auto_trial(poll_command=command)
        self.assertTrue(any(r.get('stage')=='pico_tracking' for r in self.rows))
        self.assertEqual([r.operation.value for r in self.transport.requests],['PREPARE','COMMIT','CANCEL'])
        sequences=[c.sequence for c in self.transport.commands]
        self.assertEqual(sequences,list(range(1,len(sequences)+1)))
        self.assert_cancelled()

    def test_auto_stale_feedback_never_proceeds_to_fixed_or_external(self):
        self.auto_setup()
        original=self.transport.state_fault
        def stale(s):
            s=original(s)
            return replace(s,ts_state_pub_us=s.ts_state_pub_us-200_000) if self.clock.monotonic()>101 else s
        self.transport.state_fault=stale
        with self.assertRaises(TimeoutError):self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])


    def test_auto_fixed_timeout_cancels_and_never_enters_external(self):
        self.auto_setup()
        def no_activation(name):
            self.startup_calls.append((name,self.clock.monotonic()))
        self.request_action=no_activation
        with self.assertRaisesRegex(TimeoutError,'FIXED_STAND did not settle'):self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND','CANCEL'])
        self.assertEqual(self.transport.requests,[])

    def test_auto_invalid_fixed_gains_reject_before_motion(self):
        self.auto_setup()
        config=robot_config()
        config['actions']['FIXED_STAND']['params']['keyframes'][0]['kp']=[999.]*42
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'outside'):self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])

    def test_auto_early_pico_request_is_not_queued_during_policy_lead_in(self):
        self.auto_setup('FIXED_STAND')
        self.args.wait_pico=True
        self.auto_trial(poll_command=lambda:'track' if len(self.transport.commands)==20 else None)
        self.assertTrue(self.worker.connected)
        self.assertTrue(any(r.get('record_type')=='pico_switch_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

    def test_auto_uses_policy_history_from_first_published_frame(self):
        self.auto_setup('FIXED_STAND')
        def published(command):
            self.assertEqual(len(self.runtime.policy.history),10)
        self.transport.after_publish=published
        result=self.auto_trial()
        self.assertTrue(result['onnx_control_enabled'])


    def test_auto_config_change_after_fixed_startup_cancels_before_prepare(self):
        self.auto_setup('FIXED_STAND')
        from deployment.vita_s100_aorta_30dof import actor_pico_startup as startup
        original=startup.fixed_stand_startup
        def changed(*a,**kw):
            result=original(*a,**kw)
            config=robot_config();config['actions']['EXTERNAL']['params']['command_timeout_ms']=2000
            self.config.write_text(json.dumps(config))
            return result
        with patch.object(startup,'fixed_stand_startup',changed):
            with self.assertRaisesRegex(RuntimeError,'config file changed during startup'):self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['CANCEL'])
        self.assertEqual(self.transport.requests,[])
        self.assertEqual(self.transport.value.current_action,'PASSIVE')

    def test_auto_stop_terminal_command_is_polled_during_damping_wait(self):
        self.auto_setup()
        def stop_command():
            if self.clock.monotonic()>101:self.stop.set()
        self.auto_trial(poll_command=stop_command)
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])

    def test_auto_initial_retired_action_still_requests_and_verifies_fixed_pose(self):
        self.auto_setup('')
        self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND'])
        self.assert_cancelled()


    def test_auto_damping_completion_clears_action_name_without_false_failure(self):
        self.auto_setup('PASSIVE')
        original=self.transport.state_fault
        def retired(s):
            if 102.1<self.clock.monotonic()<104.:
                self.transport.value=replace(self.transport.value,current_action='')
            return original(s)
        self.transport.state_fault=retired
        self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND'])
        self.assertGreaterEqual(self.startup_calls[0][1],104.)
        self.assert_cancelled()

    def test_auto_empty_action_does_not_replace_fixed_stand_feedback_validation(self):
        self.auto_setup('')
        def action(name):
            self.startup_calls.append((name,self.clock.monotonic()))
            self.transport.value=replace(self.transport.value,
                current_action='' if name=='FIXED_STAND' else 'PASSIVE')
        self.request_action=action
        def wrong_pose(state):
            q=np.zeros(31,np.float32);q[0]=.2
            return replace(state,q=q)
        self.transport.state_fault=wrong_pose
        with self.assertRaisesRegex(TimeoutError,'FIXED_STAND did not settle'):self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND','CANCEL'])
        self.assertEqual(self.transport.requests,[])

    def online_setup(self, initial='FIXED_STAND'):
        self.auto_setup(initial)
        self.args.entry_policy='policy-online'
        self.args.seconds=5.

    def test_online_entry_runs_onnx_during_lead_in_from_straight_knees(self):
        self.online_setup()
        result=self.auto_trial()
        self.assertTrue(result['onnx_control_enabled'])
        self.assertFalse(any(r.get('stage')=='pose_prepare' for r in self.rows))
        self.assertTrue(any(r.get('stage')=='prepare_stand' for r in self.rows))
        self.assertGreater(len(self.transport.commands),100)
        self.assert_cancelled()

    def test_online_entry_uses_measured_reference_without_clipping_observation(self):
        self.online_setup()
        original=self.runtime.infer_motion_command
        first=[]
        def inference(motion,**kw):
            if self.transport.requests and not self.transport.commands:
                np.testing.assert_allclose(motion.q[0],kw['q'])
                knee=self.runtime.model.contract.names.index('left_knee_joint')
                self.assertLess(kw['q'][knee],self.runtime.policy.lower[knee])
                first.append(True)
            return original(motion,**kw)
        with patch.object(self.runtime,'infer_motion_command',inference):self.auto_trial()
        self.assertTrue(first)

    def test_online_does_not_require_all_joints_at_fixed_zero_target(self):
        self.online_setup()
        original=self.transport.state_fault
        def bent(s):
            s=original(s)
            if not self.transport.commands:
                q=s.q.copy();q[18]=.25;return replace(s,q=q)
            return s
        self.transport.state_fault=bent
        self.auto_trial()
        confirmed=next(r for r in self.rows if r.get('record_type')=='fixed_stand_confirmed')
        self.assertAlmostEqual(confirmed['q'][18],.25)
        self.assert_cancelled()

    def test_online_requires_actual_active_fixed_stand_before_handoff(self):
        self.online_setup('')
        def never_active(name):
            self.startup_calls.append((name,self.clock.monotonic()))
            if name=='CANCEL':self.transport.value=replace(self.transport.value,current_action='PASSIVE')
        self.request_action=never_active
        with self.assertRaisesRegex(TimeoutError,'FIXED_STAND did not settle'):self.auto_trial()
        self.assertEqual(self.transport.requests,[])

    def test_online_large_step_within_robot_ranges_matches_cpp_policy_output_contract(self):
        self.online_setup()
        self.transport.reply=self.capability_reply(5,q_min=-.43625,q_max=.43625)
        original_state=self.transport.state_fault
        def initial(state):
            state=original_state(state)
            if not self.transport.commands:
                q=state.q.copy();q[5]=.288;return replace(state,q=q)
            return state
        self.transport.state_fault=initial
        original=self.runtime.infer_motion_command
        def output(*a,**kw):
            cmd,obs,row=original(*a,**kw)
            joints=list(cmd.joints);joints[5]=replace(joints[5],q=-.392699)
            return replace(cmd,joints=tuple(joints)),obs,row
        self.transport.after_publish=lambda *_:self.stop.set()
        with patch.object(self.runtime,'infer_motion_command',output):self.auto_trial()
        self.assertEqual(len(self.transport.commands),1)
        self.assertGreater(abs(self.transport.commands[0].joints[5].q-.288),.35)
        self.assertLessEqual(abs(self.transport.commands[0].joints[5].q),.43625)
        self.assert_cancelled()

    def test_online_wait_pico_starts_receiver_without_enabling_following(self):
        self.online_setup();self.args.wait_pico=True
        self.auto_trial()
        self.assertTrue(self.worker.connected)
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

    def test_online_auto_pico_follows_only_after_preparation_in_same_session(self):
        self.online_setup();self.args.wait_pico=self.args.auto_pico=True
        self.auto_trial()
        self.assertTrue(any(r.get('stage')=='pico_tracking' for r in self.rows))
        self.assertEqual([r.operation.value for r in self.transport.requests],['PREPARE','COMMIT','CANCEL'])
        self.assert_cancelled()

    def test_online_auto_pico_uses_600ms_age_without_neutral_gate(self):
        for stale in (True,False):
            self.setUp();self.online_setup();self.args.wait_pico=self.args.auto_pico=True
            original=self.worker.snapshot
            def sample():
                value=original()
                if stale:value['received_monotonic']-=.601
                else:value['q']=value['q']+.2
                return value
            self.worker.snapshot=sample
            self.auto_trial()
            tracked=any(r.get('stage')=='pico_tracking' for r in self.rows)
            self.assertEqual(tracked,not stale)
            self.assert_cancelled()

    def test_terminal_short_pico_command_still_requires_explicit_operator_line(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import TerminalCommands
        terminal=TerminalCommands('TEST-ONLY',self.stop)
        with patch('sys.stdin',io.StringIO('PICO\n')),patch('select.select',return_value=([1],[],[])):
            self.assertEqual(terminal.poll(),'track')
        self.assertFalse(self.stop.is_set())

    def test_rejected_policy_candidate_preserves_full_inputs_before_hardware_validation(self):
        self.online_setup()
        original=self.runtime.infer_motion_command
        def jumping(*a,**kw):
            command,obs,row=original(*a,**kw)
            joints=list(command.joints)
            if any(r.get('record_type')=='fixed_stand_confirmed' for r in self.rows):
                joints[0]=replace(joints[0],q=99.)
            return replace(command,joints=tuple(joints)),obs,row
        with patch.object(self.runtime,'infer_motion_command',jumping):
            with self.assertRaisesRegex(ValueError,'outside'):self.auto_trial()
        candidate=next(r for r in self.rows if r.get('record_type')=='prepared_policy_candidate')
        self.assertEqual(len(candidate['observation']),self.runtime.model.contract.input_dim)
        self.assertEqual(len(candidate['input_state']['q']),28)
        self.assertEqual(len(candidate['input_state']['body_q']),31)
        self.assertTrue(candidate['reference']['q'])
        self.assertEqual(self.transport.commands,[])

    def test_complete_candidate_replay_and_feedback_evidence(self):
        import subprocess
        self.online_setup();self.args.seconds=1.
        self.auto_trial()
        candidate=next(r for r in self.rows if r.get('record_type')=='prepared_policy_candidate')
        path=Path(self.tmp.name)/'candidate.jsonl';path.write_text(json.dumps(candidate)+'\n')
        process=subprocess.run([sys.executable,str(ROOT/'scripts/replay_actor_candidate.py'),
            '--root',str(ROOT),'--backend','ort','--input',str(path)],
            capture_output=True,text=True,timeout=15)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        report=json.loads(process.stdout)
        self.assertEqual(report['status'],'PASS_EXACT_CANDIDATE_REPLAY')
        self.assertEqual(report['hardware_envelope_check'],'not_measured')
        self.assertGreaterEqual(report['largest_target_delta_rad'],0.)
        self.assertEqual(report['actions_sent'],0)
        published=next(r for r in self.rows if r.get('stage')=='prepare_stand')
        self.assertEqual(len(published['feedback']['q_body']),31)
        self.assertEqual(len(published['feedback']['root_wxyz']),4)
        self.assertIn('applied_sequence',published['feedback'])

    def test_online_startup_does_not_invent_two_degree_tilt_gate(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import Inputs
        from deployment.vita_s100_aorta_30dof.actor_pico_startup import StartupPlan,fixed_stand_startup
        self.online_setup()
        original=self.transport.state_fault
        def leaned(state):
            state=original(state);imu=state.imu_values.copy()
            imu[0,:4]=[np.cos(.1),0.,np.sin(.1),0.]
            return replace(state,imu_values=imu)
        self.transport.state_fault=leaned
        params,_=load_robot_config(self.config)
        envelope=CommandEnvelope(self.runtime,self.transport.reply,params)
        plan=StartupPlan(self.runtime,envelope,self.config)
        inputs=Inputs(self.runtime,self.controller,None,self.clock)
        with redirect_stdout(io.StringIO()):
            ready=fixed_stand_startup(plan,envelope,self.runtime,self.args,self.controller,inputs,
                self.stop,lambda row:self.rows.append(row),self.clock,request_action=self.request_action)
        self.assertTrue(ready)
        evidence=next(r for r in self.rows if r.get('record_type')=='fixed_stand_confirmed')
        self.assertGreater(evidence['tilt_deg'],11.)
        self.assertEqual(self.transport.requests,[])
        self.assertEqual(self.transport.commands,[])
        self.assertNotIn('online_start_tilt_limit_degrees',plan.review)

    def test_fixed_stand_settle_tolerates_joint_velocity_noise(self):
        for speed, settles in ((.15, True), (.25, False)):
            with self.subTest(speed=speed):
                self.setUp(); self.online_setup()
                original = self.transport.state_fault
                def noisy(state):
                    state = original(state)
                    return replace(state, dq=np.full(31, speed, np.float32)) if not self.transport.commands else state
                self.transport.state_fault = noisy
                if settles:
                    self.auto_trial()
                    self.assertTrue(any(r.get('record_type')=='fixed_stand_confirmed' for r in self.rows))
                    self.assert_cancelled()
                else:
                    with self.assertRaisesRegex(TimeoutError, 'FIXED_STAND did not settle'): self.auto_trial()
                    self.assertEqual(self.transport.requests, [])

    def test_online_body_gyro_must_settle_before_external(self):
        self.online_setup()
        original=self.transport.state_fault
        def spinning(state):
            state=original(state);imu=state.imu_values.copy();imu[0,4]=.2
            return replace(state,imu_values=imu)
        self.transport.state_fault=spinning
        with self.assertRaisesRegex(TimeoutError,'gyro_norm=0.200'):self.auto_trial()
        self.assertEqual(self.transport.requests,[])

    def test_status_freeze_is_detected(self):
        guard = StatusGuard(self.clock)
        status = self.controller.status
        guard.check(status)
        self.clock.sleep(.51)
        with self.assertRaisesRegex(TimeoutError, 'stopped advancing'): guard.check(status)

    def test_peer_mode_is_read_from_actual_profile_without_logging_secrets(self):
        p = Path(self.tmp.name)/'private.json5'; p.touch(mode=0o600)
        p.write_text('''{ // mode: "client"
          mode: 'peer', auth: {password: "mode:client //{}"},
        }''')
        require_peer_profile(p)
        for content in ('{mode:"client"}', '{mode:"router", a:{mode:"peer"}}',
                        '{mode:"peer",mode:"client"}', '{}'):
            p.write_text(content)
            with self.assertRaisesRegex(ValueError, 'mode=peer'): require_peer_profile(p)
        p.write_text('{mode:"peer"}'); p.chmod(0o644)
        with self.assertRaisesRegex(ValueError, '0600'): require_peer_profile(p)

    def test_state_check_uses_no_control_handles_and_closes_reader(self):
        from locomotion_aorta.pull import PullStateSource, PullController
        from deployment.vita_s100_aorta_30dof import sdk021_guard
        import aorta
        spec = importlib.util.spec_from_file_location('actor_cli_state_test', ROOT/'scripts/actor_pico.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        profile = Path(self.tmp.name)/'peer.json5'; profile.touch(mode=0o600)
        profile.write_text('{mode:"peer"}')
        source = PullStateSource(group='default', _transport=self.transport)
        args = NS(group='default', seconds=.1)
        with (patch.dict(os.environ, ZENOH_SESSION_CONFIG_URI=str(profile)),
              patch.object(sdk021_guard, 'verify_vendor_sources', return_value={}),
              patch('locomotion_aorta.pull.PullStateSource', return_value=source),
              patch.object(PullController, '__init__', side_effect=AssertionError('control forbidden')),
              patch.object(aorta.Node, '__init__', side_effect=AssertionError('real Node forbidden')),
              patch.object(cli, 'time', self.clock)):
            result = cli.state_check(self.runtime, args, lambda row:self.rows.append(row))
        self.assertGreater(result['sampled_fresh_frames'], 0)
        self.assertEqual((result['actions_sent'], result['control_handles_created']), (0, 0))
        self.assertEqual((self.transport.requests, self.transport.commands), ([], []))
        self.assertTrue(self.transport.closed)


class PicoButtonTests(unittest.TestCase):
    def setUp(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_buttons import DualTriggerToggle
        self.toggle=DualTriggerToggle();self.stamp=0

    def sample(self,left,right,now,age=0.,available=True):
        self.stamp+=1
        return self.toggle.update((available,self.stamp,age,left,right),now)

    def test_press_at_connect_requires_release_and_hold_fires_only_once(self):
        self.assertFalse(self.sample(1.,1.,0.))
        self.assertFalse(self.sample(0.,0.,.1))
        self.assertTrue(self.sample(1.,1.,.2))
        self.assertFalse(self.sample(1.,1.,1.))

    def test_one_trigger_dipping_does_not_rearm_the_chord(self):
        self.sample(0.,0.,0.);self.assertTrue(self.sample(1.,1.,.1))
        self.assertFalse(self.sample(0.,1.,.5))
        self.assertFalse(self.sample(1.,1.,.6))
        self.sample(0.,0.,.7);self.assertTrue(self.sample(1.,1.,.8))

    def test_bridge_threshold_and_debounce_consume_bounce_without_late_refire(self):
        self.sample(.7,.7,0.);self.assertTrue(self.sample(.71,.71,.1))
        self.sample(0.,0.,.15);self.assertFalse(self.sample(1.,1.,.2))
        self.assertFalse(self.sample(1.,1.,.6))
        self.sample(0.,0.,.7);self.assertTrue(self.sample(1.,1.,.8))

    def test_stale_inputs_disarm_until_new_release(self):
        self.sample(0.,0.,0.)
        self.assertFalse(self.sample(1.,1.,.1,age=.601))
        self.assertFalse(self.sample(1.,1.,.2))
        self.sample(0.,0.,.3);self.assertTrue(self.sample(1.,1.,.4))

    def test_duplicate_or_regressing_packet_and_invalid_depth_do_not_fire(self):
        self.sample(0.,0.,0.)
        self.assertFalse(self.toggle.update((True,self.stamp,0.,1.,1.),.1))
        self.assertFalse(self.sample(2.,1.,.2))
        self.assertFalse(self.sample(1.,1.,.3))
        self.sample(0.,0.,.4)
        self.assertFalse(self.toggle.update((True,1,0.,1.,1.),.5))


class LocalStartTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'outputs').mkdir()
        self.machine=self.root/'machine-id';self.machine.write_text('this-board\n')
        self.selector=self.root/'end_effector';self.selector.write_text('none\n')
        self.config=self.root/'robot.yaml';self.config.write_text(json.dumps(robot_config()))
        self.profile=self.root/'outputs/actor_pico_local.json'
        self.profile.touch(mode=0o600)
        self.cfg=dict(schema='vita_actor_local_start_v1',machine_id_sha256=hashlib.sha256(b'this-board').hexdigest(),
            robot_id='10.100.23.96',robot_config=str(self.config),
            robot_config_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest(),
            end_effector='none',seconds=120,pico_handoff='manual')
        self.save()

    def save(self):self.profile.write_text(json.dumps(self.cfg))

    def load(self,addresses=('10.100.23.96',)):
        from deployment.vita_s100_aorta_30dof.actor_pico_local import load_local_start
        with patch('platform.machine',return_value='aarch64'),patch('platform.system',return_value='Linux'):
            return load_local_start(self.root,machine_id_path=self.machine,selector_path=self.selector,addresses=addresses)

    def test_launcher_requires_one_explicit_policy_and_selects_device_bound_motion(self):
        import subprocess
        scripts=self.root/'scripts';scripts.mkdir()
        launcher=scripts/'start_actor_pico.sh';launcher.write_bytes((ROOT/'scripts/start_actor_pico.sh').read_bytes())
        output=self.root/'arguments.json'
        fake=self.root/'fake-python'
        fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["ACTOR_LAUNCH_TEST_OUTPUT"]).write_text(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        startup=self.root/'bash-env'
        startup.write_text('source() { if [[ "$1" == /app/script/env.sh ]]; then return 0; fi; builtin source "$@"; }\n')
        env=dict(os.environ,BASH_ENV=str(startup),VITA_ACTOR_PYTHON=str(fake),ACTOR_LAUNCH_TEST_OUTPUT=str(output))
        def launch(*argv):
            output.unlink(missing_ok=True)
            subprocess.run(['bash',str(launcher),*argv],env=env,check=True,capture_output=True,text=True,timeout=10,
                           stdin=subprocess.DEVNULL)
            return json.loads(output.read_text())
        def refused(*argv):
            output.unlink(missing_ok=True)
            result=subprocess.run(['bash',str(launcher),*argv],env=env,capture_output=True,text=True,timeout=10,
                                  stdin=subprocess.DEVNULL)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertFalse(output.exists(),'nothing may start without one explicit policy')
            return result.stderr
        # No default policy: without a terminal to choose from, nothing starts.
        self.assertIn('Select the actor policy explicitly',refused())
        self.assertIn('Select the actor policy explicitly',refused('--seconds','5'))
        self.assertIn('Select the actor policy explicitly',refused('check','--steps','1'))
        self.assertIn('Conflicting',refused('sonic','--policy','supermimic'))
        self.assertIn("Unknown policy 'g1'",refused('--policy','g1'))
        for policy in ('sonic','supermimic'):
            args=launch(policy)
            self.assertEqual(args[1:4],['live','--policy',policy])
            for flag in ('--auto-start','--wait-pico','--manual-pico','--confirm-motion','--local-start'):
                self.assertIn(flag,args)
            self.assertNotIn('10.100.23.96',args)
        args=launch('supermimic','--seconds','5')
        self.assertIn('--local-start',args)
        self.assertEqual(args[-2:],['--seconds','5'])
        self.assertEqual(launch('--policy','supermimic','--seconds','5'),args)
        self.assertEqual(launch('--seconds','5','--policy=supermimic'),args)
        self.assertEqual(launch('supermimic','check','--steps','1')[1:],['check','--policy','supermimic','--steps','1'])
        self.assertEqual(launch('check','sonic','--steps','1')[1:],['check','--policy','sonic','--steps','1'])
        self.assertEqual(launch('shadow','--policy','sonic','--seconds','3')[1:],['shadow','--policy','sonic','--seconds','3'])
        # Interactive menu: a number selects; an empty answer starts nothing.
        import pty
        for answer,expected in ((b'2\n','supermimic'),(b'1\n','sonic'),(b'\n',None)):
            output.unlink(missing_ok=True)
            master,slave=pty.openpty()
            try:
                process=subprocess.Popen(['bash',str(launcher)],env=env,stdin=slave,stdout=subprocess.DEVNULL,stderr=slave)
                os.write(master,answer)
                code=process.wait(timeout=10)
            finally:
                os.close(slave);os.close(master)
            if expected is None:
                self.assertEqual(code,2);self.assertFalse(output.exists())
            else:
                self.assertEqual(code,0);self.assertEqual(json.loads(output.read_text())[1:4],['live','--policy',expected])
        startup.write_text('source() { echo "unexpected robot environment access" >&2; exit 99; }\n')
        subprocess.run(['bash',str(launcher),'--help'],env=env,check=True,capture_output=True,text=True,timeout=10)
        args=json.loads(output.read_text());self.assertEqual(args[1:],["check","--help"])
        subprocess.run(['bash',str(launcher),'configure','--help'],env=env,check=True,capture_output=True,text=True,timeout=10)
        args=json.loads(output.read_text())
        self.assertEqual(Path(args[0]).name,'configure_actor_s100.py')
        self.assertEqual(args[1:],["--help"])

    def test_device_profile_selects_local_target_and_receives_pico_but_not_auto_follow(self):
        args=self.load()
        self.assertEqual(args['robot_id'],'10.100.23.96')
        self.assertEqual(args['robot_config'],self.config)
        self.assertTrue(args['wait_pico']);self.assertFalse(args['auto_pico'])
        self.assertNotIn('confirm_motion',args)

    def test_profile_wrong_device_or_nonlocal_address_fails(self):
        with self.assertRaisesRegex(ValueError,'not a local'):self.load(addresses=('192.0.2.1',))
        self.machine.write_text('another-board')
        with self.assertRaisesRegex(ValueError,'another device'):self.load()

    def test_profile_config_or_selector_change_requires_review(self):
        self.config.write_text('changed')
        with self.assertRaisesRegex(ValueError,'configuration changed'):self.load()
        self.selector.write_text('gripper')
        with self.assertRaisesRegex(ValueError,'selection changed'):self.load()

    def test_missing_or_public_profile_fails_before_control(self):
        self.profile.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError,'local startup profile'):self.load()
        self.profile.unlink()
        with self.assertRaisesRegex(RuntimeError,'local startup profile'):self.load()

    def test_profile_invalid_handoff_or_negative_duration_fails(self):
        self.cfg['seconds']=-1;self.save()
        with self.assertRaises(ValueError):self.load()
        self.cfg['seconds']=120;self.cfg['pico_handoff']='anything';self.save()
        with self.assertRaises(ValueError):self.load()

    def test_profile_zero_selects_continuous_run(self):
        self.cfg['seconds']=0;self.save()
        self.assertEqual(self.load()['seconds'],0)

    def test_cli_local_duration_default_and_explicit_override(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_local
        spec=importlib.util.spec_from_file_location('actor_cli_duration_test',ROOT/'scripts/actor_pico.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        self.cfg['seconds']=0;self.save();defaults=self.load()
        runtime=NS(config={'artifacts':{'model':{}},'retargeting':{'implementation':'motion_stream_bridge_gmr_vita02a'}},reference=NS(delay=.2),
                retarget=NS(accelerated=True),
                model=NS(contract=NS(input_dim=1520)))
        for tail,expected in [([],0),(['--seconds','5'],5),(['--seconds','600'],600)]:
            captured=[]
            def live(runtime,args,write):
                captured.append(args);return dict(actions_sent=0)
            with (self.subTest(tail=tail),redirect_stdout(io.StringIO()),
                  patch.object(cli,'ActorPicoRuntime',return_value=runtime),
                  patch.object(cli,'live',side_effect=live),
                  patch.object(actor_pico_local,'load_local_start',return_value=defaults),
                  patch.object(sys,'argv',['actor_pico.py','live','--local-start','--auto-start',
                      '--confirm-motion','--root',str(self.root),*tail])):
                cli.main()
            self.assertEqual(captured[0].seconds,expected)


class SchedulingTests(unittest.TestCase):
    def test_control_priority_matches_original_operator(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_scheduling as sched
        with (patch.object(sched.os,'sched_setscheduler') as set_policy,
              patch.object(sched.os,'sched_getscheduler',return_value=os.SCHED_RR),
              patch.object(sched.os,'sched_getparam',return_value=os.sched_param(1))):
            self.assertEqual(sched.control_scheduler(),{'policy':'SCHED_RR','priority':1})
        self.assertEqual(set_policy.call_args.args[0:2],(0,os.SCHED_RR))
        self.assertEqual(set_policy.call_args.args[2].sched_priority,1)

    def test_gmr_worker_does_not_inherit_control_priority(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_scheduling as sched
        with patch.object(sched.os,'sched_setscheduler') as set_policy:
            sched.input_worker_scheduler()
        self.assertEqual(set_policy.call_args.args[0:2],(0,os.SCHED_OTHER))
        self.assertEqual(set_policy.call_args.args[2].sched_priority,0)

    def test_missing_realtime_permission_is_a_startup_error(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_scheduling as sched
        with patch.object(sched.os,'sched_setscheduler',side_effect=PermissionError('not permitted')):
            with self.assertRaises(PermissionError):sched.control_scheduler()


if __name__ == '__main__': unittest.main()
