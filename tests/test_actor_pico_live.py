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
    def __init__(self, clock): self.clock, self.fault, self.last_stamp = clock, None, None
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
            retarget_ms=.1, ik_max_residual_rad=0.)


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
        self.pico = Pico(self.clock)
        self.args = NS(robot_config=self.config, robot_id='TEST-ONLY', startup_timeout=5.,
                       seconds=3.12, stand_only=False)
        self.confirm = lambda *_:self.clock.sleep(.02)

    def run_trial(self, **kwargs):
        write = kwargs.pop('write', lambda row, obs=None:self.rows.append(row))
        with redirect_stdout(io.StringIO()):
            return run_live(self.runtime, self.args, write,
                self.controller, self.pico, self.stop, service_alive=lambda:True,
                confirm=self.confirm, clock=self.clock, **kwargs)

    def assert_cancelled(self):
        self.assertEqual(self.controller.status.current_action, 'PASSIVE')
        self.assertIn('CANCEL', [r.operation.value for r in self.transport.requests])

    def test_real_sdk_protocol_and_full_commands_through_prepare_and_tracking(self):
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
        worker.fault=lambda:dict(state='ready',received_monotonic=self.clock.monotonic()-.2)
        result=self.run_trial(manual_pico=worker,poll_command=command)
        self.assertTrue(result['exit_confirmed'])
        self.assertTrue(any(r.get('record_type')=='pico_switch_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

    def test_manual_non_neutral_reference_is_never_queued(self):
        worker, command=self.manual_trial()
        original=worker.snapshot
        def non_neutral():
            snapshot=original()
            if snapshot['state']=='ready':snapshot['q']=snapshot['q']+.2
            return snapshot
        with patch.object(worker,'snapshot',non_neutral):
            self.run_trial(manual_pico=worker,poll_command=command)
        self.assertTrue(any(r.get('record_type')=='pico_switch_rejected' for r in self.rows))
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

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
              patch.object(sdkmain_adapter,'RealtimeBodyTransport',return_value=self.transport),
              patch('locomotion_aorta.pull.PullController',return_value=self.controller),
              patch.object(pico_input,'create_pico_source',side_effect=AssertionError('Pico reader forbidden')),
              patch.object(cli.subprocess,'Popen',side_effect=AssertionError('Pico service forbidden')),
              patch.object(actor_pico_live,'run_live',return_value={'actions_sent':0}) as runner):
            result=cli.live(runtime,self.args,lambda *_:None)
        self.assertEqual(result['actions_sent'],0)
        self.assertIsNone(runner.call_args.args[4])
        self.assertIsNone(runner.call_args.kwargs['manual_pico'])

    def test_confirmation_declined_has_no_control_request_or_command(self):
        def decline(*_): raise RuntimeError('declined')
        self.confirm = decline
        with self.assertRaisesRegex(RuntimeError, 'declined'): self.run_trial()
        self.assertEqual((self.transport.commands, self.transport.requests), ([], []))

    def test_interrupt_during_confirmation_never_enters(self):
        self.confirm = lambda *_:self.stop.set()
        self.assertFalse(self.run_trial()['entered'])
        self.assertEqual(self.transport.requests, [])

    def test_interrupt_after_publish_cancels_once_without_extra_command(self):
        self.transport.after_publish = lambda c:self.stop.set()
        self.assertEqual(self.run_trial()['actions_sent'], 1)
        self.assert_cancelled()

    def test_stale_feedback_after_entry_is_not_published(self):
        self.transport.after_publish = lambda c:setattr(self.transport, 'state_fault',
            lambda s:replace(s, ts_state_pub_us=s.ts_state_pub_us-200_000))
        with self.assertRaisesRegex(TimeoutError, 'source is stale'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
        self.assert_cancelled()

    def test_stale_pico_after_entry_cancels_without_retry(self):
        self.transport.after_publish = lambda c:setattr(self.pico, 'fault', lambda:None)
        with self.assertRaisesRegex(TimeoutError, 'Pico input'): self.run_trial()
        self.assertLess(len(self.transport.commands), 10)
        self.assert_cancelled()

    def test_reject_drop_timeout_and_ownership_changes_stop_publication(self):
        for change in (dict(reject_count=1), dict(ingress_drop_count=1), dict(timeout_hold=True),
                       dict(client_id='other'), dict(generation=8)):
            with self.subTest(change=change):
                self.setUp()
                self.transport.after_publish = lambda c:setattr(self.transport, 'status_fault',
                    lambda s:replace(s, **change))
                with self.assertRaisesRegex(RuntimeError, 'ownership/state'): self.run_trial()
                self.assertEqual(len(self.transport.commands), 1)
                self.assertIn('CANCEL', [r.operation.value for r in self.transport.requests])

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

    def test_scheduler_delay_cannot_hide_publication_gap_with_fresh_new_state(self):
        self.args.stand_only = True
        original = self.clock.sleep
        def delayed(seconds):
            original(seconds)
            if self.transport.commands: original(.101)
        with patch.object(self.clock,'sleep',delayed):
            with self.assertRaisesRegex(TimeoutError,'publication gap'): self.run_trial()
        self.assertEqual(len(self.transport.commands),1)
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

    def test_config_mismatch_or_changed_after_confirmation_refuses_entry(self):
        def changed(*_):
            root = robot_config(); root['parallel_ankle']['pd']['strategy'] = 'position_motor_gains'
            self.config.write_text(json.dumps(root))
        self.confirm = changed
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

    def test_publication_stall_records_already_sent_frame_and_cancels(self):
        self.transport.after_publish = lambda c:self.clock.sleep(.101)
        with self.assertRaisesRegex(TimeoutError, 'publication stalled'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
        row = next(r for r in self.rows if r.get('record_type') == 'publish_stall')
        self.assertEqual((row['actions_sent'],row['session_id'],row['sequence']),(1,7,1))
        self.assert_cancelled()

    def test_logging_stall_cannot_publish_another_frame_after_long_gap(self):
        self.args.stand_only = True
        def write(row, obs=None):
            self.rows.append(row)
            if 'sent_body_targets' in row: self.clock.sleep(.101)
        with self.assertRaisesRegex(TimeoutError,'publication gap'): self.run_trial(write=write)
        self.assertEqual(len(self.transport.commands),1)
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

    def test_auto_waits_full_damping_once_then_fixed_pose_and_onnx_without_pico(self):
        self.auto_setup()
        result = self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND'])
        self.assertGreaterEqual(self.startup_calls[0][1],104.)
        stages = [r['stage'] for r in self.rows if 'stage' in r]
        self.assertIn('pose_prepare',stages);self.assertIn('prepare_stand',stages)
        self.assertNotIn('pico_tracking',stages)
        self.assertFalse(self.worker.connected)
        self.assertEqual([r.operation.value for r in self.transport.requests],['PREPARE','COMMIT','CANCEL'])
        self.assertTrue(result['exit_confirmed'])
        pose = [r for r in self.rows if r.get('stage')=='pose_prepare']
        q = np.array([[j['q'] for j in r['sent_body_targets']] for r in pose])
        self.assertLessEqual(np.max(np.abs(np.diff(q,axis=0)))/.02,.150001)
        self.assertTrue(all(not r['onnx_control_enabled'] for r in pose))
        np.testing.assert_allclose(q[0],np.zeros(31),atol=1e-6)
        self.assertGreater(q[-1,3],.39)
        self.assert_cancelled()

    def test_auto_existing_fixed_stand_does_not_cycle_through_passive(self):
        self.auto_setup('FIXED_STAND')
        self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assert_cancelled()

    def test_auto_confirmation_decline_has_no_fixed_or_external_requests(self):
        self.auto_setup()
        self.confirm = lambda *_:(_ for _ in ()).throw(RuntimeError('declined'))
        with self.assertRaisesRegex(RuntimeError,'declined'):self.auto_trial()
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

    def test_auto_stuck_joint_stops_pose_and_cancels_before_onnx_control(self):
        self.auto_setup('FIXED_STAND')
        self.transport.state_fault=lambda s:replace(s,q=np.zeros(31,np.float32))
        with self.assertRaisesRegex(RuntimeError,'tracking error'):self.auto_trial()
        self.assertTrue(self.transport.commands)
        self.assertFalse(any(r.get('stage')=='prepare_stand' for r in self.rows))
        self.assert_cancelled()

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

    def test_auto_pose_timeout_does_not_enable_onnx_control(self):
        self.auto_setup('FIXED_STAND');self.args.seconds=20.
        original=self.transport.state_fault
        # Hold a fixed 0.06 rad residual near the endpoint without accumulating
        # lag on every frame; tracking error stays below the 0.10 rad abort.
        def residual(s):
            s=original(s)
            if self.transport.commands:
                q=s.q.copy()
                target=self.transport.commands[-1].joints[3].q
                q[3]=max(0.,target-.06)
                return replace(s,q=q)
            return s
        self.transport.state_fault=residual
        with self.assertRaisesRegex(TimeoutError,'did not settle'):self.auto_trial()
        self.assertFalse(any(r.get('stage')=='prepare_stand' for r in self.rows))
        self.assert_cancelled()

    def test_auto_fixed_timeout_cancels_and_never_enters_external(self):
        self.auto_setup()
        def no_activation(name):
            self.startup_calls.append((name,self.clock.monotonic()))
        self.request_action=no_activation
        with self.assertRaisesRegex(TimeoutError,'FIXED_STAND did not settle'):self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND','CANCEL'])
        self.assertEqual(self.transport.requests,[])

    def test_auto_invalid_fixed_gains_reject_before_confirmation_or_motion(self):
        self.auto_setup()
        config=robot_config()
        config['actions']['FIXED_STAND']['params']['keyframes'][0]['kp']=[999.]*42
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'outside'):self.auto_trial()
        self.assertEqual(self.startup_calls,[])
        self.assertEqual(self.transport.requests,[])

    def test_auto_early_pico_request_is_not_queued_during_pose_preparation(self):
        self.auto_setup('FIXED_STAND')
        self.auto_trial(poll_command=lambda:'connect' if len(self.transport.commands)==20 else None)
        self.assertFalse(self.worker.connected)
        self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))

    def test_auto_preparation_preserves_zero_policy_history_until_pose_settles(self):
        self.auto_setup('FIXED_STAND')
        def published(command):
            if command.sequence <= 200:self.assertEqual(len(self.runtime.policy.history),0)
        self.transport.after_publish=published
        result=self.auto_trial()
        self.assertTrue(result['pose_prepared'])
        self.assertTrue(result['onnx_control_enabled'])

    def test_auto_excessive_measured_speed_during_pose_cancels(self):
        self.auto_setup('FIXED_STAND')
        original=self.transport.state_fault
        def fast(s):
            s=original(s)
            return replace(s,dq=np.full(31,.30,np.float32)) if self.transport.commands else s
        self.transport.state_fault=fast
        with self.assertRaisesRegex(RuntimeError,'joint speed'):self.auto_trial()
        self.assertEqual(len(self.transport.commands),1)
        self.assert_cancelled()

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

    def test_auto_fixed_completion_between_status_frames_requires_settled_feedback(self):
        self.auto_setup('')
        def action(name):
            self.startup_calls.append((name,self.clock.monotonic()))
            self.transport.value=replace(self.transport.value,
                current_action='' if name=='FIXED_STAND' else 'PASSIVE')
        self.request_action=action
        self.auto_trial()
        self.assertEqual([x[0] for x in self.startup_calls],['FIXED_STAND'])
        self.assertTrue(any(r.get('record_type')=='fixed_stand_confirmed' for r in self.rows))
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
                self.assertLess(kw['q'][10],self.runtime.policy.lower[10])
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

    def test_online_auto_pico_stale_or_non_neutral_stays_in_onnx_stand(self):
        for stale in (True,False):
            self.setUp();self.online_setup();self.args.wait_pico=self.args.auto_pico=True
            original=self.worker.snapshot
            def bad():
                value=original()
                if stale:value['received_monotonic']-=.2
                else:value['q']=value['q']+.2
                return value
            self.worker.snapshot=bad
            self.auto_trial()
            self.assertFalse(any(r.get('stage')=='pico_tracking' for r in self.rows))
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
        self.assertEqual(len(candidate['observation']),1520)
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
        plan=StartupPlan(self.runtime,envelope,self.config,online=True)
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

    def test_no_argument_launcher_selects_attended_device_bound_motion_and_keeps_check_explicit(self):
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
        subprocess.run(['bash',str(launcher)],env=env,check=True,capture_output=True,text=True,timeout=10)
        args=json.loads(output.read_text())
        self.assertEqual(args[1],'live')
        for flag in ('--auto-start','--wait-pico','--manual-pico','--confirm-motion','--local-start'):
            self.assertIn(flag,args)
        self.assertNotIn('10.100.23.96',args)
        subprocess.run(['bash',str(launcher),'check','--steps','1'],env=env,check=True,capture_output=True,text=True,timeout=10)
        args=json.loads(output.read_text());self.assertEqual(args[1:],["check","--steps","1"])

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

    def test_profile_invalid_handoff_or_unbounded_duration_fails(self):
        self.cfg['seconds']=1000;self.save()
        with self.assertRaises(ValueError):self.load()
        self.cfg['seconds']=120;self.cfg['pico_handoff']='anything';self.save()
        with self.assertRaises(ValueError):self.load()


if __name__ == '__main__': unittest.main()
