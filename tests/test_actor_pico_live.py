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
)


class Clock:
    def __init__(self): self.now = 100.
    def monotonic(self): return round(self.now, 9)
    def time_ns(self): return 1_800_000_000_000_000_000+round(self.now*1e9)
    def sleep(self, seconds): self.now += max(0., seconds)


def robot_config():
    return dict(parallel_ankle=dict(pd=dict(strategy='virtual_pd'),
                geometry=dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235)),
        actions=dict(EXTERNAL=dict(generator='builtin::external', params=dict(
            ankle_virtual_pd_intent=True, command_timeout_ms=1000,
            safe_hold=dict(kp=[10.]*42, kd=[1.]*42), command_limits=dict(default=dict(
                max_kp=500., max_kd=50., max_abs_dq_rad_s=25., allowed_modes=[10]))))))


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
        with redirect_stdout(io.StringIO()):
            return run_live(self.runtime, self.args, lambda row, obs=None:self.rows.append(row),
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
        result = self.run_trial()
        self.assertTrue(result['exit_confirmed'])
        self.assertFalse(any(r.get('stage') == 'pico_tracking' for r in self.rows))

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

    def test_expensive_inference_is_never_sent_and_cancels(self):
        original = self.runtime.infer_motion_command
        def expensive(*a, **kw):
            result = original(*a, **kw)
            if self.transport.commands: self.clock.sleep(.021)
            return result
        with patch.object(self.runtime, 'infer_motion_command', expensive):
            with self.assertRaisesRegex(TimeoutError, '20 ms'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
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

    def test_logging_overrun_after_publish_cancels_without_catchup(self):
        self.transport.after_publish = lambda c:self.clock.sleep(.021)
        with self.assertRaisesRegex(TimeoutError, 'Complete policy cycle'): self.run_trial()
        self.assertEqual(len(self.transport.commands), 1)
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


if __name__ == '__main__': unittest.main()
