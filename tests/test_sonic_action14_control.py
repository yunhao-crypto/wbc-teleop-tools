"""Fake bridge and CLI checks. No Pico, ONNX session, or robot connection."""
import json
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from deployment.vita_s100_aorta_30dof.sonic_action14_bridge import JsonControlBridge
from deployment.vita_s100_aorta_30dof.sonic_action14_cli import main


BRIDGE = textwrap.dedent('''
import json, sys, time
mode = sys.argv[1]
print(json.dumps({"type": "hello", "protocol": 1, "exclusive_lease": False}), flush=True)
if mode == "silent":
    time.sleep(5)
    raise SystemExit
for line in sys.stdin:
    cmd = json.loads(line)
    if mode == "foreign" and cmd.get("op") == "event":
        reply = {"type": "reply", "ok": True, "op": "event", "req_id": "other"}
    elif mode == "reject":
        reply = {"type": "reply", "ok": False, "op": cmd.get("op"), "error": "no"}
    else:
        reply = {"type": "reply", "ok": True, "op": cmd.get("op"),
                 "req_id": cmd.get("req_id"), "status": {"current_action": "FIXED_STAND",
                 "status_age_ms": 1, "active": False}}
    print(json.dumps(reply), flush=True)
    if cmd.get("op") == "close":
        break
''')


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / 'bridge.py'
        self.script.write_text(BRIDGE)
        self.script.chmod(self.script.stat().st_mode | stat.S_IEXEC)

    def tearDown(self):
        self.tmp.cleanup()

    def bridge(self, mode, timeout=.3):
        wrapper = Path(self.tmp.name) / f'bridge-{mode}'
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {self.script} {mode} "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        return JsonControlBridge(wrapper, timeout=timeout)

    def test_correlated_reply_and_close(self):
        bridge = self.bridge('ok')
        try:
            status = bridge.status()
            self.assertEqual(status['current_action'], 'FIXED_STAND')
            reply = bridge.event('req-1', 'humanoid_tracking.query', {})
            self.assertEqual(reply['req_id'], 'req-1')
        finally:
            bridge.close()
            self.assertIsNotNone(bridge.process.poll())

    def test_foreign_request_and_rejection_do_not_hang(self):
        foreign = self.bridge('foreign')
        try:
            with self.assertRaises(RuntimeError):
                foreign.event('req-1', 'humanoid_tracking.query', {})
        finally:
            foreign.close()
        rejected = self.bridge('reject')
        try:
            with self.assertRaises(RuntimeError):
                rejected.activate('req-2', 'profile_wtt_native')
        finally:
            rejected.close()

    def test_lost_reply_times_out_and_close_returns(self):
        bridge = self.bridge('silent', timeout=.2)
        try:
            with self.assertRaises(TimeoutError):
                bridge.status()
        finally:
            bridge.close()
            self.assertIsNotNone(bridge.process.poll())


class PostureTests(unittest.TestCase):
    def test_quiet_passive_waits_then_requests_stand_once(self):
        from deployment.vita_s100_aorta_30dof.sonic_action14_cli import ensure_supported_posture
        now = [0.]
        calls = []
        states = [{'current_action': 'QUIET_PASSIVE', 'status_age_ms': 20}]

        def clock():
            return now[0]

        def sleep(seconds):
            now[0] += seconds
            if now[0] > 4. and states[0]['current_action'] == 'QUIET_PASSIVE' and calls:
                states[0] = {'current_action': 'FIXED_STAND', 'status_age_ms': 30}

        def request():
            calls.append(now[0])

        class Control:
            def status(self):
                return dict(states[0])

        self.assertEqual(ensure_supported_posture(Control(), request, clock=clock, sleep=sleep), 'FIXED_STAND')
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0], 4.)

    def test_ready_posture_does_not_request_stand(self):
        from deployment.vita_s100_aorta_30dof.sonic_action14_cli import ensure_supported_posture

        class Control:
            def status(self):
                return {'current_action': 'HUMANOID', 'status_age_ms': 10}

        def request():
            raise AssertionError('request')

        self.assertEqual(ensure_supported_posture(
            Control(), request, clock=lambda: 0., sleep=lambda _seconds: None), 'HUMANOID')

    def test_ctrl_c_waits_until_passive_status(self):
        from deployment.vita_s100_aorta_30dof.sonic_action14_cli import confirm_passive
        now = [0.]
        calls = []
        state = {'current_action': 'HUMANOID_TRACKING', 'status_age_ms': 20}

        def clock():
            return now[0]

        def sleep(seconds):
            now[0] += seconds
            if calls:
                state['current_action'] = 'PASSIVE'

        class Control:
            def status(self):
                return dict(state)

        self.assertEqual(confirm_passive(Control(), lambda: calls.append(now[0]), clock=clock, sleep=sleep), 'PASSIVE')
        self.assertEqual(calls, [0.])

    def test_passive_cancel_is_sent_when_status_bridge_is_dead(self):
        from deployment.vita_s100_aorta_30dof.sonic_action14_cli import confirm_passive
        calls = []

        class Control:
            def status(self):
                raise RuntimeError('action14 bridge exited')

        with self.assertRaises(TimeoutError):
            confirm_passive(Control(), lambda: calls.append(1), clock=lambda: 0., sleep=lambda _seconds: None, timeout=0.)
        self.assertEqual(calls, [1])

    def test_other_actions_are_not_replaced(self):
        from deployment.vita_s100_aorta_30dof.sonic_action14_cli import ensure_supported_posture

        class Control:
            def status(self):
                return {'current_action': 'HUMANOID_TELEOP', 'status_age_ms': 10}

        with self.assertRaises(RuntimeError):
            ensure_supported_posture(Control(), lambda: None, clock=lambda: 0., sleep=lambda _seconds: None)


class CliTests(unittest.TestCase):
    def test_plan_has_no_model_or_endpoint(self):
        with patch('deployment.vita_s100_aorta_30dof.sonic_action14_cli.verify_model_file',
                   side_effect=AssertionError('model read')), \
             patch('socket.socket', side_effect=AssertionError('network')):
            self.assertEqual(main(['--plan']), 0)

    def test_plan_gripper_does_not_construct_a_client(self):
        from io import StringIO
        from contextlib import redirect_stdout
        example = str(Path(__file__).resolve().parents[1] / 'config' / 'pico_op3_gripper.example.json')
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.load_sdk',
                   side_effect=AssertionError('sdk')), \
             patch('deployment.vita_s100_aorta_30dof.actor_pico_gripper.GripperRunner',
                   side_effect=AssertionError('runner')), \
             patch('socket.socket', side_effect=AssertionError('network')):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['--plan', '--gripper-config', example]), 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload['gripper_opened'])
        self.assertEqual(payload['gripper_device'], 'AGILINK_OP3')

    def test_live_flags_are_refused_without_creating_a_session(self):
        with patch('deployment.vita_s100_aorta_30dof.sonic_action14_cli.verify_model_file',
                   side_effect=AssertionError('model read')):
            with self.assertRaises(SystemExit):
                main([])
            with self.assertRaises(SystemExit):
                main(['--confirm-motion'])

    def test_import_does_not_start_a_process(self):
        import deployment.vita_s100_aorta_30dof.sonic_action14_cli as cli
        import deployment.vita_s100_aorta_30dof.sonic_action14_orchestrator as orch
        self.assertTrue(callable(cli.main))
        self.assertTrue(callable(orch.Action14Session))


if __name__ == '__main__':
    unittest.main()
