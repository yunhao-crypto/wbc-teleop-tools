"""Offline profiling checks: no Aorta session, controller or device is created."""
from dataclasses import dataclass
import json
import importlib.util
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_profiling import ProfilingSession
from deployment.vita_s100_aorta_30dof.actor_pico_live import Inputs, StaleStateError



def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def body_state(age=0):
    return NS(runtime_id=10, state_id=123, roster_version=1, q=(0.,), dq=(0.,), error_code=(0,),
        received_monotonic_ns=time.monotonic_ns()-int(age*1e9),
        received_realtime_ns=time.time_ns()-int(age*1e9),
        ts_state_real_us=time.time_ns()//1000-int(age*1e6),
        ts_state_pub_us=time.time_ns()//1000-int(age*1e6))


@dataclass(frozen=True)
class Ref:
    state_id: int


@dataclass(frozen=True)
class Target:
    pos: tuple = (.055,)


@dataclass(frozen=True)
class EEState:
    state_id: int = 42
    ts_state_pub_us: int = 123456
    status_valid_mask: int = 7
    ready: bool = True
    output_allowed: bool = True
    inhibit_reason: int = 0
    touch_enabled: bool = False
    pos: tuple = (.05,)
    sample_ref: Ref = Ref(42)


class ProfileTests(unittest.TestCase):
    def test_disabled_body_path_adds_no_clock_file_or_recorder_calls(self):
        state = body_state()
        controller = NS(wait_state=lambda timeout: state)
        inputs = Inputs(NS(config={'pico_max_age_s': 1.}), controller, None, time)
        with (patch.object(inputs, 'validate_state', return_value=(1, 2, 3, 4)),
              patch('pathlib.Path.open', side_effect=AssertionError('profiling file opened')),
              patch('time.monotonic_ns', side_effect=AssertionError('profiling clock read')),
              patch('time.time_ns', side_effect=AssertionError('profiling clock read'))):
            self.assertEqual(inputs.read(), (state, 1, 2, 3, 4))
        self.assertIsNone(inputs.profiling)
        self.assertIsNone(inputs.profile_read)

    def test_stale_body_read_is_recorded_before_existing_guard_rejects(self):
        state = body_state(.14)
        seen = []
        controller = NS(wait_state=lambda timeout: seen.append(timeout) or state)
        runtime = NS(config={'pico_max_age_s': 1.}, policy=NS(state=lambda _:None))
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'profile'
            with ProfilingSession(prefix) as profile:
                inputs = Inputs(runtime, controller, None, time, profiling=profile)
                with patch.dict(sys.modules, {
                        'deployment.vita_s100_aorta_30dof.sdkmain_adapter':
                        NS(validate_body_arrays=lambda _: None)}):
                    with self.assertRaises(StaleStateError):
                        inputs.read(0)
            reads = [r for r in rows(str(prefix)+'.body.jsonl') if r['record_type']=='body_state_read']
            self.assertEqual(seen, [0])
            self.assertEqual(len(reads), 1)
            self.assertEqual(reads[0]['state_id'], 123)
            self.assertEqual(reads[0]['source_state_ts_us'], state.ts_state_real_us)
            self.assertGreaterEqual(reads[0]['feedback_age_ns'], 140_000_000)

    def test_gripper_hook_keeps_one_feedback_snapshot_and_original_publication(self):
        import test_actor_pico_gripper as fixture
        case = fixture.RunnerTests('test_startup_validates_both_sides_before_entry')
        case.setUp()
        observed = []
        def publish(client, target, latest, position, received, **kw):
            observed.append((client.side, position, received))
            return client.publish(target, input_observations=(position.sample_ref,))
        runner = fixture.GripperRunner(case.config, case.shared, connection='controller',
            sdk=case.sdk, clock=case.clock, profiling=NS(publish_gripper=publish))
        case.clients = {client.side: client for client in fixture.FakeClient.created}
        self.addCleanup(runner.close)
        for client in case.clients.values():
            client.set(case.clock.now)
        case.grips(0., 0., 1.)
        with patch.object(case.clients['left'], 'motion_feedback', wraps=case.clients['left'].motion_feedback) as read:
            runner.tick()
        self.assertEqual(read.call_count, 1)
        self.assertEqual([r[0] for r in observed], ['left', 'right'])
        self.assertEqual(observed[0][2], case.clients['left'].received)
        self.assertEqual(case.clients['left'].published[0][1], (observed[0][1].sample_ref,))

    def test_body_publication_retains_input_and_call_association(self):
        state = body_state()
        calls = []
        command = NS(input_state_id=state.state_id, input_state_ts_real_us=state.ts_state_real_us,
                     joints=(NS(q=.1),))
        sent = NS(session_id=77, sequence=88)
        controller = NS(wait_state=lambda _: state,
                        publish=lambda cmd, **kw: calls.append((cmd, kw)) or sent)
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'profile'
            with ProfilingSession(prefix) as profile:
                measured, read = profile.body.read_body(controller, 0, phase='test')
                result = profile.body.publish_body(controller, command, measured, read,
                    phase='test', target_names=('joint',), required_fields=31,
                    model_timing=dict(inference_ms=4.8, observation_ms=4.0))
            self.assertIs(result, sent)
            self.assertIs(calls[0][0], command)
            records = rows(str(prefix)+'.body.jsonl')
            cycle = next(r for r in records if r['record_type']=='body_cycle')
            self.assertEqual(cycle['input_read_cycle'], read['cycle'])
            self.assertEqual(cycle['session_id'], 77)
            self.assertEqual(cycle['input_state_id'], state.state_id)
            self.assertEqual(cycle['model_timing_ms']['inference_ms'], 4.8)
            self.assertGreaterEqual(cycle['receipt_to_publish_return_ns'], cycle['publish_call_ns'])

    def test_gripper_uses_original_reference_and_receipt_without_an_extra_read(self):
        state, target, observed = EEState(), Target(), []
        sent = NS(runtime_id=10, layout_id=11, control_id=12, key=NS(session_id=77, sequence=89),
                  ts_client_created_us=111, ts_client_send_us=112, target=target,
                  input_observations=(state.sample_ref,))
        client = NS(side='left', publish=lambda target, **kw: observed.append(kw) or sent)
        received = time.monotonic()-.007
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'profile'
            with ProfilingSession(prefix) as profile:
                profile.ee_owned_by_worker = True
                result = profile.publish_gripper(client, target, state, state, received,
                                                  phase='wtt_open', rate=50)
                profile.finish_grippers()
            self.assertIs(result, sent)
            self.assertEqual(observed, [dict(input_observations=(state.sample_ref,))])
            body, ee = rows(str(prefix)+'.body.jsonl'), rows(str(prefix)+'.ee.jsonl')
            self.assertEqual(body[0]['run_id'], ee[0]['run_id'])
            cycle = next(r for r in ee if r['record_type']=='cycle')
            self.assertEqual(cycle['position_received_monotonic_ns'], int(received*1e9))
            self.assertEqual(cycle['sequence'], 89)
            self.assertGreaterEqual(cycle['feedback_age_ns'], 7_000_000)

    def test_sdk_call_timing_excludes_slow_jsonl_writes(self):
        clock = NS(now=10.)
        state = body_state()
        state.received_monotonic_ns = 9_993_000_000
        command = NS(input_state_id=123, input_state_ts_real_us=state.ts_state_real_us,
                     joints=(NS(q=.1),))
        def publish(*args, **kwargs):
            clock.now += .001
            return NS(session_id=77, sequence=88)
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'profile'
            with ProfilingSession(prefix) as profile:
                profile.body.monotonic = lambda: clock.now
                original_write = profile.body.write
                def slow_write(row):
                    clock.now += .010
                    original_write(row)
                with patch.object(profile.body, 'write', side_effect=slow_write):
                    profile.body.publish_body(NS(publish=publish), command, state,
                        dict(cycle=0, state_read_call_ns=10), phase='test',
                        target_names=('joint',), required_fields=31)
            cycle = next(r for r in rows(str(prefix)+'.body.jsonl') if r['record_type']=='body_cycle')
            self.assertAlmostEqual(cycle['publish_call_ns'], 1_000_000, delta=2)
            self.assertAlmostEqual(cycle['receipt_to_publish_return_ns'], 8_000_000, delta=2)
            self.assertGreaterEqual(clock.now, 10.020)

    def test_log_collision_never_overwrites_an_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d)/'profile'
            Path(str(prefix)+'.body.jsonl').write_text('keep this\n')
            with self.assertRaises(FileExistsError):
                ProfilingSession(prefix)
            self.assertEqual(Path(str(prefix)+'.body.jsonl').read_text(), 'keep this\n')

    def test_publication_fault_survives_both_write_and_close_errors(self):
        state = body_state()
        command = NS(input_state_id=123, input_state_ts_real_us=state.ts_state_real_us,
                     joints=(NS(q=.1),))
        primary = TimeoutError('original SDK fault')
        def fail(*args, **kwargs):
            raise primary
        with tempfile.TemporaryDirectory() as d:
            profile = ProfilingSession(Path(d)/'profile')
            with self.assertRaises(TimeoutError) as raised:
                with profile:
                    with (patch.object(profile.body, 'write', side_effect=OSError('disk full')),
                          patch.object(profile.body.stream, 'close', side_effect=OSError('close failed'))):
                        try:
                            profile.body.publish_body(NS(publish=fail), command, state,
                                dict(cycle=0, state_read_call_ns=10), phase='test',
                                target_names=('joint',), required_fields=31)
                        except BaseException as error:
                            profile.body.close(error)
                            raise
            self.assertIs(raised.exception, primary)
            profile.body.stream.close()


class SwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('profile_actor_cli', ROOT/'scripts/actor_pico.py')
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def test_default_and_explicit_off_do_not_load_recorder_or_enable_sdk_timing(self):
        for options in ([], ['--no-profiling'], ['--profiling']):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as d:
                runtime = NS(config={'artifacts': {'model': {}}, 'body_profile': {}, 'retargeting': {}},
                    reference=NS(delay=.2), retarget=None, reference_source='pico', head=None,
                    model=NS(contract=NS(input_dim=1520)))
                runtime.policy = NS(contract=runtime.model.contract, profile={'ankle_command_space':'policy'})
                captured = []
                def live(runtime, args, write):
                    captured.append(args)
                    self.assertEqual(os.environ['LOCOMOTION_AORTA_PUBLISH_TIMING'],
                                     '1' if args.profiling else '0')
                    return dict(actions_sent=0)
                with (redirect_stdout(io.StringIO()), patch.dict(os.environ),
                      patch.object(self.cli, 'ActorPicoRuntime', return_value=runtime),
                      patch.object(self.cli, 'live', side_effect=live),
                      patch.object(sys, 'argv', ['actor_pico.py','live','--confirm-motion',
                          '--robot-id','offline-test','--robot-config','/not/read.yaml',
                          '--root', d, *options])):
                    self.cli.main()
                self.assertEqual(captured[0].profiling, '--profiling' in options)
                if not captured[0].profiling:
                    self.assertIsNone(captured[0].profiling_log)
                self.assertEqual(list(Path(d).rglob('*.body.jsonl')), [])
                self.assertEqual(list(Path(d).rglob('*.ee.jsonl')), [])

    def test_disabled_import_does_not_load_profiling_module(self):
        import subprocess
        result = subprocess.run([sys.executable, '-c',
            "import importlib.util,sys; "
            "s=importlib.util.spec_from_file_location('cli',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            "assert 'deployment.vita_s100_aorta_30dof.actor_pico_profiling' not in sys.modules",
            str(ROOT/'scripts/actor_pico.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_switches_fail_before_runtime_or_network_creation(self):
        for options in (['check', '--profiling'],
                        ['live', '--no-profiling', '--profiling-log', '/not-created'],
                        ['live', '--profiling', '--no-profiling']):
            with (self.subTest(options=options), redirect_stderr(io.StringIO()),
                  patch.object(self.cli, 'ActorPicoRuntime', side_effect=AssertionError('runtime created')),
                  patch.object(sys, 'argv', ['actor_pico.py', *options]),
                  self.assertRaises(SystemExit) as result):
                self.cli.main()
            self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
