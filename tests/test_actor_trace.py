"""Actual FBS records around WTT's model entry points; no device/model assets."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_sonic_planner import PlannerPipeline, Command, G1_INDEX
from deployment.vita_s100_aorta_30dof.actor_trace import STAGES
from locomotion_aorta.trace import Trace, Kind, Outcome
from locomotion_aorta.trace.reader import Reader


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'run.ftrace'

    def records(self):
        reader = Reader(self.path)
        records = list(reader)
        self.assertTrue(reader.integrity.complete, reader.integrity.issues)
        return [row for row in records if row['kind'] == Kind.USER_SPAN]

    def pipeline(self, trace):
        p = PlannerPipeline.__new__(PlannerPipeline)
        p.runtime = NS(trace=trace)
        p.seed = 1234
        p.rng = np.random.default_rng(p.seed)
        p.allowed_counts = (48, 52, 56, 60, 64)
        p.g1 = G1_INDEX
        p.order = list(range(28))
        p.moving_lookahead = 22
        p.idle_lookahead = 2
        p.observed_intervals = []
        def infer(outputs, inputs):
            began = time.monotonic_ns() // 1000
            result = np.zeros((1, 64, 36), np.float32)
            result[:, :, 3] = 1
            output = result, np.array([64], np.int32)
            p.observed_intervals.append((began, time.monotonic_ns() // 1000))
            return output
        p.planner = NS(run=infer)
        def bridge(outputs, inputs):
            began = time.monotonic_ns() // 1000
            output = [np.zeros((len(inputs['window_features']), 28), np.float32)]
            p.observed_intervals.append((began, time.monotonic_ns() // 1000))
            return output
        p.bridge = NS(run=bridge)
        return p

    def test_initial_background_map_and_control_thread_replan_map_are_separate(self):
        with Trace(self.path, stages=STAGES) as trace:
            p = self.pipeline(trace)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(p.start, np.zeros(28)).result(timeout=2)
                prediction = pool.submit(p.predict, 10, Command()).result(timeout=2)
                p.accept(prediction, 10)
        rows = sorted((r for r in self.records() if r['stage'] in (12+STAGES.index('planner_infer'), 12+STAGES.index('planner_map'))), key=lambda row: row['timestamp_us'])
        self.assertEqual(len(rows), len(p.observed_intervals))
        for row, (begin, end) in zip(rows, p.observed_intervals):
            self.assertLessEqual(row['timestamp_us'], begin)
            self.assertGreaterEqual(row['timestamp_us'] + row['duration_us'], end)
            self.assertGreaterEqual(row['duration_us'], 1)
        self.assertEqual([r['stage'] for r in rows], [12+STAGES.index('planner_infer'), 12+STAGES.index('planner_map')]*2)
        self.assertEqual(len({r['producer'] for r in rows[:3]}), 1)
        self.assertNotEqual(rows[2]['producer'], rows[3]['producer'])
        self.assertEqual(len({r['inference_id'] for r in rows}), 4)
        self.assertTrue(all(r['outcome'] == Outcome.OK for r in rows))

    def test_planner_failure_keeps_original_exception_and_failed_span(self):
        failure = RuntimeError('model failed')
        with Trace(self.path, stages=STAGES) as trace:
            p = self.pipeline(trace)
            with patch.object(p.planner, 'run', side_effect=failure):
                with self.assertRaises(RuntimeError) as raised:
                    p.start(np.zeros(28))
                self.assertIs(raised.exception, failure)
        rows = [r for r in self.records() if r['stage'] == 12+STAGES.index('planner_infer')]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['outcome'], Outcome.FAILED)

    def test_actor_observe_infer_decode_measure_actual_calls_separately(self):
        with Trace(self.path, stages=STAGES) as trace:
            runtime = ActorPicoRuntime.__new__(ActorPicoRuntime)
            runtime.trace = trace
            runtime.policy_name = 'synthetic-test'
            intervals = []
            def measured(function):
                def call(*args):
                    began = time.monotonic_ns() // 1000
                    result = function(*args)
                    intervals.append((began, time.monotonic_ns() // 1000))
                    return result
                return call
            runtime.policy = NS(order=(), observe=measured(lambda *args: np.zeros(3)),
                                decode=measured(lambda action: (NS(joints=()), {})))
            runtime.model = NS(backend='synthetic-test', infer=measured(lambda observation: np.ones(2)))
            command, observation, row = runtime.infer_motion_command(None, q=(), dq=(), root=(), gyro=())
            self.assertEqual(observation.shape, (3,))
            self.assertIsNone(row['raw_action'])
            identity = runtime.last_trace_inference_id
        rows = sorted(self.records(), key=lambda r: r['timestamp_us'])
        self.assertEqual(len(rows), len(intervals))
        for span, (begin, end) in zip(rows, intervals):
            self.assertLessEqual(span['timestamp_us'], begin)
            self.assertGreaterEqual(span['timestamp_us'] + span['duration_us'], end)
        self.assertEqual(rows[1]['inference_id'], identity)
        raw = list(Reader(self.path))
        action = [r for r in raw if r['kind'] == Kind.USER_MARK and r['stage'] == 12+STAGES.index('actor_action')]
        self.assertEqual(action[0]['values'], [1., 1.])

    def test_offline_cli_uses_one_run_recorder_and_off_has_no_trace_resource(self):
        spec = importlib.util.spec_from_file_location('actor_trace_cli', ROOT/'scripts/actor_pico.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                runtime = NS(config={'artifacts': {'model': {}}, 'body_profile': {}, 'retargeting': {}},
                    reference=NS(delay=.2), retarget=None, reference_source='pico', head=None,
                    model=NS(contract=NS(input_dim=1520)))
                runtime.policy = NS(contract=runtime.model.contract, profile={'ankle_command_space':'policy'})
                seen = []
                def check(runtime, args, write):
                    seen.append(runtime.trace)
                    if enabled:
                        self.assertIs(cli.trace_options(args)['trace_recorder'], runtime.trace)
                        with runtime.trace.span('actor_infer'):
                            pass
                    else:
                        self.assertEqual(cli.trace_options(args), {})
                    return {'actions_sent': 0}
                options = ['--trace=fbs', '--trace-output', str(self.path)] if enabled else []
                with (redirect_stdout(io.StringIO()), patch.object(cli, 'ActorPicoRuntime', return_value=runtime),
                      patch.object(cli, 'check', side_effect=check),
                      patch.object(sys, 'argv', ['actor_pico.py', 'check', '--root', self.temp.name, *options])):
                    cli.main()
                if enabled:
                    self.assertTrue(seen[0].session._closed)
                    self.assertEqual(len(self.records()), 1)
                else:
                    self.assertEqual(seen, [None])
                    self.assertFalse(self.path.exists())

    def test_actual_future_ready_accept_and_fusion_keep_old_and_new_result_ids(self):
        from test_actor_sonic_planner import Tests
        with Trace(self.path, stages=STAGES) as trace:
            reference = Tests().pending_reference(10)
            reference.runtime.trace = trace
            reference.pipeline = self.pipeline(trace)
            reference.pipeline.start(np.zeros(28))
            reference.trace_result_id = old = trace.new_id()
            with ThreadPoolExecutor(max_workers=1) as pool:
                reference.pool = pool
                reference._submit(reference.pipeline.predict, 10, Command())
                reference.future.result(timeout=2)
                incoming = reference.pending_trace['result_id']
                reference.tick(None, None, 1., [1, 0, 0, 0], robot_q=np.zeros(28))
                accepted = reference.trace_result_id
        reader = Reader(self.path)
        rows = list(reader)
        self.assertTrue(reader.integrity.complete, reader.integrity.issues)
        stage = lambda name: 12 + STAGES.index(name)
        ready, = [r for r in rows if r['stage'] == stage('planner_ready')]
        admit, = [r for r in rows if r['stage'] == stage('planner_accept')]
        fusion, = [r for r in rows if r['stage'] == stage('planner_fusion')]
        self.assertEqual(ready['reference_id'], incoming)
        self.assertEqual(admit['reference_id'], accepted)
        self.assertEqual(admit['integers'][0], incoming)
        self.assertEqual(fusion['reference_id'], accepted)
        self.assertEqual(fusion['integers'][:2], [old, incoming])
        self.assertNotEqual(ready['producer'], admit['producer'])
        self.assertEqual(reference.runtime.trace_planner_result_id, accepted)

    def test_grip_trace_preserves_invalid_input_open_and_records_real_decision(self):
        import test_actor_pico_gripper as fixture
        fixture.FakeClient.infos, fixture.FakeClient.created = {}, []
        clock = fixture.Clock()
        shared = fixture.shared(left=float('nan'), right=.7, received=clock(), stamp=1)
        sdk = NS(**vars(fixture.SDK), EndEffectorClient=fixture.FakeClient)
        with Trace(self.path, stages=STAGES) as trace:
            runner = fixture.GripperRunner(fixture.load_config(fixture.EXAMPLE), shared,
                connection='synthetic-controller', sdk=sdk, clock=clock, trace_recorder=trace)
            try:
                for client in runner.clients.values(): client.received=clock()
                runner.tick()
                self.assertEqual(runner.clients['left'].published[0][0].pos, (.112,))
                self.assertFalse(runner.gates['left'].valid)
            finally:
                runner.close()
        reader = Reader(self.path)
        rows = list(reader)
        self.assertTrue(reader.integrity.complete, reader.integrity.issues)
        inputs, = [r for r in rows if r['kind'] == Kind.INPUT_EVENT]
        decision = next(r for r in rows if r['kind'] == Kind.GRIP_DECISION and r['stream'] == 2)
        self.assertEqual(inputs['validity'][:2], [0, 1])
        self.assertEqual(decision['reference_id'], inputs['attempt_id'])
        self.assertEqual(decision['integers'], [0, 0, 0, 3])

    def test_gmr_ipc_cache_reuses_source_identity_and_original_endpoints(self):
        import queue
        from deployment.vita_s100_aorta_30dof.actor_pico_worker import PicoReferenceProcess
        with Trace(self.path, stages=STAGES) as trace:
            worker = PicoReferenceProcess.__new__(PicoReferenceProcess)
            worker.trace, worker.trace_previous, worker.trace_input_id = trace, None, 0
            worker.output = queue.Queue(maxsize=2)
            worker.process = NS(is_alive=lambda: True)
            worker.head_pose = NS(snapshot=lambda: None)
            worker.controls = None
            worker.previous = dict(state='ready', trace_process='synthetic-worker',
                pico_timestamp_ns=1794885630000123000, received_monotonic=1.,
                trace_points_us=(100, 150, 151, 170, 171, 90, 1794885630000123))
            first = worker.snapshot()
            second = worker.snapshot()
            self.assertEqual(first['trace_input_id'], second['trace_input_id'])
            self.assertEqual(first['received_monotonic'], second['received_monotonic'])
        reader = Reader(self.path)
        rows = list(reader)
        self.assertTrue(reader.integrity.complete, reader.integrity.issues)
        input_events = [r for r in rows if r['kind'] == Kind.INPUT_EVENT]
        self.assertEqual(len(input_events), 1)
        measured = [r for r in rows if r['kind'] == Kind.USER_SPAN]
        self.assertEqual([(r['timestamp_us'], r['duration_us']) for r in measured], [(100, 51), (151, 20)])
        self.assertTrue(all(r['reference_id'] == first['trace_input_id'] for r in measured))


if __name__ == '__main__':
    unittest.main()
