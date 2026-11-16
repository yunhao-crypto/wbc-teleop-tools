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
        with Trace(self.path, stages=('planner_infer', 'planner_map')) as trace:
            p = self.pipeline(trace)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(p.start, np.zeros(28)).result(timeout=2)
                prediction = pool.submit(p.predict, 10, Command()).result(timeout=2)
                p.accept(prediction, 10)
        rows = sorted(self.records(), key=lambda row: row['timestamp_us'])
        self.assertEqual(len(rows), len(p.observed_intervals))
        for row, (begin, end) in zip(rows, p.observed_intervals):
            self.assertLessEqual(row['timestamp_us'], begin)
            self.assertGreaterEqual(row['timestamp_us'] + row['duration_us'], end)
            self.assertGreaterEqual(row['duration_us'], 1)
        self.assertEqual([r['stage'] for r in rows], [12, 13, 12, 13])
        self.assertEqual(len({r['producer'] for r in rows[:3]}), 1)
        self.assertNotEqual(rows[2]['producer'], rows[3]['producer'])
        self.assertEqual(len({r['inference_id'] for r in rows}), 4)
        self.assertTrue(all(r['outcome'] == Outcome.OK for r in rows))

    def test_planner_failure_keeps_original_exception_and_failed_span(self):
        failure = RuntimeError('model failed')
        with Trace(self.path, stages=('planner_infer', 'planner_map')) as trace:
            p = self.pipeline(trace)
            with patch.object(p.planner, 'run', side_effect=failure):
                with self.assertRaises(RuntimeError) as raised:
                    p.start(np.zeros(28))
                self.assertIs(raised.exception, failure)
        rows = self.records()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['outcome'], Outcome.FAILED)

    def test_actor_observe_infer_decode_measure_actual_calls_separately(self):
        with Trace(self.path, stages=('actor_observe', 'actor_infer', 'actor_decode')) as trace:
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
            self.assertEqual(row['raw_action'], [1., 1.])
            identity = runtime.last_trace_inference_id
        rows = sorted(self.records(), key=lambda r: r['timestamp_us'])
        self.assertEqual(len(rows), len(intervals))
        for span, (begin, end) in zip(rows, intervals):
            self.assertLessEqual(span['timestamp_us'], begin)
            self.assertGreaterEqual(span['timestamp_us'] + span['duration_us'], end)
        self.assertEqual(rows[1]['inference_id'], identity)

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


if __name__ == '__main__':
    unittest.main()
