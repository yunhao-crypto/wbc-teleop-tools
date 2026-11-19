"""Offline fault injection: no Pico SDK, hardware, or services."""
import io
import json
import os
import queue
import tempfile
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/deployment/unitree_g1_ros2_29dof/src')]
from deployment.vita_s100_aorta_30dof.actor_pico_source_writer import SourceWriter
from deployment.vita_s100_aorta_30dof import actor_pico_worker as worker


class BlockedSink(io.StringIO):
    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.threads = []

    def write(self, text):
        self.threads.append(threading.get_ident())
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError('test did not release sink')
        return super().write(text)


class SourceWriterTests(unittest.TestCase):
    def test_slow_io_saturation_owned_buffers_and_bounded_close(self):
        sink = BlockedSink()
        writer = SourceWriter(sink=sink, header={'model': 'test'}, capacity=2)
        self.assertTrue(sink.entered.wait(1))
        array = np.ones((24, 7))
        row = dict(record_type='gmr_frame', body_poses=array, timing={'x': [1]})
        self.assertTrue(writer.enqueue(row))
        array[:] = 9
        row['timing']['x'][0] = 8
        self.assertTrue(writer.enqueue(row))
        begin = time.monotonic()
        for _ in range(1000):
            self.assertFalse(writer.enqueue(row))
        self.assertLess(time.monotonic()-begin, .3)
        begin = time.monotonic()
        stats = writer.close(.01)
        self.assertLess(time.monotonic()-begin, .15)
        self.assertFalse(stats['complete'])
        self.assertEqual(stats['queued'], 2)
        self.assertEqual(stats['dropped_full'], 1000)
        sink.release.set()
        stats = writer.close(1)
        self.assertTrue(stats['complete'])
        self.assertEqual(stats['written'], 2)
        rows = list(map(json.loads, sink.getvalue().splitlines()))
        self.assertEqual(rows[1]['body_poses'][0][0], 1)
        self.assertEqual(rows[1]['timing']['x'][0], 1)
        self.assertEqual(rows[2]['body_poses'][0][0], 9)
        self.assertGreater(rows[2]['source_log_previous_write']['write_flush_ns'], 0)
        self.assertTrue(rows[-1]['complete'])
        self.assertEqual(rows[-1]['complete_scope'], 'accepted_records_drained')
        self.assertIsNone(rows[-1]['full_source_coverage'])
        self.assertFalse(rows[-1]['close_confirmed'])
        self.assertFalse(rows[-1]['durability_confirmed'])
        self.assertEqual(rows[-1]['counters']['dropped_full'], 1000)
        self.assertNotIn(threading.get_ident(), sink.threads)
        self.assertEqual(stats['attempted'], stats['enqueued']+stats['dropped_full'])

    def test_open_and_serialization_failure_are_reported_not_raised(self):
        def fail(path):
            raise OSError('bad disk')
        writer = SourceWriter('unused', opener=fail)
        writer._done.wait(1)
        self.assertFalse(writer.enqueue({'x': 1}))
        stats = writer.close()
        self.assertFalse(stats['complete'])
        self.assertEqual(stats['errors'], 1)
        self.assertEqual(stats['dropped_error'], 1)
        sink = io.StringIO()
        writer = SourceWriter(sink=sink)
        writer.enqueue({'invalid': float('nan')})
        stats = writer.close(1)
        self.assertFalse(stats['complete'])
        self.assertEqual(stats['abandoned'], 1)
        self.assertEqual(stats['written'], 0)
        self.assertEqual(sink.getvalue(), '')

    def test_background_json_and_short_write(self):
        sink = io.StringIO()
        caller = threading.get_ident()
        original = json.dumps
        threads = []
        def dumps(*args, **kwargs):
            threads.append(threading.get_ident())
            return original(*args, **kwargs)
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_source_writer.json.dumps', dumps):
            writer = SourceWriter(sink=sink)
            writer.enqueue({'record_type': 'gmr_frame', 'x': np.arange(5)})
            self.assertTrue(writer.close(1)['complete'])
        self.assertTrue(threads)
        self.assertNotIn(caller, threads)
        class Short(io.StringIO):
            def write(self, text):
                return 0
        writer = SourceWriter(sink=Short())
        writer.enqueue({'x': 1})
        self.assertIn('short', writer.close(1)['error'])

    def test_capture_logging_open_failure_does_not_skip_cleanup(self):
        from deployment.vita_s100_aorta_30dof import pico_input
        stop = threading.Event()
        commands, output = queue.Queue(), queue.Queue()
        commands.put((1., 0., 0., 0.))
        source, service = Mock(), Mock()
        service.poll.return_value = None
        def step():
            stop.set()
            return None
        source.step.side_effect = step
        def fail(path):
            raise OSError('injected disk failure')
        writers = []
        def factory(*args, **kwargs):
            writer = SourceWriter(*args, **kwargs, opener=fail)
            writers.append(writer)
            return writer
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'outputs').mkdir()
            config = dict(pico_max_age_s=.6, retargeting={}, artifacts={'model': {}})
            with (patch.object(worker.os, 'setsid'),
                  patch.object(worker, 'thread_affinity', return_value=[]),
                  patch.object(worker, 'sdk_math', return_value=None),
                  patch.object(worker, 'ReferenceStream', return_value=NS()),
                  patch.object(worker, 'SourceWriter', side_effect=factory),
                  patch.object(worker.subprocess, 'Popen', return_value=service),
                  patch.object(pico_input, 'create_pico_source', return_value=source)):
                worker._capture(tmp, config, None, None, None, commands, output, stop)
        source.stop.assert_called_once()
        service.terminate.assert_called_once()
        service.wait.assert_called_once()
        self.assertFalse(writers[0].stats()['complete'])
        self.assertEqual(writers[0].stats()['errors'], 1)
        self.assertFalse(any(row.get('state') == 'failed' for row in output.queue))

    def test_thread_affinity_may_leave_the_inherited_mask_but_not_the_machine(self):
        original = os.sched_getaffinity(0)
        if len(original) < 2:
            self.skipTest('Needs two allowed CPUs')
        low, high = sorted(original)[0], sorted(original)[-1]
        seen = {}
        def run():
            os.sched_setaffinity(0, {low})
            seen['wide'] = worker.thread_affinity((high,))
            try:
                worker.thread_affinity((max(original)+64,))
            except ValueError as error:
                seen['error'] = str(error)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(5)
        self.assertEqual(seen['wide'], [high])
        self.assertIn('unavailable', seen['error'])
        self.assertEqual(os.sched_getaffinity(0), original)

    def test_vendor_service_is_forked_after_worker_affinity(self):
        from deployment.vita_s100_aorta_30dof import pico_input
        stop = threading.Event()
        commands, output = queue.Queue(), queue.Queue()
        commands.put((1., 0., 0., 0.))
        source, service = Mock(), Mock()
        service.poll.return_value = None
        source.step.side_effect = lambda: stop.set()
        order = []
        def affinity(cpus):
            order.append(('affinity', tuple(cpus)))
            return list(cpus)
        def popen(*args, **kwargs):
            order.append(('service',))
            return service
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'outputs').mkdir()
            config = dict(pico_max_age_s=.6, retargeting={}, artifacts={'model': {}},
                          pico_worker=dict(worker_cpus=[0, 2, 3], gmr_cpus=[1]))
            with (patch.object(worker.os, 'setsid'),
                  patch.object(worker, 'thread_affinity', side_effect=affinity),
                  patch.object(worker, 'sdk_math', return_value=None),
                  patch.object(worker, 'ReferenceStream', return_value=NS()),
                  patch.object(worker.subprocess, 'Popen', side_effect=popen),
                  patch.object(pico_input, 'create_pico_source', return_value=source)):
                worker._capture(tmp, config, None, None, None, commands, output, stop)
        self.assertEqual(order, [('affinity', (0, 2, 3)), ('service',), ('affinity', (1,))])
        service.terminate.assert_called_once()

    def test_stream_continues_while_sink_blocked_and_reader_gaps_are_not_network(self):
        class Reference:
            origin_ns = last_stamp_ns = last_receive_monotonic = None
            def append(self, stamp, q, quat, received_monotonic, base):
                if self.origin_ns is None:
                    self.origin_ns = stamp
                self.last_stamp_ns, self.last_receive_monotonic = stamp, received_monotonic
            def ready(self):
                return False
        retarget = NS(last_qpos=np.arange(37.), last_foot_support={'nested': [1]},
                      solve=lambda *a, **kw: (np.zeros(28), np.array([1., 0, 0, 0]), 0.))
        sink = BlockedSink()
        writer = SourceWriter(sink=sink, header={}, capacity=2)
        self.assertTrue(sink.entered.wait(1))
        with (patch.object(worker, 'create_reference', return_value=Reference()),
              patch.object(worker, 'create_retargeter', return_value=retarget),
              patch.object(worker, 'stream_settings', return_value=None)):
            stream = worker.ReferenceStream(None, {}, None, None, source_log=writer)
        body = np.zeros((24, 7))
        for i, seq in enumerate([1, 5, 6, 7, 8]):
            snapshot = stream.push(body, dict(timestamp_ns=1_000_000_000+i*20_000_000,
                                   timestamp_monotonic=time.monotonic(), reader_frame_sequence=seq), None)
            self.assertEqual(snapshot['state'], 'collecting')
            stream.publish_timing = (100, 120)
            stream.flush_log()
        self.assertEqual(writer.stats()['dropped_full'], 3)
        sink.release.set()
        self.assertTrue(writer.close(1)['complete'])
        rows = list(map(json.loads, sink.getvalue().splitlines()))
        self.assertEqual(rows[2]['reader_frames_observed_not_retargeted'], 3)
        self.assertEqual(rows[2]['reader_sequence_delta'], 4)
        self.assertEqual(rows[2]['snapshot_publish_call_ns'], 20)
        self.assertGreaterEqual(rows[2]['postprocess_end_monotonic'], rows[2]['gmr_end_monotonic'])
        self.assertAlmostEqual(rows[2]['source_gap_s'], .02)
        self.assertNotIn('network_drop', rows[2])
        # A logging failure/closed logger still cannot suppress the next snapshot.
        snapshot = stream.push(body, dict(timestamp_ns=2_000_000_000,
                               timestamp_monotonic=time.monotonic(), reader_frame_sequence=9), None)
        self.assertEqual(snapshot['state'], 'collecting')
        self.assertFalse(stream.flush_log())


if __name__ == '__main__':
    unittest.main()
