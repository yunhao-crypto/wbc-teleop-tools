"""Reader/head progress and raw recording while the GMR process is blocked.

Uses the production worker and reader with a synthetic SDK; no robot transport.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/deployment/unitree_g1_ros2_29dof/src')]
from deployment.vita_s100_aorta_30dof.actor_pico_latest import LatestPicoPose
from deployment.vita_s100_aorta_30dof.actor_pico_worker import PicoReferenceProcess

STAMP = 1787908100306779904


def blocked_capture(args, entered, release):
    from deployment.vita_s100_aorta_30dof import actor_pico_worker as worker, pico_input
    from humanoid_policy.local_smpl_source import LocalPicoSmplSource
    os.environ['VITA_PICO_CAPTURE_DIR'] = str(Path(args[0])/'raw')
    class SDK:
        started = time.monotonic()
        def get_body_pose_snapshot(self):
            tick = int((time.monotonic()-self.started)*90)
            body = np.zeros((24, 7), dtype=np.float32)
            body[:, 6] = 1
            body[:, 0] = tick
            return True, STAMP+tick*11111111, body
    class Stream:
        calls = 0
        def push(self, body, metadata, anchor):
            self.calls += 1
            if self.calls == 2:
                entered.set()
                release.wait(10.)
            return dict(state='ready', raw_body_poses=body.tolist(),
                        pico_timestamp_ns=metadata['timestamp_ns'],
                        received_monotonic=metadata['timestamp_monotonic'])
        def flush_log(self):
            pass
    def source(**kwargs):
        return LocalPicoSmplSource(logger=kwargs['logger'], max_source_age=kwargs['maximum_age'],
            sdk=SDK(), duplicate_poll_interval_s=kwargs['duplicate_poll_interval_s'],
            on_frame=kwargs['on_frame'])
    service = Mock()
    service.poll.return_value = None
    with (patch.object(worker, 'sdk_math', return_value=None),
          patch.object(worker, 'ReferenceStream', return_value=Stream()),
          patch.object(worker.subprocess, 'Popen', return_value=service),
          patch.object(pico_input, 'create_pico_source', side_effect=source)):
        worker._capture(*args)


class LatestPoseTests(unittest.TestCase):
    def test_atomic_stamp_and_contention_keep_original_receipt(self):
        slot = LatestPicoPose(mp.get_context('spawn'))
        sample = dict(body_poses=np.ones((24, 7), dtype=np.float32), timestamp_ns=STAMP,
                      reader_monotonic_ns=123456789001, reader_frame_sequence=7)
        self.assertIsNone(slot.snapshot())
        self.assertTrue(slot.publish(sample))
        first = slot.snapshot()
        self.assertEqual(first['pico_timestamp_ns'], STAMP)
        self.assertEqual(first['reader_monotonic_ns'], 123456789001)
        with slot.lock:
            self.assertFalse(slot.publish(dict(sample, timestamp_ns=STAMP+1)))
            self.assertIs(slot.snapshot(), first)
        self.assertEqual(slot.snapshot()['pico_timestamp_ns'], STAMP)

    def test_reader_head_and_process_capture_progress_while_gmr_waits(self):
        context = mp.get_context('spawn')
        entered, release = context.Event(), context.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'outputs').mkdir()
            config = dict(pico_max_age_s=.6, retargeting={}, artifacts={'model':{}},
                          pico_worker={'reader_poll_interval_s':.002})
            runtime = NS(root=root, config=config, model=NS(contract=None), policy=NS(lower=None, upper=None))
            worker = PicoReferenceProcess(runtime, buttons=False)
            self.assertFalse(worker.process.daemon)
            args = (directory, config, None, None, None, worker.commands, worker.output,
                    worker.stop, None, worker.head_pose)
            worker.process = context.Process(target=blocked_capture, args=(args, entered, release))
            worker.start()
            worker.connect((1, 0, 0, 0))
            try:
                self.assertTrue(entered.wait(8.), 'Worker never reached its second GMR solve')
                first = worker.snapshot()
                deadline = time.monotonic()+1.
                while first['state'] != 'ready' and time.monotonic()<deadline:
                    time.sleep(.005)
                    first = worker.snapshot()
                self.assertEqual(first['state'], 'ready')
                time.sleep(.12)
                latest = worker.snapshot()
                self.assertEqual(latest['pico_timestamp_ns'], first['pico_timestamp_ns'])
                self.assertEqual(latest['received_monotonic'], first['received_monotonic'])
                self.assertGreater(latest['head_snapshot']['pico_timestamp_ns'],
                                   first['head_snapshot']['pico_timestamp_ns'])
                self.assertGreater(latest['head_snapshot']['reader_frame_sequence'],
                                   first['head_snapshot']['reader_frame_sequence'])
                self.assertLess(time.monotonic()-latest['head_snapshot']['received_monotonic'], .1)
            finally:
                worker.stop.set()
                release.set()
                worker.close()
            self.assertFalse(worker.process.is_alive())
            rows = [json.loads(line) for line in next((root/'raw').glob('*.jsonl')).read_text().splitlines()]
            self.assertNotEqual(rows[0]['writer_pid'], worker.process.pid)
            self.assertEqual(rows[-1]['record_type'], 'summary')
            self.assertEqual(rows[-1]['dropped'], 0)
            self.assertEqual(rows[-1]['written'], len(rows)-2)
            self.assertGreater(rows[-1]['written'], 5)
            with self.assertRaises(ProcessLookupError):
                os.kill(rows[0]['writer_pid'], 0)


if __name__ == '__main__':
    unittest.main()
