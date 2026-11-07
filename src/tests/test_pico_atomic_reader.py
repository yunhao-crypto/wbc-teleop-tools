"""Reader must never combine legacy timestamp and body getters."""
from pathlib import Path
import time
import unittest
import json
import os
import tempfile
import sys
import signal
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/deployment/unitree_g1_ros2_29dof/src'))
from humanoid_policy import pico_body_pose_reader as module
class Logger:
    def error(self,*a):pass
    def warn(self,*a):pass

class TornLegacySDK:
    # Exactly the observed race: global timestamp has advanced before body.
    def get_time_stamp_ns(self):raise AssertionError('Unsafe timestamp getter called (would return1773)')
    def get_body_joints_pose(self):raise AssertionError('Unsafe body getter called (would return1772)')
    def is_body_data_available(self):raise AssertionError('Must use availability from same snapshot')
    def get_body_pose_snapshot(self):return True,1772,np.full((24,7),1772.,np.float64)

class AtomicReaderTests(unittest.TestCase):
    def test_optional_capture_preserves_frames_before_latest_overwrite(self):
        class SDK:
            count=0
            def get_body_pose_snapshot(self):
                self.count=min(20,self.count+1)
                return True,self.count,np.full((24,7),float(self.count))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, VITA_PICO_CAPTURE_DIR=directory):
            reader=module.PicoBodyPoseReader(logger=Logger(),sdk=SDK(),duplicate_poll_interval_s=.002)
            reader.start()
            capture=reader._capture
            try:
                deadline=time.monotonic()+2
                while (reader.latest() or {}).get('reader_frame_sequence',0)<20 and time.monotonic()<deadline:
                    time.sleep(.001)
            finally:reader.stop()
            rows=[json.loads(line) for line in next(Path(directory).glob('pico_reader_*.jsonl')).read_text().splitlines()]
            frames=[r for r in rows if r['record_type']=='pico_reader_frame']
            self.assertEqual([r['reader_frame_sequence'] for r in frames],list(range(1,21)))
            self.assertEqual([r['timestamp_ns'] for r in frames],list(range(1,21)))
            self.assertTrue(all(r['body_poses'][0][0]==r['timestamp_ns'] for r in frames))
            self.assertTrue(all(r['reader_monotonic_ns']>0 and r['reader_realtime_ns']>0 for r in frames))
            self.assertEqual(rows[-1],dict(record_type='summary',written=20,dropped=0))
            self.assertNotEqual(rows[0]['writer_pid'],os.getpid())
            self.assertEqual(rows[0]['writer_pid'],capture.process.pid)
            self.assertIsNone(capture.error)
            self.assertFalse(capture.process.is_alive())
            capture.close()  # Idempotent, including closed pipes/queues.

    def test_full_capture_queue_drops_without_blocking_and_drains_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            capture=module.PicoFrameLog(directory,capacity=4)
            try:
                os.kill(capture.process.pid,signal.SIGSTOP)
                started=time.monotonic()
                for i in range(100):
                    capture.append(dict(timestamp_ns=1787908100306779904+i,
                                        body_poses=np.full((24,7),i,dtype=np.float32)))
                self.assertLess(time.monotonic()-started,1.)
                self.assertGreater(capture.dropped,0)
                self.assertEqual(capture.accepted+capture.dropped,100)
            finally:
                os.kill(capture.process.pid,signal.SIGCONT)
                capture.close()
            rows=[json.loads(line) for line in capture.path.read_text().splitlines()]
            self.assertIsNone(capture.error)
            self.assertEqual(rows[-1],dict(record_type='summary',written=capture.accepted,dropped=capture.dropped))
            self.assertEqual([r['timestamp_ns'] for r in rows[1:-1]],
                             list(range(1787908100306779904,1787908100306779904+capture.accepted)))
            self.assertFalse(capture.process.is_alive())

    def test_failed_writer_is_reported_and_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            capture=module.PicoFrameLog(directory)
            capture.append(dict(timestamp_ns=1,body_poses=np.full((24,7),np.nan)))
            capture.close()
            self.assertIsNotNone(capture.error)
            self.assertFalse(capture.process.is_alive())

    def test_reader_hook_sees_owned_frames_before_latest_overwrite(self):
        body=np.zeros((24,7),dtype=np.float32)
        class SDK:
            count=0
            def get_body_pose_snapshot(self):
                self.count=min(20,self.count+1)
                body.fill(self.count)
                return True,self.count,body
        seen=[]
        reader=module.PicoBodyPoseReader(logger=Logger(),sdk=SDK(),on_frame=seen.append)
        reader.start()
        try:
            deadline=time.monotonic()+2
            while len(seen)<20 and time.monotonic()<deadline:time.sleep(.001)
        finally:reader.stop()
        self.assertEqual(len(seen),20)
        self.assertTrue(all(np.all(row['body_poses']==row['timestamp_ns']) for row in seen))

    def test_duplicate_poll_waits_without_delaying_new_frames(self):
        sdk=TornLegacySDK();stamps=iter((1772,1773,1773))
        sdk.get_body_pose_snapshot=lambda:(True,next(stamps),np.zeros((24,7)))
        reader=module.PicoBodyPoseReader(logger=Logger(),sdk=sdk,duplicate_poll_interval_s=.002)
        waits=[]
        class Stop:
            stopped=False
            def is_set(self):return self.stopped
            def wait(self,seconds):
                waits.append((seconds,reader.latest()['timestamp_ns']))
                self.stopped=True
        reader._stop_event=Stop();reader._run()
        self.assertEqual(waits,[(.002,1773)])
        self.assertEqual(reader._read_errors,0)

    def test_invalid_duplicate_poll_interval_is_rejected(self):
        for value in (0.,-.001,.1,float('nan'),float('inf'),True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.PicoBodyPoseReader(logger=Logger(),sdk=TornLegacySDK(),duplicate_poll_interval_s=value)

    def test_torn_legacy_state_is_never_read(self):
        reader=module.PicoBodyPoseReader(logger=Logger(),sdk=TornLegacySDK())
        reader.start()
        try:
            deadline=time.monotonic()+1
            while reader.latest() is None and time.monotonic()<deadline:time.sleep(.001)
            value=reader.latest();self.assertIsNotNone(value)
            self.assertEqual(value['timestamp_ns'],1772)
            self.assertTrue(np.all(value['body_poses']==1772))
            self.assertEqual(reader._read_errors,0)
        finally:reader.stop()

    def test_old_binding_is_rejected_before_thread_start(self):
        with self.assertRaisesRegex(RuntimeError,'atomic get_body_pose_snapshot'):
            module.PicoBodyPoseReader(logger=Logger(),sdk=object())

    def test_unavailable_snapshot_does_not_publish(self):
        sdk=TornLegacySDK();sdk.get_body_pose_snapshot=lambda:(False,0,np.zeros((24,7)))
        reader=module.PicoBodyPoseReader(logger=Logger(),sdk=sdk);reader.start()
        try:
            time.sleep(.01);self.assertIsNone(reader.latest());self.assertEqual(reader._read_errors,0)
        finally:reader.stop()

if __name__=='__main__':unittest.main()
