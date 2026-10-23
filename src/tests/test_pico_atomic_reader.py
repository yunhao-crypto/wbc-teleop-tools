"""Reader must never combine legacy timestamp and body getters."""
import importlib.util
from pathlib import Path
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'src/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/pico_body_pose_reader.py'
spec=importlib.util.spec_from_file_location('atomic_pico_reader_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
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
