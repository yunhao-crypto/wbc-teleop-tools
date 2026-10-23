"""Offline mapping and subscriber-lifecycle tests; never instantiate real SDK."""
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'packages/state_schema'))
import flatbuffers
from lowlevel.HumanoidLowState import HumanoidLowState,HumanoidLowStateT
from deployment.vita_s100_aorta_30dof.onboard_live_io import decode_live_state,live_robot_state,OnboardLiveStateSource
from deployment.vita_s100_aorta_30dof.onboard_state import _SCHEMA_HASH
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_POLICY_STATE_TO_STATE_SLOTS

class LiveStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=(ROOT/'outputs/live_state_velocity_30s.lowstate.bin').read_bytes()
        count=int.from_bytes(raw[:4],'little');cls.payload=raw[4:4+count]
        cls.alignment=np.array([1,0,0,0],np.float32)
    def decode(self,payload=None):return decode_live_state(payload or self.payload,alignment=self.alignment,received_monotonic_ns=123)
    def changed(self,fn):
        value=HumanoidLowStateT.InitFromObj(HumanoidLowState.GetRootAs(self.payload,0))
        fn(value)
        builder=flatbuffers.Builder(8192);builder.Finish(value.Pack(builder));return bytes(builder.Output())
    def test_nonpolicy_positions_preserved_all44_and_reordering(self):
        expected=self.decode().q_by_slot.copy()
        def change(value):
            for motor in value.motorStates:
                if motor.slotIndex not in VITA_POLICY_STATE_TO_STATE_SLOTS:
                    motor.q=(motor.slotIndex+1)/100.;expected[motor.slotIndex]=motor.q
            value.motorStates.reverse()
        result=self.decode(self.changed(change))
        np.testing.assert_array_equal(result.q_by_slot,expected)
        np.testing.assert_array_equal(result.policy_state.robot_dof_pos,self.decode().policy_state.robot_dof_pos)
    def test_nonfinite_unused_position_is_rejected(self):
        def change(value):
            motor=next(m for m in value.motorStates if m.slotIndex not in VITA_POLICY_STATE_TO_STATE_SLOTS)
            motor.q=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite measured slot position'):self.decode(self.changed(change))
    def test_conversion_rejects_alignment_change(self):
        with self.assertRaisesRegex(ValueError,'alignment does not match'):
            live_robot_state(self.decode(),base_from_hip_quaternion_wxyz=np.array([0,1,0,0],np.float32))
    def test_publication_timestamp_preserved(self):
        def change(value):value.tsStatePub=1791522000123456
        state=self.decode(self.changed(change))
        self.assertEqual(state.ts_state_pub_us,1791522000123456)
    def fake_sdk(self,*,subscribe_failure=False):
        closed=[]
        class Subscriber:
            def close(self):closed.append('subscriber')
        class Node:
            def __init__(self,name,group):self.group=lambda:group
            def create_subscriber(self,*a,**k):
                if subscribe_failure:raise RuntimeError('subscribe failed')
                return Subscriber()
            def create_publisher(self,*a,**k):raise AssertionError('Must not create publisher')
            def close(self):closed.append('node')
        sdk=types.SimpleNamespace(Node=Node,QoS=types.SimpleNamespace(sensor_data=lambda:None),SubscriberOptions=lambda **k:None)
        return sdk,closed
    def test_newest_read_timeout_close_and_no_global_node_patch(self):
        sdk,closed=self.fake_sdk();method=sdk.Node.create_publisher
        with patch.dict(sys.modules,{'aorta':sdk,'humanoid_low_state_schema_meta':types.SimpleNamespace(SCHEMA_HASH_HEX=_SCHEMA_HASH)}):
            source=OnboardLiveStateSource(group='default',alignment=self.alignment)
        self.assertIs(sdk.Node.create_publisher,method)
        source._callback(self.payload,types.SimpleNamespace(receive_stamp_ns=1791522000123456789))
        state=source.read(.1);self.assertTrue(state.attitude_valid)
        self.assertEqual(state.native_receive_stamp_ns,1791522000123456789)
        with self.assertRaises(TimeoutError):source.read(.001)
        source.close()
        with self.assertRaises(RuntimeError):source.read(.1)
        self.assertEqual(closed,['subscriber','node'])
    def test_subscription_failure_closes_node(self):
        sdk,closed=self.fake_sdk(subscribe_failure=True)
        with patch.dict(sys.modules,{'aorta':sdk,'humanoid_low_state_schema_meta':types.SimpleNamespace(SCHEMA_HASH_HEX=_SCHEMA_HASH)}):
            with self.assertRaisesRegex(RuntimeError,'subscribe failed'):
                OnboardLiveStateSource(group='default',alignment=self.alignment)
        self.assertEqual(closed,['node'])

if __name__=='__main__':unittest.main()
