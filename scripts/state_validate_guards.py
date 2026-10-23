#!/usr/bin/env python3
"""Exercise selected-input structural/finite guards on a real captured view."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lowlevel.HumanoidLowState import HumanoidLowState
from deployment.vita_s100_aorta_30dof.onboard_state import decode_policy_lowstate_view
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_POLICY_STATE_TO_STATE_SLOTS
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);a=p.parse_args()
raw=a.capture.read_bytes();size=int.from_bytes(raw[:4],'little');view=HumanoidLowState.GetRootAs(raw[4:4+size],0)
class Override:
    def __init__(self,obj,**values):self.obj=obj;self.values=values
    def __getattr__(self,key):return self.values[key] if key in self.values else getattr(self.obj,key)
def decode(value):return decode_policy_lowstate_view(value,base_from_hip_quaternion_wxyz=np.array([1,0,0,0],np.float32),received_monotonic_ns=123)
reference=decode(view)
reordered=decode(Override(view,MotorStates=lambda i:view.MotorStates(43-i)))
assert np.array_equal(reference.robot_dof_pos,reordered.robot_dof_pos)
selected=next(i for i in range(44) if view.MotorStates(i).SlotIndex() in VITA_POLICY_STATE_TO_STATE_SLOTS)
def motor_override(index,**fields):return Override(view,MotorStates=lambda i:Override(view.MotorStates(i),**fields) if i==index else view.MotorStates(i))
hip_index=next(i for i in range(view.ImuStatesLength()) if view.ImuStates(i).Location()==0)
def hip_override(**fields):return Override(view,ImuStates=lambda i:Override(view.ImuStates(i),**fields) if i==hip_index else view.ImuStates(i))
hip=view.ImuStates(hip_index).State()
cases={
 'motor_count':Override(view,MotorStatesLength=lambda:43),
 'attitude':Override(view,AttitudeValid=lambda:False),
 'duplicate_slot':motor_override(1,SlotIndex=lambda:view.MotorStates(0).SlotIndex()),
 'slot_range':motor_override(0,SlotIndex=lambda:44),
 'slot_role':motor_override(0,SlotRole=lambda:999),
 'nonfinite_q':motor_override(selected,Q=lambda:float('nan')),
 'nonfinite_dq':motor_override(selected,Dq=lambda:float('inf')),
 'imu_count':Override(view,ImuStatesLength=lambda:0),
 'hip_status':hip_override(Status=lambda:0),
 'unknown_imu':hip_override(Location=lambda:8),
 'missing_imu_state':hip_override(State=lambda:None),
 'quat_length':hip_override(State=lambda:Override(hip,QuaternionLength=lambda:3)),
 'zero_quat':hip_override(State=lambda:Override(hip,Quaternion=lambda i:0)),
 'nan_gyro':hip_override(State=lambda:Override(hip,Gyroscope=lambda i:float('nan'))),
}
for name,value in cases.items():
    try:decode(value)
    except (ValueError,RuntimeError):pass
    else:raise AssertionError('Accepted invalid input: '+name)
print('PASS: reordered44 slots exact; '+str(len(cases))+' malformed/unsafe policy inputs rejected')
