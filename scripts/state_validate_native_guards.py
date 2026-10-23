#!/usr/bin/env python3
"""Byte-level native decoder guard checks and deterministic mutation smoke."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lowlevel.HumanoidLowState import HumanoidLowState
from deployment.vita_s100_aorta_30dof.onboard_state import decode_policy_lowstate_buffer
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_POLICY_STATE_TO_STATE_SLOTS
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('VITA_STATE_LIBRARY'),'Native library must be enabled'
raw=a.capture.read_bytes();size=int.from_bytes(raw[:4],'little');payload=raw[4:4+size]
view=HumanoidLowState.GetRootAs(payload,0)
def decode(buf):return decode_policy_lowstate_buffer(buf,base_from_hip_quaternion_wxyz=np.array([1,0,0,0],np.float32),received_monotonic_ns=123)
def field(obj,entry):
    offset=obj._tab.Offset(entry)
    assert offset,'Mutation requires materialized field'
    return obj._tab.Pos+offset
def patch(pos,fmt,value):
    buf=bytearray(payload);struct.pack_into(fmt,buf,pos,value);return bytes(buf)
reference=decode(payload)
selected=next(view.MotorStates(i) for i in range(44) if view.MotorStates(i).SlotIndex() in VITA_POLICY_STATE_TO_STATE_SLOTS and view.MotorStates(i)._tab.Offset(30))
other=next(view.MotorStates(i) for i in range(44) if view.MotorStates(i).SlotIndex()!=selected.SlotIndex())
hip=next(view.ImuStates(i) for i in range(view.ImuStatesLength()) if view.ImuStates(i).Location()==0)
vector=view._tab.Vector(view._tab.Offset(20))
vtable=view._tab.Pos-struct.unpack_from('<i',payload,view._tab.Pos)[0]
cases={
 'root_oob':patch(0,'<I',0xffffffff),
 'vtable_before_start':patch(view._tab.Pos,'<i',0x7fffffff),
 'vtable_beyond_end':patch(view._tab.Pos,'<i',-0x7fffffff),
 'vtable_invalid_size':patch(vtable,'<H',3),
 'motor_vector_count':patch(vector-4,'<I',0xffffffff),
 'motor_offset_oob':patch(vector,'<I',0xffffffff),
 'duplicate_slot':patch(field(selected,30),'<B',other.SlotIndex()),
 'slot_out_of_range':patch(field(selected,30),'<B',44),
 'slot_role':patch(field(selected,32),'<B',255),
 'nonfinite_q':patch(field(selected,12),'<f',float('nan')),
 'nonfinite_dq':patch(field(selected,14),'<f',float('inf')),
 'attitude':patch(field(view,24),'<B',0),
 'hip_status':patch(field(hip,10),'<B',0),
 'hip_nan_quaternion':patch(field(hip,6),'<f',float('nan')),
 'hip_nan_gyro':patch(field(hip,6)+16,'<f',float('nan')),
}
zero=bytearray(payload);zero[field(hip,6):field(hip,6)+16]=bytes(16);cases['zero_quaternion']=bytes(zero)
for name,buf in cases.items():
    try:decode(buf)
    except ValueError:pass
    else:raise AssertionError('Invalid payload accepted: '+name)
# Vector table entries may reorder, provided each relative pointer is rebuilt.
reordered=bytearray(payload)
targets=[vector+4*i+struct.unpack_from('<I',payload,vector+4*i)[0] for i in range(44)]
for i,target in enumerate(reversed(targets)):struct.pack_into('<I',reordered,vector+4*i,target-(vector+4*i))
reordered_state=decode(reordered)
assert np.array_equal(reference.robot_dof_pos,reordered_state.robot_dof_pos)
assert np.array_equal(reference.robot_dof_vel,reordered_state.robot_dof_vel)
# Valid trailing unused data may be truncated; every accepted prefix must retain exact consumed output.
prefix_rejects=0
for length in range(len(payload)):
    try:state=decode(payload[:length])
    except ValueError:prefix_rejects+=1;continue
    for key in ('robot_dof_pos','robot_dof_vel','robot_root_quat_wxyz','robot_root_angvel_local'):
        assert np.array_equal(getattr(reference,key),getattr(state,key)),(length,key)
report=dict(passed=True,guard_cases=len(cases),reordered_slots_exact=True,prefixes_tested=len(payload),prefix_rejections=prefix_rejects,action_sent=False)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
