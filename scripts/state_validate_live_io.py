#!/usr/bin/env python3
"""Offline all44/live-policy state equality on captured LowState payloads."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flow_common import digest,stats
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] == 'locomotion_aorta':
            raise RuntimeError('Offline validation forbids control package import: '+fullname)
        return None
sys.meta_path.insert(0,NoSDK())
# ARM generated schemas import the installed aorta package, including errors
# and binding helpers. Importing is allowed; Node construction is forbidden.
import aorta
_sdk_node_class_available = hasattr(aorta, 'Node')
_node_attempts = 0
_guarded_node_methods = []
def forbidden_node(*args, **kwargs):
    global _node_attempts
    _node_attempts += 1
    raise RuntimeError('Offline state validation forbids Node/control construction')
if _sdk_node_class_available:
    aorta.Node.__init__ = forbidden_node
    _guarded_node_methods.append('__init__')
    for _method in dir(aorta.Node):
        if _method.startswith('create_'):
            setattr(aorta.Node, _method, forbidden_node)
            _guarded_node_methods.append(_method)
from deployment.vita_s100_aorta_30dof.state_adapter import decode_humanoid_lowstate_buffer
from deployment.vita_s100_aorta_30dof.live_shadow import policy_state_from_lowstate
from deployment.vita_s100_aorta_30dof.onboard_state import decode_policy_lowstate_buffer
from deployment.vita_s100_aorta_30dof.onboard_live_io import decode_live_state,live_robot_state
from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState,VITA_CONTROLLER_JOINT_NAMES

p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if not os.environ.get('VITA_STATE_LIBRARY'):raise RuntimeError('Set VITA_STATE_LIBRARY to tested native decoder')
alignments=[np.array([1,0,0,0],np.float32),np.array([.9238795,0,.3826834,0],np.float32)]
times=[];report=[];comparisons=0
for path in a.capture:
    raw=path.read_bytes();payloads=[];offset=0
    while offset<len(raw):
        if offset+4>len(raw):raise ValueError('Truncated record length')
        size=int.from_bytes(raw[offset:offset+4],'little');offset+=4
        if not size or offset+size>len(raw):raise ValueError('Truncated record payload')
        payloads.append(raw[offset:offset+size]);offset+=size
    if not payloads:raise ValueError('Empty capture')
    for index,payload in enumerate(payloads):
        received=index+123
        original=decode_humanoid_lowstate_buffer(payload,received_monotonic_ns=received)
        for alignment in alignments:
            native=decode_policy_lowstate_buffer(payload,base_from_hip_quaternion_wxyz=alignment,received_monotonic_ns=received)
            before=time.monotonic_ns()
            actual=decode_live_state(payload,alignment=alignment,received_monotonic_ns=received)
            times.append((time.monotonic_ns()-before)/1e6)
            np.testing.assert_array_equal(actual.q_by_slot,original.q_by_slot,err_msg=f'{path.name}:{index}:all44q')
            for field in ('state_id','received_monotonic_ns','attitude_valid','ts_state_real_us'):
                if getattr(actual,field)!=getattr(original,field):raise AssertionError((path.name,index,field))
            old_policy=policy_state_from_lowstate(original,base_from_hip_quaternion_wxyz=alignment)
            for field in vars(native):
                x,y,z=getattr(actual.policy_state,field),getattr(native,field),getattr(old_policy,field)
                if isinstance(x,np.ndarray):
                    np.testing.assert_array_equal(x,y,err_msg=f'{index}:{field}:native')
                    np.testing.assert_array_equal(x,z,err_msg=f'{index}:{field}:original')
                elif x!=y or x!=z:raise AssertionError((index,field,x,y,z))
            robot=live_robot_state(actual,base_from_hip_quaternion_wxyz=alignment)
            expected=VitaRobotState(VITA_CONTROLLER_JOINT_NAMES,old_policy.robot_dof_pos,old_policy.robot_dof_vel,
                old_policy.robot_root_quat_wxyz,old_policy.robot_root_angvel_local,
                state_id=original.state_id,received_monotonic_ns=received,attitude_valid=original.attitude_valid)
            for field in vars(expected):
                x,y=getattr(robot,field),getattr(expected,field)
                if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y,err_msg=f'{index}:{field}:robot')
                elif x!=y:raise AssertionError((index,field,x,y))
            comparisons+=1
    report.append(dict(path=str(path),frames=len(payloads),sha256=digest(path)))
result=dict(passed=True,captures=report,alignments=len(alignments),frame_alignment_comparisons=comparisons,
    all44_positions='Bit exact versus original full decoder',policy_state='Bit exact versus original and accepted native decoder',
    robot_state='Bit exact including final quaternion normalization',decode_live_state_ms=stats(times),
    state_library_sha256=digest(Path(os.environ['VITA_STATE_LIBRARY'])),action_sent=False,
    robot_sdk_imported=_sdk_node_class_available,aorta_package_imported=True,
    sdk_nodes_created=0,sdk_control_handles_created=0,forbidden_construction_attempts=_node_attempts,
    guarded_node_methods=_guarded_node_methods)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
