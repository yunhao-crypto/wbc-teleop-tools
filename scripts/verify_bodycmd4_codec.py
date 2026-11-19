#!/usr/bin/env python3
"""Native BodyCmd4 bytes-only regression; creates no Node/controller/publisher."""
from dataclasses import replace
import json
from pathlib import Path
import sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources
verify_vendor_sources()
from locomotion_aorta import _body_cmd_native as native
from locomotion_aorta.body_types import BodyCmd,BodyJointCmd,BODY_CMD_MOTOR_THETA_PROTOCOL_VERSION
from lowlevel.BodyCmd import BodyCmd as Wire
if BODY_CMD_MOTOR_THETA_PROTOCOL_VERSION!=4:raise RuntimeError('SDK lacks explicit native AB coordinate declaration')
joints=[BodyJointCmd() for _ in range(31)]
for i in (4,5,10,11):joints[i]=BodyJointCmd(q=-.25+.01*i,kp=20.,kd=1.)
stamp=time.time_ns()//1000
cmd=BodyCmd(joints=tuple(joints),protocol_version=4,runtime_id=1,control_id=2,session_id=3,sequence=4,
    input_state_id=5,input_state_ts_real_us=stamp,ts_client_created_us=stamp,ts_client_send_us=stamp)
checks=[]
for version in (3,4):
    original=replace(cmd,protocol_version=version)
    payload,_=native.encode(original,time.monotonic_ns(),bytes(range(16)),1)
    wire=Wire.GetRootAs(payload,0)
    if (wire.ProtocolVersion()!=version or wire.JointsLength()!=31 or wire.RuntimeId()!=1
            or wire.ControlId()!=2 or wire.SessionId()!=3 or wire.Sequence()!=4):
        raise RuntimeError('Native encoder changed protocol/session/body layout')
    for i,source in enumerate(original.joints):
        target=wire.Joints(i)
        for name in ('q','dq','tau','kp','kd'):
            if abs(getattr(target,name.capitalize())()-getattr(source,name))>1e-6:
                raise RuntimeError('Native encoder changed a target or gain: '+name)
    checks.append(version)
try:native.encode(replace(cmd,protocol_version=5),time.monotonic_ns(),bytes(range(16)),1)
except (ValueError,RuntimeError,TypeError):pass
else:raise RuntimeError('Unknown BodyCmd coordinate version was not refused')
receipt=dict(status='PASS_NATIVE_BODYCMD4_SERIALIZATION',verified_versions=checks,
    body_cmd_protocol_version=4,ankle_targets='motor_AB',feedback='virtual_pitch_roll',
    bytes_only=True,actions_sent=0,control_handles_created=0,publishers_created=0,
    robot_configuration_modified=False,protection_relaxed=False)
target=ROOT/'build/bodycmd4_codec_check.json'
target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
