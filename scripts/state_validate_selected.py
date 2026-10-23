#!/usr/bin/env python3
"""Compare selected policy decoding to full reference on captured payloads."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deployment.vita_s100_aorta_30dof.state_adapter import decode_humanoid_lowstate_buffer
from deployment.vita_s100_aorta_30dof.live_shadow import policy_state_from_lowstate
from deployment.vita_s100_aorta_30dof.onboard_state import decode_policy_lowstate_buffer
from flow_common import stats, digest

p=argparse.ArgumentParser()
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
raw=a.capture.read_bytes();offset=0;payloads=[]
while offset<len(raw):
    if offset+4>len(raw):raise ValueError('Truncated record size')
    size=int.from_bytes(raw[offset:offset+4],'little');offset+=4
    if not size or offset+size>len(raw):raise ValueError('Truncated record')
    payloads.append(raw[offset:offset+size]);offset+=size
if not payloads:raise ValueError('Empty capture')
alignments=[np.asarray([1,0,0,0],np.float32),np.asarray([.9238795,0,.3826834,0],np.float32)]
errors={};old_times=[];new_times=[]
for alignment in alignments:
    for index,payload in enumerate(payloads):
        start=time.monotonic_ns()
        reference=policy_state_from_lowstate(decode_humanoid_lowstate_buffer(payload,received_monotonic_ns=index),base_from_hip_quaternion_wxyz=alignment)
        middle=time.monotonic_ns()
        candidate=decode_policy_lowstate_buffer(payload,received_monotonic_ns=index,base_from_hip_quaternion_wxyz=alignment)
        end=time.monotonic_ns()
        old_times.append((middle-start)/1e6);new_times.append((end-middle)/1e6)
        for key in vars(reference):
            x,y=getattr(reference,key),getattr(candidate,key)
            if isinstance(x,np.ndarray):
                err=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
                errors[key]=max(errors.get(key,0),err)
                if not np.array_equal(x,y):raise AssertionError((index,key,err))
            elif x!=y:raise AssertionError((index,key,x,y))
report=dict(passed=True,frames=len(payloads),alignments=len(alignments),maximum_errors=errors,
    full_decoder_ms=stats(old_times),selected_decoder_ms=stats(new_times),capture_sha256=digest(a.capture),
    scope='Exact equality of consumed policy inputs and metadata; unused telemetry intentionally not decoded',action_sent=False)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
