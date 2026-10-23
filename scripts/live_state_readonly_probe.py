#!/usr/bin/env python3
"""Bounded subscriber-only test of OnboardLiveStateSource and live conversion."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from flow_common import stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--hz',type=float,default=50.)
    parser.add_argument('--group',default='default')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.seconds<=30 or not 1<=args.hz<=100:raise ValueError('Probe supports at most30 seconds and1..100Hz')
    if not os.environ.get('VITA_STATE_LIBRARY'):
        raise RuntimeError('Set VITA_STATE_LIBRARY to the accepted native decoder')
    sys.path.insert(0,str(args.root/'src'))
    import aorta
    def forbidden(*a,**kw):raise RuntimeError('Read-only probe forbids publishing/control handles')
    blocked=[]
    for name in dir(aorta.Node):
        if name.startswith('create_') and any(key in name for key in ('publisher','client','action','service')):
            setattr(aorta.Node,name,forbidden);blocked.append(name)
    if not blocked:raise RuntimeError('Cannot establish expected read-only Node factory guard')
    from deployment.vita_s100_aorta_30dof.onboard_live_io import OnboardLiveStateSource,live_robot_state
    from deployment.vita_s100_aorta_30dof.live_shadow import live_alignment_quaternion
    alignment=live_alignment_quaternion(json.loads((args.root/'src/deployment/vita_s100_aorta_30dof/config/vita02a_imu_alignment.json').read_text()))
    source=None;rows=[];failure=None;callbacks=0
    try:
        source=OnboardLiveStateSource(group=args.group,alignment=alignment)
        if source.group!=args.group:raise RuntimeError('Unexpected effective Aorta group')
        # First message discovery is startup work, excluded from measured ticks.
        initial=source.read(10.)
        live_robot_state(initial,base_from_hip_quaternion_wxyz=alignment)
        deadline=time.monotonic();stop=deadline+args.seconds;period=1/args.hz
        while time.monotonic()<stop:
            wait=deadline-time.monotonic()
            if wait>0:time.sleep(wait)
            begin=time.monotonic_ns()
            raw=source.read(.1);decoded=time.monotonic_ns()
            state=live_robot_state(raw,base_from_hip_quaternion_wxyz=alignment)
            end=time.monotonic_ns()
            if raw.q_by_slot.shape!=(44,) or not np.isfinite(raw.q_by_slot).all():raise ValueError('Invalid44-slot measured positions')
            if state.joint_pos.shape!=(30,) or state.joint_vel.shape!=(30,):raise ValueError('Invalid policy joint vectors')
            if state.state_id!=raw.state_id or state.received_monotonic_ns!=raw.received_monotonic_ns:
                raise ValueError('State conversion metadata mismatch')
            rows.append(dict(state_id=state.state_id,received_monotonic_ns=raw.received_monotonic_ns,
                state_real_us=raw.ts_state_real_us,state_pub_us=raw.ts_state_pub_us,
                native_receive_stamp_ns=raw.native_receive_stamp_ns,callback_wall_ns=raw.received_wall_ns,
                started_monotonic_ns=begin,read_decode_ms=(decoded-begin)/1e6,conversion_ms=(end-decoded)/1e6,
                read_to_converted_ms=(end-begin)/1e6,callback_to_converted_ms=(end-raw.received_monotonic_ns)/1e6,
                attitude_valid=bool(raw.attitude_valid),action_sent=False))
            deadline+=period
            while deadline<time.monotonic():deadline+=period
    except Exception as exc:failure=repr(exc)
    finally:
        if source is not None:
            callbacks=source.received
            try:source.close()
            except Exception as exc:failure=(failure or '')+'; cleanup: '+repr(exc)
    starts=np.asarray([r['started_monotonic_ns'] for r in rows],np.int64)
    transitions=[(b['state_id']-a['state_id'])%65536 for a,b in zip(rows,rows[1:])]
    result=dict(passed=failure is None and len(rows)>0 and not any(delta>=32768 for delta in transitions),failure=failure,action_sent=False,
        control_handles_created=0,blocked_node_factories=blocked,policy_inference_run=False,
        requested_seconds=args.seconds,requested_hz=args.hz,frames=len(rows),source_callbacks=callbacks,
        distinct_state_ids=len({r['state_id'] for r in rows}),state_backward_transitions=sum(delta>=32768 for delta in transitions),
        interarrival_ms=stats(np.diff(starts)/1e6),
        measured_hz=float((len(starts)-1)*1e9/(starts[-1]-starts[0])) if len(starts)>1 else None,
        timings={field:stats([row[field] for row in rows]) for field in ('read_decode_ms','conversion_ms','read_to_converted_ms','callback_to_converted_ms')},
        scope='Only SDK LowState subscription and state conversion; callback receipt is not verified physical source latency',rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
