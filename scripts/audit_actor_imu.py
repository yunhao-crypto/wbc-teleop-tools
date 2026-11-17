#!/usr/bin/env python3
"""OFFLINE 2–8 Hz IMU diagnostic; requires explicit clock/frame metadata."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from deployment.vita_s100_aorta_30dof.actor_imu_audit import audit_imu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('npz',type=Path);p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--start-s',type=float,default=0.,help='Seconds since FIRST selected valid input timestamp, not actor-log origin')
    p.add_argument('--end-s',type=float);p.add_argument('--max-gap-s',type=float,default=.05);p.add_argument('--min-segment-s',type=float,default=4.)
    p.add_argument('--output',type=Path);args=p.parse_args()
    try:
        result=audit_imu(args.npz,args.metadata,args.start_s,args.end_s,args.max_gap_s,args.min_segment_s)
        code=0 if result['status']=='DIAGNOSTIC_ONLY' else 2
    except (ValueError,KeyError,TypeError,OSError) as exc:
        result=dict(schema='wtt.imu_audit.v1',status='INVALID_INPUT',reason=str(exc));code=2
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        if args.output.resolve() in (args.npz.resolve(),args.metadata.resolve()):p.error('Output must not overwrite an input')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='');return code


if __name__=='__main__':raise SystemExit(main())
