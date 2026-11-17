#!/usr/bin/env python3
"""OFFLINE frozen-input preparation/parity checks; no robot or control imports."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from deployment.vita_s100_aorta_30dof.actor_control_compare import model_layout
from deployment.vita_s100_aorta_30dof.actor_control_shadow import prepare_fixture,compare_shadow


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prepare=sub.add_parser('prepare',help='Freeze recorded sensors and actual selected virtual references')
    prepare.add_argument('log',type=Path);prepare.add_argument('--onnx',required=True,type=Path);prepare.add_argument('--output',required=True,type=Path)
    compare=sub.add_parser('compare',help='Compare a native-exec offline trace to recorded SDK expectations')
    compare.add_argument('fixture',type=Path);compare.add_argument('trace',type=Path);compare.add_argument('--output',type=Path)
    compare.add_argument('--observation-atol',type=float,default=1e-6);compare.add_argument('--action-atol',type=float,default=5e-5)
    args=p.parse_args()
    try:
        if args.mode=='prepare':result=prepare_fixture(args.log,args.output,model_layout(args.onnx))
        else:result=compare_shadow(args.fixture,args.trace,args.observation_atol,args.action_atol)
        code=2 if result.get('status')=='NUMERICAL_MISMATCH' else 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        result=dict(schema='wtt.shadow_error.v1',status='INVALID_INPUT',reason=str(exc));code=2
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.mode=='compare' and args.output:
        if args.output.resolve() in (args.fixture.resolve(),args.trace.resolve()):p.error('Output must not overwrite an input')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='');return code


if __name__=='__main__':raise SystemExit(main())
