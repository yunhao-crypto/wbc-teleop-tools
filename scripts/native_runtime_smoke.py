#!/usr/bin/env python3
"""Offline native-runtime compatibility check. No SDK, control lease or actions."""
from __future__ import annotations
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np

class NoControlImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*_):
        blocked=('aorta','locomotion_aorta','deployment.vita_s100_aorta_30dof.aorta_adapter',
                 'deployment.vita_s100_aorta_30dof.real_teleop','deployment.vita_s100_aorta_30dof.cli')
        if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
            raise RuntimeError('Offline runtime smoke forbids control import: '+fullname)
        return None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--velocity-backend',choices=['ort','hbm'],default='ort')
    args=parser.parse_args()
    sys.path.insert(0,str(args.root/'src'));sys.meta_path.insert(0,NoControlImports())
    from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.command_guard import command_envelope_from_contract
    from deployment.common.model_contract import sha256_file
    interface=args.root/'artifacts/output_envelopes.interface.json'
    base=command_envelope_from_contract(json.loads(interface.read_text()))
    with OnboardDualPolicyRuntime.from_onboard_root(args.root,velocity_backend=args.velocity_backend,
            deployment_scope='suspended_validation') as runtime:
        warmup=runtime.warmup(args.iterations)
        envelopes={name:runtime.command_envelope(base,name) for name in ('motion','velocity')}
        stand=runtime.stand_step()
        if stand.target_joint_pos.shape!=(30,) or not np.isfinite(stand.target_joint_pos).all():
            raise RuntimeError('Invalid stand target')
        reference=runtime.create_online_reference()
        state=runtime.motion_runner.backend.state()
        if state['motion_step']!=0 or np.any(runtime.last_motion_action) or np.any(runtime.last_velocity_action):
            raise RuntimeError('Warmup sequence state was not reset')
        report=dict(passed=True,action_sent=False,robot_sdk_imported=False,
            scope='Offline compatibility only; suspended scope is metadata, not ownership or actuation authorization',
            warmup=warmup,native_final_state=state,reference_class=type(reference).__name__,
            bundle=str(runtime.bundle.manifest_path),bundle_schema=runtime.bundle.schema,
            policies={name:dict(path=str(runtime.bundle.artifacts[name].path),sha256=runtime.bundle.artifacts[name].sha256)
                for name in ('motion','velocity')},
            envelope_slots={name:len(value.joint_names) for name,value in envelopes.items()},
            stand_target_shape=list(stand.target_joint_pos.shape),
            source_sha256=sha256_file(args.root/'src/deployment/vita_s100_aorta_30dof/onboard_runtime.py'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
