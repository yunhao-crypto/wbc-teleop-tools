#!/usr/bin/env python3
"""Exactly replay a captured pre-ownership candidate; no transport or motion."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--backend',choices=('auto','ort','sdk'),default='auto')
    args=p.parse_args()
    rows=[json.loads(line) for line in args.input.read_text().splitlines()]
    candidate=next((row for row in reversed(rows) if row.get('record_type')=='prepared_policy_candidate'),None)
    if candidate is None:raise ValueError('Log has no complete prepared_policy_candidate; original input cannot be reconstructed exactly')
    runtime=ActorPicoRuntime(args.root,backend=args.backend)
    state,ref=candidate['input_state'],candidate['reference']
    motion=runtime.math.Motion(ref['q'],ref['base'],ref['quat'])
    command,obs,row=runtime.infer_motion_command(motion,
        q=state['q'],dq=state['dq'],root=state['root'],gyro=state['gyro'],
        ref_time=ref['ref_time'],yaw_offset=ref['yaw_offset'])
    expected=np.asarray([[j[k] for k in ('q','dq','tau','kp','kd')] for j in candidate['body_targets']])
    actual=np.asarray([[getattr(j,k) for k in ('q','dq','tau','kp','kd')] for j in command.joints])
    obs_diff=float(np.max(np.abs(obs-candidate['observation'])))
    action_diff=float(np.max(np.abs(np.asarray(row['raw_action'])-candidate['raw_action'])))
    command_diff=float(np.max(np.abs(actual-expected)))
    if obs_diff>1e-5 or action_diff>1e-4 or command_diff>1e-4:
        raise RuntimeError(f'Candidate replay differs: obs={obs_diff}, action={action_diff}, command={command_diff}')
    target_delta=np.abs(actual[:,0]-state['body_q'])
    largest=int(np.argmax(target_delta))
    print(json.dumps(dict(status='PASS_EXACT_CANDIDATE_REPLAY',backend=runtime.model.backend,
        observation_max_diff=obs_diff,action_max_diff=action_diff,command_max_diff=command_diff,
        largest_target_delta_rad=float(target_delta[largest]),
        largest_target_delta_joint=candidate['body_order'][largest],
        hardware_envelope_check='not_measured',actions_sent=0),indent=2))


if __name__=='__main__':main()
