#!/usr/bin/env python3
"""Verify a pinned model/reference trial before the actor can create transport."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(root,label,require_sim=True):
    root=Path(root).resolve();trial=root/'trials/tracking_20261106'/label
    manifest=json.loads((trial/'trial_manifest.json').read_text())
    if manifest['label']!=label or manifest['target_robot']!='10.100.23.42':
        raise ValueError('Wrong experiment or robot')
    for rel,expected in manifest['files'].items():
        if digest(root/rel)!=expected:raise ValueError('Trial file changed: '+rel)
    cfg=json.loads((trial/'config'/manifest['config_name']).read_text())
    if cfg['artifacts']['model']['sha256']!=manifest['model_sha256'] or digest(trial/cfg['artifacts']['model']['path'])!=manifest['model_sha256']:
        raise ValueError('Trial model/config mismatch')
    motion=json.loads((trial/'config/actor_motion_test.json').read_text())
    if motion['sha256']!=manifest['reference_sha256'] or digest(trial/motion['path'])!=manifest['reference_sha256']:
        raise ValueError('Trial reference/config mismatch')
    sim=json.loads((trial/'continuous_sim_validation.json').read_text())
    if sim['model_sha256']!=manifest['model_sha256'] or sim['adapted_sha256']!=manifest['reference_sha256']:
        raise ValueError('Simulation belongs to a different model/reference')
    if require_sim and (not sim['completed'] or sim.get('diagnostic_only')):
        raise ValueError('Full continuous simulation did not pass; live trial is blocked')
    profile=json.loads((trial/'outputs/actor_pico_local.json').read_text())
    if profile['robot_id']!=manifest['target_robot'] or digest(profile['robot_config'])!=profile['robot_config_sha256']:
        raise ValueError('Robot startup profile changed; review it before testing')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--trial',choices=['A31000','B25000','Sonic16000','Sonic20000','MorphMimic72000'],required=True);p.add_argument('--offline-only',action='store_true');a=p.parse_args()
    m=verify(a.root,a.trial,require_sim=not a.offline_only)
    print('TRACKING_TRIAL: '+a.trial+' | robot=10.100.23.42 | 01-09,11 | source clips <=10s | full observations in live mode')
    print('MODEL_SHA256: '+m['model_sha256']);print('REFERENCE_SHA256: '+m['reference_sha256'])

if __name__=='__main__':main()
