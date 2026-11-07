#!/usr/bin/env python3
"""Compare two completed live NPZ ankle-path trials on shared clip frames."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_trial(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    def one(kind):
        selected = [r for r in rows if r.get('record_type') == kind]
        if len(selected) != 1:
            raise ValueError(f'{path}: expected one {kind} record')
        return selected[0]
    header, started, summary = one('header'), one('motion_started'), one('summary')
    if header.get('reference_source') != 'npz' or not summary.get('exit_confirmed'):
        raise ValueError('Expected a live NPZ trial with confirmed exit')
    frames = {}
    for r in rows:
        if r.get('stage') == 'motion_tracking' and r.get('motion_phase') == 'clip':
            frame = r['motion_tick']-started['lead_frames']
            if frame in frames or 'feedback' not in r:
                raise ValueError('Duplicate clip frame or missing measured feedback')
            frames[frame] = r
    configs = [r['robot_config'] for r in rows if 'robot_config' in r]
    hashes = {r.get('ankle_path_comparison_sha256') for r in configs}
    if len(hashes) != 1 or None in hashes:
        raise ValueError('Missing/inconsistent robot configuration comparison receipt')
    return dict(path=str(path), header=header, summary=summary, frames=frames, robot_hash=next(iter(hashes)))


def compare(a, b):
    ha, hb = a['header'], b['header']
    for name, left, right in [
        ('policy', ha['policy'], hb['policy']),
        ('model', ha['model']['sha256'], hb['model']['sha256']),
        ('NPZ', ha['motion']['sha256'], hb['motion']['sha256']),
        ('actor settings', ha['comparison_config_sha256'], hb['comparison_config_sha256']),
        ('robot settings except wire space', a['robot_hash'], b['robot_hash']),
        ('projection', ha['motion']['limit_corrections'], hb['motion']['limit_corrections']),
        ('reference hold', ha['motion']['end_hold_s'], hb['motion']['end_hold_s']),
        ('root selection', ha['motion']['root_body_index'], hb['motion']['root_body_index']),
        ('joint order', ha['motion']['joint_names'], hb['motion']['joint_names'])]:
        if left != right:
            raise ValueError('A/B confound: different '+name)
    spaces = [h['contract']['required_ankle_command_space'] for h in (ha, hb)]
    if set(spaces) != {'motor_theta', 'virtual_pitch_roll'} or any(
            h['contract']['ankle_control_space'] != 'motor_theta' or
            h['contract']['required_ankle_virtual_pd_intent'] is not False for h in (ha, hb)):
        raise ValueError('Need the SAME motor-theta policy with native and legacy command paths, both motor PD')
    common = sorted(set(a['frames']) & set(b['frames']))
    if not common:
        raise ValueError('No shared clip frames')
    names = ha['motion']['joint_names']
    ankles = [i for i,n in enumerate(names) if '_ankle_' in n]
    result = dict(policy=ha['policy'], model_sha256=ha['model']['sha256'], npz_sha256=ha['motion']['sha256'],
        common_clip_frames=len(common), source_clip_frames=ha['motion']['policy_frames'], trials=[],
        interpretation='Descriptive comparison of separate live trials, not proof of causation or equal initial state.')
    first_states = []
    refs = []
    for trial, space in zip((a,b), spaces):
        errors, tilt, inference, refs_this = [], [], [], []
        for frame in common:
            row = trial['frames'][frame]
            measured = np.asarray(row['feedback']['q_body'])[[row['body_order'].index(n) for n in names]]
            reference = row['motion_reference']
            idx = int(round(reference['ref_time']/trial['header']['motion']['period_s']))
            desired = np.asarray(reference['q'][idx]); refs_this.append(desired)
            errors.append(measured-desired)
            quat = np.asarray(row['feedback']['root_wxyz']); quat = quat/np.linalg.norm(quat)
            tilt.append(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1))))
            inference.append(row['compute_ms'])
            if frame == common[0]: first_states.append(measured)
        error = np.asarray(errors)
        refs.append(np.asarray(refs_this))
        result['trials'].append(dict(path=trial['path'], command_space=space,
            clip_frames_recorded=len(trial['frames']), stop_reason=trial['summary'].get('stop_reason'),
            joint_rmse_deg=dict(zip(names, np.degrees(np.sqrt(np.mean(error**2, axis=0))).tolist())),
            ankle_rmse_deg=float(np.degrees(np.sqrt(np.mean(error[:,ankles]**2)))),
            max_root_tilt_deg=float(max(tilt)), compute_p95_ms=float(np.percentile(inference,95)),
            missed_policy_ticks=trial['summary'].get('missed_policy_ticks'),
            duplicate_body_ticks=trial['summary'].get('duplicate_body_ticks')))
    if not np.allclose(refs[0],refs[1],atol=1e-8,rtol=0):
        raise ValueError('A/B confound: reference joint poses differ on matched clip frames')
    result['initial_measured_pose_max_difference_deg'] = float(np.degrees(np.max(np.abs(first_states[0]-first_states[1]))))
    result['both_clips_complete'] = all(len(t['frames']) == ha['motion']['policy_frames'] and
        t['summary'].get('stop_reason') == 'motion_complete' for t in (a,b))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path); parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(compare(read_trial(args.first),read_trial(args.second)), indent=2)
    if args.output:
        with args.output.open('x') as stream: stream.write(result+'\n')
    print(result)


if __name__ == '__main__':
    main()
