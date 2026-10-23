#!/usr/bin/env python3
"""Replay saved CUDA oracle on NumPy CPU; no GPU or robot dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--skeleton',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--output-npz');ap.add_argument('--max-observation-error',type=float,default=5e-5)
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from deployment.vita_s100_aorta_30dof.onboard_preprocessing import CPUOnlineSmplReference,CPUSmplMotionObservationBuilder
    with np.load(a.input,allow_pickle=False) as f: data={k:f[k] for k in f.files}
    ref=CPUOnlineSmplReference(n_future_frames=10,policy_hz=50,skeleton_path=a.skeleton)
    builder=CPUSmplMotionObservationBuilder(n_future_frames=10,policy_hz=50,default_dof_pos=data['default_dof_pos'])
    reset=set(data['reset_raw_frames'].tolist());lookup={int(v):i for i,v in enumerate(data['raw_frame_indices'])}
    assert not ref.has_reference and ref.current_root_rot() is None and not ref.has_future_sequence()
    try:ref.device_observation_sequence();raise AssertionError('Premature sequence accepted')
    except RuntimeError:pass
    errors=[];elapsed=[];obs=[]
    for raw,(body,stamp) in enumerate(zip(data['body_poses'],data['sample_times'])):
        start=time.perf_counter()
        assert ref.store_pico(body,sample_time=float(stamp),current_time=float(stamp),frame_index=raw)
        if raw in reset:ref.reset_root_anchor()
        if raw not in lookup:continue
        i=lookup[raw]
        result=builder.build(smpl_reference=ref,robot_root_quat_wxyz=data['robot_root_quat'][i],
            robot_root_angvel_local=data['robot_angvel'][i],robot_dof_pos=data['robot_dof_pos'][i],
            robot_dof_vel=data['robot_dof_vel'][i],last_action=data['last_action'][i],
            reference_yaw_alignment_wxyz=data['yaw_alignment'][i]).copy()
        elapsed.append((time.perf_counter()-start)*1000)
        assert result.shape==(1,3126) and result.dtype==np.float32 and result.flags.c_contiguous
        d=np.abs(result[0].astype(np.float64)-data['gpu_observations'][i].astype(np.float64));errors.append(float(d.max()));obs.append(result[0])
        assert ref.has_future_sequence() and not ref.has_future_sequence(9)
        assert np.isfinite(result).all()
        before=(ref.seen_frames,ref._version,ref._write_slot,ref.last_receive_time)
        assert not ref.store_pico(body,sample_time=float(stamp),current_time=float(stamp)+100)
        assert before==(ref.seen_frames,ref._version,ref._write_slot,ref.last_receive_time)
    # Invalid data must not advance reference state.
    before=ref.seen_frames
    bad=data['body_poses'][-1].copy();bad[2,3:]=0
    try:ref.store_pico(bad,sample_time=float(data['sample_times'][-1])+1);raise AssertionError('Zero quaternion accepted')
    except ValueError:pass
    bad=data['body_poses'][-1].copy();bad[0,0]=np.nan
    try:ref.store_pico(bad,sample_time=float(data['sample_times'][-1])+1);raise AssertionError('NaN accepted')
    except ValueError:pass
    assert before==ref.seen_frames
    # Switching resets only anchor, preserving ring/history and timestamp origin.
    times=ref._time_ring.copy();bodies=ref._body_ring.copy();count=ref.seen_frames
    ref.reset_root_anchor();ref.current_root_rot()
    assert np.array_equal(times,ref._time_ring) and np.array_equal(bodies,ref._body_ring) and ref.seen_frames==count
    assert 'torch' not in sys.modules and 'warp' not in sys.modules
    source=root/'src/deployment/vita_s100_aorta_30dof/onboard_preprocessing.py'
    result={'scope':'CPU full preprocessing replay against saved CUDA oracle, not final action acceptance',
        'frames':len(errors),'max_abs_observation_error':max(errors),'observation_threshold':a.max_observation_error,
        'pass':max(errors)<=a.max_observation_error,'operation':'store_raw_pico + anchor_reset_if_requested + uncached_sequence_gather + full3126_build',
        'cpu_preprocess_ms':{'mean':float(np.mean(elapsed)),'p95':float(np.quantile(elapsed,.95,method='higher')),'max':max(elapsed)},
        'state_checks':'readiness,duplicates,invalidquaternions,NaNs,anchor_preserves_ring,shape,dtype,finite',
        'no_torch_or_warp_imported':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        'cpu_observation_vs_first_cpu_oracle_max_abs':float(np.max(np.abs(np.asarray(obs)-data['cpu_observations'])))}
    if a.output_npz:
        data['cpu_observations']=np.asarray(obs);np.savez(a.output_npz,**data)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
