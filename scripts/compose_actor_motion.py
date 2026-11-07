#!/usr/bin/env python3
"""Compile named-joint NPZ clips into one fixed reference; no robot I/O.

Clips retain their relative root motion. A rigid yaw/XY transform joins world
frames; cosine time ramps bring clip endpoint speeds to zero, and quintic
transitions connect poses. This is reference continuity, not a balance proof.
Run a continuous closed-loop simulation before hardware playback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoothstep(t):
    t = np.asarray(t)
    return t**3 * (10 + t * (-15 + 6*t))


def source_times(duration, fps, ramp_s):
    """C1 time map: zero endpoint speed, unit speed in the interior."""
    ramp = min(ramp_s, duration / 2)
    total = duration + ramp
    count = int(math.ceil(total * fps - 1e-9)) + 1
    playback = np.arange(count) / fps
    t = np.minimum(playback, total)
    source = t - ramp/2
    if ramp:
        first = t < ramp
        source[first] = .5*t[first] - ramp/(2*np.pi)*np.sin(np.pi*t[first]/ramp)
        last = t > total-ramp
        remaining = total-t[last]
        source[last] = duration-(.5*remaining-ramp/(2*np.pi)*np.sin(np.pi*remaining/ramp))
    source = np.clip(source, 0, duration)
    source[0], source[-1] = 0, duration
    return source


def interpolate(q, base, quat, times, target):
    qi = np.stack([np.interp(target, times, col) for col in q.T], axis=1)
    bi = np.stack([np.interp(target, times, col) for col in base.T], axis=1)
    rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    ri = Slerp(times, rotations)(target).as_quat()[:, [3, 0, 1, 2]]
    return qi, bi, ri


def yaw(quat):
    w, x, y, z = quat
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def align_root(base, quat, previous_base, previous_quat):
    rot = Rotation.from_rotvec([0, 0, yaw(previous_quat)-yaw(quat[0])])
    aligned = rot.apply(base-base[0])
    aligned[:, :2] += previous_base[:2]
    aligned[:, 2] += base[0, 2]  # Retain absolute reference height.
    roots = (rot*Rotation.from_quat(quat[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    return aligned, roots


def compose(spec_path, policy_config_path, output):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec = json.loads(spec_path.read_text())
    cfg = json.loads(Path(policy_config_path).read_text())
    if spec.get('schema') != 'vita_actor_sequence_v1' or not spec.get('clips'):
        raise ValueError('Expected vita_actor_sequence_v1 with nonempty clips')
    fps = float(spec.get('fps', 50)); ramp = float(spec.get('endpoint_ramp_s', .6))
    transition = float(spec.get('transition_s', 3)); speed = float(spec.get('transition_max_speed_rad_s', 1.0))
    if not all(math.isfinite(v) for v in (fps,ramp,transition,speed)) or fps != 50 or not .1 <= ramp <= 3 or not .5 <= transition <= 15 or not .1 <= speed <= 2:
        raise ValueError('Invalid sequence timing/speed settings')
    qs, bs, rs, source_ids, source_ts, phases = [], [], [], [], [], []
    timeline, names, n, previous_id = [], None, 0, None
    def append(q, b, r, ids, ts, phase):
        nonlocal n
        qs.append(q);bs.append(b);rs.append(r)
        source_ids.extend(ids);source_ts.extend(ts);phases.extend([phase]*len(q));n += len(q)
    for clip in spec['clips']:
        path = (spec_path.parent/clip['path']).resolve()
        sha = digest(path)
        if sha != clip['sha256']:
            raise ValueError('Clip SHA256 mismatch: '+str(path))
        with np.load(path, allow_pickle=False) as f:
            nn = list(f['joint_names'].astype(str))
            if names is None:
                names = nn
                controlled = [names.index(k) for k in cfg['body_profile']['joint_limits']]
            if nn != names or len(set(nn)) != len(nn) or str(f['joint_space']) != 'virtual_pitch_roll':
                raise ValueError('Inconsistent named virtual-joint schema')
            q, b, r = [np.asarray(f[k],dtype=float) for k in ('joint_pos','base_pos_w','base_quat_w')]
            rate = float(f['fps']); times = np.arange(len(q))/rate
        if q.shape != (len(times),len(names)) or b.shape != (len(times),3) or r.shape != (len(times),4) or len(q)<2 or not all(np.isfinite(x).all() for x in (q,b,r)):
            raise ValueError('Invalid motion arrays')
        if np.max(abs(np.linalg.norm(r,axis=1)-1)) > .02:
            raise ValueError('Expected unit wxyz quaternions')
        start, end = float(clip.get('start_s', 0)), float(clip.get('end_s', times[-1]))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= times[-1]+1e-8):
            raise ValueError('Invalid clip interval')
        target = source_times(end-start, fps, ramp)+start
        q, b, r = interpolate(q,b,r,times,np.minimum(target,times[-1]))
        if n:
            b,r = align_root(b,r,bs[-1][-1],rs[-1][-1])
            delta = q[0]-qs[-1][-1]
            seconds = max(transition, 1.875*np.max(abs(delta[controlled]))/speed)
            steps = int(math.ceil(seconds*fps))
            alpha = smoothstep(np.arange(1,steps)/steps)
            tq = qs[-1][-1]+alpha[:,None]*delta
            tb = bs[-1][-1]+alpha[:,None]*(b[0]-bs[-1][-1])
            roots = Rotation.from_quat(np.stack([rs[-1][-1],r[0]])[:,[1,2,3,0]])
            tr = Slerp([0.,1.],roots)(alpha).as_quat()[:,[3,0,1,2]]
            timeline.append(dict(kind='transition',from_id=previous_id,to_id=clip['id'],start_frame=n-1,end_frame=n-1+steps,start_s=(n-1)/fps,end_s=(n-1+steps)/fps,duration_s=steps/fps,analytic_max_joint_speed_rad_s=float(1.875*np.max(abs(delta[controlled]))/(steps/fps))))
            append(tq,tb,tr,[-1]*len(tq),[float('nan')]*len(tq),'transition')
        else:
            b[:,:2] -= b[0,:2].copy()
        timeline.append(dict(kind='clip',id=clip['id'],title=clip.get('title',''),source_path=str(path),source_sha256=sha,source_start_s=start,source_end_s=end,start_frame=n,end_frame=n+len(q)-1,start_s=n/fps,end_s=(n+len(q)-1)/fps,endpoint_ramp_s=min(ramp,(end-start)/2)))
        append(q,b,r,[clip['id']]*len(q),target.tolist(),'clip')
        previous_id = clip['id']
    q,b,r = map(np.concatenate,(qs,bs,rs))
    # No relaxation/projection here. Final standard loader retains its strict checks.
    for name,(low,high) in cfg['body_profile']['joint_limits'].items():
        col=q[:,names.index(name)]
        if np.any(col<low-1e-4) or np.any(col>high+1e-4):
            raise ValueError('Composed motion exceeds reference limits: '+name)
    for k in range(1,len(r)):
        if np.dot(r[k-1],r[k])<0:r[k]*=-1
    if n>180000:
        raise ValueError('Sequence exceeds loader frame limit')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    filename='sequence.npz'
    np.savez_compressed(output/filename,joint_names=np.array(names),joint_pos=q,base_pos_w=b,base_quat_w=r,fps=np.array(fps),framerate=np.array(fps),joint_space=np.array('virtual_pitch_roll'),quaternion_order=np.array('wxyz'),source_clip_id=np.array(source_ids),source_time_s=np.array(source_ts),sequence_phase=np.array(phases))
    motion_cfg=dict(schema='vita_actor_motion_v1',path=filename,sha256=digest(output/filename),root_body_index=0,end_hold_s=1.0,limit_projection_max_rad=.0001)
    (output/'motion.json').write_text(json.dumps(motion_cfg,indent=2)+'\n')
    receipt=dict(schema='vita_actor_sequence_result_v1',frames=n,duration_s=(n-1)/fps,fps=fps,sha256=motion_cfg['sha256'],spec_sha256=digest(spec_path),policy_config_sha256=digest(policy_config_path),timeline=timeline,root_alignment='rigid yaw and XY at each boundary; absolute height preserved',endpoint_timing='cosine speed ramps, unit speed in each interior',transition='quintic position and SLERP with zero analytic endpoint velocity and acceleration',validation='reference limits checked; continuous closed-loop validation still required',actions_sent=0)
    (output/'sequence_manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    (output/'sequence_spec.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--policy-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compose(a.spec,a.policy_config,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='timeline'},indent=2))

if __name__=='__main__':main()
