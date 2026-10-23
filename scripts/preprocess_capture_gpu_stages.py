#!/usr/bin/env python3
"""Capture existing GPU intermediate arithmetic for CPU port diagnosis only."""
import argparse,os,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'src'),str(root/'src/deployment/unitree_g1_ros2_29dof/src')]
 os.environ['HOLOMOTION_SMPL_MODEL_PATH']=a.model
 from humanoid_policy.smpl_reference import GpuOnlineSmplReference
 r=GpuOnlineSmplReference(n_future_frames=10,policy_hz=50)
 with np.load(a.input) as f: raw=f['body_poses'];ts=f['sample_times']
 out={k:[] for k in ['pose','trans','positions','rotations','body','feet','root','pico_global']}
 for i,(body,t) in enumerate(zip(raw,ts)):
  r.store_pico(body,sample_time=float(t),frame_index=i)
  for name,value in [('pico_global',r._pico_converter.global_quat_dev.numpy()[0]),('pose',r._pico_converter.pose_dev.numpy()[0]),('trans',r._pico_converter.trans_dev.numpy()[0]),('positions',r._global_pos.numpy()[0]),('rotations',r._global_quat.numpy()[0]),('body',r._body_ring.numpy()[(r._write_slot-1)%r.capacity]),('feet',r._foot_z_ring.numpy()[(r._write_slot-1)%r.capacity]),('root',r._root_quat_ring.numpy()[(r._write_slot-1)%r.capacity])]:out[name].append(value)
 np.savez(a.out,**{k:np.asarray(v) for k,v in out.items()});print('GPU_STAGES_READY',len(raw))
if __name__=='__main__':main()
