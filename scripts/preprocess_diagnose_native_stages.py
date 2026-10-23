import ctypes,json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from deployment.vita_s100_aorta_30dof.onboard_preprocessing import CPUPicoToSmplConverter,load_skeleton,_ptr
lib=ctypes.CDLL(str(root/'artifacts/preprocessing_native_probe_x86.so'));fp=ctypes.POINTER(ctypes.c_float);ip=ctypes.POINTER(ctypes.c_int32)
lib.pp_probe_fk.argtypes=[fp,ip,fp,fp,fp];lib.pp_probe_roundtrip.argtypes=[fp,fp,fp,fp];lib.pp_pico.argtypes=[fp,ip,fp,fp]
parents,offsets,_=load_skeleton(root/'artifacts/preprocessing_skeleton.npz')
data=np.load(root/'reference/preprocessing_synthetic.npz');gpu=np.load(root/'reference/preprocessing_gpu_stages.npz')
errs={};saved={}
def compare(k,c,g):
 d=np.abs(c.astype(np.float64)-g.astype(np.float64));m=float(d.max());same=int(np.count_nonzero(c==g));total=c.size
 errs[k]={'max':m,'matching':same,'total':total}
poses=[];pos=[];qs=[];gpos=[];gqs=[];roundpos=[];roundlocal=[]
for raw in data['body_poses']:
 pose=np.empty((24,3),np.float32);trans=np.empty(3,np.float32)
 lib.pp_pico(_ptr(raw),_ptr(parents),_ptr(pose),_ptr(trans));poses.append(pose)
for i,pose in enumerate(poses):
 for p,ps,rs in [(pose,pos,qs),(gpu['pose'][i],gpos,gqs)]:
  pp=np.empty((24,3),np.float32);qq=np.empty((24,4),np.float32)
  lib.pp_probe_fk(_ptr(p),_ptr(parents),_ptr(offsets),_ptr(pp),_ptr(qq));ps.append(pp);rs.append(qq)
 rp=[];rl=[]
 for j in range(1,4):
  pp=np.empty(3,np.float32);ll=np.empty(3,np.float32)
  lib.pp_probe_roundtrip(_ptr(gpu['rotations'][i,0]),_ptr(offsets[j]),_ptr(pp),_ptr(ll));rp.append(pp);rl.append(ll[[2,0,1]])
 roundpos.append(rp);roundlocal.append(rl)
compare('pico_pose',np.array(poses).reshape(-1,72),gpu['pose'])
compare('own_pose_fk_positions',np.array(pos),gpu['positions']);compare('own_pose_rootquats',np.array(qs)[:,0],gpu['rotations'][:,0])
compare('gpu_pose_fk_positions',np.array(gpos),gpu['positions']);compare('gpu_pose_rootquats',np.array(gqs)[:,0],gpu['rotations'][:,0])
compare('gpu_rootquat_rootchildren_worldpositions',np.array(roundpos),gpu['positions'][:,1:4])
compare('gpu_rootquat_rootchildren_localpositions',np.array(roundlocal),gpu['body'][:,:9].reshape(-1,3,3))
for i in [54,55,65,137,138,148]:
 print('FRAME',i,'rootquatcpu',np.array(qs)[i,0].tolist(),'gpu',gpu['rotations'][i,0].tolist(),'bodygpu',gpu['body'][i,:9].tolist(),'roundtripgpuroot',np.array(roundlocal)[i].reshape(-1).tolist())
print(json.dumps(errs,indent=2));(root/'reference/preprocessing_native_stage_diagnosis.json').write_text(json.dumps(errs,indent=2)+'\n')
