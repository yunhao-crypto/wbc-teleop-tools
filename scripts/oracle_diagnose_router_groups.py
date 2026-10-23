"""CPU-only counterfactuals through original FP32 router; no policy or preprocessing edits."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--router',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.data)
groups={'current_gravity':(0,3),'current_linvel':(3,6),'current_angvel':(6,9),'current_body':(9,267),'current_height':(267,268),'current_yaw':(268,270),'robot_gravity':(270,273),'robot_angvel':(273,276),'robot_q':(276,306),'robot_dq':(306,336),'last_action':(336,366),'future_body':(366,2946),'future_height':(2946,2956),'future_gravity':(2956,2986),'future_linvel':(2986,3016),'future_angvel':(3016,3046),'future_yaw':(3046,3066),'future_rot6d':(3066,3126)}
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;s=ort.InferenceSession(str(a.router),sess_options=opts,providers=['CPUExecutionProvider'])
def evaluate(obs):
 logits,ids,gates=s.run(None,{'obs':obs[None]});v=logits.ravel();order=np.argsort(v)[-4:][::-1]
 return dict(ids=ids.ravel().tolist(),gap_806_minus_129=float(v[806])-float(v[129]),top4=[dict(id=int(i),logit=float(v[i])) for i in order])
rows=[]
for step in [87,125,174,175,176,177,178,179,180,181,302,303,304,305,306]:
 cpu=d['cpu_observations'][step];gpu=d['gpu_observations'][step];diff=cpu.astype(np.float64)-gpu
 row=dict(step=step,raw_frame=int(d['raw_frame_indices'][step]),cpu_baseline=evaluate(cpu),gpu_baseline=evaluate(gpu),groups={},max_root_abs=float(abs(d['cpu_root'][step]-d['gpu_root'][step]).max()))
 for name,(lo,hi) in groups.items():
  part=diff[lo:hi];im=int(abs(part).argmax())+lo;entry=dict(max_abs=float(abs(part).max()),l2=float(np.linalg.norm(part)),worst_dim=im,cpu=float(cpu[im]),gpu=float(gpu[im]))
  if step==87 and np.any(part):
   x=gpu.copy();x[lo:hi]=cpu[lo:hi];entry['gpu_except_this_cpu']=evaluate(x)
   x=cpu.copy();x[lo:hi]=gpu[lo:hi];entry['cpu_except_this_gpu']=evaluate(x)
  row['groups'][name]=entry
 if step==87:
  row['single_coordinate_counterfactuals']=[]
  for dim in np.argsort(abs(diff))[-30:][::-1]:
   x=gpu.copy();x[dim]=cpu[dim];r=evaluate(x);r.update(dim=int(dim),cpu=float(cpu[dim]),gpu=float(gpu[dim]),abs_difference=float(abs(diff[dim])));row['single_coordinate_counterfactuals'].append(r)
 rows.append(row)
a.output.write_text(json.dumps(dict(kind='Original CPU router counterfactual grouping; its GPU baseline route must match original full CUDA before causal inference',groups=groups,cases=rows),indent=2));print(json.dumps(rows[0]),flush=True)
