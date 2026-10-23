"""Compare arithmetic implementation candidates using one original CUDA session."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
s=ort.InferenceSession(a.model,sess_options=opt,providers=[('CUDAExecutionProvider',{'use_tf32':0}),'CPUExecutionProvider']);assert s.get_providers()[0]=='CUDAExecutionProvider'
result={}
for path in a.data:
 d=np.load(path);frames=d['raw_frame_indices'];resets=set(d['reset_raw_frames'].tolist());acts=[];ids=[]
 for observations in (d['gpu_observations'],d['cpu_observations']):
  kv=np.zeros((1,2,1,32,4,64),np.float32);step=0;stream=[];routes=[]
  for i,(obs,frame) in enumerate(zip(observations,frames)):
   if i==0 or int(frame) in resets:kv.fill(0);step=0
   out=s.run(None,{'obs':obs[None],'past_key_values':kv,'step_idx':np.array([step],np.int64)});kv=out[1];step+=1;stream.append(out[0][0]);routes.append(out[2].reshape(2))
  acts.append(np.array(stream));ids.append(np.array(routes))
 diff=np.abs(acts[1].astype(np.float64)-acts[0]);worst=np.unravel_index(diff.argmax(),diff.shape)
 result[path]={'max_abs_action':float(diff.max()),'coordinates_over_1e4':int((diff>1e-4).sum()),'routes_equal':bool(np.array_equal(ids[0],ids[1])),
 'worst_step':int(worst[0]),'worst_raw_frame':int(frames[worst[0]]),'worst_coord':int(worst[1]),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
 print(path,result[path],flush=True)
Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
