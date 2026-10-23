"""Original full CUDA policy sensitivity to paired CPU/GPU preprocessing, no robot I/O."""
import argparse,json,time
from pathlib import Path
import numpy as np,onnxruntime as ort
from oracle_policy_replay import digest
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
assert digest(a.model)=='881d936f560e22e81302483b394cd6a89ba5648e6c2893fa153172dec38033a7'
d=np.load(a.data);gpu=d['gpu_observations'];cpu=d['cpu_observations'];frames=d['raw_frame_indices'];resetframes=set(d['reset_raw_frames'].tolist());n=len(cpu)
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
s=ort.InferenceSession(str(a.model),sess_options=opts,providers=[('CUDAExecutionProvider',{'use_tf32':0}),'CPUExecutionProvider']);assert s.get_providers()[0]=='CUDAExecutionProvider' and s.get_provider_options()['CUDAExecutionProvider']['use_tf32']=='0'
caches=[np.zeros((1,2,1,32,4,64),np.float32) for _ in range(2)];step=0;actions=np.empty((2,n,30),np.float32);ids=np.empty((2,n,2),np.int64);logits=np.empty((2,n,1024),np.float32);kv=np.empty((2,n,1,2,1,32,4,64),np.float32);steps=np.empty(n,np.int64);resets=np.zeros(n,np.uint8)
for i,frame in enumerate(frames):
 if i==0 or int(frame) in resetframes:
  for cache in caches:cache.fill(0)
  step=0;resets[i]=1
 steps[i]=step
 for stream,observations in enumerate((gpu,cpu)):
  result=s.run(None,{'obs':observations[i:i+1],'past_key_values':caches[stream],'step_idx':np.array([step],np.int64)})
  for value in result:assert np.isfinite(value).all(),(i,stream)
  actions[stream,i]=result[0].reshape(30);np.copyto(caches[stream],result[1]);ids[stream,i]=result[2].reshape(2);logits[stream,i]=result[3].reshape(1024);kv[stream,i]=result[1]
 step+=1
delta=abs(actions[1].astype(np.float64)-actions[0]);worst=np.unravel_index(int(delta.argmax()),delta.shape)
for name,value in dict(observations=cpu[:,None,:],gpu_observations=gpu[:,None,:],reference_actions=actions[0],cpu_observation_original_actions=actions[1],reference_expert_indices=ids[0],cpu_observation_original_indices=ids[1],reference_logits=logits[0],reference_caches=kv[0],step_indices=steps,reset_before=resets,raw_frame_indices=frames).items():np.save(a.output/(name+'.npy'),value)
cpu.tofile(a.output/'observations.f32');actions[0].tofile(a.output/'reference_actions.f32');ids[0].tofile(a.output/'reference_expert_indices.i64');steps.tofile(a.output/'step_indices.i64');resets.tofile(a.output/'reset_before.u8')
report=dict(kind='Original full CUDA FP32 policy with paired CPU versus GPU preprocessing observations, each own recurrent KV; synthetic rawPico input',steps=n,maximum_raw_action_error=float(delta.max()),worst_step=int(worst[0]),worst_action_coordinate=int(worst[1]),worst_raw_frame=int(frames[worst[0]]),coordinates_exceeding_1e4=int((delta>1e-4).sum()),passed=bool(np.isfinite(delta).all() and delta.max()<=1e-4),ordered_routes_equal=bool(np.array_equal(ids[0],ids[1])),max_cache_error=float(abs(kv[1].astype(np.float64)-kv[0]).max()),reset_before_indices=np.flatnonzero(resets).tolist(),reset_raw_frames=d['reset_raw_frames'].tolist(),data_source=str(a.data),data_source_sha256=digest(a.data),source_model=str(a.model),source_model_sha256=digest(a.model),reference_kind='Full original CUDA FP32 use_tf32=0 consumes GPU preprocessing; candidate board should consume CPU observations and ownKV',files=[dict(file=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(a.output.iterdir())],limitations='Synthetic rawPico/robot inputs. Frozen source last_action observation fields; not candidate-driven closed loop. Does not include board hybrid arithmetic until root replay.')
(a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
