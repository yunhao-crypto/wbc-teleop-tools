"""Original-model oracle replay only. No robot SDK imports or I/O.

Recorded policy observations are replayed verbatim, including recorded previous
actions; this is policy backend validation, not candidate-driven closed-loop control.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def records(path,width):
 obs=[];line_numbers=[]
 with path.open() as f:
  for i,line in enumerate(f,1):
   item=json.loads(line)
   if item.get('record_type')!='NO_ACTION_PROFILE':continue
   x=np.asarray(item['policy_observation'],dtype=np.float32)
   assert x.size==width and np.isfinite(x).all(),(path,i,x.shape)
   obs.append(x.reshape(width));line_numbers.append(i)
 assert len(obs)>=1000,(path,len(obs))
 return np.stack(obs),np.array(line_numbers,np.int64)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['motion','velocity'],required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,required=True);a=p.parse_args()
 assert not a.output.exists(),'Use a new reference output directory'
 ismotion=a.mode=='motion';width=3126 if ismotion else 990;actionwidth=30 if ismotion else 28
 modelhash=digest(a.model);assert modelhash==('881d936f560e22e81302483b394cd6a89ba5648e6c2893fa153172dec38033a7' if ismotion else 'f78fd39eb3f420675af63782d12ce2e6434f62e3159e1fd5b17b65b4fd6d7126')
 unique,lines=records(a.recording,width);a.output.mkdir(parents=True)
 sourceindex=np.arange(a.steps,dtype=np.int64)%len(unique);obs=np.ascontiguousarray(unique[sourceindex]);obs.tofile(a.output/'observations.f32');np.save(a.output/'observations.npy',obs[:,None,:]);np.save(a.output/'source_indices.npy',sourceindex);np.save(a.output/'source_line_numbers.npy',lines[sourceindex])
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 providers=[('CUDAExecutionProvider',{'device_id':0,'use_tf32':0}),'CPUExecutionProvider'] if ismotion else ['CPUExecutionProvider']
 session=ort.InferenceSession(str(a.model),sess_options=options,providers=providers)
 assert session.get_providers()[0]==('CUDAExecutionProvider' if ismotion else 'CPUExecutionProvider')
 if ismotion:assert session.get_provider_options()['CUDAExecutionProvider']['use_tf32']=='0'
 actions=np.lib.format.open_memmap(a.output/'reference_actions.npy',mode='w+',dtype=np.float32,shape=(a.steps,actionwidth))
 steps=np.zeros(a.steps,dtype=np.int64);reset=np.zeros(a.steps,dtype=np.uint8);reset[0]=1
 if ismotion:
  ids=np.lib.format.open_memmap(a.output/'reference_expert_indices.npy',mode='w+',dtype=np.int64,shape=(a.steps,2));logits=np.lib.format.open_memmap(a.output/'reference_logits.npy',mode='w+',dtype=np.float32,shape=(a.steps,1024));cache=np.zeros((1,2,1,32,4,64),np.float32);state=np.array([0],np.int64)
  cache_indices=np.array(sorted(set(range(0,a.steps,32)) | ({0,1,31,32,33,8126,8127,8128,8129,a.steps-1}&set(range(a.steps)))),np.int64);cache_slots={int(v):j for j,v in enumerate(cache_indices)};np.save(a.output/'cache_indices.npy',cache_indices)
  caches=np.lib.format.open_memmap(a.output/'reference_caches.npy',mode='w+',dtype=np.float32,shape=(len(cache_indices),1,2,1,32,4,64))
 start=time.monotonic()
 for i in range(a.steps):
  feed={'obs':obs[i:i+1]}
  if ismotion:
   if state[0]>=8128:cache.fill(0);state[0]=0;reset[i]=1
   steps[i]=state[0];feed.update(past_key_values=cache,step_idx=state)
  result=session.run(None,feed)
  for value in result:assert np.isfinite(value).all(),(i,'nonfinite original output')
  actions[i]=result[0].reshape(actionwidth)
  if ismotion:
   np.copyto(cache,result[1]);ids[i]=result[2].reshape(2);logits[i]=result[3].reshape(1024);state[0]+=1
   if i in cache_slots:caches[cache_slots[i]]=cache
  if i%1000==0 or i==a.steps-1:print(json.dumps(dict(mode=a.mode,completed=i+1,total=a.steps,elapsed_seconds=time.monotonic()-start)),flush=True)
 actions.flush();actions.tofile(a.output/'reference_actions.f32');steps.tofile(a.output/'step_indices.i64');reset.tofile(a.output/'reset_before.u8');np.save(a.output/'step_indices.npy',steps);np.save(a.output/'reset_before.npy',reset)
 if ismotion:
  ids.flush();logits.flush();caches.flush();ids.tofile(a.output/'reference_expert_indices.i64');logits.tofile(a.output/'reference_logits.f32');caches.tofile(a.output/'reference_caches.f32')
  assert steps[8127]==8127 and steps[8128]==0 and reset.sum()==2
 files=[]
 for f in sorted(a.output.iterdir()):
  if f.is_file():files.append(dict(file=f.name,bytes=f.stat().st_size,sha256=digest(f)))
 manifest=dict(mode=a.mode,steps=a.steps,observation_width=width,action_width=actionwidth,source_model=str(a.model),source_model_sha256=modelhash,source_recording=str(a.recording),source_recording_sha256=digest(a.recording),unique_recorded_observations=len(unique),repetitions=float(a.steps/len(unique)),provenance='Prior no-action pipeline recording: real robot LowState; Motion uses synthetic SMPL reference, not recorded real Pico. Prior recorded previous-action observation fields remain fixed.',replay='Original full model; continuously feeds its own KV, even across repeated-recording seams; automatic zero-KV/step reset only before absolute replay index8128, plus initial zero state. Velocity stateless model; recorded observation history retained.',reset_before_indices=np.flatnonzero(reset).tolist(),provider=session.get_providers(),provider_options=session.get_provider_options(),reference_kind='CUDA FP32 use_tf32=0' if ismotion else 'Original production CPU FP32',elapsed_wall_seconds=time.monotonic()-start,files=files,limitations='Offline repeated observations, not a new uninterrupted real trajectory or candidate-driven closed-loop control; no robot I/O. Board candidate must feed its own KV and compare every raw action coordinate <=1e-4.',raw_layout='All raw binary arrays C-contiguous little endian; observations[N,width], actions[N,actionwidth], ids[N,2], logits[N,1024], caches[len(cache_indices),1,2,1,32,4,64] at explicit cache_indices.npy, step_indices[N], reset_before[N].')
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('ORACLE_READY',str(a.output),flush=True)

if __name__=='__main__':main()
