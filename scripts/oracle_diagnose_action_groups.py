"""CPU exact FP32 cut replay counterfactuals; diagnostic only, CUDA oracle unchanged."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=np.load(a.data);cuts=a.root/'artifacts/model_motion/cuts';opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
sessions={n:ort.InferenceSession(str(cuts/(n+'.onnx')),sess_options=opts,providers=['CPUExecutionProvider']) for n in ['frontend','router','expert_reference','aggregation_reference','backend_after_moe']}
pools=[np.memmap(a.root/f'artifacts/model_motion/pools_nk/expert_pool_{i}_nk.f32',mode='r',dtype=np.float32,shape=shape) for i,shape in enumerate([(1024,512,512),(1024,512,256)])]
F='/actor_module/norm2/Mul_1_output_0';R='/actor_module/Add_4_output_0';S='/actor_module/shared_experts/down_proj/Add_output_0';E='/actor_module/Reshape_11_output_0';G='/actor_module/Unsqueeze_14_output_0';U='/actor_module/Add_6_output_0';W0='/actor_module/Gather_8_output_0';W1='/actor_module/Gather_10_output_0'
gpu=d['gpu_observations'][:6];cpu=d['cpu_observations'][:6]
groups={'current_P':np.arange(9,78),'current_A':np.arange(78,141),'current_R':np.arange(141,267),'current_kinematics':np.arange(0,9),'root_and_robot':np.arange(267,366),'future_P':np.concatenate([np.arange(366+i*258,435+i*258) for i in range(10)]),'future_A':np.concatenate([np.arange(435+i*258,498+i*258) for i in range(10)]),'future_R':np.concatenate([np.arange(498+i*258,624+i*258) for i in range(10)]),'future_kinematics':np.arange(2946,3126)}
groups['all_P']=np.concatenate([groups['current_P'],groups['future_P']]);groups['all_body']=np.concatenate([np.arange(9,267),np.arange(366,2946)])
def replay(observations):
 cache=np.zeros((1,2,1,32,4,64),np.float32);actions=[]
 for i,obs in enumerate(observations):
  front,res,cache=sessions['frontend'].run(None,{'obs':obs[None],'past_key_values':cache,'step_idx':np.array([i],np.int64)})
  _,ids,gates=sessions['router'].run(None,{'obs':obs[None]});ids=ids.ravel();w=[np.ascontiguousarray(np.swapaxes(pool[ids],-1,-2)) for pool in pools]
  shared,routed=sessions['expert_reference'].run(None,{F:front,W0:w[0],W1:w[1]});update=sessions['aggregation_reference'].run(None,{S:shared,E:routed,G:gates})[0]
  actions.append(sessions['backend_after_moe'].run(None,{R:res,U:update})[0].ravel())
 return np.array(actions)
baseline=replay(gpu);native=replay(cpu);target=np.load(a.root/'reference/policy_motion_preprocessing_native_323/reference_actions.npy')[:6]
def metrics(x):
 delta=x.astype(np.float64)-baseline;dtarget=x.astype(np.float64)-target;return dict(max_abs_vs_sameCPU_reference=float(abs(delta).max()),max_abs_vs_originalCUDA_reference=float(abs(dtarget).max()),action27_difference_vs_CPU_reference=delta[:,27].tolist(),action27_difference_vs_CUDA_reference=dtarget[:,27].tolist())
rows=[]
for name,index in groups.items():
 x=gpu.copy();x[:,index]=cpu[:,index];y=cpu.copy();y[:,index]=gpu[:,index]
 rows.append(dict(group=name,gpu_except_groupCPU=metrics(replay(x)),cpu_except_groupGPU=metrics(replay(y))))
report=dict(kind='First6steps ownKV CPU-cut diagnostic counterfactuals; originalCUDAreference preserved; model replacement not under acceptance here',gpu_baseline=metrics(baseline),native_baseline=metrics(native),cases=rows)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
