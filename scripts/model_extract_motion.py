"""Extract Vita FP32 islands and dynamic router pack; small synthetic original oracle proof."""
import argparse, json, hashlib, struct
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper as N, TensorProto as T
import onnxruntime as ort
from model_cut_utils import streamed, cut, vi, describe, err, FRONT, RESIDUAL, SHARED, ROUTED, GATES, UPDATE, W0, W1

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-source-sha256',required=True);a=p.parse_args()
digest=hashlib.sha256()
with a.source.open('rb') as f:
 for data in iter(lambda:f.read(8*1024**2),b''):digest.update(data)
if digest.hexdigest()!=a.expected_source_sha256:raise ValueError('Source ONNX SHA256 mismatch')
a.output.mkdir(parents=True,exist_ok=False)
nodes,tensors,pools=streamed(a.source)
assert set(pools)=={'actor_module.layers.0.gate_up_proj','actor_module.layers.0.down_proj'}
obs_width=int(tensors['obs_normalizer._mean'].dims[-1]);action_width=int(tensors['actor_module.actor_head.2.bias'].dims[0]) if 'actor_module.actor_head.2.bias' in tensors else 30
assert obs_width==3126 and action_width==30
shapes={UPDATE:[1,1,512],FRONT:[1,1,512],RESIDUAL:[1,1,512],SHARED:[1,1,512],ROUTED:[1,2,512],GATES:[1,2,1],W0:[2,512,512],W1:[2,256,512],'obs':[1,obs_width],'past_key_values':[1,2,1,32,4,64],'present_key_values':[1,2,1,32,4,64],'step_idx':[1],'actions':[1,action_width],'moe_layer_0_expert_logits':[1,1,1024],'moe_layer_0_expert_indices':[1,1,2]}
def specs(names):return {n:vi(n,shapes[n],T.INT64 if n in ('step_idx','moe_layer_0_expert_indices') else T.FLOAT) for n in names}
contracts={
 'frontend':(['obs','past_key_values','step_idx'],[FRONT,RESIDUAL,'present_key_values']),
 'expert_reference':([FRONT,W0,W1],[SHARED,ROUTED]),
 'backend_after_moe':([RESIDUAL,UPDATE],['actions']),
 'aggregation_reference':([SHARED,ROUTED,GATES],[UPDATE]),
 'router':(['obs'],['moe_layer_0_expert_logits','moe_layer_0_expert_indices',GATES])}
models={};descriptions={}
for name,(ins,outs) in contracts.items():
 model=cut(nodes,tensors,specs(ins),specs(outs),'vita_original_fp32_'+name)
 path=a.output/(name+'.onnx');onnx.save(model,path);models[name]=model;descriptions[name]=describe(model,path)
 (a.output/(name+'_manifest.json')).write_text(json.dumps(descriptions[name],indent=2))
weights={t.name:N.to_array(t) for t in models['router'].graph.initializer}
constants={n.output[0]:N.to_array(next(t.t for t in n.attribute if t.name=='value')) for n in models['router'].graph.node if n.op_type=='Constant'}
indices=constants['/actor_module/Constant_2_output_0'].astype('<i4')
first=weights['actor_module.router_obs_embed.0.weight'];width=first.shape[1];padded_width=(width+15)//16*16
assert indices.shape==(width,) and width==3028 and len(set(indices.tolist()))==width
arrays=[('mean',weights['obs_normalizer._mean']),('divisor',weights['onnx::Div_565']),('weight0_padded',np.pad(first,((0,0),(0,padded_width-width)))),('bias0',weights['actor_module.router_obs_embed.0.bias']),('weight2',weights['actor_module.router_obs_embed.2.weight']),('bias2',weights['actor_module.router_obs_embed.2.bias']),('router_weight_rows',weights['onnx::MatMul_647'].T)]
with (a.output/'router_parameters.bin').open('wb') as f:
 f.write(b'VROUT001'+struct.pack('<6I',obs_width,width,padded_width,2048,512,1024))
 indices.tofile(f)
 for _,v in arrays:np.ascontiguousarray(v,dtype='<f4').tofile(f)
router_record=dict(obs_width=obs_width,indices=width,padded_width=padded_width,arrays=[dict(name=n,shape=list(v.shape),elements=int(v.size)) for n,v in arrays],precision='original FP32, no quantization',format='VROUT001+6uint32(obs,selected,padded,hidden,dmodel,experts), then int32 indices followed by float32 arrays in manifest order, little endian')
(a.output/'router_parameters.json').write_text(json.dumps(router_record,indent=2))
digest=hashlib.sha256()
with a.source.open('rb') as f:
 for data in iter(lambda:f.read(8*1024**2),b''):digest.update(data)
assert digest.hexdigest()==a.expected_source_sha256
report=dict(source=str(a.source),source_sha256_previously_verified=digest.hexdigest(),pool_metadata=pools,graphs={n:{k:v for k,v in d.items() if k not in ('node_order','initializers','tensor_constants')} for n,d in descriptions.items()})
(a.output/'manifest.json').write_text(json.dumps(report,indent=2));print('ARTIFACTS_READY',flush=True)
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
sessions={n:ort.InferenceSession(m.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider']) for n,m in models.items()}
original=ort.InferenceSession(str(a.source),sess_options=options,providers=[('CUDAExecutionProvider',{'use_tf32':0}), 'CPUExecutionProvider'])
assert original.get_providers()[0]=='CUDAExecutionProvider',original.get_providers()
rng=np.random.default_rng(73029);mean=arrays[0][1];divisor=arrays[1][1]
cache=np.zeros((1,2,1,32,4,64),np.float32);refcache=cache.copy();cases=[]
for step in range(34):
 obs=np.asarray(mean+divisor*rng.normal(0,.3,(1,obs_width)),dtype=np.float32)
 feeds={'obs':obs,'past_key_values':refcache,'step_idx':np.array([step],np.int64)};ref=original.run(None,feeds);refcache=ref[1]
 logits,ids,gates=sessions['router'].run(None,{'obs':obs});ids=ids.reshape(2)
 front,residual,nextcache=sessions['frontend'].run(None,dict(feeds,past_key_values=cache))
 selected=[]
 with a.source.open('rb') as f:
  for name in ['actor_module.layers.0.gate_up_proj','actor_module.layers.0.down_proj']:
   meta=pools[name];shape=meta['shape'][1:];n=int(np.prod(shape))*4;rows=[]
   for idx in ids:
    f.seek(meta['raw_offset']+int(idx)*n);rows.append(np.frombuffer(f.read(n),dtype='<f4').reshape(shape))
   selected.append(np.stack(rows))
 shared,routed=sessions['expert_reference'].run(None,{FRONT:front,W0:selected[0],W1:selected[1]})
 update=sessions['aggregation_reference'].run(None,{SHARED:shared,ROUTED:routed,GATES:gates})[0]
 action=sessions['backend_after_moe'].run(None,{RESIDUAL:residual,UPDATE:update})[0]
 row=dict(step=step,actions=err(action,ref[0]),cache=err(nextcache,ref[1]),routes_exact=bool(np.array_equal(ids,ref[2].reshape(2))))
 # TopK may reverse equal-score experts across CPU/CUDA. The weighted sum
 # is order independent; preserve the order diagnostic and require identical
 # expert membership plus the unchanged end-to-end action tolerance.
 row['route_set_exact']=bool(np.array_equal(np.sort(ids),np.sort(ref[2].reshape(2))))
 assert row['actions']['max_abs']<=1e-4 and row['route_set_exact'] and np.isfinite(action).all(),row
 cases.append(row)
 if step in (0,1,33):
  dest=a.output/f'case_{step:04d}';dest.mkdir(exist_ok=True)
  np.savez(dest/'case.npz',obs=obs,past_key_values=cache,step_idx=feeds['step_idx'],frontend_norm2=front,attention_residual=residual,present_key_values=nextcache,gates=gates,expert_ids=ids,actions=action,moe_update=update,reference_actions=ref[0],reference_present_key_values=ref[1])
 cache=nextcache
proof=dict(kind='34 synthetic steps, independent recurrent cache, original CUDA FP32 use_tf32=0 oracle; includes context wrap; subgraph proof only',cases=cases,maximum_action_error=max(x['actions']['max_abs'] for x in cases))
(a.output/'cut_proof.json').write_text(json.dumps(proof,indent=2));print('CUT_PROOF_PASS',proof['maximum_action_error'],flush=True)
