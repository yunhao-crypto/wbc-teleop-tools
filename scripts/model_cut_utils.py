"""Stream original ONNX into FP32 front/expert-reference/back islands; no compilation/timing."""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper as H,TensorProto as T,numpy_helper as N
import onnxruntime as ort
from model_router_utils import varint

FRONT='/actor_module/norm2/Mul_1_output_0'
RESIDUAL='/actor_module/Add_4_output_0'
SHARED='/actor_module/shared_experts/down_proj/Add_output_0'
ROUTED='/actor_module/Reshape_11_output_0'
GATES='/actor_module/Unsqueeze_14_output_0'
UPDATE='/actor_module/Add_6_output_0'
W0='/actor_module/Gather_8_output_0'
W1='/actor_module/Gather_10_output_0'

def enc(n):
 b=[]
 while n>=128:b.append((n&127)|128);n>>=7
 return bytes(b+[n])

def streamed(source):
 nodes=[];tensors={};large={}
 with source.open('rb') as f:
  while True:
   tag=varint(f);field,wire=tag>>3,tag&7
   if wire==0:varint(f);continue
   assert wire==2;size=varint(f)
   if field==7:end=f.tell()+size;break
   f.seek(size,1)
  while f.tell()<end:
   tag=varint(f);field,wire=tag>>3,tag&7;assert wire==2;size=varint(f)
   if field==1:
    node=onnx.NodeProto();node.ParseFromString(f.read(size));nodes.append(node)
   elif field==5 and size<100_000_000:
    tensor=onnx.TensorProto();tensor.ParseFromString(f.read(size));tensors[tensor.name]=tensor
   elif field==5:
    tend=f.tell()+size;meta=bytearray();raw=None
    while f.tell()<tend:
     tag=varint(f);tf,tw=tag>>3,tag&7
     if tw==0:meta.extend(enc(tag)+enc(varint(f)))
     elif tw==2:
      ts=varint(f)
      if tf==9:raw=(f.tell(),ts);f.seek(ts,1)
      else:meta.extend(enc(tag)+enc(ts)+f.read(ts))
     else:raise ValueError((tf,tw))
    tensor=onnx.TensorProto();tensor.ParseFromString(meta);assert raw
    large[tensor.name]={'shape':list(tensor.dims),'dtype':tensor.data_type,'raw_offset':raw[0],'raw_bytes':raw[1]}
   else:f.seek(size,1)
 return nodes,tensors,large

def vi(name,shape,dtype=T.FLOAT):return H.make_tensor_value_info(name,dtype,shape)

def cut(nodes,tensors,inputs,outputs,name):
 producers={v:n for n in nodes for v in n.output};required=set(outputs);pending=list(outputs);keep=set()
 while pending:
  value=pending.pop()
  if value in inputs:continue
  node=producers.get(value)
  if node is None or node.name in keep:continue
  keep.add(node.name)
  for v in node.input:
   if v and v not in required:required.add(v);pending.append(v)
 selected=[copy.deepcopy(n) for n in nodes if n.name in keep]
 inits=[copy.deepcopy(t) for k,t in tensors.items() if k in required]
 unknown=required-set(inputs)-set(tensors)-set(producers);assert not unknown,unknown
 graph=H.make_graph(selected,name,list(inputs.values()),list(outputs.values()),inits)
 model=H.make_model(graph,opset_imports=[H.make_opsetid('',17)],ir_version=8)
 model=onnx.shape_inference.infer_shapes(model);onnx.checker.check_model(model)
 return model

def tensor_record(t):
 data=N.to_array(t);return {'name':t.name,'shape':list(data.shape),'dtype':str(data.dtype),'elements':data.size,'bytes':data.nbytes,'sha256':hashlib.sha256(data.tobytes()).hexdigest()}

def describe(model,path):
 weights=[tensor_record(t) for t in model.graph.initializer]
 constants=[]
 for n in model.graph.node:
  for a in n.attribute:
   if a.type==onnx.AttributeProto.TENSOR:
    item=tensor_record(a.t);item.update(node=n.name,attribute=a.name);constants.append(item)
 def values(seq):return [{'name':v.name,'dtype':T.DataType.Name(v.type.tensor_type.elem_type),'shape':[d.dim_value for d in v.type.tensor_type.shape.dim]} for v in seq]
 return {'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'inputs':values(model.graph.input),'outputs':values(model.graph.output),'nodes':len(model.graph.node),
         'op_counts':dict(collections.Counter(n.op_type for n in model.graph.node)),
         'weight_elements':sum(t['elements'] for t in weights),'weight_bytes':sum(t['bytes'] for t in weights),
         'initializers':weights,'tensor_constants':constants,'node_order':[{'name':n.name,'op':n.op_type,'inputs':list(n.input),'outputs':list(n.output)} for n in model.graph.node]}

def err(actual,expected):
 d=actual.astype(np.float64)-expected
 return {'bitwise_equal':actual.tobytes()==expected.tobytes(),'max_abs':float(np.abs(d).max()),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(expected.astype(np.float64)),1e-12))}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--router',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
 assert not a.out_dir.exists();a.out_dir.mkdir(parents=True)
 nodes,tensors,pools=streamed(a.source)
 assert set(pools)=={'actor_module.layers.0.gate_up_proj','actor_module.layers.0.down_proj'},pools
 shapes={UPDATE:[1,1,512],FRONT:[1,1,512],RESIDUAL:[1,1,512],SHARED:[1,1,512],ROUTED:[1,2,512],GATES:[1,2,1],W0:[2,512,512],W1:[2,256,512],'obs':[1,604],'past_key_values':[1,2,1,32,4,64],'present_key_values':[1,2,1,32,4,64],'step_idx':[1],'actions':[1,29]}
 def specs(names):return {name:vi(name,shapes[name],T.INT64 if name=='step_idx' else T.FLOAT) for name in names}
 models={
  'frontend':cut(nodes,tensors,specs(['obs','past_key_values','step_idx']),specs([FRONT,RESIDUAL,'present_key_values']),'exact_fp32_frontend'),
  'expert_reference':cut(nodes,tensors,specs([FRONT,W0,W1]),specs([SHARED,ROUTED]),'exact_fp32_expert_reference'),
  'backend':cut(nodes,tensors,specs([RESIDUAL,SHARED,ROUTED,GATES]),specs(['actions']),'exact_fp32_backend'),
  'aggregation_reference':cut(nodes,tensors,specs([SHARED,ROUTED,GATES]),specs([UPDATE]),'exact_fp32_moe_aggregation'),
  'backend_after_moe':cut(nodes,tensors,specs([RESIDUAL,UPDATE]),specs(['actions']),'exact_fp32_backend_after_moe')}
 descriptions={}
 for name,model in models.items():
  path=a.out_dir/(name+'.onnx');onnx.save(model,str(path));descriptions[name]=describe(model,path)
  (a.out_dir/(name+'_manifest.json')).write_text(json.dumps(descriptions[name],indent=2)+'\n')
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
 sessions={k:ort.InferenceSession(m.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']) for k,m in models.items()}
 router=ort.InferenceSession(str(a.router),sess_options=opts,providers=['CPUExecutionProvider'])
 obs=np.load(a.data/'observations.npy');refs=np.load(a.data/'reference_1064steps.npz');cases=[];last_cache=None
 for step in [0,1,33]:
  cache=np.zeros((1,2,1,32,4,64),np.float32) if step==0 else (last_cache if step==1 else refs['caches'][step-1])
  router_out=router.run(None,{'obs':obs[step]});ids=router_out[1].reshape(2).astype(int);gate=router_out[2]
  front_values=sessions['frontend'].run(None,{'obs':obs[step],'past_key_values':cache,'step_idx':np.array([step],np.int64)})
  selected=[]
  with a.source.open('rb') as f:
   for pool_name in ['actor_module.layers.0.gate_up_proj','actor_module.layers.0.down_proj']:
    meta=pools[pool_name];assert meta['dtype']==T.FLOAT and meta['shape'][0]==1024
    row_elements=math.prod(meta['shape'][1:]);rows=[]
    for index in ids:
     f.seek(meta['raw_offset']+int(index)*row_elements*4);raw=f.read(row_elements*4);assert len(raw)==row_elements*4
     rows.append(np.frombuffer(raw,dtype='<f4').reshape(meta['shape'][1:]))
    selected.append(np.stack(rows))
  shared,routed=sessions['expert_reference'].run(None,{FRONT:front_values[0],W0:selected[0],W1:selected[1]})
  action=sessions['backend'].run(None,{RESIDUAL:front_values[1],SHARED:shared,ROUTED:routed,GATES:gate})[0]
  update=sessions['aggregation_reference'].run(None,{SHARED:shared,ROUTED:routed,GATES:gate})[0]
  after_moe=sessions['backend_after_moe'].run(None,{RESIDUAL:front_values[1],UPDATE:update})[0]
  assert after_moe.tobytes()==action.tobytes(),'Alternate MoE-update cut changed action bytes'
  action_err=err(action,refs['actions'][step]);cache_err=err(front_values[2],refs['caches'][step]);route_exact=np.array_equal(ids,refs['expert_indices'][step].reshape(2))
  assert action_err['max_abs']<2e-5 and cache_err['max_abs']<2e-5 and route_exact,(step,action_err,cache_err,route_exact)
  dest=a.out_dir/f'case_{step:04d}';dest.mkdir()
  arrays={'obs':obs[step],'past_key_values':cache,'step_idx':np.array([step],np.int64),'frontend_norm2':front_values[0],'attention_residual':front_values[1],'present_key_values':front_values[2],'shared_output':shared,'routed_output':routed,'gates':gate,'expert_ids':ids,'actions':action,'moe_update':update,'actions_after_moe':after_moe,'reference_actions':refs['actions'][step],'reference_present_key_values':refs['caches'][step]}
  np.savez(dest/'case.npz',**arrays)
  case={'step':step,'cache_source':'zero' if step==0 else ('previous composed step' if step==1 else 'original source sequential FP32 reference step32'),'actions':action_err,'backend_after_moe_actions':err(after_moe,refs['actions'][step]),'cache':cache_err,'routes_exact':bool(route_exact),'expert_ids':ids.tolist()};cases.append(case);last_cache=front_values[2]
  print(json.dumps(case),flush=True)
 report={'source':str(a.source),'source_sha256_previously_verified':'2aabb53bd86b1860dd07b24d02710f4b988f9c201655becb45a2378604599ad7','method':'protobuf stream skips expert raw payloads; reads only two selected FP32 rows per case','pool_metadata':pools,'graphs':{k:{p:v for p,v in d.items() if p not in ['node_order','initializers','tensor_constants']} for k,d in descriptions.items()},'reference_dataset':str(a.data),'stateful_cases':cases,'cpu_latency_measured':False,'target_compilation_performed':False,'accuracy_gate':'max_abs < 2e-5 for this extraction check only; not deployment acceptance'}
 (a.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('EXTRACTION_AND_THREE_CASE_PROOF_READY',flush=True)

if __name__=='__main__':main()
