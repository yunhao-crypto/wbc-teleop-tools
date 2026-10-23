"""Original FP32 shared+routed MoE with four VPU-friendly matrix-vector products."""
import argparse,copy,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N,TensorProto as T
from hbdk4.compiler import convert,save
from hbdk4.compiler.onnx import export
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
expert=onnx.load(str(a.references/'expert_reference.onnx'));aggregate=onnx.load(str(a.references/'aggregation_reference.onnx'));nodes=list(expert.graph.node)+list(aggregate.graph.node);assert len({n.name for n in nodes})==len(nodes)
initializers={t.name:t for t in expert.graph.initializer}
for t in aggregate.graph.initializer:
 if t.name in initializers:assert t.SerializeToString()==initializers[t.name].SerializeToString()
 else:initializers[t.name]=t
inputs=list(expert.graph.input)+[copy.deepcopy(aggregate.graph.input[-1])];graph=H.make_graph(nodes,'fp32_moe_vpu',inputs,list(aggregate.graph.output),list(initializers.values()));model=H.make_model(graph,opset_imports=expert.opset_import,ir_version=expert.ir_version);model=onnx.shape_inference.infer_shapes(model)
info={v.name:v for v in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info)};weights={t.name:t for t in model.graph.initializer};matmuls=[n for n in model.graph.node if n.op_type=='MatMul'];assert len(matmuls)==4;replacements={};additional=[];newinfo=[];audit=[]
shapes={'/actor_module/shared_experts/gate_up_proj/MatMul':((1,1,512),(1,1,512)), '/actor_module/shared_experts/down_proj/MatMul':((1,1,256),(1,1,512)), '/actor_module/MatMul_2':((2,1,512),(2,1,512)), '/actor_module/MatMul_3':((2,1,256),(2,1,512))}
for n in matmuls:
 x,w=n.input;y=n.output[0];xs,ys=shapes[n.name]
 # ONNX symbolic Slice inference leaves these static chunk dimensions unknown.
 for value,shape_ in [(x,xs),(y,ys)]:
  vi=info[value];known=tuple(d.dim_value for d in vi.type.tensor_type.shape.dim);assert len(known)==len(shape_) and all(a==0 or a==b for a,b in zip(known,shape_));vi.CopyFrom(H.make_tensor_value_info(value,T.FLOAT,shape_))
 ws=tuple(weights[w].dims) if w in weights else tuple(d.dim_value for d in info[w].type.tensor_type.shape.dim);assert ws[-2]==xs[-1] and ws[-1]==ys[-1]
 base=n.name+'::vpu';transpose_shape=ws[:-2]+(ws[-1],ws[-2]);ww=base+'weight_NK'
 if w in weights:
  additional.append(N.from_array(np.ascontiguousarray(np.swapaxes(N.to_array(weights[w]),-1,-2)),ww));kind='static contiguous transposed copy'
 else:
  inp=next(v for v in model.graph.input if v.name==w);inp.CopyFrom(H.make_tensor_value_info(ww,T.FLOAT,transpose_shape));kind='external transposed FP32 weights'
 prod=base+'products';summed=base+'sum';axes=base+'axes';shape=base+'shape';additional.extend([N.from_array(np.array([-1],np.int64),axes),N.from_array(np.array(ys,np.int64),shape)])
 replacements[n.name]=[H.make_node('Mul',[x,ww],[prod],name=base+'mul'),H.make_node('ReduceSum',[prod,axes],[summed],name=base+'reduce',keepdims=0),H.make_node('Reshape',[summed,shape],[y],name=base+'reshape')]
 ps=np.broadcast_shapes(xs,transpose_shape);newinfo.extend([H.make_tensor_value_info(prod,T.FLOAT,ps),H.make_tensor_value_info(summed,T.FLOAT,ps[:-1])]);audit.append(dict(node=n.name,input_shape=list(xs),original_weight_shape=list(ws),transposed_weight_shape=list(transpose_shape),weight_input=ww,weight_kind=kind,output_shape=list(ys)))
newnodes=[]
for n in model.graph.node:newnodes.extend(replacements.get(n.name,[n]))
del model.graph.node[:];model.graph.node.extend(newnodes);model.graph.initializer.extend(additional);model.graph.value_info.extend(newinfo)
needed={v for n in model.graph.node for v in n.input};inits=[t for t in model.graph.initializer if t.name in needed];del model.graph.initializer[:];model.graph.initializer.extend(inits)
# Re-infer metadata after changing the two external weight layouts.
external={v.name for v in model.graph.input};values=[v for v in model.graph.value_info if v.name not in external and (v.name in needed or any(v.name in n.output for n in model.graph.node))];del model.graph.value_info[:];model.graph.value_info.extend(values);model=onnx.shape_inference.infer_shapes(model);onnx.checker.check_model(model);onnx.save(model,str(a.output/'fp32_moe.onnx'))
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC;options.optimized_model_filepath=str(a.output/'fp32_moe_folded.onnx')
folded_session=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
model=onnx.load(str(a.output/'fp32_moe_folded.onnx'));defined={v.name for v in list(model.graph.initializer)+list(model.graph.input)+list(model.graph.output)};infos=[v for v in model.graph.value_info if v.name not in defined];del model.graph.value_info[:];model.graph.value_info.extend(infos)
rename={'/actor_module/norm2/Mul_1_output_0':'norm2_hidden','/actor_module/MatMul_2::vpuweight_NK':'selected_expert_weights_0','/actor_module/MatMul_3::vpuweight_NK':'selected_expert_weights_1','/actor_module/Unsqueeze_14_output_0':'normalized_expert_gates','/actor_module/Add_6_output_0':'moe_update'}
for n in model.graph.node:
 n.input[:]=[rename.get(v,v) for v in n.input];n.output[:]=[rename.get(v,v) for v in n.output]
for v in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info):v.name=rename.get(v.name,v.name)
ordered={v.name:copy.deepcopy(v) for v in model.graph.input};del model.graph.input[:];model.graph.input.extend(ordered[n] for n in ['selected_expert_weights_0','selected_expert_weights_1','normalized_expert_gates','norm2_hidden']);onnx.save(model,str(a.output/'fp32_moe_folded.onnx'))
assert not any(n.op_type=='Shape' for n in model.graph.node)
module=export(model);module.replace_index_tensor_type();module=convert(module,march='nash-e',advice=True,advice_path=str(a.output/'advice_'));save(module,str(a.output/'fp32_moe.bc'));(a.output/'fp32_moe.mlir').write_text(module.module.operation.get_asm(enable_debug_info=True,large_elements_limit=4))
cpus=[dict(name=o.name,signature=str(o.op.attributes['signature'])) for o in module[0].operations if o.type=='hbtl.call'];vpu=[dict(name=o.name,type=o.type) for o in module[0].operations if '::vpu' in o.name];assert not any('Linear' in o['signature'] or 'MatMul' in o['signature'] for o in cpus)
report=dict(kind='Original FP32 MoE math; four matvecs rewritten to multiply/reduce, no quantization',matvec_audit=audit,inputs=[dict(name=v.name,type=str(v.value.type)) for v in module[0].inputs],outputs=[dict(name=v.name,type=str(v.value.type)) for v in module[0].outputs],cpu_calls=cpus,matvec_operations=vpu,backend_bytes=(a.output/'fp32_moe.bc').stat().st_size);(a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
