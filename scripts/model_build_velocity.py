"""Preserve original Velocity FP32 MLP, place four Gemms on VPU with Mul/Reduce."""
import argparse,copy,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N,TensorProto as T
from hbdk4.compiler import convert,save
from hbdk4.compiler.onnx import export
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
original=onnx.load(a.source);model=copy.deepcopy(original);weights={t.name:N.to_array(t) for t in model.graph.initializer};nodes=[];audit=[]
for n in model.graph.node:
 if n.op_type!='Gemm':nodes.append(n);continue
 attrs={v.name:H.get_attribute_value(v) for v in n.attribute}
 assert attrs.get('alpha',1)==1 and attrs.get('beta',1)==1 and attrs.get('transA',0)==0
 w=weights[n.input[1]];w=w if attrs.get('transB',0) else w.T;w=np.ascontiguousarray(w)
 base=n.name+'_fp32_vpu';wn=base+'_NK';axis=base+'_axis';prod=base+'_products';summed=base+'_sum'
 model.graph.initializer.extend([N.from_array(w,wn),N.from_array(np.array([1],np.int64),axis)])
 nodes.extend([H.make_node('Mul',[n.input[0],wn],[prod],name=base+'_mul'),H.make_node('ReduceSum',[prod,axis],[summed],name=base+'_reduce',keepdims=0),H.make_node('Add',[summed,n.input[2]],[base+'_biased'],name=base+'_bias')])
 shape=base+'_shape';model.graph.initializer.append(N.from_array(np.array([1,w.shape[0]],np.int64),shape));nodes.append(H.make_node('Reshape',[base+'_biased',shape],list(n.output),name=base+'_reshape'))
 audit.append(dict(node=n.name,weight_shape_NK=list(w.shape),all_original_values=True))
assert len(audit)==4
del model.graph.node[:];model.graph.node.extend(nodes)
# ReduceSum axes became a tensor input in opset13. The remaining original ELU/Reshape/Add semantics are unchanged.
model.opset_import[0].version=17;used={v for n in nodes for v in n.input};keep=[v for v in model.graph.initializer if v.name in used];del model.graph.initializer[:];model.graph.initializer.extend(keep)
model=onnx.shape_inference.infer_shapes(model);onnx.checker.check_model(model);onnx.save(model,a.output/'velocity_fp32_vpu.onnx')
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
reference=ort.InferenceSession(original.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider']);candidate=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
rng=np.random.default_rng(7341);maximum=0.;worst=None
for i in range(96):
 obs=rng.normal(0,[.01,.1,1.,3.][i%4],(1,990)).astype(np.float32);expected=reference.run(None,{'obs':obs})[0];actual=candidate.run(None,{'obs':obs})[0];delta=abs(actual.astype(np.float64)-expected);error=float(delta.max())
 if error>maximum:maximum=error;worst=dict(case=i,coordinate=int(delta.argmax()),reference=float(expected.flat[delta.argmax()]),candidate=float(actual.flat[delta.argmax()]))
 assert np.isfinite(actual).all() and error<=1e-4,(i,error)
module=convert(export(model),march='nash-e',advice=True,advice_path=str(a.output/'advice_'));save(module,str(a.output/'velocity_fp32_vpu.bc'));(a.output/'velocity_fp32_vpu.mlir').write_text(module.module.operation.get_asm(enable_debug_info=True,large_elements_limit=4))
cpus=[dict(name=o.name,signature=str(o.op.attributes['signature'])) for o in module[0].operations if o.type=='hbtl.call'];assert not any('Linear' in c['signature'] or 'MatMul' in c['signature'] for c in cpus),cpus
backendmaximum=0.
for i in range(24):
 obs=rng.normal(0,[.01,.1,1.,3.][i%4],(1,990)).astype(np.float32);expected=reference.run(None,{'obs':obs})[0];actual=module[0](obs)[0];error=float(abs(actual.astype(np.float64)-expected).max());backendmaximum=max(backendmaximum,error);assert np.isfinite(actual).all() and error<=1e-4,(i,error)
report=dict(method='original FP32 weights, four Gemm become broadcast Mul+ReduceSum+bias; original ELU',matvecs=audit,onnx_proof_cases=96,onnx_max_abs_error=maximum,onnx_worst=worst,converted_proof_cases=24,converted_max_abs_error=backendmaximum,cpu_calls=cpus,quantized=False,board_validated=False)
(a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
