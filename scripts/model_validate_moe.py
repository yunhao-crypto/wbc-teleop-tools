"""Three exact original-policy cases versus all-FP32 VPU MoE."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort
from hbdk4.compiler import load
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--references',type=Path,required=True);parser.add_argument('--pools',type=Path,required=True);args=parser.parse_args();p=args.candidate;d=args.references;m=load(str(p/'fp32_moe.bc'))
pools=[np.memmap(args.pools/f'expert_pool_{i}_nk.f32',mode='r',dtype=np.float32,shape=shape) for i,shape in enumerate([(1024,512,512),(1024,512,256)])]
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;backend=ort.InferenceSession(str(d/'backend_after_moe.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);rows=[]
def metrics(a,b):
 delta=a.astype(np.float64)-b;return dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(b)),max_abs_error=float(abs(delta).max()))
assert [v.name for v in m[0].inputs]==['selected_expert_weights_0','selected_expert_weights_1','normalized_expert_gates','norm2_hidden']
for folder in sorted(d.glob('case_*')):
 case=np.load(folder/'case.npz');ids=case['expert_ids'];actual=m[0](np.ascontiguousarray(pools[0][ids]),np.ascontiguousarray(pools[1][ids]),case['gates'],case['frontend_norm2'])[0]
 actions=backend.run(None,{'/actor_module/Add_4_output_0':case['attention_residual'],'/actor_module/Add_6_output_0':actual})[0]
 row=dict(case=folder.name,expert_indices=ids.tolist(),moe_update=metrics(actual,case['moe_update']),actions_after_original_fp32_backend=metrics(actions,case['reference_actions']))
 assert np.isfinite(actual).all() and row['moe_update']['relative_l2']<1e-5 and row['actions_after_original_fp32_backend']['max_abs_error']<=1e-4,row
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(kind='Three original-FP32 boundary cases; all-FP32 converted VPU MoE, no full suite/timing/board claim',cases=rows);(p/'fp32_reference_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
