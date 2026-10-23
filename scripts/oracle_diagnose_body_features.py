"""Original CPU router counterfactuals isolate future P/A/R features at route flip."""
import argparse,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();d=np.load(a.root/'reference/preprocessing_synthetic.npz');gpu=d['gpu_observations'][87];cpu=d['cpu_observations'][87]
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;model=a.root/'artifacts/model_motion/cuts/router.onnx';s=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
weights={t.name:onnx.numpy_helper.to_array(t) for t in onnx.load(model).graph.initializer};div=weights['onnx::Div_565'].reshape(-1);mean=weights['obs_normalizer._mean'].reshape(-1)
def run(x):
 logits,ids,_=s.run(None,{'obs':x[None]});v=logits.ravel();return dict(ids=ids.ravel().tolist(),gap=float(v[806])-float(v[129]))
groups={}
for name,start,end in [('P',0,69),('A',69,132),('R',132,258)]:
 groups['all_future_'+name]=np.concatenate([np.arange(366+j*258+start,366+j*258+end) for j in range(10)])
 for j in range(10):groups[f'future{j+1}_{name}']=np.arange(366+j*258+start,366+j*258+end)
for j in range(10):groups[f'future{j+1}_all']=np.arange(366+j*258,366+(j+1)*258)
rows=[]
for name,index in groups.items():
 x=gpu.copy();x[index]=cpu[index];y=cpu.copy();y[index]=gpu[index]
 rows.append(dict(group=name,gpu_except_groupCPU=run(x),cpu_except_groupGPU=run(y),max_abs=float(abs(cpu[index]-gpu[index]).max()),max_normalized_difference=float(abs((cpu[index]-gpu[index])/div[index]).max())))
normdelta=(cpu-gpu)/div;indices=np.arange(366,2946);rank=indices[np.argsort(abs(normdelta[indices]))[-40:][::-1]];singles=[]
for i in rank:
 x=gpu.copy();x[i]=cpu[i];singles.append(dict(dim=int(i),future_frame=int((i-366)//258+1),body_component=int((i-366)%258),cpu=float(cpu[i]),gpu=float(gpu[i]),divisor=float(div[i]),normalized_difference=float(normdelta[i]),gpu_except_dimCPU=run(x)))
report=dict(step=87,rawframe=148,gpu=run(gpu),cpu=run(cpu),groups=rows,normalized_top_coordinates=singles)
(a.root/'outputs/oracle_preprocess_diagnosis_body_features.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
