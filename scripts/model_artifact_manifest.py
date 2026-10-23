"""Hash isolated model handoff files on laptop native filesystem."""
from pathlib import Path
import hashlib,json,struct
root=Path(__file__).resolve().parents[1]
paths=['model_motion/cuts/frontend.onnx','model_motion/cuts/backend_after_moe.onnx','model_motion/cuts/router.onnx','model_motion/cuts/router_parameters.bin','model_motion/pools_nk/expert_pool_0_nk.f32','model_motion/pools_nk/expert_pool_1_nk.f32','model_motion/moe/fp32_moe.hbm','model_velocity/original.onnx','model_velocity/vpu/velocity_fp32_vpu.hbm']
records=[]
for relative in paths:
 path=root/'artifacts'/relative;h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 records.append(dict(path='artifacts/'+relative,bytes=path.stat().st_size,sha256=h.hexdigest()))
pack=root/'artifacts/model_motion/cuts/router_parameters.bin'
with pack.open('rb') as f:header=f.read(32)
assert header[:8]==b'VROUT001' and struct.unpack('<6I',header[8:])==(3126,3028,3040,2048,512,1024)
router=json.loads((pack.parent/'router_parameters.json').read_text())
router['sha256']=next(v['sha256'] for v in records if v['path'].endswith('router_parameters.bin'))
router['source_tensors']={'indices':'/actor_module/Constant_2_output_0','mean':'obs_normalizer._mean','divisor':'onnx::Div_565','weight0_padded':'actor_module.router_obs_embed.0.weight; original3028 columns then12 zero columns','bias0':'actor_module.router_obs_embed.0.bias','weight2':'actor_module.router_obs_embed.2.weight','bias2':'actor_module.router_obs_embed.2.bias','router_weight_rows':'onnx::MatMul_647 transposed to [1024,512]'}
(pack.parent/'router_parameters.json').write_text(json.dumps(router,indent=2))
manifest=dict(kind='Vita FP32 dual-policy model artifacts, no quantization or architecture change',source_hashes=dict(motion='881d936f560e22e81302483b394cd6a89ba5648e6c2893fa153172dec38033a7',velocity='f78fd39eb3f420675af63782d12ce2e6434f62e3159e1fd5b17b65b4fd6d7126'),files=records,compiler='HBDK4 4.7.5 nash-e, optimization2, default MLIR threading',board_validated=False,restriction='No actions or actuator/control lease commands')
(root/'outputs/model_artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
