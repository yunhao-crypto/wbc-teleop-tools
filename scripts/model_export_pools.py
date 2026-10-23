"""Stream exact original FP32 expert payloads using audited ONNX tensor offsets.

Run on the laptop native filesystem: avoids round-tripping 1.5 GiB through SSHFS.
No model loading, inference, quantization or CPU latency benchmark.
Optional NK layout transposes each expert by a bit-preserving permutation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True,help='Exact streaming-cut manifest.json with pool_metadata')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--layout',choices=('original','nk'),default='original',help='nk pretransposes every expert row to [N,K] with no arithmetic')
    a=p.parse_args()
    meta=json.loads(a.metadata.read_text())
    specs=[('actor_module.layers.0.gate_up_proj',[1024,512,512]),
           ('actor_module.layers.0.down_proj',[1024,256,512])]
    assert not a.out_dir.exists(),'Use a fresh output directory'
    a.out_dir.mkdir(parents=True)
    needed=sum(math.prod(shape)*4 for _,shape in specs)
    assert shutil.disk_usage(a.out_dir).free>needed+256*1024**2,'Need 1.5 GiB plus 256 MiB free'
    result=[]
    with a.source.open('rb') as source:
        for i,(name,shape) in enumerate(specs):
            info=meta['pool_metadata'][name]
            assert info['shape']==shape and info['dtype']==1
            assert info['raw_bytes']==math.prod(shape)*4
            assert 0<=info['raw_offset']<a.source.stat().st_size-info['raw_bytes']
            suffix='_nk' if a.layout=='nk' else ''
            part=a.out_dir/f'expert_pool_{i}{suffix}.f32.part'
            destination=a.out_dir/f'expert_pool_{i}{suffix}.f32'
            source.seek(info['raw_offset']);left=info['raw_bytes'];digest=hashlib.sha256()
            with part.open('xb') as target:
                if a.layout=='nk':
                    import numpy as np
                    rowbytes=math.prod(shape[1:])*4
                    for expert in range(shape[0]):
                        data=source.read(rowbytes);assert len(data)==rowbytes
                        row=np.frombuffer(data,dtype='<f4').reshape(shape[1:])
                        transposed=np.ascontiguousarray(row.T)
                        # Transpose is a bit-preserving permutation, including signed zeros.
                        assert transposed.T.tobytes(order='C')==data
                        data=transposed.tobytes(order='C')
                        target.write(data);digest.update(data)
                else:
                    while left:
                        data=source.read(min(left,8*1024**2))
                        assert data,'Truncated source ONNX expert payload'
                        target.write(data);digest.update(data);left-=len(data)
            assert part.stat().st_size==info['raw_bytes']
            part.rename(destination)
            record={'index':i,'source_tensor':name,'file':str(destination),'shape':([shape[0],shape[2],shape[1]] if a.layout=='nk' else shape),'source_shape':shape,
                    'flat_shape':[1024,math.prod(shape[1:])],'dtype':'float32','byte_order':'little',
                    'bytes':info['raw_bytes'],'row_bytes':math.prod(shape[1:])*4,
                    'sha256':digest.hexdigest(),'source_raw_offset':info['raw_offset'],
                    'transposed':a.layout=='nk','transpose_axes':([0,2,1] if a.layout=='nk' else None),'layout':a.layout,'quantized':False}
            result.append(record)
            report={'source_onnx':str(a.source),'source_size_bytes':a.source.stat().st_size,
                    'source_sha256_previously_verified':meta['source_sha256_previously_verified'],
                    'method':('Exact source expert rows transposed [0,2,1] as bit-preserving permutation with streaming SHA256; no arithmetic' if a.layout=='nk' else 'Exact raw TensorProto payload slices copied with streaming SHA256; no datatype/layout conversion'),
                    'source_metadata':str(a.metadata),'pools':result,'complete':len(result)==2,
                    'selected_total_bytes_per_step':2*(512*512+256*512)*4}
            (a.out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print('POOL_READY',json.dumps(record),flush=True)
    print('ALL_ORIGINAL_FP32_POOLS_READY',flush=True)


if __name__=='__main__':main()
