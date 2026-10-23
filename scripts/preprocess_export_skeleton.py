#!/usr/bin/env python3
"""Export only the exact licensed neutral skeleton used by the GPU oracle.

Run in the existing development environment; the target runtime only needs NumPy.
The pickle remains with its owner; this does not grant redistribution rights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--source-root',default=str(Path(__file__).resolve().parents[1]/'src'))
    ap.add_argument('--license',action='append',default=[])
    a=ap.parse_args()
    sys.path.insert(0,a.source_root)
    from holomotion.src.motion_tracking.smpl_reference import load_neutral_smpl_skeleton,SMPL_TO_COMMAND_FRAME
    from holosmpl.converters.pico.adapter import PICO_SMPL_PARENTS_24
    model=Path(a.model).resolve();target=Path(a.output)
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError('Use a new artifact path; refusing to replace evidence')
    skeleton=load_neutral_smpl_skeleton(model)
    parents=skeleton.parents.numpy().astype(np.int32)
    offsets=skeleton.rest_parent_offsets.numpy().astype(np.float32)
    assert np.array_equal(parents,np.array(PICO_SMPL_PARENTS_24,dtype=np.int32))
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez(target,parents=parents,offsets=offsets,command_frame=np.asarray(SMPL_TO_COMMAND_FRAME,dtype=np.float32))
    licenses=[]
    for name in a.license:
        src=Path(name);dst=target.parent/('preprocessing_smpl_notice_'+src.name)
        if not dst.exists(): shutil.copyfile(src,dst)
        if sha(src)!=sha(dst): raise ValueError('License copy identity mismatch')
        licenses.append({'file':dst.name,'sha256':sha(dst)})
    meta={'schema':'vita_cpu_smpl_skeleton_v1','source_model_basename':model.name,
          'source_pickle_sha256':sha(model),'skeleton_hash':skeleton.skeleton_hash,
          'artifact_sha256':sha(target),'parents_shape':[24],'offsets_shape':[24,3],
          'offsets_dtype':'float32','rest_shape':'neutral zero-beta model',
          'source_loader':'holomotion.src.motion_tracking.smpl_reference.load_neutral_smpl_skeleton',
          'source_loader_sha256':sha(Path(a.source_root)/'holomotion/src/motion_tracking/smpl_reference.py'),
          'license_notice':'Derived from licensed SMPL asset. Original SMPL terms and access restrictions still apply. Internal deployment only; no redistribution permission granted.',
          'license_files':licenses,'full_mesh_exported':False}
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'artifact':str(target),'bytes':target.stat().st_size,'source_pickle_sha256':meta['source_pickle_sha256']}))

if __name__=='__main__':main()
