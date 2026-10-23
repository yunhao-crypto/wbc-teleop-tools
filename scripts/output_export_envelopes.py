#!/usr/bin/env python3
"""Export original policy_envelope results and hash-verified source provenance."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--contracts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.source))
    from deployment.common.model_contract import load_model_bundle
    from deployment.vita_s100_aorta_30dof.command_guard import command_envelope_from_contract
    from deployment.vita_s100_aorta_30dof.model_contract import inspect_vita_session, VitaBundleContracts
    from deployment.vita_s100_aorta_30dof.policy_limits import policy_envelope
    bundle=load_model_bundle(a.bundle)
    records=json.loads(a.contracts.read_text())
    inputs={}
    for key in ('velocity','motion','motion_urdf','deployment_vita02a_a_sample_full_interface'):
        item=bundle.artifacts[key]
        actual=digest(item.path)
        if actual != item.sha256:
            raise ValueError(f'Original bundle source hash mismatch: {key}')
        inputs[key]={'path':str(item.path),'sha256':actual}
    # Prove exported metadata against both original ONNX files, rather than
    # trusting the record's declared model identity alone. No ORT sessions.
    import onnx
    class View:
        def __init__(self, r): self.r=r
        def tensors(self, key):
            return [SimpleNamespace(name=x['name'],shape=x['shape'],type={'FLOAT':'tensor(float)','INT64':'tensor(int64)'}[x['dtype']]) for x in self.r[key]]
        def get_inputs(self): return self.tensors('inputs')
        def get_outputs(self): return self.tensors('outputs')
        def get_modelmeta(self): return SimpleNamespace(custom_metadata_map=self.r['metadata_props'])
    for name in ('velocity','motion'):
        if records[name]['sha256'] != inputs[name]['sha256']:
            raise ValueError('Model identity mismatch')
        model=onnx.load(bundle.artifacts[name].path,load_external_data=False)
        metadata={x.key:x.value for x in model.metadata_props}
        if metadata != records[name]['metadata_props']:
            raise ValueError('Original metadata mismatch')
        del model
    contracts=VitaBundleContracts(**{name:inspect_vita_session(View(records[name]),name) for name in ('motion','velocity')})
    base=command_envelope_from_contract(json.loads(bundle.artifacts['deployment_vita02a_a_sample_full_interface'].path.read_text()))
    envelopes={'base':base, **{mode:policy_envelope(base,contracts,bundle,mode) for mode in ('velocity','motion')}}
    values={name:{key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in asdict(env).items()} for name,env in envelopes.items()}
    for module in ('policy_limits.py','command_guard.py','policy_contract.py','robot_adapter.py'):
        path=a.source/'deployment/vita_s100_aorta_30dof'/module
        inputs[module]={'path':str(path),'sha256':digest(path)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for key,suffix in [('motion_urdf','.motion.urdf'),('deployment_vita02a_a_sample_full_interface','.interface.json')]:
        dest=a.output.with_suffix(suffix)
        shutil.copyfile(bundle.artifacts[key].path,dest)
        inputs[key]['copied_file']=dest.name
    result={'schema':'vita_original_output_envelopes_v1','derivation':'original deployment.vita_s100_aorta_30dof.policy_limits.policy_envelope(base, contracts, bundle, mode)',
            'bundle':{'path':str(a.bundle),'sha256':digest(a.bundle)},
            'contracts':{'path':str(a.contracts),'sha256':digest(a.contracts)},
            'sources':inputs,'envelopes':values}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    hashed=digest(a.output)
    a.output.with_suffix('.sha256').write_text(hashed+'  '+a.output.name+'\n')
    print(json.dumps({'artifact':str(a.output),'sha256':hashed,'slots':{k:len(v.joint_names) for k,v in envelopes.items()},'sources_verified':True}))

if __name__=='__main__':main()
