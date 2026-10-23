"""Bind one FP32 S100 package to its original ONNX and deployment contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import onnx
from model_router_utils import varint


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_contract(path):
    """Read ONNX interfaces/metadata without loading 1.5 GiB expert tensors."""
    result = {'inputs': [], 'outputs': [], 'metadata_props': {}}
    with Path(path).open('rb') as f:
        while (tag := varint(f)) is not None:
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                varint(f)
                continue
            if wire != 2:
                raise ValueError('Unexpected ONNX wire type')
            size = varint(f)
            end = f.tell() + size
            if field == 7:
                while f.tell() < end:
                    graph_tag = varint(f)
                    if graph_tag & 7 != 2:
                        raise ValueError('Unexpected graph wire type')
                    count = varint(f)
                    graph_field = graph_tag >> 3
                    if graph_field in (11, 12):
                        value = onnx.ValueInfoProto()
                        value.ParseFromString(f.read(count))
                        entry = {'name': value.name,
                            'dtype': onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                            'shape': [d.dim_value if d.HasField('dim_value') else d.dim_param
                                      for d in value.type.tensor_type.shape.dim]}
                        result['inputs' if graph_field == 11 else 'outputs'].append(entry)
                    else:
                        f.seek(count, 1)
            elif field == 14:
                value = onnx.StringStringEntryProto()
                value.ParseFromString(f.read(size))
                result['metadata_props'][value.key] = value.value
            else:
                f.seek(size, 1)
    return result


def validate_motion_contract(motion, baseline, *, group, expected=None):
    for key in ('inputs', 'outputs'):
        if motion[key] != baseline[key]:
            raise ValueError('New model changed the deployed ABI: ' + key)
    if group in ('P1', 'P2', 'P3'):
        if expected is None or motion != expected:
            raise ValueError('P1/P2/P3 require the exact reviewed Motion contract')
        allowed = {'joint_stiffness', 'joint_damping', 'action_scale', 'joint_physics_contract'}
        before, after = baseline['metadata_props'], motion['metadata_props']
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        if changed - allowed:
            raise ValueError('Unreviewed metadata changes: ' + str(sorted(changed - allowed)))
        physics = json.loads(after.get('joint_physics_contract', '{}'))
        if physics.get('schema_version') != 2:
            raise ValueError('P1/P2/P3 require joint physics schema 2')
        import math
        for key in ('joint_stiffness', 'joint_damping', 'action_scale'):
            values = [float(x) for x in after[key].split(',')]
            if len(values) != 30 or not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError('Invalid Motion parameter: ' + key)
    elif expected is not None or motion['metadata_props'] != baseline['metadata_props']:
        raise ValueError('New model changed the deployed contract: metadata_props')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=(*tuple('ABCDE'), 'P1', 'P2', 'P3'), required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--expected-motion-contract', type=Path,
        help='Reviewed exact ONNX contract for P1/P2/P3; old groups retain baseline equality')
    args = parser.parse_args()
    if sha(args.source) != args.expected_source_sha256 or args.iteration <= 0:
        raise ValueError('Invalid source identity/iteration')
    package = args.output
    cut = json.loads((package / 'cuts/manifest.json').read_text())
    if cut['source_sha256_previously_verified'] != args.expected_source_sha256:
        raise ValueError('Cuts belong to another source model')
    proof = json.loads((package / 'cuts/cut_proof.json').read_text())
    if proof['maximum_action_error'] > 1e-4:
        raise ValueError('Original FP32 extraction proof failed')
    old = json.loads((args.baseline_root / 'artifacts/policy_contracts.json').read_text())
    motion = inspect_contract(args.source)
    validate_motion_contract(motion, old['motion'], group=args.group,
        expected=(json.loads(args.expected_motion_contract.read_text())
                  if args.expected_motion_contract else None))
    motion['sha256'] = args.expected_source_sha256
    contracts = {'velocity': old['velocity'], 'motion': motion}
    (package / 'policy_contracts.json').write_text(json.dumps(contracts, indent=2) + '\n')
    envelopes = json.loads((args.baseline_root / 'artifacts/output_envelopes.json').read_text())
    envelopes['sources']['motion'] = {'path': str(args.source), 'sha256': args.expected_source_sha256}
    for key in ('motion_urdf', 'deployment_vita02a_a_sample_full_interface'):
        item = envelopes['sources'][key]
        source = args.baseline_root / 'artifacts' / item['copied_file']
        if sha(source) != item['sha256']:
            raise ValueError('Invalid original envelope provenance')
        shutil.copyfile(source, package / item['copied_file'])
    (package / 'output_envelopes.json').write_text(json.dumps(envelopes, indent=2) + '\n')
    paths = {'front': 'cuts/frontend.onnx', 'back': 'cuts/backend_after_moe.onnx',
        'router_onnx': 'cuts/router.onnx', 'router': 'cuts/router_parameters.bin',
        'pool0': 'pools_nk/expert_pool_0_nk.f32', 'pool1': 'pools_nk/expert_pool_1_nk.f32',
        'moe': 'moe/fp32_moe.hbm'}
    artifacts = {name: {'path': path, 'sha256': sha(package / path)} for name, path in paths.items()}
    artifacts['velocity'] = {'path': '../../artifacts/model_velocity/original.onnx',
                             'sha256': old['velocity']['sha256']}
    record = {'schema': 'vita_onboard_native_bundle_v1', 'experiment': {
        'group': args.group, 'iteration': args.iteration},
        'source_hashes': {name: contracts[name]['sha256'] for name in ('velocity', 'motion')},
        'model_contract': {'motion_obs': 3126, 'motion_actions': 30,
            'velocity_obs': 990, 'velocity_actions': 28, 'policy_hz': 50, 'rope_reset_step': 8128},
        'artifacts': artifacts}
    (package / 'model_runtime_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PACKAGE_MANIFEST_READY', args.group, args.iteration, args.expected_source_sha256)


if __name__ == '__main__':
    main()
