#!/usr/bin/env python3
"""Host-only import of a matched MimicLite ONNX/YAML into WTT's SDK ABI.

Preserves the trained graph and its normalizers. Adds input splits/reshapes,
selects action only, verifies original/wrapped ORT parity, embeds the reviewed
contract, and pins the wrapped model's size/SHA256. No robot APIs are imported.
Run before building the sealed S100 package; host needs onnx and onnxruntime.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from deployment.vita_s100_aorta_30dof.actor_mimiclite import (
    CONTRACT_KEY, MimicLiteContract, metadata_contract,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def wrap(source, deployment, yaml_sha256):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    raw = metadata_contract(deployment, sha(source), yaml_sha256)
    model = onnx.load_model_from_string(source)
    if any(onnx.external_data_helper.uses_external_data(t)
           for t in onnx.external_data_helper._get_all_tensors(model)):
        raise ValueError('External ONNX tensor files are not supported; export self-contained weights')
    opsets = {o.domain: o.version for o in model.opset_import}
    # The pinned SDK native runtime is not replaced during import. Do not
    # silently downgrade graph semantics to fit an older target runtime.
    if not 13 <= opsets.get('', 0) <= 21 or model.ir_version > 10:
        raise ValueError('Export MimicLite with ONNX opset 13–21 and IR <= 10 for the pinned SDK')
    onnx.checker.check_model(model)
    inputs = {i.name: i for i in model.graph.input}
    if set(inputs) != set(raw['input_groups']):
        raise ValueError('ONNX inputs do not match MimicLite YAML observations')
    output = next((o for o in model.graph.output if o.name == 'action'), None)
    if output is None:
        raise ValueError('MimicLite ONNX is missing the action output')

    def shape(info, width):
        tensor = info.type.tensor_type
        dims = [d.dim_value if d.HasField('dim_value') else None for d in tensor.shape.dim]
        if tensor.elem_type != TensorProto.FLOAT or dims not in ([width], [1, width]):
            raise ValueError('Expected fixed float32 vector or single-batch tensor: '+info.name)
        return dims

    input_shapes = {name: shape(inputs[name], width)
                    for name, width in zip(raw['input_groups'], raw['input_widths'])}
    shape(output, 28)
    existing = ({v for n in model.graph.node for v in (*n.input, *n.output)} | set(inputs)
                | {v.name for v in (*model.graph.initializer, *model.graph.output, *model.graph.value_info)})
    prefix = '__wtt_mimiclite_'
    while any(n.startswith(prefix) for n in existing):
        prefix += 'x'
    input_name = prefix+'observation'
    initializers = list(model.graph.initializer)
    nodes = []

    def constant(name, values):
        key = prefix+name
        initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), key))
        return key

    axis = constant('axis', [1])
    step = constant('step', [1])
    offset = 0
    for name, width in zip(raw['input_groups'], raw['input_widths']):
        sliced = prefix+name+'_slice'
        nodes.append(helper.make_node('Slice', [input_name, constant(name+'_start', [offset]),
            constant(name+'_end', [offset+width]), axis, step], [sliced]))
        nodes.append(helper.make_node('Reshape', [sliced, constant(name+'_shape', input_shapes[name])], [name]))
        offset += width
    nodes.extend(copy.deepcopy(list(model.graph.node)))
    result_name = prefix+'action'
    nodes.append(helper.make_node('Reshape', ['action', constant('output_shape', [1, 28])], [result_name]))
    del model.graph.node[:]; model.graph.node.extend(nodes)
    del model.graph.initializer[:]; model.graph.initializer.extend(initializers)
    del model.graph.input[:]
    model.graph.input.extend([helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, offset])])
    del model.graph.output[:]
    model.graph.output.extend([helper.make_tensor_value_info(result_name, TensorProto.FLOAT, [1, 28])])
    props = {p.key: p.value for p in model.metadata_props}
    props[CONTRACT_KEY] = json.dumps(raw, sort_keys=True, allow_nan=False)
    helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    return model.SerializeToString(), raw, input_shapes


def validate_parity(source, wrapped, contract, shapes):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    original = ort.InferenceSession(source, options, providers=['CPUExecutionProvider'])
    converted = ort.InferenceSession(wrapped, options, providers=['CPUExecutionProvider'])
    rng, peak = np.random.default_rng(42), 0.
    for index in range(9):
        obs = (np.zeros((1, contract['input_dim']), np.float32) if index == 0 else
               rng.normal(size=(1, contract['input_dim'])).astype(np.float32))
        args, offset = {}, 0
        for name, width in zip(contract['input_groups'], contract['input_widths']):
            args[name] = obs[:, offset:offset+width].reshape(shapes[name])
            offset += width
        expected, = original.run(['action'], args)
        actual, = converted.run(None, {converted.get_inputs()[0].name: obs})
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError('Nonfinite MimicLite output during wrapper parity test')
        np.testing.assert_allclose(actual.reshape(28), expected.reshape(28), atol=1e-5, rtol=1e-5)
        peak = max(peak, float(np.max(np.abs(actual.reshape(28)-expected.reshape(28)))))
    return dict(samples=9, max_abs_error=peak, input='synthetic finite inputs; not robot validation')


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def import_policy(root, yaml_path, onnx_path=None, *, replace=False):
    import fcntl
    import yaml
    root, yaml_path = Path(root).resolve(), Path(yaml_path).resolve()
    onnx_path = Path(onnx_path).resolve() if onnx_path else yaml_path.with_suffix('.onnx')
    if onnx_path.stat().st_size > 512*1024*1024:
        raise ValueError('MimicLite ONNX exceeds the reviewed 512 MiB import limit')
    yaml_bytes, source = yaml_path.read_bytes(), onnx_path.read_bytes()
    document = yaml.safe_load(yaml_bytes)
    if not isinstance(document, dict) or not isinstance(document.get('motor_deployment'), dict):
        raise ValueError('Supply the matching MimicLite YAML containing motor_deployment')
    wrapped, contract, shapes = wrap(source, document['motor_deployment'], sha(yaml_bytes))
    parity = validate_parity(source, wrapped, contract, shapes)
    MimicLiteContract.from_metadata(None, {CONTRACT_KEY: json.dumps(contract)})
    if yaml_path.read_bytes() != yaml_bytes or onnx_path.read_bytes() != source:
        raise ValueError('MimicLite source changed during import')
    config_path = root/'config/actor_pico_mimiclite.json'
    # The same lock protects installation/live entry. Import never replaces a
    # model/config while an onboard operator holds it.
    (root/'build').mkdir(exist_ok=True)
    with (root/'build/operator.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another onboard operator is active; MimicLite import is blocked') from error
        config = json.loads(config_path.read_text())
        if config.get('policy_kind') != contract['policy_kind']:
            raise ValueError('Not a MimicLite deployment configuration')
        item = config['artifacts']['model']
        relative = Path(item['path'])
        target = root/relative
        destination = target.resolve()
        if (relative != Path('artifacts/mimiclite/policy.onnx')
                or not destination.is_relative_to(root)
                or any(p.is_symlink() for p in (target, *target.parents)
                       if p != root and p.is_relative_to(root))):
            raise ValueError('Invalid or symlinked MimicLite destination path')
        digest = sha(wrapped)
        if (config.get('model_ready') is True and item.get('sha256') != digest
                or destination.exists() and sha(destination.read_bytes()) != digest):
            if not replace:
                raise RuntimeError('Preserving existing MimicLite model/config; use --replace for an intentional update')
        item.update(sha256=digest, bytes=len(wrapped), source_name=onnx_path.name)
        config['model_ready'] = True
        config['provenance'] = dict(importer='scripts/import_mimiclite_policy.py',
            source_onnx_sha256=sha(source), source_yaml_sha256=sha(yaml_bytes),
            model_sha256=contract['motor_deployment']['model_sha256'],
            actuator_schema=contract['motor_deployment']['actuator_schema'], wrapper_parity=parity,
            robot_validation='not_measured')
        config['reference_window'] = dict(past_frames=-min(contract['motor_deployment']['future_steps']), tail_frames=0)
        atomic_write(destination, wrapped)
        atomic_write(config_path, (json.dumps(config, indent=2, allow_nan=False)+'\n').encode())
    return dict(status='PASS_MIMICLITE_IMPORT_AND_WRAPPER_PARITY', model=str(destination),
        sha256=digest, bytes=len(wrapped), input_groups=contract['input_groups'],
        input_widths=contract['input_widths'], parity=parity,
        actions_sent=0, control_handles_created=0, robot_validation='not_measured')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-yaml', required=True, type=Path)
    parser.add_argument('--onnx', type=Path, help='Defaults to the YAML sibling .onnx')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--replace', action='store_true', help='Explicitly replace a previously pinned MimicLite model')
    args = parser.parse_args()
    print(json.dumps(import_policy(args.root, args.policy_yaml, args.onnx, replace=args.replace), indent=2))


if __name__ == '__main__':
    main()
