#!/usr/bin/env python3
"""Offline import of a matched MimicLite v2 ONNX/YAML for native locomotion."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from deployment.vita_s100_aorta_30dof.actor_mimiclite_reference import (
    CONTRACT_KEY, metadata_contract,
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
    # The native locomotion ONNX Runtime is not replaced during import. Do not
    # silently downgrade graph semantics to fit an older target runtime.
    if not 13 <= opsets.get('', 0) <= 21 or model.ir_version > 10:
        raise ValueError('Export MimicLite with ONNX opset 13–21 and IR <= 10 for the reviewed native runtime')
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


def import_policy(yaml_path, onnx_path, output, name):
    import re
    import yaml

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('Invalid native model name')
    if onnx_path.stat().st_size > 512*1024*1024:
        raise ValueError('MimicLite ONNX exceeds the 512 MiB import limit')
    yaml_bytes, source = yaml_path.read_bytes(), onnx_path.read_bytes()
    document = yaml.safe_load(yaml_bytes)
    if not isinstance(document, dict) or not isinstance(document.get('motor_deployment'), dict):
        raise ValueError('Supply the matching MimicLite YAML containing motor_deployment')
    wrapped, contract, shapes = wrap(source, document['motor_deployment'], sha(yaml_bytes))
    parity = validate_parity(source, wrapped, contract, shapes)
    contract_data = (json.dumps(contract, indent=2, allow_nan=False)+'\n').encode()
    model_file, contract_file = name+'.onnx', 'contract_'+name+'.json'
    output.mkdir(parents=True, exist_ok=False)
    (output/model_file).write_bytes(wrapped)
    (output/contract_file).write_bytes(contract_data)
    (output/(name+'.sha256')).write_text(
        f'{sha(wrapped)}  {model_file}\n{sha(contract_data)}  {contract_file}\n')
    return dict(model=str(output/model_file), contract=str(output/contract_file),
                model_sha256=sha(wrapped), parity=parity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-yaml', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, help='Defaults to the YAML sibling .onnx')
    parser.add_argument('--output', type=Path, required=True, help='New native package directory')
    parser.add_argument('--model', required=True, help='Native model name used by the launcher')
    args = parser.parse_args()
    result = import_policy(args.policy_yaml, args.onnx or args.policy_yaml.with_suffix('.onnx'),
                           args.output, args.model)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
