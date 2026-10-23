"""Extract original router without loading the large expert initializer payloads."""
import argparse
from pathlib import Path
import onnx

def varint(f):
    value = shift = 0
    while True:
        b = f.read(1)
        if not b: return None
        value |= (b[0] & 127) << shift
        if b[0] < 128: return value
        shift += 7

def extract(source, include_gates=False):
    with open(source, 'rb') as f:
        while True:
            tag = varint(f); field, wire = tag >> 3, tag & 7
            if wire == 0: varint(f)
            elif wire == 2:
                size = varint(f)
                if field == 7:
                    end = f.tell() + size
                    break
                f.seek(size, 1)
            else: raise ValueError((field, wire))
        nodes, tensors = [], []
        while f.tell() < end:
            tag = varint(f); field, wire = tag >> 3, tag & 7
            assert wire == 2
            size = varint(f)
            if field == 1:
                n = onnx.NodeProto(); n.ParseFromString(f.read(size)); nodes.append(n)
            elif field == 5 and size < 10_000_000:
                t = onnx.TensorProto(); t.ParseFromString(f.read(size)); tensors.append(t)
            else: f.seek(size, 1)
    targets = ['moe_layer_0_expert_logits', 'moe_layer_0_expert_indices']
    if include_gates:
        targets.append('/actor_module/Unsqueeze_14_output_0')
    producer = {v: n for n in nodes for v in n.output}
    required = set(targets); pending = targets.copy(); kept = set()
    while pending:
        n = producer.get(pending.pop())
        if n is None or n.name in kept: continue
        kept.add(n.name)
        for v in n.input:
            if v not in required: required.add(v); pending.append(v)
    outputs = [onnx.helper.make_tensor_value_info(targets[0], onnx.TensorProto.FLOAT, [1, 1, 1024]),
               onnx.helper.make_tensor_value_info(targets[1], onnx.TensorProto.INT64, [1, 1, 2])]
    if include_gates:
        outputs.append(onnx.helper.make_tensor_value_info(targets[2], onnx.TensorProto.FLOAT, [1, 2, 1]))
    selected_nodes = []
    for node in nodes:
        if node.name not in kept:
            continue
        # HBDK 4.7.5's Clip converter dereferences omitted optional inputs.
        # Explicit +infinity is equivalent to the original omitted upper bound.
        if node.op_type == 'Clip' and len(node.input) == 3 and not node.input[2]:
            bound = node.name + '_explicit_infinite_max'
            selected_nodes.append(onnx.helper.make_node('Constant', [], [bound], name=bound,
                value=onnx.helper.make_tensor(bound, onnx.TensorProto.FLOAT, [], [float('inf')])))
            node.input[2] = bound
        selected_nodes.append(node)
    graph = onnx.helper.make_graph(selected_nodes, 'fp32_router',
        [onnx.helper.make_tensor_value_info('obs', onnx.TensorProto.FLOAT, [1, 604])],
        outputs,
        [t for t in tensors if t.name in required])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--include-gates', action='store_true')
    a = p.parse_args(); model = extract(a.source, include_gates=a.include_gates); onnx.save(model, a.output)
    print('Extracted', len(model.graph.node), 'nodes;', len(model.graph.initializer), 'original FP32 tensors')
