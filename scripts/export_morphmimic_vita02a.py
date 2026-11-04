#!/usr/bin/env python3
"""Export a Vita02A motion checkpoint using its MorphMimic environment.

Run with MorphMimic's Python and --morph-repo. CPU only; never starts hardware.
Writes the ONNX and actor configuration in the deployment repository.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--morph-repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('artifacts/morphmimic/morphmimic_vita02a.onnx'))
    p.add_argument('--config-output', type=Path, default=Path('config/actor_pico_morphmimic.json'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.morph_repo.resolve()), str(args.morph_repo.resolve()/'src')]
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from deploy.export_onnx import _build_policy, DeployPolicy
    from morphmimic.constants import JOINT_NAMES
    from morphmimic.robots.vita02a import vita02a_spec, actuators, entity_cfg
    from morphmimic.robots.vita02a_model import make_spec
    from morphmimic.train.normalizers import RunningNormalizer
    torch.set_num_threads(1)
    checkpoint_bytes = args.checkpoint.read_bytes()
    c = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
    if (c['cfg']['robot'] != 'vita02a' or c.get('control_space') != 'motor_theta'
            or c['cfg'].get('goal_anchor', 'reference') != 'reference'
            or c['cfg'].get('policy_goal', 'motion') != 'motion'
            or c['cfg']['hist_len'] != 10 or c.get('morph') is not None):
        raise ValueError('Expected a single Vita02A motor_theta motion policy with reference goals')
    ids = [i for i in range(29) if i != 13]
    if list(c['canonical_joint_mask']) != [i != 13 for i in range(29)]:
        raise ValueError('Unexpected canonical joint mask')
    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.deploy = DeployPolicy(_build_policy(c), RunningNormalizer(1).eps)
            for name, value in (('om', c['obs_norm']['mean']), ('ov', c['obs_norm']['var']),
                                ('gm', c['goal_norm']['mean']), ('gv', c['goal_norm']['var'])):
                self.register_buffer(name, value)
        def forward(self, observation):
            action = self.deploy(observation[:, :930], observation[:, 930:], self.om, self.ov, self.gm, self.gv)
            return action[:, ids]
    wrapper = Export().eval()
    spec, model = vita02a_spec(), make_spec(visuals=False).compile()
    names = [JOINT_NAMES[i] for i in ids]
    native = [spec.joint_names[spec.translator.perm[i]] for i in ids]
    control = json.loads((root/'tests/fixtures/actor_pico/actor_g1_contract.json').read_text())
    control['joint_names'] = names
    control['robot_type'] = 'vita02a_closed_loop'
    defaults = entity_cfg().init_state.joint_pos
    control['default_joint_pos'] = [defaults[n] for n in native]
    for key in ('action_scale', 'joint_stiffness', 'joint_damping', 'joint_pos_limit_lower', 'joint_pos_limit_upper'):
        control[key] = []
    for n in native:
        cfg = next(a for a in actuators() if any(re.fullmatch(e, n) for e in a.target_names_expr))
        control['action_scale'].append(next(v for e, v in spec.action_scale.items() if re.fullmatch(e, n)))
        control['joint_stiffness'].append(cfg.stiffness)
        control['joint_damping'].append(cfg.damping)
        lo, hi = model.jnt_range[model.joint(n).id]
        control['joint_pos_limit_lower'].append(float(lo));control['joint_pos_limit_upper'].append(float(hi))
    control['ankle'] = {
        'schema_version': 3, 'control_space': 'motor_theta', 'robot_type': 'vita02a_closed_loop',
        'reference_conversion': 'virtual_pitch_roll_to_motor_theta',
        'geometry': dict(lx=.05, ly=.02175, z_top=.31, z_bot=.235),
        'motor_limits': [-.95975, .61075], 'roll_sign': dict(left=1, right=-1),
        'slots': {f'{side}_ankle_{axis}_joint': dict(joint=f'{side}_{label}_achilles_joint', motor=motor, side=side)
                  for side in ('left', 'right') for axis, label, motor in
                  (('pitch', 'A', 'theta_top'), ('roll', 'B', 'theta_bottom'))}}
    chain = []
    for name in ('waist_yaw_joint', 'waist_pitch_joint'):
        j = model.joint(name).id;b = model.jnt_bodyid[j]
        chain.append(dict(joint=name, pos=model.body_pos[b].tolist(), quat=model.body_quat[b].tolist(),
                          pivot=model.jnt_pos[j].tolist(), axis=model.jnt_axis[j].tolist()))
    metadata = dict(schema_version=1, policy_kind='morphmimic_vita02a_tracking', control=control,
        canonical_indices=ids, history_length=10, input_dim=1600, future_frames=10,
        goal_anchor='reference', velocity_rule='virtual_gradient_then_motor_jacobian', anchor_chain=chain,
        nominal_root_height=float(entity_cfg().init_state.pos[2]),
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        iteration=c['iteration'], training_commit=c.get('code_commit'),
        normalizers='baked_into_onnx', action_clip_note='Deployment uses control.action_clip; training decoder is unbounded')
    out = args.output if args.output.is_absolute() else root/args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1600)
    torch.onnx.export(wrapper, (x,), str(out), input_names=['observation'], output_names=['action'],
                      opset_version=17, dynamo=False)
    graph = onnx.load(str(out)); item = graph.metadata_props.add()
    item.key = 'morphmimic.tracking_contract.v1';item.value = json.dumps(metadata)
    onnx.checker.check_model(graph);onnx.save(graph, str(out))
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session = ort.InferenceSession(str(out), opts, providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(42);error=0.
    for _ in range(12):
        sample=rng.normal(size=(1,1600)).astype(np.float32)
        with torch.no_grad(): expected=wrapper(torch.from_numpy(sample)).numpy()
        actual=session.run(None, {'observation':sample})[0]
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
        error=max(error,float(np.max(np.abs(actual-expected))))
    config=json.loads((root/'config/actor_pico_supermimic.json').read_text())
    config['policy_kind']='morphmimic_vita02a_tracking'
    config['artifacts']['model']={'path':str(out.relative_to(root)), 'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                                  'bytes':out.stat().st_size,
                                  'source_name':args.checkpoint.name}
    config['body_profile']['ankle_command_space']='policy'
    config['body_profile']['ankle_clamp_feedforward']=False
    config['reference_window']={'past_frames':1,'tail_frames':1}
    config['provenance']={'exporter':'scripts/export_morphmimic_vita02a.py',**{k:metadata[k] for k in ('checkpoint_sha256','iteration','training_commit')}}
    config_out = args.config_output if args.config_output.is_absolute() else root/args.config_output
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(config,indent=2)+'\n')
    report=dict(onnx=str(out), max_abs_onnx_error=error, iteration=c['iteration'], samples=12)
    out.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
