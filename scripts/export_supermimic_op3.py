#!/usr/bin/env python3
"""Export an official SuperMimic full OP3 checkpoint for WTT (CPU, no robot).

Run using SuperMimic's Python plus onnxruntime. Models remain external delivery
resources; config and metadata/validation sidecars are versioned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--supermimic-repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='committed checkpoint directory')
    parser.add_argument('--output', type=Path, default=Path('artifacts/supermimic_op3/actor.onnx'))
    parser.add_argument('--config-output', type=Path, default=Path('config/actor_pico_supermimic_op3.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo = args.supermimic_repo.resolve()
    sys.path[:0] = [str(root/'src'), str(repo/'src')]
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from supermimic.models import MultiEncoderAE
    from supermimic.tasks.task_spec import TaskSpec
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.exporter import _SingleEncoderOnnx
    from supermimic.assets.vita02a_op3.robot import make_cfg, make_spec, model_info
    from supermimic.tasks.vita02a_op3 import vita02a_op3_env_cfg, POLICY_JOINTS
    from deployment.vita_s100_aorta_30dof.actor_supermimic_op3 import (
        OP3Contract, CONTRACT_KEY, POLICY_KIND, TERM_WIDTHS, OFFSETS, GEOMETRY, native_name)
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()/'rank_0000.pt'
    manifest = json.loads((checkpoint.parent/'manifest.json').read_text())
    if manifest['files'][checkpoint.name] != checkpoint.stat().st_size:
        raise ValueError('checkpoint shard size differs from committed manifest')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    sig = state['signature']
    cfg = load_runner_cfg({'runner_cfg': sig['runner']})
    if (cfg.representation != 'gencoord' or cfg.actor_obs != 'proprio' or cfg.history_length != 10
            or tuple(sig['joint_names']) != model_info()[0]
            or state['next_iteration'] != manifest['next_iteration']):
        raise ValueError('expected an official GenCoord/proprio OP3 full checkpoint')
    actor = MultiEncoderAE(90, 28, [TaskSpec('motion', 860, lambda *_: None),
                                   TaskSpec('velocity', 3, lambda *_: None)], cfg.slot_mix,
                           cfg.latent_dim, cfg.actor_hidden_dims, cfg.latent_bc_weight, cfg.history_length).eval()
    actor.load_state_dict(state['actor'], strict=True)
    wrapper = _SingleEncoderOnnx(actor, 'motion', list(TERM_WIDTHS)).eval()
    env_cfg, robot_cfg = vita02a_op3_env_cfg(), make_cfg()
    reference = list(model_info()[0])
    native = [n for n in reference if any(re.fullmatch(e, n) for e in POLICY_JOINTS)]
    names = [n.replace('_A_achilles_joint', '_ankle_pitch_joint').replace('_B_achilles_joint', '_ankle_roll_joint')
             for n in native]
    assert list(map(native_name, names)) == native and len(native) == 28
    model = make_spec(visuals=False).compile()
    raw = dict(schema_version=1, policy_kind=POLICY_KIND, robot_type='vita02a_op3',
        input_dim=1760, action_dim=28, history_length=10, policy_dt=.02,
        observation_terms=['gravity', 'gyro', 'motor_pos_rel', 'motor_vel', 'last_action'],
        history_order='term_major_oldest_first', reset='repeat_first_frame',
        goal_layout='per_frame_q40_qd40_then_relative_orientation_6d', frame_offsets=list(OFFSETS),
        rotation_6d='first_two_columns_row_major', orientation_frame='actual_robot_root',
        beyond_reference_end='clamp_last_frame_with_velocity',
        streaming_velocity='virtual_gradient_then_linkage_central_difference', action_clip=None,
        command_target_clipping='none', required_ankle_command_space='motor_theta', normalizers='baked_into_onnx',
        joint_names=names, native_joint_names=native, reference_joint_names=reference,
        default_joint_pos=[robot_cfg.init_state.joint_pos[n] for n in native],
        checkpoint_sha256=digest(checkpoint), iteration=state['next_iteration'],
        training_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        training_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)),
        checkpoint_model_sha256=sig['model_sha256'])
    for key in ('action_scale','joint_stiffness','joint_damping','joint_pos_limit_lower','joint_pos_limit_upper'):
        raw[key] = []
    for n in native:
        ac = next(a for a in robot_cfg.articulation.actuators if any(re.fullmatch(e,n) for e in a.target_names_expr))
        raw['action_scale'].append(next(v for e,v in env_cfg.actions['joint_pos'].scale.items() if re.fullmatch(e,n)))
        raw['joint_stiffness'].append(ac.stiffness); raw['joint_damping'].append(ac.damping)
        lo, hi = model.joint(n).range
        if '_achilles_joint' in n:
            lo, hi = -.95975, .61075
        raw['joint_pos_limit_lower'].append(float(lo)); raw['joint_pos_limit_upper'].append(float(hi))
    raw['ankle'] = dict(schema_version=3, control_space='motor_theta',
        reference_conversion='virtual_pitch_roll_to_motor_theta', geometry=GEOMETRY,
        motor_limits=[-.95975,.61075], roll_sign=dict(left=1,right=-1),
        slots={f'{s}_ankle_{a}_joint':dict(side=s,motor=m) for s in ('left','right')
               for a,m in (('pitch','theta_top'),('roll','theta_bottom'))})
    OP3Contract.from_metadata({CONTRACT_KEY:json.dumps(raw)})
    out = (root/args.output).resolve()
    if not out.is_relative_to(root):
        raise ValueError('output must be inside the deployment repository')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (torch.zeros(1,1760),), str(out), input_names=['observation'],
                      output_names=['action'], opset_version=17, dynamo=False)
    graph = onnx.load(out); item = graph.metadata_props.add()
    item.key, item.value = CONTRACT_KEY, json.dumps(raw)
    onnx.checker.check_model(graph); onnx.save(graph,out)
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(out),opts,providers=['CPUExecutionProvider'])
    error=0.
    for seed in range(12):
        x=np.random.default_rng(seed).normal(0,.1,(1,1760)).astype(np.float32)
        with torch.inference_mode(): expected=wrapper(torch.from_numpy(x)).numpy()
        actual=session.run(None,{'observation':x})[0]
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        error=max(error,float(np.max(abs(actual-expected))))
    config=json.loads((root/'config/actor_pico_supermimic.json').read_text())
    config['policy_kind']=POLICY_KIND
    config['artifacts']['model']=dict(path=str(out.relative_to(root)),sha256=digest(out),bytes=out.stat().st_size,
                                     source_name=f'official_op3_iter{state["next_iteration"]}')
    config['body_profile']['ankle_command_space']='motor_theta'
    config['body_profile']['ankle_clamp_feedforward']=False
    config['reference_window']=dict(past_frames=1,tail_frames=1)
    config['provenance']={k:raw[k] for k in ('checkpoint_sha256','iteration','training_commit','training_worktree_dirty','checkpoint_model_sha256')}
    config['provenance']['exporter']='scripts/export_supermimic_op3.py'
    config['reference_delay_s']=.92
    (root/args.config_output).write_text(json.dumps(config,indent=2)+'\n')
    out.with_suffix('.contract.json').write_text(json.dumps(raw,indent=2)+'\n')
    report=dict(iteration=state['next_iteration'],samples=12,max_abs_onnx_error=error,
                onnx_sha256=digest(out),hardware_tested=False)
    out.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
