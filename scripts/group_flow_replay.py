"""Offline Pico/state -> observation -> policy -> conditioned targets, no SDK."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import time

import numpy as np


class NoRobotTransport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *_):
        blocked = ('aorta', 'locomotion_aorta',
                   'deployment.vita_s100_aorta_30dof.real_teleop',
                   'deployment.vita_s100_aorta_30dof.onboard_live_session')
        if any(fullname == n or fullname.startswith(n + '.') for n in blocked):
            raise RuntimeError('Offline replay forbids robot transport: ' + fullname)
        return None


class OriginalRunner:
    def __init__(self, path, motion):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        providers = [('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider'] if motion else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if motion and self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('Original CUDA oracle unavailable')
        self.motion = motion
        self.reset()

    def reset(self):
        self.cache = np.zeros((1, 2, 1, 32, 4, 64), np.float32)
        self.step_index = np.array([0], np.int64)

    def run(self, observation):
        feed = {'obs': np.asarray(observation, np.float32)}
        if self.motion:
            if self.step_index[0] >= 8128:
                self.reset()
            feed.update(past_key_values=self.cache, step_idx=self.step_index)
        result = self.session.run(None, feed)
        if self.motion:
            self.cache = result[1]
            self.step_index += 1
        return result[0].reshape(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--group', choices=(*tuple('ABCDE'), 'P1', 'P2', 'P3'), required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oracle-onnx', type=Path)
    p.add_argument('--compare', type=Path)
    p.add_argument('--period-ms', type=float, default=0, help='Optional no-action replay cadence')
    a = p.parse_args()
    sys.meta_path.insert(0, NoRobotTransport())
    sys.path.insert(0, str(a.root / 'src'))
    from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.motion_groups import load_motion_group
    from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState, VITA_CONTROLLER_JOINT_NAMES
    from deployment.vita_s100_aorta_30dof.command_guard import command_envelope_from_contract
    from deployment.vita_s100_aorta_30dof.command_conditioning import VitaCommandAdapter
    from deployment.vita_s100_aorta_30dof.policy_contract import VITA_ACTION_TO_COMMAND_SLOTS
    from deployment.common.model_contract import sha256_file
    selection = load_motion_group(a.root, a.group)
    manifest = json.loads(selection['manifest'].read_text())
    if a.oracle_onnx:
        if sha256_file(a.oracle_onnx) != selection['source_sha256']:
            raise ValueError('Original model identity mismatch')
        runtime = OnboardDualPolicyRuntime(contracts_path=selection['contracts'],
            skeleton_path=a.root/'artifacts/preprocessing_skeleton.npz',
            runtime_root=a.root, envelopes_path=selection['envelopes'],
            envelopes_sha256=selection['envelopes_sha256'],
            expected_model_hashes=manifest['source_hashes'],
            motion_runner=OriginalRunner(a.oracle_onnx, True),
            velocity_runner=OriginalRunner(a.root/'artifacts/model_velocity/original.onnx', False))
    else:
        runtime = OnboardDualPolicyRuntime.from_onboard_root(a.root, motion_group=a.group)
        runtime.warmup()
    data = np.load(a.input, allow_pickle=False)
    base = command_envelope_from_contract(json.loads((a.root/'artifacts/output_envelopes.interface.json').read_text()))
    reference = runtime.create_online_reference()
    envelopes = {mode: runtime.command_envelope(base, mode) for mode in ('velocity', 'motion')}
    adapter = VitaCommandAdapter(envelopes['velocity'])
    hold = np.zeros(42)
    hold[list(VITA_ACTION_TO_COMMAND_SLOTS)] = data['joint_pos'][0]
    adapter.arm(hold)
    obs, actions, targets, gains, modes, costs = [], [], [], [], [], []
    mode_before = None
    starts = []
    stage_timings = []
    deadline = time.monotonic()
    try:
        for i in range(len(data['joint_pos'])):
            if a.period_ms > 0:
                time.sleep(max(0, deadline - time.monotonic()))
            started = time.monotonic_ns()
            starts.append(started)
            deadline = started / 1e9 + a.period_ms / 1000
            reference.store_pico(data['body_poses'][i], sample_time=float(data['sample_times'][i]),
                                 current_time=float(data['wall_times'][i]), frame_index=i)
            state = VitaRobotState(VITA_CONTROLLER_JOINT_NAMES, data['joint_pos'][i], data['joint_vel'][i],
                                   data['root_quat_wxyz'][i], data['root_ang_vel_b'][i])
            mode = 'motion' if 90 <= i < 190 or 250 <= i < 350 or 410 <= i < 450 else 'velocity'
            if mode != mode_before:
                runtime.reset()
                if mode == 'motion':
                    reference.reset_root_anchor()
                mode_before = mode
            if mode == 'motion':
                step = runtime.step_motion(state, reference, reference_yaw_alignment_wxyz=(1, 0, 0, 0))
            else:
                step = runtime.step_velocity(state, (0, 0, 0))
            adapter.envelope = envelopes[mode]
            target, kp, kd, _ = adapter.condition(step, hold_q=hold, period_s=.02)
            costs.append((time.monotonic_ns()-started)/1e6)
            stage_timings.append(dict(runtime.last_timing_ms))
            modes.append(mode)
            obs.append(np.pad(runtime.last_observation.reshape(-1), (0, 3126-runtime.last_observation.size)))
            actions.append(np.pad(runtime.last_raw_action.reshape(-1), (0, 30-runtime.last_raw_action.size)))
            targets.append(target)
            gains.append(np.stack((kp, kd)))
    finally:
        runtime.close()
    arrays = dict(observations=np.stack(obs), actions=np.stack(actions), targets=np.stack(targets),
                  gains=np.stack(gains), modes=np.array(modes), compute_ms=np.asarray(costs))
    report = dict(group=a.group, source_sha256=selection['source_sha256'], frames=len(modes),
        motion_frames=modes.count('motion'), switches=sum(modes[i]!=modes[i-1] for i in range(1,len(modes))),
        action_sent=False, sdk_imported=False, input_sha256=sha256_file(a.input),
        scope='Synthetic Pico/state fixture, own previous actions and independent KV; computation only, no state transport or publication',
        full_compute_ms=dict(mean=float(np.mean(costs[20:])), p99=float(np.percentile(costs[20:],99)),
                             max=float(np.max(costs[20:]))), passed=True)
    report['per_mode'] = {mode: {
        'full_compute_ms': {stat: float(fn(np.asarray(costs)[np.asarray(modes)==mode]))
            for stat,fn in [('mean',np.mean),('p99',lambda x:np.percentile(x,99)),('max',np.max)]},
        'stage_mean_ms': {key: float(np.mean([row.get(key,0) for row,m in zip(stage_timings,modes) if m==mode]))
            for key in sorted({k for row in stage_timings for k in row})}} for mode in ('velocity','motion')}
    if a.period_ms > 0:
        intervals = np.diff(starts) / 1e6
        report['cadence'] = dict(requested_period_ms=a.period_ms,
            measured_hz=float(1000 / np.mean(intervals)),
            interval_p99_ms=float(np.percentile(intervals, 99)),
            interval_max_ms=float(np.max(intervals)),
            computation_over_period=int(np.sum(np.asarray(costs) > a.period_ms)))
    if a.compare:
        expected = np.load(a.compare, allow_pickle=False)
        errors = {name: float(np.max(np.abs(arrays[name]-expected[name])))
                  for name in ('observations', 'actions', 'targets', 'gains')}
        report['max_abs_error'] = errors
        report['passed'] = (np.array_equal(arrays['modes'], expected['modes']) and errors['gains'] == 0
            and all(np.isfinite(arrays[k]).all() and errors[k] <= 1e-4 for k in ('observations','actions','targets')))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output.with_suffix('.npz'), **arrays)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    a.output.with_suffix('.timings.json').write_text(json.dumps([
        dict(frame=i, mode=mode, compute_ms=cost, **timing)
        for i, (mode, cost, timing) in enumerate(zip(modes, costs, stage_timings))
    ], separators=(',', ':'))+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
