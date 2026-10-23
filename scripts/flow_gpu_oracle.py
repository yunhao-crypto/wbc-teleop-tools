#!/usr/bin/env python3
"""Original FP32 Velocity CPU/Motion CUDA + original GPU preprocessing oracle.

Drives only the inert OnboardShadowFlow; no robot transport or publishing API.
Both policy histories, KV, and previous actions advance independently from input
fixture generation. Robot state/Pico are exogenous prerecorded synthetic inputs.
"""
from __future__ import annotations
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from flow_common import check_fixture, digest, frame_arguments, transition_record


class _ForbidControl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *_):
        blocked = ("aorta", "locomotion_aorta", "deployment.vita_s100_aorta_30dof.aorta_adapter",
                   "deployment.vita_s100_aorta_30dof.real_teleop", "deployment.vita_s100_aorta_30dof.cli")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise RuntimeError("No-actuation oracle blocks import: " + fullname)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.meta_path.insert(0, _ForbidControl())
    from deployment.common.controller_input import VelocityLimits
    from deployment.common.model_contract import load_model_bundle
    from deployment.vita_s100_aorta_30dof.runtime import VitaDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.onboard_shadow import OnboardShadowFlow
    bundle = load_model_bundle(a.bundle)
    os.environ["HOLOMOTION_SMPL_MODEL_PATH"] = str(bundle.artifacts["smpl"].path)
    runtime = VitaDualPolicyRuntime.from_bundle(bundle, require_trained=True,
        providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"],
        require_cuda=True, velocity_providers=["CPUExecutionProvider"], velocity_require_cuda=False)
    options = runtime.motion_runner.session.get_provider_options()
    assert options["CUDAExecutionProvider"]["use_tf32"] == "0"
    flow = OnboardShadowFlow(runtime, limits=VelocityLimits(forward=.9, backward=.7, lateral=0, yaw=.8, deadzone=.15))
    d = np.load(a.input, allow_pickle=False)
    count = check_fixture(d)
    active = np.zeros(count, np.uint8)
    action_widths = np.zeros(count, np.int32)
    obs_widths = np.zeros(count, np.int32)
    actions = np.zeros((count, 30), np.float32)
    observations = np.zeros((count, 3126), np.float32)
    clipped = np.zeros((count, 30), np.float32)
    targets = np.zeros((count, 30), np.float32)
    kp = np.zeros((count, 30), np.float32)
    kd = np.zeros((count, 30), np.float32)
    steps = np.zeros(count, np.int64)
    modes = []
    records = []
    begin = time.monotonic()
    for i in range(count):
        step, info = flow.tick(**frame_arguments(d, i))
        modes.append(info["mode"])
        steps[i] = int(runtime.motion_runner.step_index[0])
        if step is not None:
            active[i] = 1
            raw = np.asarray(runtime.last_raw_action).reshape(-1)
            observation = np.asarray(runtime.last_observation).reshape(-1)
            assert raw.size in (28, 30) and observation.size in (990, 3126)
            assert np.isfinite(raw).all() and np.isfinite(observation).all()
            action_widths[i], obs_widths[i] = raw.size, observation.size
            actions[i, :raw.size] = raw
            observations[i, :observation.size] = observation
            clipped[i, :step.raw_action.size] = step.raw_action
            targets[i], kp[i], kd[i] = step.target_joint_pos, step.joint_stiffness, step.joint_damping
        records.append({"frame":i, "mode":info["mode"], "reference_ready":info["reference_ready"],
            "pico_stored":info["pico_stored"], "transition":transition_record(info["transition"]),
            "action_width":int(action_widths[i]), "observation_width":int(obs_widths[i]),
            "motion_step_next":int(steps[i]), "action_sent":False})
        if (i + 1) % 64 == 0:
            print(json.dumps({"frames_completed":i+1,"mode":info["mode"]}), flush=True)
    assert any(r["transition"] and r["transition"]["event"] == "reference_lost" for r in records)
    assert modes[63] == "motion" and modes[96] == "velocity" and modes[460] == "estop"
    result = a.output / "expected.npz"
    np.savez(result, active=active, action_widths=action_widths, observation_widths=obs_widths,
        raw_actions=actions, observations=observations, clipped_actions=clipped,
        target_joint_pos=targets, stiffness=kp, damping=kd,
        motion_step_next=steps, modes=np.asarray(modes, dtype="<U16"))
    (a.output / "frames.json").write_text(json.dumps(records, indent=2)+"\n")
    source_files = [root / "src/deployment/vita_s100_aorta_30dof/runtime.py",
        root / "src/deployment/vita_s100_aorta_30dof/onboard_shadow.py",
        root / "src/deployment/common/policy_runtime.py",
        root / "src/deployment/vita_s100_aorta_30dof/teleop_engine.py",
        root / "src/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/smpl_reference.py",
        root / "src/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/smpl_motion_observation.py"]
    report = {"kind":"No-actuation full software-flow original FP32 oracle", "frames":count,
        "active_frames":int(active.sum()), "motion_frames":modes.count("motion"),
        "velocity_or_stand_frames":modes.count("velocity")+modes.count("stand"),
        "input":str(a.input),"input_sha256":digest(a.input),"expected_sha256":digest(result),
        "frames_json_sha256":digest(a.output/"frames.json"),
        "models":{key:{"sha256":bundle.artifacts[key].sha256,"provider":runtime.motion_runner.session.get_providers() if key=="motion" else runtime.velocity_runner.session.get_providers()} for key in ("motion","velocity")},
        "motion_use_tf32":False,"policy_hz":50,"reference_source":"Original unchanged GPU Pico/SMPL/observation builder + original full ONNX; new inert flow uses original software FSM",
        "state_semantics":"Own Velocity history/gait and both last_action buffers, own Motion KV/step, original mode-reset/anchor semantics. No reference actions/cache injected each step.",
        "input_kind":"Synthetic raw Pico, small plausible joint perturbations about metadata defaults; exogenous state, not simulated robot dynamics or recorded real tracking",
        "wall_seconds_including_oracle_work":time.monotonic()-begin,
        "source_files":[{"path":str(path.relative_to(root)),"sha256":digest(path)} for path in source_files],
        "action_sent":False,"robot_sdk_imported":False,
        "limitations":"START is inert software STAND semantics only; no physical stand handshake/trajectory. 480-frame switch test does not cover uninterrupted8128 reset; separate Motion8192 oracle covers that. GPU timing is not board performance."}
    (a.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
