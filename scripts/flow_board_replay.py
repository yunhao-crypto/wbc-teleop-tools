#!/usr/bin/env python3
"""Replay raw full-flow input against original GPU oracle on S100, inert only.

Root schedules target execution. No SDK/controller/actuator API is imported.
All errors and deadline misses are retained; failed precision never becomes pass.
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
from flow_common import check_fixture, digest, frame_arguments, stats, transition_record


class _ForbidControl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *_):
        if fullname == "aorta.sys" or fullname.startswith("aorta.sys."):
            return None  # FlatBuffers AortaHeader data definitions only.
        blocked = ("aorta", "locomotion_aorta", "deployment.vita_s100_aorta_30dof.aorta_adapter",
                   "deployment.vita_s100_aorta_30dof.real_teleop", "deployment.vita_s100_aorta_30dof.cli")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise RuntimeError("No-actuation replay blocks import: " + fullname)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--contracts", type=Path, required=True)
    p.add_argument("--skeleton", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preprocess-library", type=Path)
    p.add_argument("--velocity-backend", choices=("ort", "hbm"), default="ort")
    p.add_argument("--warmup-frames", type=int, default=90)
    p.add_argument("--timeout-ms", type=int, default=100)
    p.add_argument("--no-cadence", action="store_true", help="throughput diagnostic; default paces20ms")
    p.add_argument("--serial-router", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.meta_path.insert(0, _ForbidControl())
    if a.preprocess_library:
        os.environ["VITA_PREPROCESS_LIBRARY"] = str(a.preprocess_library.resolve())
    from deployment.common.controller_input import VelocityLimits
    from deployment.vita_s100_aorta_30dof.onboard_backend import NativeBackend, NativeMotionRunner, NativeVelocityRunner
    from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.onboard_shadow import OnboardShadowFlow
    with np.load(a.input, allow_pickle=False) as archive:
        count = check_fixture(archive)
        data = {key: archive[key] for key in archive.files}
    oracle_manifest = json.loads((a.oracle / "manifest.json").read_text())
    if digest(a.input) != oracle_manifest["input_sha256"]:
        raise ValueError("Fixture hash differs from oracle")
    if digest(a.oracle / "expected.npz") != oracle_manifest["expected_sha256"]:
        raise ValueError("Oracle tensor hash mismatch")
    if digest(a.oracle / "frames.json") != oracle_manifest["frames_json_sha256"]:
        raise ValueError("Oracle frame-metadata hash mismatch")
    with np.load(a.oracle / "expected.npz", allow_pickle=False) as archive:
        expected = {key: archive[key] for key in archive.files}
    expected_frames = json.loads((a.oracle / "frames.json").read_text())
    if len(expected["active"]) != count:
        raise ValueError("Oracle frame count mismatch")
    if not 0 <= a.warmup_frames <= count:
        raise ValueError("Invalid warmup frame count")
    limits = VelocityLimits(forward=.9, backward=.7, lateral=0, yaw=.8, deadzone=.15)
    actual_actions = np.zeros((count, 30), np.float32)
    actual_observations = np.zeros((count, 3126), np.float32)
    actual_targets = np.zeros((count, 30), np.float32)
    raw_cycle_ms = np.zeros(count, np.float64)
    phase_lateness_ms = np.zeros(count, np.float64)
    starts_ns = np.zeros(count, np.int64)
    schedule_slip_ms = np.zeros(count, np.float64)
    checks_ms = np.zeros(count, np.float64)
    records = []
    mode_errors = []
    action_failures = []
    worst = {"error":0.0,"frame":None,"coordinate":None,"mode":None}
    max_obs_error = 0.0
    max_target_error = 0.0
    switch_durations = []
    native = NativeBackend(a.library, a.manifest, velocity_backend=a.velocity_backend,
        parallel_router=not a.serial_router, timeout_ms=a.timeout_ms, deadline_ms=20)
    try:
        runtime = OnboardDualPolicyRuntime(contracts_path=a.contracts, skeleton_path=a.skeleton,
            motion_runner=NativeMotionRunner(native), velocity_runner=NativeVelocityRunner(native))
        warm_flow = OnboardShadowFlow(runtime, limits=limits)
        for i in range(a.warmup_frames):
            warm_flow.tick(**frame_arguments(data, i))
        # Models/workers remain resident; all externally-owned sequence state resets.
        runtime.reset()
        flow = OnboardShadowFlow(runtime, limits=limits)
        del warm_flow
        period_ns = 20_000_000
        next_tick = time.monotonic_ns()
        run_begin = next_tick
        for i in range(count):
            if not a.no_cadence:
                wait_ns = next_tick - time.monotonic_ns()
                if wait_ns > 0:
                    time.sleep(wait_ns / 1e9)
            begun = time.monotonic_ns()
            starts_ns[i] = begun
            schedule_slip_ms[i] = max(0, begun - (run_begin + i * period_ns)) / 1e6
            phase_lateness_ms[i] = max(0, begun - next_tick) / 1e6 if not a.no_cadence else 0
            arguments = frame_arguments(data, i)
            step, info = flow.tick(**arguments)
            ended = time.monotonic_ns()
            raw_cycle_ms[i] = (ended - begun) / 1e6
            # Validation and bulk trace staging are excluded from pipeline latency.
            # They still consume wall time and affect the next measured start lateness.
            check_begin = time.monotonic_ns()
            transition = transition_record(info["transition"])
            expected_record = expected_frames[i]
            native_state = native.state()
            disagreements = []
            for key in ("mode", "reference_ready", "pico_stored"):
                if info[key] != expected_record[key]:
                    disagreements.append(key)
            if transition != expected_record["transition"]:
                disagreements.append("transition")
            if bool(step is not None) != bool(expected["active"][i]):
                disagreements.append("action_present")
            if native_state["motion_step"] != int(expected["motion_step_next"][i]):
                disagreements.append("motion_step")
            frame_error = None
            if step is not None:
                raw = np.asarray(runtime.last_raw_action).reshape(-1)
                observation = np.asarray(runtime.last_observation).reshape(-1)
                width = int(expected["action_widths"][i])
                obs_width = int(expected["observation_widths"][i])
                if raw.size != width or observation.size != obs_width:
                    disagreements.append("action_or_observation_shape")
                else:
                    if not np.isfinite(raw).all() or not np.isfinite(observation).all():
                        raise RuntimeError(f"Nonfinite candidate output at frame{i}")
                    actual_actions[i, :width] = raw
                    actual_observations[i, :obs_width] = observation
                    delta = np.abs(raw.astype(np.float64) - expected["raw_actions"][i, :width].astype(np.float64))
                    frame_error = float(delta.max())
                    if frame_error > worst["error"]:
                        worst = {"error":frame_error,"frame":i,"coordinate":int(delta.argmax()),"mode":info["mode"]}
                    bad = np.flatnonzero(delta > 1e-4)
                    if len(bad):
                        action_failures.append({"frame":i,"mode":info["mode"],"coordinates":bad.tolist(),
                            "max_abs":frame_error,"worst_coordinate":int(delta.argmax())})
                    max_obs_error = max(max_obs_error, float(np.max(np.abs(observation.astype(np.float64) - expected["observations"][i, :obs_width].astype(np.float64)))))
                    max_target_error = max(max_target_error, float(np.max(np.abs(step.target_joint_pos.astype(np.float64) - expected["target_joint_pos"][i].astype(np.float64)))))
                    actual_targets[i] = step.target_joint_pos
                    if not np.array_equal(step.joint_stiffness, expected["stiffness"][i]) or not np.array_equal(step.joint_damping, expected["damping"][i]):
                        disagreements.append("gains")
            if disagreements:
                mode_errors.append({"frame":i,"fields":disagreements})
            if transition and transition["changed"]:
                switch_durations.append(raw_cycle_ms[i])
            records.append({"frame":i,"mode":info["mode"],"reference_ready":info["reference_ready"],
                "transition":transition,"timing_ms":info["timing_ms"],"full_replay_cycle_ms":float(raw_cycle_ms[i]),
                "start_lateness_ms":float(phase_lateness_ms[i]),"action_max_abs_error":frame_error,
                "motion_step_next":native_state["motion_step"],"deadline_missed":bool(raw_cycle_ms[i]>20),"action_sent":False})
            checks_ms[i] = (time.monotonic_ns() - check_begin) / 1e6
            if not a.no_cadence:
                # No burst of old ticks: late work moves the next schedule forward.
                next_tick = max(next_tick + period_ns, time.monotonic_ns())
        wall_seconds = (time.monotonic_ns() - run_begin) / 1e9
        final_state = native.state()
        build_info = native.build_info
    finally:
        native.close()
    np.savez(a.output / "candidate.npz", raw_actions=actual_actions, observations=actual_observations,
        target_joint_pos=actual_targets, full_cycle_ms=raw_cycle_ms, start_lateness_ms=phase_lateness_ms,
        validation_staging_ms=checks_ms, start_monotonic_ns=starts_ns,
        cumulative_schedule_slip_ms=schedule_slip_ms)
    (a.output / "frames.json").write_text(json.dumps(records, indent=2)+"\n")
    mode_masks = {name:np.asarray([r["mode"] == name for r in records]) for name in ("stand", "velocity", "motion", "idle", "estop")}
    report = {"kind":"Actual-board raw-input dual-policy shadow replay vs original FP32 GPU/CPU oracle",
        "frames":count,"active_frames":int(expected["active"].sum()),"warmup_frames":a.warmup_frames,
        "passed":not mode_errors and not action_failures and max_target_error<=1e-4,
        "criterion":"Every active step and every raw-action coordinate absolute error <=1e-4; exact mode/reference/reset/gain semantics",
        "worst_action_error":worst,"action_bad_steps":len(action_failures),
        "action_bad_coordinates":sum(len(item["coordinates"]) for item in action_failures),
        "action_failures":action_failures,"state_or_mode_errors":mode_errors,
        "max_observation_absolute_error_diagnostic":max_obs_error,"max_mapped_target_absolute_error":max_target_error,
        "full_cycle_ms":stats(raw_cycle_ms),"active_cycle_ms":stats(raw_cycle_ms[expected["active"].astype(bool)]),
        "by_mode_ms":{name:stats(raw_cycle_ms[mask]) for name,mask in mode_masks.items()},
        "switch_cycle_ms":stats(switch_durations),"cycles_over_20ms":int((raw_cycle_ms>20).sum()),
        "max_start_lateness_ms":float(phase_lateness_ms.max()),"validation_staging_ms":stats(checks_ms),
        "elapsed_wall_seconds":wall_seconds,"requested_pacing_hz":None if a.no_cadence else 50,
        "actual_start_rate_hz":float((count-1)*1e9/(starts_ns[-1]-starts_ns[0])),
        "interarrival_ms":stats(np.diff(starts_ns)/1e6),
        "interarrival_over_20ms":int((np.diff(starts_ns)>period_ns).sum()),
        "interarrival_over_21ms":int((np.diff(starts_ns)>period_ns+1_000_000).sum()),
        "max_cumulative_schedule_slip_ms":float(schedule_slip_ms.max()),"velocity_backend":a.velocity_backend,
        "parallel_router":not a.serial_router,"native_final_state":final_state,"native_build_info":build_info,
        "fixture_sha256":digest(a.input),"oracle_manifest_sha256":digest(a.oracle/"manifest.json"),
        "backend_library_sha256":digest(a.library),"runtime_manifest_sha256":digest(a.manifest),
        "preprocess_library_sha256":digest(a.preprocess_library) if a.preprocess_library else None,
        "preprocess_native_stages":os.environ.get("VITA_PREPROCESS_NATIVE_STAGES", "pico,features,gather,build"),
        "source_files":[{"path":str(path.relative_to(root)),"sha256":digest(path)} for path in [
            root/"src/deployment/vita_s100_aorta_30dof/onboard_runtime.py",
            root/"src/deployment/vita_s100_aorta_30dof/onboard_shadow.py",
            root/"src/deployment/vita_s100_aorta_30dof/onboard_preprocessing.py"]],
        "timing_boundary":"Includes state/controller reconstruction and complete Pico/reference/FSM/observations/native inference/feedback/mapping; excludes SDK transport (synthetic exogenous inputs), startup and postcycle numerical validation/file staging. Validation time remains in wall cadence.",
        "state_semantics":"Candidate owns KV, last_action, gait/history and mode resets; no teacher-forced expected observations/actions/cache injected.",
        "input_kind":oracle_manifest["input_kind"],"action_sent":False,"robot_sdk_imported":False,
        "limitations":"Software START/STAND only. Not physical robot dynamics or live Pico/LowState transport; switch replay does not cover8128 without separate continuous Motion test."}
    (a.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key not in ("action_failures", "source_files")}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
