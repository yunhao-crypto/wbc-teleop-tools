"""Paced no-action replay with state, Pico preprocessing, own KV and complete logs.

Input is a recorded no-action capture (including its declared Pico provenance).
No SDK controller, publisher or Node can be created. The byte sink is local only.
This measures compute capacity, never source-to-bottom delivery latency.
"""

import argparse
import gc
import importlib.abc
import importlib.machinery
import json
import os
import re
from pathlib import Path
import struct
import sys
import time
import types

import numpy as np


class NoNetwork(importlib.abc.MetaPathFinder):
    def __init__(self, readonly=False):
        self.readonly = readonly

    def find_spec(self, fullname, *_):
        if fullname == "aorta" or fullname.startswith(
            (
                "aorta.",
                "locomotion_aorta",
                "deployment.vita_s100_aorta_30dof.onboard_live_session",
            )
        ):
            if fullname.startswith("aorta.sys") or (
                self.readonly and fullname.startswith("aorta")
            ):
                return None
            raise RuntimeError("Replay forbids live transport: " + fullname)


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(
        n=len(x),
        mean=float(x.mean()),
        p95=float(np.percentile(x, 95)),
        p99=float(np.percentile(x, 99)),
        max=float(x.max()),
        ge20=int((x >= 20).sum()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--capture-prefix", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument(
        "--live-state",
        action="store_true",
        help="Subscribe to live state with every control factory blocked",
    )
    p.add_argument("--group", default="C", choices=tuple("ABCDE"))
    a = p.parse_args()
    if not 1 <= a.seconds <= 600:
        raise ValueError("Bounded 1..600 s replay")
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.root / "src"))
    # Load only generated data submodules, bypassing aorta/__init__.py.
    if a.live_state:
        import aorta

        def forbidden(*args, **kwargs):
            raise RuntimeError("Read-only replay forbids control factories")

        blocked = []
        for name in dir(aorta.Node):
            if name.startswith("create_") and any(
                k in name for k in ("publisher", "client", "action", "service")
            ):
                setattr(aorta.Node, name, forbidden)
                blocked.append(name)
        if not blocked:
            raise RuntimeError("Could not guard control factories")
    else:
        spec = importlib.machinery.PathFinder.find_spec("aorta", sys.path)
        ns = types.ModuleType("aorta")
        ns.__path__ = list(spec.submodule_search_locations)
        sys.modules["aorta"] = ns
    sys.meta_path.insert(0, NoNetwork(readonly=a.live_state))
    from deployment.vita_s100_aorta_30dof.onboard_backend import (
        NativeBackend,
        NativeMotionRunner,
        NativeVelocityRunner,
    )
    from deployment.vita_s100_aorta_30dof.onboard_runtime import (
        OnboardDualPolicyRuntime,
    )
    from deployment.vita_s100_aorta_30dof.motion_groups import (
        load_motion_group,
    )
    from deployment.vita_s100_aorta_30dof.onboard_live_io import (
        decode_live_state,
        live_robot_state,
        OnboardLiveStateSource,
    )
    from deployment.vita_s100_aorta_30dof.live_shadow import (
        live_alignment_quaternion,
    )
    from deployment.vita_s100_aorta_30dof.command_guard import (
        command_envelope_from_contract,
    )
    from deployment.vita_s100_aorta_30dof.command_conditioning import (
        VitaCommandAdapter,
    )
    from deployment.vita_s100_aorta_30dof.async_recording import (
        ProcessJsonlSessionRecorder,
    )
    from deployment.vita_s100_aorta_30dof.onboard_command_encoding import (
        CreateMotorCmd,
    )
    from deployment.common.model_contract import sha256_file
    import flatbuffers
    import lowlevel.LowCmd as low
    from aorta.sys import AortaHeader as header

    alignment = live_alignment_quaternion(
        json.loads(
            (
                a.root
                / "src/deployment/vita_s100_aorta_30dof/config/vita02a_imu_alignment.json"
            ).read_text()
        )
    )
    rows = []
    with a.capture_prefix.with_suffix(".jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            rows.append({"pico": row.get("pico")})
    raw = a.capture_prefix.with_suffix(".lowstate.bin").read_bytes()
    payloads = []
    offset = 0
    while offset < len(raw):
        n = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        payloads.append(raw[offset : offset + n])
        offset += n
    if offset != len(raw) or len(payloads) != len(rows):
        raise ValueError("Capture framing mismatch")
    indices = [i for i, r in enumerate(rows) if r["pico"] is not None]
    if not indices:
        raise ValueError("No captured Pico samples")
    picos = {i: np.asarray(rows[i]["pico"][0], np.float32) for i in indices}
    with a.control.open() as f:
        template = next(
            json.loads(line) for line in f if '"mode": "motion"' in line
        )
    selection = load_motion_group(a.root, a.group)
    manifest = json.loads(selection["manifest"].read_text())
    backend = NativeBackend(a.library, selection["manifest"])
    runtime = OnboardDualPolicyRuntime(
        contracts_path=selection["contracts"],
        skeleton_path=a.root / "artifacts/preprocessing_skeleton.npz",
        runtime_root=a.root,
        envelopes_path=selection["envelopes"],
        envelopes_sha256=selection["envelopes_sha256"],
        expected_model_hashes=manifest["source_hashes"],
        motion_runner=NativeMotionRunner(backend),
        velocity_runner=NativeVelocityRunner(backend),
    )
    runtime.warmup()
    reference = runtime.create_online_reference()
    for i in range(reference.capacity):
        reference.store_pico(
            picos[indices[i % len(indices)]],
            sample_time=i * 0.02,
            current_time=i * 0.02,
            frame_index=i,
        )
    source = (
        OnboardLiveStateSource(group="default", alignment=alignment)
        if a.live_state
        else None
    )
    initial = (
        source.read(10.0)
        if source
        else decode_live_state(
            payloads[indices[0]],
            alignment=alignment,
            received_monotonic_ns=time.monotonic_ns(),
        )
    )
    hold = initial.q_by_slot[:42].copy()
    env = runtime.command_envelope(
        command_envelope_from_contract(
            json.loads(
                (
                    a.root / "artifacts/output_envelopes.interface.json"
                ).read_text()
            )
        ),
        "motion",
    )
    adapter = VitaCommandAdapter(env)
    adapter.arm(hold)
    costs = []
    starts = []
    skipped = 0
    finite = True
    byte_count = 0
    gc_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        with ProcessJsonlSessionRecorder(
            a.output, prefix="replay"
        ) as recorder:
            start = time.monotonic_ns()
            deadline = start
            for i in range(a.seconds * 50):
                time.sleep(max(0, deadline - time.monotonic_ns()) / 1e9)
                begin = time.monotonic_ns()
                starts.append(begin)
                k = indices[i % len(indices)]
                state_raw = (
                    source.read(0.1)
                    if source
                    else decode_live_state(
                        payloads[k],
                        alignment=alignment,
                        received_monotonic_ns=begin,
                    )
                )
                state = live_robot_state(
                    state_raw, base_from_hip_quaternion_wxyz=alignment
                )
                decoded = time.monotonic_ns()
                stamp = (reference.capacity + i) * 0.02
                reference.store_pico(
                    picos[k],
                    sample_time=stamp,
                    current_time=stamp,
                    frame_index=i,
                )
                pico_done = time.monotonic_ns()
                step = runtime.step_motion(
                    state, reference, reference_yaw_alignment_wxyz=(1, 0, 0, 0)
                )
                inferred = time.monotonic_ns()
                target, kp, kd, conditioning = adapter.condition(
                    step, hold_q=hold, period_s=0.02
                )
                conditioned = time.monotonic_ns()
                # Exact generated struct/table build into bytes; never publish.
                builder = flatbuffers.Builder(2048)
                header.Start(builder)
                hoff = header.End(builder)
                low.LowCmdStartMotorCmdVector(builder, 42)
                for j in reversed(range(42)):
                    CreateMotorCmd(
                        builder,
                        1,
                        float(target[j]),
                        0.0,
                        0.0,
                        float(kp[j]),
                        float(kd[j]),
                    )
                vector = builder.EndVector()
                low.LowCmdStart(builder)
                low.LowCmdAddAortaHeader(builder, hoff)
                low.LowCmdAddMotorCmd(builder, vector)
                low.LowCmdAddCmdId(builder, i % 65535 + 1)
                low.LowCmdAddExternalVersion(builder, 1)
                low.LowCmdAddExternalSessionId(builder, 1)
                low.LowCmdAddExternalSequence(builder, i + 1)
                builder.Finish(low.LowCmdEnd(builder))
                byte_count += len(builder.Output())
                encoded = time.monotonic_ns()
                encoded_wall = time.time_ns()
                record = dict(
                    command_id=None,
                    profile_sequence=i,
                    action_sent=False,
                    controller=template.get("controller"),
                    velocity_command=(0, 0, 0),
                    state_id=state_raw.state_id,
                    state_real_us=state_raw.ts_state_real_us,
                    state_pub_us=state_raw.ts_state_pub_us,
                    state_received_monotonic_ns=state_raw.received_monotonic_ns,
                    cycle_started_monotonic_ns=begin,
                    record_type="offline_control",
                    sdk_publish_started_monotonic_ns=None,
                    published_monotonic_ns=None,
                    publish_period_s=None,
                    sdk_status_before_publish=None,
                    sdk_reject_count=None,
                    timing_wall_ns={"local_encode_done_wall_ns": encoded_wall},
                    mode="motion",
                    raw_action=step.raw_action,
                    onnx_raw_action=runtime.last_raw_action,
                    policy_observation=runtime.last_observation,
                    q_policy=state.joint_pos,
                    dq_policy=state.joint_vel,
                    root_quat_wxyz=state.root_quat_wxyz,
                    root_ang_vel_b=state.root_ang_vel_b,
                    target_policy=step.target_joint_pos,
                    target_sdk=target,
                    kp_sdk=kp,
                    kd_sdk=kd,
                    conditioning=conditioning,
                    timing_ms=runtime.last_timing_ms,
                )
                recorder.write(record)
                logged = time.monotonic_ns()
                row = dict(
                    sequence=i,
                    decode_ms=(decoded - begin) / 1e6,
                    pico_ms=(pico_done - decoded) / 1e6,
                    policy_ms=(inferred - pico_done) / 1e6,
                    condition_ms=(conditioned - inferred) / 1e6,
                    encode_ms=(encoded - conditioned) / 1e6,
                    record_enqueue_ms=(logged - encoded) / 1e6,
                    through_bytes_ms=(encoded - begin) / 1e6,
                    through_record_ms=(logged - begin) / 1e6,
                    start_lateness_ms=max(0, begin - deadline) / 1e6,
                    **runtime.last_timing_ms,
                )
                if source:
                    row.update(
                        source_to_bytes_ms=(
                            encoded_wall - state_raw.ts_state_real_us * 1000
                        )
                        / 1e6,
                        source_to_callback_ms=(
                            state_raw.received_wall_ns
                            - state_raw.ts_state_real_us * 1000
                        )
                        / 1e6,
                        core_receive_to_callback_ms=(
                            state_raw.received_wall_ns
                            - state_raw.native_receive_stamp_ns
                        )
                        / 1e6,
                        publication_to_core_receive_ms=(
                            state_raw.native_receive_stamp_ns
                            - state_raw.ts_state_pub_us * 1000
                        )
                        / 1e6,
                    )
                costs.append(row)
                recorder.write(
                    dict(record_type="cycle_timing", profile_sequence=i, **row)
                )
                finite = finite and bool(
                    np.isfinite(runtime.last_raw_action).all()
                )
                deadline += 20_000_000
                now = time.monotonic_ns()
                if now > deadline:
                    missed = (now - deadline) // 20_000_000 + 1
                    skipped += missed
                    deadline += missed * 20_000_000
            log_path = recorder.path
    finally:
        if source:
            source.close()
        state = backend.state()
        backend.close()
        if gc_enabled:
            gc.enable()
    count = 0
    with log_path.open("rb") as f:
        for line in f:
            match = re.search(rb'"profile_sequence": ([0-9]+)', line)
            if (
                not line.endswith(b"}\n")
                or match is None
                or int(match[1]) != count // 2
            ):
                raise ValueError("Log sequence gap or incomplete line")
            count += 1
    if count != a.seconds * 100:
        raise ValueError("Log count mismatch")
    report = dict(
        actions_sent=False,
        control_handles=0,
        group=a.group,
        finite=finite,
        records=count,
        frames=len(costs),
        seconds_requested=a.seconds,
        skipped_ticks=int(skipped),
        encoded_bytes=byte_count,
        log_validation="Worker fsync/close ACK plus complete-line and sequence scan; not a second full JSON decode",
        model_sha256=selection["source_sha256"],
        library_sha256=sha256_file(a.library),
        final_state=state,
        capture_prefix=str(a.capture_prefix),
        capture_sha256=sha256_file(a.capture_prefix.with_suffix(".jsonl")),
        live_readonly_state=bool(source),
        scope="Full computation with captured host-fake Pico; live read-only or recorded LowState as labelled; no command publisher or bottom delivery; replay seams retained",
        stages={
            key: stats([r[key] for r in costs])
            for key in costs[0]
            if key != "sequence"
        },
        command_interval_ms=stats(np.diff(starts) / 1e6),
        log_path=str(log_path),
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    (a.output / "timings.json").write_text(
        json.dumps(costs, separators=(",", ":"))
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
