"""Replay captured control payloads through a recorder; no SDK or robot action."""

import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import time

import numpy as np


class NoTransport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *_):
        if fullname.split(".")[0] in ("aorta", "locomotion_aorta"):
            raise RuntimeError("Recorder profile forbids robot transport")


def stats(values):
    a = np.asarray(values)
    return dict(
        n=len(a),
        mean=float(a.mean()),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        max=float(a.max()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("sync", "process"), required=True)
    p.add_argument("--steps", type=int, default=500)
    a = p.parse_args()
    if a.steps < 1:
        raise ValueError("Positive step count required")
    sys.meta_path.insert(0, NoTransport())
    sys.path.insert(0, str(a.root / "src"))
    from deployment.vita_s100_aorta_30dof.recording import JsonlSessionRecorder
    from deployment.vita_s100_aorta_30dof.async_recording import (
        ProcessJsonlSessionRecorder,
    )

    samples = []
    with a.control.open() as source:
        for line in source:
            item = json.loads(line)
            if (
                item.get("record_type") != "control"
                or item.get("mode") != "motion"
            ):
                continue
            # Production uses arrays for these entries, before JSON conversion.
            for key in (
                "policy_observation",
                "raw_action",
                "onnx_raw_action",
                "q_policy",
                "dq_policy",
                "target_policy",
                "target_sdk",
                "kp_sdk",
                "kd_sdk",
                "root_quat_wxyz",
                "root_ang_vel_b",
                "velocity_command",
            ):
                if isinstance(item.get(key), list):
                    item[key] = np.asarray(item[key], dtype=np.float32)
            samples.append(item)
            if len(samples) == 64:
                break
    if not samples:
        raise ValueError("No Motion control samples")
    a.output.mkdir(parents=True, exist_ok=False)
    recorder_type = (
        JsonlSessionRecorder
        if a.mode == "sync"
        else ProcessJsonlSessionRecorder
    )
    times, cpu, lateness = [], [], []
    with recorder_type(a.output, prefix="profile") as recorder:
        start = time.monotonic_ns()
        for i in range(a.steps):
            due = start + i * 20_000_000
            time.sleep(max(0, due - time.monotonic_ns()) / 1e9)
            before, cpu_start = time.monotonic_ns(), time.thread_time_ns()
            lateness.append(max(0, before - due) / 1e6)
            record = dict(samples[i % len(samples)], profile_sequence=i)
            recorder.write(record)
            elapsed = (time.monotonic_ns() - before) / 1e6
            times.append(elapsed)
            cpu.append((time.thread_time_ns() - cpu_start) / 1e6)
            recorder.write(
                dict(
                    record_type="cycle_timing",
                    profile_sequence=i,
                    recorder_enqueue_or_write_ms=elapsed,
                )
            )
    count = 0
    with recorder.path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["profile_sequence"] != count // 2:
                raise ValueError("Recorder sequence mismatch")
            count += 1
    if count != a.steps * 2:
        raise ValueError("Recorder lost records")
    report = dict(
        mode=a.mode,
        steps=a.steps,
        records=count,
        action_sent=False,
        control_thread_wall_ms=stats(times),
        control_thread_cpu_ms=stats(cpu),
        schedule_lateness_ms=stats(lateness),
        path=str(recorder.path),
        scope="Recorder only; includes snapshot/enqueue or serialization/write, excludes inference and SDK",
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
