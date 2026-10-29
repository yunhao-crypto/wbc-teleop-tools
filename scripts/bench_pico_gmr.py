#!/usr/bin/env python3
"""Time the live GMR worker on the frozen Pico session; no robot or Pico needed.

Pushes recorded Pico frames through the deployed ReferenceStream (same GMR,
ik_max_iter and stream conditioning as live) and reports per-frame wall and
thread-CPU milliseconds next to the on-robot retarget_ms of the same frames.
CPU close to wall means compute-bound; CPU well below wall means preemption.

  python3 scripts/bench_pico_gmr.py --frames 1000 --cpus 4
"""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import SessionFixture
from deployment.vita_s100_aorta_30dof.actor_pico_worker import ReferenceStream

DEFAULT_FIXTURE = ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz'


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {k: round(float(v), 3) for k, v in (('mean', values.mean()), ('p50', np.median(values)),
            ('p95', np.percentile(values, 95)), ('p99', np.percentile(values, 99)), ('max', values.max()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--frames', type=int, default=1000, help='leading fixture frames to push (0 = all)')
    parser.add_argument('--cpus', type=int, nargs='+', help='pin this process to these CPUs first')
    parser.add_argument('--legacy', action='store_true', help='drop pico_stream (pinned ik_max_iter 10, no filters)')
    parser.add_argument('--output', type=Path, help='optional JSON report')
    args = parser.parse_args()
    if args.cpus:
        os.sched_setaffinity(0, args.cpus)
    fixture = SessionFixture(args.fixture)
    runtime = ActorPicoRuntime(ROOT, backend='ort', policy='supermimic', human_height=fixture.human_height_m)
    config = copy.deepcopy(runtime.config)
    if args.legacy:
        config.pop('pico_stream', None)
    stream = ReferenceStream(runtime.root, config, runtime.model.contract, runtime.math)
    count = len(fixture.frame_stamp_ns) if args.frames <= 0 else min(args.frames, len(fixture.frame_stamp_ns))
    wall, cpu = [], []
    for i in range(count):
        snapshot = stream.push(fixture.frame_body[i].astype(np.float64),
                               dict(timestamp_ns=int(fixture.frame_stamp_ns[i]),
                                    timestamp_monotonic=float(fixture.frame_received_monotonic[i])),
                               (1., 0., 0., 0.))
        wall.append(snapshot['retarget_ms'])
        cpu.append(snapshot['retarget_cpu_ms'])
    affinity = sorted(os.sched_getaffinity(0))
    report = dict(host=platform.node(), machine=platform.machine(), cpu_count=os.cpu_count(),
                  affinity=affinity if len(affinity) <= 16 else f'{len(affinity)} cpus', frames=count,
                  pico_stream=config.get('pico_stream') is not None,
                  ik_max_iter=None if config.get('pico_stream') is None else config['pico_stream']['ik_max_iter'],
                  wall_ms=summary(wall), cpu_ms=summary(cpu),
                  robot_session_retarget_ms=summary(fixture.frame_robot_retarget_ms[:count]))
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
