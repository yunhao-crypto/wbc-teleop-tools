#!/usr/bin/env python3
"""Replay a frozen Pico session through the deployed worker and export policy windows.

For each variant (``legacy`` = window anchored at the newest sample, as
deployed at ca86f1c, with neither ``pico_stream`` nor ``pico_worker``;
``playback`` = the configured blocks) this writes, per 50 Hz control tick,
the reference window the live loop feeds the
SuperMimic policy, and prints how smoothly the current reference frame
advances. The output feeds trackerlab-supermimic's
scripts/rls/supermimic/replay_pico_session_mujoco.py closed-loop sim2sim.

  python scripts/replay_pico_session.py --output /tmp/robot42_windows.npz
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import (SessionFixture, advance_statistics,
                                                                        export_windows)

DEFAULT_FIXTURE = ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=('legacy', 'playback'), default=['legacy', 'playback'])
    parser.add_argument('--segments', nargs='+', type=int)
    parser.add_argument('--seconds', type=float, default=0., help='limit each following segment (0 = whole)')
    parser.add_argument('--stand', type=float, default=2., help='stand seconds exported before each segment')
    parser.add_argument('--gmr', default='robot', help="'robot' recorded GMR durations, or a constant in ms")
    args = parser.parse_args()
    fixture = SessionFixture(args.fixture)
    runtime = ActorPicoRuntime(ROOT, backend='auto', policy='supermimic', human_height=fixture.human_height_m)
    arrays, report = {}, dict(fixture=str(args.fixture), session=fixture.meta['session'], gmr=args.gmr,
                              pico_stream=runtime.config.get('pico_stream'),
                              pico_worker=runtime.config.get('pico_worker'), variants={})
    for variant in args.variants:
        config = copy.deepcopy(runtime.config)
        if variant == 'legacy':
            config.pop('pico_stream', None)
            config.pop('pico_worker', None)
        windows = export_windows(runtime, fixture, config=config, segments=args.segments, stand=args.stand,
                                 seconds=args.seconds, gmr=args.gmr if args.gmr == 'robot' else float(args.gmr))
        arrays.update({f'{variant}/{k}': v for k, v in windows.items()})
        stats = []
        for segment in np.unique(windows['segment']):
            mask = (windows['segment'] == segment) & windows['following'] & windows['valid']
            stats.append(dict(segment=int(segment), **advance_statistics(windows['current_source_s'][mask])))
        report['variants'][variant] = stats
        print(json.dumps(dict(variant=variant, advance=stats)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(dict(report, steps=runtime.reference.steps,
        current_index=runtime.reference.current_index, ref_time=runtime.reference.ref_time))), **arrays)


if __name__ == '__main__':
    main()
