#!/usr/bin/env python3
"""OFFLINE SDK versus direct-exec fixed-NPZ comparison; never starts control."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_control_compare import model_layout, read_trial, compare_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sdk_log', type=Path)
    parser.add_argument('exec_log', type=Path)
    parser.add_argument('--onnx', required=True, type=Path, help='Exact wrapped MimicLite ONNX used in BOTH trials')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--warmup-s', type=float, default=2.)
    parser.add_argument('--max-initial-joint-deg', type=float, default=3.)
    parser.add_argument('--max-initial-root-deg', type=float, default=3.)
    parser.add_argument('--max-initial-velocity-rad-s', type=float, default=.5)
    args = parser.parse_args()
    try:
        layout = model_layout(args.onnx)
        a, b = read_trial(args.sdk_log, layout), read_trial(args.exec_log, layout)
        result = compare_trials(a, b, layout, args.warmup_s, args.max_initial_joint_deg,
                                args.max_initial_root_deg, args.max_initial_velocity_rad_s)
        code = 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = dict(schema='wtt.control_compare.v1',status='NOT_COMPARABLE',reason=str(exc),
                      robot_safety_acceptance=False)
        code = 2
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        if args.output.resolve() in (args.sdk_log.resolve(),args.exec_log.resolve(),args.onnx.resolve()):
            parser.error('Output must not overwrite an input')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
