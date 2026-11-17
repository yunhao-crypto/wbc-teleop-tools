#!/usr/bin/env python3
"""Read-only WTT JSONL audit; no SDK or robot connection."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deployment.vita_s100_aorta_30dof.actor_control_audit import audit_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--onnx', type=Path, help='wrapped MimicLite model for CPU numerical replay')
    args = parser.parse_args(argv)
    inputs = [args.log] + ([args.onnx] if args.onnx else [])
    if args.output and any(args.output.resolve() == p.resolve() or
                           (args.output.exists() and p.exists() and args.output.samefile(p)) for p in inputs):
        parser.error('output must not overwrite an input')
    try:
        report = audit_log(args.log, model_path=args.onnx)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end='')
    except (OSError, ValueError) as exc:
        parser.exit(2, f'audit failed: {exc}\n')
    return 2 if report['status'] == 'INVALID' else 0


if __name__ == '__main__':
    raise SystemExit(main())
