#!/usr/bin/env python3
"""Offline only: inspect a native capture; PARTIAL is never a safety PASS."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deployment.vita_s100_aorta_30dof.actor_native_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--onnx', type=Path, help='Optional CPU replay of recorded observation (no history reconstruction)')
    parser.add_argument('--max-pending', type=int, default=256)
    parser.add_argument('--max-line-bytes', type=int, default=2*1024*1024)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    args = parser.parse_args()
    try:
        report = audit(args.capture, max_pending=args.max_pending,
                       max_line_bytes=args.max_line_bytes, onnx=args.onnx,
                       atol=args.atol, rtol=args.rtol)
    except Exception as exc:
        print(json.dumps({'status': 'PARTIAL', 'error': str(exc), 'scope': 'offline_analysis_failed'}))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    # 0 means a report was produced, not PASS. Read status/diagnostics.
    return 0


if __name__ == '__main__':
    sys.exit(main())
