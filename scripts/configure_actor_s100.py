#!/usr/bin/env python3
"""Provision local one-command startup. Writes defaults; creates no robot handles."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--robot-id', required=True, help='Actual local S100 IPv4 address')
    parser.add_argument('--robot-config', required=True, type=Path,
                        help='Absolute path to this board\'s running effective full A-sample YAML')
    parser.add_argument('--seconds', type=int, default=0, help='Default live duration; 0 means until STOP')
    parser.add_argument('--replace', action='store_true', help='Back up and replace reviewed local defaults')
    args = parser.parse_args()
    from deployment.vita_s100_aorta_30dof.actor_pico_local import configure_local_start
    root = args.root.resolve()
    (root/'build').mkdir(exist_ok=True)
    with (root/'build/operator.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another operator is using this deployment; configuration is blocked') from error
        result = configure_local_start(root, robot_id=args.robot_id,
            robot_config=args.robot_config, seconds=args.seconds, replace=args.replace)
    print(json.dumps(dict(result, actions_sent=0, control_handles_created=0,
                         motion_authorized=False, next='Run check/state-check/shadow before an authorized live trial'),
                     indent=2))


if __name__ == '__main__':
    main()
