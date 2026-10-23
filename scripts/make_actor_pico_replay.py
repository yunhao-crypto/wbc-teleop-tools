#!/usr/bin/env python3
"""Copy the versioned synthetic actor/Pico replay fixture to a new output path."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Replay output exists; choose a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/'tests/fixtures/actor_pico/pico_replay.npz', args.output)
    print(json.dumps(dict(output=str(args.output), frames=384,
                          kind='synthetic offline fixture; not hardware evidence')))


if __name__ == '__main__':
    main()
