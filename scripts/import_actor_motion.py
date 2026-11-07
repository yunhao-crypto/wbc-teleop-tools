#!/usr/bin/env python3
"""Copy one hash-verified dataset NPZ to the fixed reference path; no robot I/O."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]


def selected_bytes(archive, clip):
    relative = PurePosixPath(clip)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.npz':
        raise ValueError('clip must be a relative NPZ member path')
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        manifests = [m for m in members if PurePosixPath(m.name).name == 'SHA256SUMS']
        if len(manifests) != 1 or not manifests[0].isfile() or manifests[0].size > 1024*1024:
            raise ValueError('Expected one regular SHA256SUMS')
        entries = {}
        for line in bundle.extractfile(manifests[0]).read().decode().splitlines():
            sha, name = line.split(maxsplit=1)
            name = name.lstrip('*')
            if name in entries:
                raise ValueError('Duplicate checksum entry')
            entries[name] = sha
        if clip not in entries:
            raise ValueError('Clip not listed in SHA256SUMS')
        member_name = str(PurePosixPath(manifests[0].name).parent/relative)
        matches = [m for m in members if m.name == member_name]
        if len(matches) != 1 or not matches[0].isfile() or not 0 < matches[0].size <= 64*1024*1024:
            raise ValueError('Expected one regular NPZ of at most 64 MiB')
        data = bundle.extractfile(matches[0]).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entries[clip]:
        raise ValueError('Dataset NPZ SHA256 mismatch')
    return data, dict(archive=str(Path(archive).resolve()), clip=clip, sha256=digest,
                      bytes=len(data), original_npz_unchanged=True)


def install(archive, clip, root, replace=False):
    data, receipt = selected_bytes(archive, clip)
    root = Path(root).resolve()
    (root/'build').mkdir(parents=True, exist_ok=True)
    with (root/'build/operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = root/'reference'
        folder.mkdir(exist_ok=True)
        path = folder/'fixed_motion.npz'
        if path.exists() and not replace:
            raise FileExistsError('Reference exists; use --replace to select another clip')
        if path.is_symlink():
            raise ValueError('Reference destination must not be a symlink')
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
            temp = Path(f.name)
            try:
                f.write(data); f.flush(); os.fsync(f.fileno()); f.close()
                if replace:
                    os.replace(temp, path)
                else:
                    os.link(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        receipt.update(path=str(path), actions_sent=0)
        (folder/'fixed_motion.import.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--clip', required=True, help='NPZ member path listed in the archive SHA256SUMS')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(args.archive, args.clip, args.root, args.replace), indent=2))


if __name__ == '__main__':
    main()
