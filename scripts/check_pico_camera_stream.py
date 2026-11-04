#!/usr/bin/env python3
"""Check the PICO camera wire protocol from a host, optionally decode with PyAV."""
import argparse
from collections import Counter
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.pico_camera_bridge import (
    CONFIG, DISCONTINUITY, HEADER, KEYFRAME, MAGIC, MAX_PAYLOAD, STATUS, VERSION,
    VIDEO, independent_keyframe, nal_types,
)


def read_exact(connection, length):
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(length-len(data))
        if not chunk:
            raise EOFError('Camera connection closed')
        data.extend(chunk)
    return bytes(data)


def read(connection):
    header = HEADER.unpack(read_exact(connection, HEADER.size))
    magic, version, kind, flags, length, sequence, stamp = header
    if magic != MAGIC or version != VERSION or not 0 < length <= MAX_PAYLOAD:
        raise ValueError('Invalid camera wire header')
    return header, read_exact(connection, length)


def check(host, port=13579, *, seconds=5., decode=False):
    import importlib
    av = importlib.import_module('av') if decode else None
    counts = Counter()
    timestamps, receipts, picture_types = [], [], Counter()
    codec = None
    previous_sequence = None
    with socket.create_connection((host, port), timeout=3.) as connection:
        connection.settimeout(3.)
        header, payload = read(connection)
        if header[2] != CONFIG:
            raise ValueError('Missing camera configuration')
        config = json.loads(payload)
        if config['codec'] != 'hevc' or config['stereo_layout'] != 'mono':
            raise ValueError('Unexpected camera profile')
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            header, payload = read(connection)
            _, _, kind, flags, length, sequence, stamp = header
            if kind == STATUS:
                counts['status_packets'] += 1
                continue
            if kind != VIDEO:
                raise ValueError('Unexpected camera packet')
            types = nal_types(payload)
            key = independent_keyframe(types)
            if key != bool(flags & KEYFRAME):
                raise ValueError('Keyframe flag differs from HEVC payload')
            if previous_sequence is None or flags & DISCONTINUITY:
                if not key:
                    raise ValueError('Stream did not resume at independent keyframe')
                counts['decoder_resets'] += 1
                if av:
                    codec = av.CodecContext.create('hevc', 'r')
                    codec.thread_count = 1
            elif sequence != previous_sequence+1:
                raise ValueError('Unmarked frame loss')
            previous_sequence = sequence
            receipts.append(time.monotonic())
            timestamps.append(stamp)
            counts['frames'] += 1
            counts['bytes'] += length
            counts['keyframes'] += key
            counts['max_frame_bytes'] = max(counts['max_frame_bytes'], length)
            if codec:
                frames = codec.decode(av.Packet(payload))
                counts['decoded_frames'] += len(frames)
                picture_types.update(av.video.frame.PictureType(frame.pict_type).name for frame in frames)
        if codec:
            frames = codec.decode(None)
            counts['decoded_frames'] += len(frames)
            picture_types.update(av.video.frame.PictureType(frame.pict_type).name for frame in frames)
    if counts['frames'] < 2:
        raise RuntimeError('No continuous camera video received')
    elapsed = receipts[-1]-receipts[0]
    return dict(schema='vita_pico_camera_check_v1', configuration=config,
        counters=dict(counts), decoded_picture_types=dict(picture_types),
        received_hz=(len(receipts)-1)/elapsed,
        source_hz=(len(timestamps)-1)*1e9/(timestamps[-1]-timestamps[0]),
        maximum_receipt_gap_ms=max(b-a for a,b in zip(receipts, receipts[1:]))*1000,
        host_decode=decode, headset_display='not_measured', motion_commands_sent=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=13579)
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--decode', action='store_true', help='Requires host-only PyAV')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not 0 < args.seconds <= 300:
        p.error('Use a bounded check of 0..300 seconds')
    result = check(args.host, args.port, seconds=args.seconds, decode=args.decode)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
