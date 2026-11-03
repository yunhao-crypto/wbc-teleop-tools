#!/usr/bin/env python3
"""Serve an existing Aorta camera stream to the PICO video client; no motion."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.pico_camera_bridge import (
    CameraRelay, FrameBuffer, TOPICS, subscribe_camera,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=13579)
    p.add_argument('--topic', choices=TOPICS, default=TOPICS[0])
    p.add_argument('--seconds', type=float, default=0., help='0 runs until interrupted')
    p.add_argument('--stats', type=Path, help='Write atomic local JSON status every second')
    p.add_argument('--ready-file', type=Path, help='Owned launcher readiness record; removed on exit')
    args = p.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.seconds < float('inf'):
        p.error('Invalid port/duration')
    if not os.environ.get('ZENOH_SESSION_CONFIG_URI'):
        p.error('Set ZENOH_SESSION_CONFIG_URI to the local robot session configuration')
    if os.environ.get('AORTA_GROUP', 'default') != 'default':
        p.error('Camera requires AORTA_GROUP=default')
    buffer = FrameBuffer()
    relay = CameraRelay(buffer, listen=(args.host, args.port), topic=args.topic)
    node, check_drops = subscribe_camera(buffer, args.topic)
    failure = []

    def run():
        try:
            relay.serve()
        except BaseException as error:
            failure.append(error)
            relay.stop.set()
    worker = threading.Thread(target=run, name='pico-camera-tcp', daemon=True)
    previous = {sig: signal.signal(sig, lambda *_: relay.close())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    start = time.monotonic()
    try:
        worker.start()
        if not relay.ready.wait(3.) or failure:
            raise RuntimeError('Camera TCP listener did not start') from (failure[0] if failure else None)
        print(json.dumps(dict(event='camera_bridge_ready', address=relay.address,
            **relay.configuration)), flush=True)
        if args.ready_file:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.ready_file.with_suffix(args.ready_file.suffix+'.tmp')
            temporary.write_text(json.dumps(dict(pid=os.getpid(), address=relay.address,
                                                **relay.configuration))+'\n')
            temporary.replace(args.ready_file)
        while not relay.stop.wait(1.):
            check_drops()
            row = dict(buffer.stats(), **relay.counters, elapsed_s=time.monotonic()-start)
            print(json.dumps(row, allow_nan=False), flush=True)
            if args.stats:
                args.stats.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.stats.with_suffix(args.stats.suffix+'.tmp')
                temporary.write_text(json.dumps(row, indent=2)+'\n')
                temporary.replace(args.stats)
            if args.seconds and time.monotonic()-start >= args.seconds:
                break
    finally:
        relay.close()
        worker.join(2.)
        node.close()
        if args.ready_file:
            args.ready_file.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if failure:
        raise RuntimeError('Camera relay failed') from failure[0]


if __name__ == '__main__':
    main()
