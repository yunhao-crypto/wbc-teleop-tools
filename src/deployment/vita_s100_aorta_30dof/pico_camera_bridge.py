"""Read-only Aorta HEVC relay, independent of actor inference and ownership.

VTC1: network-endian >4sBBHIQQ header (28 bytes), then payload. Fields are
magic, version, kind, flags, length, bridge sequence, capture timestamp (ns).
Kinds: 1 JSON configuration, 2 one Annex-B HEVC access unit, 3 JSON status.
Flags: 1 independently decodable keyframe, 2 decoder discontinuity/reset.
The client initiates the only TCP connection. No robot command/RPC is used.
"""
from collections import Counter, deque
from dataclasses import dataclass
import json
import math
import re
import socket
import struct
import threading
import time


HEADER = struct.Struct('!4sBBHIQQ')
MAGIC, VERSION = b'VTC1', 1
CONFIG, VIDEO, STATUS = 1, 2, 3
KEYFRAME, DISCONTINUITY = 1, 2
MAX_PAYLOAD = 2 * 1024 * 1024
TOPICS = ('/image_front_raw/h265', '/image_left_raw/humanoid/h265',
          '/image_right_raw/humanoid/h265')
START_CODE = re.compile(b'\x00\x00(?:\x00)?\x01')


def nal_types(data):
    starts = list(START_CODE.finditer(data))
    if not starts or starts[0].start() > 1:
        raise ValueError('Expected Annex-B HEVC access unit')
    types = []
    for index, match in enumerate(starts):
        end = starts[index+1].start() if index+1 < len(starts) else len(data)
        if end-match.end() < 2:
            raise ValueError('Truncated HEVC NAL header')
        a, b = data[match.end():match.end()+2]
        if a & 128 or b & 7 == 0:
            raise ValueError('Invalid HEVC NAL header')
        types.append((a >> 1) & 63)
    return types


def independent_keyframe(types):
    # Start only at IDR/BLA with all parameter sets. CRA/RASL recovery is not
    # needed by the deployed encoder (IDR every 30 frames) and is not assumed.
    return {32, 33, 34}.issubset(types) and any(16 <= n <= 20 for n in types)


def packet(kind, payload, *, flags=0, sequence=0, stamp_ns=0):
    if not 0 < len(payload) <= MAX_PAYLOAD:
        raise ValueError('Video packet length out of bounds')
    return HEADER.pack(MAGIC, VERSION, kind, flags, len(payload), sequence, stamp_ns)+payload


def json_packet(kind, value):
    return packet(kind, json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


@dataclass(frozen=True)
class Frame:
    data: bytes
    sequence: int
    stamp_ns: int
    received: float
    keyframe: bool
    generation: int


class FrameBuffer:
    """Bounded queue. Loss/backpressure invalidates the GOP, never just a P frame."""
    def __init__(self, *, depth=6, maximum_age=.2, maximum_gap_ns=50_000_000):
        if depth < 1 or maximum_age <= 0 or maximum_gap_ns <= 0:
            raise ValueError('Invalid video buffering limits')
        self.depth, self.maximum_age = depth, maximum_age
        self.maximum_gap_ns = maximum_gap_ns
        self.condition = threading.Condition()
        self.queue = deque()
        self.waiting_keyframe = True
        self.generation = 0
        self.sequence = 0
        self.last_stamp = None
        self.last_received = None
        self.counts = Counter()

    def _reset(self, reason):
        self.counts['discarded_queued'] += len(self.queue)
        self.queue.clear()
        self.waiting_keyframe = True
        self.generation += 1
        self.counts['reset_'+reason] += 1

    def reset(self, reason):
        with self.condition:
            self._reset(reason)
            self.condition.notify_all()

    def push(self, data, stamp_ns, *, now=None):
        now = time.monotonic() if now is None else now
        if (not isinstance(data, bytes) or not 0 < len(data) <= MAX_PAYLOAD
                or type(stamp_ns) is not int or not 0 < stamp_ns < 1 << 64
                or not math.isfinite(now)):
            raise ValueError('Invalid video sample')
        types = nal_types(data)
        keyframe = independent_keyframe(types)
        with self.condition:
            self.counts['received_frames'] += 1
            self.counts['received_bytes'] += len(data)
            self.sequence += 1
            if self.last_stamp is not None and not 0 < stamp_ns-self.last_stamp <= self.maximum_gap_ns:
                self._reset('source_gap')
            self.last_stamp, self.last_received = stamp_ns, now
            if keyframe:
                self.counts['keyframes'] += 1
            if len(self.queue) >= self.depth:
                self._reset('overflow')
            if self.waiting_keyframe:
                if not keyframe:
                    self.counts['waiting_keyframe_drops'] += 1
                    return
                self.waiting_keyframe = False
            self.queue.append(Frame(data, self.sequence, stamp_ns, now, keyframe, self.generation))
            self.condition.notify()

    def pop(self, timeout=.25, *, clock=time.monotonic):
        deadline = clock()+timeout
        with self.condition:
            while True:
                now = clock()
                if self.queue:
                    if not 0 <= now-self.queue[0].received <= self.maximum_age:
                        self._reset('age')
                    else:
                        return self.queue.popleft()
                remaining = deadline-now
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

    def stats(self):
        with self.condition:
            return dict(self.counts, queued=len(self.queue), generation=self.generation,
                        waiting_keyframe=self.waiting_keyframe,
                        source_age_s=None if self.last_received is None else
                            max(0., time.monotonic()-self.last_received))


class CameraRelay:
    def __init__(self, buffer, *, listen=('0.0.0.0', 13579), topic=TOPICS[0],
                 width=960, height=768, fps=30, send_timeout=.2):
        self.buffer, self.listen = buffer, listen
        self.send_timeout = send_timeout
        self.configuration = dict(schema='vita_pico_camera_v1', codec='hevc',
            width=width, height=height, fps=fps, stereo_layout='mono', topic=topic,
            max_payload=MAX_PAYLOAD)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.address = None
        self.server = None
        self.client = None
        self.counters = Counter(connections=0, sent_frames=0, sent_bytes=0, disconnects=0)

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self.server = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(self.listen)
            server.listen(1)
            server.settimeout(.25)
            self.address = server.getsockname()
            self.ready.set()
            while not self.stop.is_set():
                try:
                    client, peer = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop.is_set():
                        break
                    raise
                self.client = client
                with client:
                    try:
                        client.settimeout(self.send_timeout)
                        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        # Limit stale bytes hidden in the kernel during Wi-Fi loss.
                        client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64*1024)
                        self.buffer.reset('connect')
                        self.counters['connections'] += 1
                        client.sendall(json_packet(CONFIG, self.configuration))
                        generation = None
                        while not self.stop.is_set():
                            frame = self.buffer.pop(.25)
                            if frame is None:
                                client.sendall(json_packet(STATUS, self.buffer.stats()))
                                continue
                            reset = frame.generation != generation
                            if reset and not frame.keyframe:
                                self.buffer.reset('unsafe_resume')
                                continue
                            flags = (KEYFRAME if frame.keyframe else 0) | (DISCONTINUITY if reset else 0)
                            client.sendall(packet(VIDEO, frame.data, flags=flags,
                                sequence=frame.sequence, stamp_ns=frame.stamp_ns))
                            generation = frame.generation
                            self.counters['sent_frames'] += 1
                            self.counters['sent_bytes'] += len(frame.data)
                    except (OSError, TimeoutError):
                        self.counters['disconnects'] += 1
                        self.buffer.reset('disconnect')
                    finally:
                        self.client = None
            self.server = None

    def close(self):
        self.stop.set()
        if self.client is not None:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.buffer.reset('shutdown')


def subscribe_camera(buffer, topic):
    # Imports stay here so protocol/queue tests need neither SDK nor robot I/O.
    import aorta
    from foxglove.CompressedVideo import CompressedVideo
    node = aorta.Node('pico_camera_bridge', group='default')
    last_drop_count = [0]

    def receive(payload, context):
        try:
            msg = CompressedVideo.GetRootAs(payload, 0)
            if msg.Format() != b'h265':
                raise ValueError('Camera must publish format=h265')
            stamp = msg.Timestamp()
            if stamp is None:
                raise ValueError('Camera timestamp missing')
            buffer.push(msg.DataAsNumpy().tobytes(), stamp.Sec()*10**9+stamp.Nsec())
        except Exception:
            buffer.reset('invalid_sample')

    subscriber = node.create_subscriber(topic, receive, qos=aorta.QoS.sensor_data(),
        options=aorta.SubscriberOptions(receive_depth=8))

    def check_drops():
        stats = subscriber.runtime_stats()
        count = stats.executor_drop_count+stats.local_drop_count
        if count != last_drop_count[0]:
            last_drop_count[0] = count
            buffer.reset('subscriber_drop')
    return node, check_drops
