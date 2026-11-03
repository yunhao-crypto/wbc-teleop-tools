"""Protocol, inter-frame recovery and actual TCP lifecycle without robot I/O."""
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.pico_camera_bridge import (
    CameraRelay, CONFIG, DISCONTINUITY, FrameBuffer, HEADER, KEYFRAME, MAGIC,
    MAX_PAYLOAD, STATUS, VIDEO, independent_keyframe, nal_types, packet,
)


def nal(kind):
    return b'\0\0\0\1'+bytes((kind << 1, 1, 0x80))


KEY = b''.join(nal(n) for n in (32, 33, 34, 19))
DELTA = nal(1)


def read_exact(connection, count):
    data = bytearray()
    while len(data) < count:
        chunk = connection.recv(count-len(data))
        if not chunk:
            raise EOFError('Connection closed')
        data.extend(chunk)
    return bytes(data)


def read_packet(connection):
    header = HEADER.unpack(read_exact(connection, HEADER.size))
    return header, read_exact(connection, header[4])


class BufferTests(unittest.TestCase):
    def test_only_complete_independent_keyframe_opens_gop(self):
        buffer = FrameBuffer()
        buffer.push(DELTA, 1, now=1.)
        self.assertEqual(len(buffer.queue), 0)
        buffer.push(nal(19), 33_333_334, now=1.)
        self.assertEqual(len(buffer.queue), 0)
        buffer.push(KEY, 66_666_667, now=1.)
        buffer.push(DELTA, 100_000_000, now=1.)
        self.assertEqual([f.keyframe for f in buffer.queue], [True, False])
        self.assertTrue(independent_keyframe(nal_types(KEY)))
        self.assertFalse(independent_keyframe([32, 33, 34, 21]))

    def test_overflow_discards_the_entire_gop_until_next_keyframe(self):
        buffer = FrameBuffer(depth=2)
        buffer.push(KEY, 1, now=1.)
        buffer.push(DELTA, 33_333_334, now=1.)
        buffer.push(DELTA, 66_666_667, now=1.)
        self.assertEqual(len(buffer.queue), 0)
        self.assertTrue(buffer.waiting_keyframe)
        buffer.push(KEY, 100_000_000, now=1.)
        self.assertTrue(buffer.queue[0].keyframe)
        self.assertEqual(buffer.queue[0].generation, 1)

    def test_source_gap_duplicate_and_regression_each_require_new_keyframe(self):
        for stamp in (1, 0, 100_000_000):
            with self.subTest(stamp=stamp):
                buffer = FrameBuffer()
                buffer.push(KEY, 2, now=1.)
                # stamp=0 is not a valid timestamp; rejected input is distinct.
                if stamp == 0:
                    with self.assertRaises(ValueError): buffer.push(DELTA, 0, now=1.)
                    continue
                buffer.push(DELTA, stamp, now=1.)
                self.assertEqual(len(buffer.queue), 0)
                self.assertTrue(buffer.waiting_keyframe)
        buffer = FrameBuffer()
        buffer.push(KEY, 2, now=1.)
        buffer.push(DELTA, 2, now=1.)
        self.assertTrue(buffer.waiting_keyframe)

    def test_stale_queue_is_not_replayed(self):
        buffer = FrameBuffer(maximum_age=.01)
        buffer.push(KEY, 1, now=time.monotonic()-.02)
        self.assertIsNone(buffer.pop(.001))
        self.assertEqual(buffer.stats()['reset_age'], 1)

    def test_malformed_nals_and_payload_bounds_are_rejected(self):
        for data in (b'', b'not annex b', b'\0\0\1\x26', b'\0\0\1\x26\0',
                     b'\0\0\1\xff\x01', b'x'*(MAX_PAYLOAD+1)):
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                FrameBuffer().push(data, 1)

    def test_network_header_is_big_endian_and_preserves_64bit_timestamp(self):
        wire = packet(VIDEO, KEY, flags=3, sequence=1234, stamp_ns=1_800_000_000_012_345_678)
        self.assertEqual(len(wire), 28+len(KEY))
        header = HEADER.unpack(wire[:28])
        self.assertEqual(header, (MAGIC, 1, VIDEO, 3, len(KEY), 1234, 1_800_000_000_012_345_678))


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.buffer = FrameBuffer()
        self.relay = CameraRelay(self.buffer, listen=('127.0.0.1', 0))
        self.worker = threading.Thread(target=self.relay.serve, daemon=True)
        self.worker.start()
        self.assertTrue(self.relay.ready.wait(2))
        self.stamp = 1_000_000_000

    def tearDown(self):
        self.relay.close()
        self.worker.join(2)
        self.assertFalse(self.worker.is_alive())

    def connect(self):
        client = socket.create_connection(self.relay.address, timeout=2)
        client.settimeout(2)
        self.addCleanup(client.close)
        header, payload = read_packet(client)
        self.assertEqual(header[2], CONFIG)
        config = json.loads(payload)
        self.assertEqual((config['codec'], config['width'], config['height']), ('hevc', 960, 768))
        return client

    def send_frame(self, data):
        self.stamp += 33_333_333
        self.buffer.push(data, self.stamp)

    def next_video(self, client):
        for _ in range(10):
            header, payload = read_packet(client)
            if header[2] == VIDEO: return header, payload
        self.fail('No video received')

    def test_real_tcp_receives_keyframe_then_delta_and_status(self):
        client = self.connect()
        self.send_frame(DELTA)
        self.send_frame(KEY)
        header, data = self.next_video(client)
        self.assertEqual((header[3], data), (KEYFRAME | DISCONTINUITY, KEY))
        self.send_frame(DELTA)
        header, data = self.next_video(client)
        self.assertEqual((header[3], data), (0, DELTA))
        header, data = read_packet(client)
        self.assertEqual(header[2], STATUS)
        self.assertIn('source_age_s', json.loads(data))

    def test_recovery_marks_discontinuity_and_never_sends_intermediate_delta(self):
        client = self.connect()
        self.send_frame(KEY)
        self.next_video(client)
        self.buffer.reset('test_loss')
        self.send_frame(DELTA)
        self.send_frame(KEY)
        header, data = self.next_video(client)
        self.assertEqual(header[3], KEYFRAME | DISCONTINUITY)
        self.assertEqual(data, KEY)

    def test_client_disconnect_can_reconnect_at_new_keyframe(self):
        first = self.connect()
        self.send_frame(KEY)
        self.next_video(first)
        first.shutdown(socket.SHUT_RDWR)
        first.close()
        second = self.connect()
        self.send_frame(KEY)
        header, data = self.next_video(second)
        self.assertEqual(header[3], KEYFRAME | DISCONTINUITY)
        self.assertEqual(data, KEY)


if __name__ == '__main__':
    unittest.main()
