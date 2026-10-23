"""Recovery regressions using deterministic wire time and localhost only."""
import errno
import importlib.util
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from deployment.common.controller_input import Button, ControllerSample
from deployment.common.wifi_controller import (
    PROTOCOL, ReceiverProtocol, WifiXboxController, bind_sender_interface,
    data_packet, decode_packet, encode_packet,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('send_xbox_wifi_recovery', ROOT/'scripts/send_xbox_wifi.py')
sender_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sender_module)
Sender = sender_module.ReconnectingXboxSender
KEY = b'k'*32
PEER = ('127.0.0.1', 12345)
CLIENT = 'a'*32


def sample(buttons=(), axes=None):
    return ControllerSample(0, 0., frozenset(buttons), axes or {})


def handshake(protocol, now, client=CLIENT):
    response = protocol.ingest(encode_packet(KEY, dict(protocol=PROTOCOL, kind='hello', client=client)), PEER, now)
    return decode_packet(KEY, response)['challenge']


class RecoveryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = ReceiverProtocol(KEY, allow_link_loss=True)
        self.challenge = handshake(self.protocol, 10.)
        self.send(0, 10.01, [Button.START], {'left_y': -.8})

    def send(self, seq, now, buttons=(), axes=None, client=CLIENT, challenge=None):
        frame = data_packet(client, challenge or self.challenge, seq, sample(buttons, axes))
        self.protocol.ingest(encode_packet(KEY, frame), PEER, now)

    def reconnect(self):
        lost = self.protocol.current(10.22)
        self.assertFalse(lost.buttons)
        self.assertFalse(lost.axes)
        self.assertFalse(self.protocol.status(10.22)['connected'])
        self.challenge = handshake(self.protocol, 10.23, 'b'*32)

    def test_loss_retires_session_and_rejects_delayed_controls(self):
        old = self.challenge
        self.reconnect()
        self.assertNotEqual(old, self.challenge)
        self.send(999, 10.24, [Button.START], {'left_y': 1.}, challenge=old)
        self.assertFalse(self.protocol.current(10.24).buttons)
        self.assertIsNone(self.protocol.latest)
        old_hello = dict(protocol=PROTOCOL, kind='hello', client=CLIENT)
        self.assertIsNone(self.protocol.ingest(encode_packet(KEY, old_hello), PEER, 10.25))
        self.assertEqual(self.protocol.client, 'b'*32)

    def test_rearm_requires_continuous_neutral_then_fresh_a(self):
        self.reconnect()
        # A inherited from before the reconnection cannot enable controls.
        for seq, now in enumerate([10.24, 10.34, 10.44]):
            self.send(seq, now, [Button.A, Button.START], {'left_y': 1.}, client='b'*32)
            self.assertFalse(self.protocol.current(now).buttons)
        for seq, now in enumerate([10.54, 10.64, 10.74, 10.84, 10.94, 11.05], 3):
            self.send(seq, now, client='b'*32)
        self.assertFalse(self.protocol.controls_enabled)
        # Any excursion cancels the completed dwell.
        self.send(9, 11.06, axes={'left_x': .5}, client='b'*32)
        self.send(10, 11.07, [Button.A], client='b'*32)
        self.assertFalse(self.protocol.controls_enabled)
        for seq, now in enumerate([11.08, 11.18, 11.28, 11.38, 11.48, 11.59], 11):
            self.send(seq, now, client='b'*32)
        self.send(17, 11.60, [Button.A], client='b'*32)
        self.assertTrue(self.protocol.controls_enabled)
        self.send(18, 11.61, [Button.START], {'left_y': -.4}, client='b'*32)
        self.assertEqual(self.protocol.current(11.61).axes['left_y'], -.4)

    def test_held_stop_suppressed_until_release_in_new_generation(self):
        self.reconnect()
        for seq, now in enumerate([10.24, 10.34]):
            self.send(seq, now, [Button.X, Button.BACK], client='b'*32)
            self.assertFalse(self.protocol.current(now).buttons)
        self.send(2, 10.44, client='b'*32)
        self.send(3, 10.45, [Button.X, Button.BACK, Button.START], client='b'*32)
        self.assertEqual(self.protocol.current(10.45).buttons, frozenset((Button.X, Button.BACK)))
        self.assertFalse(self.protocol.controls_enabled)


class FakeSocket:
    def __init__(self):
        self.incoming = []
        self.sent = []
        self.failure = None
        self.closed = False
    def connect(self, peer): self.peer = peer
    def setblocking(self, value): pass
    def recv(self, size):
        if not self.incoming: raise BlockingIOError()
        return self.incoming.pop(0)
    def send(self, payload):
        if self.failure: raise self.failure
        self.sent.append(decode_packet(KEY, payload))
    def close(self): self.closed = True


class SenderRecoveryTests(unittest.TestCase):
    def test_network_errors_retry_rebind_new_nonce_and_new_challenge(self):
        for error in (errno.ENETUNREACH, errno.ECONNREFUSED):
            with self.subTest(error=error):
                sockets = []
                def factory(*args):
                    sock = FakeSocket(); sockets.append(sock); return sock
                binder = Mock()
                sender = Sender(KEY, PEER, interface='wlan0', socket_factory=factory, route_binder=binder, logger=lambda _: None)
                self.assertFalse(sender.tick(sample(), 0.))
                old_client = sender.client
                challenge = dict(protocol=PROTOCOL, kind='challenge', client=old_client, challenge='c'*32)
                sockets[0].incoming.append(encode_packet(KEY, challenge))
                self.assertTrue(sender.tick(sample(), .01))
                sockets[0].failure = OSError(error, 'simulated network loss')
                self.assertFalse(sender.tick(sample([Button.START]), .02))
                self.assertTrue(sockets[0].closed)
                self.assertFalse(sender.tick(sample(), .1))
                self.assertEqual(len(sockets), 1)
                self.assertFalse(sender.tick(sample(), .28))
                self.assertNotEqual(sender.client, old_client)
                self.assertEqual(binder.call_count, 2)
                # Replayed acknowledgment cannot authenticate the new socket.
                sockets[1].incoming.append(encode_packet(KEY, challenge))
                self.assertFalse(sender.tick(sample(), .29))
                self.assertTrue(all(packet['kind']=='hello' for packet in sockets[1].sent))
                challenge.update(client=sender.client, challenge='d'*32)
                sockets[1].incoming.append(encode_packet(KEY, challenge))
                self.assertTrue(sender.tick(sample(), .30))
                self.assertEqual(sockets[1].sent[-1]['sequence'], 0)
                self.assertEqual(sockets[1].sent[-1]['challenge'], 'd'*32)
                sender.close()

    def test_missing_return_acks_forces_new_handshake(self):
        sockets = []
        def factory(*args):
            sock = FakeSocket(); sockets.append(sock); return sock
        sender = Sender(KEY, PEER, socket_factory=factory, logger=lambda _: None)
        sender.tick(sample(), 0.)
        sockets[0].incoming.append(encode_packet(KEY, dict(protocol=PROTOCOL, kind='challenge', client=sender.client, challenge='c'*32)))
        self.assertTrue(sender.tick(sample(), .01))
        self.assertFalse(sender.tick(sample(), .52))
        self.assertTrue(sockets[0].closed)
        self.assertIsNone(sender.challenge)

    def test_temporarily_missing_route_is_recoverable_without_binding(self):
        sock = Mock(); reply = bytearray(256); reply[20:24] = socket.inet_aton('192.168.50.7')
        with patch('deployment.common.wifi_controller.fcntl.ioctl', return_value=bytes(reply)), patch(
            'deployment.common.wifi_controller.subprocess.run', side_effect=subprocess.CalledProcessError(2, ['ip'])):
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify route'):
                bind_sender_interface(sock, 'wlan0', PEER[0])
        sock.bind.assert_not_called()


class LocalhostRecoveryTests(unittest.TestCase):
    def test_heartbeat_loss_and_receiver_socket_rebind_recover_with_neutral_gate(self):
        def until(predicate, timeout=4.):
            deadline = time.monotonic()+timeout
            while time.monotonic()<deadline:
                if predicate(): return
                time.sleep(.01)
            self.fail('localhost recovery condition timed out')
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            key = Path(folder)/'key'; key.write_bytes(KEY); key.chmod(0o600)
            receiver = WifiXboxController(bind_host='127.0.0.1', port=0, key_file=key, startup_timeout=2., allow_link_loss=True)
            sender = Sender(KEY, receiver.address, logger=lambda _: None)
            stop = threading.Event(); pause = threading.Event(); errors = []
            state = [sample([Button.START], {'left_y': -.7})]
            def pump():
                try:
                    while not stop.is_set():
                        if not pause.is_set(): sender.tick(state[0])
                        stop.wait(.02)
                except BaseException as exc: errors.append(exc)
            thread = threading.Thread(target=pump); thread.start()
            try:
                with receiver:
                    self.assertTrue(receiver.link_status['controls_enabled'])
                    pause.set()
                    until(lambda: not receiver.link_status['connected'])
                    neutral = receiver.sample()
                    self.assertFalse(neutral.buttons); self.assertFalse(neutral.axes)
                    generation = receiver.link_status['generation']
                    pause.clear()
                    until(lambda: receiver.link_status['connected'] and receiver.link_status['generation']>generation)
                    self.assertFalse(receiver.link_status['controls_enabled'])
                    self.assertFalse(receiver.sample().buttons)
                    state[0] = sample()
                    until(lambda: 'press A' in receiver.link_status['reason'])
                    state[0] = sample([Button.A])
                    until(lambda: receiver.link_status['controls_enabled'])
                    # Simulate the receiving socket disappearing while the link is up.
                    generation = receiver.link_status['generation']
                    with receiver._lock:
                        old_socket = receiver._socket
                        old_socket.close()
                    until(lambda: receiver._socket is not None and receiver._socket is not old_socket)
                    until(lambda: receiver.link_status['connected'] and receiver.link_status['generation']>generation)
                    self.assertFalse(receiver.link_status['controls_enabled'])
                    self.assertFalse(receiver.sample().buttons)  # Held A cannot rearm.
                    state[0] = sample()
                    until(lambda: 'press A' in receiver.link_status['reason'])
                    state[0] = sample([Button.A])
                    until(lambda: receiver.link_status['controls_enabled'])
            finally:
                stop.set(); thread.join(timeout=2.); sender.close(); receiver.close()
            self.assertEqual(errors, [])
            self.assertFalse(thread.is_alive()); self.assertFalse(receiver._thread.is_alive())


if __name__ == '__main__': unittest.main()
