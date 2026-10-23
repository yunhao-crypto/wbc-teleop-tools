"""Deterministic protocol invariants plus localhost UDP/thread integration."""
import json
import os
from pathlib import Path
import secrets
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch
from deployment.common.controller_input import Button,ControllerSample,ControllerEdgeTracker
from deployment.common.wifi_controller import (PROTOCOL,MAX_DATAGRAM,ControllerLinkError,ReceiverProtocol,
    WifiXboxController,encode_packet,decode_packet,data_packet,load_key,bind_sender_interface)

KEY=b'k'*32
PEER=('127.0.0.1',34567)
CLIENT='a'*32

def hello(client=CLIENT):return {'protocol':PROTOCOL,'kind':'hello','client':client}
def connect(protocol,now=10.,peer=PEER,client=CLIENT):
    answer=protocol.ingest(encode_packet(KEY,hello(client)),peer,now)
    return decode_packet(KEY,answer)['challenge']
def data(challenge,seq=0,buttons=(),axes=None,client=CLIENT):
    return data_packet(client,challenge,seq,ControllerSample(seq,0.,frozenset(buttons),axes or {}))

class ProtocolTests(unittest.TestCase):
    def test_authenticated_handshake_pins_one_session_and_receiver_clock(self):
        p=ReceiverProtocol(KEY);c=connect(p)
        p.ingest(encode_packet(KEY,data(c,0,[Button.START],{'left_y':-.5})),PEER,10.01)
        x=p.current(10.10);self.assertEqual(x.monotonic_time,10.01);self.assertEqual(x.sequence,0)
        self.assertEqual(x.buttons,frozenset([Button.START]));self.assertEqual(x.axes['left_y'],-.5)
        p.ingest(encode_packet(KEY,hello('b'*32)),PEER,10.11)
        with self.assertRaisesRegex(ControllerLinkError,'session changed'):p.current(10.12)
    def test_hmac_and_peer_pin(self):
        p=ReceiverProtocol(KEY,allowed_peer='127.0.0.1')
        self.assertIsNone(p.ingest(encode_packet(b'z'*32,hello()),PEER,10.))
        self.assertIsNone(p.ingest(encode_packet(KEY,hello()),('127.0.0.2',1),10.))
        self.assertIsNone(p.client)
        c=connect(p);p.ingest(encode_packet(KEY,data(c)),('127.0.0.1',PEER[1]+1),10.01)
        self.assertIsNone(p.latest)
    def test_sequence_duplicate_reorder_do_not_refresh_timeout(self):
        p=ReceiverProtocol(KEY);c=connect(p)
        p.ingest(encode_packet(KEY,data(c,4)),PEER,10.01)
        for seq in [4,3,0]:p.ingest(encode_packet(KEY,data(c,seq,[Button.START])),PEER,10.1)
        self.assertEqual(p.sequence,4);self.assertEqual(p.accepted,1);self.assertEqual(p.received_at,10.01)
        with self.assertRaises(ControllerLinkError):p.current(10.211)
        p.ingest(encode_packet(KEY,data(c,5)),PEER,10.212)
        with self.assertRaises(ControllerLinkError):p.current(10.213)
    def test_prior_receiver_session_replay_and_expired_challenge_rejected(self):
        old=ReceiverProtocol(KEY);old_c=connect(old)
        old_frame=encode_packet(KEY,data(old_c,10,[Button.START]))
        fresh=ReceiverProtocol(KEY);new_c=connect(fresh)
        self.assertNotEqual(old_c,new_c);fresh.ingest(old_frame,PEER,10.01);self.assertIsNone(fresh.latest)
        fresh.ingest(encode_packet(KEY,data(new_c)),PEER,12.01);self.assertIsNone(fresh.latest)
    def test_bad_data_and_bounds(self):
        p=ReceiverProtocol(KEY);c=connect(p)
        malformed=[]
        for seq in [True,-1,2**63,1.2]:d=data(c);d['sequence']=seq;malformed.append(d)
        for axes in [{'left_y':2},{'left_y':True},{'unknown':0},{'left_x':'0'}]:d=data(c);d['axes']=axes;malformed.append(d)
        for buttons in [['unknown'],['start','start'],[{}]]:d=data(c);d['buttons']=buttons;malformed.append(d)
        for d in malformed:p.ingest(encode_packet(KEY,d),PEER,10.01)
        for raw in [b'{}',b'x'*(MAX_DATAGRAM+1),b'not json']:p.ingest(raw,PEER,10.01)
        self.assertIsNone(p.latest);self.assertEqual(p.dropped,len(malformed)+3)
        with self.assertRaises(ValueError):encode_packet(KEY,{'axis':float('nan')})
    def test_bye_is_authenticated_session_scoped(self):
        p=ReceiverProtocol(KEY);c=connect(p);p.ingest(encode_packet(KEY,data(c)),PEER,10.01)
        bye={'protocol':PROTOCOL,'kind':'bye','client':CLIENT,'challenge':c}
        p.ingest(encode_packet(b'z'*32,bye),PEER,10.02);self.assertIsNone(p.fault)
        p.ingest(encode_packet(KEY,bye),PEER,10.03)
        with self.assertRaisesRegex(ControllerLinkError,'stopped'):p.current(10.04)
    def test_key_permissions_and_length(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'key';f.write_bytes(KEY);f.chmod(0o600);self.assertEqual(load_key(f),KEY)
            f.chmod(0o644)
            with self.assertRaises(ValueError):load_key(f)
            f.chmod(0o600);f.write_bytes(b'short')
            with self.assertRaises(ValueError):load_key(f)

class SocketTests(unittest.TestCase):
    def test_localhost_heartbeat_survives_blocking_consumer_then_latches_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'key';f.write_bytes(KEY);f.chmod(0o600)
            receiver=WifiXboxController(bind_host='127.0.0.1',port=0,key_file=f,startup_timeout=1.)
            stop=threading.Event();errors=[]
            def sender():
                try:
                    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
                        s.settimeout(.2);s.sendto(encode_packet(KEY,hello()),receiver.address)
                        packet,_=s.recvfrom(MAX_DATAGRAM);c=decode_packet(KEY,packet)['challenge'];seq=0
                        while not stop.is_set():
                            s.sendto(encode_packet(KEY,data(c,seq)),receiver.address);seq+=1;stop.wait(.02)
                except BaseException as e:errors.append(e)
            t=threading.Thread(target=sender);t.start()
            try:
                with receiver:
                    first=receiver.sample();time.sleep(.30) # Simulated blocking lease handshake; receiver thread stays fresh.
                    last=receiver.sample();self.assertGreater(last.sequence,first.sequence)
                    # More frequent polling must advance local edge-tracker sequence
                    # without inventing a second wire packet or repeating button edges.
                    tracker=ControllerEdgeTracker();tracker.update(last)
                    from deployment.common.controller_input import VelocityLimits
                    from deployment.vita_s100_aorta_30dof.teleop_engine import VitaTeleopEngine
                    engine=VitaTeleopEngine(VelocityLimits())
                    engine.update(last,robot_ready=True,motion_reference_ready=True)
                    for _ in range(5):
                        again=receiver.sample();self.assertGreater(again.sequence,last.sequence)
                        self.assertEqual(tracker.update(again),frozenset())
                        engine.update(again,robot_ready=True,motion_reference_ready=True)
                        last=again
                    stop.set();t.join();time.sleep(.23)
                    with self.assertRaisesRegex(ControllerLinkError,'heartbeat'):receiver.sample()
            finally:stop.set();t.join();receiver.close()
            self.assertEqual(errors,[]);self.assertFalse(receiver._thread.is_alive())
    def test_startup_timeout_closes_socket_thread(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'key';f.write_bytes(KEY);f.chmod(0o600)
            r=WifiXboxController(bind_host='127.0.0.1',port=0,key_file=f,startup_timeout=.05)
            with self.assertRaisesRegex(ControllerLinkError,'connection timed out'):
                with r:pass
            self.assertFalse(r._thread.is_alive());self.assertEqual(r._socket.fileno(),-1)


class InterfaceTests(unittest.TestCase):
    def test_bind_source_ipv4_after_matching_route_without_privileged_device_bind(self):
        sock=Mock();reply=bytearray(256);reply[20:24]=socket.inet_aton('192.168.50.7')
        result=Mock(stdout='[{"dev":"wlp0s20f3"}]')
        with patch('deployment.common.wifi_controller.fcntl.ioctl',return_value=bytes(reply)),patch('deployment.common.wifi_controller.subprocess.run',return_value=result) as command:
            self.assertEqual(bind_sender_interface(sock,'wlp0s20f3','192.168.50.8'),'192.168.50.7')
        sock.bind.assert_called_once_with(('192.168.50.7',0))
        self.assertEqual(command.call_args.args[0],['ip','-j','route','get','192.168.50.8','from','192.168.50.7'])
        sock.setsockopt.assert_not_called()
    def test_wrong_wired_route_fails_before_bind_or_send(self):
        sock=Mock();reply=bytearray(256);reply[20:24]=socket.inet_aton('192.168.50.7')
        with patch('deployment.common.wifi_controller.fcntl.ioctl',return_value=bytes(reply)),patch('deployment.common.wifi_controller.subprocess.run',return_value=Mock(stdout='[{"dev":"eth0"}]')):
            with self.assertRaisesRegex(RuntimeError,'not requested'):bind_sender_interface(sock,'wlp0s20f3','192.168.50.8')
        sock.bind.assert_not_called();sock.send.assert_not_called()

if __name__=='__main__':unittest.main()
