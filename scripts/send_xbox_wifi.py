#!/usr/bin/env python3
"""Send authenticated Xbox telemetry only; reconnect without network mutations."""
import argparse
from pathlib import Path
import secrets
import socket
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from deployment.common.wifi_controller import DEFAULT_PORT,MAX_DATAGRAM,PROTOCOL,load_key,encode_packet,decode_packet,data_packet,_nonce,bind_sender_interface
from deployment.common.xbox_joystick import LocalXboxController


class ReconnectingXboxSender:
    """Nonblocking authenticated telemetry with bounded challenge acknowledgments."""
    def __init__(self,key,peer,*,interface=None,clock=time.monotonic,socket_factory=socket.socket,
                 route_binder=bind_sender_interface,logger=print,retry_interval=.25):
        self.key=key;self.peer=peer;self.interface=interface;self.clock=clock
        self.socket_factory=socket_factory;self.route_binder=route_binder;self.logger=logger
        self.retry_interval=retry_interval;self.sock=None;self.client=None;self.challenge=None
        self.sequence=0;self.next_retry=0.;self.last_hello=-float('inf');self.last_ack=None
        self.opened_at=None;self.reason=None;self.generation=0
    def _note(self,reason):
        if reason!=self.reason:self.reason=reason;self.logger(reason)
    def _disconnect(self,reason,now):
        if self.sock is not None:self.sock.close()
        self.sock=None;self.client=None;self.challenge=None;self.last_ack=None
        self.next_retry=now+self.retry_interval
        self._note('Reconnection waiting: '+reason+'. Center sticks/release buttons; after recovery press A.')
    def _open(self,now):
        sock=self.socket_factory(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            if self.interface:self.route_binder(sock,self.interface,self.peer[0])
            sock.connect(self.peer);sock.setblocking(False)
        except BaseException:sock.close();raise
        self.sock=sock;self.client=secrets.token_hex(16);self.challenge=None;self.sequence=0
        self.opened_at=now;self.last_hello=-float('inf');self.last_ack=None
        self._note('Waiting for authenticated receiver challenge; keep sticks centered and buttons released.')
    def tick(self,sample,now=None):
        now=self.clock() if now is None else now
        if self.sock is None:
            if now<self.next_retry:return False
            try:self._open(now)
            except (OSError,RuntimeError,ValueError) as exc:
                self._disconnect(str(exc),now);return False
        try:
            # Keep reading replies after the initial handshake. Periodic HELLO
            # detects receiver restart/new challenge and proves return reachability.
            for _ in range(16):
                try:packet=self.sock.recv(MAX_DATAGRAM+1)
                except (BlockingIOError,socket.timeout):break
                try:
                    body=decode_packet(self.key,packet)
                    if set(body)!={'protocol','kind','client','challenge'} or body['kind']!='challenge' or body['client']!=self.client or not _nonce(body['challenge']):continue
                except (ValueError,TypeError,KeyError,UnicodeError,OverflowError,RecursionError):continue
                if body['challenge']!=self.challenge:
                    self.challenge=body['challenge'];self.sequence=0;self.generation+=1
                self.last_ack=now
                self._note('Authenticated telemetry connected; recovery controls require neutral500ms then a fresh A.')
            if self.last_ack is not None and now-self.last_ack>.5:
                self._disconnect('receiver challenge acknowledgment expired',now);return False
            if self.challenge is None and now-self.opened_at>1.:
                self._disconnect('receiver has not authenticated this session',now);return False
            if now-self.last_hello>=.1:
                self.sock.send(encode_packet(self.key,{'protocol':PROTOCOL,'kind':'hello','client':self.client}))
                self.last_hello=now
            if self.challenge is None:return False
            self.sock.send(encode_packet(self.key,data_packet(self.client,self.challenge,self.sequence,sample)))
            self.sequence+=1
            return True
        except OSError as exc:
            self._disconnect(str(exc),now);return False
    def close(self):
        if self.sock is not None:
            if self.challenge is not None:
                try:self.sock.send(encode_packet(self.key,{'protocol':PROTOCOL,'kind':'bye','client':self.client,'challenge':self.challenge}))
                except OSError:pass
            self.sock.close();self.sock=None
        self.challenge=None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=DEFAULT_PORT)
    p.add_argument('--interface',help='Wi-Fi interface; verify and bind its source IPv4 on every reconnect')
    p.add_argument('--key-file',required=True);p.add_argument('--joystick',type=Path,default=Path('/dev/input/js0'))
    a=p.parse_args();key=load_key(a.key_file);peer=(socket.gethostbyname(a.host),a.port)
    sender=ReconnectingXboxSender(key,peer,interface=a.interface,logger=lambda message:print(message,flush=True))
    try:
        with LocalXboxController(a.joystick) as controller:
            deadline=time.monotonic()
            while True:
                delay=deadline-time.monotonic()
                if delay>0:time.sleep(delay)
                sender.tick(controller.sample(timeout=0.))
                deadline+=.02
                if deadline<time.monotonic():deadline=time.monotonic()+.02
    except KeyboardInterrupt:pass
    finally:sender.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
