"""Authenticated Xbox telemetry over UDP; contains no robot/control SDK calls.

Strict receivers latch connection loss. Optional live recovery supplies neutral
input while disconnected, authenticates a fresh challenge, and requires a
centered/released500ms interval plus a fresh A before restoring controls.
"""
from __future__ import annotations
import hashlib
import fcntl
import hmac
import json
import math
from pathlib import Path
import secrets
import socket
import stat
import struct
import subprocess
import threading
import time
from .controller_input import Button, ControllerSample

PROTOCOL='vita-xbox-v1'
DEFAULT_PORT=16502
MAX_DATAGRAM=2048
AXES=frozenset(('left_x','left_y','right_x','right_y'))

class ControllerLinkError(RuntimeError):pass


def load_key(path):
    path=Path(path);info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError('Controller key must be a private regular file (mode 600)')
    key=path.read_bytes()
    if len(key)!=32:raise ValueError('Controller key file must contain exactly 32 random bytes')
    return key


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')

def encode_packet(key,body):
    raw=_json(body)
    packet=_json({'body':body,'mac':hmac.new(key,raw,hashlib.sha256).hexdigest()})
    if len(packet)>MAX_DATAGRAM:raise ValueError('Controller datagram too large')
    return packet


def decode_packet(key,packet):
    if len(packet)>MAX_DATAGRAM:raise ValueError('Controller datagram too large')
    envelope=json.loads(packet)
    if not isinstance(envelope,dict) or set(envelope)!= {'body','mac'}:raise ValueError('Malformed envelope')
    body=envelope['body'];mac=envelope['mac']
    if not isinstance(body,dict) or not isinstance(mac,str) or len(mac)!=64:raise ValueError('Malformed authentication')
    if not hmac.compare_digest(mac,hmac.new(key,_json(body),hashlib.sha256).hexdigest()):raise ValueError('Authentication failed')
    if body.get('protocol')!=PROTOCOL:raise ValueError('Unknown controller protocol')
    return body


def _nonce(value):
    return isinstance(value,str) and len(value)==32 and all(c in '0123456789abcdef' for c in value)


def data_packet(client,challenge,seq,sample):
    return {'protocol':PROTOCOL,'kind':'data','client':client,'challenge':challenge,'sequence':seq,
            'buttons':sorted(b.value for b in sample.buttons),'axes':dict(sample.axes)}


class ReconnectControlGate:
    """Transport-independent post-loss gate; caller owns freshness and sequence.

    Call reset() on every loss/new connection, observe() on fresh input only,
    then filter() on sampled input. Initial enabled=True preserves the original
    real-loop startup gate. X/BACK require release in this recovery generation.
    """
    def __init__(self,*,neutral_seconds=.5,deadzone=.15):
        self.neutral_seconds=neutral_seconds;self.deadzone=deadzone
        self.reset();self.enabled=True;self.reason='Controls enabled'
    def reset(self):
        self.enabled=False;self.reason='Reconnected: center sticks and release buttons for 500 ms'
        self.neutral_since=None;self.neutral_ready=False;self.previous_buttons=frozenset();self.stop_release_seen=set()
    def observe(self,buttons,axes,now):
        if self.enabled:return
        for stop in (Button.X,Button.BACK):
            if stop not in buttons:self.stop_release_seen.add(stop)
        centered=all(abs(float(value))<=self.deadzone for value in axes.values())
        fresh_a=buttons==frozenset((Button.A,)) and Button.A not in self.previous_buttons
        if centered and fresh_a and self.neutral_ready:
            self.enabled=True;self.reason='Controls re-enabled by fresh A'
        elif centered and not buttons:
            if self.neutral_since is None:self.neutral_since=now
            self.neutral_ready=now-self.neutral_since>=self.neutral_seconds
            self.reason=('Reconnected: press A to re-enable controls' if self.neutral_ready else
                         'Reconnected: center sticks and release buttons for 500 ms')
        else:
            self.neutral_since=None;self.neutral_ready=False
            self.reason='Reconnected: center sticks and release buttons for 500 ms'
        self.previous_buttons=buttons
    def filter(self,sample):
        if self.enabled:return sample
        return ControllerSample(sample.sequence,sample.monotonic_time,
            frozenset(sample.buttons.intersection(self.stop_release_seen)),{})


class ReceiverProtocol:
    """Authenticated wire state; optional recovery never reuses old controls.

    The socket owner serializes access with its condition lock. Strict mode is
    the historical diagnostic behavior. Recovery mode invalidates the challenge
    on every loss and requires neutral500ms plus a fresh A after reconnection.
    """
    REARM_NEUTRAL_SECONDS=.5
    REARM_DEADZONE=.15
    def __init__(self,key,*,max_age=.2,allowed_peer=None,allow_link_loss=False):
        if len(key)!=32:raise ValueError('32-byte key required')
        if not math.isfinite(max_age) or not .02<=max_age<=1.:raise ValueError('max_age must be .02..1 seconds')
        self.key=key;self.max_age=max_age;self.allowed_peer=allowed_peer;self.allow_link_loss=bool(allow_link_loss)
        self.client=None;self.challenge=None;self.peer=None;self.issued_at=None
        self.latest=None;self.received_at=None;self.sequence=-1;self.fault=None;self.accepted=0;self.dropped=0
        self.ever_connected=False;self.controls_enabled=False;self.generation=0
        self.reason='Waiting for authenticated sender';self._last_received_at=None
        self._neutral_since=None;self._neutral_ready=False;self._previous_buttons=frozenset()
        self._stop_release_seen=set();self._retired_clients=set();self.rearm=ReconnectControlGate()

    def link_lost(self,reason,now):
        if not self.allow_link_loss:
            self.fault=reason;self.reason=reason;self.controls_enabled=False
            return
        if self.client is not None and self.ever_connected:
            self._retired_clients.add((self.client,self.peer))
        self.client=None;self.challenge=None;self.peer=None;self.issued_at=None
        self.latest=None;self.received_at=None;self.sequence=-1
        self.controls_enabled=False;self.reason=reason
        self._neutral_since=None;self._neutral_ready=False
        self._previous_buttons=frozenset();self._stop_release_seen.clear();self.rearm.reset()

    def _check_age(self,now):
        if self.received_at is not None and now-self.received_at>self.max_age:
            self.link_lost('Controller heartbeat lost; authenticate a new session',now)
        if self.fault:raise ControllerLinkError(self.fault)

    def status(self,now):
        try:self._check_age(now)
        except ControllerLinkError:pass
        connected=self.latest is not None and self.received_at is not None and not self.fault
        age=None if self._last_received_at is None else max(0.,now-self._last_received_at)
        return dict(connected=bool(connected),controls_enabled=bool(connected and self.controls_enabled),
                    reason=self.reason,generation=self.generation,age_s=age)

    def current(self,now):
        self._check_age(now)
        if self.latest is None:
            if self.allow_link_loss and self.ever_connected:
                return ControllerSample(0,now,frozenset(),{})
            raise ControllerLinkError('Controller has not connected')
        if self.controls_enabled:return self.latest
        # During rearm, an inherited held X/BACK is suppressed. A release must
        # have been observed in THIS authenticated generation before its press.
        return self.rearm.filter(ControllerSample(self.latest.sequence,now,self.latest.buttons,{}))

    def _begin_session(self,client,peer,now):
        self.client=client;self.peer=peer;self.challenge=secrets.token_hex(16);self.issued_at=now
        self.sequence=-1;self.latest=None;self.received_at=None;self.controls_enabled=False
        self.generation+=1;self._neutral_since=None;self._neutral_ready=False
        self._previous_buttons=frozenset();self._stop_release_seen.clear();self.rearm.reset()
        self.reason='Authenticating controller session'

    def _accept_controls(self,parsed,axes,now):
        if not self.ever_connected:
            self.ever_connected=True;self.controls_enabled=True
            self.rearm.enabled=True;self.reason='Controls enabled'
            return
        if self.controls_enabled:return
        self.rearm.observe(parsed,axes,now)
        self.controls_enabled=self.rearm.enabled;self.reason=self.rearm.reason

    def ingest(self,packet,peer,now):
        if self.fault:return None
        try:self._check_age(now)
        except ControllerLinkError:return None
        if self.allowed_peer is not None and peer[0]!=self.allowed_peer:self.dropped+=1;return None
        try:
            body=decode_packet(self.key,packet);kind=body.get('kind');client=body.get('client')
            if not _nonce(client):raise ValueError('Bad client nonce')
            if kind=='hello':
                if set(body)!= {'protocol','kind','client'}:raise ValueError('Bad hello')
                if (client,peer) in self._retired_clients:raise ValueError('Retired sender session')
                if self.latest is not None and (client!=self.client or peer!=self.peer):
                    self.link_lost('Authenticated controller session changed; authenticate again',now)
                    if not self.allow_link_loss:return None
                if client!=self.client or peer!=self.peer or (self.latest is None and now-self.issued_at>2.):
                    self._begin_session(client,peer,now)
                return encode_packet(self.key,{'protocol':PROTOCOL,'kind':'challenge','client':self.client,'challenge':self.challenge})
            if client!=self.client or peer!=self.peer or body.get('challenge')!=self.challenge:
                raise ValueError('Unrecognized session')
            if self.latest is None and now-self.issued_at>2.:raise ValueError('Expired handshake')
            if kind=='bye':
                if set(body)!= {'protocol','kind','client','challenge'}:raise ValueError('Bad bye')
                self.link_lost('Controller sender stopped; authenticate a new session',now);return None
            if kind!='data' or set(body)!= {'protocol','kind','client','challenge','sequence','buttons','axes'}:raise ValueError('Bad data')
            seq=body['sequence'];buttons=body['buttons'];axes=body['axes']
            if type(seq) is not int or not 0<=seq<2**63 or seq<=self.sequence:raise ValueError('Replayed or unordered sequence')
            if not isinstance(buttons,list) or len(buttons)>len(Button) or len(set(buttons))!=len(buttons):raise ValueError('Bad buttons')
            if not isinstance(axes,dict) or set(axes)-AXES:raise ValueError('Bad axes')
            if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1 for v in axes.values()):raise ValueError('Bad axis value')
            parsed=frozenset(Button(b) for b in buttons)
            if not self.allow_link_loss:
                self.ever_connected=True;self.controls_enabled=True;self.reason='Controls enabled'
            else:self._accept_controls(parsed,axes,now)
            self.latest=ControllerSample(seq,now,parsed,dict(axes));self.sequence=seq
            self.received_at=now;self._last_received_at=now;self.accepted+=1
            self._previous_buttons=parsed
        except (ValueError,TypeError,KeyError,OverflowError,UnicodeError,RecursionError):self.dropped+=1
        return None


class WifiXboxController:
    """Drop-in controller with strict default or neutral-on-loss recovery mode."""
    def __init__(self,*,bind_host='0.0.0.0',port=DEFAULT_PORT,key_file,max_age=.2,allowed_peer=None,
                 startup_timeout=30.,allow_link_loss=False):
        if not math.isfinite(startup_timeout) or startup_timeout<=0:raise ValueError('startup_timeout must be positive')
        self.protocol=ReceiverProtocol(load_key(key_file),max_age=max_age,allowed_peer=allowed_peer,
                                       allow_link_loss=allow_link_loss)
        self.allow_link_loss=bool(allow_link_loss);self._sample_sequence=0
        self.startup_timeout=startup_timeout;self._lock=threading.Condition();self._stop=threading.Event();self._thread=None
        self._bind_host=bind_host;self._bind_port=port;self._socket=self._open_socket()
        self.address=self._socket.getsockname();self._bind_port=self.address[1]

    def _open_socket(self):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:sock.bind((self._bind_host,self._bind_port));sock.settimeout(.025)
        except BaseException:sock.close();raise
        return sock

    def _socket_lost(self,sock,reason):
        with self._lock:
            if self._stop.is_set():return
            self.protocol.link_lost(reason,time.monotonic());self._lock.notify_all()
            if self._socket is sock:self._socket=None
        sock.close()

    def _receive(self):
        window=time.monotonic();count=0
        while not self._stop.is_set():
            now=time.monotonic()
            with self._lock:
                try:self.protocol._check_age(now)
                except ControllerLinkError:self._lock.notify_all();return
                sock=self._socket
            if sock is None:
                if not self.allow_link_loss:return
                try:replacement=self._open_socket()
                except OSError:self._stop.wait(.1);continue
                with self._lock:
                    if self._stop.is_set():replacement.close();return
                    self._socket=replacement;self.address=replacement.getsockname()
                continue
            try:packet,peer=sock.recvfrom(MAX_DATAGRAM+1)
            except socket.timeout:continue
            except OSError:
                if self._stop.is_set():return
                self._socket_lost(sock,'Controller socket failed; waiting to rebind')
                if self.allow_link_loss:self._stop.wait(.1);continue
                return
            now=time.monotonic()
            if now-window>=1.:window=now;count=0
            count+=1
            if count>500:
                self._stop.wait(.002);continue
            with self._lock:
                response=self.protocol.ingest(packet,peer,now);self._lock.notify_all()
            if response is not None:
                try:sock.sendto(response,peer)
                except OSError:
                    if self.allow_link_loss:self._socket_lost(sock,'Controller reply socket failed; waiting to rebind')

    def __enter__(self):
        if self._thread is not None:raise RuntimeError('Controller context cannot be entered twice')
        self._thread=threading.Thread(target=self._receive,name='xbox-wifi',daemon=True);self._thread.start()
        try:
            deadline=time.monotonic()+self.startup_timeout
            with self._lock:
                while self.protocol.latest is None:
                    if self.protocol.fault:raise ControllerLinkError(self.protocol.fault)
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise ControllerLinkError('Controller connection timed out; start sender first')
                    self._lock.wait(min(remaining,.05))
            return self
        except BaseException:self.close();raise

    @property
    def link_status(self):
        with self._lock:return dict(self.protocol.status(time.monotonic()))

    def sample(self,timeout=0.):
        if self._thread is None:raise ControllerLinkError('Controller context is not entered')
        if self._stop.is_set():raise ControllerLinkError('Controller closed')
        if not math.isfinite(timeout) or timeout<0:raise ValueError('Invalid controller timeout')
        with self._lock:
            if timeout:self._lock.wait(min(timeout,self.protocol.max_age))
            latest=self.protocol.current(time.monotonic())
            sample=ControllerSample(self._sample_sequence,latest.monotonic_time,latest.buttons,dict(latest.axes))
            self._sample_sequence+=1
            return sample

    def close(self):
        self._stop.set()
        with self._lock:
            self.protocol.link_lost('Controller closed',time.monotonic());self._lock.notify_all()
            sock=self._socket
        if sock is not None:sock.close()
        if self._thread is not None:self._thread.join(timeout=.5)
    def __exit__(self,*args):self.close()


WifiControllerReceiver=WifiXboxController


def bind_sender_interface(sock,interface,destination):
    """Bind source IPv4 and verify the route selects the requested interface.

    No SO_BINDTODEVICE/root privileges and no network configuration mutations.
    Fail closed if an overlapping wired route would carry this source address.
    """
    if not isinstance(interface,str) or not interface or len(interface.encode('ascii'))>15 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-' for c in interface):
        raise ValueError('Invalid network interface name')
    try:
        request=struct.pack('256s',interface.encode('ascii'))
        reply=fcntl.ioctl(sock.fileno(),0x8915,request) # Linux SIOCGIFADDR
        source=socket.inet_ntoa(reply[20:24])
    except OSError as exc:raise RuntimeError(f'Interface {interface} has no usable IPv4 address') from exc
    try:
        route=subprocess.run(['ip','-j','route','get',destination,'from',source],check=True,capture_output=True,text=True)
    except (OSError,subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'Cannot verify route to {destination} on {interface}') from exc
    routes=json.loads(route.stdout)
    if not routes or routes[0].get('dev')!=interface:
        selected=routes[0].get('dev') if routes else None
        raise RuntimeError(f'Route to {destination} from {source} selects {selected}, not requested {interface}; fix Wi-Fi routing before sending')
    sock.bind((source,0))
    return source
