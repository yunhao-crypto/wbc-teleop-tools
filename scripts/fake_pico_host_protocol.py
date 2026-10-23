"""Host-generated synthetic Pico Body frames through official PCService TCP.

Only device registration, heartbeat and state telemetry are emitted. No Aorta,
actuator, lease or PCService device-control APIs exist in this module.
Accepts socket-like sendall/recv transports, including a Paramiko relay channel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import struct
import time

CLIENT_HEAD=0x3F
SERVER_HEAD=0xCF
TAIL=0xA5
CONNECT=0x19
HEARTBEAT=0x23
DEVICE_STATE_JSON=0x6D
DEFAULT_DEVICE_PORT=63901
HEADER=struct.Struct('<BBI')
TRAILER=struct.Struct('<QB')
MAX_PAYLOAD=1<<20

def assert_codex_host():
    if not Path('/home/yucheng.wang').is_dir():
        raise RuntimeError('Synthetic input must be generated on the Codex host workspace, not the robot/DPX')

def encode_frame(command:int,payload:bytes,*,tail_seconds:int|None=None)->bytes:
    if command not in (CONNECT,HEARTBEAT,DEVICE_STATE_JSON):
        raise ValueError('Only registration, heartbeat and input telemetry are supported')
    if not isinstance(payload,bytes) or len(payload)>MAX_PAYLOAD:
        raise ValueError('Invalid bounded protocol payload')
    stamp=int(time.time()) if tail_seconds is None else int(tail_seconds)
    if not 0<=stamp<2**64:raise ValueError('Tail timestamp out of range')
    return HEADER.pack(CLIENT_HEAD,command,len(payload))+payload+TRAILER.pack(stamp,TAIL)

class FrameDecoder:
    """Bounded stream decoder for diagnostics/tests; handles TCP fragmentation."""
    def __init__(self):self.pending=bytearray()
    def feed(self,data:bytes):
        self.pending.extend(data);frames=[]
        while len(self.pending)>=HEADER.size:
            head,command,size=HEADER.unpack_from(self.pending)
            if head not in (CLIENT_HEAD,SERVER_HEAD):raise ValueError('Invalid frame head')
            if size>MAX_PAYLOAD:raise ValueError('Unbounded frame length')
            total=HEADER.size+size+TRAILER.size
            if len(self.pending)<total:break
            seconds,tail=TRAILER.unpack_from(self.pending,HEADER.size+size)
            if tail!=TAIL:raise ValueError('Invalid frame tail')
            frames.append(dict(head=head,command=command,payload=bytes(self.pending[HEADER.size:HEADER.size+size]),tail_seconds=seconds))
            del self.pending[:total]
        return frames

def body_state_payload(body,stamp_ns:int)->bytes:
    if len(body)!=24 or not 0<int(stamp_ns)<2**63:raise ValueError('Expected24 joints and positive int64 timestamp')
    joints=[]
    for row in body:
        values=[float(v) for v in row]
        if len(values)!=7 or not all(math.isfinite(v) for v in values):raise ValueError('Expected finite24x7 Pico pose')
        norm=sum(v*v for v in values[3:])
        if abs(norm-1)>1e-4:raise ValueError('Expected normalized XYZW quaternion')
        joints.append({'p':','.join(format(v,'.17g') for v in values),'t':int(stamp_ns),'va':'0,0,0,0,0,0','wva':'0,0,0,0,0,0'})
    inner={'Controller':{},'Hand':{},'Body':{'timeStampNs':int(stamp_ns),'joints':joints},'timeStampNs':int(stamp_ns)}
    # Official pybind expects value to be a STRING containing a second JSON object.
    return json.dumps({'value':json.dumps(inner,separators=(',',':'),allow_nan=False)},separators=(',',':'),allow_nan=False).encode('utf-8')

class SyntheticBodyGenerator:
    """Deterministic smooth synthetic global Pico poses, computed on Codex host.

    Joint translations form a simple body-shaped fixture; rotations are global
    XYZW. This is protocol/preprocessing input, never a desired robot pose.
    """
    def __init__(self,seed=20260929):
        assert_codex_host();self.seed=int(seed);rng=random.Random(self.seed)
        self.amplitudes=[[rng.uniform(.03,.20) for _ in range(3)] for _ in range(24)]
        self.phases=[[rng.uniform(-math.pi,math.pi) for _ in range(3)] for _ in range(24)]
        # SMPL/Pico24 ordering; y is vertical in Pico input convention.
        self.positions=[(0,1.,0),(-.09,.92,0),(.09,.92,0),(0,1.10,0),(-.09,.53,0),(.09,.53,0),(0,1.23,0),(-.09,.10,0),(.09,.10,0),(0,1.36,0),(-.09,.04,.12),(.09,.04,.12),(0,1.52,0),(-.10,1.43,0),(.10,1.43,0),(0,1.70,0),(-.23,1.43,0),(.23,1.43,0),(-.48,1.30,0),(.48,1.30,0),(-.68,1.16,0),(.68,1.16,0),(-.74,1.12,0),(.74,1.12,0)]
    def frame(self,index:int,*,hz=50.):
        if index<0 or not math.isfinite(hz) or hz<=0:raise ValueError('Invalid frame index/rate')
        t=index/hz;dx=.04*math.sin(.7*t);dy=.01*math.cos(.5*t);dz=.03*math.sin(.4*t);result=[]
        for j,position in enumerate(self.positions):
            aa=[self.amplitudes[j][k]*math.sin(.8*t+self.phases[j][k]) for k in range(3)]
            theta=math.sqrt(sum(v*v for v in aa));scale=math.sin(theta/2)/theta if theta>1e-12 else .5
            q=[v*scale for v in aa]+[math.cos(theta/2)]
            result.append([position[0]+dx,position[1]+dy,position[2]+dz,*q])
        return result

class FakePicoClient:
    def __init__(self,transport,*,device_id='CODEX_SYNTHETIC_PICO_NO_ACTION'):
        if not device_id or '|' in device_id or len(device_id)>128:raise ValueError('Invalid device ID')
        self.transport=transport;self.device_id=device_id;self.registered=False;self.last_stamp_ns=0;self.frames=0;self.bytes=0;self.last_heartbeat=-math.inf
    def _send(self,command,payload):
        message=encode_frame(command,payload);self.transport.sendall(message);self.bytes+=len(message)
    def register(self,*,settle_seconds=.15):
        if not 0<=settle_seconds<=2:raise ValueError('Invalid registration settle delay')
        payload=(self.device_id+'|1').encode('utf-8')
        # First frame identifies the temporary socket; service consumes it while
        # transferring ownership to the UID model. Second frame notifies SDK.
        self._send(CONNECT,payload);time.sleep(settle_seconds)
        self._send(CONNECT,payload);time.sleep(settle_seconds)
        self.registered=True;self.heartbeat(force=True)
    def heartbeat(self,*,force=False):
        if not self.registered:raise RuntimeError('Register before telemetry')
        now=time.monotonic()
        if force or now-self.last_heartbeat>=5:
            self._send(HEARTBEAT,self.device_id.encode());self.last_heartbeat=now
    def send_body(self,body,*,timestamp_ns=None):
        if not self.registered:raise RuntimeError('Register before telemetry')
        stamp=time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        if stamp<=self.last_stamp_ns:raise ValueError('Body timestamp must strictly increase')
        payload=body_state_payload(body,stamp);self.heartbeat();self._send(DEVICE_STATE_JSON,payload)
        self.last_stamp_ns=stamp;self.frames+=1
        return {'frame_index':self.frames-1,'timestamp_ns':stamp,'payload_sha256':hashlib.sha256(payload).hexdigest(),'payload_bytes':len(payload)}

def stream_synthetic(transport,*,frames=1000,hz=50.,seed=20260929,device_id='CODEX_SYNTHETIC_PICO_NO_ACTION',on_frame=None):
    assert_codex_host()
    if not 1<=int(frames)<=1_000_000 or not math.isfinite(hz) or not .1<=hz<=240:raise ValueError('Invalid bounded stream length/rate')
    generator=SyntheticBodyGenerator(seed);client=FakePicoClient(transport,device_id=device_id);client.register()
    begin=time.monotonic();target=begin;overruns=0;first=None;last=None
    for i in range(int(frames)):
        delay=target-time.monotonic()
        if delay>0:time.sleep(delay)
        elif i and -delay>1/hz:overruns+=1;target=time.monotonic()
        body=generator.frame(i,hz=hz);record=client.send_body(body,timestamp_ns=max(time.time_ns(),client.last_stamp_ns+1))
        if on_frame:on_frame(i,body,record)
        first=first or record;last=record;target+=1/hz
    return dict(kind='Codex-host synthetic Pico TCP input; no robot actions',generation_host=socket.gethostname(),generation_pid=os.getpid(),generation_workspace='/home/yucheng.wang',seed=seed,frames=client.frames,hz=hz,wall_seconds=time.monotonic()-begin,bytes_sent=client.bytes,scheduling_overruns=overruns,first=first,last=last,device_id=device_id,protocol_device_port_default=DEFAULT_DEVICE_PORT,native_sdk_confirmation='Receiver must independently verify native SDK24x7 values and timestamps; successful send alone is not reception proof')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=DEFAULT_DEVICE_PORT);p.add_argument('--frames',type=int,default=1000);p.add_argument('--hz',type=float,default=50);p.add_argument('--seed',type=int,default=20260929);p.add_argument('--device-id',default='CODEX_SYNTHETIC_PICO_NO_ACTION');p.add_argument('--report',type=Path);a=p.parse_args();assert_codex_host()
    with socket.create_connection((a.host,a.port),timeout=10) as transport:
        report=stream_synthetic(transport,frames=a.frames,hz=a.hz,seed=a.seed,device_id=a.device_id)
    text=json.dumps(report,indent=2);print(text)
    if a.report:a.report.write_text(text+'\n')

if __name__=='__main__':main()
