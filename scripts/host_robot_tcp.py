"""Socketlike test transport: Codex host -> SSH DPX relay -> robot TCP.

Only bytes are relayed on DPX; all fake-pose generation stays on the Codex host,
and all policy/SMPL computation stays onboard. This adds test transport overhead
and must not be described as a direct-headset network latency measurement.
"""
import os
from pathlib import Path
import shlex
import socket
import sys

sys.path.insert(0,'/home/yucheng.wang/.s100-tools')
from remote import client


class RobotTCPTransport:
    def __init__(self, ssh_command_file):
        self.command=Path(ssh_command_file).read_text().strip()
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.settimeout(15)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_BINDTODEVICE,b'enx6c1ff7c94514\0')
        self.sock.connect(('192.168.124.100',22))
        self.ssh=client('192.168.124.100','dpx',os.environ['S100_LAPTOP_PASSWORD'],self.sock)
        self.channels=[]

    def connect(self, port):
        port=int(port)
        if not 1024<=port<=65535:raise ValueError('Test port outside unprivileged range')
        # No robot control protocol is emitted: this simply relays the caller's
        # documented fake Pico framing to the isolated local Pico service.
        program='''import socket,sys,selectors,os
s=socket.create_connection(('127.0.0.1',PORT),10)
poll=selectors.DefaultSelector();poll.register(s,selectors.EVENT_READ);poll.register(sys.stdin.buffer,selectors.EVENT_READ)
while True:
 for key,_ in poll.select():
  if key.fileobj is s:
   b=s.recv(65536)
   if not b:sys.exit(0)
   sys.stdout.buffer.write(b);sys.stdout.buffer.flush()
  else:
   b=os.read(sys.stdin.fileno(),65536)
   if not b:sys.exit(0)
   s.sendall(b)
'''.replace('PORT',str(port))
        cmd='python3 -u -c '+shlex.quote(program)
        channel=self.ssh.get_transport().open_session()
        channel.exec_command(self.command+' '+shlex.quote(cmd))
        self.channels.append(channel)
        return channel

    def close(self):
        for channel in self.channels:channel.close()
        self.ssh.close();self.sock.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
