#!/usr/bin/env python3
"""Read-only G1 reference retrieval via DPX; credentials entered without echo."""
import base64
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import sys
import time
import paramiko

sys.path.insert(0,'/home/yucheng.wang/.s100-tools')
from remote import client

out=Path('/home/yucheng.wang/s100/vita-onboard/outputs/network_reference')
out.mkdir(exist_ok=True,parents=True);out.chmod(0o700)
laptop_password=getpass.getpass('Authorized DPX credential: ')
robot_password=getpass.getpass('Authorized G1 credential: ')
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
    sock.settimeout(10);sock.setsockopt(socket.SOL_SOCKET,socket.SO_BINDTODEVICE,b'enx6c1ff7c94514\0');sock.connect(('192.168.124.100',22))
    with client('192.168.124.100','dpx',laptop_password,sock) as laptop:
        _,stdout,stderr=laptop.exec_command('ssh-keygen -F 192.168.124.161')
        trusted=stdout.read().decode()
        if stdout.channel.recv_exit_status()!=0:raise RuntimeError('DPX trusted G1 host key absent')
        keys=paramiko.HostKeys()
        for line in trusted.splitlines():
            if not line or line.startswith('#'):continue
            fields=line.split()
            entry=paramiko.hostkeys.HostKeyEntry.from_line('192.168.124.161 '+fields[-2]+' '+fields[-1])
            keys.add('192.168.124.161',entry.key.get_name(),entry.key)
        if not keys.lookup('192.168.124.161'):raise RuntimeError('No parsable trusted G1 host key')
        jump=laptop.get_transport().open_channel('direct-tcpip',('192.168.124.161',22),('127.0.0.1',0),timeout=5)
        robot=paramiko.SSHClient()
        for entry in keys.values():
            for kind,key in entry.items():robot.get_host_keys().add('192.168.124.161',kind,key)
        robot.set_missing_host_key_policy(paramiko.RejectPolicy())
        robot.connect('192.168.124.161',username='unitree',password=robot_password,sock=jump,allow_agent=False,look_for_keys=False,timeout=5,auth_timeout=5,banner_timeout=5)
        try:
            code="import socket,os,json,hashlib;from pathlib import Path;p=Path('/proc/device-tree/model');print(json.dumps({'hostname':socket.gethostname(),'machine':os.uname().machine,'home':str(Path.home()),'model':p.read_text().strip(chr(0)) if p.exists() else None,'machine_id_sha256':hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest()}))"
            _,stdout,stderr=robot.exec_command('python3 -c '+shlex.quote(code),timeout=5)
            identity=json.loads(stdout.read().decode())
            report={'target':'unitree@192.168.124.161','via':'dpx@192.168.124.100','identity':identity,'host_key_source':'Existing trusted DPX known_hosts; no acceptance of new key','read_only':True,'network_changed':False,'actions_sent':False,'directories':{},'files':[]}
            for key,command in [('devices','nmcli -f DEVICE,TYPE,STATE,CONNECTION device status'),('default_routes','ip -4 route show default')]:
                _,stdout,stderr=robot.exec_command(command,timeout=5)
                report[key]=stdout.read().decode()
            with robot.open_sftp() as sftp:
                for directory in ['/',identity['home'],'/root']:
                    try:entries=sftp.listdir_attr(directory)
                    except OSError as e:report['directories'][directory]={'readable':False,'error':type(e).__name__};continue
                    report['directories'][directory]={'readable':True,'top_level_names':[e.filename for e in entries]}
                    for entry in entries:
                        if len(report['files'])>=16:break
                        if not stat.S_ISREG(entry.st_mode):continue
                        name=entry.filename
                        if not re.search(r'wifi|asus|horizon|network|切网|switch.*net',name,re.I):continue
                        if not name.endswith(('.sh','.py','.bash','.conf','.yaml','.yml','.txt')):continue
                        if entry.st_size>131072:continue
                        path=directory.rstrip('/')+'/'+name
                        with sftp.open(path,'rb') as f:data=f.read()
                        basename=directory.strip('/').replace('/','_')+'__'+name
                        dest=out/basename
                        fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                        with os.fdopen(fd,'wb') as f:f.write(data)
                        dest.chmod(0o600)
                        report['files'].append({'remote_path':path,'saved_file':basename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mode':'0600','may_contain_wifi_credentials':bool(re.search(rb'password|passwd|psk|key-mgmt',data,re.I))})
            (out/'identity_and_inventory.json').write_text(json.dumps(report,indent=2)+'\n');(out/'identity_and_inventory.json').chmod(0o600)
            print(json.dumps({'target':report['target'],'identity':identity,'files':report['files'],'network_changed':False},indent=2))
        finally:robot.close()
