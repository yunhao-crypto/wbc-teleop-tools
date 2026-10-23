#!/usr/bin/env python3
"""Run ONLY on the Codex host; send synthetic Pico through the test SSH relay."""
import argparse
import json
from pathlib import Path
from host_robot_tcp import RobotTCPTransport
from fake_pico_host_protocol import stream_synthetic, DEFAULT_DEVICE_PORT


def main():
    p=argparse.ArgumentParser();p.add_argument('--ssh-command-file',type=Path,required=True)
    p.add_argument('--frames',type=int,default=1500);p.add_argument('--hz',type=float,default=50.)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    def record(index,body,metadata):rows.append(dict(**metadata,body=body))
    with RobotTCPTransport(a.ssh_command_file) as bridge:
        channel=bridge.connect(DEFAULT_DEVICE_PORT)
        try:result=stream_synthetic(channel,frames=a.frames,hz=a.hz,on_frame=record)
        finally:
            a.out.with_suffix('.frames.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    result['transport']='Codex host -> encrypted SSH DPX byte relay -> robot-local native Pico TCP63901; DPX performs no pose or policy computation'
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
