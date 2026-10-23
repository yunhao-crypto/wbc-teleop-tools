#!/usr/bin/env python3
"""Bounded local Pico receive-only readiness probe. No robot SDK imported."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert 1<=a.seconds<=120
    sdk=a.root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join(str(x) for x in [a.root/'build/pico',sdk,sdk/'lib',sdk/'SDK/arm64'])
    env['QT_PLUGIN_PATH']=str(sdk/'plugins');env['QT_QPA_PLATFORM']='offscreen'
    runtime=a.root/'build/pico/xdg';runtime.mkdir(exist_ok=True,mode=0o700)
    env['XDG_RUNTIME_DIR']=str(runtime)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    log=a.out.with_suffix('.service.log').open('w')
    service=subprocess.Popen([str(sdk/'RoboticsServiceProcess')],cwd=sdk,env=env,
                             stdout=log,stderr=subprocess.STDOUT)
    frames=[];error=None
    try:
        time.sleep(2)
        if service.poll() is not None:raise RuntimeError('Pico service exited during startup')
        sys.path.insert(0,str(a.root/'build/pico'))
        import xrobotoolkit_sdk as client
        # These are the only SDK calls: init, availability, timestamps, poses, close.
        client.init();last=None;start=time.monotonic()
        while time.monotonic()-start<a.seconds:
            if service.poll() is not None:raise RuntimeError('Pico service exited during receive probe')
            if client.is_body_data_available():
                stamp=int(client.get_time_stamp_ns())
                if stamp!=last:
                    frames.append(dict(timestamp_ns=stamp,received_monotonic_ns=time.monotonic_ns(),
                                       received_wall_ns=time.time_ns(),body=client.get_body_joints_pose()))
                    last=stamp
            time.sleep(.001)
        # Native close can block if service disconnects; parent timeout bounds probe.
        client.close()
    except Exception as exc:
        error=repr(exc)
    finally:
        service.terminate()
        try:service.wait(timeout=5)
        except subprocess.TimeoutExpired:service.kill();service.wait()
        log.close()
    result=dict(scope='Pico local ARM service/SDK receive only; no robot control APIs',
                seconds=a.seconds,frames_received=len(frames),error=error,action_sent=False,
                service_returncode=service.returncode,frames=frames)
    a.out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
    return 0 if error is None else 2


if __name__=='__main__':raise SystemExit(main())
