#!/usr/bin/env python3
"""Own private Pico service and one operator child; cancellation precedes service cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    if len(sys.argv)<2 or sys.argv[1] not in ('check','input-check','suspended-test'):
        raise ValueError('Explicit operator mode required')
    mode=sys.argv[1]
    sdk=root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join(map(str,[root/'build/pico',root/'vendor_packages/ort1232/lib',sdk,sdk/'lib',sdk/'SDK/arm64']))+':'+env.get('LD_LIBRARY_PATH','')
    env['PYTHONPATH']=':'.join(map(str,[root/'src',root/'src/deployment/unitree_g1_ros2_29dof/src',root/'build/pico']))
    env['QT_PLUGIN_PATH']=str(sdk/'plugins');env['QT_QPA_PLATFORM']='offscreen'
    runtime=root/'build/pico/xdg';runtime.mkdir(exist_ok=True,mode=0o700);env['XDG_RUNTIME_DIR']=str(runtime)
    children=[];stopping=False;stop_signal=None;log=None
    def stop(signum,frame):
        nonlocal stopping,stop_signal
        # A signal can arrive inside Popen before its return value is assigned.
        # Never infer the policy role from children[-1] in a signal handler.
        stopping=True;stop_signal=signum
    old_handlers={number:signal.signal(number,stop) for number in (signal.SIGINT,signal.SIGTERM)}
    try:
        service=None
        if mode!='check':
            log=(root/'outputs/pico_service_operator.log').open('w')
            service=subprocess.Popen([str(sdk/'RoboticsServiceProcess')],cwd=sdk,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(service)
        if stopping:return 128+stop_signal
        command=[sys.executable,str(root/'scripts/onboard_operator.py'),*sys.argv[1:]]
        if mode=='suspended-test':command=['chrt','-r','1',*command]
        child=subprocess.Popen(command,env=env,start_new_session=True);children.append(child)
        while child.poll() is None:
            if stopping:break
            if service is not None and service.poll() is not None:
                raise RuntimeError('Pico service exited; canceling operator')
            try:child.wait(timeout=.25)
            except subprocess.TimeoutExpired:pass
        if stopping:return 128+stop_signal
        return child.returncode if child.returncode>=0 else 128-child.returncode
    finally:
        cleanup_errors=[]
        # Policy is appended last. It receives cancellation and is reaped before
        # Pico service teardown, including failure and signal-during-spawn paths.
        for child in reversed(children):
            try:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait(timeout=2)
            except Exception as exc:
                cleanup_errors.append(exc)
        if log is not None:log.close()
        for number,handler in old_handlers.items():signal.signal(number,handler)
        if cleanup_errors:
            raise RuntimeError('Operator cleanup could not confirm child termination: '
                               + '; '.join(str(error) for error in cleanup_errors)) from cleanup_errors[0]

if __name__=='__main__':raise SystemExit(main())
