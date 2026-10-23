#!/usr/bin/env python3
"""Own a temporary local Pico service for one bounded NO-ACTION shadow run."""
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    sdk=root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join(str(x) for x in [root/'build/pico',root/'vendor_packages/ort1232/lib',sdk,sdk/'lib',sdk/'SDK/arm64'])+':'+env.get('LD_LIBRARY_PATH','')
    env['QT_PLUGIN_PATH']=str(sdk/'plugins');env['QT_QPA_PLATFORM']='offscreen'
    env['PYTHONPATH']=':'.join(str(x) for x in [root/'src',root/'build/pico'])
    runtime=root/'build/pico/xdg';runtime.mkdir(exist_ok=True,mode=0o700)
    env['XDG_RUNTIME_DIR']=str(runtime)
    children=[];telemetry=[]
    def interrupt(signum,frame):
        for child in reversed(children):
            if child.poll() is None:child.terminate()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    log=(root/'outputs/pico_service_shadow.log').open('w')
    try:
        service=subprocess.Popen([str(sdk/'RoboticsServiceProcess')],cwd=sdk,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append(service)
        command=[sys.executable,str(root/'scripts/run_onboard_shadow.py'),*sys.argv[1:]]
        priority=env.get('VITA_RR_PRIORITY')
        if priority is not None:
            if int(priority)!=1:raise ValueError('Validated SCHED_RR priority is 1')
            command=['chrt','-r',priority,*command]
        env['OPENBLAS_NUM_THREADS']='1'
        child=subprocess.Popen(command,env=env)
        children.append(child)
        while child.poll() is None:
            service_status=service.poll()
            if service_status is not None:
                raise RuntimeError(f'Pico service exited during shadow run: {service_status}; see {log.name}')
            sample={'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
                    'policy_pid':child.pid,'pico_service_pid':service.pid}
            for path in [Path('/proc/loadavg'),Path('/proc/meminfo'),
                         *Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'),
                         *Path('/sys/class/thermal').glob('thermal_zone*/temp')]:
                try:sample[str(path)]=path.read_text().strip()
                except OSError:pass
            telemetry.append(sample)
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        service_status=service.poll()
        if service_status is not None:
            raise RuntimeError(f'Pico service exited during shadow run: {service_status}; see {log.name}')
        return child.returncode
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()
        (root/'outputs/onboard_shadow_telemetry.json').write_text(json.dumps(telemetry,indent=2)+'\n')


if __name__=='__main__':raise SystemExit(main())
