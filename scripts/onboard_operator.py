#!/usr/bin/env python3
"""Operator entry: offline check or explicitly requested supported robot test."""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

PINNED = {
 'build/libvita_backend.so':'256bc6643c9af8e55704509ca5e4c6fec34cdddd4a4c88ae7dacc5d35567b122',
 'artifacts/preprocessing_native_arm_v2.so':'20038faba69bce44083d477cb579217c1880b7f0eac35eda185777d93436dfd2',
 'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so':'6b26f5d424ef25f63511e93ea69532707679fda2026c1253dfccb677d8f409c0',
}
def configuration(root):
    from deployment.vita_s100_aorta_30dof.config import load_vita_config
    original=load_vita_config(root/'src/deployment/vita_s100_aorta_30dof/config/vita02a_s100.yaml')
    return replace(original,aorta_profile=Path('/app_param/zenoh/s100_session_peer.json5'),
        logs=root/'logs/live',sdk_lease_duration_seconds=7200.0)

def verify_static(root,config):
    from deployment.vita_s100_aorta_30dof.aorta_adapter import suspended_validation_alignment_quaternion
    alignment=suspended_validation_alignment_quaternion(config.imu_alignment_contract,
                                                       robot_serial=config.robot_serial)
    for relative,expected in PINNED.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Unvalidated native binary: '+relative)
    for variable,relative in [('VITA_PREPROCESS_LIBRARY','artifacts/preprocessing_native_arm_v2.so')]:
        if Path(os.environ.get(variable,'')).resolve()!=(root/relative).resolve():
            raise RuntimeError('Wrong native library environment: '+variable)
    from deployment.vita_s100_aorta_30dof.sdkmain_adapter import verify_state_decoder
    verify_state_decoder()
    import xrobotoolkit_sdk
    if Path(xrobotoolkit_sdk.__file__).resolve()!=(root/'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so').resolve():
        raise RuntimeError('Wrong Pico binding import path')
    if not callable(getattr(xrobotoolkit_sdk,'get_body_pose_snapshot',None)):
        raise RuntimeError('Atomic Pico snapshot API is required')
    from deployment.vita_s100_aorta_30dof.onboard_live_session import verify_vendor_sources
    verify_vendor_sources()
    return alignment

def controller_factory(args, key):
    if args.controller == 'bluetooth':
        from deployment.common.bluetooth_xbox import BluetoothXboxController, load_bluetooth_config
        config = load_bluetooth_config(args.bluetooth_config)
        return lambda _: BluetoothXboxController(
            address=config['address'], adapter=config['adapter'], startup_timeout=30.)
    if args.controller in ('local','usb'):
        from deployment.common.resilient_xbox import ResilientLocalXboxController
        return ResilientLocalXboxController
    from deployment.common.wifi_controller import WifiControllerReceiver
    return lambda _: WifiControllerReceiver(
        bind_host='0.0.0.0', port=args.port, key_file=key, allowed_peer=args.peer,
        max_age=.2, startup_timeout=60, allow_link_loss=True)

def input_check(root,cfg,alignment,key,args):
    """Read actual state/Pico/controller without constructing control handles."""
    import time
    import aorta
    def forbidden(*_,**__):raise RuntimeError('Input check forbids control handles')
    for name in dir(aorta.Node):
        if name.startswith('create_') and any(part in name for part in ('publisher','client','action','service')):
            setattr(aorta.Node,name,forbidden)
    from deployment.vita_s100_aorta_30dof.onboard_live_io import OnboardLiveStateSource
    from deployment.vita_s100_aorta_30dof.pico_input import create_pico_source
    controller_source = controller_factory(args, key)
    if not 0<args.seconds<=120:raise ValueError('Input check duration must be0..120seconds')
    pico=create_pico_source(maximum_age=cfg.pico_max_age,logger=logging.getLogger('input-check'))
    frames=0;pico_seen=False;last_pico_at=None;buttons=set()
    try:
        pico.start()
        with controller_source(cfg.joystick) as controller, OnboardLiveStateSource(group=cfg.aorta_group,alignment=alignment) as states:
            states.read(cfg.aorta_startup_timeout)
            deadline=time.monotonic()+args.seconds
            while time.monotonic()<deadline:
                raw=states.read(cfg.state_max_age);sample=controller.sample()
                buttons.update(b.value for b in sample.buttons);frames+=1
                body=pico.step();pico_seen=pico_seen or body is not None
                if body is not None:last_pico_at=time.monotonic()
                if frames%50==0:print(json.dumps(dict(state_id=raw.state_id,buttons=sorted(b.value for b in sample.buttons),axes=dict(sample.axes),pico_seen=pico_seen,controller_link=getattr(controller,'link_status',None))),flush=True)
                time.sleep(.02)
            controller_connected=getattr(controller,'link_status',{}).get('connected',True)
            pico_fresh=last_pico_at is not None and time.monotonic()-last_pico_at<=cfg.pico_max_age
    finally:pico.stop()
    result=dict(passed=frames>0 and pico_seen and pico_fresh and controller_connected,controller_connected_at_end=controller_connected,pico_fresh_at_end=pico_fresh,state_frames=frames,pico_seen=pico_seen,buttons_seen=sorted(buttons),actions_sent=0,control_handles_created=0)
    (root/'outputs/operator_input_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 0 if result['passed'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','input-check','suspended-test'))
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--controller',choices=('wifi','bluetooth','local','usb'),default='wifi')
    p.add_argument('--bluetooth-config',type=Path,help='Pinned paired Xbox identity, default config/xbox_bluetooth.json')
    p.add_argument('--joystick',type=Path,help='Local Linux joystick device (wireless gamepad), default from config')
    p.add_argument('--port',type=int,default=16502)
    p.add_argument('--key-file',type=Path)
    p.add_argument('--peer')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--motion-group',choices=(*tuple('ABCDE'), 'P1', 'P2', 'P3'),help='Select a hash-verified Motion experiment')
    a=p.parse_args();root=a.root.resolve()
    if a.command in ('check','suspended-test') and a.motion_group is None:
        p.error('--motion-group A/B/C/D/E/P1/P2/P3 is required for model check and robot test')
    a.bluetooth_config=a.bluetooth_config or root/'config/xbox_bluetooth.json'
    if a.controller=='bluetooth' and a.joystick is not None:
        p.error('Bluetooth selects the input device by paired MAC; omit --joystick')
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s %(message)s')
    if a.command=='suspended-test' and os.environ.get('HOLOMOTION_REAL_ROBOT')!='1':
        raise RuntimeError('Use the explicit suspended-test shell entry')
    if a.command!='suspended-test' and os.environ.get('HOLOMOTION_REAL_ROBOT'):
        raise RuntimeError('Action authorization must not enter check mode')
    (root/'outputs').mkdir(exist_ok=True)
    cfg=configuration(root)
    if a.joystick is not None:cfg=replace(cfg,joystick=a.joystick)
    alignment=verify_static(root,cfg)
    from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime
    from deployment.vita_s100_aorta_30dof.command_guard import command_envelope_from_contract
    key=a.key_file or root/'config/xbox_wifi.key'
    if a.controller=='wifi':
        if not key.is_file() or len(key.read_bytes())!=32 or key.stat().st_mode&0o077:
            raise RuntimeError('Wi-Fi controller requires private 0600 32-byte key file')
    if a.command=='input-check':
        return input_check(root,cfg,alignment,key,a)
    with OnboardDualPolicyRuntime.from_onboard_root(root,velocity_backend='ort',
             deployment_scope='suspended_validation',motion_group=a.motion_group) as runtime:
        print('MOTION_SELECTION '+json.dumps(runtime.motion_selection),flush=True)
        base=command_envelope_from_contract(json.loads(cfg.interface_contract.read_text()))
        for mode in ('velocity','motion'):runtime.command_envelope(base,mode)
        if a.command=='check':
            result={'passed':True,'actions_sent':False,'sdk_control_handles_created':0,
                    'scope':'suspended_validation','controller':a.controller,
                    'requested_lease_seconds':cfg.sdk_lease_duration_seconds,
                    'warmup':runtime.warmup(),'native_binaries':PINNED,
                    'motion_selection':runtime.motion_selection,
                    'hardware_inputs_checked':False,
                    'next':'input-check then operator-invoked suspended-test; START is manual'}
            (root/f'outputs/operator_check_{a.motion_group}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2));return 0
        from deployment.vita_s100_aorta_30dof.real_teleop import run_real_teleop
        from deployment.vita_s100_aorta_30dof.onboard_live_io import OnboardLiveStateSource,live_robot_state
        from deployment.vita_s100_aorta_30dof.onboard_live_session import OnboardLiveSession
        from deployment.vita_s100_aorta_30dof.async_recording import ProcessJsonlSessionRecorder
        selected_controller_factory=controller_factory(a,key)
        if a.controller=='wifi':
            logging.warning('Waiting for authenticated Xbox input on UDP %d; start the laptop sender.',a.port)
        elif a.controller=='bluetooth':
            logging.warning('Reading paired Bluetooth Xbox directly on S100; identity config: %s',a.bluetooth_config)
        else:
            logging.warning('Reading Xbox directly on S100 at %s; laptop sender is not required.',cfg.joystick)
        logging.warning('SUPPORTED/SUSPENDED TEST. START requests control and sends commands. '
                        'X/BACK/Ctrl-C cancel to PASSIVE; they are NOT a hardware emergency stop.')
        enabled=gc.isenabled()
        try:
            gc.collect();gc.disable()
            run_real_teleop(config=cfg,runtime=runtime,suspended_validation=True,
                state_source_factory=lambda **kw:OnboardLiveStateSource(alignment=alignment,**kw),
                state_converter=live_robot_state,session_factory=OnboardLiveSession,
                controller_factory=selected_controller_factory,
                recorder_factory=ProcessJsonlSessionRecorder)
        finally:
            if enabled:gc.enable()
    return 0
if __name__=='__main__':raise SystemExit(main())
