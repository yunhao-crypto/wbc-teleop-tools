"""Private, device-bound defaults for the no-argument S100 launcher.

This file stores configuration, never on-site authorization. Every motion run
still requires current interactive confirmation of the displayed robot.
"""
import hashlib
import ipaddress
import json
from pathlib import Path
import platform
import subprocess


def load_local_start(root, *, machine_id_path=Path('/etc/machine-id'),
                     selector_path=Path('/app_param/locomotion/end_effector'), addresses=None):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('No-argument motion startup requires the configured local S100')
    path = Path(root)/'outputs/actor_pico_local.json'
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError('Missing/private local startup profile: '+str(path)+
                           '; provision this device or use explicit auto target/config options')
    cfg = json.loads(path.read_text())
    if cfg.get('schema') != 'vita_actor_local_start_v1':
        raise ValueError('Unknown local startup profile schema')
    if hashlib.sha256(Path(machine_id_path).read_bytes().strip()).hexdigest() != cfg['machine_id_sha256']:
        raise ValueError('Local startup profile belongs to another device')
    target = str(ipaddress.IPv4Address(cfg['robot_id']))
    if addresses is None:
        reply = subprocess.run(['ip','-j','addr','show'],check=True,capture_output=True,text=True,timeout=3.)
        addresses = [a['local'] for interface in json.loads(reply.stdout)
                     for a in interface.get('addr_info',[]) if a.get('family')=='inet']
    if target not in addresses or ipaddress.ip_address(target).is_loopback:
        raise ValueError('Configured robot identity is not a local S100 address')
    selector = Path(selector_path).read_text().strip() or 'none'
    if selector != cfg['end_effector']:
        raise ValueError('End-effector selection changed; review the effective local robot config')
    config = Path(cfg['robot_config'])
    if not config.is_absolute() or hashlib.sha256(config.read_bytes()).hexdigest() != cfg['robot_config_sha256']:
        raise ValueError('Local effective robot configuration changed; review the device profile')
    if (cfg['pico_handoff'] not in ('manual','auto') or type(cfg['seconds']) is not int
            or not 1 <= cfg['seconds'] <= 120):
        raise ValueError('Invalid local Pico handoff or bounded motion duration')
    return dict(robot_id=target,robot_config=config,seconds=cfg['seconds'],
                wait_pico=True,auto_pico=cfg['pico_handoff']=='auto')
