"""Private, device-bound defaults for the no-argument S100 launcher.

This file stores device configuration, never on-site authorization. The local
launcher requests motion directly; there is no typed MOVE prompt.
"""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def local_addresses():
    reply = subprocess.run(['ip', '-j', 'addr', 'show'], check=True,
                           capture_output=True, text=True, timeout=3.)
    return [a['local'] for interface in json.loads(reply.stdout)
            for a in interface.get('addr_info', []) if a.get('family') == 'inet']


def configure_local_start(root, *, robot_id, robot_config, seconds=0, replace=False,
                          machine_id_path=Path('/etc/machine-id'),
                          selector_path=Path('/app_param/locomotion/end_effector'), addresses=None):
    """Write this board's defaults only; never connect, authorize or move it."""
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Configure local startup on the actual Linux ARM64 S100')
    target = str(ipaddress.IPv4Address(robot_id))
    addresses = local_addresses() if addresses is None else addresses
    if target not in addresses or ipaddress.ip_address(target).is_loopback:
        raise ValueError('Configured robot identity is not a local S100 address')
    if type(seconds) is not int or seconds < 0:
        raise ValueError('seconds must be an integer >= 0; 0 means until STOP')
    config = Path(robot_config)
    if not config.is_absolute():
        raise ValueError('robot_config must be the absolute running effective YAML path')
    from .actor_pico_live import load_robot_config
    load_robot_config(config)
    machine_id = Path(machine_id_path).read_bytes().strip()
    if not machine_id:
        raise ValueError('Local machine-id must not be empty')
    selector = Path(selector_path).read_text().strip() or 'none'
    profile = dict(schema='vita_actor_local_start_v1',
        machine_id_sha256=hashlib.sha256(machine_id).hexdigest(), robot_id=target,
        robot_config=str(config), robot_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        end_effector=selector, seconds=seconds, pico_handoff='manual')
    path = Path(root)/'outputs/actor_pico_local.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Local profile must not be a symlink')
    backup = None
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except ValueError:
            existing = None
        if existing == profile and not path.stat().st_mode & 0o077:
            return dict(path=str(path), changed=False, backup=None)
        if not replace:
            raise RuntimeError('Preserving existing local profile; use --replace after reviewing this board')
        with tempfile.NamedTemporaryFile('wb', prefix=path.name+'.bak.', dir=path.parent,
                                          delete=False) as saved:
            saved.write(path.read_bytes())
            backup = str(saved.name)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            os.chmod(temporary, 0o600)
            output.write(json.dumps(profile, indent=2)+'\n')
            output.flush()
            os.fsync(output.fileno())
            output.close()
            if replace:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)  # Refuse a concurrent creation instead of overwriting it.
        finally:
            temporary.unlink(missing_ok=True)
    load_local_start(root, machine_id_path=machine_id_path, selector_path=selector_path,
                     addresses=addresses)
    return dict(path=str(path), changed=True, backup=backup)


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
        addresses = local_addresses()
    if target not in addresses or ipaddress.ip_address(target).is_loopback:
        raise ValueError('Configured robot identity is not a local S100 address')
    selector = Path(selector_path).read_text().strip() or 'none'
    if selector != cfg['end_effector']:
        raise ValueError('End-effector selection changed; review the effective local robot config')
    config = Path(cfg['robot_config'])
    if not config.is_absolute() or hashlib.sha256(config.read_bytes()).hexdigest() != cfg['robot_config_sha256']:
        raise ValueError('Local effective robot configuration changed; review the device profile')
    if (cfg['pico_handoff'] not in ('manual','auto') or type(cfg['seconds']) is not int
            or cfg['seconds'] < 0):
        raise ValueError('Invalid local Pico handoff or motion duration (0 means until STOP)')
    return dict(robot_id=target,robot_config=config,seconds=cfg['seconds'],
                wait_pico=True,auto_pico=cfg['pico_handoff']=='auto')
