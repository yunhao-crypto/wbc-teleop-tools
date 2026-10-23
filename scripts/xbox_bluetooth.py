#!/usr/bin/env python3
"""Explicit Bluetooth gamepad setup; default status never changes Bluetooth state.

Pairing requires the operator to hold the controller pairing button first.
Only explicit pair/connect/disconnect/discover commands change radio/device state.
No adapter power, pairability, discoverability, service or vendor settings change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
DEFAULT_CONFIG = ROOT/'config/xbox_bluetooth.json'


def address(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', value):
        raise ValueError('Expected an exact Bluetooth MAC address, e.g. AA:BB:CC:DD:EE:FF')
    return value.upper()


def adapter_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'hci[0-9]+', value):
        raise ValueError('Adapter must be an exact hciN name')
    return value


def run_command(args, timeout=5.):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed (exit {result.returncode}): {(result.stderr or result.stdout).strip() or "no diagnostic output"}')
    return result.stdout


def resolve_input(mac):
    from deployment.common.bluetooth_xbox import resolve_bluetooth_joystick
    identity = resolve_bluetooth_joystick(mac)
    return str(identity.joystick_path)


def clean_terminal(text):
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text).replace('\r', '\n')


def observed_devices(transcript):
    """Keep scan observations even if BlueZ removes unpaired objects on exit."""
    observed = {}
    for match in re.finditer(r'(?:\[NEW\]\s+)?Device\s+((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})\s+([^\n]+)', clean_terminal(transcript)):
        mac, name = match.groups(); mac = address(mac); name = name.strip()
        if name.startswith(('RSSI:', 'TxPower:', 'ManufacturerData', 'ServiceData', 'Connected:', 'Paired:', 'UUIDs:', 'ServicesResolved:')):
            continue
        if name.startswith(('Name:', 'Alias:')): name = name.split(':', 1)[1].strip()
        observed[mac] = dict(address=mac, name=name)
    return list(observed.values())


class BluetoothCtlSession:
    """One bounded client owns discovery until pairing has been verified."""
    def __init__(self, *, popen=subprocess.Popen):
        self.process = popen(['bluetoothctl', '--agent', 'NoInputNoOutput', '--timeout', '50'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def send(self, command):
        if self.process.poll() is not None: raise RuntimeError('Bluetooth discovery client exited unexpectedly')
        self.process.stdin.write((command+'\n').encode('ascii')); self.process.stdin.flush()

    def read(self, timeout):
        if not self.selector.select(timeout): return ''
        payload = os.read(self.process.stdout.fileno(), 8192)
        if not payload: raise RuntimeError('Bluetooth discovery client closed its output')
        return clean_terminal(payload.decode('utf-8', errors='replace'))

    def close(self):
        try:
            if self.process.poll() is None:
                try: self.send('scan off'); self.send('quit')
                except (OSError, RuntimeError): pass
                try: self.process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try: self.process.wait(timeout=1.)
                    except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=1.)
        finally:
            self.selector.close()
            self.process.stdin.close(); self.process.stdout.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class BluetoothSetup:
    def __init__(self, *, runner=run_command, input_resolver=resolve_input, clock=time.monotonic, sleeper=time.sleep,
                 session_factory=BluetoothCtlSession):
        self.run = runner; self.input_resolver = input_resolver
        self.clock = clock; self.sleep = sleeper
        self.session_factory = session_factory

    def scalar_properties(self, path, interface, properties):
        output = self.run(['busctl', '--system', '--auto-start=no', '--timeout=2s', 'get-property',
                           'org.bluez', path, interface, *properties], timeout=3.)
        lines = output.strip().splitlines()
        if len(lines)!=len(properties): raise ValueError('Incomplete BlueZ scalar properties')
        result = {}
        for name, line in zip(properties, lines):
            fields = shlex.split(line)
            if len(fields)!=2: raise ValueError('Malformed BlueZ scalar property '+name)
            kind, value = fields
            if kind=='b' and value in ('true', 'false'): value = value=='true'
            elif kind!='s': raise ValueError('Unexpected BlueZ property type for '+name)
            result[name] = dict(type=kind, data=value)
        return result

    def objects(self, mac=None, adapter='hci0'):
        # systemd249 busctl JSON cannot serialize ManufacturerData's uint16 dict
        # keys. Enumerate paths and request only scalar properties, never GetAll.
        paths = self.run(['busctl', '--system', '--auto-start=no', '--timeout=2s', '--list', 'tree', 'org.bluez'], timeout=3.)
        objects = {}; target = None if mac is None else f'/org/bluez/{adapter}/dev_{mac.replace(":", "_")}'
        for path in paths.splitlines():
            path = path.strip()
            if re.fullmatch(r'/org/bluez/hci[0-9]+', path):
                objects[path] = {'org.bluez.Adapter1': self.scalar_properties(path, 'org.bluez.Adapter1', ('Address', 'Powered'))}
            elif re.fullmatch(r'/org/bluez/hci[0-9]+/dev_(?:[0-9A-Fa-f]{2}_){5}[0-9A-Fa-f]{2}', path):
                props = dict(Address=dict(type='s', data=path.rsplit('dev_', 1)[1].replace('_', ':')))
                if path==target:
                    props = self.scalar_properties(path, 'org.bluez.Device1', ('Address', 'Alias', 'Paired', 'Connected', 'ServicesResolved', 'Trusted'))
                objects[path] = {'org.bluez.Device1': props}
        return objects

    @staticmethod
    def properties(obj, interface):
        return {key: value['data'] for key, value in obj.get(interface, {}).items()}

    def snapshot(self, mac=None, adapter='hci0'):
        adapter = adapter_name(adapter)
        mac = None if mac is None else address(mac)
        objects = self.objects(mac, adapter); adapters = []; devices = []
        for path, obj in objects.items():
            if 'org.bluez.Adapter1' in obj:
                adapters.append(dict(path=path, **self.properties(obj, 'org.bluez.Adapter1')))
            if 'org.bluez.Device1' in obj:
                props = self.properties(obj, 'org.bluez.Device1')
                devices.append(dict(path=path, **props))
        report = dict(adapters=adapters, devices=devices, ready=False, address=mac, adapter=adapter)
        if mac is None:
            report['reason'] = 'No controller selected; select or pair an exact MAC address.'
            return report
        mac = address(mac); report['address'] = mac
        path = f'/org/bluez/{adapter}/dev_{mac.replace(":", "_")}'
        target = next((device for device in devices if device['path']==path and device.get('Address', '').upper()==mac), None)
        report['device'] = target
        if target is None:
            report['reason'] = 'Selected device is not known to this adapter; discover then pair it.'
            return report
        name = str(target.get('Name', target.get('Alias', '')))
        report['gamepad_name_hint'] = bool(re.search(r'xbox|gamepad|controller', name, re.I))
        # Names are diagnostics, never identity or authorization.
        reasons = [f'{flag} is not true' for flag in ('Paired', 'Connected', 'ServicesResolved') if target.get(flag) is not True]
        adapter_props = next((item for item in adapters if item['path']==f'/org/bluez/{adapter}'), {})
        if adapter_props.get('Powered') is not True: reasons.append('Adapter Powered is not true')
        try:
            report['joystick'] = self.input_resolver(mac)
        except (OSError, ValueError) as exc:
            report['joystick'] = None; reasons.append(str(exc))
        report['ready'] = not reasons
        report['reason'] = '; '.join(reasons) if reasons else 'Bluetooth radio link and local input identity verified; run the input probe to validate button mapping.'
        return report

    def _cli_adapter(self, adapter):
        report = self.snapshot(adapter=adapter)
        adapters = report['adapters']
        if len(adapters)!=1 or adapters[0]['path']!=f'/org/bluez/{adapter}':
            raise RuntimeError('bluetoothctl pairing/discovery requires exactly one matching adapter; no adapter selection is changed automatically')
        if adapters[0].get('Powered') is not True:
            raise RuntimeError('Bluetooth adapter is not powered; helper will not change global adapter power')

    def discover(self, adapter, seconds):
        if not 1<=seconds<=30: raise ValueError('Discovery must be bounded to 1..30 seconds')
        self._cli_adapter(adapter)
        transcript = self.run(['bluetoothctl', '--timeout', str(seconds), 'scan', 'on'], timeout=seconds+3)
        result = dict(adapter=adapter, observed_devices=observed_devices(transcript), ready=False,
                      reason='Scan observations retained; pair an exact MAC while the controller remains in pairing mode.')
        try: result['current_state'] = self.snapshot(adapter=adapter)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            result['status_error'] = str(exc)
        return result

    def _wait_client(self, client, predicate, seconds, description):
        deadline = self.clock()+seconds; transcript = ''
        while self.clock()<deadline:
            transcript = (transcript+client.read(min(.2, max(0., deadline-self.clock()))))[-262144:]
            if re.search(r'Failed to (?:start discovery|pair)|AuthenticationFailed|AuthenticationCanceled|No default controller available', transcript):
                raise RuntimeError(transcript[-2000:])
            if predicate(transcript): return transcript
        raise RuntimeError(f'Timed out {description}; keep the selected controller in pairing mode. '+transcript[-1000:])

    def wait_for(self, mac, adapter, predicate, seconds):
        deadline = self.clock()+seconds
        while True:
            report = self.snapshot(mac, adapter)
            if predicate(report): return report
            if self.clock()>=deadline: raise RuntimeError(report['reason'])
            self.sleep(.2)

    def pair(self, mac, adapter):
        mac = address(mac); adapter = adapter_name(adapter)
        self._cli_adapter(adapter)
        report = self.snapshot(mac, adapter)
        if not report.get('device') or report['device'].get('Paired') is not True:
            with self.session_factory() as client:
                client.send('scan on')
                client.send('devices')
                self._wait_client(client, lambda text: bool(re.search(r'\bDevice\s+'+re.escape(mac)+r'\b', text, re.I)), 10., 'discovering '+mac)
                # Another operator may have paired it while discovery was running.
                current = self.snapshot(mac, adapter)
                if not current.get('device') or current['device'].get('Paired') is not True:
                    client.send('pair '+mac)
                    self._wait_client(client, lambda text: 'Pairing successful' in text or bool(re.search(
                        r'Device\s+'+re.escape(mac)+r'\s+Paired:\s+yes\b', text, re.I)), 30., 'pairing '+mac)
                self.wait_for(mac, adapter, lambda r: bool(r.get('device') and r['device'].get('Paired') is True), 3.)
                return self._trust_paired(mac, adapter)
        # Never invoke pair on an existing bond: newer bluetoothctl may remove it.
        return self._trust_paired(mac, adapter)

    def _trust_paired(self, mac, adapter):
        path = f'/org/bluez/{adapter}/dev_{mac.replace(":", "_")}'
        self.run(['busctl', '--system', '--auto-start=no', 'set-property', 'org.bluez', path, 'org.bluez.Device1', 'Trusted', 'b', 'true'])
        return self.wait_for(mac, adapter, lambda r: bool(r.get('device') and r['device'].get('Paired') is True and r['device'].get('Trusted') is True), 3.)

    def connection(self, mac, adapter, connect):
        mac = address(mac); adapter = adapter_name(adapter)
        report = self.snapshot(mac, adapter)
        target = report.get('device')
        if target is None: raise RuntimeError(report['reason'])
        if connect and target.get('Paired') is not True:
            raise RuntimeError('Run explicit pair for this MAC first; connect does not change pairing/trust')
        method = 'Connect' if connect else 'Disconnect'
        self.run(['busctl', '--system', '--auto-start=no', '--timeout=15s', 'call', 'org.bluez', target['path'], 'org.bluez.Device1', method], timeout=18)
        return self.wait_for(mac, adapter, lambda r: r['ready'] if connect else (not r.get('device') or r['device'].get('Connected') is False), 5.)


def load_selection(path):
    if not path.exists(): return None, 'hci0'
    data = json.loads(path.read_text())
    if type(data.get('schema_version')) is not int or data['schema_version']!=1:
        raise ValueError('Unsupported Bluetooth selection schema')
    return address(data['address']), adapter_name(data.get('adapter', 'hci0'))


def save_selection(path, mac, adapter):
    data = dict(schema_version=1, address=address(mac), adapter=adapter_name(adapter))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n'); temporary.replace(path)


def main(argv=None, *, setup=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--adapter', help='Exact BlueZ adapter, default selected adapter or hci0')
    commands = parser.add_subparsers(dest='command')
    for command in ('status', 'list'):
        commands.add_parser(command)
    discover = commands.add_parser('discover'); discover.add_argument('--seconds', type=int, default=10)
    for command in ('pair', 'select', 'connect', 'disconnect'):
        commands.add_parser(command).add_argument('address')
    args = parser.parse_args(argv)
    setup = setup or BluetoothSetup()
    try:
        selected, saved_adapter = load_selection(args.config)
        adapter = adapter_name(args.adapter or saved_adapter)
        command = args.command or 'status'
        if command in ('status', 'list'):
            result = setup.snapshot(selected if command=='status' else None, adapter)
        elif command=='discover': result = setup.discover(adapter, args.seconds)
        else:
            mac = address(args.address)
            if command=='pair':
                print('Hold the selected controller pairing button until it flashes before pairing.', file=sys.stderr)
                result = setup.pair(mac, adapter)
                save_selection(args.config, mac, adapter)
            elif command=='select':
                result = setup.snapshot(mac, adapter)
                if not result.get('device') or result['device'].get('Paired') is not True:
                    raise RuntimeError('Select requires an already paired exact MAC; use pair first')
                save_selection(args.config, mac, adapter)
            else: result = setup.connection(mac, adapter, command=='connect')
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps(dict(error=str(exc), ready=False)), file=sys.stderr)
        return 1


if __name__=='__main__': raise SystemExit(main())
