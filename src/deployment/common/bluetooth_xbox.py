"""Pinned Bluetooth Xbox input with a bounded, read-only BlueZ monitor.

This module never pairs, connects, trusts, powers, or restarts Bluetooth. A
background busctl query observes BlueZ; the policy thread performs no D-Bus or
subprocess calls. A joystick event stream is not used as a radio heartbeat.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time

from deployment.common.resilient_xbox import ResilientLocalXboxController
from deployment.common.xbox_joystick import LinuxJoystick


def canonical_mac(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', value):
        raise ValueError('A complete Bluetooth MAC address is required')
    return value.upper()


def _adapter(value):
    if not isinstance(value, str) or not re.fullmatch(r'hci[0-9]+', value):
        raise ValueError('Bluetooth adapter must be an exact hciN name')
    return value


def load_bluetooth_config(path):
    config = json.loads(Path(path).read_text())
    if (not isinstance(config, dict) or set(config) - {'schema_version', 'address', 'adapter'}
            or type(config.get('schema_version')) is not int or config['schema_version'] != 1):
        raise ValueError('Unsupported pinned Bluetooth controller configuration')
    return dict(schema_version=1, address=canonical_mac(config.get('address')),
                adapter=_adapter(config.get('adapter', 'hci0')))


@dataclass(frozen=True)
class BluetoothInputIdentity:
    joystick_path: Path
    device_number: int
    sysfs_device: Path
    mac: str


def resolve_bluetooth_joystick(mac, *, sys_class_input=Path('/sys/class/input'),
                               dev_input=Path('/dev/input'), file_stat=os.stat):
    """Require exactly one Bluetooth js node whose kernel uniq is the pinned MAC."""
    mac = canonical_mac(mac)
    matches = []
    for entry in Path(sys_class_input).glob('js[0-9]*'):
        if not re.fullmatch(r'js[0-9]+', entry.name):
            continue
        device = entry / 'device'
        try:
            address = canonical_mac((device / 'uniq').read_text().strip())
            bus = int((device / 'id/bustype').read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        if address != mac or bus != 0x0005:
            continue
        path = Path(dev_input) / entry.name
        info = file_stat(path)
        if not stat.S_ISCHR(info.st_mode):
            raise OSError(f'Matching joystick is not a character device: {path}')
        matches.append(BluetoothInputIdentity(path, info.st_rdev, device.resolve(), mac))
    if len(matches) != 1:
        raise OSError(f'Expected one Bluetooth joystick for {mac}; found {len(matches)}')
    return matches[0]


@dataclass(frozen=True)
class BluetoothSnapshot:
    address: str
    adapter: str
    observed_at: float
    service_available: bool = False
    paired: bool = False
    connected: bool = False
    services_resolved: bool = False
    trusted: bool = False
    identity: BluetoothInputIdentity | None = None
    reason: str = 'Waiting for BlueZ monitor'
    generation: int = 0
    adapter_powered: bool = False

    @property
    def ready(self):
        return (self.service_available and self.adapter_powered and self.paired and self.connected
                and self.services_resolved and self.identity is not None)


def parse_bluez_snapshot(payload, *, address, adapter, observed_at,
                         resolver=resolve_bluetooth_joystick):
    """Parse systemd busctl's typed GetManagedObjects JSON, rejecting ambiguity."""
    address, adapter = canonical_mac(address), _adapter(adapter)
    document = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if (not isinstance(document, dict) or document.get('type') != 'a{oa{sa{sv}}}'
            or not isinstance(document.get('data'), list) or len(document['data']) != 1
            or not isinstance(document['data'][0], dict)):
        raise ValueError('Malformed BlueZ managed-object snapshot')
    objects = document['data'][0]
    path = f'/org/bluez/{adapter}/dev_' + address.replace(':', '_')
    props = objects.get(path, {}).get('org.bluez.Device1')
    snapshot = BluetoothSnapshot(address, adapter, observed_at, service_available=True)
    adapter_props = objects.get(f'/org/bluez/{adapter}', {}).get('org.bluez.Adapter1')
    if not isinstance(adapter_props, dict):
        return replace(snapshot, reason='Pinned Bluetooth adapter is not present in BlueZ')
    powered = adapter_props.get('Powered')
    if not isinstance(powered, dict) or powered.get('type') != 'b' or type(powered.get('data')) is not bool:
        raise ValueError('Missing or invalid Adapter1.Powered')
    snapshot = replace(snapshot, adapter_powered=powered['data'])
    if not snapshot.adapter_powered:
        return replace(snapshot, reason='Bluetooth adapter is powered off')
    if not isinstance(props, dict):
        return replace(snapshot, reason='Pinned Bluetooth device is not present in BlueZ')
    def prop(name, signature, kind):
        value = props.get(name)
        if (not isinstance(value, dict) or value.get('type') != signature
                or type(value.get('data')) is not kind):
            raise ValueError(f'Missing or invalid Device1.{name}')
        return value['data']
    if canonical_mac(prop('Address', 's', str)) != address:
        raise ValueError('BlueZ device address does not match pinned MAC')
    if prop('Adapter', 'o', str) != f'/org/bluez/{adapter}':
        raise ValueError('BlueZ device adapter does not match pinned adapter')
    snapshot = replace(snapshot, paired=prop('Paired', 'b', bool),
                       connected=prop('Connected', 'b', bool),
                       services_resolved=prop('ServicesResolved', 'b', bool),
                       trusted=prop('Trusted', 'b', bool))
    for value, reason in ((snapshot.paired, 'Bluetooth device is not paired'),
                          (snapshot.connected, 'Bluetooth radio link disconnected'),
                          (snapshot.services_resolved, 'Bluetooth services are not resolved')):
        if not value:
            return replace(snapshot, reason=reason)
    try:
        identity = resolver(address)
    except (OSError, ValueError) as error:
        return replace(snapshot, reason=f'Bluetooth joystick unavailable: {error}')
    if identity.mac != address:
        raise ValueError('Resolved joystick does not match pinned MAC')
    return replace(snapshot, identity=identity, reason='Bluetooth radio and matching joystick ready')


class BlueZMonitor:
    """Poll pinned scalar properties in two bounded calls off the control thread.

    Once BlueZ exposes a change, polling normally sees it after a 100ms interval
    plus a query bounded by a shared 500ms deadline. Physical RF-loss detection also depends on Bluetooth
    link supervision: a 500ms cache-age limit is NOT an RF-loss detection bound.
    Polling can miss transitions shorter than the observation period.

    Never request GetManagedObjects/GetAll: systemd249 cannot JSON-encode some
    ManufacturerData dictionaries, including those on unrelated nearby devices.
    """
    def __init__(self, address, *, adapter='hci0', interval=.1, query_timeout=.2,
                 max_query_age=.5,
                 resolver=resolve_bluetooth_joystick, run=subprocess.run,
                 clock=time.monotonic):
        self.address, self.adapter = canonical_mac(address), _adapter(adapter)
        if not all(math.isfinite(x) and x > 0 for x in (interval, query_timeout, max_query_age)):
            raise ValueError('Monitor interval and timeout must be finite and positive')
        self.interval, self.query_timeout = interval, query_timeout
        self.max_query_age = max_query_age
        self.resolver, self.run, self.clock = resolver, run, clock
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._thread = None
        self._snapshot = None
        self._generation = 0
        self._token = None

    def _probe(self):
        observed = self.clock()  # Query start: a slow response is not newly fresh.
        try:
            deadline = observed + self.max_query_age
            def query(path, interface, names, signatures):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError('BlueZ scalar snapshot exceeded its shared deadline')
                result = self.run(['busctl', '--system', '--auto-start=no',
                    f'--timeout={min(self.query_timeout, remaining)}s', '--json=short',
                    'get-property', 'org.bluez', path, interface, *names],
                    capture_output=True, text=True, check=True,
                    timeout=min(self.query_timeout + .1, remaining))
                values = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
                if len(values) != len(names):
                    raise ValueError('Incomplete BlueZ scalar property response')
                for value, signature in zip(values, signatures):
                    expected = bool if signature == 'b' else str
                    if (not isinstance(value, dict) or value.get('type') != signature
                            or type(value.get('data')) is not expected):
                        raise ValueError('Invalid typed BlueZ scalar property')
                if self.clock() > deadline:
                    raise TimeoutError('BlueZ scalar snapshot exceeded its shared deadline')
                return dict(zip(names, values))
            adapter_path = f'/org/bluez/{self.adapter}'
            device_path = adapter_path + '/dev_' + self.address.replace(':', '_')
            powered = query(adapter_path, 'org.bluez.Adapter1', ['Powered'], ['b'])
            if not powered['Powered']['data']:
                return BluetoothSnapshot(self.address, self.adapter, observed,
                    service_available=True, reason='Bluetooth adapter is powered off')
            try:
                properties = query(device_path, 'org.bluez.Device1',
                    ['Address', 'Adapter', 'Paired', 'Connected', 'ServicesResolved', 'Trusted'],
                    ['s', 'o', 'b', 'b', 'b', 'b'])
            except subprocess.CalledProcessError as error:
                detail = str(error.stderr or '').lower()
                missing_get = ('method "get" with signature "ss" on interface '
                               '"org.freedesktop.dbus.properties" doesn\'t exist') in detail
                if missing_get or any(marker in detail for marker in
                       ('unknown object', 'unknownobject', 'no such interface')):
                    return BluetoothSnapshot(self.address, self.adapter, observed,
                        service_available=True, adapter_powered=True,
                        reason='Pinned Bluetooth device is not present in BlueZ')
                raise
            payload = {'type': 'a{oa{sa{sv}}}', 'data': [{
                adapter_path: {'org.bluez.Adapter1': powered},
                device_path: {'org.bluez.Device1': properties}}]}
            return parse_bluez_snapshot(payload, address=self.address,
                adapter=self.adapter, observed_at=observed, resolver=self.resolver)
        except Exception as error:
            return BluetoothSnapshot(self.address, self.adapter, observed,
                                     reason=f'BlueZ monitor unavailable: {error}')

    def _loop(self):
        while not self._stop.is_set():
            snapshot = self._probe()
            token = (snapshot.ready, snapshot.identity)
            with self._condition:
                if token != self._token:
                    self._generation += 1
                    self._token = token
                self._snapshot = replace(snapshot, generation=self._generation)
                self._condition.notify_all()
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is not None:
            raise RuntimeError('BlueZ monitor already started')
        self._thread = threading.Thread(target=self._loop, name='xbox-bluez', daemon=True)
        self._thread.start()

    def snapshot(self):
        with self._condition:
            return self._snapshot

    def wait_first(self, timeout):
        with self._condition:
            self._condition.wait_for(lambda: self._snapshot is not None, timeout)
            return self._snapshot

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.max_query_age + .2)


class BluetoothXboxController(ResilientLocalXboxController):
    """Accept only fresh BlueZ status and the joystick for one pinned MAC."""
    def __init__(self, address, *, adapter='hci0', monitor=None, max_status_age=.5,
                 startup_timeout=30., reconnect_interval=1., clock=time.monotonic,
                 joystick_factory=LinuxJoystick, state_factory=None, fstat=os.fstat,
                 resolver=resolve_bluetooth_joystick):
        self.address, self.adapter = canonical_mac(address), _adapter(adapter)
        if not math.isfinite(max_status_age) or max_status_age <= 0:
            raise ValueError('max_status_age must be finite and positive')
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError('startup_timeout must be finite and positive')
        if state_factory is None:
            from deployment.common.xbox_linux_mapping import build_state_for_joystick
            state_factory = build_state_for_joystick
        self._monitor = monitor or BlueZMonitor(self.address, adapter=self.adapter, clock=clock)
        self._status_age = max_status_age
        self._bluetooth_clock = clock
        self._device_factory = joystick_factory
        self._fstat = fstat
        self._resolver = resolver
        self._seen_generation = None
        self._radio_closed = False
        self._monitor.start()
        try:
            deadline = time.monotonic() + startup_timeout
            self._monitor.wait_first(startup_timeout)
            while True:
                snapshot, reason = self._status()
                if reason is None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OSError(reason)
                time.sleep(min(.05, remaining))
            super().__init__(snapshot.identity.joystick_path, reconnect_interval=reconnect_interval,
                             joystick_factory=self._open_matching, state_factory=state_factory,
                             clock=clock)
            self._seen_generation = snapshot.generation
        except BaseException:
            self._monitor.close()
            raise

    def _status(self):
        snapshot = self._monitor.snapshot()
        if snapshot is None:
            return None, 'BlueZ monitor has not produced a snapshot'
        age = self._bluetooth_clock() - snapshot.observed_at
        if snapshot.address != self.address or snapshot.adapter != self.adapter:
            return snapshot, 'BlueZ monitor identity mismatch'
        if not math.isfinite(age) or not 0 <= age <= self._status_age:
            return snapshot, 'BlueZ monitor snapshot stale'
        if not snapshot.ready:
            return snapshot, snapshot.reason
        return snapshot, None

    def _require_ready(self):
        snapshot, reason = self._status()
        if reason is not None:
            raise OSError(reason)
        return snapshot

    def _open_matching(self, _path):
        snapshot = self._require_ready()
        identity = snapshot.identity
        if self._resolver(self.address) != identity:
            raise OSError('Bluetooth joystick sysfs identity changed before open')
        joystick = self._device_factory(identity.joystick_path)
        try:
            info = self._fstat(joystick.descriptor)
            if not stat.S_ISCHR(info.st_mode) or info.st_rdev != identity.device_number:
                raise OSError('Opened joystick device identity changed')
            if self._resolver(self.address) != identity:
                raise OSError('Bluetooth joystick sysfs identity changed during open')
            if self._require_ready().generation != snapshot.generation:
                raise OSError('Bluetooth connection changed during joystick open')
        except BaseException:
            joystick.close()
            raise
        return joystick

    def sample(self, timeout=0.):
        snapshot, reason = self._status()
        generation = snapshot.generation if snapshot is not None else None
        if reason is not None or generation != self._seen_generation:
            if self.joystick is not None:
                self._drop(self._bluetooth_clock(), reason or 'Bluetooth connection generation changed')
            self._seen_generation = generation
        # The guarded factory cannot reopen while BlueZ status is unavailable.
        return super().sample(timeout)

    @property
    def mapping_info(self):
        return dict(getattr(self.state, 'mapping_info', {}))

    @property
    def link_status(self):
        result = super().link_status
        snapshot, reason = self._status()
        ready = reason is None and not self._radio_closed
        result.update(connected=bool(ready and result['connected']),
                      controls_enabled=bool(ready and result['controls_enabled']),
                      address=self.address, adapter=self.adapter,
                      radio_connected=bool(snapshot and snapshot.connected),
                      paired=bool(snapshot and snapshot.paired),
                      services_resolved=bool(snapshot and snapshot.services_resolved),
                      adapter_powered=bool(snapshot and snapshot.adapter_powered),
                      trusted=bool(snapshot and snapshot.trusted),
                      monitor_generation=None if snapshot is None else snapshot.generation,
                      joystick_path=None if snapshot is None or snapshot.identity is None else
                          str(snapshot.identity.joystick_path),
                      monitor_age_s=None if snapshot is None else
                          max(0., self._bluetooth_clock() - snapshot.observed_at))
        if reason is not None:
            result['reason'] = reason
        return result

    def close(self):
        if self._radio_closed:
            return
        self._radio_closed = True
        try:
            super().close()
        finally:
            self._monitor.close()


__all__ = ['BluetoothXboxController', 'BlueZMonitor', 'BluetoothSnapshot',
           'BluetoothInputIdentity', 'canonical_mac', 'resolve_bluetooth_joystick',
           'parse_bluez_snapshot', 'load_bluetooth_config']
