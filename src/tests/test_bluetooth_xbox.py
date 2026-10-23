"""Offline BlueZ/device identity/recovery tests. No host Bluetooth is queried."""
from dataclasses import replace
import json
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from deployment.common.bluetooth_xbox import (BluetoothInputIdentity, BluetoothSnapshot,
    BluetoothXboxController, BlueZMonitor, canonical_mac, load_bluetooth_config,
    parse_bluez_snapshot, resolve_bluetooth_joystick)
from deployment.common.controller_input import Button
from deployment.common.xbox_joystick import JS_EVENT_BUTTON, XboxJoystickState

MAC = 'AA:BB:CC:DD:EE:FF'
IDENTITY = BluetoothInputIdentity(Path('/fake/js2'), 42, Path('/fake/input5'), MAC)


def document(**changes):
    props = {'Address': {'type': 's', 'data': MAC},
             'Adapter': {'type': 'o', 'data': '/org/bluez/hci0'}}
    for name in ('Paired', 'Connected', 'ServicesResolved', 'Trusted'):
        props[name] = {'type': 'b', 'data': True}
    for name, value in changes.items(): props[name]['data'] = value
    return {'type': 'a{oa{sa{sv}}}', 'data': {
        '/org/bluez/hci0': {'org.bluez.Adapter1': {'Powered': {'type': 'b', 'data': True}}},
        '/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF': {'org.bluez.Device1': props}}}


def payload(**changes):
    value = document(**changes); value['data'] = [value['data']]
    return value


def scalar_output(args, data=None):
    data = payload() if data is None else data
    start = args.index('get-property')
    props = data['data'][0][args[start + 2]][args[start + 3]]
    return '\n'.join(json.dumps(props[name]) for name in args[start + 4:]) + '\n'


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now


class Monitor:
    def __init__(self, clock):
        self.value = BluetoothSnapshot(MAC, 'hci0', clock(), True, True, True,
                                       True, True, IDENTITY, 'ready', 1, True)
        self.started = 0; self.closed = 0
    def start(self): self.started += 1
    def snapshot(self): return self.value
    def wait_first(self, timeout): return self.value
    def close(self): self.closed += 1


class Device:
    def __init__(self): self.descriptor = 7; self.closed = 0; self.events = []
    def read_ready_events(self, timeout):
        result, self.events = self.events, []
        return result
    def close(self): self.closed += 1


class BluetoothParsingTests(unittest.TestCase):
    def test_typed_snapshot_requires_radio_pairing_services_and_exact_identity(self):
        calls = []
        def resolve(mac): calls.append(mac); return IDENTITY
        result = parse_bluez_snapshot(payload(Trusted=False), address=MAC, adapter='hci0',
                                      observed_at=3., resolver=resolve)
        self.assertTrue(result.ready); self.assertFalse(result.trusted)
        self.assertEqual(calls, [MAC])
        for key in ('Paired', 'Connected', 'ServicesResolved'):
            result = parse_bluez_snapshot(payload(**{key: False}), address=MAC,
                adapter='hci0', observed_at=3., resolver=lambda _: self.fail('No input lookup while radio unavailable'))
            self.assertFalse(result.ready)
        for key, value in [('Address', '00:00:00:00:00:00'), ('Adapter', '/org/bluez/hci1'), ('Connected', 'true')]:
            with self.assertRaises(ValueError):
                parse_bluez_snapshot(payload(**{key: value}), address=MAC,
                    adapter='hci0', observed_at=3., resolver=resolve)

    def test_missing_device_malformed_json_and_missing_input_are_unavailable(self):
        empty = {'type': 'a{oa{sa{sv}}}', 'data': [{}]}
        self.assertFalse(parse_bluez_snapshot(empty, address=MAC, adapter='hci0', observed_at=0).ready)
        for malformed in ('garbage', {}, {'type': 'wrong', 'data': [{}]}):
            with self.assertRaises(ValueError):
                parse_bluez_snapshot(malformed, address=MAC, adapter='hci0', observed_at=0)
        def missing(_): raise OSError('no matching device')
        result = parse_bluez_snapshot(payload(), address=MAC, adapter='hci0', observed_at=0, resolver=missing)
        self.assertFalse(result.ready); self.assertIn('no matching device', result.reason)

    def test_adapter_powered_is_required_even_if_device_still_says_connected(self):
        data = payload()
        data['data'][0]['/org/bluez/hci0']['org.bluez.Adapter1']['Powered']['data'] = False
        result = parse_bluez_snapshot(data, address=MAC, adapter='hci0', observed_at=0,
                                     resolver=lambda _: self.fail('Powered-off adapter must not resolve input'))
        self.assertFalse(result.ready); self.assertIn('powered off', result.reason)

    def test_config_exact_mac_and_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pin.json'
            path.write_text(json.dumps(dict(schema_version=1, address=MAC.lower())))
            self.assertEqual(load_bluetooth_config(path), dict(schema_version=1, address=MAC, adapter='hci0'))
            for change in (dict(address='Xbox'), dict(adapter='hci0;echo'), dict(schema_version=True), dict(extra=True)):
                path.write_text(json.dumps(dict(schema_version=1, address=MAC) | change))
                with self.assertRaises(ValueError): load_bluetooth_config(path)
        with self.assertRaises(ValueError): canonical_mac('AA:BB')

    def test_sysfs_requires_matching_mac_bluetooth_bus_and_unique_character_node(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sysfs = root / 'sys'; dev = root / 'dev'
            def make(name, address, bus):
                device = sysfs / name / 'device'; (device / 'id').mkdir(parents=True)
                (device / 'uniq').write_text(address); (device / 'id/bustype').write_text(bus)
            make('js0', MAC, '0003'); make('js1', '00:00:00:00:00:01', '0005')
            make('js2', MAC.lower(), '0005')
            kwargs = dict(sys_class_input=sysfs, dev_input=dev,
                          file_stat=lambda _: SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=42))
            result = resolve_bluetooth_joystick(MAC, **kwargs)
            self.assertEqual(result.joystick_path, dev / 'js2')
            with self.assertRaises(OSError):
                resolve_bluetooth_joystick(MAC, **(kwargs | {'file_stat': lambda _: SimpleNamespace(st_mode=stat.S_IFREG)}))
            make('js3', MAC, '0005')
            with self.assertRaisesRegex(OSError, 'found 2'): resolve_bluetooth_joystick(MAC, **kwargs)

    def test_monitor_query_is_bounded_read_only_and_keeps_start_time(self):
        clock = Clock(); calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs)); clock.now += .1
            return SimpleNamespace(stdout=scalar_output(args))
        monitor = BlueZMonitor(MAC, run=run, resolver=lambda _: IDENTITY, clock=clock)
        result = monitor._probe()
        self.assertTrue(result.ready); self.assertEqual(result.observed_at, 10.)
        self.assertEqual(len(calls), 2)
        for args, kwargs in calls:
            self.assertIn('--auto-start=no', args); self.assertIn('get-property', args)
            self.assertNotIn('GetManagedObjects', args); self.assertNotIn('GetAll', args)
            self.assertLessEqual(kwargs['timeout'], .31)
            self.assertNotIn('shell', kwargs)
        self.assertIn('/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF', calls[1][0])

    def test_scalar_monitor_rejects_partial_or_wrong_typed_batches(self):
        for bad in ('', '{"type":"b","data":"true"}', '{"type":"b","data":true}\n'*2):
            monitor = BlueZMonitor(MAC, run=lambda *_, **__: SimpleNamespace(stdout=bad))
            self.assertFalse(monitor._probe().service_available)
        def failed(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0], output='{"type":"b","data":true}')
        self.assertFalse(BlueZMonitor(MAC, run=failed)._probe().service_available)

    def test_scalar_queries_share_deadline_and_slow_reply_is_not_fresh(self):
        clock = Clock(); timeouts = []
        def run(args, **kwargs):
            timeouts.append(kwargs['timeout']); clock.now += .3
            return SimpleNamespace(stdout=scalar_output(args))
        monitor = BlueZMonitor(MAC, run=run, resolver=lambda _: IDENTITY, clock=clock)
        result = monitor._probe()
        self.assertFalse(result.service_available)
        self.assertEqual(result.observed_at, 10.)
        self.assertLess(timeouts[1], .201)

    def test_missing_pinned_device_does_not_mean_bluez_service_is_down(self):
        def run(args, **kwargs):
            if 'org.bluez.Device1' in args:
                raise subprocess.CalledProcessError(1, args,
                    stderr='Failed to get property Address on interface org.bluez.Device1: Method "Get" with signature "ss" on interface "org.freedesktop.DBus.Properties" doesn\'t exist\n')
            return SimpleNamespace(stdout=scalar_output(args))
        result = BlueZMonitor(MAC, run=run)._probe()
        self.assertTrue(result.service_available); self.assertTrue(result.adapter_powered)
        self.assertFalse(result.ready); self.assertIn('not present', result.reason)

    def test_monitor_failure_snapshot_and_bounded_lifecycle(self):
        def run(*args, **kwargs): raise subprocess.TimeoutExpired('busctl', .2)
        monitor = BlueZMonitor(MAC, run=run, interval=.01)
        monitor.start()
        try:
            result = monitor.wait_first(.3)
            self.assertIsNotNone(result); self.assertFalse(result.service_available)
            self.assertIn('unavailable', result.reason)
        finally: monitor.close()
        self.assertFalse(monitor._thread.is_alive())


class BluetoothControllerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(); self.monitor = Monitor(self.clock)
        self.devices = []; self.paths = []
        def factory(path):
            self.paths.append(path); device = Device(); self.devices.append(device); return device
        self.factory = factory
        self.controller = None
    def tearDown(self):
        if self.controller is not None: self.controller.close()
    def open(self, **kwargs):
        args = dict(address=MAC, monitor=self.monitor, clock=self.clock,
                    joystick_factory=self.factory, state_factory=lambda _: XboxJoystickState(),
                    startup_timeout=.01, resolver=lambda _: self.monitor.value.identity,
                    fstat=lambda _: SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=42))
        self.controller = BluetoothXboxController(**(args | kwargs))
        return self.controller
    def refresh(self, dt=0, **kwargs):
        self.clock.now += dt
        self.monitor.value = replace(self.monitor.value, observed_at=self.clock(), **kwargs)
    def lose(self):
        self.refresh(connected=False, generation=2, reason='radio disconnected')
        result = self.controller.sample()
        self.assertFalse(result.buttons); self.assertFalse(result.axes)
        self.assertFalse(self.controller.link_status['connected'])
        return result

    def test_radio_disconnect_closes_device_and_requires_new_neutral_and_a(self):
        controller = self.open(); controller.sample(); old = self.devices[-1]
        lost = self.lose(); self.assertEqual(old.closed, 1)
        self.refresh(1.1, connected=True, generation=3)
        neutral = controller.sample()
        self.assertGreater(neutral.sequence, lost.sequence)
        self.assertEqual(len(self.devices), 2)
        self.refresh(.26); controller.sample()
        self.refresh(.26); controller.sample()
        self.assertFalse(controller.link_status['controls_enabled'])
        self.devices[-1].events = [(JS_EVENT_BUTTON, 0, 1)]
        self.assertEqual(controller.sample().buttons, frozenset([Button.A]))
        self.assertTrue(controller.link_status['controls_enabled'])

    def test_service_loss_or_stale_snapshot_drops_even_when_device_read_is_idle(self):
        controller = self.open(); controller.sample()
        self.clock.now += .501
        self.assertFalse(controller.sample().axes)
        self.assertEqual(self.devices[0].closed, 1)
        self.assertIn('stale', controller.link_status['reason'])
        self.refresh(1.1, service_available=False, generation=2, reason='BlueZ service unavailable')
        controller.sample()
        self.assertEqual(len(self.devices), 1)

    def test_fresh_radio_status_needs_no_joystick_event_heartbeat(self):
        controller = self.open()
        for _ in range(10):
            self.refresh(1); controller.sample()
        self.assertTrue(controller.link_status['controls_enabled'])
        self.assertEqual(len(self.devices), 1)

    def test_missed_brief_outage_generation_and_changed_js_node_force_rearm(self):
        controller = self.open(); controller.sample()
        identity = replace(IDENTITY, joystick_path=Path('/fake/js4'))
        self.refresh(identity=identity, generation=3)
        self.assertFalse(controller.sample().buttons)
        self.assertEqual(self.devices[0].closed, 1)
        self.refresh(1.1)
        controller.sample()
        self.assertEqual(self.paths[-1], Path('/fake/js4'))
        self.assertFalse(controller.link_status['controls_enabled'])

    def test_startup_unpaired_and_wrong_opened_device_fail_before_control(self):
        self.monitor.value = replace(self.monitor.value, paired=False, reason='not paired')
        with self.assertRaisesRegex(OSError, 'not paired'): self.open()
        self.assertEqual(self.monitor.closed, 1); self.assertFalse(self.devices)
        self.monitor = Monitor(self.clock)
        with self.assertRaisesRegex(OSError, 'identity changed'):
            self.open(fstat=lambda _: SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=99))
        self.assertEqual(self.devices[0].closed, 1); self.assertEqual(self.monitor.closed, 1)

    def test_mapping_failure_closes_startup_and_neutralizes_failed_reopen(self):
        def bad(_): raise ValueError('unsupported HID mapping')
        with self.assertRaisesRegex(OSError, 'mapping failed'): self.open(state_factory=bad)
        self.assertEqual(self.devices[-1].closed, 1); self.assertEqual(self.monitor.closed, 1)
        self.monitor = Monitor(self.clock)
        calls = []
        def changing(_):
            calls.append(1)
            if len(calls) > 1: raise ValueError('mapping changed')
            return XboxJoystickState()
        controller = self.open(state_factory=changing); self.lose()
        self.refresh(1.1, connected=True, generation=3)
        self.assertFalse(controller.sample().buttons)
        self.assertEqual(self.devices[-1].closed, 1)
        self.assertFalse(controller.link_status['controls_enabled'])

    def test_same_minor_reassignment_is_rejected_by_sysfs_recheck(self):
        calls = []
        def resolver(_):
            calls.append(1)
            return IDENTITY if len(calls) == 1 else replace(IDENTITY, sysfs_device=Path('/fake/input99'))
        with self.assertRaisesRegex(OSError, 'sysfs identity changed during open'):
            self.open(resolver=resolver)
        self.assertEqual(self.devices[-1].closed, 1)
        self.assertEqual(self.monitor.closed, 1)

    def test_startup_waits_for_ready_snapshot_within_deadline(self):
        original = self.monitor.snapshot
        calls = []
        def pending():
            calls.append(1)
            if len(calls) <= 2:
                return replace(original(), connected=False, reason='still connecting')
            return original()
        self.monitor.snapshot = pending
        controller = self.open(startup_timeout=.3)
        self.assertTrue(controller.link_status['connected'])
        self.assertGreaterEqual(len(calls), 3)

    def test_control_samples_never_probe_bluez_and_close_stops_monitor(self):
        controller = self.open()
        for _ in range(20): controller.sample()
        self.assertEqual(self.monitor.started, 1)
        self.assertEqual(len(self.devices), 1)
        controller.close(); controller.close()
        self.assertEqual(self.monitor.closed, 1)
        self.assertFalse(controller.link_status['connected'])


if __name__ == '__main__': unittest.main()
