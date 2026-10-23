"""Offline BlueZ setup tests; no subprocess or radio is used."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('xbox_bluetooth_setup', ROOT/'scripts/xbox_bluetooth.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
MAC = 'AA:BB:CC:DD:EE:FF'
PATH = '/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF'


def variants(properties):
    return {key: dict(type='b' if isinstance(value, bool) else 's', data=value) for key, value in properties.items()}


class BlueZFixture:
    def __init__(self, paired=True):
        self.calls = []
        self.visible = True
        self.props = dict(Address=MAC, Name='Xbox Wireless Controller', Paired=paired,
                          Connected=True, ServicesResolved=True, Trusted=False)
        self.adapters = {'/org/bluez/hci0': {'org.bluez.Adapter1': variants(dict(Address='00:11:22:33:44:55', Powered=True))}}
    def __call__(self, args, timeout=5.):
        self.calls.append(args)
        if 'tree' in args:
            return '\n'.join([*self.adapters, *([PATH] if self.visible else [])])+'\n'
        if 'get-property' in args:
            index = args.index('get-property'); path, interface = args[index+2:index+4]
            if interface=='org.bluez.Device1': props = {**self.props, 'Alias': self.props['Name']}
            else: props = {key: value['data'] for key, value in self.adapters[path][interface].items()}
            return '\n'.join('b '+str(props[key]).lower() if isinstance(props[key], bool) else 's '+json.dumps(props[key]) for key in args[index+4:])+'\n'
        if 'pair' in args: self.props['Paired'] = True
        elif 'set-property' in args: self.props['Trusted'] = True
        elif args[-1]=='Disconnect': self.props['Connected'] = False
        elif args[-1]=='Connect': self.props['Connected'] = True
        elif args[-2:]==['scan', 'on']: pass
        else: raise AssertionError(args)
        return ''
    @property
    def mutations(self): return [args for args in self.calls if 'tree' not in args and 'get-property' not in args]


class FakeDiscoverySession:
    def __init__(self, fixture, fail_pair=False):
        self.fixture = fixture; self.commands = []; self.queue = []
        self.closed = False; self.fail_pair = fail_pair
    def send(self, command):
        self.commands.append(command)
        if command=='scan on':
            self.fixture.visible = True
            self.queue.append('[NEW] Device '+MAC+' Xbox Wireless Controller\n')
        elif command=='pair '+MAC:
            if self.fail_pair: self.queue.append('Failed to pair: org.bluez.Error.AuthenticationFailed\n')
            else:
                self.fixture.props['Paired'] = True
                self.queue.append('Pairing successful\n')
    def read(self, timeout): return self.queue.pop(0) if self.queue else ''
    def __enter__(self): return self
    def __exit__(self, *args):
        self.closed = True
        self.fixture.visible = self.fixture.props['Paired']


class BluetoothSetupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = BlueZFixture()
        self.session = FakeDiscoverySession(self.fixture)
        self.setup = helper.BluetoothSetup(runner=self.fixture, input_resolver=lambda mac: '/dev/input/js2',
                                          session_factory=lambda: self.session)

    def test_default_status_has_no_mutations_or_configuration_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            config = Path(folder)/'selection.json'
            with contextlib.redirect_stdout(io.StringIO()):
                result = helper.main(['--config', str(config)], setup=self.setup)
            self.assertEqual(result, 0); self.assertFalse(config.exists())
        self.assertEqual(self.fixture.mutations, [])
        self.assertTrue(self.fixture.calls)
        self.assertTrue(all('--auto-start=no' in args for args in self.fixture.calls))
        self.assertTrue(all('GetManagedObjects' not in args and 'GetAll' not in args and '--json=short' not in args for args in self.fixture.calls))

    def test_status_requires_bluez_flags_and_input_identity_not_display_name(self):
        self.fixture.props['Name'] = 'Unhelpful device name'
        report = self.setup.snapshot(MAC)
        self.assertTrue(report['ready']); self.assertFalse(report['gamepad_name_hint'])
        # Trusted is diagnostic; a user-selected existing bond remains usable.
        self.assertFalse(report['device']['Trusted'])
        for flag in ('Paired', 'Connected', 'ServicesResolved'):
            self.fixture.props[flag] = False
            self.assertFalse(self.setup.snapshot(MAC)['ready'])
            self.fixture.props[flag] = True
        self.setup.input_resolver = lambda mac: (_ for _ in ()).throw(ValueError('No exact input identity'))
        self.assertFalse(self.setup.snapshot(MAC)['ready'])
        self.assertEqual(self.fixture.mutations, [])

    def test_scalar_snapshot_ignores_manufacturer_numeric_dictionary(self):
        # This BlueZ property broke systemd249's JSON GetManagedObjects path.
        self.fixture.props['ManufacturerData'] = {6: [0, 1, 255]}
        report = self.setup.snapshot(MAC)
        self.assertTrue(report['ready'])
        self.assertTrue(all('ManufacturerData' not in args and 'GetManagedObjects' not in args and 'GetAll' not in args for args in self.fixture.calls))

    def test_pair_preserves_existing_bond_and_trusts_only_selected_device(self):
        report = self.setup.pair(MAC.lower(), 'hci0')
        self.assertTrue(report['device']['Trusted'])
        self.assertEqual(self.fixture.mutations, [['busctl', '--system', '--auto-start=no', 'set-property', 'org.bluez', PATH,
                                                 'org.bluez.Device1', 'Trusted', 'b', 'true']])

    def test_new_pair_uses_bounded_agent_and_exact_mac_then_records_verified_selection(self):
        self.fixture.props['Paired'] = False
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            config = Path(folder)/'selection.json'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = helper.main(['--config', str(config), 'pair', MAC.lower()], setup=self.setup)
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(config.read_text()), dict(schema_version=1, address=MAC, adapter='hci0'))
        self.assertEqual(self.session.commands, ['scan on', 'devices', 'pair '+MAC])
        self.assertTrue(self.session.closed)

    def test_select_already_paired_device_writes_only_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            config = Path(folder)/'selection.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(helper.main(['--config', str(config), 'select', MAC], setup=self.setup), 0)
            self.assertEqual(helper.load_selection(config), (MAC, 'hci0'))
        self.assertEqual(self.fixture.mutations, [])

    def test_connect_disconnect_call_only_exact_object(self):
        self.assertFalse(self.setup.connection(MAC, 'hci0', False)['device']['Connected'])
        self.assertTrue(self.setup.connection(MAC, 'hci0', True)['ready'])
        self.assertEqual([args[-1] for args in self.fixture.mutations], ['Disconnect', 'Connect'])
        self.assertTrue(all(PATH in args for args in self.fixture.mutations))
        self.assertTrue(all('--auto-start=no' in args for args in self.fixture.calls))

    def test_discovery_bounded_and_does_not_toggle_global_adapter_settings(self):
        for seconds in (0, 31):
            with self.assertRaises(ValueError): self.setup.discover('hci0', seconds)
        self.setup.discover('hci0', 2)
        self.assertEqual(self.fixture.mutations, [['bluetoothctl', '--timeout', '2', 'scan', 'on']])

    def test_invalid_address_or_adapter_rejected_before_mutations(self):
        for invalid in ('AA:BB', 'AA:BB:CC:DD:EE:FF;power off', '*', 'xbox'):
            with self.assertRaises(ValueError): self.setup.pair(invalid, 'hci0')
        with self.assertRaises(ValueError): self.setup.pair(MAC, 'hci0/../hci1')
        self.assertEqual(self.fixture.mutations, [])

    def test_multiple_or_unpowered_adapters_fail_without_mutation(self):
        self.fixture.adapters['/org/bluez/hci1'] = {'org.bluez.Adapter1': variants(dict(Address='00:11:22:33:44:56', Powered=True))}
        with self.assertRaisesRegex(RuntimeError, 'exactly one'): self.setup.discover('hci0', 2)
        del self.fixture.adapters['/org/bluez/hci1']
        self.fixture.adapters['/org/bluez/hci0']['org.bluez.Adapter1'] = variants(dict(Address='00:11:22:33:44:55', Powered=False))
        with self.assertRaisesRegex(RuntimeError, 'not powered'): self.setup.pair(MAC, 'hci0')
        self.assertEqual(self.fixture.mutations, [])

    def test_pair_failure_does_not_save_selection_or_claim_success(self):
        self.fixture.props['Paired'] = False
        self.session.fail_pair = True
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            config = Path(folder)/'selection.json'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(helper.main(['--config', str(config), 'pair', MAC], setup=self.setup), 1)
            self.assertFalse(config.exists())
        self.assertEqual(self.fixture.mutations, [])
        self.assertTrue(self.session.closed)

    def test_scan_retains_observations_when_device_removed_or_status_fails(self):
        original = self.setup.run
        def scan(args, timeout=5.):
            if args[-2:]==['scan', 'on']:
                self.fixture.visible = False
                return '\x1b[0;92m[NEW]\x1b[0m Device '+MAC+' Xbox Wireless Controller\r\n[CHG] Device '+MAC+' RSSI: -55\n'
            return original(args, timeout)
        self.setup.run = scan
        report = self.setup.discover('hci0', 2)
        self.assertEqual(report['observed_devices'], [dict(address=MAC, name='Xbox Wireless Controller')])
        self.assertEqual(report['current_state']['devices'], [])
        with patch.object(self.setup, 'snapshot', side_effect=[dict(adapters=[dict(path='/org/bluez/hci0', Powered=True)]), RuntimeError('busctl exit -6')]):
            report = self.setup.discover('hci0', 2)
        self.assertEqual(report['observed_devices'][0]['address'], MAC)
        self.assertIn('exit -6', report['status_error'])

    def test_pair_rediscovers_missing_unpaired_object_with_same_client_until_verified(self):
        self.fixture.visible = False; self.fixture.props['Paired'] = False
        original = self.setup.run
        def ensure_alive(args, timeout=5.):
            if 'set-property' in args:
                self.assertFalse(self.session.closed)
                self.assertTrue(self.fixture.visible)
            return original(args, timeout)
        self.setup.run = ensure_alive
        report = self.setup.pair(MAC, 'hci0')
        self.assertTrue(report['device']['Paired']); self.assertTrue(report['device']['Trusted'])
        self.assertEqual(self.session.commands, ['scan on', 'devices', 'pair '+MAC])
        self.assertTrue(self.session.closed)

    def test_wrong_discovered_mac_never_gets_pair_command_and_timeout_closes_client(self):
        self.fixture.visible = False; self.fixture.props['Paired'] = False
        tick = iter(range(0, 100))
        self.setup.clock = lambda: next(tick)
        original = self.session.send
        def wrong(command):
            original(command)
            if command=='scan on': self.session.queue[:] = ['[NEW] Device 11:22:33:44:55:66 Other Controller\n']
        self.session.send = wrong
        with self.assertRaisesRegex(RuntimeError, 'Timed out discovering'):
            self.setup.pair(MAC, 'hci0')
        self.assertEqual(self.session.commands, ['scan on', 'devices'])
        self.assertTrue(self.session.closed)

    def test_owned_subprocess_client_cleanup_on_exception(self):
        processes = []
        def local_process(args, **kwargs):
            self.assertEqual(args, ['bluetoothctl', '--agent', 'NoInputNoOutput', '--timeout', '50'])
            process = subprocess.Popen([sys.executable, '-u', '-c',
                'import sys\nfor line in sys.stdin:\n print(line.strip(), flush=True)\n if line.strip()=="quit": break'], **kwargs)
            processes.append(process); return process
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with helper.BluetoothCtlSession(popen=local_process) as client:
                client.send('scan on')
                self.assertIn('scan on', client.read(2.))
                raise RuntimeError('test failure')
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed); self.assertTrue(processes[0].stdout.closed)


if __name__=='__main__': unittest.main()
