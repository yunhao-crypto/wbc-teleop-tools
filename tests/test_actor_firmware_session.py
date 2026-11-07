"""Private CLI routing and single-request behavior; no robot or network."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('firmware_session_prepare', ROOT/'scripts/prepare_zenoh_session.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
from deployment.vita_s100_aorta_30dof.actor_pico_startup import FirmwareActions

SOURCE = '''{
  // mode: "router" must remain a comment.
  namespace: "robot-test", mode: 'peer',
  connect: { endpoints: ["tcp/127.0.0.1:7447"] },
  auth: {password: "mode: 'router' // literal", mode: "peer"},
  transport: {shared_memory: {enabled: true,
    transport_optimization: {enabled: true, pool_size: 8388608}}}
}'''


class FirmwareSessionTests(unittest.TestCase):
    def test_client_override_only_edits_root_mode_and_outgoing_shm(self):
        expected = SOURCE.replace("mode: 'peer'", 'mode: "client"').replace(
            'transport_optimization: {enabled: true', 'transport_optimization: {enabled: false')
        self.assertEqual(prepare.patch_config(SOURCE, session_mode='client'), expected)
        self.assertEqual(prepare.patch_config(SOURCE), SOURCE.replace(
            'transport_optimization: {enabled: true', 'transport_optimization: {enabled: false'))

    def test_duplicate_missing_or_invalid_mode_is_rejected(self):
        for source in (SOURCE.replace("mode: 'peer',", ''),
                       SOURCE.replace("mode: 'peer',", "mode:'peer', mode:'client',"),
                       SOURCE.replace("mode: 'peer'", 'mode: true')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                prepare.patch_config(source, session_mode='client')

    def test_private_configs_preserve_vendor_source_and_hide_contents_in_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'vendor.json5'; source.write_text(SOURCE)
            data, cli = root/'data.json5', root/'cli.json5'
            prepare.prepare(source, data)
            receipt = prepare.prepare(source, cli, session_mode='client')
            self.assertEqual(source.read_text(), SOURCE)
            self.assertEqual(data.stat().st_mode & 0o777, 0o600)
            self.assertEqual(cli.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('password', str(receipt))
            self.assertEqual(receipt['session_mode'], 'client')
            with self.assertRaises(ValueError):
                prepare.prepare(source, source, session_mode='client')

    def test_fixed_stand_and_cancel_only_override_child_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp)/'cli.json5'; cli.write_text('{}'); cli.chmod(0o600)
            for action in ('FIXED_STAND', 'PASSIVE'):
                process = Mock(returncode=0)
                process.poll.side_effect = [None, 0, 0]
                process.communicate.return_value = ('success', None)
                monitor = Mock(); clock = Mock(); clock.monotonic.return_value = 0.
                with patch.dict(os.environ, VITA_ACTOR_CLI_SESSION_CONFIG=str(cli),
                                ZENOH_SESSION_CONFIG_URI='original-data-peer.json5'):
                    with patch('deployment.vita_s100_aorta_30dof.actor_pico_startup.subprocess.Popen',
                               return_value=process) as popen:
                        FirmwareActions(threading.Event(), clock, monitor)(action)
                    self.assertEqual(os.environ['ZENOH_SESSION_CONFIG_URI'], 'original-data-peer.json5')
                self.assertEqual(popen.call_args.kwargs['env']['ZENOH_SESSION_CONFIG_URI'], str(cli))
                self.assertEqual(popen.call_count, 1)
                monitor.assert_called_once()

    def test_public_or_missing_profile_fails_before_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cli.json5'
            for exists in (False, True):
                if exists:
                    path.write_text('{}'); path.chmod(0o644)
                with patch.dict(os.environ, VITA_ACTOR_CLI_SESSION_CONFIG=str(path)), patch(
                        'deployment.vita_s100_aorta_30dof.actor_pico_startup.subprocess.Popen') as popen:
                    with self.assertRaisesRegex(ValueError, '0600'):
                        FirmwareActions(threading.Event(), Mock(), Mock())('FIXED_STAND')
                    popen.assert_not_called()

    def test_rpc_failure_is_reported_without_retry(self):
        process = Mock(returncode=3); process.poll.return_value = 0
        process.communicate.return_value = ('service call timed out', None)
        clock = Mock(); clock.monotonic.return_value = 0.
        with patch.dict(os.environ, VITA_ACTOR_CLI_SESSION_CONFIG=''), patch(
                'deployment.vita_s100_aorta_30dof.actor_pico_startup.subprocess.Popen',
                return_value=process) as popen:
            with self.assertRaisesRegex(RuntimeError, 'service call timed out'):
                FirmwareActions(threading.Event(), clock, Mock())('FIXED_STAND')
            self.assertEqual(popen.call_count, 1)


if __name__ == '__main__':
    unittest.main()
