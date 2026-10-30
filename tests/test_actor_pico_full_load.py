"""Full-load benchmark CLI and non-actuation/reporting boundaries; no SDK init."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_load_benchmark', ROOT/'scripts/bench_pico_full_load.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class FullLoadBenchmarkTests(unittest.TestCase):
    def cli(self, output, *args):
        with (patch.object(sys, 'argv', ['bench', '--output', str(output), *args]),
              patch.object(benchmark.subprocess, 'Popen', side_effect=AssertionError('Receiver must not start'))):
            with self.assertRaises(SystemExit) as error:
                benchmark.main()
        self.assertEqual(error.exception.code, 2)

    def test_invalid_modes_and_timeouts_fail_before_receiver_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'new-results'
            for args in [('--sequence', 'BD'), ('--sequence', ''), ('--frames', '49'),
                         ('--startup-timeout', '0'), ('--startup-timeout', 'nan'),
                         ('--startup-timeout', 'inf')]:
                with self.subTest(args=args):
                    self.cli(output, *args)
                    self.assertFalse(output.exists())

    def test_existing_results_are_preserved_before_receiver_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sentinel = output/'reports.json'
            sentinel.write_text('previous result')
            self.cli(output, '--pico', 'replay', '--feedback', 'recorded')
            self.assertEqual(sentinel.read_text(), 'previous result')
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_readonly_source_blocks_publication_and_rejects_publishers(self):
        publish = Mock()
        source = SimpleNamespace(transport=SimpleNamespace(publishers={}, _body_publisher=None, publish=publish))
        benchmark.require_readonly_source(source)
        with self.assertRaisesRegex(RuntimeError, 'Publishing forbidden'):
            source.transport.publish('body_cmd', object())
        publish.assert_not_called()
        for publishers, native in [({'body_cmd': object()}, None), ({}, object())]:
            source = SimpleNamespace(transport=SimpleNamespace(publishers=publishers, _body_publisher=native))
            with self.subTest(publishers=bool(publishers)), self.assertRaisesRegex(RuntimeError, 'publisher-free'):
                benchmark.require_readonly_source(source)

    def test_reports_reject_different_outputs_counts_or_actuation(self):
        good = dict(qpos_sha256='same', gmr_frames=1000, actions_sent=0, control_publishers=0)
        benchmark.require_identical_outputs([good, dict(good)])
        for rows in [[], [good, dict(good, qpos_sha256='different')],
                     [good, dict(good, gmr_frames=999)], [dict(good, actions_sent=1)],
                     [dict(good, control_publishers=1)]]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                benchmark.require_identical_outputs(rows)

    def test_script_resolves_the_fixture_from_a_checkout(self):
        self.assertEqual(benchmark.ROOT, ROOT)
        self.assertTrue((benchmark.ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz').is_file())


if __name__ == '__main__':
    unittest.main()
