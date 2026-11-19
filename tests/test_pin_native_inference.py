"""Inference-thread selection for the runtime pin helper; no perf or robot access."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pin_native_inference', ROOT/'scripts/pin_native_inference.py')
pin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin)

SCRIPT = '\n'.join(
    ['  1001 [004]  ffff8000 (/app/locomotion/lib/libonnxruntime.so.1.20.1)']*30 +
    ['  1001 [002]  ffff8001 (/app/locomotion/lib/libonnxruntime.so.1.20.1)']*5 +
    ['  1001 [004]  aaaa0001 (/opt/native/locomotion_exec)']*4 +
    ['  1002 [004]  ffff0001 ([kernel.kallsyms])']*7 +
    ['  1003 [001]  ffff8002 (/app/locomotion/lib/libonnxruntime.so.1.20.1)']*2 +
    ['malformed', '  x [004] 0 (libonnxruntime.so)'])


class PinNativeInferenceTests(unittest.TestCase):
    def test_counts_only_onnx_runtime_samples_per_thread_and_cpu(self):
        ort, every = pin.ort_samples(SCRIPT)
        self.assertEqual(dict(ort[1001]), {4: 30, 2: 5})
        self.assertEqual(dict(ort[1003]), {1: 2})
        self.assertNotIn(1002, ort)
        self.assertEqual(dict(every[1002]), {4: 7})
        self.assertEqual(sum(every[1001].values()), 39)

    def test_choose_requires_enough_samples_and_one_dominant_thread(self):
        ort, _ = pin.ort_samples(SCRIPT)
        self.assertEqual(pin.choose(ort), 1001)
        self.assertIsNone(pin.choose(ort, minimum=100))
        split, _ = pin.ort_samples('\n'.join(['1 [4] 0 (libonnxruntime.so)']*15 + ['2 [4] 0 (libonnxruntime.so)']*15))
        self.assertIsNone(pin.choose(split))
        self.assertIsNone(pin.choose({}))

    def test_cpu_lists(self):
        self.assertEqual(pin.parse_cpus('4'), {4})
        self.assertEqual(pin.parse_cpus('0-2,5'), {0, 1, 2, 5})
        for bad in ('', '3-1', '-1', 'x'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pin.parse_cpus(bad)

    def test_failed_perf_sample_is_retried_then_thread_pinned(self):
        import json, subprocess, sys, tempfile
        from unittest import mock
        ort, every = pin.ort_samples(SCRIPT)
        calls = iter([subprocess.CalledProcessError(255, ['perf'], stderr=b'thread exited'), (ort, every), (ort, every)])

        def fake_sample(*_):
            item = next(calls)
            if isinstance(item, Exception):
                raise item
            return item
        masks = {1001: {0, 1, 2, 3, 4, 5}}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(pin, 'locomotion_pid', return_value=7), \
                mock.patch.object(pin, 'other_threads', return_value=[1001, 1002]), \
                mock.patch.object(pin, 'sample', side_effect=fake_sample), \
                mock.patch.object(pin.time, 'sleep'), \
                mock.patch.object(pin.os, 'sched_getaffinity', side_effect=lambda t: masks.get(t, {0, 2, 3})), \
                mock.patch.object(pin.os, 'sched_setaffinity', side_effect=lambda t, c: masks.__setitem__(t, set(c))), \
                mock.patch.object(pin.signal, 'signal'), \
                mock.patch.object(sys, 'argv', ['pin', '--cpus', '4', '--log', f'{tmp}/pin.log']):
            self.assertEqual(pin.main(), 0)
            events = [json.loads(line) for line in open(f'{tmp}/pin.log')]
        self.assertEqual([e['event'] for e in events], ['start', 'sample_failed', 'sample', 'pinned', 'verify'])
        self.assertEqual(events[1]['stderr'], 'thread exited')
        self.assertEqual(events[3]['after'], [4])
        self.assertEqual(masks[1001], {4})


if __name__ == '__main__':
    unittest.main()
