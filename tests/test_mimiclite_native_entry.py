"""Native entry routing through an inert shell stub; no SDK or robot access."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'scripts/start_mimiclite_native.sh'


class NativeEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='native-entry-')
        self.addCleanup(temporary.cleanup)
        self.package = Path(temporary.name)/'package with spaces'
        self.package.mkdir()
        self.calls = self.package/'calls.json'
        # Record exact argv and propagate a distinctive exit code. No installed
        # launcher, transport, controller or model runtime is called.
        (self.package/'start.sh').write_text(
            f'''#!/bin/bash
exec {shlex.quote(sys.executable)} - "$@" <<'PY'
import json, os, sys
from pathlib import Path
Path({str(self.calls)!r}).write_text(json.dumps(sys.argv[1:]))
Path({str(self.package/'cpus.json')!r}).write_text(json.dumps(sorted(os.sched_getaffinity(0))))
raise SystemExit(37)
PY
''')
        for model in ('student2000', 'knee-reg1000', 'knee-reg2000'):
            (self.package/('contract_'+model+'.json')).write_text('{}')
            (self.package/(model+'.onnx')).write_text('inert test placeholder')

    def run_entry(self, *args, inference_cpus='off'):
        env = dict(os.environ, MIMICLITE_INFERENCE_CPUS=inference_cpus)
        return subprocess.run(['bash', str(ENTRY), *args], capture_output=True, text=True, env=env)

    def test_all_models_reach_selected_package_and_preserve_exit_status(self):
        for model in ('student2000', 'knee-reg1000', 'knee-reg2000'):
            with self.subTest(model=model):
                result = self.run_entry('--model', model, '--native-root', str(self.package), '--seconds', '0')
                self.assertEqual(result.returncode, 37, result.stderr)
                self.assertEqual(json.loads(self.calls.read_text()), ['--model', model, '--seconds', '0'])

    @unittest.skipUnless({0, 2, 3} <= os.sched_getaffinity(0), 'needs CPUs 0, 2 and 3')
    def test_package_runs_off_gmr_inference_and_realtime_cpus(self):
        result = self.run_entry('--native-root', str(self.package))
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(json.loads((self.package/'cpus.json').read_text()), [0, 2, 3])
        self.assertIn('inference_cpus=off', result.stdout)
        plan = self.run_entry('--native-root', str(self.package), '--plan', inference_cpus='4')
        self.assertTrue(plan.stdout.startswith('taskset -c 0,2,3 bash '), plan.stdout)
        self.assertIn('inference thread CPUs: 4', plan.stdout)

    def test_invalid_inference_cpu_list_refuses_delegation(self):
        for value in ('', 'four', '4;true', '4-'):
            with self.subTest(value=value):
                result = self.run_entry('--native-root', str(self.package), inference_cpus=value)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.calls.exists())

    def test_stand_and_motion_arguments_are_not_shell_expanded(self):
        result = self.run_entry('--native-root', str(self.package), '--stand-only', '--seconds', '1.5')
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(json.loads(self.calls.read_text()), ['--model', 'student2000', '--stand-only', '--seconds', '1.5'])
        motion = str(self.package/'reference $(false); with spaces.npz')
        result = self.run_entry('--native-root', str(self.package), '--motion', motion)
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(json.loads(self.calls.read_text()), ['--model', 'student2000', '--motion', motion])

    def test_help_and_plan_never_invoke_the_package(self):
        for option in ('--help', '--plan'):
            with self.subTest(option=option):
                result = self.run_entry('--native-root', str(self.package), option)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(self.calls.exists())
        result = self.run_entry('--native-root', '/nonexistent/native-package', '--plan')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/nonexistent/native-package/start.sh', result.stdout)

    def test_current_robot67_model_defaults_are_explicit(self):
        for model, directory in [('student2000', 'mimiclite_native_20261117'),
                                 ('knee-reg1000', 'mimiclite_native_20261117'),
                                 ('knee-reg2000', 'mimiclite_knee_reg2000_20261117')]:
            with self.subTest(model=model):
                result = self.run_entry('--model', model, '--plan')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('/userdata/'+directory+'/start.sh', result.stdout)

    def test_missing_model_contract_or_launcher_refuses_delegation(self):
        for name in ('start.sh', 'contract_student2000.json', 'student2000.onnx'):
            with self.subTest(name=name):
                path = self.package/name
                original = path.read_bytes()
                path.unlink()
                result = self.run_entry('--native-root', str(self.package))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('missing native package file', result.stderr)
                self.assertFalse(self.calls.exists())
                path.write_bytes(original)

    def test_audit_requires_exact_package_marker(self):
        marker=self.package/'audit-package-protocol.txt'
        for text in (None, '', '{}', 'mimiclite-native-audit-package-v2', 'x'*129):
            if text is not None:marker.write_text(text)
            result=self.run_entry('--native-root',str(self.package),'--audit')
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertFalse(self.calls.exists())
        marker.write_text('mimiclite-native-audit-package-v1\n')
        for model in ('student2000','knee-reg1000','knee-reg2000'):
            result=self.run_entry('--native-root',str(self.package),'--audit','--model',model)
            self.assertEqual(result.returncode,37,result.stderr)
            self.assertIn('--audit',json.loads(self.calls.read_text()))

    def test_audit_plan_missing_package_is_read_only(self):
        result=self.run_entry('--native-root','/nonexistent/native-package','--audit','--plan')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--audit',result.stdout)
        self.assertFalse(self.calls.exists())

    def test_invalid_or_conflicting_options_do_not_reach_package(self):
        cases = [('--model', 'old814'), ('--model',), ('--native-root',),
                 ('--native-root', 'relative/path'), ('--seconds', '-1'), ('--seconds', 'nan'),
                 ('--stand-only', '--motion', '/unused.npz'), ('--bridge', '/unused'),
                 ('--planner-config', '/unused'), ('--model', 'student2000', '--model', 'knee-reg2000')]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_entry(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
