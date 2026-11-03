"""Real child-process lifecycle tests; no SDK, robot commands or camera hardware."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/pico_session_services.py'
spec = importlib.util.spec_from_file_location('pico_session_services_tests', SCRIPT)
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = self.root/'events'
        self.ready = self.root/'camera.ready'
        self.primary_pid = self.root/'primary.pid'
        self.camera_pid = self.root/'camera.pid'
        self.messages = []

    def code(self, name, source):
        path = self.root/(name+'.py')
        path.write_text('import os,sys,time,signal,json\nfrom pathlib import Path\n'+source)
        return [sys.executable, str(path)]

    def camera(self):
        return self.code('camera', f'''
events=Path({str(self.events)!r})
Path({str(self.camera_pid)!r}).write_text(str(os.getpid()))
Path({str(self.ready)!r}).write_text(json.dumps(dict(pid=os.getpid(),address=['127.0.0.1',13579],topic='/image_front_raw/h265')))
def stop(*_):
    with events.open('a') as f:f.write('camera_stop\\n')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
while True:time.sleep(.01)
''')

    def primary(self, *, wait_for_camera=True, result=0):
        return self.code('primary', f'''
events=Path({str(self.events)!r})
Path({str(self.primary_pid)!r}).write_text(str(os.getpid()))
deadline=time.monotonic()+3
while {wait_for_camera!r} and not Path({str(self.ready)!r}).exists() and time.monotonic()<deadline:time.sleep(.01)
time.sleep(.15)
with events.open('a') as f:f.write('primary_exit\\n')
raise SystemExit({result})
''')

    def run_session(self, primary, camera=None, **values):
        return session.supervise(primary, camera_command=camera,
            camera_log=self.root/'camera.log', camera_ready=self.ready,
            report=lambda text, **_:self.messages.append(text), **values)

    def wait_file(self, path):
        deadline = time.monotonic()+5
        while not path.exists():
            if time.monotonic() >= deadline:
                self.fail('Child did not become ready: '+str(path))
            time.sleep(.01)

    def test_primary_finishes_before_camera_cleanup_and_unrelated_process_survives(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda:session.stop_child(unrelated, 1.))
        result = self.run_session(self.primary(), self.camera())
        self.assertEqual(result, 0)
        self.assertEqual(self.events.read_text().splitlines(), ['primary_exit', 'camera_stop'])
        self.assertIsNone(unrelated.poll())
        self.assertFalse(self.ready.exists())
        self.assertTrue(any('PICO_VIDEO: ready' in text for text in self.messages))

    def test_failed_camera_does_not_cancel_primary_or_change_its_exit_status(self):
        camera = self.code('failed_camera', 'raise SystemExit(4)\n')
        result = self.run_session(self.primary(wait_for_camera=False, result=7), camera)
        self.assertEqual(result, 7)
        self.assertEqual(self.events.read_text().strip(), 'primary_exit')
        self.assertTrue(any('primary session continues' in text for text in self.messages))

    def test_missing_primary_reaps_the_camera(self):
        children = []
        original = subprocess.Popen
        def tracked(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        with patch.object(session.subprocess, 'Popen', tracked):
            with self.assertRaises(FileNotFoundError):
                self.run_session(['/missing/actor'], self.camera())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_unresponsive_primary_is_reaped_without_signalling_parent_group(self):
        command = self.code('unresponsive', f'''
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path({str(self.primary_pid)!r}).write_text(str(os.getpid()))
while True:time.sleep(.01)
''')
        child = subprocess.Popen(command)
        self.addCleanup(lambda:session.stop_child(child, .1))
        self.wait_file(self.primary_pid)
        session.stop_child(child, .1)
        self.assertEqual(child.returncode, -signal.SIGKILL)

    def test_sigterm_is_forwarded_and_primary_shutdown_precedes_video(self):
        primary = self.code('signal_primary', f'''
def stop(*_):
    with Path({str(self.events)!r}).open('a') as f:f.write('primary_stop\\n')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path({str(self.primary_pid)!r}).write_text(str(os.getpid()))
while True:time.sleep(.01)
''')
        wrapper = self.code('wrapper', f'''
sys.path.insert(0,{str(ROOT/'scripts')!r})
import pico_session_services as session
raise SystemExit(session.supervise({primary!r},camera_command={self.camera()!r},
    camera_log={str(self.root/'camera.log')!r},camera_ready={str(self.ready)!r}))
''')
        child = subprocess.Popen(wrapper, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 start_new_session=True)
        self.addCleanup(lambda:session.stop_child(child, 1.))
        self.wait_file(self.primary_pid); self.wait_file(self.ready)
        child.send_signal(signal.SIGTERM)
        output, _ = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 143, output)
        self.assertEqual(self.events.read_text().splitlines(), ['primary_stop', 'camera_stop'])
        for path in (self.primary_pid, self.camera_pid):
            with self.assertRaises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)

    def test_connection_startup_failure_is_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, 'before listening'):
            self.run_session(self.code('empty_pc', 'raise SystemExit(0)\n'), ready_port=1)
        self.assertFalse(any('PICO_CONNECT: ready' in text for text in self.messages))

    def test_missing_camera_profile_keeps_primary_available(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
            '--camera-session-profile', str(self.root/'absent.json5'), '--',
            sys.executable, '-c', 'print("primary_completed")'], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('profile missing', result.stdout)
        self.assertIn('primary_completed', result.stdout)

    def test_connect_help_needs_no_model_policy_robot_environment_or_services(self):
        result = subprocess.run(['bash', str(ROOT/'scripts/start_actor_pico.sh'), 'connect', '--help'],
            env=dict(os.environ, VITA_ACTOR_PYTHON=sys.executable), capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--connect-only', result.stdout)
        self.assertNotIn('Select the actor policy', result.stdout)

    def test_actor_stdin_remains_available_for_operator_commands(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
            '--no-camera', '--', sys.executable, '-c', 'print("operator_received="+input())'],
            input='stop_following\n', capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('operator_received=stop_following', result.stdout)


if __name__ == '__main__':
    unittest.main()
