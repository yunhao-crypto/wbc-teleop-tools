"""Receive-only XRoboToolkit service lifecycle, with explicit process ownership.

No robot SDK, configuration writes, shell command composition or system service
installation. Only a child created here may be terminated here.
"""
import getpass
import hashlib
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import tempfile
import time


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.05): return True
    except OSError: return False


def resolve_executable(root, explicit=None, sdk_dir=None):
    root = Path(root).resolve()
    selected = explicit or os.environ.get('VITA_PICO_SERVICE_EXE')
    if selected:
        candidates = [Path(selected).expanduser()]
    elif os.name == 'nt':
        candidates = [root/'vendor_packages/xrobotoolkit-windows/RoboticsServiceProcess.exe',
            root/'vendor_packages/xrobotoolkit-windows/roboticsservice/RoboticsServiceProcess.exe']
        if sdk_dir:
            sdk = Path(sdk_dir).resolve()
            candidates += [sdk/'RoboticsServiceProcess.exe', sdk.parent/'RoboticsServiceProcess.exe',
                sdk/'roboticsservice/RoboticsServiceProcess.exe']
        found = shutil.which('RoboticsServiceProcess.exe')
        if found: candidates.append(Path(found))
    elif platform.system() == 'Linux' and platform.machine().lower() in ('aarch64', 'arm64'):
        candidates = [root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice/RoboticsServiceProcess']
    else:
        candidates = []
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('No compatible Pico RoboticsServiceProcess found. Set --service-exe or VITA_PICO_SERVICE_EXE to the installed executable; no runtime/configuration is installed or changed.')
    with path.open('rb') as f: magic = f.read(4)
    if os.name == 'nt' and magic == b'\x7fELF':
        raise ValueError('Linux/ARM Pico service cannot run on Windows')
    if os.name != 'nt' and magic[:2] == b'MZ':
        raise ValueError('Windows Pico service cannot run on Linux')
    return path


def default_sdk_directory(root):
    """Same compiled-binding location as the existing S100 launcher."""
    folder = Path(root).resolve()/'build/pico'
    suffix = '.pyd' if os.name == 'nt' else '.so'
    return folder if folder.is_dir() and any(folder.glob('xrobotoolkit_sdk*'+suffix)) else None


def library_directories(root, executable=None):
    paths = [Path(root).resolve()/'build/pico']
    if executable:
        base = Path(executable).resolve().parent
        paths += [base, base/'lib', base/'SDK'/('x64' if os.name == 'nt' else 'arm64')]
    if os.name != 'nt': paths.append(Path('/usr/hobot/lib'))
    return list(dict.fromkeys(str(p) for p in paths if p.is_dir()))


class Lease:
    """Per-user, cross-worktree OS lock; stale files do not retain a lease."""
    def __init__(self, port):
        user = str(os.getuid()) if hasattr(os, 'getuid') else getpass.getuser()
        key = hashlib.sha256(user.encode()).hexdigest()[:16]
        folder = Path(tempfile.gettempdir())/('vita_pico_inputs_'+key)
        folder.mkdir(mode=0o700, exist_ok=True)
        self.path = folder/f'service_{port}.lock'; self.file = None

    def acquire(self):
        stream = self.path.open('a+b')
        try:
            stream.seek(0, 2)
            if not stream.tell(): stream.write(b'0'); stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError('Another pico_smpl/pico_teleop input session owns this Pico service port') from exc
        self.file = stream

    def close(self):
        if self.file is not None:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0); msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close(); self.file = None


class PicoService:
    def __init__(self, root, log_path, *, mode='auto', executable=None, arguments=(), sdk_dir=None,
                 port=63901, ready_timeout=10., stopped=lambda: False):
        if mode not in ('auto', 'external') or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid Pico service mode/port')
        if not math.isfinite(ready_timeout) or not 0 < ready_timeout <= 60:
            raise ValueError('Invalid Pico service readiness budget')
        if mode == 'external' and (executable is not None or arguments):
            raise ValueError('External mode never launches an executable')
        self.root = Path(root).resolve(); self.log_path = Path(log_path)
        self.mode = mode; self.executable = executable; self.arguments = list(arguments); self.sdk_dir = sdk_dir
        self.port = port; self.ready_timeout = ready_timeout; self.stopped = stopped
        self.child = None; self.output = None; self.lease = Lease(port); self.closed = False
        self.report = dict(mode=mode, owned=False, pid=None, ready=False, port=port,
            executable=None, log=str(self.log_path.resolve()), forced_shutdown=False, exitcode=None,
            robot_configuration_modified=False, service_configuration_modified=False)

    def start(self):
        if self.closed or self.lease.file is not None: raise RuntimeError('Pico service session already used')
        self.lease.acquire()
        try:
            occupied = listening(self.port)
            if self.mode == 'external':
                if not occupied: raise RuntimeError(f'External Pico service is not listening on TCP {self.port}')
                self.report.update(ready=True, ownership='external_not_managed')
                return self.report
            if occupied:
                raise RuntimeError(f'TCP {self.port} is already in use. Existing process is untouched; use --pico-service external only to intentionally reuse the running Pico service.')
            executable = resolve_executable(self.root, self.executable, self.sdk_dir)
            self.output = self.log_path.open('xb')
            options = dict(cwd=executable.parent, stdout=self.output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if os.name == 'nt':
                options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            env = os.environ.copy()
            dirs = library_directories(self.root, executable)
            key = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
            env[key] = os.pathsep.join(dirs+([env[key]] if env.get(key) else []))
            self.child = subprocess.Popen([str(executable), *self.arguments], env=env, **options)
            self.report.update(owned=True, pid=self.child.pid, executable=str(executable), ownership='created_child')
            deadline = time.monotonic()+self.ready_timeout
            while True:
                if self.stopped(): raise InterruptedError('Pico service startup interrupted')
                code = self.child.poll()
                if code is not None: raise RuntimeError(f'Pico service exited during startup: {code}; see {self.log_path}')
                if listening(self.port):
                    # A listener alone is not a validated Pico skeleton stream.
                    self.check(); self.report['ready'] = True
                    return self.report
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Pico service did not listen on TCP {self.port}; see {self.log_path}')
                time.sleep(.02)
        except BaseException:
            self.close(); raise

    def check(self):
        if self.closed: raise RuntimeError('Pico service session closed')
        if self.child is not None and self.child.poll() is not None:
            raise RuntimeError(f'Owned Pico service exited: {self.child.returncode}; see {self.log_path}')

    def close(self):
        if self.closed: return dict(self.report)
        try:
            if self.child is not None:
                if self.child.poll() is None:
                    self.child.terminate()
                    try: self.child.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        self.report['forced_shutdown'] = True
                        self.child.kill(); self.child.wait(timeout=2.)
                self.report['exitcode'] = self.child.returncode
        finally:
            try:
                if self.output is not None: self.output.close()
            finally:
                self.lease.close(); self.closed = True
        return dict(self.report)
