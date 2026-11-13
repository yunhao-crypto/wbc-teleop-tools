"""Bounded stdio RPC to the native control helper; no process on import."""
import json
import queue
import subprocess
import threading


class JsonControlBridge:
    def __init__(self, executable, *, timeout=20.):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.messages = queue.Queue(maxsize=8)
        # A new session keeps Ctrl-C in the launcher. The bridge must stay up
        # long enough to observe the PASSIVE status after that interrupt.
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1,
                                        start_new_session=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            hello = self._receive()
            if hello.get('type') != 'hello' or hello.get('protocol') != 1:
                raise RuntimeError('unsupported native bridge protocol')
        except BaseException:
            self._terminate()
            raise

    def _put(self, value):
        while True:
            try:
                self.messages.put(value, timeout=.1)
                return
            except queue.Full:
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    continue

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except ValueError:
                    value = RuntimeError('non-JSON bridge output')
                self._put(value)
        finally:
            self._put(RuntimeError('bridge stdout closed'))

    def _receive(self):
        try:
            result = self.messages.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError('native bridge reply deadline exceeded') from None
        if isinstance(result, BaseException):
            raise result
        if not isinstance(result, dict):
            raise RuntimeError('invalid bridge reply shape')
        return result

    def _call(self, op, **fields):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('action14 bridge exited')
            self.process.stdin.write(json.dumps(dict(op=op, **fields), allow_nan=False)+'\n')
            self.process.stdin.flush()
            result = self._receive()
            if (result.get('type') != 'reply' or result.get('op') != op or
                    result.get('ok') is not True):
                raise RuntimeError('action14 bridge rejected '+op+': '+repr(result))
            if 'req_id' in fields and result.get('req_id') != fields['req_id']:
                raise RuntimeError('action14 bridge request correlation mismatch')
            return result

    def activate(self, request_id, profile):
        return self._call('activate', req_id=request_id, profile=profile)

    def event(self, request_id, name, params):
        return self._call('event', req_id=request_id, name=name, params=params)

    def status(self):
        return self._call('status')['status']

    def _terminate(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.reader.is_alive():
            self.reader.join(timeout=1)

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self._call('close')
                except Exception:
                    pass
        finally:
            self._terminate()
