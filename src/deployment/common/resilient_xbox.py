"""Local joystick recovery without treating an idle event stream as a timeout.

Applies to Linux js devices, including supported wireless receivers/Bluetooth;
this adapter does not configure or pair a wireless controller.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

from deployment.common.controller_input import ControllerSample
from deployment.common.wifi_controller import ReconnectControlGate
from deployment.common.xbox_joystick import JS_EVENT_INIT, LinuxJoystick, XboxJoystickState


class ResilientLocalXboxController:
    """Open normally at startup; neutralize unplugged controls until rearmed.

    ``path`` is compatible with LocalXboxController's positional constructor.
    ``joystick_path`` is an optional explicit alias. Reopen attempts occur only
    when sampled, at most once per reconnect_interval. No background thread or
    event heartbeat is needed for a local Linux joystick.
    """

    def __init__(self, path=Path('/dev/input/js0'), *, joystick_path=None,
                 reconnect_interval=1.0, joystick_factory=None, state_factory=None,
                 clock=None):
        if not math.isfinite(reconnect_interval) or reconnect_interval <= 0:
            raise ValueError('reconnect_interval must be finite and positive')
        self.path = Path(path if joystick_path is None else joystick_path)
        self._factory = joystick_factory or LinuxJoystick
        self._state_factory = state_factory or (lambda joystick: XboxJoystickState())
        self._clock = clock or time.monotonic
        self._retry_interval = float(reconnect_interval)
        # Missing device at startup must fail before acquiring robot control.
        self.joystick, self.state = self._open_device()
        self._gate = ReconnectControlGate()
        self._sequence = 0
        self._generation = 1
        self._last_read = None
        self._next_retry = 0.0
        self._closed = False
        self._reason = 'Local controller connected'

    def _open_device(self):
        joystick = self._factory(self.path)
        try:
            state = self._state_factory(joystick)
        except Exception as error:
            try:
                joystick.close()
            except OSError:
                pass
            raise OSError(f'Joystick mapping failed: {error}') from error
        return joystick, state

    @property
    def link_status(self):
        """Report joystick-device availability, not proven RF connectivity.

        ``connected`` means the device is open without a reported read error.
        ``age_s`` measures time since a successful poll, including an empty
        poll; neither establishes that a wireless link is currently healthy.
        """
        connected = self.joystick is not None and not self._closed
        return dict(connected=connected,
                    controls_enabled=connected and self._gate.enabled,
                    reason=self._gate.reason if connected else self._reason,
                    generation=self._generation,
                    age_s=None if self._last_read is None else
                    max(0.0, self._clock() - self._last_read))

    def _drop(self, now, error):
        joystick, self.joystick = self.joystick, None
        if joystick is not None:
            try:
                joystick.close()
            except OSError:
                pass
        self.state = XboxJoystickState()
        self._gate.reset()
        self._reason = f'Local controller disconnected: {error}'
        self._next_retry = now + self._retry_interval

    def sample(self, timeout=0.0):
        if self._closed:
            raise RuntimeError('Local controller is closed')
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('timeout must be finite and non-negative')
        now = self._clock()
        if self.joystick is None and now >= self._next_retry:
            try:
                self.joystick, self.state = self._open_device()
            except OSError as error:
                self._reason = f'Local controller unavailable: {error}'
                self._next_retry = now + self._retry_interval
            else:
                self._gate.reset()
                self._generation += 1
        if self.joystick is not None:
            try:
                events = self.joystick.read_ready_events(min(timeout, .02))
            except OSError as error:
                self._drop(self._clock(), error)
            else:
                now = self._clock()
                for event_type, number, value in events:
                    self.state.apply_event(event_type, number, value)
                    # Preserve intermediate excursions/releases even if an
                    # entire event batch ends centered with no buttons held.
                    # INIT is a snapshot assembled across events: absence
                    # before that snapshot completes is not a button release.
                    if not event_type & JS_EVENT_INIT:
                        self._gate.observe(frozenset(self.state.buttons), self.state.axes, now)
                self._last_read = now
                self._gate.observe(frozenset(self.state.buttons), self.state.axes, now)
        result = ControllerSample(self._sequence, self._clock(),
                                  frozenset(self.state.buttons), dict(self.state.axes))
        self._sequence += 1
        return self._gate.filter(result)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._drop(self._clock(), 'closed')
        self._reason = 'Local controller closed'

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Local controller is closed')
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ['ResilientLocalXboxController']
