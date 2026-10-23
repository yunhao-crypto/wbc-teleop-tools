"""Offline local-device recovery tests; no joystick/robot transport is opened."""
from pathlib import Path
import unittest

from deployment.common.controller_input import Button, ControllerEdgeTracker
from deployment.common.resilient_xbox import ResilientLocalXboxController
from deployment.common.xbox_joystick import JS_EVENT_AXIS, JS_EVENT_BUTTON, JS_EVENT_INIT


class Clock:
    def __init__(self): self.now = 10.0
    def __call__(self): return self.now
    def advance(self, dt): self.now += dt


class Device:
    def __init__(self): self.events = []; self.error = None; self.closed = 0
    def read_ready_events(self, timeout):
        if self.error: raise self.error
        result, self.events = self.events, []
        return result
    def close(self): self.closed += 1
    def button(self, number, value, initial=False):
        self.events.append((JS_EVENT_BUTTON | (JS_EVENT_INIT if initial else 0), number, value))
    def axis(self, value): self.events.append((JS_EVENT_AXIS, 1, value))


class Factory:
    def __init__(self, *items): self.items = list(items); self.paths = []
    def __call__(self, path):
        self.paths.append(path)
        result = self.items.pop(0)
        if isinstance(result, Exception): raise result
        return result


class ResilientLocalJoystickTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(); self.first = Device(); self.second = Device()
        self.factory = Factory(self.first, self.second)
        self.controller = ResilientLocalXboxController(Path('/fake/js0'),
            joystick_factory=self.factory, clock=self.clock)
    def tearDown(self): self.controller.close()
    def disconnect(self):
        self.first.error = OSError('unplugged')
        sample = self.controller.sample()
        self.assertFalse(sample.buttons); self.assertFalse(sample.axes)
        self.assertFalse(self.controller.link_status['connected'])
        return sample
    def reconnect(self):
        self.clock.advance(1.0)
        return self.controller.sample()

    def test_missing_startup_device_fails_and_alias_is_accepted(self):
        factory = Factory(FileNotFoundError('absent'))
        with self.assertRaises(FileNotFoundError):
            ResilientLocalXboxController(joystick_path='/fake/missing', joystick_factory=factory)
        self.assertEqual(factory.paths, [Path('/fake/missing')])

    def test_idle_device_has_no_heartbeat_and_sequences_advance(self):
        self.first.button(7, 1)
        first = self.controller.sample()
        self.clock.advance(3600)
        second = self.controller.sample()
        self.assertEqual(second.buttons, frozenset([Button.START]))
        self.assertGreater(second.sequence, first.sequence)
        self.assertTrue(self.controller.link_status['connected'])
        self.assertTrue(self.controller.link_status['controls_enabled'])
        self.assertEqual(self.controller.link_status['age_s'], 0)

    def test_disconnect_clears_axes_buttons_and_limits_reopen_attempts(self):
        self.first.axis(-20000); self.first.button(1, 1)
        initial = self.controller.sample(); lost = self.disconnect()
        self.assertGreater(lost.sequence, initial.sequence)
        self.factory.items.insert(0, FileNotFoundError('absent'))
        for _ in range(20): self.controller.sample()
        self.assertEqual(len(self.factory.paths), 1)
        self.clock.advance(1); self.controller.sample()
        self.assertEqual(len(self.factory.paths), 2)
        self.clock.advance(.9); self.controller.sample()
        self.assertEqual(len(self.factory.paths), 2)
        self.clock.advance(.11); sample = self.controller.sample()
        self.assertEqual(len(self.factory.paths), 3)
        self.assertFalse(sample.axes); self.assertFalse(sample.buttons)
        self.assertEqual(self.controller.link_status['generation'], 2)

    def test_reconnect_discards_held_buttons_and_requires_neutral_then_fresh_a(self):
        self.disconnect()
        self.second.button(0, 1, initial=True); self.second.button(7, 1, initial=True)
        self.second.axis(-22000)
        self.assertFalse(self.reconnect().buttons)
        self.clock.advance(2); self.assertFalse(self.controller.sample().buttons)
        self.assertFalse(self.controller.link_status['controls_enabled'])
        self.second.button(0, 0); self.second.button(7, 0); self.second.axis(0)
        self.controller.sample()
        self.clock.advance(.49); self.controller.sample()
        self.assertFalse(self.controller.link_status['controls_enabled'])
        self.clock.advance(.02); self.controller.sample()
        self.second.button(0, 1)
        sample = self.controller.sample()
        self.assertEqual(sample.buttons, frozenset([Button.A]))
        self.assertTrue(self.controller.link_status['controls_enabled'])
        self.second.axis(-20000)
        self.assertLess(self.controller.sample().axes['left_y'], -.5)

    def test_stop_buttons_need_release_but_work_during_rearm(self):
        self.disconnect(); edges = ControllerEdgeTracker()
        self.second.button(2, 1, initial=True); self.second.button(6, 1, initial=True)
        self.assertFalse(edges.update(self.reconnect()))
        self.second.button(2, 0); self.second.button(6, 0)
        self.assertFalse(edges.update(self.controller.sample()))
        self.second.button(2, 1); self.second.button(6, 1)
        self.assertEqual(edges.update(self.controller.sample()), frozenset([Button.X, Button.BACK]))
        self.assertFalse(self.controller.link_status['controls_enabled'])
        self.assertFalse(edges.update(self.controller.sample()))

    def test_second_disconnect_resets_neutral_progress_and_rearm(self):
        self.disconnect(); self.reconnect()
        self.clock.advance(.6); self.controller.sample()
        self.second.error = OSError('again')
        self.controller.sample()
        third = Device(); third.button(0, 1, initial=True)
        self.factory.items.append(third)
        self.clock.advance(1)
        self.assertFalse(self.controller.sample().buttons)
        self.assertFalse(self.controller.link_status['controls_enabled'])

    def test_batched_axis_excursion_restarts_neutral_dwell(self):
        self.disconnect(); self.reconnect()
        self.clock.advance(.51)
        self.second.axis(20000); self.second.axis(0)
        self.controller.sample()
        self.second.button(0, 1)
        self.assertFalse(self.controller.sample().buttons)
        self.assertFalse(self.controller.link_status['controls_enabled'])
        self.second.button(0, 0); self.controller.sample()
        self.clock.advance(.51); self.controller.sample()
        self.second.button(0, 1)
        self.assertEqual(self.controller.sample().buttons, frozenset([Button.A]))

    def test_batched_button_press_release_restarts_neutral_dwell(self):
        self.disconnect(); self.reconnect()
        self.clock.advance(.51)
        self.second.button(1, 1); self.second.button(1, 0)
        self.controller.sample()
        self.second.button(0, 1)
        self.assertFalse(self.controller.sample().buttons)
        self.assertFalse(self.controller.link_status['controls_enabled'])
        self.second.button(0, 0); self.controller.sample()
        self.clock.advance(.51); self.controller.sample()
        self.second.button(0, 1)
        self.assertEqual(self.controller.sample().buttons, frozenset([Button.A]))

    def test_close_is_idempotent_and_does_not_reopen(self):
        self.controller.close(); self.controller.close()
        self.assertEqual(self.first.closed, 1)
        self.assertFalse(self.controller.link_status['connected'])
        with self.assertRaises(RuntimeError): self.controller.sample()
        self.assertEqual(len(self.factory.paths), 1)


if __name__ == '__main__': unittest.main()
