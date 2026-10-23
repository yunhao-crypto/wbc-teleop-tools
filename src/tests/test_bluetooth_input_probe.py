"""Standalone diagnostic proves reconnect evidence without any robot runtime."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from deployment.common.controller_input import Button, ControllerSample

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bt_probe', ROOT / 'scripts/probe_bluetooth_xbox.py')
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)


@pytest.mark.parametrize('recover,rearm,expected', [(True, True, True), (True, False, False), (False, False, False)])
def test_reconnect_probe_requires_loss_recovery_and_confirmation(recover, rearm, expected):
    clock = SimpleNamespace(now=0.)
    class Controller:
        mapping_info = {'mock': True}
        closed = False
        def __enter__(self): return self
        def __exit__(self, *_): self.closed = True
        def sample(self, timeout):
            return ControllerSample(int(clock.now * 1000), clock.now,
                                    frozenset([Button.A]) if clock.now > .7 and rearm else frozenset(),
                                    {'left_x': 0.})
        @property
        def link_status(self):
            connected = clock.now < .2 or (clock.now > .5 and recover)
            enabled = clock.now < .2 or (clock.now > .7 and rearm and recover)
            return dict(connected=connected, controls_enabled=enabled, reason='fixture', generation=1 if clock.now < .2 else 2)
    controller = Controller()
    result = probe_module.probe(controller, 1., require_reconnect=True,
        clock=lambda: clock.now, sleep=lambda dt: setattr(clock, 'now', clock.now + dt), emit=lambda _: None)
    assert result['passed'] is expected
    assert result['disconnect_observed']
    assert result['reconnection_observed'] is recover
    assert result['rearmed_after_reconnect'] is rearm
    assert controller.closed and result['actions_sent'] == 0


def test_stable_input_does_not_claim_disconnect_test_passed():
    clock = SimpleNamespace(now=0.)
    class Controller:
        mapping_info = {}
        link_status = dict(connected=True, controls_enabled=True)
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def sample(self, timeout): return ControllerSample(int(clock.now * 1000), clock.now)
    result = probe_module.probe(Controller(), .1, require_reconnect=True,
        clock=lambda: clock.now, sleep=lambda dt: setattr(clock, 'now', clock.now + dt), emit=lambda _: None)
    assert not result['passed']
    assert not result['disconnect_observed']


@pytest.mark.parametrize('activity', ['none', 'axis', 'button'])
def test_connected_static_axes_do_not_pass_without_real_input_activity(activity):
    clock = SimpleNamespace(now=0.)
    class Controller:
        mapping_info = {'mock': True}
        link_status = dict(connected=True, controls_enabled=True)
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def sample(self, timeout):
            changed = clock.now >= .06
            return ControllerSample(int(clock.now * 1000), clock.now,
                frozenset([Button.B]) if changed and activity == 'button' else frozenset(),
                dict(left_x=0. if changed and activity == 'axis' else -1.,
                     left_y=-1., right_x=-1., right_y=-1.))
    result = probe_module.probe(Controller(), .2,
        clock=lambda: clock.now, sleep=lambda dt: setattr(clock, 'now', clock.now + dt), emit=lambda _: None)
    assert result['link_passed']
    assert result['input_activity_observed'] is (activity != 'none')
    assert result['passed'] is (activity != 'none')
    assert 'not complete physical mapping validation' in result['scope']
