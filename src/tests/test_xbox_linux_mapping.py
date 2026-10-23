"""Linux UAPI mapping fixtures, without opening any real input device."""
import array
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from deployment.common import xbox_linux_mapping as mapping
from deployment.common.controller_input import Button
from deployment.common.xbox_joystick import JS_EVENT_AXIS, JS_EVENT_BUTTON

BUTTONS = [0x130, 0x131, 0x133, 0x134, 0x136, 0x137, 0x13A, 0x13B]


def test_reordered_axes_and_buttons_follow_codes_not_indices():
    axes, buttons, layout = mapping.mapping_from_codes([5, 4, 0, 3, 1, 2], list(reversed(BUTTONS)))
    assert axes == {1: 'right_y', 2: 'left_x', 3: 'right_x', 4: 'left_y'}
    assert buttons[0] == Button.START and buttons[4] == Button.X
    assert layout == 'ABS_X_Y_RX_RY'


def test_bluetooth_hid_separate_trigger_layout():
    axes, buttons, layout = mapping.mapping_from_codes([0, 1, 2, 5, 9, 10, 16, 17], BUTTONS)
    assert axes[2] == 'right_x' and axes[3] == 'right_y'
    assert 4 not in axes and 5 not in axes
    assert buttons[2] == Button.Y and buttons[3] == Button.X
    assert layout == 'ABS_X_Y_Z_RZ'


@pytest.mark.parametrize('axes,buttons', [
    ([0, 1, 2, 5, 16, 17], BUTTONS),  # ambiguous Z/RZ could be triggers
    ([0, 1, 3, 3, 4], BUTTONS),
    ([1, 3, 4], BUTTONS),
    ([0, 1, 3, 4], BUTTONS[:-1]),
    ([0, 1, 3, 4], BUTTONS + [0x130]),
])
def test_incomplete_or_ambiguous_mapping_rejected(axes, buttons):
    with pytest.raises(ValueError):
        mapping.mapping_from_codes(axes, buttons)


def test_ioctl_reads_mapping_and_builds_physical_button_state():
    axes = [0, 1, 2, 3, 4, 5]
    requests = []
    def ioctl(fd, request, buffer, mutate):
        assert fd == 42 and mutate is True
        requests.append(request)
        if request == mapping.JSIOCGAXES: buffer[0] = len(axes)
        elif request == mapping.JSIOCGBUTTONS: buffer[0] = len(BUTTONS)
        elif request == mapping.JSIOCGAXMAP: buffer[:len(axes)] = array.array('B', axes)
        elif request == mapping.JSIOCGBTNMAP: buffer[:len(BUTTONS)] = array.array('H', BUTTONS)
        elif request == mapping.JSIOCGNAME_128: buffer[:4] = b'Xbox'
        else: raise AssertionError(request)
    with patch.object(mapping.fcntl, 'ioctl', ioctl):
        state = mapping.build_state_for_joystick(SimpleNamespace(descriptor=42))
    state.apply_event(JS_EVENT_BUTTON, 3, 1)  # index 3 is BTN_WEST, X
    state.apply_event(JS_EVENT_AXIS, 4, -32767)
    sample = state.sample(0)
    assert sample.buttons == frozenset([Button.X])
    assert sample.axes['right_y'] == -1
    assert state.mapping_info['device_name'] == 'Xbox'
    assert len(requests) == 5
