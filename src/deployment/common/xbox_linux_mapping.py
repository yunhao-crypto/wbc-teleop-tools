"""Read Linux joystick mappings; never infer button meaning from event indices.

The ioctl definitions and ABS/BTN codes are Linux UAPI (joystick.h and
input-event-codes.h). These calls only read the device configuration.
"""
from __future__ import annotations

import array
import fcntl
from .controller_input import Button
from .xbox_joystick import XboxJoystickState

JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JSIOCGAXMAP = 0x80406A32
JSIOCGBTNMAP = 0x84006A34
JSIOCGNAME_128 = 0x80806A13

BUTTON_CODES = {
    0x130: Button.A, 0x131: Button.B, 0x134: Button.X, 0x133: Button.Y,
    0x136: Button.LB, 0x137: Button.RB,
    0x13A: Button.BACK, 0x13B: Button.START,
}
REQUIRED_BUTTONS = frozenset(BUTTON_CODES.values())


def mapping_from_codes(axis_codes, button_codes):
    """Accept standard gamepad mappings, rejecting ambiguous/missing controls."""
    axis_codes = list(map(int, axis_codes))
    button_codes = list(map(int, button_codes))
    if len(set(axis_codes)) != len(axis_codes):
        raise ValueError('Duplicate Linux joystick axis codes')
    if len(set(button_codes)) != len(button_codes):
        raise ValueError('Duplicate Linux joystick button codes')
    desired_axes = {0: 'left_x', 1: 'left_y'}  # ABS_X/Y
    if 3 in axis_codes and 4 in axis_codes:  # ABS_RX/RY
        desired_axes.update({3: 'right_x', 4: 'right_y'})
        layout = 'ABS_X_Y_RX_RY'
    elif (2 in axis_codes and 5 in axis_codes
          and 3 not in axis_codes and 4 not in axis_codes
          and (set(axis_codes) == {0, 1, 2, 5}
               or {9, 10}.issubset(axis_codes))):
        # Some Bluetooth HID descriptors expose sticks as Z/RZ and triggers
        # separately as GAS/BRAKE. Never treat an ambiguous trigger pair as
        # the right stick.
        desired_axes.update({2: 'right_x', 5: 'right_y'})
        layout = 'ABS_X_Y_Z_RZ'
    else:
        raise ValueError('Unsupported right-stick mapping; inspect physical input before use')
    axes = {i: desired_axes[c] for i, c in enumerate(axis_codes) if c in desired_axes}
    buttons = {i: BUTTON_CODES[c] for i, c in enumerate(button_codes) if c in BUTTON_CODES}
    if set(axes.values()) != {'left_x', 'left_y', 'right_x', 'right_y'}:
        raise ValueError('Missing required joystick axes')
    missing = REQUIRED_BUTTONS - set(buttons.values())
    if missing:
        raise ValueError('Missing required gamepad buttons: ' + ', '.join(sorted(b.value for b in missing)))
    return axes, buttons, layout


def read_joystick_mapping(descriptor, *, ioctl=None):
    ioctl = ioctl or fcntl.ioctl
    counts = []
    for request in (JSIOCGAXES, JSIOCGBUTTONS):
        result = array.array('B', [0])
        ioctl(descriptor, request, result, True)
        counts.append(int(result[0]))
    axes_count, buttons_count = counts
    if not 1 <= axes_count <= 64 or not 1 <= buttons_count <= 255:
        raise ValueError('Invalid joystick axis/button count')
    axes = array.array('B', [0]) * 64
    buttons = array.array('H', [0]) * 512
    if buttons.itemsize != 2:
        raise RuntimeError('Linux joystick ABI requires 16-bit unsigned short')
    ioctl(descriptor, JSIOCGAXMAP, axes, True)
    ioctl(descriptor, JSIOCGBTNMAP, buttons, True)
    axis_codes = list(axes[:axes_count])
    button_codes = list(buttons[:buttons_count])
    axis_map, button_map, layout = mapping_from_codes(axis_codes, button_codes)
    name = bytearray(128)
    try:
        ioctl(descriptor, JSIOCGNAME_128, name, True)
        device_name = bytes(name).split(b'\0', 1)[0].decode('utf-8', errors='replace')
    except OSError:
        device_name = 'unavailable'
    return dict(device_name=device_name, axis_codes=axis_codes, button_codes=button_codes,
                axis_map=axis_map, button_map=button_map, layout=layout)


def build_state_for_joystick(joystick):
    """Read mappings on each open, including after device re-enumeration."""
    mapping = read_joystick_mapping(joystick.descriptor)
    state = XboxJoystickState(axis_map=mapping['axis_map'], button_map=mapping['button_map'])
    state.mapping_info = {**mapping, 'button_map': {k: v.value for k, v in mapping['button_map'].items()}}
    return state
