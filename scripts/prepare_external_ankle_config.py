#!/usr/bin/env python3
"""Write a separate EXTERNAL command-space configuration; never operates a robot.

Install this YAML with the corresponding vita-robot build before selecting it
as --robot-config. The SDK cannot negotiate or confirm this setting remotely.
"""
import argparse
import math
from pathlib import Path
import re

import yaml


# Smallest float literal at or above the A-sample SONIC ankle torque-saturation
# bound (50 + 1*12.25)/20 = 3.1125 rad that the firmware accepts (<= pi).
DEFAULT_POSITION_TARGET_MARGIN_RAD = 3.14159


def prepare(source, space, virtual_pd_intent=None, margin=DEFAULT_POSITION_TARGET_MARGIN_RAD):
    if space not in ('motor_theta', 'virtual_pitch_roll'):
        raise ValueError('Unknown ankle command space')
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 < margin <= math.pi:
        raise ValueError('position_target_margin_rad must be finite and in (0, pi]')
    intent = space == 'virtual_pitch_roll' if virtual_pd_intent is None else virtual_pd_intent
    if type(intent) is not bool or (space == 'motor_theta' and intent):
        raise ValueError('Native motor commands require virtual_pd_intent=false')
    root = yaml.safe_load(source)
    action = root['actions']['EXTERNAL']
    if action['generator'] != 'builtin::external' or 'parallel_ankle' not in root:
        raise ValueError('Expected a parallel-ankle EXTERNAL configuration')
    # Preserve the robot's remaining configuration and comments verbatim.
    match = re.search(r'^  EXTERNAL:\s*\n(?:(?: {4}[^\n]*|[ \t]*)\n)*', source, re.M)
    if not match:
        raise ValueError('Expected standard two-space EXTERNAL YAML indentation')
    block = match.group()
    block, count = re.subn(r'^      ankle_virtual_pd_intent:.*$',
                          '      ankle_virtual_pd_intent: '+('true' if intent else 'false'),
                          block, flags=re.M)
    if count != 1:
        raise ValueError('Expected exactly one EXTERNAL ankle_virtual_pd_intent')
    line = '      ankle_command_space: '+space
    if re.search(r'^      ankle_command_space:', block, re.M):
        block = re.sub(r'^      ankle_command_space:.*$', line, block, flags=re.M)
    else:
        block = block.replace('    params:\n', '    params:\n'+line+'\n', 1)
    raw_targets = space == 'motor_theta'
    line = '      allow_position_targets_outside_mechanical_range: '+('true' if raw_targets else 'false')
    block, count = re.subn(r'^      allow_position_targets_outside_mechanical_range:.*$',
                          line, block, flags=re.M)
    if count > 1:
        raise ValueError('Expected at most one EXTERNAL position-target opt-in')
    if not count:
        block = block.replace('    params:\n', '    params:\n'+line+'\n', 1)
    margin_line = '      position_target_margin_rad: '+repr(float(margin))+'\n'
    block, count = re.subn(r'^      position_target_margin_rad:.*\n', margin_line if raw_targets else '',
                          block, flags=re.M)
    if count > 1:
        raise ValueError('Expected at most one EXTERNAL position_target_margin_rad')
    if not count and raw_targets:
        block = re.sub(r'^      allow_position_targets_outside_mechanical_range:.*\n',
                       lambda m: m.group()+margin_line, block, count=1, flags=re.M)
    interpolation = re.search(r'^      interpolation:\s*\n(?: {8}[^\n]*\n)*', block, re.M)
    if not interpolation or not isinstance(action['params'].get('interpolation'), dict):
        raise ValueError('Expected a block-style EXTERNAL interpolation configuration')
    direct = interpolation.group()
    direct, count = re.subn(r'^        enabled:.*$', '        enabled: false', direct, flags=re.M)
    if count > 1:
        raise ValueError('Expected at most one EXTERNAL interpolation.enabled')
    if not count:
        direct = direct.replace('      interpolation:\n', '      interpolation:\n        enabled: false\n', 1)
    block = block[:interpolation.start()]+direct+block[interpolation.end():]
    result = source[:match.start()]+block+source[match.end():]
    parsed = yaml.safe_load(result)
    expected = root['actions']['EXTERNAL']['params']
    expected['interpolation']['enabled'] = False
    expected['ankle_command_space'] = space
    expected['ankle_virtual_pd_intent'] = intent
    expected['allow_position_targets_outside_mechanical_range'] = raw_targets
    if raw_targets:
        expected['position_target_margin_rad'] = float(margin)
    else:
        expected.pop('position_target_margin_rad', None)
    if parsed != root:
        raise ValueError('Configuration rewrite changed an unrelated setting')
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--space', choices=('motor_theta', 'virtual_pitch_roll'), required=True)
    parser.add_argument('--virtual-pd-intent', choices=('true', 'false'),
                        help='Use false for BOTH command paths of a motor-theta model; default follows space')
    parser.add_argument('--position-target-margin-rad', type=float, default=DEFAULT_POSITION_TARGET_MARGIN_RAD,
                        help='Native mode only: firmware clamps unclipped q targets to travel +/- this margin')
    args = parser.parse_args()
    result = prepare(args.input.read_text(), args.space,
                     None if args.virtual_pd_intent is None else args.virtual_pd_intent == 'true',
                     args.position_target_margin_rad)
    # Exclusive creation also prevents accidentally replacing a live selection.
    with args.output.open('x') as stream:
        stream.write(result)
    print(f'Prepared {args.output}; EXTERNAL ankle_command_space={args.space}; not installed or activated')


if __name__ == '__main__':
    main()
