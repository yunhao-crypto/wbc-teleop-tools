#!/usr/bin/env python3
"""Write a separate EXTERNAL command-space configuration; never operates a robot.

Install this YAML with the corresponding vita-robot build before selecting it
as --robot-config. The SDK cannot negotiate or confirm this setting remotely.
"""
import argparse
from pathlib import Path
import re

import yaml


def prepare(source, space):
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
                          '      ankle_virtual_pd_intent: '+('false' if space == 'motor_theta' else 'true'),
                          block, flags=re.M)
    if count != 1:
        raise ValueError('Expected exactly one EXTERNAL ankle_virtual_pd_intent')
    line = '      ankle_command_space: '+space
    if re.search(r'^      ankle_command_space:', block, re.M):
        block = re.sub(r'^      ankle_command_space:.*$', line, block, flags=re.M)
    else:
        block = block.replace('    params:\n', '    params:\n'+line+'\n', 1)
    result = source[:match.start()]+block+source[match.end():]
    parsed = yaml.safe_load(result)
    expected = root['actions']['EXTERNAL']['params']
    expected['ankle_command_space'] = space
    expected['ankle_virtual_pd_intent'] = space == 'virtual_pitch_roll'
    if parsed != root:
        raise ValueError('Configuration rewrite changed an unrelated setting')
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--space', choices=('motor_theta', 'virtual_pitch_roll'), required=True)
    args = parser.parse_args()
    result = prepare(args.input.read_text(), args.space)
    # Exclusive creation also prevents accidentally replacing a live selection.
    with args.output.open('x') as stream:
        stream.write(result)
    print(f'Prepared {args.output}; EXTERNAL ankle_command_space={args.space}; not installed or activated')


if __name__ == '__main__':
    main()
