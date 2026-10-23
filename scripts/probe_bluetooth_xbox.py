#!/usr/bin/env python3
"""Read Bluetooth Xbox input only: no Pico, policy models or robot SDK imports."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def probe(controller, seconds, *, require_reconnect=False, clock=time.monotonic,
          sleep=time.sleep, emit=print):
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError('Probe duration must be in (0, 600] seconds')
    started = clock()
    next_print = started
    samples = connected_samples = 0
    buttons_seen = set()
    ranges = {}
    transitions = []
    previous_signature = None
    connected_before = loss_observed = recovery_observed = rearmed = False
    last_link = {}
    with controller:
        initial_mapping = controller.mapping_info
        emit(json.dumps({'event': 'mapping', 'mapping': initial_mapping}, ensure_ascii=False))
        while clock() - started < seconds:
            sample = controller.sample(timeout=0.)
            link = controller.link_status
            now = clock()
            samples += 1
            buttons_seen.update(button.value for button in sample.buttons)
            for name, value in sample.axes.items():
                if name not in ranges:
                    ranges[name] = [value, value]
                else:
                    ranges[name][0] = min(ranges[name][0], value)
                    ranges[name][1] = max(ranges[name][1], value)
            connected = bool(link.get('connected'))
            if connected:
                connected_samples += 1
                if loss_observed:
                    recovery_observed = True
                    rearmed = rearmed or bool(link.get('controls_enabled'))
                connected_before = True
            elif connected_before:
                loss_observed = True
            signature = tuple(link.get(k) for k in ('connected', 'controls_enabled', 'generation', 'reason'))
            if signature != previous_signature:
                event = {'event': 'controller_link', 'elapsed_s': now - started, **link}
                transitions.append(event)
                emit(json.dumps(event, ensure_ascii=False))
                previous_signature = signature
            if now >= next_print:
                emit(json.dumps({'event': 'input', 'elapsed_s': now - started,
                                 'buttons': sorted(b.value for b in sample.buttons),
                                 'axes': dict(sample.axes), 'link': link}, ensure_ascii=False))
                next_print = now + 1.
            last_link = link
            sleep(.02)
    link_passed = bool(connected_samples and last_link.get('connected'))
    input_activity_observed = bool(buttons_seen or any(
        maximum > minimum for minimum, maximum in ranges.values()))
    passed = link_passed and input_activity_observed
    if require_reconnect:
        passed = passed and loss_observed and recovery_observed and rearmed and bool(last_link.get('controls_enabled'))
    return {'passed': passed, 'link_passed': link_passed,
            'input_activity_observed': input_activity_observed,
            'duration_s': clock() - started, 'samples': samples,
            'connected_samples': connected_samples, 'buttons_seen': sorted(buttons_seen),
            'axis_ranges_seen': ranges, 'disconnect_observed': loss_observed,
            'reconnection_observed': recovery_observed, 'rearmed_after_reconnect': rearmed,
            'require_reconnect': require_reconnect, 'last_link': last_link, 'initial_mapping': initial_mapping,
            'transitions': transitions, 'robot_sdk_imported': False, 'actions_sent': 0,
            'scope': 'Link and some input activity only; not complete physical mapping validation. '
                     'Operator must verify every required button and stick label/range.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/xbox_bluetooth.json')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--require-reconnect', action='store_true',
                        help='Turn controller off/on during this probe, then center/release 500ms and press A')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/bluetooth_input_probe.json')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
        parser.error('--seconds must be in (0, 600]')
    from deployment.common.bluetooth_xbox import BluetoothXboxController, load_bluetooth_config
    try:
        config = load_bluetooth_config(args.config)
        controller = BluetoothXboxController(address=config['address'], adapter=config['adapter'])
        result = probe(controller, args.seconds, require_reconnect=args.require_reconnect,
                       emit=lambda line: print(line, flush=True))
    except (OSError, ValueError, RuntimeError) as error:
        result = {'passed': False, 'link_passed': False, 'input_activity_observed': False,
                  'error': str(error), 'robot_sdk_imported': False, 'actions_sent': 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
