#!/usr/bin/env python3
"""Exercise the real BlueZ monitor only, without pairing or opening gamepads."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from deployment.common.bluetooth_xbox import BlueZMonitor


def main():
    # This deliberately unpaired address is a read-only parser/worker probe.
    # It is never saved to the operator selection or used in a Connect call.
    monitor = BlueZMonitor('00:00:00:00:00:00')
    readings = []
    monitor.start()
    try:
        first = monitor.wait_first(2.)
        if first is None:
            raise RuntimeError('BlueZ monitor did not produce a snapshot')
        deadline = time.monotonic() + 3.
        while time.monotonic() < deadline:
            started = time.monotonic_ns()
            snapshot = monitor.snapshot()
            elapsed_ns = time.monotonic_ns() - started
            readings.append(dict(service_available=snapshot.service_available,
                adapter_powered=snapshot.adapter_powered, ready=snapshot.ready,
                observed_at=snapshot.observed_at, age_s=time.monotonic()-snapshot.observed_at,
                snapshot_read_us=elapsed_ns/1000., reason=snapshot.reason))
            time.sleep(.02)
    finally:
        monitor.close()
    unique_queries = len({x['observed_at'] for x in readings})
    result = dict(passed=bool(readings) and unique_queries >= 10
        and all(x['service_available'] and x['adapter_powered'] and not x['ready']
                and 0 <= x['age_s'] <= .5 for x in readings)
        and not monitor._thread.is_alive(),
        snapshots=len(readings), distinct_queries=unique_queries,
        max_snapshot_age_s=max(x['age_s'] for x in readings),
        max_snapshot_read_us=max(x['snapshot_read_us'] for x in readings),
        background_thread_stopped=not monitor._thread.is_alive(),
        actions_sent=0, pairing_calls=0, bluetooth_state_mutations=0,
        gamepad_opened=False, physical_controller_validated=False,
        scope='Actual S100 read-only BlueZ query/parser/background lifecycle; deliberately absent address.',
        readings=readings)
    (ROOT/'outputs/bluetooth_monitor_readonly_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='readings'}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
