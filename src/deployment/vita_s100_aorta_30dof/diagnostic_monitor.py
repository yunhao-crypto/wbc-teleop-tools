"""Independent read-only Vita LowState recorder; stop with Ctrl-C."""

from __future__ import annotations
import signal
import time
from pathlib import Path
from typing import Any

from deployment.vita_s100_aorta_30dof.aorta_adapter import VitaRawStateSource
from deployment.vita_s100_aorta_30dof.config import VitaDeploymentConfig
from deployment.vita_s100_aorta_30dof.recording import JsonlSessionRecorder


def run_diagnostic_monitor(
    config: VitaDeploymentConfig, *, output_root: Path | None = None
) -> Path:
    stop_requested = False

    def stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {
        number: signal.signal(number, stop)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    recorder = JsonlSessionRecorder(
        output_root or config.logs / "diagnostics", prefix="lowstate"
    )
    try:
        with recorder, VitaRawStateSource(group=config.aorta_group) as source:
            while not stop_requested:
                raw = source.read(0.5, wait_for_new=True)
                recorder.write(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "state_id": raw.state_id,
                        "ts_state_real_us": raw.ts_state_real_us,
                        "received_monotonic_ns": raw.received_monotonic_ns,
                        "attitude_valid": raw.attitude_valid,
                        "q_by_slot": raw.q_by_slot,
                        "dq_by_slot": raw.dq_by_slot,
                        "tau_est_by_slot": raw.tau_est_by_slot,
                        "temperature_by_slot": raw.temperature_by_slot,
                        "lost_by_slot": raw.lost_by_slot,
                        "imu": [
                            {
                                "location": imu.location,
                                "status": imu.status,
                                "quaternion_wxyz": imu.quaternion_wxyz,
                                "gyroscope_rad_s": imu.gyroscope_rad_s,
                                "accelerometer_m_s2": imu.accelerometer_m_s2,
                            }
                            for imu in raw.imu_samples
                        ],
                    }
                )
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    return recorder.directory


__all__ = ["run_diagnostic_monitor"]
