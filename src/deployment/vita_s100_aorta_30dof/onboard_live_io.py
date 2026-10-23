"""Main-thread native-array state input; this module never sends commands."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .sdkmain_adapter import decode_body_raw
from .aorta_adapter import VitaRawStateSource
from .live_shadow import policy_state_from_lowstate
from .robot_adapter import VitaRobotState, VITA_CONTROLLER_JOINT_NAMES

@dataclass(frozen=True)
class LiveRawState:
    state_id: int
    received_monotonic_ns: int
    attitude_valid: bool
    q_by_slot: np.ndarray
    ts_state_real_us: int
    policy_state: object
    alignment: np.ndarray
    ts_state_pub_us: int | None = None
    received_wall_ns: int = 0
    read_selected_wall_ns: int = 0
    read_finished_wall_ns: int = 0
    native_receive_stamp_ns: int = 0
    applied_session_id: int = 0
    applied_sequence: int = 0
    ts_first_output_us: int = 0
    imu_timestamps_us: tuple[tuple[int, int, int, int, int], ...] = ()
    joint_timestamps_us: tuple[tuple[int, int, int], ...] = ()

def decode_live_state(payload, *, alignment, received_monotonic_ns):
    """Decode SDK body31 and retain the original policy alignment/order."""
    raw = decode_body_raw(payload, received_monotonic_ns=received_monotonic_ns)
    return live_state_from_raw(raw, alignment=alignment)


def live_state_from_raw(raw, *, alignment, arrays=None, finished_wall_ns=0):
    state = policy_state_from_lowstate(raw, base_from_hip_quaternion_wxyz=alignment)
    if arrays is not None:
        finished_wall_ns = time.time_ns()
    return LiveRawState(raw.state_id, raw.received_monotonic_ns, raw.attitude_valid,
                        raw.q_by_slot, raw.ts_state_real_us, state,
                        np.asarray(alignment, np.float32).copy(),
                        ts_state_pub_us=raw.ts_state_pub_us,
                        received_wall_ns=0 if arrays is None else arrays.received_realtime_ns,
                        read_selected_wall_ns=0 if arrays is None else arrays.received_realtime_ns,
                        read_finished_wall_ns=finished_wall_ns,
                        native_receive_stamp_ns=0 if arrays is None else arrays.pull_context_receive_stamp_ns,
                        applied_session_id=raw.applied_session_id,
                        applied_sequence=raw.applied_sequence,
                        ts_first_output_us=raw.ts_first_output_us,
                        imu_timestamps_us=tuple((s.location, s.ts_sampled_us,
                            s.ts_acquired_hw_us, s.ts_ingested_us, s.ts_assembled_us)
                            for s in raw.imu_samples),
                        joint_timestamps_us=raw.joint_timestamps_us)

def live_robot_state(raw, *, base_from_hip_quaternion_wxyz):
    if not np.array_equal(raw.alignment,np.asarray(base_from_hip_quaternion_wxyz,np.float32)):
        raise ValueError("Live state alignment does not match loop alignment")
    s=raw.policy_state
    return VitaRobotState(VITA_CONTROLLER_JOINT_NAMES,s.robot_dof_pos,s.robot_dof_vel,
                          s.robot_root_quat_wxyz,s.robot_root_angvel_local,
                          state_id=raw.state_id,received_monotonic_ns=raw.received_monotonic_ns,
                          attitude_valid=raw.attitude_valid)

class OnboardLiveStateSource(VitaRawStateSource):
    """Read-only startup ring, then shared controller ring on the main thread.

    Convert each advancing frame once. An empty ring cannot republish the
    previous state; freshness includes the original source publication age.
    """
    def __init__(self, *, group, alignment):
        self.alignment = np.asarray(alignment, np.float32)
        self._last_policy_raw = self._last_policy_state = None
        super().__init__(group=group)

    def read(self,timeout):
        raw = super().read(timeout)
        if raw is not self._last_policy_raw:
            self._last_policy_state = live_state_from_raw(raw, alignment=self.alignment,
                arrays=self._latest_arrays, finished_wall_ns=self._mapped_wall_ns)
            self._last_policy_raw = raw
        return self._last_policy_state
