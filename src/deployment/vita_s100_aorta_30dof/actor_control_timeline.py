"""Optional, clock-injected control-loop timing telemetry; no SDK/client imports."""
import math


def _safe(value):
    if hasattr(value, 'tolist'): value = value.tolist()
    if isinstance(value, (list, tuple)): return [_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def association_metadata(state):
    """Retain protocol3 validity separately from numeric-zero session/sequence."""
    return {name:_safe(getattr(state,name,None)) for name in
            ('protocol_version','runtime_id','ts_control_us','applied_key_valid','association_source_type')}


def sensor_timing(state):
    """Copy only existing arrays. Unknown/missing/nonfinite entries become null.

    These are consumed cached samples, NOT an independent high-rate IMU stream.
    Hardware/ingest/assembly clock relationships are intentionally undeclared.
    """
    return {name:_safe(getattr(state,name,None)) for name in
            ('joint_stamps_us','error_code','imu_stamps_us','imu_location','imu_status',
             'imu_state_present','imu_values','ts_consumed_us','received_realtime_ns')}

SCHEMA = 'wtt.control_timeline.v1'
CONTROL_PATH = 'sdk'
CLOCK_DOMAIN = 'host_monotonic'
UNITS = 'ns'
EVENT_NAMES = (
    'feedback_read', 'observation_begin', 'observation_end',
    'onnx_begin', 'onnx_end', 'beforepublish', 'publishreturn',
)


class ControlTimeline:
    """Record named host monotonic samples without touching control behavior."""
    def __init__(self, clock):
        self.clock = clock
        self.events = []

    def mark(self, event):
        if event not in EVENT_NAMES:
            raise ValueError('unknown control timeline event: '+str(event))
        monotonic_ns = getattr(self.clock, 'monotonic_ns', None)
        stamp = int(monotonic_ns() if monotonic_ns is not None else
                    round(float(self.clock.monotonic()) * 1e9))
        self.events.append({'event': event, 'host_monotonic_ns': stamp})
        return stamp

    def as_dict(self):
        return dict(schema=SCHEMA, control_path=CONTROL_PATH, units=UNITS,
                    clock_domain=CLOCK_DOMAIN,
                    clock_domains=dict(host_timestamps=CLOCK_DOMAIN,
                                       source_state_us='robot_realtime_us',
                                       ts_first_output_us='robot_realtime_us'),
                    events=list(self.events))


def header_metadata(enabled):
    """Unavailable means absent in actor JSONL, not necessarily absent in MCAP.

    Host monotonic and robot realtime are NEVER declared mutually aligned here.
    ``units`` applies only to timeline event stamps, not robot firmware fields.
    """
    return dict(enabled=bool(enabled), schema=SCHEMA, control_path=CONTROL_PATH,
                units=UNITS, clock_domains=dict(host_timestamps=CLOCK_DOMAIN,
                                               source_state_us='robot_realtime_us',
                                               ts_first_output_us='robot_realtime_us'),
                timestamp_fields={
                    'control_timeline.events[].host_monotonic_ns': {'domain':'host_monotonic','unit':'ns'},
                    'source_state_us': {'domain':'robot_realtime','unit':'us'},
                    'feedback.ts_first_output_us': {'domain':'robot_realtime','unit':'us'},
                    'feedback_timing.publication_us': {'domain':'robot_realtime','unit':'us'},
                    'feedback_timing.received_monotonic_ns': {'domain':'host_monotonic','unit':'ns'},
                    'feedback_timing.pull_context_receive_stamp_ns': {'domain':'unknown','unit':'ns'}},
                scope='successfully_published_commands_only',
                clock_alignment='not_measured',
                optional_sensor_fields={
                    'joint_stamps_us': {'columns':['acquired_hw','ingested','assembled'],'unit':'us','domains':'unknown'},
                    'imu_stamps_us': {'columns':['sampled','acquired_hw','ingested','assembled'],'unit':'us','domains':'unknown'},
                    'imu_values': {'columns':['qw','qx','qy','qz','gx','gy','gz','ax','ay','az','roll_deg','pitch_deg','yaw_deg'],
                                   'validity':'imu_state_present AND valid imu_status; calibration not verified'}},
                sensor_scope='consumed cached samples only; retain MCAP for high-rate IMU',
                unavailable=['firmware_command_receive_timestamp', 'physical_response_onset'])
