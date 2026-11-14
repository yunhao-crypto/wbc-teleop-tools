"""Opt-in timing of existing control calls; never receives feedback or controls I/O.

Import this module only when profiling is requested. The normal actor log remains
the source for policy/ONNX diagnostics; command keys join it to these streams.
"""
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import time

NOT_MEASURED = 'NOT_MEASURED'


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _percentiles(samples):
    if not samples:
        return NOT_MEASURED
    values = sorted(samples)
    def at(fraction):
        index = (len(values)-1)*fraction
        lo, hi = math.floor(index), math.ceil(index)
        return (values[lo]+(values[hi]-values[lo])*(index-lo))/1e6
    return dict(p50=at(.5), p95=at(.95), p99=at(.99), max=values[-1]/1e6)


class _Recorder:
    def __init__(self, path, metadata):
        self.stream = Path(path).open('x', encoding='utf-8')
        self.monotonic = time.monotonic
        self.count, self.previous, self.metrics, self.overheads = 0, {}, {}, []
        self.closed = False
        try:
            self.write(dict(record_type='run', schema='wtt.control_timing.v1', **metadata))
        except BaseException:
            self.stream.close()
            raise

    def write(self, record):
        self.stream.write(json.dumps(_plain(record), separators=(',', ':'), allow_nan=False)+'\n')
        self.stream.flush()

    def _record(self, row, *, suppress_errors=False):
        self.count += 1
        try:
            start = time.monotonic_ns()
            self.write(row)
            elapsed = time.monotonic_ns()-start
            self.write(dict(record_type='logging_overhead', cycle=row['cycle'],
                            cycle_record_write_ns=elapsed))
            self.overheads.append(time.monotonic_ns()-start)
        except BaseException:
            if not suppress_errors:
                raise

    def describe_body(self, info):
        self.write(dict(record_type='body_info', info=info))

    def describe(self, infos):
        self.write(dict(record_type='end_effector_info', endpoints=infos))

    def read_body(self, controller, timeout, *, phase):
        start = int(self.monotonic()*1e9)
        row = dict(record_type='body_state_read', cycle=self.count, phase=phase,
                   read_started_monotonic_ns=start)
        try:
            state = controller.wait_state(timeout)
        except BaseException as error:
            row.update(read_result='raised', error_type=type(error).__name__, error=str(error))
            row.update(state_read_call_ns=int(self.monotonic()*1e9)-start, S2=time.time_ns())
            self._record(row, suppress_errors=True)
            raise
        ended = int(self.monotonic()*1e9)
        receipt = getattr(state, 'received_monotonic_ns', 0)
        row.update(read_result='returned', runtime_id=state.runtime_id, state_id=state.state_id,
            read_finished_monotonic_ns=ended, state_read_call_ns=ended-start, S2=time.time_ns(),
            S0=state.ts_state_pub_us*1000 or NOT_MEASURED,
            S1=getattr(state, 'received_realtime_ns', 0) or NOT_MEASURED,
            source_state_ts_us=state.ts_state_real_us, publication_ts_us=state.ts_state_pub_us,
            state_received_monotonic_ns=receipt or NOT_MEASURED,
            pull_context_receive_stamp_ns=getattr(state, 'pull_context_receive_stamp_ns', 0) or NOT_MEASURED,
            feedback_age_ns=ended-receipt if receipt else NOT_MEASURED,
            applied_session_id=getattr(state, 'applied_session_id', 0),
            applied_sequence=getattr(state, 'applied_sequence', 0),
            ts_first_output_us=getattr(state, 'ts_first_output_us', 0))
        self._record(row)
        return state, row

    def _finish_publication(self, row, received, start, end, realtime, *, suppress_errors=False):
        side = row['side']
        row.update(publish_started_monotonic_ns=start, S6=realtime, publish_call_ns=end-start,
            control_interval_ns=start-self.previous[side] if side in self.previous else NOT_MEASURED,
            feedback_age_ns=start-received if received else NOT_MEASURED,
            state_received_monotonic_ns=received or NOT_MEASURED,
            receipt_to_publish_return_ns=end-received if received else NOT_MEASURED)
        self.previous[side] = start
        metrics = self.metrics.setdefault(side, dict(attempts=0, returned=0, control_interval_ns=[],
            publish_call_ns=[], feedback_age_ns=[], receipt_to_publish_return_ns=[]))
        metrics['attempts'] += 1
        metrics['returned'] += row['publish_result'] == 'returned'
        for key in ('control_interval_ns', 'publish_call_ns', 'feedback_age_ns', 'receipt_to_publish_return_ns'):
            if isinstance(row[key], int):
                metrics[key].append(row[key])
        self._record(row, suppress_errors=suppress_errors)

    def _publish(self, publish, row, received, enrich):
        row.update(cycle=self.count, S5=time.time_ns())
        start = int(self.monotonic()*1e9)
        try:
            sent = publish()
        except BaseException as error:
            end, realtime = int(self.monotonic()*1e9), time.time_ns()
            row.update(publish_result='raised', error_type=type(error).__name__, error=str(error))
            self._finish_publication(row, received, start, end, realtime, suppress_errors=True)
            raise
        end, realtime = int(self.monotonic()*1e9), time.time_ns()
        row['publish_result'] = 'returned'
        enrich(row, sent)
        self._finish_publication(row, received, start, end, realtime)
        return sent

    def publish_body(self, controller, command, state, read, *, phase, target_names,
                     required_fields, model_timing=None):
        row = dict(record_type='body_cycle', side='body', phase=phase,
            input_read_cycle=read['cycle'], state_read_call_ns=read['state_read_call_ns'],
            runtime_id=state.runtime_id, state_id=state.state_id,
            input_state_id=command.input_state_id, input_state_ts_real_us=command.input_state_ts_real_us,
            measured_q=state.q, measured_dq=state.dq,
            target_q=[j.q for j in command.joints], target_names=target_names,
            required_fields=required_fields,
            model_timing_ms={key: value for key, value in (model_timing or {}).items() if key.endswith('_ms')})
        def enrich(record, sent):
            record.update(session_id=sent.session_id, sequence=sent.sequence,
                          publish_timing=getattr(controller, 'publish_timing', None) or NOT_MEASURED)
        return self._publish(lambda: controller.publish(command, target_names=target_names,
            required_fields=required_fields), row, getattr(state, 'received_monotonic_ns', 0), enrich)

    def publish_gripper(self, client, target, latest, position, received, *, phase, rate):
        receipt_ns = int(received*1e9) if received is not None else 0
        row = dict(record_type='cycle', side=client.side, phase=phase, requested_hz=rate,
            S0=position.ts_state_pub_us*1000 or NOT_MEASURED,
            position_received_monotonic_ns=receipt_ns or NOT_MEASURED,
            position_sample_ref=position.sample_ref, position_state=position,
            target_intent=target, latest_status=dict(state_id=latest.state_id,
                status_valid_mask=latest.status_valid_mask, ready=latest.ready,
                output_allowed=latest.output_allowed, inhibit_reason=latest.inhibit_reason,
                touch_enabled=latest.touch_enabled))
        def enrich(record, sent):
            record.update(session_id=sent.key.session_id, sequence=sent.key.sequence,
                runtime_id=sent.runtime_id, layout_id=sent.layout_id, control_id=sent.control_id,
                ts_client_created_us=sent.ts_client_created_us, ts_client_send_us=sent.ts_client_send_us,
                input_observations=sent.input_observations, submitted_target=sent.target)
        return self._publish(lambda: client.publish(target, input_observations=(position.sample_ref,)),
                             row, receipt_ns, enrich)

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        try:
            if error is not None:
                self.write(dict(record_type='error', error_type=type(error).__name__, error=str(error)))
            sides = {}
            for side, metrics in self.metrics.items():
                intervals = metrics['control_interval_ns']
                sides[side] = dict(attempts=metrics['attempts'], returned=metrics['returned'],
                    observed_attempt_hz=len(intervals)*1e9/sum(intervals) if intervals and sum(intervals)>0 else NOT_MEASURED,
                    **{key.replace('_ns', '_ms'): _percentiles(value)
                       for key, value in metrics.items() if isinstance(value, list)})
            self.write(dict(record_type='summary', publications=sides,
                            logging_overhead_ms=_percentiles(self.overheads)))
        except BaseException:
            if error is None:
                raise
        finally:
            try:
                self.stream.close()
            except BaseException:
                if error is None:
                    raise


class ProfilingSession:
    def __init__(self, prefix, *, policy_log=None, policy=None):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(run_id=f'{time.time_ns()}-{os.getpid()}', policy=policy,
            policy_log=str(policy_log) if policy_log is not None else None,
            elapsed_clock='CLOCK_MONOTONIC', timestamp_clock='CLOCK_REALTIME',
            scope='existing control calls only; no new subscriber or feedback reads',
            limitations='software timestamps; no CAN transmission or physical arrival proof')
        self.ee_owned_by_worker = False
        self.body = _Recorder(str(prefix)+'.body.jsonl', dict(metadata, writer='body control thread'))
        try:
            self.ee = _Recorder(str(prefix)+'.ee.jsonl', dict(metadata, writer='OP3 gripper thread'))
        except BaseException as error:
            self.body.close(error)
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        # run_live has already attempted CANCEL. The OP3 worker owns its stream
        # once started, including when a blocked worker outlives close's timeout.
        try:
            if not self.ee_owned_by_worker:
                self.finish_grippers(error)
        finally:
            self.body.close(error)
        return False

    def begin_grippers(self, infos):
        self.ee.describe(infos)

    def finish_grippers(self, error=None):
        self.ee.close(error)

    def publish_gripper(self, *args, **kwargs):
        return self.ee.publish_gripper(*args, **kwargs)

    def publish_fault_open(self, client, target, position):
        refs = () if position is None else (position.sample_ref,)
        # No valid position is required on the existing fault-open path.
        started = time.monotonic_ns()
        try:
            sent = client.publish(target, input_observations=refs)
        except BaseException as error:
            try:
                self.ee.write(dict(record_type='fault_open_publish', side=client.side,
                    publish_result='raised', error=repr(error), publish_call_ns=time.monotonic_ns()-started))
            except BaseException:
                pass
            raise
        self.ee.write(dict(record_type='fault_open_publish', side=client.side,
            publish_result='returned', publish_call_ns=time.monotonic_ns()-started,
            session_id=sent.key.session_id, sequence=sent.key.sequence, input_observations=refs))
        return sent
