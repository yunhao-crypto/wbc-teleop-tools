"""Offline audit of WTT actor_inference JSONL, never a robot safety approval.

Only numpy and (when requested) CPU ONNX Runtime are imported. Robot realtime
source/echo stamps are correlated by originating command, never by reading row.
"""
from collections import Counter, OrderedDict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'wtt.control_audit.v1'
CONTRACT_KEY = 'mimiclite.tracking_contract.v1'
ALIASES = {f'{s}_{b}_achilles_joint': f'{s}_ankle_{a}_joint'
           for s in ('left', 'right') for a, b in (('pitch', 'A'), ('roll', 'B'))}
TOLERANCE = 5e-5


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def uint64(value):
    """No float round trip: SDK sessions routinely exceed 2**53."""
    return type(value) is int and 0 <= value < 2**64


def _stats(values):
    if not values:
        return dict(status='MISSING', count=0, p50_ms=None, p95_ms=None,
                    p99_ms=None, max_ms=None)
    return dict(status='AVAILABLE', count=len(values),
                **{f'p{p}_ms': float(np.percentile(values, p)) for p in (50, 95, 99)},
                max_ms=float(max(values)))


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite JSON number')
    if isinstance(value, dict):
        for item in value.values(): _finite_json(item)
    elif isinstance(value, list):
        for item in value: _finite_json(item)


def audit_log(path, model_path=None):
    """Stream rows; retain ten sensor/action histories and at most 10k commands.

    Scalar timing samples are retained for exact percentiles, not observations.
    COMPLETE describes available evidence only, never fitness to run hardware.
    """
    findings = Counter()
    def issue(code, field):
        findings[(code, field)] += 1

    def vector(value, size, field):
        if value is None:
            issue('MISSING', field)
            return None
        if isinstance(value, list) and any(x is None for x in value):
            issue('MISSING', field)
            return None
        if (not isinstance(value, list) or len(value) != size or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
            issue('INVALID', field)
            return None
        return np.asarray(value, dtype=np.float64)

    session = metadata = deployment = None
    model_hash = None
    replay = dict(status='NOT_REQUESTED', count=0, max_abs_error=None, tolerance=TOLERANCE)
    checks = {}
    def compare(name, actual, expected):
        if actual is None:
            return
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        item = checks.setdefault(name, dict(count=0, max_abs_error=0., tolerance=TOLERANCE))
        item['count'] += 1
        item['max_abs_error'] = max(item['max_abs_error'], error)
        if not math.isfinite(error) or error > TOLERANCE:
            issue('INVALID', name)

    if model_path is not None:
        model_hash = sha256_file(model_path)
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if (len(inputs) != 1 or inputs[0].shape != [1, 1224] or
                    inputs[0].type != 'tensor(float)' or len(outputs) != 1 or
                    outputs[0].shape != [1, 28]):
                raise ValueError('requires wrapped single float input [1,1224] -> [1,28]')
            metadata = json.loads(session.get_modelmeta().custom_metadata_map[CONTRACT_KEY])
            deployment = metadata['motor_deployment']
            names = deployment['joint_names']
            defaults = np.asarray(deployment['default_position'], dtype=float)
            scales = np.asarray(deployment['action_scale'], dtype=float)
            if (len(names) != 28 or len(set(names)) != 28 or defaults.shape != (28,) or
                    scales.shape != (28,) or not np.isfinite(defaults).all() or
                    not np.isfinite(scales).all() or metadata['input_dim'] != 1224 or
                    metadata['action_dim'] != 28 or metadata['action_semantics'] !=
                    'default_plus_scale_times_raw_action_no_clipping'):
                raise ValueError('unsupported tracking contract')
            terms, offset = [], 0
            for group in metadata['input_groups']:
                for term in deployment['observations'][group]:
                    width = term['width']
                    if type(width) is not int or width <= 0:
                        raise ValueError('invalid term width')
                    kind = term.get('kind')
                    if kind in ('joint_pos_history', 'joint_vel_history'):
                        joint_names = term.get('joint_names')
                        steps = term.get('history_steps')
                        if (not isinstance(joint_names, list) or len(joint_names) != 28 or
                                set(joint_names) != set(names) or not isinstance(steps, list) or
                                steps != list(range(10)) or width != 280):
                            raise ValueError('requires named ten-step joint histories')
                    if kind == 'prev_actions' and (term.get('steps') != 3 or width != 84):
                        raise ValueError('requires three previous raw actions')
                    terms.append((offset, offset + width, term))
                    offset += width
            if offset != 1224:
                raise ValueError('invalid term layout')
            replay['status'] = 'AVAILABLE'
        except Exception as exc:  # ORT exposes several non-RuntimeError load exceptions.
            issue('INVALID', 'model_contract_or_runtime')
            replay.update(status='UNSUPPORTED', reason=str(exc))
            session = deployment = None

    headers, summaries, rows = 0, 0, 0
    header, summary = {}, {}
    commands = OrderedDict()
    histories = deque(maxlen=10)
    timing = {k: [] for k in ('source_to_first_output', 'source_age_at_read',
              'source_age_before_publish', 'observation', 'inference', 'compute', 'publish',
              'timeline_observation', 'timeline_inference', 'timeline_publish',
              'timeline_read_to_publish')}
    stages = {}
    event_counts, sensor_coverage = Counter(), Counter()
    overruns = {}
    skipped_ticks = ingress_increments = ingress_resets = 0
    sensor_previous = {}
    source_previous = {}
    source_history = deque(maxlen=10)
    source_intervals, source_spans = [], []
    command_rows = 0
    matched = duplicate_echoes = orphan_echoes = zero_echoes = evicted_unobserved = 0
    repeated_commands = 0
    history_session = None
    warmup_rows = 0
    last_type = None
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                _finite_json(row)
                if not isinstance(row, dict):
                    raise ValueError('row is not object')
            except (ValueError, UnicodeError):
                issue('INVALID', 'json_line')
                continue
            kind = row.get('record_type')
            last_type = kind
            if isinstance(kind, str): event_counts[kind] += 1
            if kind == 'header':
                headers += 1
                if headers == 1:
                    header = row
                if rows or summaries:
                    issue('INVALID', 'header_order')
                continue
            if kind == 'summary':
                summaries += 1
                summary = row
                continue
            if kind != 'actor_inference':
                if kind == 'policy_period_overrun':
                    phase = row.get('phase', 'unknown')
                    overruns.setdefault(str(phase), []).append(row)
                elif kind == 'policy_ticks_skipped':
                    value = row.get('missed_ticks')
                    if uint64(value): skipped_ticks += value
                    else: issue('WARNING', 'missed_ticks')
                elif kind == 'firmware_ingress_drop':
                    previous, current = row.get('previous'), row.get('count', row.get('current'))
                    if uint64(previous) and uint64(current):
                        ingress_increments += max(0, current - previous)
                        ingress_resets += current < previous
                    else: issue('WARNING', 'firmware_ingress_drop_counter')
                continue
            if summaries:
                issue('INVALID', 'rows_after_summary')
            rows += 1
            if headers != 1:
                issue('MISSING', 'preceding_unique_header')
            stage = row.get('stage', 'MISSING')
            if not isinstance(stage, str):
                issue('INVALID', 'stage')
                stage = 'INVALID'
            stage_metrics = stages.setdefault(stage, {k: [] for k in timing})
            def sample(name, value):
                if value is None:
                    issue('MISSING', name)
                elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    issue('INVALID', name)
                else:
                    timing[name].append(value)
                    stage_metrics[name].append(value)

            for name in ('observation', 'inference', 'compute', 'publish'):
                sample(name, row.get(name + '_ms'))
            ft = row.get('feedback_timing')
            ft = ft if isinstance(ft, dict) else {}
            for when in ('at_read', 'before_publish'):
                sub = ft.get(when)
                sample('source_age_' + when, sub.get('source_age_ms') if isinstance(sub, dict) else None)

            # Only explicitly declared, same-domain host monotonic timeline differences.
            timeline = row.get('control_timeline')
            if timeline is None:
                issue('MISSING', 'control_timeline')
            else:
                declaration = header.get('audit_timing', header.get('trace_clock_schema', {}))
                if ('audit_timing' in header and 'trace_clock_schema' in header and
                        header['audit_timing'] != header['trace_clock_schema']):
                    issue('INVALID', 'conflicting_clock_declarations')
                if (not isinstance(timeline, dict) or not isinstance(declaration, dict) or
                        declaration.get('schema') != 'wtt.control_timeline.v1' or
                        declaration.get('units') != 'ns' or
                        not isinstance(declaration.get('clock_domains'), dict) or
                        declaration['clock_domains'].get('host_timestamps') != 'host_monotonic' or
                        timeline.get('schema') != 'wtt.control_timeline.v1' or
                        timeline.get('clock_domain') != 'host_monotonic' or timeline.get('units') != 'ns'):
                    issue('INVALID', 'timeline_clock_declaration')
                else:
                    events = {}
                    for event in timeline.get('events', []):
                        if not isinstance(event, dict):
                            issue('INVALID', 'timeline_event'); continue
                        name, stamp = event.get('event'), event.get('host_monotonic_ns')
                        if not isinstance(name, str) or name in events or not uint64(stamp) or stamp <= 0:
                            issue('INVALID', 'timeline_event'); continue
                        events[name] = stamp
                    ordered = ['feedback_read', 'observation_begin', 'observation_end',
                               'onnx_begin', 'onnx_end', 'beforepublish', 'publishreturn']
                    present = [events[n] for n in ordered if n in events]
                    if any(b < a for a, b in zip(present, present[1:])):
                        issue('INVALID', 'timeline_event_order')
                    for name, start, end in (
                        ('observation', 'observation_begin', 'observation_end'),
                        ('inference', 'onnx_begin', 'onnx_end'),
                        ('publish', 'beforepublish', 'publishreturn'),
                        ('read_to_publish', 'feedback_read', 'beforepublish')):
                        sample('timeline_' + name, (events[end]-events[start])/1e6
                               if start in events and end in events else None)

            fb = row.get('feedback')
            if not isinstance(fb, dict):
                issue('MISSING', 'feedback'); fb = {}
            protocol3 = fb.get('protocol_version') == 3
            runtime = row.get('runtime_id')
            if runtime is None: runtime = fb.get('runtime_id')
            if row.get('runtime_id') is not None and fb.get('runtime_id') is not None and row['runtime_id'] != fb['runtime_id']:
                issue('WARNING', 'row_feedback_runtime_mismatch')
            source_type = row.get('command_source_type')
            if fb.get('protocol_version') not in (None, 2, 3):
                issue('MISSING', 'supported_feedback_protocol')
            if protocol3 and (not uint64(runtime) or not uint64(source_type) or source_type not in range(1, 7)):
                issue('MISSING', 'protocol3_runtime_or_command_source')
            declaration = header.get('audit_timing', header.get('trace_clock_schema', {}))
            inference_only = (header.get('scope') == 'inference_only' or
                              isinstance(declaration, dict) and declaration.get('scope') == 'inference_only')
            if not inference_only: command_rows += 1
            source, sid, seq = (row.get(k) for k in ('source_state_us', 'session_id', 'sequence'))
            scope_key = (runtime if uint64(runtime) else None, sid if uint64(sid) else None)
            if uint64(source) and source > 0:
                previous = source_previous.get(scope_key)
                if previous is not None:
                    if source < previous[0]: issue('WARNING', 'source_timestamp_regression')
                    else: source_intervals.append((source-previous[0])/1000.)
                    if source == previous[0] and row.get('state_id') is not None and row.get('state_id') == previous[1]:
                        issue('WARNING', 'duplicate_source_state_identity')
                source_previous[scope_key] = (source, row.get('state_id'))
            sensors = row.get('sensor_timing')
            if not isinstance(sensors, dict):
                issue('MISSING', 'sensor_timing')
            else:
                for field, width in (('joint_stamps_us', 3), ('imu_stamps_us', 4)):
                    value = sensors.get(field)
                    if value is None:
                        issue('MISSING', field); continue
                    if not isinstance(value, list) or any(not isinstance(v, list) or len(v) != width for v in value):
                        issue('WARNING', field + '_shape'); continue
                    sensor_coverage[field + '_rows'] += 1
                    for i, stamps in enumerate(value):
                        for j, stamp in enumerate(stamps):
                            if stamp is None:
                                sensor_coverage[field + '_unavailable'] += 1; continue
                            if not uint64(stamp):
                                issue('WARNING', field + '_integer'); continue
                            if stamp == 0:
                                sensor_coverage[field + '_unobserved'] += 1; continue
                            sensor_coverage[field + '_observed'] += 1
                            skey = (runtime if uint64(runtime) else None, field, i, j)
                            if skey in sensor_previous and stamp < sensor_previous[skey]:
                                issue('WARNING', field + '_regression')
                            sensor_previous[skey] = stamp
                imu_count = len(sensors['imu_stamps_us']) if isinstance(sensors.get('imu_stamps_us'), list) else None
                for field in ('imu_location', 'imu_status', 'imu_state_present', 'imu_values'):
                    value = sensors.get(field)
                    sensor_coverage[field + '_rows'] += value is not None
                    if value is None: continue
                    if not isinstance(value, list) or (imu_count is not None and len(value) != imu_count):
                        issue('WARNING', field + '_shape'); continue
                    if field == 'imu_values':
                        if any(v is not None and (not isinstance(v, list) or len(v) != 13 or
                               any(x is not None and type(x) not in (int, float) for x in v)) for v in value):
                            issue('WARNING', field + '_shape_or_numeric')
                    elif any(v is not None and (type(v) is not bool if field == 'imu_state_present' else not uint64(v)) for v in value):
                        issue('WARNING', field + '_integer_or_boolean')
            for field, value in (('source_state_us', source), ('session_id', sid), ('sequence', seq)):
                if value is None:
                    issue('MISSING', field)
                elif not uint64(value) or (value == 0 and (field == 'source_state_us' or not protocol3)):
                    issue('INVALID', field)
            key = None
            if uint64(sid) and uint64(seq) and not inference_only:
                if protocol3:
                    if uint64(runtime) and type(source_type) is int and source_type in range(1, 7):
                        key = (runtime, source_type, sid, seq)
                elif fb.get('protocol_version') in (None, 2) and (runtime is None or uint64(runtime)) and sid > 0 and seq > 0:
                    key = (runtime, sid, seq)
            if key is not None:
                if key in commands:
                    repeated_commands += 1
                    if commands[key]['source'] != source:
                        issue('INVALID', 'conflicting_command_source')
                else:
                    commands[key] = dict(source=source, echo=None, stage=stage)
                if len(commands) > 10000:
                    _, removed = commands.popitem(last=False)
                    evicted_unobserved += removed['echo'] is None
            echo_values = [fb.get(k) for k in ('applied_session_id', 'applied_sequence', 'ts_first_output_us')]
            if any(v is None for v in echo_values):
                issue('MISSING', 'feedback_echo')
            elif not all(uint64(v) for v in echo_values):
                issue('INVALID', 'feedback_echo_uint64')
            elif (echo_values[2] == 0 or (not protocol3 and not all(v > 0 for v in echo_values))):
                zero_echoes += 1
            else:
                esid, eseq, stamp = echo_values
                if protocol3:
                    echo_runtime = fb.get('runtime_id')
                    association = fb.get('association_source_type')
                    if (fb.get('applied_key_valid') is not True or not uint64(echo_runtime) or
                            type(association) is not int or association not in range(1, 7)):
                        issue('MISSING', 'protocol3_valid_echo_association')
                        echo_key = None
                    else:
                        echo_key = (echo_runtime, association, esid, eseq)
                else:
                    echo_key = (runtime, esid, eseq) if fb.get('protocol_version') in (None, 2) and (runtime is None or uint64(runtime)) else None
                command = commands.get(echo_key)
                if command is None:
                    orphan_echoes += 1
                elif command['echo'] is not None:
                    if command['echo'] == stamp:
                        duplicate_echoes += 1
                    else:
                        issue('INVALID', 'conflicting_echo_timestamp')
                else:
                    command['echo'] = stamp
                    origin = command['source']
                    clock_schema = header.get('audit_timing', header.get('trace_clock_schema'))
                    domains = clock_schema.get('clock_domains', {}) if isinstance(clock_schema, dict) else {}
                    known_clock = clock_schema is None or (isinstance(domains, dict) and domains.get('source_state_us') ==
                        domains.get('ts_first_output_us') == 'robot_realtime_us')
                    if not known_clock:
                        issue('MISSING', 'verified_robot_clock_domain')
                    elif uint64(origin) and origin > 0:
                        latency = (stamp-origin)/1000.
                        if latency < 0:
                            issue('INVALID', 'source_to_first_output_negative')
                        else:
                            matched += 1
                            timing['source_to_first_output'].append(latency)
                            stages[command['stage']]['source_to_first_output'].append(latency)

            order = row.get('body_order')
            if (not isinstance(order, list) or len(order) != 31 or
                    any(not isinstance(n, str) for n in order) or len(set(order)) != 31):
                issue('INVALID' if order is not None else 'MISSING', 'body_order')
                order = None
            q = vector(fb.get('q_body'), 31, 'feedback.q_body')
            dq = vector(fb.get('dq_body'), 31, 'feedback.dq_body')
            quat = vector(fb.get('root_wxyz'), 4, 'feedback.root_wxyz')
            gyro = vector(fb.get('gyro'), 3, 'feedback.gyro')
            gravity = None
            if quat is not None:
                norm = np.linalg.norm(quat)
                if norm == 0: issue('INVALID', 'feedback.root_wxyz_norm')
                else:
                    w, x, y, z = quat / norm
                    gravity = np.array([2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))])
            obs = vector(row.get('observation'), 1224, 'observation')
            raw = vector(row.get('raw_action'), 28, 'raw_action')
            decoded = row.get('decoded', {})
            position = row.get('position', decoded.get('target') if isinstance(decoded, dict) else None)
            position = vector(position, 28, 'position_or_decoded.target')
            for field, expected_dim in (('observation_dim', 1224), ('action_dim', 28)):
                if field in row and (type(row[field]) is not int or row[field] != expected_dim):
                    issue('INVALID', field)
            for field in ('body_targets', 'sent_body_targets'):
                targets = row.get(field)
                if targets is None:
                    issue('MISSING', field)
                elif not isinstance(targets, list) or len(targets) != 31:
                    issue('INVALID', field)
                else:
                    for target in targets:
                        if not isinstance(target, dict):
                            issue('INVALID', field); continue
                        vector([target.get(k) for k in ('q', 'dq', 'tau', 'kp', 'kd')], 5, field)
            if scope_key != history_session:
                histories.clear()
                source_history.clear()
                history_session = scope_key
            if uint64(source) and source > 0:
                source_history.append(source)
                if len(source_history) == 10 and source_history[-1] >= source_history[0]:
                    source_spans.append((source_history[-1]-source_history[0])/1000.)
            if len(histories) < 10:
                warmup_rows += 1
            if session is not None and obs is not None and raw is not None:
                output = session.run(None, {session.get_inputs()[0].name: obs.astype(np.float32)[None]})[0][0]
                if not np.isfinite(output).all():
                    issue('INVALID', 'onnx_output')
                else:
                    compare('raw_action_replay', output, raw)
                    replay['count'] += 1
                    replay['max_abs_error'] = checks['raw_action_replay']['max_abs_error']
            if deployment is not None and raw is not None:
                expected = defaults + scales * raw
                # Legacy traces use decoded.target, newer runtimes may emit position.
                compare('affine_target', position, expected)
                try:
                    body_indices = [order.index(ALIASES.get(n, n)) for n in names]
                    for field in ('body_targets', 'sent_body_targets'):
                        targets = row.get(field)
                        if not isinstance(targets, list) or len(targets) != 31:
                            continue
                        # Serial joints only: motor/virtual ankle semantics are not interchangeable.
                        for j, n in enumerate(names):
                            if n in ALIASES:
                                continue
                            target = targets[body_indices[j]]
                            for component, wanted in (('q', expected[j]), ('kp', deployment['joints'][n]['kp']),
                                                      ('kd', deployment['joints'][n]['kd'])):
                                value = target.get(component)
                                if type(value) in (float, int) and math.isfinite(value):
                                    compare(field + '_serial_' + component, [value], [wanted])
                    if obs is not None and len(histories) == 10 and q is not None and dq is not None:
                        current = dict(q=q, dq=dq, raw=raw, order=order, gyro=gyro, gravity=gravity)
                        sensor_rows = [current] + list(reversed(histories))
                        action_rows = list(reversed(histories))
                        for start, end, term in terms:
                            kind = term['kind']
                            if kind == 'prev_actions':
                                selected = [h['raw'] for h in action_rows[:term['steps']]]
                                if any(v is None for v in selected):
                                    issue('MISSING', 'previous_raw_actions_history')
                                else:
                                    compare('previous_raw_actions_history', obs[start:end], np.concatenate(selected))
                            elif kind in ('root_ang_vel_history', 'projected_gravity_history'):
                                field = 'gyro' if kind == 'root_ang_vel_history' else 'gravity'
                                selected = [sensor_rows[step][field] for step in term['history_steps']]
                                if any(v is None for v in selected):
                                    issue('MISSING', kind)
                                else:
                                    compare(kind, obs[start:end], np.concatenate(selected))
                            elif kind in ('joint_pos_history', 'joint_vel_history'):
                                if any(sensor_rows[step]['q' if kind == 'joint_pos_history' else 'dq'] is None or
                                       sensor_rows[step]['order'] is None for step in term['history_steps']):
                                    issue('MISSING', kind); continue
                                term_names = term['joint_names']
                                block = obs[start:end].reshape(len(term['history_steps']), len(term_names))
                                serial = [j for j, n in enumerate(term_names) if n not in ALIASES]
                                expected_history = []
                                for step in term['history_steps']:
                                    h = sensor_rows[step]
                                    values = [h['q' if kind == 'joint_pos_history' else 'dq'][h['order'].index(term_names[j])]
                                              for j in serial]
                                    if kind == 'joint_pos_history' and term.get('relative_to_default'):
                                        values = np.asarray(values) - [defaults[names.index(term_names[j])] for j in serial]
                                    expected_history.append(values)
                                compare(kind + '_serial', block[:, serial], expected_history)
                except (ValueError, TypeError, KeyError, IndexError):
                    issue('MISSING', 'history_or_target_mapping')
            histories.append(dict(q=q, dq=dq, raw=raw, order=order, gyro=gyro, gravity=gravity))

    for field, count in (('header', headers), ('summary', summaries)):
        if count != 1:
            issue('MISSING' if count == 0 else 'INVALID', 'exactly_one_' + field)
    if last_type != 'summary':
        issue('MISSING', 'terminal_summary')
    if rows == 0:
        issue('MISSING', 'actor_inference_rows')
    if summaries == 1:
        if summary.get('actions_sent') is None:
            issue('MISSING', 'summary_actions_sent')
        elif not uint64(summary['actions_sent']) or summary['actions_sent'] != command_rows:
            issue('INVALID', 'summary_actions_sent')
    if header.get('record_type') == 'header' and (header.get('mode') in ('supported_external_trial', 'supported_exec_trial') or
            header.get('scope') in ('supported_external_trial', 'supported_exec_trial') or
            (header.get('audit_timing') or {}).get('scope') in ('supported_external_trial', 'supported_exec_trial') or
            header.get('trial_mode') in ('supported_external_trial', 'supported_exec_trial') or
            header.get('session_kind') in ('supported_external_trial', 'supported_exec_trial')):
        if summary.get('exit_confirmed') is not True: issue('MISSING', 'summary_exit_confirmed')
    if 'audit_timing' in header and 'trace_clock_schema' in header and header['audit_timing'] != header['trace_clock_schema']:
        issue('INVALID', 'conflicting_clock_declarations')
    for field in ('robot_id', 'policy'):
        if not header.get(field):
            issue('MISSING', 'header.' + field)
    model = header.get('model') or {}
    motion = header.get('motion') or {}
    if not isinstance(model, dict):
        issue('INVALID', 'header.model'); model = {}
    if not isinstance(motion, dict):
        issue('INVALID', 'header.motion'); motion = {}
    if not model.get('sha256'):
        issue('MISSING', 'header.model.sha256')
    if model_hash is not None and model_hash != model.get('sha256'):
        issue('INVALID', 'model_sha256_mismatch')
    if not motion.get('sha256'):
        issue('MISSING', 'header.motion.sha256')
    if not timing['timeline_publish']:
        issue('MISSING', 'host_monotonic_timeline')
    if not timing['source_to_first_output']:
        issue('MISSING', 'matched_source_to_output')
    if session is None:
        issue('MISSING', 'numeric_model_replay')
    missing_echo = evicted_unobserved + sum(c['echo'] is None for c in commands.values())
    if missing_echo:
        issue('MISSING', 'commands_without_observed_echo')
    for item in checks.values():
        item['status'] = 'PASS' if item['max_abs_error'] <= TOLERANCE else 'FAIL'
    if replay['count']:
        replay['status'] = 'PASS' if replay['max_abs_error'] <= TOLERANCE else 'FAIL'
    result = dict(schema=SCHEMA, status='INVALID' if any(k[0] == 'INVALID' for k in findings)
                  else 'PARTIAL' if findings else 'COMPLETE',
        informational_only=True, safety_approval=False,
        identity=dict(control_path=header.get('control_path', (header.get('audit_timing') or header.get('trace_clock_schema') or {}).get('control_path')), log_path=str(Path(path)), log_sha256=digest.hexdigest(),
                      model_sha256=model.get('sha256'), supplied_model_sha256=model_hash,
                      motion_sha256=motion.get('sha256'), robot_id=header.get('robot_id'),
                      policy=header.get('policy'), header_count=headers, summary_count=summaries),
        timing={k: _stats(v) for k, v in timing.items()},
        stages={s: {k: _stats(v) for k, v in metrics.items()} for s, metrics in stages.items()},
        contract=checks, replay=replay,
        findings=[dict(code=code, field=field, count=count) for (code, field), count in sorted(findings.items())],
        capabilities=dict(robot_realtime_correlation=True, host_cross_clock_subtraction=False,
                          numeric_replay=session is not None, robot_control=False,
                          logged_timing_provenance='*_ms and feedback_timing source_age_ms are collector-reported, not independently clock-synchronized; timeline differences require declared host_monotonic ns',
                          timestamp_fields=(header.get('audit_timing') or header.get('trace_clock_schema') or {}).get('timestamp_fields'),
                          clock_provenance=('header.audit_timing' if header.get('audit_timing') else 'header.trace_clock_schema' if header.get('trace_clock_schema') else
                              'legacy SDK documented source_state_us / ts_first_output_us robot realtime; assumed same robot clock, not independently synchronized'),
                          association_provenance='protocol3 requires explicit validity/runtime/source; absent protocol uses legacy v2 positive-key assumption, never inferred from MCAP',
                          optional_sensor_fields=dict(sensor_coverage),
                          unavailable=['sensor_clock_alignment', 'full_motor_virtual_ankle_validation', 'physical_execution_or_safety']),
        events=dict(counts=dict(event_counts), policy_ticks_skipped=skipped_ticks,
                    firmware_ingress_drop_increments=ingress_increments,
                    firmware_ingress_counter_resets=ingress_resets,
                    policy_period_overruns={k: dict(count=len(v), elapsed=_stats([r['elapsed_ms'] for r in v
                        if type(r.get('elapsed_ms')) in (int, float) and r['elapsed_ms'] >= 0])) for k,v in overruns.items()},
                    inferred_row_statistics_exclude_failed_and_skipped_cycles=True),
        source_progression=dict(intervals=_stats(source_intervals), nine_step_history_span=_stats(source_spans)),
        sensor_timing=dict(coverage=dict(sensor_coverage), domains='unknown; no cross-clock subtraction',
                           scope='consumed cached samples, not physical sample age or high-rate stream'),
        coverage=dict(inference_rows=rows, command_rows=command_rows, repeated_commands=repeated_commands,
                      matched_commands=matched, duplicate_echoes=duplicate_echoes,
                      unmatched_echo_rows=orphan_echoes, nonpositive_echo_rows=zero_echoes,
                      commands_without_observed_echo=missing_echo,
                      echo_absence_is_not_loss=True, command_cache_limit=10000,
                      evicted_unobserved_commands=evicted_unobserved,
                      history_warmup_rows_excluded=warmup_rows,
                      serial_joints_checked=24 if deployment else 0,
                      ankle_joints_excluded=list(ALIASES.values()),
                      history_scope='24 serial q/dq, gyro, normalized quaternion gravity and previous raw actions; excludes first ten rows per runtime/session',
                      observation_scope='numeric replay uses all 1224 values; reference and ankle reconstruction not independently validated'))
    # Self-digest is over canonical report excluding this field, not a circular file hash.
    result['report_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':'),
                                                       allow_nan=False).encode()).hexdigest()
    return result
