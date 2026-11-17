"""Bounded, offline native JSONL inspection. Never a causal or safety certificate."""
import collections
import copy
import hashlib
import json
import math


PRESENCE = {
    0: ('sensor',), 1: ('reference',), 2: ('observation',),
    3: ('raw_action',), 4: ('decoded',), 5: ('source_commands',),
    6: ('final_commands',), 7: ('command_key',), 8: ('body_sample',),
    9: ('joint_source_count', 'joint_captured_count', 'joint_timestamps_truncated', 'joint_timestamps'),
    10: ('primary_imu',), 14: ('vita_clock_us',), 15: ('system_clock_us',),
    16: ('steady_ns',), 17: ('worker',), 18: ('dispatch',),
    19: ('robot_runtime_id',), 20: ('state_timing',), 21: ('inference_timing',),
    22: ('history_epoch', 'inference_step', 'history_sensor_committed', 'history_action_committed'),
    23: ('activation_id',), 24: ('timeline_sampling',),
}
LIMITATIONS = [
    'No independent observation/history reconstruction; pre-capture history is unknown.',
    'No NPZ/reference artifact replay or sensor-to-observation reconstruction.',
    'No physical ACK, movement, safety, clock-domain correlation or durability validation.',
    'Duplicate detection and correlation are limited to the configured pending window.',
    'No full command transformation replay or artifact provenance proof.',
]


def integer(v, name, signed=False):
    lo, hi = (-(1 << 63), (1 << 63)-1) if signed else (0, (1 << 64)-1)
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError('invalid integer: ' + name)


def loads(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    def constant(v):
        raise ValueError('non-JSON number: ' + v)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def vector(v, n, name):
    if not isinstance(v, list) or len(v) != n:
        raise ValueError('invalid vector: ' + name)
    for x in v:
        if type(x) not in (int, float) or not math.isfinite(x):
            raise ValueError('nonfinite/missing vector value: ' + name)


def validate(r):
    """Validate one raw or expanded wire row; reference resolution needs AuditDecoder."""
    if not isinstance(r, dict) or type(r.get('schema_version')) is not int or r['schema_version'] not in (1, 2):
        raise ValueError('unsupported schema_version')
    if r.get('type') not in ('header', 'record', 'footer'):
        raise ValueError('unknown line type')
    # Preserve exact uint64 values, including genuine zeros. Do not mix clocks.
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.endswith(('_id', '_epoch', '_us', '_ns')) or k in (
                    'sequence', 'generation', 'inference_step', 'delivery_sequence',
                    'source_id', 'position_valid_mask', 'rewrite_reasons'):
                    integer(v, k, k in ('receipt_steady_ns', 'offset_ns'))
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(r)
    if r['schema_version'] == 2 and r['type'] == 'header':
        if r.get('command_encoding') != 'last_full_snapshot_ref_v1':
            raise ValueError('unsupported command_encoding')
    if r['type'] != 'record':
        return
    if r['schema_version'] == 2:
        integer(r.get('record_sequence'), 'record_sequence')
        if r['record_sequence'] == 0:
            raise ValueError('zero record_sequence')
    integer(r.get('present'), 'present')
    p = r['present']
    if p >> 25:
        raise ValueError('unknown presence bits')
    if type(r.get('kind')) is not int or not 1 <= r['kind'] <= 6:
        raise ValueError('unknown record kind')
    for name in ('engine_id', 'capture_epoch'):
        integer(r.get(name), name)
    if not isinstance(r.get('reason'), str):
        raise ValueError('missing reason')
    for bit, names in PRESENCE.items():
        for name in names:
            if name in ('source_commands', 'final_commands'):
                ref = name + '_ref'
                if ref in r:
                    if r['schema_version'] != 2:
                        raise ValueError('command reference requires schema_version 2')
                    integer(r[ref], ref)
                    if not 0 < r[ref] < r['record_sequence']:
                        raise ValueError('non-backward command reference: ' + ref)
                if int(name in r) + int(ref in r) != int(bool(p & (1 << bit))):
                    raise ValueError('presence mismatch: ' + name)
            elif (name in r) != bool(p & (1 << bit)):
                raise ValueError('presence mismatch: ' + name)
    for bit, name in ((11, 'sequence'), (12, 'capture_system_us'), (13, 'source_capture_us')):
        if (name in r.get('reference', {})) != bool(p & (1 << bit)) or (p & (1 << bit) and not p & 2):
            raise ValueError('reference presence mismatch: ' + name)
    for name, n in (('observation', 1224), ('raw_action', 28), ('decoded', 28)):
        if name in r:
            vector(r[name], n, name)
    for section, widths in (('sensor', {'q':28, 'dq':28, 'root_wxyz':4, 'gyro':3}),
                            ('reference', {'indices':8, 'q':224, 'dq':224, 'rootpos':24, 'rootquat_wxyz':32})):
        if section in r:
            for name, n in widths.items():
                vector(r[section].get(name), n, section + '.' + name)
    if 'reference' in r:
        ref = r['reference']
        for field in ('kind', 'clip_frame_count', 'center'):
            integer(ref.get(field), 'reference.' + field)
        for index in ref['indices']:
            integer(index, 'reference.index')
    for name in ('history_sensor_committed', 'history_action_committed', 'joint_timestamps_truncated'):
        if name in r and type(r[name]) is not bool:
            raise ValueError('invalid boolean: ' + name)
    if 'body_sample' in r:
        b = r['body_sample']
        if type(b.get('valid')) is not bool:
            raise ValueError('invalid body validity')
        for name in ('id', 'runtime_id', 'ts_state_real_us', 'position_valid_mask'):
            if (name in b) != b['valid']:
                raise ValueError('invalid body identity presence')
            if name in b:
                integer(b[name], name)
    if 'joint_timestamps' in r:
        a, n = r['joint_timestamps'], r['joint_captured_count']
        integer(n, 'joint_captured_count')
        integer(r['joint_source_count'], 'joint_source_count')
        if not isinstance(a, list) or len(a) != n or n > 64 or r['joint_source_count'] < n:
            raise ValueError('invalid timestamp counts')
        if r['joint_timestamps_truncated'] != (r['joint_source_count'] > n):
            raise ValueError('timestamp truncation mismatch')
        for item in a:
            integer(item.get('source_id'), 'source_id')
    for name in ('source_commands', 'final_commands'):
        if name not in r:
            continue
        if not isinstance(r[name], list) or len(r[name]) > 64:
            raise ValueError('invalid commands')
        for cmd in r[name]:
            if not isinstance(cmd, dict):
                raise ValueError('invalid command object')
            for field in ('q', 'dq', 'tau', 'kp', 'kd'):
                if field in cmd and cmd[field] is not None:
                    value = cmd[field]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError('invalid command number: ' + field)
            # Native emits explicit null for nonfinite measurements; preserve
            # that missing numeric evidence rather than inventing a zero.
            for field in ('hardware_slot', 'controller_status'):
                if field in cmd and (type(cmd[field]) is not int or not -(1 << 31) <= cmd[field] < (1 << 31)):
                    raise ValueError('invalid command integer: ' + field)
            if 'context' not in cmd:
                continue
            c = cmd['context']
            if not isinstance(c, dict):
                raise ValueError('invalid command context')
            if type(c.get('source_valid')) is not bool or type(c.get('allow_joint_projection')) is not bool:
                raise ValueError('invalid context validity')
            integer(c.get('rewrite_reasons'), 'rewrite_reasons')
            if ('source' in c) != c['source_valid']:
                raise ValueError('reset context must not invent source')
            if c['source_valid']:
                for key in ('session_id', 'sequence', 'writer_epoch', 'type'):
                    integer(c['source'].get(key), key)
    for name, fields in (('state_timing', ('delivery_sequence', 'input_state_absl_us', 'source_state_ts_real_us')),
                         ('inference_timing', ('before_infer_vita_us', 'after_infer_vita_us'))):
        if name in r:
            for field in fields:
                integer(r[name].get(field), field)
    if 'timeline_sampling' in r:
        t = r['timeline_sampling']
        if t.get('clock') != 'linux_clock_monotonic':
            raise ValueError('unknown timeline sampling clock')
        for key in ('tick_ns', 'coverage_first_ns', 'coverage_last_ns', 'offset_ns'):
            integer(t.get(key), key, signed=True)
        integer(t.get('buffer_rows'), 'buffer_rows')
        integer(t.get('source_origin_ns'), 'source_origin_ns')
        if not 2 <= t['buffer_rows'] <= 128 or t['coverage_first_ns'] >= t['coverage_last_ns']:
            raise ValueError('invalid recorded timeline extent')
    if 'inference_timing' in r and r['inference_timing']['after_infer_vita_us'] < r['inference_timing']['before_infer_vita_us']:
        raise ValueError('reversed same-domain inference clock')
    if 'primary_imu' in r:
        integer(r['primary_imu'].get('id'), 'imu.id')
        if type(r['primary_imu'].get('read_valid')) is not bool:
            raise ValueError('invalid IMU validity')
    if 'command_key' in r:
        for field in ('session_id', 'sequence', 'type'):
            integer(r['command_key'].get(field), field)
    if 'worker' in r:
        integer(r['worker'].get('generation'), 'generation')
        if type(r['worker'].get('accepted')) is not bool:
            raise ValueError('invalid worker accepted')
    if 'dispatch' in r:
        d = r['dispatch']
        if type(d.get('attempted')) is not bool:
            raise ValueError('invalid dispatch attempted')
        for key in ('success', 'status'):
            if (key in d) != (r['kind'] == 5):
                raise ValueError('dispatch stage mismatch')


class AuditDecoder:
    """One capture, two last-FULL snapshots only; returned rows never alias caches.

    decode() accepts parsed wire rows and returns expanded rows (refs removed).
    Any malformed row invalidates both bases, so later references cannot invent
    evidence across unknown input. last_diagnostics describes sequence holes.
    Use iter_audit_records for bounded JSON input; direct callers bound row size.
    """

    def __init__(self):
        self._header = None
        self._closed = False
        self._sequence = 0
        self._snapshots = {}
        self.last_diagnostics = ()

    def decode(self, row):
        self.last_diagnostics = ()
        try:
            return self._decode(row)
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
            self._snapshots.clear()
            raise

    def invalidate(self):
        """Call after JSON/transport errors before attempting to resume."""
        self._snapshots.clear()

    def _decode(self, row):
        validate(row)
        if self._closed:
            raise ValueError('data after footer')
        t = row['type']
        if t == 'header':
            if self._header is not None:
                raise ValueError('duplicate/misplaced header')
            for name in ('engine_id', 'capture_epoch'):
                integer(row.get(name), name)
            self._header = (row['schema_version'], row['engine_id'], row['capture_epoch'])
            return copy.deepcopy(row)
        if self._header is None:
            raise ValueError('cross-epoch/engine or missing header')
        version, engine, epoch = self._header
        if row['schema_version'] != version:
            raise ValueError('mixed schema_version within capture')
        if t == 'footer':
            for key, expected in (('engine_id', engine), ('capture_epoch', epoch)):
                if key in row and row[key] != expected:
                    raise ValueError('cross-epoch/engine footer')
            self._closed = True
            self._snapshots.clear()
            return copy.deepcopy(row)
        if (row['engine_id'], row['capture_epoch']) != (engine, epoch):
            raise ValueError('cross-epoch/engine or missing header')
        if version == 1:
            return copy.deepcopy(row)
        seq = row['record_sequence']
        if seq <= self._sequence:
            raise ValueError('non-increasing record_sequence')
        if seq != self._sequence + 1:
            self.last_diagnostics = ('missing_record_sequence_not_physical_loss',)
            # An omitted row could have replaced either most recent FULL base.
            self._snapshots.clear()
        self._sequence = seq
        expanded = copy.deepcopy(row)
        updates = {}
        for name in ('source_commands', 'final_commands'):
            ref = name + '_ref'
            if ref in row:
                base = self._snapshots.get(name)
                if base is None or base[0] != row[ref]:
                    raise ValueError('unresolved command reference: ' + ref)
                expanded[name] = copy.deepcopy(base[1])
                del expanded[ref]
            elif name in row:
                updates[name] = (seq, copy.deepcopy(row[name]))
        self._snapshots.update(updates)
        return expanded


def iter_audit_records(path, *, max_line_bytes=2*1024*1024, on_diagnostic=None):
    """Yield expanded header/record/footer rows with bounded line + two-base memory.

    Invalid/unresolved/truncated input raises ValueError, never substitutes zeros.
    Sequence holes raise unless on_diagnostic accepts the explicit diagnostic.
    A new iterator/decoder is required for each capture.
    """
    if type(max_line_bytes) is not int or max_line_bytes < 1:
        raise ValueError('invalid line bound')
    decoder = AuditDecoder()
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                raise ValueError('line_limit_exceeded')
            if not line.endswith(b'\n'):
                raise ValueError('truncated_final_line')
            row = decoder.decode(loads(line))
            for diagnostic in decoder.last_diagnostics:
                if on_diagnostic is None:
                    raise ValueError(diagnostic)
                on_diagnostic(diagnostic)
            yield row
    if decoder._header is None:
        raise ValueError('missing_header')
    if not decoder._closed:
        raise ValueError('missing_footer')


def audit(path, *, max_pending=256, max_line_bytes=2*1024*1024, onnx=None, atol=1e-5, rtol=1e-5):
    """Memory bounded by line cap and pending cap; diagnostics aggregated by category."""
    if max_pending < 1 or max_line_bytes < 1 or not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError('invalid bounds/tolerances')
    issues = collections.Counter()
    counts = collections.Counter()
    pending = collections.OrderedDict()
    decoder = AuditDecoder()
    header = footer = metadata = session = None
    ort = np = None
    checked = set()  # keys removed with pending entries
    def issue(name):
        # Invalid input may invent unlimited distinct keys/messages.
        if name not in issues and len(issues) >= 128:
            name = 'additional_diagnostic_categories'
        issues[name] += 1
    def finish(key, group):
        expected = {1, 2, 3} if key[0] == 'inference' else {4, 5}
        if not expected.issubset(group):
            issue('missing_correlated_records')
        checked.discard(key)
    def compare(key, group):
        if key[0] != 'inference' or key in checked or not {1, 2}.issubset(group):
            return
        checked.add(key)
        counts['observation_result_pairs'] += 1
        obs, result = group[1], group[2]
        for field in ('activation_id', 'robot_runtime_id', 'history_epoch', 'inference_step'):
            if field in obs and field in result and obs[field] != result[field]:
                issue('inference_context_identity_mismatch:' + field)
        if 'observation' not in obs or 'raw_action' not in result or 'decoded' not in result:
            issue('missing_inference_arrays')
            return
        if not onnx or session is None:
            return
        try:
            raw = np.asarray(result['raw_action'], dtype=np.float32)
            output = session.run(None, {session.get_inputs()[0].name: np.asarray(obs['observation'], dtype=np.float32).reshape(1, 1224)})[0]
            if output.shape != (1, 28) or not np.allclose(output[0], raw, atol=atol, rtol=rtol):
                issue('onnx_raw_action_mismatch')
            else:
                counts['onnx_raw_matches'] += 1
            contract = loads(metadata['embedded_contract_text'])
            if contract.get('action_semantics') != 'default_plus_scale_times_raw_action_no_clipping':
                issue('unsupported_decode_semantics')
                return
            for name in ('default_q', 'action_scale'):
                vector(metadata[name], 28, name)
            decoded = np.asarray(metadata['default_q'], dtype=np.float64) + np.asarray(metadata['action_scale'], dtype=np.float64) * raw.astype(np.float64)
            if not np.allclose(decoded, result['decoded'], atol=atol, rtol=rtol):
                issue('decoded_action_mismatch')
            else:
                counts['decoded_matches'] += 1
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            issue('replay_error:' + type(e).__name__)
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline(max_line_bytes + 1)
            if not line:
                break
            counts['lines'] += 1
            if len(line) > max_line_bytes:
                issue('line_limit_exceeded')
                break
            if not line.endswith(b'\n'):
                issue('truncated_final_line')
                break
            try:
                decoder.last_diagnostics = ()
                r = decoder.decode(loads(line))
                for diagnostic in decoder.last_diagnostics:
                    issue(diagnostic)
                decoder.last_diagnostics = ()
                t = r['type']
                if footer is not None:
                    raise ValueError('data after footer')
                if t == 'header':
                    if header is not None or counts['lines'] != 1:
                        raise ValueError('duplicate/misplaced header')
                    for name in ('engine_id', 'capture_epoch'):
                        integer(r.get(name), name)
                    if r.get('history_complete_from_reset') is not False:
                        raise ValueError('unsupported history coverage claim')
                    metadata = loads(r['metadata_json_text'])
                    if not isinstance(metadata, dict) or metadata.get('engine_id') != r['engine_id']:
                        raise ValueError('metadata engine mismatch')
                    header = r
                    if onnx:
                        import numpy as np
                        import onnxruntime as ort
                        session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
                        digest = hashlib.sha256()
                        with open(onnx, 'rb') as model:
                            for chunk in iter(lambda: model.read(1024*1024), b''):
                                digest.update(chunk)
                        if any(metadata.get(k) != digest.hexdigest() for k in ('model_sha256_before_load', 'model_sha256_after_load')):
                            issue('model_hash_missing_or_mismatch')
                elif t == 'footer':
                    footer = r
                else:
                    counts['records'] += 1
                    if 'timeline_sampling' in r:
                        ts = r['timeline_sampling']
                        counts['timeline_sampling_records'] += 1
                        if (ts['coverage_first_ns'] > ts['tick_ns'] - 160_000_000 or
                                ts['coverage_last_ns'] < ts['tick_ns'] + 80_000_000):
                            issue('recorded_timeline_undercoverage_not_physical_loss')
                    if header is None or any(r[k] != header[k] for k in ('engine_id', 'capture_epoch')):
                        raise ValueError('cross-epoch/engine or missing header')
                    if r.get('joint_timestamps_truncated'):
                        issue('joint_timestamp_capture_truncated_not_sensor_loss')
                    for cmd in r.get('final_commands', []):
                        c = cmd.get('context')
                        counts['slot_source_valid' if c and c.get('source_valid') else 'slot_source_unknown_or_reset'] += 1
                    kind = r['kind']
                    if kind == 6:
                        counts['events'] += 1
                        continue
                    idname = 'inference_id' if kind <= 3 else 'output_attempt_id'
                    integer(r.get(idname), idname)
                    # Presence, not truthiness, distinguishes missing from genuine zero.
                    key = ('inference' if kind <= 3 else 'output', r['capture_epoch'], r['engine_id'], r[idname])
                    if key not in pending:
                        if len(pending) >= max_pending:
                            old, group = pending.popitem(last=False)
                            finish(old, group)
                            issue('pending_window_eviction')
                        pending[key] = {}
                    group = pending[key]
                    if kind in group:
                        raise ValueError('duplicate record kind/ID')
                    group[kind] = r
                    if kind >= 4 and len(group) == 2:
                        if group[4].get('inference_id') != group[5].get('inference_id'):
                            issue('output_inference_identity_mismatch')
                        counts['output_pairs'] += 1
                        infkey = ('inference', r['capture_epoch'], r['engine_id'], r.get('inference_id'))
                        if infkey not in pending or 3 not in pending[infkey]:
                            issue('output_worker_link_unresolved')
                        else:
                            worker = pending[infkey][3]
                            if not worker.get('worker', {}).get('accepted'):
                                issue('output_worker_not_accepted')
                            if worker.get('command_key') != r.get('command_key') or 'command_key' not in worker:
                                issue('output_worker_command_key_unresolved')
                    compare(key, group)
                    counts['peak_pending'] = max(counts['peak_pending'], len(pending))
            except (ValueError, KeyError, TypeError, AttributeError, UnicodeError, OverflowError, RecursionError) as e:
                decoder.invalidate()
                for diagnostic in decoder.last_diagnostics:
                    issue(diagnostic)
                issue('invalid_line:' + str(e)[:120])
    for key, group in pending.items():
        finish(key, group)
    if header is None:
        issue('missing_header')
    if footer is None:
        issue('missing_footer')
    else:
        if footer.get('complete_scope') != 'accepted_records_drained' or footer.get('complete') is not True:
            issue('accepted_records_not_confirmed_drained')
        if footer.get('full_inference_coverage', 'absent') is not None:
            issue('invalid_footer_coverage_claim')
        try:
            for k in ('attempted', 'enqueued', 'written', 'unwritten', 'dropped_full', 'dropped_contention', 'dropped_disabled', 'dropped_stale', 'serialization_failures', 'write_failures', 'truncated_tail'):
                integer(footer.get(k), k)
                if k.startswith('dropped_') or k in ('unwritten', 'serialization_failures', 'write_failures', 'truncated_tail'):
                    if footer[k]:
                        issue('recorder_counter_nonzero:' + k)
            if footer['attempted'] != footer['enqueued'] + footer['dropped_full'] or footer['unwritten'] != footer['enqueued'] - footer['written'] or footer['written'] != counts['records']:
                issue('footer_counter_inconsistency')
            if not isinstance(footer.get('counter_scope'), dict):
                issue('missing_counter_scope')
        except (ValueError, KeyError, TypeError):
            issue('invalid_footer_counters')
    if not onnx:
        issue('onnx_replay_not_requested')
    return {'status': 'PARTIAL', 'scope': 'recorded_software_evidence_only',
            'counts': dict(counts), 'diagnostics': dict(issues), 'limitations': LIMITATIONS,
            'accepted_records_drained_claim': bool(footer and footer.get('complete') is True and footer.get('complete_scope') == 'accepted_records_drained'),
            'physical_loss_inferred': False, 'atol': atol, 'rtol': rtol}
