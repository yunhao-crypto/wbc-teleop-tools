"""Synthetic offline checks; no robot/native runtime calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deployment.vita_s100_aorta_30dof.actor_native_audit import AuditDecoder, audit, iter_audit_records, validate


def header(metadata=None):
    return dict(schema_version=1, type='header', engine_id=2**64-1, capture_epoch=1,
                history_complete_from_reset=False,
                metadata_json_text=json.dumps(metadata or {'engine_id': 2**64-1}))


def record(kind=1, ident=1, **extra):
    r = dict(schema_version=1, type='record', engine_id=2**64-1, capture_epoch=1,
             kind=kind, present=0, reason='', inference_id=ident)
    r.update(extra)
    return r


def footer(n, **extra):
    r = dict(schema_version=1, type='footer', complete_scope='accepted_records_drained',
             complete=True, full_inference_coverage=None, close_confirmed=False,
             durability_confirmed=False, counter_scope={'session': [], 'lifetime_snapshot': []},
             attempted=n, enqueued=n, written=n, unwritten=0, dropped_full=0,
             dropped_contention=0, dropped_disabled=0, dropped_stale=0,
             serialization_failures=0, write_failures=0, truncated_tail=0)
    r.update(extra)
    return r


class NativeAuditTest(unittest.TestCase):
    def run_capture(self, records, tail=b'', **kwargs):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'audit.jsonl'
            p.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in records)+tail)
            return audit(p, **kwargs)

    def test_uint64_and_zero_validity(self):
        r = record(present=(1 << 19) | (1 << 20), robot_runtime_id=0,
                   state_timing=dict(delivery_sequence=0, input_state_absl_us=0,
                                     source_state_ts_real_us=2**64-1, receipt_steady_ns=-(2**63)))
        validate(r)
        for bad in (-1, 2**64, True, 1.0):
            with self.assertRaises(ValueError):
                validate(dict(r, engine_id=bad))

    def test_presence_and_clock(self):
        for r in (record(observation=[0]*1224), record(present=4),
                  record(present=1 << 21, inference_timing={'before_infer_vita_us': 3, 'after_infer_vita_us': 2})):
            with self.assertRaises(ValueError):
                validate(r)

    def test_reference_transport_at_watchdog(self):
        timing = dict(sequence=439, ingest_monotonic_ns=10, check_monotonic_ns=20,
                      packet_age_ms=102., source_age_ms=130., source_limit_ms=250.)
        row = record(present=1 << 25, reference_transport=timing, reason='timeline_watchdog_expired')
        validate(row)
        for change in ({'check_monotonic_ns': 9}, {'packet_age_ms': -1}, {'source_limit_ms': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(row, reference_transport=dict(timing, **change)))
        with self.assertRaises(ValueError):
            validate(dict(row, present=0))

    def test_timeline_sampling_clock_and_complete_coverage(self):
        timing = dict(clock='linux_clock_monotonic', tick_ns=2**53+17,
                      coverage_first_ns=2**53+17-160_000_000,
                      coverage_last_ns=2**53+17+80_000_000,
                      offset_ns=-111, buffer_rows=128, source_origin_ns=2**64-1)
        r = record(present=1 << 24, timeline_sampling=timing)
        validate(r)
        for change in ({'clock': 'system_clock'}, {'buffer_rows': 129},
                       {'coverage_last_ns': timing['coverage_first_ns']},
                       {'tick_ns': float(timing['tick_ns'])}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(r, timeline_sampling=dict(timing, **change)))
        with self.assertRaises(ValueError):
            validate(dict(r, present=0))
        # Guard failures are legitimate recorded evidence, not malformed data.
        failed = dict(r, timeline_sampling=dict(timing, coverage_last_ns=timing['coverage_last_ns']-1),
                      reason='timeline_future_undercoverage')
        validate(failed)
        report = self.run_capture([header(), failed])
        self.assertIn('recorded_timeline_undercoverage_not_physical_loss', report['diagnostics'])

    def test_duplicate_cross_epoch_malformed_and_partial(self):
        report = self.run_capture([header(), record(), record(), record(capture_epoch=2)], tail=b'{bad}\n{"')
        self.assertEqual(report['status'], 'PARTIAL')
        keys = ' '.join(report['diagnostics'])
        for text in ('duplicate', 'cross-epoch', 'invalid_line', 'missing_footer', 'truncated_final_line', 'missing_correlated'):
            self.assertIn(text, keys)
        self.assertFalse(report['physical_loss_inferred'])

    def test_footer_drained_is_not_coverage(self):
        report = self.run_capture([header(), record(), footer(1, attempted=2, dropped_full=1, dropped_contention=3)])
        self.assertTrue(report['accepted_records_drained_claim'])
        self.assertEqual(report['status'], 'PARTIAL')
        self.assertIn('recorder_counter_nonzero:dropped_full', report['diagnostics'])
        report = self.run_capture([header(), footer(0, complete_scope='all_safe')])
        self.assertFalse(report['accepted_records_drained_claim'])

    def test_bounded_pending_and_line(self):
        report = self.run_capture([header()]+[record(ident=i) for i in range(1, 20)], max_pending=2)
        self.assertEqual(report['counts']['peak_pending'], 2)
        self.assertIn('pending_window_eviction', report['diagnostics'])
        report = self.run_capture([header()], max_line_bytes=5)
        self.assertIn('line_limit_exceeded', report['diagnostics'])

    def test_reset_slot_context(self):
        cmd = {'hardware_slot': 5, 'context': {'source_valid': False, 'allow_joint_projection': False, 'rewrite_reasons': 8}}
        r = record(5, present=64, final_commands=[cmd], output_attempt_id=1)
        validate(r)
        report = self.run_capture([header(), r])
        self.assertEqual(report['counts']['slot_source_unknown_or_reset'], 1)
        cmd['context']['source'] = {'session_id': 1}
        with self.assertRaises(ValueError):
            validate(r)

    def test_output_pair_worker_link(self):
        key = {'session_id': 0, 'sequence': 0, 'type': 0}
        rows = [record(1), record(2), record(3, present=(1 << 17) | 128,
                worker={'generation': 0, 'accepted': True}, command_key=key)]
        rows += [record(k, output_attempt_id=2, present=128 | (1 << 18), command_key=key,
                        dispatch=({'attempted': True} if k == 4 else {'attempted': True, 'success': True, 'status': 0})) for k in (4, 5)]
        report = self.run_capture([header()]+rows+[footer(5)])
        self.assertEqual(report['counts']['output_pairs'], 1)
        self.assertNotIn('output_worker_link_unresolved', report['diagnostics'])

    def test_timestamp_truncation_not_physical_loss(self):
        r = record(present=512, joint_source_count=70, joint_captured_count=0,
                   joint_timestamps_truncated=True, joint_timestamps=[])
        report = self.run_capture([header(), r])
        self.assertIn('joint_timestamp_capture_truncated_not_sensor_loss', report['diagnostics'])
        self.assertFalse(report['physical_loss_inferred'])

    def test_v2_two_full_bases_and_owned_expansion(self):
        decoder = AuditDecoder()
        decoder.decode(dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1'))
        source = [{'hardware_slot': 0, 'q': -0.0, 'context': {
            'source_valid': True, 'allow_joint_projection': False, 'rewrite_reasons': 0,
            'source': {'session_id': 0, 'sequence': 0, 'writer_epoch': 0, 'type': 0}}}]
        first = dict(record(6, present=96, source_commands=source, final_commands=[]),
                     schema_version=2, record_sequence=1)
        expanded = decoder.decode(first)
        first['source_commands'][0]['q'] = 999
        expanded['source_commands'][0]['q'] = 888
        for seq in (2, 3):
            wire = dict(record(6, present=96, source_commands_ref=1, final_commands_ref=1),
                        schema_version=2, record_sequence=seq)
            validate(wire)
            result = decoder.decode(wire)
            validate(result)
            self.assertEqual(result['source_commands'][0]['q'], 0.0)
            self.assertEqual(result['source_commands'][0]['context']['source']['session_id'], 0)
            self.assertEqual(result['final_commands'], [])
            self.assertNotIn('source_commands_ref', result)
            result['source_commands'][0]['q'] = 777
        self.assertEqual(len(decoder._snapshots), 2)
        # Each field has an independent most-recent FULL, refs do not advance it.
        decoder.decode(dict(record(6, present=32, source_commands=[]), schema_version=2, record_sequence=4))
        ok = dict(record(6, present=96, source_commands_ref=4, final_commands_ref=1),
                  schema_version=2, record_sequence=5)
        self.assertEqual(decoder.decode(ok)['source_commands'], [])
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            decoder.decode(dict(ok, record_sequence=6, source_commands_ref=1))

    def test_v2_malformed_wire_and_scope(self):
        h = dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1')
        base = dict(record(6, present=32, source_commands=[]), schema_version=2, record_sequence=1)
        ref = dict(record(6, present=32, source_commands_ref=1), schema_version=2, record_sequence=2)
        for change in ({'source_commands': []}, {'present': 0}, {'source_commands_ref': 0},
                       {'source_commands_ref': 2}, {'source_commands_ref': 3},
                       {'source_commands_ref': True}, {'source_commands_ref': 1.0},
                       {'source_commands_ref': 2**64}, {'record_sequence': 0},
                       {'record_sequence': True}, {'schema_version': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(ref, **change))
        for bad in (dict(ref, capture_epoch=2), dict(ref, engine_id=0),
                    dict(base), dict(ref, source_commands_ref=1, present=64,
                                     final_commands_ref=1), dict(record(6), schema_version=1)):
            decoder = AuditDecoder()
            decoder.decode(h)
            decoder.decode(base)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decoder.decode(bad)
            # No stale cache after any invalid row, even if valid-looking ref follows.
            with self.assertRaises(ValueError):
                decoder.decode(dict(ref, record_sequence=3))
        decoder = AuditDecoder()
        decoder.decode(h)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            decoder.decode(ref)
        self.assertIn('missing_record_sequence_not_physical_loss', decoder.last_diagnostics)
        with self.assertRaises(ValueError):
            decoder.decode(h)

    def test_v2_field_scope_and_changed_same_inference(self):
        decoder = AuditDecoder()
        decoder.decode(dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1'))
        for seq, commands in enumerate(([{'hardware_slot': 0, 'q': 1}],
                                         [{'hardware_slot': 0, 'q': 2, 'context': {
                                             'source_valid': False, 'allow_joint_projection': False,
                                             'rewrite_reasons': 8}}]), 1):
            row = dict(record(4, ident=0, output_attempt_id=seq, present=32,
                              source_commands=commands), schema_version=2, record_sequence=seq)
            self.assertEqual(decoder.decode(row)['source_commands'], commands)
        # Even a real FULL sequence for source cannot resolve final commands.
        with self.assertRaisesRegex(ValueError, 'unresolved.*final_commands_ref'):
            decoder.decode(dict(record(6, present=64, final_commands_ref=2),
                                schema_version=2, record_sequence=3))

    def test_v2_corrupt_full_never_becomes_a_reference_base(self):
        bad_commands = [{'q': 'broken'}, {'q': True}, {'dq': float('inf')},
                        {'tau': json.loads('1e309')}, {'kp': float('nan')},
                        {'hardware_slot': 0.5}, {'hardware_slot': True},
                        {'controller_status': 2**31}, {'context': None}, 'not-an-object']
        for cmd in bad_commands:
            with self.subTest(cmd=cmd):
                decoder = AuditDecoder()
                decoder.decode(dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1'))
                with self.assertRaises(ValueError):
                    decoder.decode(dict(record(6, present=64, final_commands=[cmd]),
                                        schema_version=2, record_sequence=1))
                self.assertEqual(len(decoder._snapshots), 0)
                with self.assertRaisesRegex(ValueError, 'unresolved'):
                    decoder.decode(dict(record(6, present=64, final_commands_ref=1),
                                        schema_version=2, record_sequence=2))
        decoder = AuditDecoder()
        decoder.decode(dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1'))
        full = dict(record(6, present=64, final_commands=[{'q': None, 'dq': -0.0, 'hardware_slot': -1}]),
                    schema_version=2, record_sequence=1)
        decoder.decode(full)
        restored=decoder.decode(dict(record(6, present=64, final_commands_ref=1),schema_version=2,record_sequence=2))
        self.assertIsNone(restored['final_commands'][0]['q'])
        self.assertEqual(json.dumps(restored['final_commands']), json.dumps(full['final_commands']))

    def test_v2_iterator_and_report_missing_not_zero(self):
        h = dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1')
        rows = [h, dict(record(6, present=64, final_commands=[{'hardware_slot': 0}]),
                        schema_version=2, record_sequence=1),
                dict(record(6, present=64, final_commands_ref=1), schema_version=2, record_sequence=2),
                dict(footer(2), schema_version=2)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'audit.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            expanded = list(iter_audit_records(p))
            self.assertEqual(expanded[1]['final_commands'], expanded[2]['final_commands'])
            report = audit(p)
            self.assertEqual(report['counts']['slot_source_unknown_or_reset'], 2)
            self.assertFalse(any(k.startswith('invalid_line') for k in report['diagnostics']))
            with self.assertRaisesRegex(ValueError, 'line_limit'):
                list(iter_audit_records(p, max_line_bytes=3))
            # Dropping the FULL row cannot turn the ref into an empty command array.
            p.write_text(''.join(json.dumps(r)+'\n' for r in (rows[0], rows[2], rows[3])))
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                list(iter_audit_records(p))
            report = audit(p)
            self.assertNotIn('slot_source_unknown_or_reset', report['counts'])
            self.assertIn('missing_record_sequence_not_physical_loss', report['diagnostics'])
            self.assertTrue(any('unresolved command reference' in k for k in report['diagnostics']))
        report = self.run_capture([header(), record(1, ident=0), record(2, ident=0), footer(2)])
        self.assertEqual(report['counts']['observation_result_pairs'], 1)
        self.assertFalse(any('unassigned' in k for k in report['diagnostics']))

    def test_v2_sequence_hole_recovery_and_v1_iterator(self):
        h = dict(header(), schema_version=2, command_encoding='last_full_snapshot_ref_v1')
        decoder = AuditDecoder()
        decoder.decode(h)
        decoder.decode(dict(record(6, present=32, source_commands=[]), schema_version=2, record_sequence=1))
        # A missing row may have changed the most recent full snapshot: fail closed.
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            decoder.decode(dict(record(6, present=32, source_commands_ref=1), schema_version=2, record_sequence=3))
        self.assertEqual(len(decoder._snapshots), 0)
        decoder.decode(dict(record(6, present=32, source_commands=[]), schema_version=2, record_sequence=4))
        decoder.decode(dict(record(6, present=32, source_commands_ref=4), schema_version=2, record_sequence=5))
        decoder.decode(dict(footer(4), schema_version=2))
        with self.assertRaisesRegex(ValueError, 'after footer'):
            decoder.decode(dict(record(6), schema_version=2, record_sequence=6))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'v1.jsonl'
            rows = [header(), record(), footer(1)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(list(iter_audit_records(p)), rows)
            p.write_text(json.dumps(header())+'\n')
            with self.assertRaisesRegex(ValueError, 'missing_footer'):
                list(iter_audit_records(p))
            p.write_text(json.dumps(header()))
            with self.assertRaisesRegex(ValueError, 'truncated'):
                list(iter_audit_records(p))

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'optional ONNX dependencies absent')
    def test_cpu_replay_and_action_mismatch(self):
        import onnx
        from onnx import helper, TensorProto
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'constant.onnx'
            tensor = helper.make_tensor('raw', TensorProto.FLOAT, [1, 28], [2.0]*28)
            graph = helper.make_graph([helper.make_node('Constant', [], ['action'], value=tensor)], 'synthetic',
                [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1,1224])],
                [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1,28])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
            model.ir_version = 8
            onnx.save(model, p)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            meta = {'engine_id': 2**64-1, 'default_q':[1]*28, 'action_scale':[3]*28,
                    'embedded_contract_text':json.dumps({'action_semantics':'default_plus_scale_times_raw_action_no_clipping'}),
                    'model_sha256_before_load':digest, 'model_sha256_after_load':digest}
            rows = [header(meta), record(1, present=4, observation=[0]*1224),
                    record(2, present=24, raw_action=[2]*28, decoded=[7]*28), record(3), footer(3)]
            report = self.run_capture(rows, onnx=p)
            self.assertEqual(report['counts']['onnx_raw_matches'], 1)
            self.assertEqual(report['counts']['decoded_matches'], 1)
            rows[2]['raw_action'][0] = 99
            report = self.run_capture(rows, onnx=p)
            self.assertIn('onnx_raw_action_mismatch', report['diagnostics'])
            self.assertIn('decoded_action_mismatch', report['diagnostics'])
            self.assertEqual(report['status'], 'PARTIAL')


if __name__ == '__main__':
    unittest.main()
