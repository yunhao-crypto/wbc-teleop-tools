"""Synthetic offline checks; no robot/native runtime calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deployment.vita_s100_aorta_30dof.actor_native_audit import audit, validate


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
