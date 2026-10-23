"""Adversarial trace checks: no SDK or robot dependencies."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/audit_command_delivery.py"
spec = importlib.util.spec_from_file_location("delivery_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def command(ident, source):
    start = source + 5000
    return dict(
        command_id=ident,
        state_real_us=source,
        cycle_skipped_ticks=0,
        sdk_reject_count=4,
        sdk_publish_started_monotonic_ns=start * 1000,
        timing_wall_ns=dict(
            sdk_publish_started_wall_ns=start * 1000,
            sdk_returned_wall_ns=(start + 1000) * 1000,
            state_received_wall_ns=(source + 1000) * 1000,
            cycle_started_wall_ns=(source + 2000) * 1000,
        ),
    )


def echo(ident, adopted):
    row = np.zeros(13)
    row[3], row[5], row[9] = adopted + 2000, ident, adopted
    row[4] = adopted + 3000
    return row


class DeliveryTest(unittest.TestCase):
    def test_mean_target_is_separate_from_twenty_ms_tail_gate(self):
        for delay, passes in [(16999, True), (17000, False), (19000, False)]:
            r = module.audit([command(1, 100000)], [echo(1, 100000 + delay)])
            self.assertTrue(r['trace_gate_pass'])
            self.assertEqual(r['latency_target_pass'], passes)
        r = module.audit([command(1, 100000), command(2, 120000)], [echo(1, 110000), echo(2, 140001)])
        self.assertTrue(r['mean_target_pass'])
        self.assertFalse(r['latency_target_pass'])

    def test_exactly_twenty_fails_not_rounded_to_pass(self):
        rows = [command(1, 100000), command(2, 120000)]
        r = module.audit(rows, np.stack([echo(1, 119999), echo(2, 140000)]))
        self.assertFalse(r["numeric_latency_pass"])
        self.assertEqual([x["command_id"] for x in r["exceedances"]], [2])

    def test_missing_command_remains_in_denominator(self):
        r = module.audit([command(1, 100000), command(2, 120000)], [echo(1, 110000)])
        self.assertEqual((r["motion_rows"], r["matched_rows"]), (2, 1))
        self.assertFalse(r["trace_gate_pass"])

    def test_wrap_and_first_cache_not_later_echo(self):
        rows = [command(65535, 100000), command(1, 120000), command(1, 1000000)]
        low = [echo(65535, 110000), echo(1, 134000), echo(1, 130000), echo(1, 1010000)]
        r = module.audit(rows, low)
        self.assertTrue(r["trace_gate_pass"])
        self.assertEqual(
            [x["adopted_us"] for x in r["matched"]], [110000, 130000, 1010000]
        )
        self.assertFalse(r["complete_goal_proven"])

    def test_large_tail_is_reported_and_outside_window_is_missing(self):
        r = module.audit([command(1, 100000)], [echo(1, 167292)])
        self.assertAlmostEqual(r["latency_ms"]["source_to_bottom_ms"]["max"], 67.292)
        r = module.audit([command(1, 100000)], [echo(1, 400000)])
        self.assertEqual(len(r["unmatched"]), 1)
        self.assertFalse(r["numeric_latency_pass"])

    def test_clock_order_duplicate_and_bad_lowstate_cannot_pass(self):
        row = command(1, 100000)
        r = module.audit([row, row], [echo(1, 110000)])
        self.assertFalse(r["numeric_latency_pass"])
        r = module.audit([row], [echo(1, 104000)])
        self.assertFalse(r["numeric_latency_pass"])
        low = echo(1, 110000)
        low[9] = float("nan")
        r = module.audit([row], [low])
        self.assertEqual(r["malformed_lowstate_rows"], 1)
        self.assertFalse(r["numeric_latency_pass"])

    def test_counter_increase_and_skipped_tick_are_not_latency_success(self):
        rows = [command(1, 100000), command(2, 120000)]
        rows[1]["sdk_reject_count"] = 5
        rows[1]["cycle_skipped_ticks"] = 1
        r = module.audit(rows, [echo(1, 110000), echo(2, 130000)])
        self.assertTrue(r["numeric_latency_pass"])
        self.assertFalse(r["trace_gate_pass"])
        self.assertEqual(r["sdk_reject_increases"], 1)

    def test_new_frame_can_arrive_after_loop_start(self):
        row = command(1, 100000)
        row["timing_wall_ns"]["state_received_wall_ns"] = 104000 * 1000
        self.assertTrue(module.audit([row], [echo(1, 110000)])["trace_gate_pass"])

    def test_echo_state_can_precede_command_update_but_not_publication(self):
        row = echo(1, 110000)
        row[3] = 109000
        self.assertTrue(module.audit([command(1, 100000)], [row])["trace_gate_pass"])
        row[4] = 109500
        self.assertFalse(module.audit([command(1, 100000)], [row])["trace_gate_pass"])

    def test_other_control_modes_split_interval_not_motion_latency(self):
        rows = [command(1, 100000), command(5, 500000)]
        rows[0]["control_ordinal"] = 1
        rows[1]["control_ordinal"] = 5
        r = module.audit(rows, [echo(1, 110000), echo(5, 510000)])
        self.assertEqual(r["matched_rows"], 2)
        self.assertIsNone(r["publish_entry_interval_ms"])

    def test_empty_or_offline_replay_is_not_evidence(self):
        self.assertFalse(module.audit([], np.empty((0, 13)))["numeric_latency_pass"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "control.jsonl"
            p.write_text(
                json.dumps(dict(mode="motion", record_type="offline_control")) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Offline replay"):
                module.read_controls(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
