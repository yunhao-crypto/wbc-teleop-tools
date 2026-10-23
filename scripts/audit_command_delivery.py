"""Audit real Motion command delivery using control JSONL and decoded LowState.

The NPZ schema is the 13-column `low` array from the archived MCAP extractor:
record_seconds, state_id, input_cmd_id, ts_state_real, ts_state_pub, cmd_id,
input_state_id, input_state_ts_real, ts_infer_input_state,
ts_input_cmd_into_motor, ts_before_infer, ts_after_infer, ts_consumed_us.
All columns except record_seconds are in their native integer/us units.

This checks timestamp evidence, not CAN completion, mechanical stability, model
identity, or the correctness of the upstream MCAP decoder. No robot imports.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def statistics(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    return dict(
        n=len(x),
        mean=float(x.mean()),
        p95=float(np.percentile(x, 95)),
        p99=float(np.percentile(x, 99)),
        max=float(x.max()),
        min=float(x.min()),
        ge20=int(np.sum(x >= 20)),
    )


def read_controls(path):
    rows = []
    ordinal = 0
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            if row.get("record_type") == "control":
                ordinal += 1
            if row.get("mode") != "motion":
                continue
            if (
                row.get("record_type") == "offline_control"
                or row.get("action_sent") is False
            ):
                raise ValueError("Offline replay cannot establish command delivery")
            if row.get("record_type") == "control":
                rows.append(dict(row, source_line=line_number, control_ordinal=ordinal))
    return rows


def audit(rows, low, match_window_ms=200):
    low = np.asarray(low, dtype=float)
    if low.ndim != 2 or low.shape[1] != 13:
        raise ValueError("Expected archived 13-column LowState schema")
    if not 20 < match_window_ms <= 1000:
        raise ValueError(
            "Association window must be (20,1000] ms, not a success threshold"
        )
    echoes = defaultdict(list)
    malformed_lowstate = 0
    for index, row in enumerate(low):
        if np.isfinite(row[[3, 4, 5, 9]]).all() and row[5] == int(row[5]):
            echoes[int(row[5])].append(index)
        else:
            malformed_lowstate += 1
    echoes = {k: np.asarray(v, dtype=int) for k, v in echoes.items()}
    matched, unmatched, invalid = [], [], []
    starts, returns, offsets, reject_counts = [], [], [], []
    ordinals = []
    seen = set()
    next_same_id = {}
    upper = [float("inf")] * len(rows)
    for index in reversed(range(len(rows))):
        row = rows[index]
        ident = row["command_id"]
        upper[index] = next_same_id.get(ident, float("inf"))
        next_same_id[ident] = (
            row["timing_wall_ns"]["sdk_publish_started_wall_ns"] / 1000
        )
    for index, row in enumerate(rows):
        ident, source = row["command_id"], row["state_real_us"]
        ref = dict(index=index, command_id=ident, source_line=row.get("source_line"))
        if type(ident) is not int or not 1 <= ident <= 65535:
            invalid.append(dict(ref, reason="invalid_command_id"))
            continue
        wall = row["timing_wall_ns"]
        start = wall["sdk_publish_started_wall_ns"] / 1000
        end = wall["sdk_returned_wall_ns"] / 1000
        callback = wall["state_received_wall_ns"] / 1000
        cycle = wall["cycle_started_wall_ns"] / 1000
        key = (ident, start)
        if key in seen:
            invalid.append(dict(ref, reason="duplicate_control_record"))
        seen.add(key)
        if not np.isfinite([source, start, end, callback, cycle]).all():
            invalid.append(dict(ref, reason="nonfinite_timestamp"))
            continue
        # A fresh callback may arrive after the loop starts but before read().
        if not (0 < source <= callback <= start <= end and 0 < cycle <= start):
            invalid.append(dict(ref, reason="client_timestamp_order"))
        if starts and start <= starts[-1]:
            invalid.append(dict(ref, reason="nonincreasing_publish_time"))
        starts.append(start)
        returns.append(end)
        ordinals.append(row.get("control_ordinal", index))
        if row.get("sdk_publish_started_monotonic_ns") is not None:
            offsets.append(
                (
                    wall["sdk_publish_started_wall_ns"]
                    - row["sdk_publish_started_monotonic_ns"]
                )
                / 1e6
            )
        if row.get("sdk_reject_count") is not None:
            reject_counts.append(row["sdk_reject_count"])
        indices = echoes.get(ident, np.empty(0, dtype=int))
        stamps = low[indices, 9]
        # Temporal association only; late/missing commands fail coverage rather
        # than being removed from the control denominator. No 20 ms prefilter.
        selected = indices[
            (stamps > 0)
            & (np.abs(stamps - start) < match_window_ms * 1000)
            & (stamps < upper[index])
        ]
        if not len(selected):
            unmatched.append(dict(ref, source_us=source, sdk_begin_us=start))
            continue
        j = int(selected[np.argmin(low[selected, 9])])
        adopted = float(low[j, 9])
        # This packet's motor-state capture can precede its command update.
        # The cache timestamp must precede publication, not state capture.
        if not start <= adopted <= low[j, 4]:
            invalid.append(dict(ref, reason="bottom_timestamp_order", lowstate_row=j))
        matched.append(
            dict(
                ref,
                state_real_us=source,
                sdk_begin_us=start,
                adopted_us=adopted,
                lowstate_row=j,
                source_to_bottom_ms=(adopted - source) / 1000,
                sdk_to_bottom_ms=(adopted - start) / 1000,
                sdk_return_to_bottom_ms=(adopted - end) / 1000,
                source_to_callback_ms=(callback - source) / 1000,
                callback_to_cycle_ms=(cycle - callback) / 1000,
                cycle_to_sdk_ms=(start - cycle) / 1000,
            )
        )
    exceedances = [r for r in matched if r["source_to_bottom_ms"] >= 20]
    reject_increases = sum(
        max(0, b - a) for a, b in zip(reject_counts, reject_counts[1:])
    )
    skipped = sum(int(r.get("cycle_skipped_ticks", 0)) for r in rows)
    offset_span = float(np.ptp(offsets)) if offsets else None
    clock_review = offset_span is None or offset_span > 1.0
    contiguous = np.diff(ordinals) == 1
    numeric_pass = (
        bool(rows)
        and len(matched) == len(rows)
        and not invalid
        and not exceedances
        and malformed_lowstate == 0
    )
    trace_pass = numeric_pass and not clock_review and skipped == 0 and reject_increases == 0
    mean_latency = float(np.mean([r['source_to_bottom_ms'] for r in matched])) if matched else None
    return dict(
        endpoint="Source state to same cmd_id first observed ts_input_cmd_into_motor; not CAN/motor completion",
        scope="Decoded timestamp audit; MCAP decoder provenance, clock semantics and physical result require separate review",
        motion_rows=len(rows),
        matched_rows=len(matched),
        unmatched=unmatched,
        malformed_lowstate_rows=malformed_lowstate,
        invalid=invalid,
        exceedances=exceedances,
        matched=matched,
        latency_ms={
            k: statistics([r[k] for r in matched])
            for k in (
                "source_to_bottom_ms",
                "sdk_to_bottom_ms",
                "sdk_return_to_bottom_ms",
                "source_to_callback_ms",
                "callback_to_cycle_ms",
                "cycle_to_sdk_ms",
            )
        },
        publish_entry_interval_ms=statistics((np.diff(starts) / 1000)[contiguous]),
        publish_return_interval_ms=statistics((np.diff(returns) / 1000)[contiguous]),
        motion_interval_scope="Consecutive Motion control records; intervening other modes split intervals, no tail filtering",
        skipped_ticks=skipped,
        sdk_reject_increases=reject_increases,
        realtime_minus_monotonic_span_ms=offset_span,
        clock_review_required=clock_review,
        clock_review_note="Missing clock pairs or >1ms offset span is a review flag, not clock correction",
        numeric_latency_pass=numeric_pass,
        trace_gate_pass=trace_pass,
        mean_target_ms=17,
        mean_target_pass=mean_latency is not None and mean_latency < 17,
        latency_target_pass=trace_pass and mean_latency is not None and mean_latency < 17,
        match_window_ms=match_window_ms,
        threshold_ms=20,
        threshold_comparison="strictly less",
        complete_goal_proven=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--lowstate-npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.lowstate_npz, allow_pickle=False) as data:
        result = audit(read_controls(a.control), data["low"])
    result["inputs"] = {
        name: dict(path=str(path.resolve()), sha256=sha256(path))
        for name, path in [("control", a.control), ("lowstate_npz", a.lowstate_npz)]
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("matched", "exceedances")},
            indent=2,
        )
    )
    return 0 if result["latency_target_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
