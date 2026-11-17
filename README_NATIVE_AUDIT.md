# Native audit review candidate

Companion native branch: `VitaDynamics/vita-robot:feat/mimiclite-native-online-audit`.
The native implementation still requires production build/link and integration
verification. Do not install it solely because these workstation tests passed.

## Offline report (no robot connection)

```sh
python scripts/audit_mimiclite_native.py capture.jsonl --onnx policy.onnx > report.json
```

The reader preserves integer timestamps, bounds correlation storage, diagnoses
missing/truncated/dropped records and verifies optional recorded-observation→CPU
ONNX/raw-action→decoded numerics. It does not reconstruct actual online
observation/history independently, certify fixed-NPZ equivalence, or prove CAN
completion/mechanical response. Reports remain `PARTIAL`; exit 0 only means a
report was produced. Read numerical mismatch and structural diagnostics.

## Bounded v1/v2 offline decoder

`actor_native_audit.validate(row)` checks individual raw **or expanded** v1/v2
rows; it does not resolve references. Use `AuditDecoder.decode(row)` for ordered
parsed rows, or `iter_audit_records(path, max_line_bytes=2*1024*1024)` for bounded
JSONL streaming. The iterator yields header, expanded records, then footer;
consume it fully to check the footer. Invalid/unresolved/truncated input raises
`ValueError`. Each iterator/decoder is scoped to one capture; create a fresh one
for another file. Direct decoder callers must bound their own input row sizes.

V2 requires header `command_encoding="last_full_snapshot_ref_v1"` and positive
uint64 `record_sequence` on every record, starting at 1 and strictly increasing.
A present source/final command field has exactly one full array or corresponding
`source_commands_ref` / `final_commands_ref`. Each ref must point strictly
backwards to the **most recent FULL** of that same field, engine and capture.
Refs do not advance the base. Absent fields do not erase a known base. Two
snapshots only are cached; returned data never aliases these caches. Unknown or
invalid bases are not replaced with empty arrays, zero identities or other
fields' snapshots. Genuine zero identities remain valid integers.

Sequence holes are explicit `missing_record_sequence_not_physical_loss`
diagnostics and invalidate both bases (a missing row could have replaced one).
The iterator raises on holes by default; an optional `on_diagnostic` callback
can accept a hole followed by a new full snapshot. Malformed/unresolved rows
always raise. The report API continues with explicit `invalid_line` diagnostics,
clears bases after errors, and never counts unresolved commands as output
evidence. New full snapshots can re-establish bases. Mixed schemas, repeated or
reversed sequences, cross-capture/engine refs and rows after footer are invalid.
V1 is still accepted unchanged; old v1-only readers reject v2 rather than silently
misreading it. Footer `complete` still means accepted records drained, not full
inference/control coverage. Extra recorder counters are forward-compatible;
existing drop counters remain evidence loss, not physical command loss.

These are offline reader checks, not board-side performance or hardware safety
validation. Companion integration branch: `feat/mimiclite-native-robot67-20261117`;
WTT: `feat/mimiclite-student2000-wtt`.

## Operator capture, only after separate native acceptance

`start_mimiclite_native.sh --audit` is an explicit **live** startup option, not a
read-only inspector. It requires the new package marker and runtime capability;
old packages reject it. `--plan` remains read-only even with `--audit` and never
starts/checks the package. The owning launcher waits for recorder readiness
before reference following; initial activation/prewarm frames are not covered.
No automatic installation/restart is added. See `README_MIMICLITE.md` and native
`tools/MIMICLITE_NATIVE.md` for capability configuration and collection workflow.

Timeline companion: native branch `feat/mimiclite-reference-timeline`, WTT
`feat/mimiclite-reference-timeline-wtt`. The reader accepts optional bit24
`timeline_sampling` (exact tick/coverage/offset/source-origin integers). A guard's
recorded undercoverage remains evidence with a diagnostic, not malformed data or
inferred physical loss. Source epoch identity is not a host-clock timestamp.
See `README_REFERENCE_TIMELINE.md` in the source checkout and native
`REFERENCE_TIMELINE_DELIVERY.md` for implementation and outstanding gates.

Local timeline regression: 166 tests +29 subtests passed (one upstream pytest
deprecation warning), including existing audits/V2 tests, native reader/CPU ORT,
entrypoint guards and offline packaging. This is not native binary validation.
