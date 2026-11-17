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
