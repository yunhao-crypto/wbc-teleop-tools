# Native reference timeline candidate

Native implementation and follow-up fixes are integrated on
`VitaDynamics/vita-robot:feat/mimiclite-native-robot67-20261117` (the original
`feat/mimiclite-reference-timeline` feature is already included). The companion
WTT integration branch is `feat/mimiclite-student2000-wtt`. WTT does not run native inference and cannot repair the installed
controller merely by changing its shell entrypoint.

The new native/Python package must agree on
`mimiclite.reference_timeline.v1` and `linux_clock_monotonic`. A package marker is
only packaging provenance; the launcher must verify the running native runtime's
capability before following. Do not use an old package/binary and assume the new
time-axis behavior exists. Installing a candidate, restarting, collecting data
and hardware validation remain separate operator decisions.

This candidate is intended to remove packet-held reference time: native samples
on its inference clock only when the entire model past/future window is covered.
It must never manufacture missing future frames, refresh stale source stamps,
queue obsolete reference packets to catch up, or silently re-engage after guard
failure. Sending, status/business queries and JSON file writing use separate
bounded paths, with explicit timing/drop diagnostics.

No heavier low-pass filtering, PD changes, new model weights or robot excitation
is part of this work. Software timing and reference-consumption tests do not
establish ground-contact stability or physical actuator calibration. See the
native delivery notes for exact implemented scope, validation and open gates.

## Required support and source logging follow-up

Native's Python builder now trims only optional future headroom at a gap;
complete `[-160ms,+80ms]` support and the50ms segment guard remain mandatory.
Frozen source mapping, the100ms transport reserve and watchdogs are unchanged.
A real required-window gap still disarms, without fabrication or automatic re-arm.
Builder decision evidence is captured before cleanup RPC, rather than treating
its later loss-log timestamp as the failure instant.

`actor_pico_worker` publishes each existing reference snapshot before enqueueing
source evidence to `actor_pico_source_writer.SourceWriter`. The writer owns data
copies, has a64-row queue, and alone opens/JSON-serializes/writes/flushes the file
on a SCHED_OTHER background thread. Saturation drops evidence with counters,
never a reference to rescue its log. Close waits at most0.5s; blocked kernel IO
can outlive that wait. Accepted data may remain unwritten if the process exits.
Footer `complete` means `accepted_records_drained`, not full source coverage,
confirmed close or durability. Open/write failure cannot skip existing cleanup.

GMR frame timing adds postprocess/publish-call/log-preparation/enqueue duration;
background records expose prior JSON/write costs. Capture receipt/available and
retarget wall/CPU definitions remain unchanged. `reader_frame_sequence`,
`reader_sequence_delta`, `reader_frames_observed_not_retargeted` and
`reader_sequence_nonincreasing` report observed identities/skips/regressions;
unknown identity stays null/unknown, never invented as zero or described as
network loss. Snapshot log counters give
an independent view if the source evidence file is full/unwritable. No unbounded
raw-reader queue is introduced: stale poses are not replayed to catch up.

Native audit file schema2 plus bounded v1/v2 WTT reader are documented in
`README_NATIVE_AUDIT.md`. Use updated tooling for reference expansion; raw
reference records are not self-contained command arrays.

Offline validation:187 WTT tests +81 subtests passed, including blocked writer,
saturation, ownership, bounded close, error cleanup, continued fake-source
publication, strict decoder and packaging tests. One cross-policy model-dependent
case is explicitly excluded because its pinned Sonic/SuperMimic artifacts are
unavailable; no model was substituted. Full model-dependent playback suites are
not claimed. The original retained audit re-encoded offline is72.33% smaller,
with all48,952 records preserved by independent public-decoder comparison; this
is not measured on-board throughput or recovery of historical missing records.

See native `REFERENCE_PIPELINE_FIX_DELIVERY.md` for detailed tests and gaps.
The9 historical candidate gaps avoid only premature extra-headroom rejection;
all still reject on entering required support. Async logging removes a blocking
path, but neither its causal contribution nor continuous follow/vibration
improvement is established. The new native edits still require production
ARM/Bazel build/package validation and separately approved supported hardware
trials; no deployment or robot connection was performed by this repair.
