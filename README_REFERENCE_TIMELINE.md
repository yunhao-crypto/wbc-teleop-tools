# Native reference timeline candidate

Native implementation is developed separately on
`VitaDynamics/vita-robot:feat/mimiclite-reference-timeline`, based on the opt-in
audit branch. WTT does not run native inference and cannot repair the installed
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
