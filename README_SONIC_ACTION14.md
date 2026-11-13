# Native SONIC action14 reference-only Pico/GMR

`scripts/start_sonic_action14.sh --plan` and `--help` read only the local JSON contract. They do not source `/app/script/env.sh`, take the operator lock, open a device, run inference, or start Pico. On the robot, `scripts/start_sonic_action14.sh` with no extra flags is the one-click live entry: it confirms motion, follows the Pico head, and uses `bin/humanoid_tracking_client`. If the robot is in PASSIVE or QUIET_PASSIVE, the entry waits out that non-interruptible damping curve and sends one FIXED_STAND request, then activates action14 only after a fresh FIXED_STAND or HUMANOID status. Other actions are left as they are. Extra arguments such as `--seconds` or `--gripper-config` are kept. `--fixed-neck` remains available only when passed explicitly together with `--confirm-motion` and `--bridge`. `--follow-head` publishes Pico neck targets on the same reference; the native profile keeps neck kp/kd at 20/1 and follows that q only while it is finite and inside joint limits. Optional `--gripper-config` attaches the existing OP3 runner to the action14 transport and does not request EXTERNAL. `--plan` can validate a gripper JSON file and still does not construct a client. Existing `scripts/start_actor_pico.sh` (EXTERNAL) is unchanged and is not an automatic fallback.

The live shell, when it is eventually used on the robot, sources the same `/app/script/env.sh` and sets the same `PYTHONPATH`, `LD_LIBRARY_PATH`, Pico build path and operator lock as the EXTERNAL entry. `--plan` and `--help` skip all of that and use the system interpreter so an ARM `venv/bin/python` is not executed on the development host.

`sonic_action14_source.py` uses the pinned GMR solver with `SimpleNamespace(names=28 joints)` and `math.quaternion`. It does not construct ActorModel, an ONNX session, or a policy future window. `load_source_config` checks pinned GMR artifacts and skips the ONNX file. The approved model SHA256 remains `2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d`. The worker's latest-value queue can drop source frames; dropped frames are not replayed and are not given a new receipt.

`sonic_action14_resample.py` builds a causal 20 ms grid. Root orientation uses shortest-path normalized lerp (nlerp), not the original WTT slerp, so the grids are not claimed to be numerically identical. Each output keeps a receipt on the segment between the two real Pico receipts. A source gap above 600 ms fails closed. The playback queue is a 150 ms source jitter buffer, not a second policy future window.

`sonic_action14_orchestrator.py` activates action14 once, then keeps `active` (native stand) separate from `following` (Pico stream). A trigger engages only after this process has published a reference and `humanoid_tracking.query` reports a fresh accepted frame. `stream_start` refusal, including `no_stream`, stays in stand and does not queue another engage. The next follow needs a new trigger. Overdue grid slots are dropped and the newest due pose is published as the next contiguous sequence. A retained grid index would be a C++ sequence gap and would freeze the hold. Ownership is re-read at 100 ms. RobotInfo is checked once a second after the due frame, with a 50 ms query cap, so neither call runs on every 20 ms publish.

Ctrl-C and SIGTERM send one firmware `cancel` (PASSIVE) and return only after a fresh status names `PASSIVE`. That drops stiffness. A `--seconds` exit still requests `humanoid_tracking.return_to_stand` and leaves the native policy running. Disengage requests `humanoid_tracking.return_to_stand` only while the same request still correlates. The result is `lead_in_accepted` only when the business payload says a stand lead-in started and follow did not resume. Timeout, foreign ownership, or `stay_state=hold_measured` is `ambiguous`. The CLI prints `ACTION14_POSTURE_AMBIGUOUS` in that case and does not report that the robot is standing. A native action can keep inferring after the client exits. Ownership checks are best-effort: the bridge reports `exclusive_lease=false`, and the gap between a status check and a later publish remains.

`humanoid_tracking.query` is where capability fields are read: raw affine mode, margin, auto start/resume disabled, and `native_runtime_config` equal to `actions/policy_profiles/vt_human_v2_tracking_asample_wtt_native.yaml`. Margin 1.5 rad is only the explicit ankle PD-target envelope. It does not make the raw affine targets equivalent to the original EXTERNAL path when those targets fall outside the envelope.

Pure checks:

```bash
PYTHONPATH=src:venv/lib/python3.10/site-packages \
  python3.10 -m unittest discover -s tests -p 'test_sonic_action14*.py'
bash scripts/start_sonic_action14.sh --plan
```

Use the system `python3.10`. `venv/bin/python` in this tree is ARM and does not run on the development host. Hardware, inference, and motion checks need a person present and a separate authorization.
