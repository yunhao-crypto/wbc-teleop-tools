# Vita02A policy + Pico on S100

This entry runs the reviewed joint-reference policy through the installed SDK
0.2.1 WBC observation/decoder math. The model is supplied separately and excluded
from Git. Its identity, installation path and SHA256 are pinned in
`config/actor_pico.json` and the delivery manifest. Supply the model matching
that contract; the example filename `policy.onnx` is only a local pathname.

The entry supports model checks, offline Pico replay, live-input shadow
inference, and a separately armed supported EXTERNAL trial. Shadow subscribes to Body31 feedback and reads Pico through its owned vendor
service. It writes computed targets to JSONL; it never creates a controller,
requests EXTERNAL or publishes a robot command. Only `live` requests control and moves joints. The existing P1/P2/P3 runtime,
launchers, SDK installation and protocol guards are not replaced.

## Policy selection

`start_actor_pico.sh` requires an explicit policy for every mode except
`configure`: `sonic` (`config/actor_pico.json`) or `supermimic`
(`config/actor_pico_supermimic.json`), given as the first word
(`start_actor_pico.sh supermimic shadow ...`), after the mode, or as `--policy`.
Without one, a terminal shows a menu with no default; an empty answer or a
non-interactive run starts nothing. Conflicting selections are rejected. A
missing/mismatched model of the selected policy is an error; there is no
substitute policy. (`scripts/actor_pico.py` itself still defaults to `sonic`
for installer and delivery checks.) Each configuration
pins its own external model; the shared GMR input, Body31 decode, gains, limits
and startup/live guards are unchanged.

| | `sonic` | `supermimic` |
|---|---|---|
| Model | `artifacts/actor_g1/actor_g1.onnx` | `artifacts/supermimic/supermimic_vita02a.onnx` |
| Input values | 1520 | 1670 (10x96 observation history + 10x71 goal frames) |
| Reference delay | 200 ms | 220 ms (10 future goal frames + 1 velocity tail frame) |
| Root input | quaternion only | quaternion plus GMR base position |

The SuperMimic model is the serial-ankle `resume_nopush` training checkpoint
(iteration 72188) exported by trackerlab-supermimic
`scripts/rls/supermimic/export_vita_deployment.py`; its observation/goal contract
is embedded in the ONNX metadata and checked at load. It needs the firmware's
virtual ankle PD path, exactly as the SONIC actor does. Provision it with
`bootstrap_actor_s100.py --policy supermimic --model /path/to/supermimic_vita02a.onnx`
and pass the same `--policy` to `check`, `replay`, `shadow` and `live`:

```bash
bash scripts/start_actor_pico.sh supermimic check --steps 100
bash scripts/start_actor_pico.sh supermimic shadow --seconds 60 --group default
bash scripts/start_actor_pico.sh supermimic        # default local motion entry
```

Offline parity: `tests/test_actor_supermimic.py` compares the deployed history,
goal and action with the training-side fixture. A trackerlab MuJoCo closed loop
(`validate_vita_deployment_mujoco.py`) compares the training policy, this
observation builder and the causal Pico window on the same clips. Neither is
hardware validation.
## Data path and calibration

```text
Atomic Pico 24x7 positions + xyzw quaternions
  -> original bridge XRobotStreamer Unity-to-GMR conversion
  -> original bridge Vita02A GMR (mink / MuJoCo / DAQP)
  -> named Vita02A joint selection + root wxyz
  -> model-specific 200 ms causal reference window
  -> SDK observation / ONNX / Body31 command
```

The solver, geometry helpers, coordinate conversion, `xrobot_to_vita02a.json`
and `vita02a_nomesh.xml` are copied byte-for-byte from motion-stream-bridge
commit `763f5607c3f0accb61a415cfd676af54b72e1232`. Hashes and provenance live in
`third_party/motion_stream_bridge/SOURCE.json`; actor configuration checks the
actual source/config/model files before creating control handles.
The old orientation-only IK has been removed. GMR retains its position and
orientation tasks, foot/toe targets, ground snap, posture regularizers,
input pose correction, joint limits and qpos smoothing with their original
settings. It uses shipped offsets; the first received pose is **not** declared
neutral or automatically calibrated onto the actor defaults. The bridge's
optional interactive recalibration UI and other action/button combos are not
part of this actor entry.

The actor runs this pinned solver through a subclass in `actor_pico_gmr.py`
that only removes work whose results GMR never reads: `mj_kinematics` +
`mj_comPos` instead of full `mj_forward` after ground snap and smoothing, a
direct DAQP call instead of the qpsolvers wrapper, inlined body FrameTask
residuals, a vectorized Unity conversion, and one residual evaluation for the
two identical IK tables. Its qpos, root and residual are bit-identical to the
pinned engine on the 384-frame Pico replay and the bridge golden frames
(x86 host: median 9.1 -> 6.2 ms, p99 14.5 -> 10.9 ms per frame). The subclass
requires mink 1.3.0, daqp 0.10.3 and qpsolvers 4.13.0 (the pinned wheelhouse);
other versions raise at startup instead of silently running the slower pinned
engine. Log headers record `gmr_engine`. The input worker no longer
sleeps 5 ms after each solved frame; it polls every 1 ms only while idle.

Operator height defaults to **1.75 metres**. Override it for every launch:

```bash
bash scripts/start_actor_pico.sh sonic --human-height 1.68
bash scripts/start_actor_pico.sh sonic shadow --seconds 30 --human-height 1.68
```

Height scales the original GMR human-body table and is recorded in the log
header. This selects the no-hand Vita02A model; no hand or neck tracking is
introduced. Joints are resolved by name, including wrist yaw/pitch, independently
of bridge wire order. Each engagement aligns reference yaw to current measured
robot yaw. Following logs include raw Pico frames and full reference windows. The input
worker additionally records every solved raw frame and complete GMR qpos in
`outputs/actor_pico_gmr_input_*.jsonl` before snapshot coalescing;
`gmr_task_error` is a weighted mixed position/orientation norm, **not radians**
and not an admission threshold.

This remains a GMR-to-G1-reference actor deployment, not the separate original
HoloMotion SMPL-input Motion/Velocity neural-policy pair. Source/numerical
parity does not establish live tracking quality or physical stability.

## Offline verification on this x86 host

Use the host Python, which has NumPy and ONNX Runtime. The ARM venv interpreter
cannot run on x86. Offline mode reads the installed SDK's Python math and DTOs
without loading its ARM inference extension or constructing any Aorta node.
For a fresh clone without an installed SDK, use the staged-export verifier:
`python3 scripts/verify_actor_delivery.py --host-inference --model /path/to/policy.onnx`.
It uses the explicit, hash-verified external model and versioned wheels/resources,
and materializes SDK sources in a temporary
export. The direct commands below also work in the original snapshot or an
environment where the SDK has already been installed at `venv/`.

```bash
cd /host/vita_p123_20261015_sdkmain_opt021
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 tests/test_actor_pico.py -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/actor_pico.py check --steps 100

python3 scripts/make_actor_pico_replay.py --output /tmp/actor_pico_input.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/actor_pico.py replay --input /tmp/actor_pico_input.npz \
  --steps 100 --record-observations
```

The regression tests compare the SDK Body31 observation builder against the
independent C++ golden observations inherited from `humanoid-sdk-tracking`.
They also compare 48 GMR frames against the original bridge solver,
causal reference timing/velocity, stale and invalid inputs, real model
inference, and Body31 ankle torque/neck/hole mapping. Fixture provenance is in
`tests/fixtures/actor_pico/tracking_fixture_provenance.json`.
The replay fixture is synthetic, and its output is not hardware validation.
The documented 100-step replay covers its initial segment. Finite large fit
residuals remain visible in diagnostics; they do not establish real-human
reference quality or physical balance.

## Run on S100

Clone the repository into a new directory on S100, then run:

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --model /path/to/policy.onnx
```
The 21 offline wheels, Pico service, binding, sources and synthetic replay
fixture are tracked in Git. The model must be supplied separately; its SHA256
is pinned in `config/actor_pico.json` and the delivery manifest.
The bootstrap creates a fresh environment and verifies/extracts the resources;
it does not borrow an existing development environment. See
[`deploy/actor_s100/README.md`](deploy/actor_s100/README.md) for prerequisites and
the optional `--build-pico` source build on S100. The launcher selects SDK
0.2.1's bundled native CPU ONNX runner on aarch64; a separate pip ORT install
is not needed. Neither the existing HoloMotion MoE backend nor BPU acceleration
is used by this actor. 50 Hz scheduling is configured, but S100 timing remains
unmeasured until an actual run; overruns and skipped ticks are logged.

The target must provide `/app/script/env.sh`, its matching vendor runtime, and
the reviewed `/app_param/zenoh/s100_session_peer.json5`. The launcher derives a
process-local session profile with the existing preparation script. It supplies
the relocated NumPy dependency path and Pico library paths explicitly rather
than relying on the old absolute `.pth` path. It shares `build/operator.lock`
with the existing operator entry.

From the **actual S100 deployment root**, run:

```bash
bash scripts/start_actor_pico.sh sonic check --steps 100
bash scripts/start_actor_pico.sh sonic state-check --seconds 5 --group default
bash scripts/start_actor_pico.sh sonic shadow --seconds 60 --group default
```

Connect the Pico XRoboToolkit app to the S100's reachable LAN IP, enable
Full body and Send, and keep the trackers calibrated. The shadow entry starts
and owns `RoboticsServiceProcess`; do not start a second service for that run.
It needs no Xbox. Ctrl-C ends shadow and closes its reader/subscriptions/service.

Each run creates a unique `outputs/actor_pico_*.jsonl`; the Pico service writes
`outputs/pico_service_actor.log`. `--record-observations` also records all 1520
input values. Successful inference records include `actions_sent=0`, the
retarget residual, input age, the 200 ms reference delay, inference timing and
all 31 inert output slots. An overrun record includes logging time. `check`
does not open the Pico service or check hardware inputs.
`state-check` subscribes to Body31 only and validates finite joint/IMU values,
protocol, roster and freshness. It creates no control handles or Pico service.
No feedback/timeout is reported as a failed check; it never starts robot services.

The bootstrap creates the interpreter with the proper permissions and restores
executable mode on the service from the sealed archive. The actor shadow entry
requires no Xbox configuration or Wi-Fi controller key.

## Default local startup

On the configured S100, from the deployment root, run:

```bash
bash scripts/start_actor_pico.sh sonic        # or: supermimic
```

**A policy name alone (or no arguments, then a menu choice) selects an attended
motion trial, not an offline check.** The policy has no default; see
[policy selection](#policy-selection). Use
`bash scripts/start_actor_pico.sh sonic check` for offline inference, or
`sonic state-check --seconds 5` for read-only feedback. The default motion entry uses
`outputs/actor_pico_local.json`, a private 0600 file provisioned for this board.
It binds the local IPv4 address, machine-id digest, selected end effector,
effective YAML digest and configured duration (`0` means until STOP). It stores no authorization.
Generate the file on the actual board after installation and read-only checks:

```bash
bash scripts/start_actor_pico.sh configure \
  --robot-id '<actual local S100 IPv4>' \
  --robot-config /absolute/path/to/running/effective.yaml
```

This command stores defaults only and sends no actions. It keeps an existing
different profile unless `--replace` is specified, in which case it saves a
private backup. It defaults to manual dual-trigger handoff and an unlimited
client duration; add `--seconds N` for a bounded default. Configuration does not
record authorization or prove that the supplied YAML is the running selection.

Another device, changed network identity/selector/YAML or missing profile fails
before robot handles. A generic bundle has no device profile; use explicit
`auto --robot-id <local-robot-ip> --robot-config <effective-yaml>` or provision
reviewed local defaults. Do not copy another robot's private profile.

The default displays target, actual gains and startup sequence, then requests
motion directly without a typed `MOVE` prompt. Reliable support, clear
movement/pinch zones, an attendant and usable E-stop remain required.
The sequence is:

1. Wait the configured PASSIVE or QUIET_PASSIVE damping duration plus two
   seconds on continuous fresh feedback (currently 42 s / 6 s on the selected
   board). SDK `action_phase=IDLE` is the EXTERNAL session phase, not damping
   completion. A retired keyframe can clear the action name; that alone does
   not count as FIXED_STAND readiness.
2. Request FIXED_STAND once, and require actual active FIXED_STAND plus healthy,
   hardware-bounded, stationary feedback (every joint |dq| <=0.2 rad/s) for 0.5 s.
   Policy-online startup also requires gyro norm <=0.10 rad/s. Measured joint
   positions may exceed advertised RobotInfo q ranges by 0.03 rad (hard stops,
   calibration); commanded q always uses the exact ranges. HIP IMU tilt is displayed and checked
   against the reviewed profile's unchanged 0.8 rad fall threshold; there is
   no additional 2-degree admission rule. This fall threshold alone does not
   prove a command is safe to apply. In `policy-online` mode the
   policy reference starts at the measured pose: exact zero joint positions
   are not an activation prerequisite. Progress and failure logs name the
   largest joint position error and speed; no request is retried after failure.
3. Preflight the first native policy command using the newly measured pose,
   cosine reference lead-in, real observations and configured heading alignment.
   Check hardware/output limits and feedback freshness before requesting
   EXTERNAL. Record inference exceeding the nominal 20 ms period. A failed preflight
   cancels startup and records the actual state.
4. Enter one SDK EXTERNAL session. Keep ONNX active throughout the reference
   lead-in into the model stand, using the pinned profile's 1.2 rad/s reference
   speed and 0.3--3 s duration. This matches the supplied tracking profile's
   `stand_on_activate` / online-reference design; it does not first drive an
   independent open-loop trajectory to the training pose. Raw measured q is
   not clipped to model output limits; model targets remain clamped to the
   trained/physical intersection. Check each command against RobotInfo ranges, state freshness,
   ownership/status and body tilt. Schedule at 50 Hz, record overruns and skip
   missed ticks rather than publishing a burst of accumulated commands.
5. Start the isolated Pico receiver automatically while ONNX stand continues.
   No Pico input is required for model inference. After `ONNX_READY` and
   `PICO_INPUT: ready` and `PICO_BUTTONS: controller input ready`, keep the
   release both triggers, then squeeze both past **0.7** to
   follow (at least 3 s of initial policy preparation is required). Release
   both and squeeze again to return smoothly to ONNX stand, without leaving
   EXTERNAL. A hold fires once; both must release to/below 0.7 to re-arm, with
   300 ms debounce. The local default never auto-switches to following. The session,
   command sequence and policy history are preserved. Early/invalid requests
   are rejected and not queued; source loss after handoff cancels without
   automatic motion retry/reconnection.

The live control thread uses the original onboard HoloMotion entry's
`SCHED_RR` priority 1. This is process-local; no robot service priorities or
persistent motor/network settings are changed. The isolated GMR input process
explicitly uses `SCHED_OTHER`, so it and the vendor service do not inherit the
control priority. Linux scheduling permission is required before any control
handle is created; failure stops startup rather than silently falling back.

Body31 feedback must be within 100 ms at read and again before publication.
As in the SDK WBC demo, the gate is local receipt age plus producer publication
age (`ts_state_pub_us`, CLOCK_REALTIME, at most 5 ms in the future); sensor
stamp age is reported only. A stale tick is skipped rather than ending the
trial: no command is published and the firmware keeps interpolating/holding
the last one (interpolation max 100 ms, watchdog 1000 ms). The third
consecutive stale tick cancels. Each `body_state_age_failure` record holds
phase, ages, compute/policy time and `action` (`skip_tick` or `cancel`).
A stale command is never published or retried.
If computation finishes on an expired state, its measured observation remains
in policy history, but `last_action` stays at the last successfully published
action. The next tick never uses an unsent candidate as its previous action.

Input checks follow the original HoloMotion deployment's
`config/vita02a_s100.yaml`: robot feedback age is 100 ms; Pico source/reference
age is 600 ms (`config/actor_pico.json:pico_max_age_s`), also used for controller
input and edge delivery. Source frame intervals are diagnostics, not data age:
a 110--122 ms interval does not clear a ready buffer or interrupt following.
Timestamped samples remain in the same causal window and are interpolated on
the model's 20 ms grid. There is no reference-epoch gate, segment-fit cutoff,
or 0.1 rad all-joint neutral admission rule. The model's 200 ms initial window,
finite/shape/quaternion checks and strictly increasing source timestamps remain.
While following, a reference older than `pico_hold_age_s` (250 ms) holds its
current frame with zero velocity (the training rule beyond a reference's end)
instead of replaying a frozen moving window; rows record `reference_hold`.
Pico age exceeding 600 ms or other genuine input faults stop following through
the existing checked exit; changing input thresholds does not copy HoloMotion's
separate Velocity-policy fallback into this single-actor deployment.

The trigger interaction follows motion-stream-bridge `sources/pico.py` at
763f560 (`_TRIGGER_ENGAGE=0.7`, `_TOGGLE_DEBOUNCE_S=0.3`, both-release latch).
This application reads a same-packet dual-trigger snapshot with native receipt
age in an independent input thread, then carries a monotonic event counter to
the control loop. Cached/stale inputs cannot engage; startup/recovery additionally
requires a fresh release. Returning to stand seeds the existing profile's
cosine lead-in from measured joints/orientation and preserves policy history.
This is an intentional reference transition; input faults during following still
cancel and never automatically resume. Trigger toggling is not an emergency stop.
No bridge repository/runtime is needed on S100. Terminal `PICO` remains an
optional diagnostic entry command; normal operation needs no typed handoff.

The reference for the activation design is the user-selected
`vt_human_v2_tracking_asample.yaml`: measured-pose reference, continuous policy
inference, heading alignment, metadata-derived reference timing and stand hold
before live references. This application still uses the separately supplied
configured ONNX policy and Body31 SDK EXTERNAL contract. It does not load the firmware
HUMANOID_TRACKING model or copy its legacy 42-slot/finger/planner interfaces.
Model hash, 20 ms reference spacing, gains, order and native/schema guards
remain pinned by this standalone delivery; no monorepo is needed on S100.

For an explicit target, `auto` selects the same online-reference entry:

```bash
bash scripts/start_actor_pico.sh supermimic auto --seconds 120 \
  --robot-id 10.100.23.96 \
  --robot-config /app/locomotion/config/locomotion_foot_humanoid_asample_none.yaml
```

Startup always runs the policy on the measured-pose reference. The earlier
independent `joint-ramp` experiment was removed; `--entry-policy policy-online`
remains accepted for existing launch commands. `live --manual-pico` retains manual
receiver connection with `CONNECT`. `--auto-pico` is an explicit opt-in to
following a fresh valid reference automatically; it is not enabled by this
board's selected local profile.

The configured local default uses `seconds=0`: no application duration expiry.
Use `bash scripts/start_actor_pico.sh sonic --seconds 300` for a five-minute run;
an explicit duration overrides the local profile. Positive duration counts from
EXTERNAL entry; damping wait and fixed-stand activation are additional bounded
time. `STOP` or Ctrl-C sends
checked cancellation to PASSIVE. Incomplete startup cancels only a recognized
non-EXTERNAL state/generation, records incomplete exits and never retries
motion. An explicit positive duration also cancels to PASSIVE when it expires;
the summary reports `stop_reason=duration_elapsed`. Operator stop reports
`stop_reason=operator_stop`. EXTERNAL has no ownership expiry; PASSIVE removes position stiffness
and can let loaded limbs fall. Keep support throughout exit. Reference parity,
mock-I/O tests and native read-only inference do not prove physical balance:
actual standing, full-loop timing and Pico handoff remain `not_measured` until
an attended hardware run. No service restart or persistent motor change is
part of this launcher.

## Supported EXTERNAL trial

The deployed Body31 firmware takes command ranges from `get_robot_info`, not
YAML `command_limits` (which it rejects). The client checks supported fields
and every advertised range, including model and safe-hold gains. For the four
A-sample parallel ankle axes, `supported_fields=31, valid_limits=1` means only
q has a joint-domain range; dq/tau/kp/kd remain validated by the firmware's
motor-space adapter. Those absent ranges are not zero limits. Other populated
axes must advertise all five ranges; missing or degraded capabilities fail
before requesting ownership. The full A-sample YAML still supplies watchdog,
recovery, interpolation, safe-hold and virtual-PD intent. Do not add old limits
to robot YAML or weaken output/hardware checks to get past preflight.

`live` is experimental **supported** control. It is not acceptance
for unsupported standing/walking. Obtain authorization for the actual robot
and motion; keep reliable support, clear motion/pinch zones, an attendant and
a usable hardware emergency stop. Start from a steady capture pose for the
GMR shipped offsets and use small movements during initial checks.
The 200 ms reference delay remains for Pico following. `--stand-only` still moves the whole body:
it runs the online actor with a default-stand reference, not a fixed pose.
It requires no Pico device/service/reader. Body feedback and all control guards
remain mandatory; an entry pose outside physical hardware limits is rejected.

After installing into a new S100 directory, run the existing `check` and
bounded `shadow` commands first. Verify the matching robot release/schema,
the full A-sample hardware configuration and the **running effective** YAML.
Supply its absolute path below; do not use an old checkout/config snapshot.
This program reads that file but cannot prove which file a running service
loaded. Missing/disabled `parallel_ankle.pd.strategy=virtual_pd` or
`actions.EXTERNAL.params.ankle_virtual_pd_intent=true` stops the trial.
It never edits the file, restarts services or activates an OTA slot.

From the actual deployment root, for a separately authorized preparation test:

```bash
bash scripts/start_actor_pico.sh sonic live --stand-only --seconds 5 \
  --robot-id '<actual robot identity>' --confirm-motion \
  --robot-config /absolute/path/to/running/locomotion.yaml
```

After reviewing that test and authorizing Pico following, omit `--stand-only`:

```bash
bash scripts/start_actor_pico.sh sonic live --seconds 10 \
  --robot-id '<actual robot identity>' --confirm-motion \
  --robot-config /absolute/path/to/running/locomotion.yaml
```

This immediate-following mode uses the same isolated Pico/GMR input process as
the manual mode, without the controller-trigger reader. It connects on the first
Body31 feedback during preflight; the control loop never runs GMR and only reads
the latest reference snapshot. Preflight waits for a ready reference window, and
following starts after the stand preparation with yaw aligned to the measured
robot heading. A failed worker or stale snapshot cancels the trial.

To enter ONNX stand first and manually attach Pico in the **same EXTERNAL
session**, use the separately armed interactive mode (on S100, from the actual
deployment root on persistent `/userdata`, not overlay `/root`):

```bash
bash scripts/start_actor_pico.sh sonic live --manual-pico --seconds 120 \
  --robot-id '<actual robot identity>' --confirm-motion \
  --robot-config /absolute/path/to/running/locomotion_foot_humanoid_asample_none.yaml
```

After preflight, it runs default-stand ONNX
inference without Pico. After at least three seconds of preparation, type
`CONNECT <actual robot identity>` to start the owned Pico service/capture.
Connect the calibrated headset and enable Full body/Send while keeping the
wearer neutral. When `PICO_INPUT: ready` and fresh controller input appear,
release and squeeze both triggers to request following. A premature/stale/invalid
request is rejected while stand continues; it is **not queued for automatic
activation**. `CONNECT` by itself never starts following. Another released/squeezed
dual-trigger pulse returns to ONNX stand in the same session. Typed `STOP` or Ctrl-C
cancels. A positive application duration expires normally; `--seconds 0`
continues until operator stop or fault.

Pico initialization and retargeting run in an isolated process so blocking
vendor initialization cannot hold the control interpreter's GIL. The process
starts idle, opens no robot controller, and initializes Pico only on CONNECT.
Its bounded snapshot queue retains original source receipt timestamps. Switching
preserves the policy history/last action and the SDK session/increasing sequence.
After following begins, a stale/failed Pico input cancels the trial; there is
no automatic reconnection or fallback motion. Full-loop timing and hardware
handoff remain unmeasured until a separately authorized supported trial.

These commands create SDK control handles but send no control request until
fresh inputs, live RobotInfo, physical body actuators, gains and command limits
pass preflight. Only the immediate-following mode requires Pico during entry;
stand-only and manual-Pico modes can enter with fresh body feedback alone.
The terminal displays the named model defaults and KP/KD,
fixed neck targets and config hash. Starting `live --confirm-motion` or the
automatic local entry requests motion directly, with no typed MOVE prompt.
The CLI identity is an operator label, not discovered robot
serial evidence. Confirm the actual device/connection and effective config.

Entry uses the SDK's PREPARE/COMMIT and verifies owned EXTERNAL/RUNNING.
It discards preflight policy/reference history and leads the reference from the
measured body pose to actor defaults. As in the SDK tracking example, the
reference advances by one 20 ms frame per inference, capped at its final frame;
wall-clock delays do not skip reference frames. Fixed neck targets use the
reviewed profile directly. At the handoff, the entire buffered Pico reference
must be complete, finite and within the configured Pico age budget.
The complete decoded BodyCmd is sent, including ankle compensation torque.
The unpopulated waist-roll slot remains inert; no hand command is sent.

`--seconds 0` disables only application duration expiry. Body feedback age must
stay within 100 ms (up to two consecutive stale ticks are skipped), Pico age
within 600 ms (held beyond 250 ms), and ControlStatus must advance within
500 ms. The 20 ms period is a
50 Hz scheduling target, matching the C++ executor's treatment of occasional
overruns. Compute/full-cycle overruns and skipped ticks are recorded; delayed
commands are never replayed in a catch-up burst. Feedback is rechecked just
before publication. Publish/log timing is recorded without an additional
100 ms publication-gap/stall cancellation rule. The effective firmware
watchdog/interpolation settings remain unchanged.
The Body31 actor path validates native arrays directly without materializing
an unused legacy44 state. Logs use a 1 MiB buffer, flushed on entry, exit,
candidate capture and errors; abrupt process termination may lose buffered rows.
Faults, timeouts, command
rejections, ownership changes and SIGINT/SIGTERM stop the trial without
automatic retry. Firmware `ingress_drop_count` counts a valid command superseded
by a newer one before its control tick (latest wins); each change is logged as
`firmware_ingress_drop`, and more than 10 within 1 s cancels. The first poll
counts increments from the confirmed entry counter, rather than treating an
already increased counter as its baseline. Logging records requested commands, session/sequence and
input state IDs; publication alone is not proof of actuator execution.

Every attempted entry exits through SDK CANCEL and requires PASSIVE/IDLE.
An incomplete exit is reported with the actual available status. **PASSIVE
can let loaded limbs fall**; keep support through exit. EXTERNAL has no automatic
expiry: the application duration is not a server lease, and a killed/crashed
client may retain ownership. Ctrl-C/software CANCEL is not an emergency stop
and cannot restore an action cleared on entry. Use the hardware emergency stop
if safety is uncertain. Do not retry a fault until its actual cause/state has
been reviewed. The historical P1/P2/P3 `7200` config is not used by this entry.

## Replay NPZ schema

`body_poses [T,24,7]` contains Pico world XYZ/XYZW.
`sample_times_ns [T]` is strictly increasing integer nanoseconds.
`robot_joint_names [J]` explicitly names the columns of
`robot_joint_pos [T,J]` and `robot_joint_vel [T,J]`.
`robot_root_quat_wxyz [T,4]` and `robot_root_angvel_local [T,3]` contain the
model's HIP-frame state convention. Units are radians, rad/s and seconds/ns as
named. Every one of the actor's 28 joints must be present. Replay inference
starts only after the full model-derived window is available (200 ms for the current contract).

## Pico stream conditioning (SuperMimic)

`config/actor_pico_supermimic.json` `pico_stream` changes the live Pico path:

- The worker sends conditioned raw GMR samples and a source-clock offset. The
  control loop resamples the policy window at
  `now - clock_offset - playback_margin_s - reference delay`, so the reference
  moves 20 ms per tick no matter when GMR frames arrive.
- `ik_max_iter` caps GMR refinement passes per frame. The pinned default is 10.
- `joint_filter`/`root_filter` are One-Euro filters. `root_max_tilt_deg` clamps
  the reference root tilt and keeps its heading.
- `reader_poll_interval_s` waits between duplicate body snapshots (2 ms on
  SuperMimic); new frames are processed immediately when observed.
- `worker_cpus` is inherited by SDK/body/trigger threads. Only after those
  threads start does `gmr_cpus` narrow the calling GMR thread. The reviewed
  six-core S100 profile uses CPU 0–4 for receivers and CPU 1 for GMR, avoiding
  CPU 5's measured contention. These indices must be reviewed on other CPU
  layouts; unavailable CPUs fail startup. The vendor service keeps its
  inherited mask. No CPU frequency or vendor-service affinity is changed.

Without the block, the previous newest-sample window is used unchanged.

### Frozen session replay (sim2sim)

`tests/fixtures/actor_pico/robot42_pico_session_20261029.npz` freezes the Pico
input, receipt times and on-robot GMR durations of session
`robot42_test_20261029T220318` (`scripts/freeze_pico_session.py` rebuilds it).
Stage 1 runs in the deployment venv and exports the per-tick reference windows
for the `legacy` and `playback` variants:

```bash
python3 scripts/replay_pico_session.py --output /tmp/windows.npz
```

Stage 2 is trackerlab-supermimic
`scripts/rls/supermimic/replay_pico_session_mujoco.py --windows /tmp/windows.npz`.
It closes the loop in MuJoCo with the deployed ONNX. The `legacy` variant
reproduces the robot's per-segment pitch std, >1 Hz pitch power, roll std and
action step RMS. This is simulation, not hardware validation.

### TODO: GMR timing under full control load

Status: root cause calibrated and receiver/thread changes verified on S100
on 2026-10-30. Full control-load acceptance remains open.

The earlier 33–36 ms online versus 2.7 ms x86 comparison did not isolate
input-thread contention. CPU 5 placement, repeated full-snapshot copies and
SDK/reader/GMR sharing a core all increased latency in controlled tests.
Putting the whole worker on a single core made live reception slower. The
reader now waits 2 ms on duplicate packets; SDK/readers retain CPU 0–4 while
only the GMR calling thread narrows to CPU 1. OpenBLAS stays single-threaded.

The [on-robot verification receipt](deploy/actor_s100/gmr_benchmark_20261030.json)
records the same 1000 frozen inputs with real Pico reception, trigger reader,
50 Hz queue draining and raw GMR logging, at unchanged 1.5 GHz:

| Receiver configuration | Mean GMR ms | P99 GMR ms | Mean thread CPU ms |
| --- | ---: | ---: | ---: |
| 0.1 ms duplicate polling, unrestricted threads | 21.167 | 47.763 | 13.079 |
| 2 ms duplicate polling, receiver CPU 0–4 / GMR CPU 1 | 12.834 | 29.212 | 9.798 |

Mean and P99 improved 39.37% and 38.84%; all 1000 GMR qpos outputs have the
same SHA-256. IK iterations, filters and weights were identical. This is a
read-only performance comparison, not an EXTERNAL motion trial. The prior
short calibration at 2 GHz is not a production frequency change.

To reproduce, connect Pico Full body/Send for each run; neither command opens
a robot controller. Reports refuse to overwrite an existing file:

```bash
venv/bin/python scripts/bench_pico_gmr_live.py --baseline --frames 1000 --output /tmp/gmr_baseline.json
venv/bin/python scripts/bench_pico_gmr_live.py --frames 1000 --output /tmp/gmr_configured.json
```

The offline-only `scripts/bench_pico_gmr.py` remains available and now selects
the native SDK backend on ARM. Capture headers report both `worker_affinity`
and `gmr_affinity`, plus the duplicate poll interval. Session regressions,
including stale-hold freezing and late-sample clock behavior, are included
in `verify_actor_delivery.py --host-inference --supermimic-model ...`.

Remaining: mean GMR below 10 ms and P99 below 20 ms at 1.5 GHz are **not yet
achieved** in this measurement. Validate sustained timing with the full
control/inference/logging load and physical Pico A/B. If further CPU work is
needed, profile the Python/NumPy task-objective assembly (about 41% of GMR
CPU in the calibration); DAQP itself was about 5%. Historical logs lack
per-frame GMR thread CPU/placement, so their entire delay cannot be attributed
retroactively to one cause.

### Full-compute shadow replay and alternating trials

The [full-load replay receipt](deploy/actor_s100/full_load_replay_20261030.json)
retains 25 trials of 1000 frozen GMR inputs and 18,821 native inference ticks
from 2026-10-30. Pico was unavailable and the real right-wrist pitch feedback
reported TIMEOUT (8), so these measurements use recorded robot feedback and
a 90 Hz replay SDK shim. They do not validate live Pico reception, healthy
real-time joint feedback or EXTERNAL motion.

The committed runner reproduces the 13-trial placement matrix: N is the
read-only subscription/queue/log baseline; A adds 50 Hz policy compute with
unrestricted control threads; B restricts control to CPU 2–4 with GMR on
CPU 1; C uses the same control mask with GMR allowed on CPU 0–1. Each trial
uses a fresh process and records actual masks, frequency, temperature,
wall/thread CPU time, full observations and skipped ticks. It uses only
`PullStateSource`, rejects control publishers, and never sends actions.

On the installed S100 checkout, use the deployment venv/model and the private
peer configuration prepared by the documented state-check/shadow startup:

```bash
venv/bin/python scripts/bench_pico_full_load.py \
  --pico replay --feedback recorded --frames 1000 \
  --sequence NABABABACACAC --output /tmp/gmr_full_load_new
```

The output directory must be empty. `--pico live` tests actual reception;
`--feedback live` requires healthy real feedback and does not fall back to
recordings. The runner retains one receiver service across trials to avoid
headset reconnects, and closes its owned processes on completion or failure.
Input starts **before** control affinity is narrowed: applying `taskset` to
the entire launcher would also restrict the input child's inherited mask.

| Replay condition | Trials | Pooled mean GMR ms | Pooled P99 GMR ms |
| --- | ---: | ---: | ---: |
| A: unrestricted control, GMR CPU 1 | 6 | 17.880 | 36.782 |
| B: control CPU 2–4, GMR CPU 1 | 9 | 12.675 | 24.628 |
| C: control CPU 2–4, GMR CPU 0–1 | 3 | 14.037 | 27.218 |

All 25,000 GMR qpos hashes matched and the shadow loops had no skipped ticks
or compute cycles over 20 ms. The receipt also preserves six **experimental**
native-objective trials: the second prototype reached pooled 10.118 ms mean
and 19.464 ms P99, but only one of its three runs met both targets. Its native
code, build scripts and binaries are excluded from this delivery, and D/E
are not supported runner conditions. Large logs and prototype sources remain
in the separately checksummed archive identified by the receipt.

These results support testing layout B next with live Pico; they do not change
the production control-affinity defaults or establish stable timing acceptance.

### TODO: robot42 hardware and deployment follow-ups

Status: open. Source: session `robot42_test_20261029T220318`. Training-side
items (fore-aft sway, learned standing asymmetry, ankle roll at its limit,
missing joint friction) are in trackerlab-supermimic
`docs/vita02a_robot42_followups.md`.

1. **Robot A/B of `pico_stream`.** Read-only receiver performance A/B is now
   recorded above; EXTERNAL motion A/B remains pending. In frozen
   session simulation, action step RMS drops about 32% and the
   per-segment sway (pitch std, >1 Hz pitch power) is unchanged. Run the same
   motions with this config, then freeze the new session with
   `scripts/freeze_pico_session.py --gmr-input ... --actor-log ... --session ... --output ...`.
   Compare per segment against robot42: reference advance per tick
   (`playback_time_s`), action step RMS, root pitch std and >1 Hz pitch power.
2. **Left upper ankle motor (slot 4).** Suspended SysID on 2026-10-28
   (candidate fit, not validated) gives Coulomb friction of 1.06 Nm, against
   0.32–0.38 Nm for slots 5/10/11. Its 0.5 Hz ±0.1 rad response has an
   amplitude ratio of 0.43 (others 0.70–0.76) and a lag of 251 ms (others
   115–131 ms). Inspect the left ankle linkage, bearings and lubrication, then
   repeat the slot-4 mapping run. In simulation this friction does not
   change the sway.
3. **Left ankle pitch static residual.** Standing on the robot needs about
   8 Nm more left ankle pitch torque than MuJoCo at the same angle (target - q
   +0.094 vs -0.107 rad). In the stand probe, three causes fit equally well:
   about 30% weaker torque on both ankles, about 4 Nm forward load on both
   ankles (mass/CoM), or a +0.05 rad zero on the left lower motor. Measure
   the sole attitude with an inclinometer and check robot mass/CoM against
   the model before changing any zero or torque constant.
4. **FIXED_STAND keyframe.** The vendor
   `locomotion_foot_humanoid_asample_none.yaml` FIXED_STAND keyframe has all
   42 positions at 0.0, marked as a placeholder. The robot then stands
   straight-legged with IMU roll about +4.5 deg and pitch about -11 deg until
   the policy takes over (under 1 deg afterwards). Replace it with a real
   standing pose in the vendor configuration.
## Validation boundary

Host CPU inference and synthetic Pico replay are exercised. S100 inference
latency, live Pico/tracker quality, real robot feedback, and robot actuation
are **not_measured**. The new live runner's offline tests exercise real SDK
session/math against simulated I/O, not real transport or hardware. Moving
the robot additionally requires per-device review of the retargeted reference
distribution, control transitions and the robot-side virtual-ankle PD contract.
The shadow mode remains incapable of actuation.

## Reproduce a rejected policy candidate

Startup writes `prepared_policy_candidate` before validating the command. It
contains the exact measured input, reference arrays, observation and decoded
output so rejected frames can be replayed without robot control:

```bash
python3 scripts/replay_actor_candidate.py --backend ort --input /path/to/live.jsonl
```

The replay checks observation/action/command parity and reports target deltas
as diagnostics. Live RobotInfo validation remains a separate hardware check. Earlier logs lacking this record cannot reconstruct the original
IMU/reference exactly. Actual live publish records include measured Body31 q/dq,
HIP orientation/gyro and firmware applied session/sequence evidence.

Optional developer physics validation uses an explicitly supplied compatible
MuJoCo scene/assets, never robot transport:

```bash
python3 scripts/validate_actor_stand_mujoco.py --scene /path/to/scene.xml \
  --initial zero --seconds 10 --output /tmp/actor-physics-result.json
```

MuJoCo and these external assets are not SDK deployment prerequisites. This
simulation uses serial virtual ankle joints, not the robot's parallel motor
adapter; its success is separate from hardware control and balance acceptance.

The captured failed hardware preflight was reproduced exactly on host ORT
(observation difference 0; action difference about 7e-7). The previously added
Python 2-degree startup gate and 0.35 rad policy-output delta gate were both
withdrawn: neither is part of the reviewed C++ tracking contract. Target
deltas and measured tilt remain diagnostic data, never replacements for real
observations or evidence that a command is safe by themselves.

C++ parity was checked at the deployed robot revision c9b4e80c: policy
DecodeAndClampVtHumanAction applies the trained/profile range intersection,
BuildMotorFrameFromPolicyQ writes the result and ankle clamp feedforward,
and EXTERNAL ValidateFrameLocked checks the actual BodyCapabilityRegistry
fields/ranges. The parallel adapter owns its motor-space caps and entry blend.
No uniform policy-output delta threshold is applied by these paths. The
profile's stream.max_step_rad=2.0 rejects glitches in incoming Pico reference
frames; it is not a policy-output bound and is not substituted for one.
Hardware limits, finite-value checks, gains, ankle feedforward/adapter caps,
watchdog/interpolation, ownership and checked exit remain in place. Removing
an extra Python gate restores this contract; hardware stability still requires
an attended motion trial.
