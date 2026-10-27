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

Operator height defaults to **1.75 metres**. Override it for every launch:

```bash
bash scripts/start_actor_pico.sh --human-height 1.68
bash scripts/start_actor_pico.sh shadow --seconds 30 --human-height 1.68
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
bash scripts/start_actor_pico.sh check --steps 100
bash scripts/start_actor_pico.sh state-check --seconds 5 --group default
bash scripts/start_actor_pico.sh shadow --seconds 60 --group default
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
bash scripts/start_actor_pico.sh
```

**No arguments now selects an attended motion trial, not an offline check.**
Use `bash scripts/start_actor_pico.sh check` for offline inference, or
`state-check --seconds 5` for read-only feedback. The default motion entry uses
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
   hardware-bounded, stationary feedback for 0.5 s. Policy-online startup also
   requires gyro norm <=0.10 rad/s. HIP IMU tilt is displayed and checked
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

Body31 remains limited to 100 ms at read and again before publication. Fault
messages include source, publication and local receipt ages. A
`body_state_age_failure` record captures ages at read and at failure, plus
compute/policy time; the stale command is never published. No stale-command
retry, automatic re-entry or threshold extension is added.

Input checks follow the original HoloMotion deployment's
`config/vita02a_s100.yaml`: robot feedback age is 100 ms; Pico source/reference
age is 600 ms (`config/actor_pico.json:pico_max_age_s`), also used for controller
input and edge delivery. Source frame intervals are diagnostics, not data age:
a 110--122 ms interval does not clear a ready buffer or interrupt following.
Timestamped samples remain in the same causal window and are interpolated on
the model's 20 ms grid. There is no reference-epoch gate, segment-fit cutoff,
or 0.1 rad all-joint neutral admission rule. The model's 200 ms initial window,
finite/shape/quaternion checks and strictly increasing source timestamps remain.
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
bash scripts/start_actor_pico.sh auto --seconds 120 \
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
Use `bash scripts/start_actor_pico.sh --seconds 300` for a five-minute run;
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
bash scripts/start_actor_pico.sh live --stand-only --seconds 5 \
  --robot-id '<actual robot identity>' --confirm-motion \
  --robot-config /absolute/path/to/running/locomotion.yaml
```

After reviewing that test and authorizing Pico following, omit `--stand-only`:

```bash
bash scripts/start_actor_pico.sh live --seconds 10 \
  --robot-id '<actual robot identity>' --confirm-motion \
  --robot-config /absolute/path/to/running/locomotion.yaml
```

To enter ONNX stand first and manually attach Pico in the **same EXTERNAL
session**, use the separately armed interactive mode (on S100, from the actual
deployment root on persistent `/userdata`, not overlay `/root`):

```bash
bash scripts/start_actor_pico.sh live --manual-pico --seconds 120 \
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
stay within 100 ms, Pico age within 600 ms, and ControlStatus must advance within
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
rejections/drops, ownership changes and SIGINT/SIGTERM stop the trial without
automatic retry. Logging records requested commands, session/sequence and
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
