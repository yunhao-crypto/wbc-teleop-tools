# actor_g1 + Pico on S100

This entry runs `model_step_431500_g1.onnx` through the installed SDK 0.2.1 WBC
observation/decoder math. The model is supplied separately (excluded from Git)
and copied by the bootstrap to `artifacts/actor_g1/actor_g1.onnx`
and SHA256 pinned in `config/actor_pico.json`. Its metadata identifies
`vita02a`, G1 encoder, input `[1,1520]`, output `[1,28]`, and 50 Hz.
It is the Vita28 tracking actor, regardless of the `g1` filename.

The entry supports model checks, offline Pico replay, live-input shadow
inference, and a separately armed supported EXTERNAL trial. Shadow subscribes to Body31 feedback and reads the existing Pico
service. It writes computed targets to JSONL; it never creates a controller,
requests EXTERNAL or publishes a robot command. Only `live` requests control and moves joints. The existing P1/P2/P3 runtime,
launchers, SDK installation and protocol guards are not replaced.

## Data path and calibration

```text
Pico/XRoboToolkit atomic 24x7 snapshot
  -> existing Pico world-to-canonical SMPL rotation conversion
  -> calibrated URDF segment-orientation inverse kinematics
  -> timestamped 28-joint reference buffer
  -> SDK WbcTrackingPolicy.observe (1520 floats)
  -> actor_g1 ONNX (28 actions)
  -> SDK WbcTrackingPolicy.decode (inert Body31 DTO)
  -> JSONL: targets, KP/KD, ankle feedforward torque, reference/compute timing
```

The first valid Pico frame calibrates segment offsets against the model's
default robot pose. Stand upright and still in the intended neutral capture
pose when starting shadow. Restart the shadow process to recalibrate.
The initial measured HIP orientation anchors the reference root orientation.
The current actor uses joint positions, joint velocities and root orientation
as reference inputs; its metadata does not request root height or translation.

The new retargeter matches torso, upper/lower legs, feet, upper/lower arms and
wrists using the bundled Vita URDF. It uses NumPy only and requires no mesh,
SMPL pickle, Torch, SciPy, CUDA, GMR or planner on S100. Segment mappings,
iterations and a maximum segment orientation residual are explicit in config.
Excess residual, malformed poses, regressing timestamps or input gaps stop the
run with a recorded error. This retargeter does not enforce contacts, foot
placement, collision avoidance or dynamic balance, and has no hardware motion
acceptance. Its kinematic checks are separate from training-distribution
or real-human motion quality validation.

The selected step-431500 actor requests 10 reference frames spaced 20 ms apart.
The causal input buffer waits for the 180 ms horizon plus a 20 ms forward-difference tail.
The resulting reference delay is **200 ms**, in addition to capture/compute
latency. These are real past samples interpolated at 50 Hz; future motion is
not predicted or filled with repeated terminal poses. Removing that delay
requires a validated predictor or a differently trained/exported model.
This file is distinct from the old `f0dfa046…` actor: its SHA256 is
`a341391a346e050caf11c40c6989336c658c445e3e124f12dcbe292d91324474`.
The old actor declared 100 ms spacing and needed 920 ms buffering. Shape alone
does not distinguish those models; timing and joint limits come from each
model's own metadata. The current actor narrows ankle-pitch limits and changes
shoulder-yaw limits; effective targets use the intersection with robot limits.

The actor path retains the SDK's 10-frame, term-major proprioception history,
clipped-action feedback and ankle clamp feedforward correction. It uses the
actor's healthy HIP IMU convention from `humanoid-sdk-tracking`, not the older
HoloMotion Motion observation layout. Policy targets are named and mapped to
Body31; the unpopulated waist-roll slot is inert and the two neck joints use
the explicit fixed profile. The copied SDK profile is documented in config.

## Offline verification on this x86 host

Use the host Python, which has NumPy and ONNX Runtime. The ARM venv interpreter
cannot run on x86. Offline mode reads the installed SDK's Python math and DTOs
without loading its ARM inference extension or constructing any Aorta node.
For a fresh clone without an installed SDK, use the staged-export verifier:
`python3 scripts/verify_actor_delivery.py --host-inference --model /path/to/actor_g1.onnx`.
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
They also test known-pose IK recovery, an independent numerical Jacobian,
causal reference timing/velocity, stale and invalid inputs, real model
inference, and Body31 ankle torque/neck/hole mapping. Fixture provenance is in
`tests/fixtures/actor_pico/tracking_fixture_provenance.json`.
The replay fixture is synthetic, and its output is not hardware validation.
The documented 100-step replay covers its initial segment. An extended run
over the whole HoloMotion synthetic fixture reaches poses that exceed this
retargeter's 0.6 rad segment-residual limit and is rejected. This source is not
a guarantee that every arbitrary human pose is feasible for Vita28. The
residual limit is retained; it is not raised to make that replay pass.

## Run on S100

Clone the repository into a new directory on S100, then run
`python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx`.
The eight offline wheels, Pico service, binding, sources and synthetic replay
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
effective YAML digest and 120-second live budget. It stores no authorization.
Another device, changed network identity/selector/YAML or missing profile fails
before robot handles. A generic bundle has no device profile; use explicit
`auto --robot-id <local-robot-ip> --robot-config <effective-yaml>` or provision
reviewed local defaults. Do not copy another robot's private profile.

The default displays target, actual gains and startup sequence, and requires
one current on-site `MOVE <robot-id>` confirmation. Reliable support, clear
movement/pinch zones, an attendant and usable E-stop remain required.
The sequence is:

1. Wait the configured PASSIVE or QUIET_PASSIVE damping duration plus two
   seconds on continuous fresh feedback (currently 42 s / 6 s on the selected
   board). SDK `action_phase=IDLE` is the EXTERNAL session phase, not damping
   completion. A retired keyframe can clear the action name; that alone does
   not count as FIXED_STAND readiness.
2. Request FIXED_STAND once, and require actual active FIXED_STAND plus healthy,
   hardware-bounded, stationary feedback for 0.5 s. In `policy-online` mode the
   policy reference starts at the measured pose: exact zero joint positions
   are not an activation prerequisite. Progress and failure logs name the
   largest joint position error and speed; no request is retried after failure.
3. Preflight the first native policy command using the newly measured pose,
   cosine reference lead-in, real observations and configured heading alignment.
   Check hardware/output limits, a 0.35 rad initial target-step bound and the
   20 ms inference budget before requesting EXTERNAL. A failed preflight
   cancels startup and records the actual state.
4. Enter one SDK EXTERNAL session. Keep ONNX active throughout the reference
   lead-in into the model stand, using the pinned profile's 1.2 rad/s reference
   speed and 0.3--3 s duration. This matches the supplied tracking profile's
   `stand_on_activate` / online-reference design; it does not first drive an
   independent open-loop trajectory to the training pose. Raw measured q is
   not clipped to model output limits; model targets remain clamped to the
   trained/physical intersection. Check each target change, state freshness,
   ownership/status, body tilt and the complete 20 ms cycle budget.
5. Start the isolated Pico receiver automatically while ONNX stand continues.
   No Pico input is required for model inference. After `ONNX_READY` and
   `PICO_INPUT: ready`, keep the wearer neutral and type **`PICO`** to follow
   (at least 3 s of policy preparation is required). `PICO <robot-id>` remains
   accepted. The local default never auto-switches to following. The session,
   command sequence and policy history are preserved. Early/invalid requests
   are rejected and not queued; source loss after handoff cancels without
   automatic motion retry/reconnection.

The reference for the activation design is the user-selected
`vt_human_v2_tracking_asample.yaml`: measured-pose reference, continuous policy
inference, heading alignment, metadata-derived reference timing and stand hold
before live references. This application still uses the separately supplied
step431500 ONNX and Body31 SDK EXTERNAL contract. It does not load the firmware
HUMANOID_TRACKING model or copy its legacy 42-slot/finger/planner interfaces.
Model hash, 20 ms reference spacing, gains, order and native/schema guards
remain pinned by this standalone delivery; no monorepo is needed on S100.

For an explicit target, `auto` selects the same online-reference entry:

```bash
bash scripts/start_actor_pico.sh auto --seconds 120 \
  --robot-id 10.100.23.96 \
  --robot-config /app/locomotion/config/locomotion_foot_humanoid_asample_none.yaml
```

`--entry-policy joint-ramp` retains the earlier supported, independently
bounded preparation experiment with its strict FIXED_STAND zero-target and
actor-entry checks; it is not the default. `live --manual-pico` retains manual
receiver connection with `CONNECT`. `--auto-pico` is an explicit opt-in to
following a fresh neutral reference automatically; it is not enabled by this
board's selected local profile.

The 120-second live budget starts with EXTERNAL; confirmation, damping wait
and fixed-stand activation are additional bounded time. `STOP` or Ctrl-C sends
checked cancellation to PASSIVE. Incomplete startup cancels only a recognized
non-EXTERNAL state/generation, records incomplete exits and never retries
motion. EXTERNAL has no ownership expiry; PASSIVE removes position stiffness
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
to robot YAML or weaken model entry-pose checks to get past preflight.

`live` is an experimental, bounded **supported** trial. It is not acceptance
for unsupported standing/walking. Obtain authorization for the actual robot
and motion; keep reliable support, clear motion/pinch zones, an attendant and
a usable hardware emergency stop. The Pico wearer must stay neutral during
preflight and the first three seconds after entry, then use small movements.
The 200 ms reference delay remains for Pico following. `--stand-only` still moves the whole body:
it runs the online actor with a default-stand reference, not a fixed pose.
It requires no Pico device/service/reader. Body feedback and all control guards
remain mandatory; an entry pose outside actor/hardware limits is rejected.

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

After preflight and `MOVE <actual robot identity>`, it runs default-stand ONNX
inference without Pico. After at least three seconds of preparation, type
`CONNECT <actual robot identity>` to start the owned Pico service/capture.
Connect the calibrated headset and enable Full body/Send while keeping the
wearer neutral. When `PICO_INPUT: ready` appears, type
`PICO <actual robot identity>` to request following. A premature/stale/nonneutral
request is rejected while stand continues; it is **not queued for automatic
activation**. `CONNECT` by itself never starts following. Typed `STOP` or Ctrl-C
cancels. The trial still expires at the specified application duration.

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
fixed neck targets and config hash, then requires `MOVE <actual robot identity>`
within 60 seconds as current on-site confirmation. No unattended confirmation
flag is provided. The CLI identity is an operator label, not discovered robot
serial evidence. Confirm the actual device/connection and effective config.

Entry uses the SDK's PREPARE/COMMIT and verifies owned EXTERNAL/RUNNING.
It discards preflight policy/reference history, leads the reference from the
measured body pose to actor defaults, and ramps neck positions to zero over
at least three seconds. At the handoff, the entire buffered Pico reference
must remain within 0.10 rad of defaults; otherwise the trial cancels.
The complete decoded BodyCmd is sent, including ankle compensation torque.
The unpopulated waist-roll slot remains inert; no hand command is sent.

Trials are limited to 120 seconds. Body feedback and Pico age must stay within
100 ms, advancing ControlStatus within 500 ms, and the complete 50 Hz cycle
(including publication/logging) within 20 ms. Faults, timeouts, command
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
starts only after the full model-derived window is available (200 ms for step 431500).

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

The replay checks observation/action/command parity and reports the unchanged
step guard. Earlier logs lacking this record cannot reconstruct the original
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
