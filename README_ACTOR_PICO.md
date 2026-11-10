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
`configure` and the connection-only `connect` mode: `sonic` (`config/actor_pico.json`),
`supermimic` (`config/actor_pico_supermimic.json`) or `morphmimic`
(`config/actor_pico_morphmimic.json`), given as the first word
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
| Model | `artifacts/actor_g1/model_step_009700_longrun_closed_loop_g1.onnx` | `artifacts/supermimic/F_ckpt_038000_supermimic_vita02a.onnx` |
| Input values | 1520 | 1670 (10x96 observation history + 10x71 goal frames) |
| Reference delay | 200 ms | 220 ms (10 future goal frames + 1 velocity tail frame) |
| Root input | quaternion only | quaternion plus GMR base position |
| Ankle type | from ONNX: `virtual_pitch_roll` or `motor_theta` (see below) | `motor_theta` |
| `ankle_virtual_pd_intent` | `true` or `false`, must match the ankle type | `false` |

The SuperMimic model is the closed-loop-ankle checkpoint
`train_v3_8gpu_24h_20261029` iteration 23000, exported by trackerlab-supermimic
`scripts/rls/supermimic/export_vita_deployment.py`; its observation/goal contract
is embedded in the ONNX metadata and checked at load. Its ankle contract is
`motor_theta`: the policy observes and commands the two Achilles motor angles in
the ankle pitch/roll slots. The actor maps measured state and references to
motor angles using training IK and the Jacobian. With the default
`body_profile.ankle_command_space: policy`, motor-theta policies send the A/B
angle targets and their independent motor PD gains directly in the existing
Body31 ankle pitch/roll slots. The robot must use
`actions.EXTERNAL.params.ankle_command_space: motor_theta` and
`ankle_virtual_pd_intent: false`. EXTERNAL tags those commands for the existing
motor passthrough adapter: no target FK/IK roundtrip, virtual-box clamp or
virtual-torque compensation. Rate/torque/gain caps still apply; motor travel
does not clip the policy's PD position setpoints.

SDK 0.2.1 and its wire schema are unchanged. `BodyState` and `RobotInfo` still
use virtual pitch/roll coordinates. EXTERNAL's command space is immutable for
the robot process; there is no remote SDK negotiation. The client checks the
provided effective YAML against the model/profile, but that file is not proof
of the running firmware configuration. Install the matching robot build and
YAML before starting the client. Do not mix an old virtual-command client with
a robot configured for native A/B.

For a legacy robot, explicitly set the actor profile to
`ankle_command_space: virtual_pitch_roll`. A motor-theta policy then retains
the FK/IK motor-PD path, without position clipping or clamp feedforward; the robot
must also use `virtual_pitch_roll` and `ankle_virtual_pd_intent: false`.
A virtual-ankle policy requires virtual commands and virtual PD (`true`).
For virtual-ankle SONIC, the decoder now sends the affine PD position target
without position clipping, matching the upstream policy contract. There is no
removed position term to restore through clamp feedforward; the old
`ankle_clamp_feedforward=true` requirement no longer applies. Rate, torque,
gain, finite-value and measured-state checks remain in force. Robot-side
configuration must still match the model's coordinate space.

Provision the SuperMimic model with
`bootstrap_actor_s100.py --policy supermimic --model /path/to/supermimic_vita02a.onnx`
and pass the same `--policy` to `check`, `replay`, `shadow` and `live`:

```bash
bash scripts/start_actor_pico.sh supermimic check --steps 100
bash scripts/start_actor_pico.sh supermimic shadow --seconds 60 --group default
bash scripts/start_actor_pico.sh supermimic        # default local motion entry
```

Offline parity: `tests/test_actor_supermimic.py` compares history and goal
features with the training-side fixture. Golden actions are checked against
that fixture's original ONNX only when `VITA_SUPERMIMIC_PARITY_MODEL` names
an explicitly supplied file (SHA256 pinned inside the fixture). Without it,
only the optional golden-action test is skipped; feature checks still run.
The configured deployment model has a finite-output/native-target smoke test; its weights are
not expected to reproduce a different checkpoint's golden actions. A trackerlab MuJoCo closed loop
(`validate_vita_deployment_mujoco.py`) compares the training policy, this
observation builder and the causal Pico window on the same clips. Neither is
hardware validation.

## SONIC has two ankle types: know which one you run

The default deployment is now **closed-loop SONIC iteration 9700**, pinned by
SHA256 `2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d`.
Its ONNX contract selects `motor_theta`; `ankle_command_space: policy` therefore
publishes native A/B targets, with `action_clip: 20`. Robot EXTERNAL must use
`ankle_command_space: motor_theta` and `ankle_virtual_pd_intent: false`.
The previous local 57400 virtual-ankle selection is not the deployment default.


`--policy sonic` covers two different kinds of GR00T/SONIC export. They share
the 1520-value input layout but command the ankle in different spaces, and each
needs a different robot EXTERNAL ankle mode. Before any `live` run, know which
type the installed `actor_g1.onnx` is and set the robot to match.

| SONIC type | Typical checkpoint | ONNX `ankle.control_space` | Robot YAML `actions.EXTERNAL.params.ankle_virtual_pd_intent` | Robot ankle PD |
|---|---|---|---|---|
| Serial ankle (virtual) | legacy `model_step_431500` (`TRL_Vita02A_Baseline`) | `virtual_pitch_roll` | `true` | `parallel_ankle.pd.strategy: virtual_pd` |
| Closed-loop ankle (motor) | GR00T `vita02a_closed_loop` (`TRL_Vita02A_ClosedLoop`) exports | `motor_theta` | `false` | Native A/B: per-command motor PD; `caps` = model `motor_limits` |

The actor decides the type only from the ONNX metadata
(`ankle.control_space` of `groot_wbc.tracking_contract.v1`), never from the file
name, the configuration or the robot:

- `virtual_pitch_roll`: the SDK contract and observations are reused; SONIC
  targets use the affine action rule below.
- `motor_theta`: the SDK layout is kept, but the ankle slots of the
  proprioception history and of the future joint references carry the two
  Achilles motor angles (IK of the pose, rates through the Jacobian, as in
  training). With command space `policy`, these A/B
  targets and model gains are transmitted directly (`actor_groot_ankle.py`).

GR00T/SONIC, SuperMimic and MorphMimic clip only the dimensionless action to
their exported `action_clip` (20 for the current SONIC training run), then
send `default + scale * action` in each policy's control coordinates.
Neither exported 0.9 soft limits, physical motor travel nor profile position
ranges clip these PD setpoints. The soft limits describe the training penalty
region for actual joint positions. The active command envelope allows these
policy targets while retaining finite-value, timing, velocity, torque and gain
checks; measured/startup poses and fixed joints still use their position bounds.
The robot driver/firmware must implement the same target semantics. This does
not change the physical plant's position stops or motor effort envelope.

Native affine-target actors require all three EXTERNAL settings:

```yaml
ankle_command_space: motor_theta
ankle_virtual_pd_intent: false
allow_position_targets_outside_mechanical_range: true
```

Use firmware implementing this explicit opt-in (vita-robot `d9e1a0a32` or a
compatible later build). A missing/false opt-in is rejected before a native
actor can create control handles; non-boolean values and an enabled opt-in on
the virtual wire are rejected too. Firmware retains confirmed protocol
position bounds and applies its physical-range rules to startup/safe hold.
The command `tau` field is feedforward: checking it alone does not establish
the bound on total motor-local PD effort.

Firmware `d7c5dbc81` adds `EXTERNAL.params.interpolation.enabled: false` for
same-tick body command output. WBC requires this explicit setting: true,
an omitted switch, strings and integers are rejected before requesting control.
The configuration preparation tool always writes `enabled: false`; there is no
WBC opt-in to command interpolation. `min_period_ms` and `max_period_ms` remain
required for firmware safety-generated hold ramps. The receipt records
`command_interpolation_enabled: false`, and the setting remains part of the
configuration comparison identity.
After reviewing a firmware/YAML change, rerun read-only checks and re-provision
the device-local startup profile and frozen trial receipts; do not bypass a
configuration fingerprint mismatch or reuse the old validation receipt.

Log headers record `contract.position_target_semantics`, and the effective
robot configuration receipt records the opt-in. Keep these identities when
comparing captures: the same model SHA may have been run with the older
secondary target-clipping decoder.

For the legacy virtual wire, a motor target must have a representable FK pose;
an unreachable request is rejected rather than projected. Use native
`motor_theta` commands for the full action range of motor-angle policies.
The legacy `ankle_clamp_feedforward` profile flag has no effect on these actors.

The actor **never changes** `ankle_virtual_pd_intent` or any other robot setting.
Prepare a separate YAML using `scripts/prepare_external_ankle_config.py`
(`--input SOURCE --output NEW_FILE --space motor_theta` or `virtual_pitch_roll`).
The script writes a new file only; it never installs it or restarts a service.
It enables the position-target opt-in for native mode and explicitly disables
it when generating a virtual-wire configuration, including from an already
enabled source YAML. Local startup defaults must reference the configuration
actually selected by the running locomotion service; regenerate device-bound
defaults when that selection changes.
Install the matching robot configuration and pass it as `--robot-config`. If the YAML
does not match the model type (wrong intent, gains, motor limits, geometry or
`virtual_limits`), `live` stops before requesting EXTERNAL or sending any
command.

To see the type of the installed model, run `check`; its report and every log
header give `ankle_control_space`, `required_ankle_command_space` and the required
`ankle_virtual_pd_intent`:

```bash
bash scripts/start_actor_pico.sh sonic check --steps 10
#   "ankle_control_space": "motor_theta",
#   "required_ankle_command_space": "motor_theta",
#   "required_ankle_virtual_pd_intent": false,
```

Switching SONIC between the two types is a model change: export the ONNX,
update `config/actor_pico.json` (model path, `sha256`, `bytes`), reseal the
manifests and install it with
`bootstrap_actor_s100.py --policy sonic --model /path/to/actor_g1.onnx`; then
change the robot YAML to the matching row above.
## MorphMimic Vita02A

The versioned `native_serial_and_coupled_motor_workspace_v2` reference guard
supports the 137670 `vita02a_sonic_v2` export through
`actor_morphmimic_sonic.py`. Despite its training profile name, this export uses
the **MorphMimic 1600-dimensional interface**, not SONIC's 1520-dimensional
interface. The adapter validates its declared bounded-v3 training to affine-v4
execution transfer and retains native A/B target decoding. Its reference
projection uses serial travel and the coupled motor workspace; existing
`virtual_box_and_coupled_motor_v1` models retain their original reference guard.

SuperMimic schema 2 retains the 1670-dimensional feature layout and explicitly
declares affine PD targets. The loader requires its no-position-clip semantics,
native motor command space and unchanged root-reference features, and checks
declared actuation overrides against exported gain/scale arrays. Schema 1
remains supported. Select candidate models in separate deployment roots so
their metadata, configuration and recordings remain identifiable; this support
does not change the default model files.

The default deployment model is the teleop fine-tune iteration 31000. Its model
path, SHA256 and size are specified in `config/actor_pico_morphmimic.json`.
Model weights and export/experiment reports are supplied separately; reports
are not source-controlled deployment inputs.
The model metadata enables `virtual_box_and_coupled_motor_v1`: virtual ankle
references are projected into the coupled motor workspace before reference
velocities are computed. Measured feedback is unchanged. Older models without
this metadata retain their existing reference path. This guard is identical
for native and legacy command trials. The candidate GMR configuration also
narrows ankle IK ranges; fixed NPZ playback does not use GMR.
The same launcher manages the camera relay and PICO view switching. MorphMimic
also uses the shared head-following defaults (40 ms filter, no independent
slew-rate cap); the physical head joint limits still apply.

`morphmimic` uses 1600 inputs (10 × 93 proprioception + 10 × 67 reference)
and 28 outputs, including native A/B ankles. Observation/goal normalizers are
baked into the exported ONNX. The reference includes the waist anchor and its
absolute height; standing references use the training robot's nominal root
height. Feedback remains virtual and is converted by the same IK/Jacobian
used in training. There is no SDK modification.

Export and training-input parity checks belong to the training repository.
Supply an ONNX containing the `morphmimic.tracking_contract.v1` metadata,
update the model path/SHA256/size in `config/actor_pico_morphmimic.json`, and
reseal the delivery manifests. Training export reports are external evidence.
The deployment checks the supplied model's contract and runs offline inference:

```bash
python scripts/prepare_external_ankle_config.py --input /path/to/asample.yaml \
  --output /path/to/asample-native-ab.yaml --space motor_theta
python scripts/bootstrap_actor_s100.py --policy morphmimic --model /path/to/policy.onnx
bash scripts/start_actor_pico.sh morphmimic check --steps 100
```

SONIC's serial-ankle checkpoint cannot become a native A/B policy by changing
a flag. Supply a real closed-loop motor-theta export and update its pinned
model identity. The three native decoders are covered by offline tests; this
does not constitute physical robot validation.

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
header. This selects the no-hand Vita02A model; finger tracking is not
introduced. Neck following is composed separately as described below.
Joints are resolved by name, including wrist yaw/pitch, independently
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
cd /host/wbc-teleop-tools
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 tests/test_actor_pico.py -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/actor_pico.py check --steps 100

python3 scripts/make_actor_pico_replay.py --output /tmp/actor_pico_input.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/actor_pico.py replay --input /tmp/actor_pico_input.npz \
  --steps 100 --record-observations
```

### Fixed NPZ reference playback

`scripts/start_actor_motion.sh` plays the same named robot-reference clip with
`sonic`, `supermimic`, or `morphmimic`. The selected policy constructs its own
observation, history and future goals, including virtual-to-motor ankle
conversion. This is a reference source, not a recorded observation/action player.
Live observations always use current SDK robot feedback.

Put the clip at `reference/fixed_motion.npz`, or edit
[`config/actor_motion.json`](config/actor_motion.json). Relative clip paths are
resolved against the repository root, not the shell's current directory.
The clip is external deployment data and is not included in Git. Import an
original NPZ explicitly selected with `--clip` from a dataset archive. This verifies its SHA256SUMS entry and copies it byte for byte to the
fixed path, without modifying the archive or starting services:

```bash
python3 scripts/import_actor_motion.py --archive /path/to/dataset.tar.gz --clip clips/example.npz
# Select another clip explicitly; --replace refuses to run while this repo's operator lock is held.
python3 scripts/import_actor_motion.py --archive /path/to/dataset.tar.gz \
  --clip clips/another.npz --replace
```

The import receipt is `reference/fixed_motion.import.json`. Import does not
establish that a clip is runnable; use the selected policy's `check` next.

```bash
# Validate the whole file and run up to 100 inference steps on synthetic stand
# feedback. No robot connection, Pico receiver, GMR or camera service is started.
bash scripts/start_actor_motion.sh supermimic check --steps 100 --record-observations

# S100 only: uses the existing device-bound local startup profile and requests
# motion. FIXED_STAND -> policy stand preparation -> clip lead-in -> one pass
# -> final-pose reference hold -> checked CANCEL/PASSIVE. STOP/Ctrl-C also exits.
bash scripts/start_actor_motion.sh supermimic --record-observations
bash scripts/start_actor_motion.sh morphmimic --record-observations
bash scripts/start_actor_motion.sh sonic --record-observations

# Optional configuration override; the policy remains an explicit selection.
bash scripts/start_actor_motion.sh supermimic check --motion-config /path/to/clip.json
```

The NPZ schema is common to all three policies:

| Field | Shape | Meaning |
|---|---|---|
| `joint_names` | `[J]` strings | Unique named robot joints; must include every selected policy joint |
| `joint_pos` | `[N,J]` | Joint angles in radians; ankles are **virtual pitch/roll**, never A/B motor angles |
| `body_pos_w` | `[N,B,3]` | Body positions in metres, right-handed Z-up world |
| `body_quat_w` | `[N,B,4]` | Unit quaternions in **wxyz** order |
| `fps` | scalar | Source sampling frequency in Hz |

`root_body_index` in the configuration explicitly selects the pelvis/free-root
column in the body arrays (default `0`); it must not select a waist/head anchor.
SDK-format clips may instead use `base_pos_w [N,3]`, `base_quat_w [N,4]` and
scalar `framerate`. Both root layouts may coexist only when the base arrays
exactly equal the selected body arrays. Conflicting or incomplete aliases are
rejected. When `body_names`/`root_body` are present, the selected column must
match the named root. If both frame-rate fields exist, their values must agree.
The supplied dataset has 40 named joints: the 28 policy reference joints and
12 Achilles mechanism joints. Its `control_space=motor_theta` describes the
actuators; named ankle pitch/roll reference columns remain virtual coordinates.
The separately named A/B columns are not substituted into those references. Optional scalar `joint_space` must equal
`virtual_pitch_roll`. Stored `joint_vel`/body velocities are not used: each
policy derives reference velocities using its own existing training-compatible
rule. A nonzero extra `waist_roll_joint` is rejected because the 28-DOF policy
cannot represent it; other ignored joint names are listed in the receipt.

The loader rejects malformed/nonfinite arrays, missing or duplicate names,
non-unit quaternions, and virtual joint references outside the deployment
profile beyond `limit_projection_max_rad`. The shipped motion configuration
explicitly allows at most `0.0001 rad` of reference-boundary projection; absent
this setting the loader permits only `0.000001 rad`. Larger settings are
rejected. This accommodates the dataset's ankle boundary difference of about
`0.000082 rad` and shoulder rounding of about `0.000005 rad`. Every corrected
frame/joint, original value, final value and delta is recorded in
`motion.limit_corrections`; the source NPZ/hash is unchanged. Command and
firmware limits are not relaxed. Set this value to `0` for strict references.
It resamples once to the policy's 50 Hz grid (linear positions, quaternion
SLERP), removes the initial world XY origin while preserving relative movement
and root height, and aligns initial yaw to the measured robot heading.
References are neither GMR-retargeted nor filtered with Pico's OneEuro settings;
future frames are available from disk, so no Pico playback delay is added.

The existing measured-pose lead-in is reused for clip entry. MorphMimic's
absolute reference root height also blends from its nominal stand height.
The policy's action/history state remains continuous across stand and playback.
Only a bounded, model-specific reference window is processed per tick, so
runtime cost does not grow with the length of the clip. Reference time advances
one frame per successful command publication; skipped/duplicate/stale feedback
does not fast-forward the clip or cause a catch-up command burst. The logged
`motion_tick`/`motion_time_s` are therefore separate from elapsed wall time.

`end_hold_s` (default `1.0`) holds the final **reference** while policy inference
continues, then uses the existing checked exit. Playback does not loop. The
existing `--seconds` limit, if supplied, includes stand preparation and can
end the trial earlier. Neck joints use the fixed policy profile, not Pico head
tracking. This entry starts no Pico/camera services; independently running
services are not stopped. Robot identity, configuration checks, command bounds,
feedback freshness, exclusive operator lock and EXTERNAL ownership checks are
the same as the Pico entry.

Each run logs the file/config SHA256, source/target rates, joint mapping and
reference windows. `sha256` in `actor_motion.json` can pin an approved clip;
`null` accepts the selected file and records its hash. `check` uses synthetic
stand feedback and is **not** closed-loop simulation or hardware validation.
Recorded `supermimic_observations.npz` files are a different schema and cannot
be played by this entry. Tests in `tests/test_actor_motion.py` cover all three
real ONNX policies and simulated live lifecycle/fault paths; they do not prove
physical stability of a chosen clip.

#### Controlled ankle command-path A/B trial

Use the **same motor-theta model** (SuperMimic or MorphMimic, or a SONIC export
whose contract explicitly says motor-theta), NPZ/hash, gains, reference settings
and robot build. Keep video and other background workloads in the same state.
A virtual-angle SONIC model cannot be used as the native half of this test.

| Trial | `--ankle-command-space` / EXTERNAL `ankle_command_space` | EXTERNAL `ankle_virtual_pd_intent` | `allow_position_targets_outside_mechanical_range` |
|---|---|---|---|
| Native | `motor_theta` | `false` | `true` |
| Legacy conversion | `virtual_pitch_roll` | `false` | `false` |

In both trials observations/references use the same virtual-to-motor conversion
and the same ONNX. Only the output path changes. Native sends A/B targets and
motor gains; legacy converts them to virtual pitch/roll, then firmware converts
back and applies motor PD. Legacy virtual clipping/FK residuals may change the
result; they are part of the behavior being tested. Keep
`ankle_clamp_feedforward` identical between trials. The legacy path additionally
checks that the firmware's fixed motor gains equal the model gains.
The comparison receipt excludes these two correlated command-path selectors,
while retaining gains and all other robot settings. The comparator rejects
mixed decoder semantics (including a new capture paired with an older log
that did not identify its decoder). This remains a comparison of complete
output paths, including legacy clipping, rather than an equal-target proof.

Prepare candidate YAMLs from the SAME effective robot configuration; these
commands only create files, never install them or restart locomotion:

```bash
python3 scripts/prepare_external_ankle_config.py --input /path/to/effective.yaml \
  --output /path/to/native.yaml --space motor_theta --virtual-pd-intent false
python3 scripts/prepare_external_ankle_config.py --input /path/to/effective.yaml \
  --output /path/to/legacy.yaml --space virtual_pitch_roll --virtual-pd-intent false

# Both are offline and can be run before changing any robot configuration.
bash scripts/start_actor_motion.sh morphmimic check --steps 1000 \
  --ankle-command-space motor_theta
bash scripts/start_actor_motion.sh morphmimic check --steps 1000 \
  --ankle-command-space virtual_pitch_roll
```

EXTERNAL command space is a process-level robot setting, not a remote mode
switch. Between live trials, stop the policy, install/select the corresponding
reviewed firmware YAML, restart locomotion with the robot supported, and refresh
the device-bound profile against the **running effective YAML**:

```bash
python3 scripts/configure_actor_s100.py --robot-id 10.100.23.173 \
  --robot-config /absolute/path/to/running-effective.yaml --replace
```

Only after that configuration matches the desired trial, start the corresponding
command manually (run these as separate trials, never concurrently):

```bash
bash scripts/start_actor_motion.sh morphmimic --ankle-command-space motor_theta \
  --record-observations --log outputs/motion_native.jsonl
bash scripts/start_actor_motion.sh morphmimic --ankle-command-space virtual_pitch_roll \
  --record-observations --log outputs/motion_legacy.jsonl
python3 scripts/compare_actor_motion.py outputs/motion_native.jsonl \
  outputs/motion_legacy.jsonl --output outputs/motion_ankle_comparison.json
```

The CLI override changes only the in-memory actor command profile; it does not
edit policy JSON or robot YAML. Mismatched configurations fail before motion.
The SDK cannot verify the running firmware mode remotely: a matching YAML path
alone is not evidence that locomotion restarted with it.

The comparison refuses different model/reference hashes, actor settings, root
selection, reference corrections, or robot YAML content beyond command space.
It aligns on original clip frame indices (excluding lead-in/end hold), checks
reference equality, and reports measured virtual ankle/joint RMSE, root tilt,
compute timing, skipped ticks, coverage and starting-pose differences. Partial
runs are identified. Logs without a confirmed exit are rejected; retain fault
logs for separate diagnosis rather than interpreting them as completed trials.
These metrics describe separate closed-loop trials; they do not prove causation.
Repeat each condition (for example native/legacy/legacy/native) before drawing
conclusions about the ankle path.

Start with `L1_01_weight_shift_small`. The deep-bend diagnostic exceeds the
existing tilt protection and is not a live acceptance clip. `L3_06_crouch_adjust`
starts crouched/tilted and requires a separately validated entry. Dataset
sim2sim results used iter12500; they do not validate another installed model.


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

## Pico head following (SONIC and SuperMimic)

Optional front-camera video return and A+B+X+Y view switching are described in
[PICO camera viewer](deploy/pico_viewer/README.md), including the versioned Android
APK and a new-headset setup guide. `bash scripts/start_actor_pico.sh connect`
starts the PC Service and video relay without a policy or robot control. Exit
that session before starting an actor. Live/auto/shadow entries now supervise
their own optional video relay (`--no-camera` disables it), while the actor
retains ownership of its PC Service and motion cleanup. This separate read-only HEVC
relay/client path leaves actor inference and double-trigger following intact.

Both actor configurations enable `head_control`. The 28-action policies and
GMR stay unchanged; a shared head controller replaces the fixed neck targets
in the same Body31 command, before the existing RobotInfo envelope check.
Arms and wrists remain policy-controlled; fingers are not part of this path.

**Look straight ahead relative to your chest when enabling Pico following.**
Each engagement calibrates the current `Head` relative to `Spine3` as neutral.
Subsequent head/chest relative rotation is expressed using the pinned GMR
chest-frame offset, then projected onto the robot's yaw/pitch gaze direction.
Whole-body turns cancel, and head roll is not reproduced by the two-axis neck.
This calibration affects only the neck; body GMR calibration is unchanged.

Live head control uses a separate shared snapshot updated directly by the
Pico reader, without waiting for a GMR solve or its output queue. Its pose,
integer nanosecond timestamp and reader receipt time are copied atomically;
a busy slot keeps the previous sample and its original age. The body policy
still uses the matching GMR result and timestamp. The head is independent of the
200/220 ms body reference window (and SuperMimic playback margin). It therefore
has a different reference time from the body; it follows relative articulation,
not absolute world gaze. An exponential filter (time constant 0.04 s) smooths
its targets. `head_control.max_velocity_rad_s` is `null`, so there is no separate
position slew limit; a finite positive value opts back into that limit (for
example, `0.8` restores the former 0.8 rad/s limit). The former filter time
constant was 0.12 s. These settings apply to tracking and standby return.
Configured targets are limited
to yaw +/-60 degrees and pitch [-30, +20] degrees in URDF joint coordinates,
intersected with actual RobotInfo limits. Positive pitch tilts forward/down in
the URDF frame. The existing fixed-joint KP/KD (20/1) are preserved.

Standby smoothly returns the neck to zero. The first live tick starts from
measured neck position. When the head reader sample exceeds `pico_hold_age_s`
(0.25 s), neck position freezes; fresh input resumes following. Body reference
hold uses the GMR sample's own age, so a delayed solve does not freeze a fresh
head sample. Either input exceeding `pico_max_age_s` (0.6 s) still reaches the
existing live fault path and cancels EXTERNAL.
Missing/invalid head data also fails the trial. Calibration/filter/slew state
advances only after a successful live publication. Logs include `head_control`
diagnostics (including the actual head sample's timestamp, reader sequence,
receipt clock and chest/head quaternions) and the final combined
`body_targets`/`sent_body_targets`. Replay/shadow without an independent reader
snapshot use the supplied frame, identified as `gmr_snapshot` in diagnostics.

Replay and shadow also compute head targets; neither sends commands. Set
`head_control.enabled` to `false` in the selected actor JSON to restore the
fixed-neck behavior. No robot YAML, Pico binding or policy re-export is needed.
Head following requires its own small-angle hardware validation; offline tests
do not establish physical tracking quality or balance.

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

1. Wait the configured PASSIVE or QUIET_PASSIVE damping duration from the
   first fresh observation. The updated A-sample YAMLs use 4 seconds for both;
   no extra two-second delay is added. An older 40-second YAML still waits 40
   seconds. The firmware continues to gate FIXED_STAND admission. SDK `action_phase=IDLE` is the EXTERNAL session phase, not damping
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
   not clipped to model output limits. All three actors preserve affine
   policy targets without a second position-range clip.
   Check the applicable command fields against RobotInfo ranges, state freshness,
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
the last one without command interpolation (watchdog 1000 ms). The third
consecutive stale tick cancels. Each `body_state_age_failure` record holds
phase, ages, compute/policy time and `action` (`skip_tick` or `cancel`).
A stale command is never published or retried.
When `(state_id, ts_state_real_us)` repeats, the loop records
`body_state_not_advanced` and waits for a new measurement without advancing
policy history, the stand reference or the command sequence. Heartbeat copies
cannot extend this wait indefinitely by refreshing their receipt/publication
clocks: the original sample receipt still has the same 100 ms budget and the
existing consecutive-stale cancellation rule applies. This protects the model
from treating cached feedback as new observations; it does not repair delayed
feedback delivery.
Successful inference records include `feedback_timing` with the ages at read
and before send, producer publication time and local receipt time. The Core
pull-context stamp is an assembly/dequeue diagnostic, not a network-arrival
measurement. Native joint-input failures name the affected joints, body slots,
numeric/symbolic error codes and state ID (for example `STALL=13`), so a motor
fault is distinguishable from delayed delivery.
For a diagnostic capture, set `VITA_PICO_CAPTURE_DIR` to an output directory.
The reader then records each distinct frame it observes (24x7 poses, a local
frame sequence and paired monotonic/realtime stamps) through a bounded queue to
a separate writer process. Array-to-list conversion, JSON encoding and disk
writes run outside the GMR interpreter. The header records `writer_pid`; normal
shutdown drains accepted frames and verifies the final written count after
flushing the file. Queue overflow drops capture frames without blocking the
reader; overflow, writer failure or a shutdown timeout reports an incomplete
capture. Its summary reports written and dropped frames; this is reader first
visibility, not SDK callback or network arrival. Raw network timing requires
a separate packet capture. Each GMR source row also retains that reader
identity, solver start/end monotonic and thread-CPU clocks, wall/CPU durations,
and the resulting qpos. `available_monotonic` includes reference processing
after the solver and must not be substituted for pure solver completion.
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
loaded. For a `virtual_pitch_roll` model, missing/disabled
`parallel_ankle.pd.strategy=virtual_pd` or
`actions.EXTERNAL.params.ankle_virtual_pd_intent=true` stops the trial. For a
`motor_theta` model (SuperMimic, or a closed-loop SONIC export),
`ankle_virtual_pd_intent` must be `false` and `kp_motor`/`kd_motor`, motor
limits, geometry and `virtual_limits` must match the model and actor profile.
The two ankle modes therefore need different EXTERNAL settings in the running
YAML; a mismatch stops the trial before any command and the YAML is never edited.
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
neck hold gains, head-control settings and config hash. Starting `live --confirm-motion` or the
automatic local entry requests motion directly, with no typed MOVE prompt.
The CLI identity is an operator label, not discovered robot
serial evidence. Confirm the actual device/connection and effective config.

Entry uses the SDK's PREPARE/COMMIT and verifies owned EXTERNAL/RUNNING.
It discards preflight policy/reference history and leads the reference from the
measured body pose to actor defaults. As in the SDK tracking example, the
reference advances by one 20 ms frame per inference, capped at its final frame;
wall-clock delays do not skip reference frames. With head control enabled, the
neck slews from measured position toward zero during standby and follows Pico
after handoff; disabling it uses the fixed profile directly.
At the handoff, the entire buffered Pico reference
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
The launcher creates a separate private client-mode Zenoh session for the
short-lived firmware CLI used by FIXED_STAND and startup cancellation. It
connects through the existing router with the same namespace/authentication;
the SDK feedback and continuous control session remains in peer mode. A failed
or timed-out motion request is never retried automatically.
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
- `joint_filter`/`root_filter` are One-Euro filters. `root_max_tilt_deg` clamps
  the reference root tilt and keeps its heading.

Without the block, the previous newest-sample window is used unchanged. The
SONIC reference does not implement this playback, so its configuration must
not contain `pico_stream`.

## Pico worker: GMR and CPU layout (all policies)

`pico_worker` sets the GMR worker, receiver and control scheduling. SONIC
(`config/actor_pico.json`) and SuperMimic ship the same block, so both run the
same GMR chain with the same parameters:

- `ik_max_iter` caps GMR refinement passes per frame. The pinned default is
  10; both policies use 3.
- `reader_poll_interval_s` waits between duplicate body snapshots (2 ms);
  new frames are processed immediately when observed. Shadow mode uses the
  same interval for its in-process reader.
- `worker_cpus` is inherited by SDK/body/trigger threads. Only after those
  threads start does `gmr_cpus` narrow the calling GMR thread. The reviewed
  six-core S100 profile uses CPU 0–4 for receivers and CPU 1 for GMR, avoiding
  CPU 5's measured contention. These indices must be reviewed on other CPU
  layouts; unavailable CPUs fail startup. The vendor service keeps its
  inherited mask. No CPU frequency or vendor-service affinity is changed.
- `control_cpus` pins every thread of the live control process (SCHED_RR
  policy loop, native inference, transport) after the Pico worker has been
  spawned, so the worker and vendor service keep the unrestricted mask. It
  must not overlap `gmr_cpus`: a real-time control thread on the GMR CPU
  preempts the pinned GMR thread, which cannot migrate. The S100 profile uses
  CPU 2–4 (layout B in the full-load replay below). The live log records the
  applied mask as `control_affinity`. Stand-only runs apply it too.

Without the block the worker is unchanged: 10 iterations, 0.1 ms polling and
no CPU pinning. The GMR input log header records `pico_worker` and the
applied worker/GMR masks. The timing measurements below used the SuperMimic
configuration; SONIC runs the same GMR code and settings but has not been
timed on the robot.

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

### GMR performance checks

Measure timing on the target hardware using separately saved reports. Historical
receipts are not deployment inputs; offline replay does not validate live Pico
reception, healthy joint feedback or EXTERNAL motion.

```bash
venv/bin/python scripts/bench_pico_gmr_live.py --baseline --frames 1000 --output /tmp/gmr_baseline.json
venv/bin/python scripts/bench_pico_gmr_live.py --frames 1000 --output /tmp/gmr_configured.json
venv/bin/python scripts/bench_pico_full_load.py \
  --pico replay --feedback recorded --frames 1000 \
  --sequence NABABABACACAC --output /tmp/gmr_full_load_new
```

The live GMR commands need Pico Full body/Send; they do not open a robot
controller. The full-load runner uses read-only feedback and never publishes
actions. Its output directory must be empty. `--pico live` tests actual reception;
`--feedback live` requires healthy feedback without falling back to recordings.
Compare recorded CPU placement, frequency, temperature, skipped ticks and GMR
latencies under the intended workload before accepting a deployment.

### TODO: robot42 hardware and deployment follow-ups

Status: open. Source: session `robot42_test_20261029T220318`. Training-side
items (fore-aft sway, learned standing asymmetry, ankle roll at its limit,
missing joint friction) are in trackerlab-supermimic
`docs/vita02a_robot42_followups.md`.

1. **Robot A/B of `pico_stream`.** Read-only receiver performance A/B is now
   measured with the commands above; EXTERNAL motion A/B remains pending. In frozen
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

The replay restores the recorded ankle command space (legacy captures default
to virtual pitch/roll), checks observation/action/command parity and reports target deltas
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

C++ parity was previously checked at robot revision c9b4e80c: policy
DecodeAndClampVtHumanAction applies the trained/profile range intersection,
BuildMotorFrameFromPolicyQ writes the result and ankle clamp feedforward,
and EXTERNAL ValidateFrameLocked checks the actual BodyCapabilityRegistry
fields/ranges. That historical target-clipping/feedforward behavior is
superseded by the affine targets described above; robot firmware must be
updated to match. The parallel adapter owns its effort caps and entry blend.
No uniform policy-output delta threshold is applied by these paths. The
profile's stream.max_step_rad=2.0 rejects glitches in incoming Pico reference
frames; it is not a policy-output bound and is not substituted for one.
Physical travel limits, finite-value checks, gains, adapter effort caps,
watchdog/interpolation, ownership and checked exit remain in place. Removing
an extra Python gate restores this contract; hardware stability still requires
an attended motion trial.
