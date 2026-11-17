# MimicLite Vita02A OP3 deployment

`mimiclite` replaces the former MorphMimic deployment option. There is no legacy
alias and old MorphMimic weights cannot be used. SONIC and SuperMimic are unchanged.
Model support does not replace the pinned model: this observation-ABI upgrade
leaves `config/actor_pico_mimiclite.json` and its existing model identity unchanged.
A new hist10 model must be imported as an explicitly reviewed replacement.
Unconfigured packages (`model_ready: false`) refuse inference/installation.
No ARM execution or real-robot acceptance is implied by host parity checks.

For optional timestamp capture, offline contract/IMU audits, frozen-input shadow
checks and controlled SDK-versus-exec comparison, see
[README_ACTOR_CONTROL_AUDIT.md](README_ACTOR_CONTROL_AUDIT.md). These tools never
start robot motion; the native-exec implementation remains external.

## Native locomotion actor entry

For a robot with the separate native MimicLite deployment already installed:

```bash
# Read-only command preview; works without a robot or native package:
bash scripts/start_mimiclite_native.sh --model knee-reg2000 --plan
# On the prepared robot67, explicitly start its installed native package:
bash scripts/start_mimiclite_native.sh --model knee-reg2000
```

This WTT entry calls the installed package's `start.sh`. Actor inference and
history run in locomotion; the native package's Python process supplies Pico/GMR
references. The native implementation, build/tests, profiles and installation
tools remain in [vita-robot's native deployment guide](https://github.com/VitaDynamics/vita-robot/blob/feat/mimiclite-native-robot67-20261117/src/application/locomotion/humanoid/tools/MIMICLITE_NATIVE.md).
The WTT entry is included in both base and planner delivery packages; native
binaries, models and device configuration are separate prerequisites.

`student2000` (the default here) means the **1224-input hist10 v2** native model,
not this branch's pinned original 814-input WTT model. `student2000` and
`knee-reg1000` use `/userdata/mimiclite_native_20261117`; `knee-reg2000` uses
`/userdata/mimiclite_knee_reg2000_20261117`. These defaults describe the current
robot67 installation. Use `--native-root /absolute/path` for a different reviewed
installation; its matching `start.sh`, `contract_MODEL.json` and `MODEL.onnx`
must exist. An incomplete installation is refused before invoking the package.

Pico following is the native package's default. `--stand-only` omits Pico;
`--motion /absolute/path/reference.npz` selects fixed NPZ; `--seconds 0` runs
until Ctrl-C. After stand preparation, release then press both triggers to toggle
following; Ctrl-C requests PASSIVE through the installed launcher. The existing
native package keeps its operator lock, ownership/posture/freshness guards,
planner/camera-off settings and recording location. This wrapper does not
reconfigure/restart locomotion, change WTT's pinned model or provide a Python
actor fallback. The earlier WTT inference path remains
`start_actor_pico.sh mimiclite`; native startup requires matching native firmware
on the target.

## 1. Export a real matching policy

Use the MimicLite OP3 training-side `scripts/play.py` export with the exact
checkpoint/algorithm/observation configuration. Supply both `policy.onnx` and
its matching `policy.yaml` containing `motor_deployment`. Export the deployable
PPO actor or ROA **finetune student**, never a privileged teacher.

The accepted model identity is the canonical `mimiclite.vita02a.native.v2`
OP3 plant (SHA256 `2f015a57135ea0fd76e2f76f60633a3d55e835c6978432fb3c43b2de4fa78a75`),
with actuator schema `mimiclite.vita02a.achilles_friction.v2`. Older checkpoints
need an explicit training-side fine-tune/new export; editing a metadata version
is not a migration or a validation. Other robot geometries or observation terms
require a separate review rather than weakening the parser.

## 2. Host-side graph import (no robot connection)

Install the host dependencies in `deploy/actor_s100/requirements-host-check.txt`
(including `onnx` and `onnxruntime`), then run from the source checkout:

```bash
python3 scripts/import_mimiclite_policy.py --policy-yaml /absolute/path/policy.yaml
# Use --onnx /absolute/path/policy.onnx if it is not beside the YAML.
```

The importer validates the YAML and ONNX input shapes, pins source file hashes,
and adds only input splits/reshapes and an action-only output around the trained
graph. Original weights and embedded normalization remain unchanged. It checks
nine synthetic inputs against the original ONNX using host ORT before installing
`artifacts/mimiclite/policy.onnx` and updating the configuration's hash/size.
It does not fabricate a checkpoint or certify tracking quality. The wrapper
embeds `mimiclite.tracking_contract.v1`; an arbitrary single-input model with a
new filename is not accepted. Self-contained float32 source ONNX with fixed
vector or `[1,D]` inputs, opset 13–21 and IR <= 10 is required for the reviewed
SDK ABI; newer graph semantics are rejected, not silently downgraded.

A different previously imported model is preserved unless `--replace` is
explicitly supplied. The shared operator lock prevents import during a live or
install session. Commit/review the updated JSON/provenance before a release;
weights stay outside Git. Run this importer **before** package sealing, not
inside an extracted sealed package.

## 3. Build/install the S100 package

```bash
# Build host, after reviewing the source/config update:
python3 scripts/build_actor_delivery.py --variant base
# For an explicitly labelled development package only: add --allow-dirty.
```

Copy the built archive, its SHA256 and the **wrapped** model to S100. Follow the
normal fresh-directory/bootstrap procedure; use the selected model's configured
hash, not its local filename:

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --policy mimiclite --model /path/to/wrapped-policy.onnx
bash scripts/start_actor_pico.sh mimiclite check --steps 100
bash scripts/start_actor_motion.sh mimiclite check --steps 100 --record-observations
```

The base package keeps the same SDK-native inference API and 21-wheel ARM
closure. Python ONNX/ORT are host importer/check dependencies, not additional
base S100 runtime dependencies. SDK 0.3.0, matching firmware, device-local peer
credentials and reviewed effective robot YAML are still prerequisites.
An unconfigured package can serve the other policies, but MimicLite installation
is refused until a configured package is built. There is no policy fallback.

## Observation and control contract

`motor_deployment.observation_schema` selects the observation ABI. A missing
field is the preserved `mimic_lite.motor_observation.v1`; the only new value is
`mimic_lite.motor_observation.v2`. Unknown values and unknown terms are rejected.
V1 remains the 814-wide contract described below. V2 is exact: command terms add
`ref_joint_vel_future` using the same future indices as `ref_joint_pos_future`;
all four sensor histories use `[0,1,2,3,4,5,6,7,8,9]`,
`joint_pos_history.relative_to_default: true`, and `prev_actions.steps: 3`.
Its resolved widths are command=520, policy=704, total=1224. V2 joint history
is motor radians relative to the named policy motor default; reference positions
remain absolute motor radians.

The policy API accepts exact parity fixtures through
`observe(..., reference_motor_q=<[future,28] motor rad>,
reference_joint_velocity=<[future,28] motor rad/s>)` (or
`reference={'joint_position': ..., 'joint_velocity': ...}`). These explicit
selected-frame joint arrays default to the contract's `joint_names` (policy motor)
order; optional `reference['joint_names']` explicitly names another ordering and
is validated/reordered. The actor then uses `reference_joint_names` for the feature.
The reference mapping can also supply
selected `root_position` and `root_quaternion` arrays. This retains the training
float16 reference quaternion's near-unit norm drift for projected heading; v2
matches the training non-renormalizing projected-yaw formula, while orientation
matrices use normalized rotations. Production Motion/Pico
uses physically coherent virtual-joint rates transformed to motor rates by the
existing ankle Jacobian. NPZ references should provide `joint_vel` (or
`joint_velocity`) in virtual-joint rad/s. A deterministic dense derivative is
available only when the Motion config explicitly sets
`velocity_fallback: dense_derivative`; it differentiates the complete declared-
FPS clip before resampling, never a sparse reference window. Missing v2 velocity
is otherwise an error, and rates are never silently zero-filled.
Live Pico references have no recorded joint-rate channel: their explicit rule is
`np.gradient(q, source_timestamps, edge_order=1)` over the retained full source
timeline, then interpolation to the dense causal window and ankle-Jacobian
conversion. This is a documented reconstruction, not a claim that live GMR rates
are identical to independently recorded training-dataset velocities. Lead-in rates
follow the actual dense lead-in trajectory. Stand, stale-reference hold and
out-of-range endpoint padding explicitly use zero rates. V2 rejects position
projection when it would invalidate supplied explicit velocities.

The following sparse-history defaults describe **V1**, not V2:

- The wrapper joins input groups in an embedded, validated order, then restores
  each original input's rank. Default recipe: `command=296`, `policy=518`, total
  814. Counts are derived from each export, not hardcoded as an unversioned ABI.
- Histories are term-major/newest-first, at offsets `[0,1,2,3,4,8,16]`; initial
  sensor histories repeat the first frame and previous actions reset to zero.
  Joint positions are calibrated motor coordinates, not nominal-relative q.
- Reference offsets are `[-8,-4,-2,0,1,2,3,4]` by default, rooted at `base_link`.
  Orientation uses the first two **rows** of relative rotation, flattened;
  quaternions are wxyz and gyro axes are HIP/base_link. VecNorm remains in ONNX.
- Pico is resampled causally at 50 Hz with the export's past/future window.
  Default +4 lookahead requires 80 ms delay plus the configured stream margin
  (60 ms by default); collecting the initial -8..+4 window needs 240 ms of data.
  Sparse proprioception history is a separate signal history.
- Fixed NPZ and GMR references remain named **virtual** ankle pitch/roll.
  Policy inputs map these references and SDK virtual feedback to A/B angles and
  rates by the reviewed IK/Jacobian. References outside coupled motor travel
  are rejected, not silently projected into another reference motion. The
  conservative existing GMR box remains; no hardware range is widened.
- The export's named depth-first joint order is explicitly mapped into Body31.
  Wrists and ankle aliases must never be copied by numeric slot coincidence.
- Raw finite action is decoded as `default + scale * action`, with **no action
  clipping or PD-target clipping**; raw actions also enter the history. Native
  A/B targets/gains are sent directly, without a second ankle IK or friction
  feedforward. Simulated friction is not an extra hardware compensation term.

## Safety and staged validation

Only native commands are supported for MimicLite. An explicit virtual-wire
trial is rejected. Required effective EXTERNAL settings are unchanged:

```yaml
ankle_command_space: motor_theta
ankle_virtual_pd_intent: false
allow_position_targets_outside_mechanical_range: true
position_target_margin_rad: 3.14159
interpolation:
  enabled: false
  min_period_ms: 2
  max_period_ms: 100
```

Do not copy this fragment as a full robot configuration. The running firmware
must support the opt-in and target margin. All existing measured-pose, finite
float32, gain/command envelope, freshness, exclusive ownership, watchdog,
operator-selection and CANCEL/PASSIVE checks remain. Model support is not
permission to bypass firmware or an effective-YAML fingerprint mismatch.

After offline checks, separately review device configuration and run read-only
`state-check`, then `shadow` with real feedback/Pico. Review observation parity,
calibration, timing, native sim2sim, stopping and saturation before any explicitly
authorized supported live trial. No new live command is run by the importer or
tests. SDK virtual feedback reconstruction is not a direct raw-motor encoder
measurement; compliance, zero/sign calibration and firmware behavior still
require hardware validation.

## Tests and evidence limits

`tests/test_actor_mimiclite.py` generates synthetic ONNX/YAML and checks import
parity, sparse histories, references, mapping, raw action semantics, fail-closed
validation, NPZ windows and the real SDK Python/fake-transport lifecycle.
When the sibling MimicLite checkout exists, it additionally compares observations
and history against its actual lightweight `MotorAngleInterface`; that checkout
is only a test oracle, never a deployment dependency. Shared ankle/safety tests
keep SONIC/SuperMimic behavior and cover MimicLite's native path.

`tests/test_actor_mimiclite_v2.py` and `test_actor_mimiclite_v2_edges.py` cover
1224-wide wrapper parity, ten-frame ordering, named/default-relative joints,
reference velocity provenance, reset/hold/window boundaries, and quantized
near-unit reference quaternions. Training-checkout oracle tests are optional;
actual-checkpoint trajectory receipts should accompany a release.

These tests do not substitute for a real trained policy, S100 native binary
inference, HIL, real Pico quality or physical robot stability acceptance.
