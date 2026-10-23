# actor_g1 + Pico on S100

This entry runs `/host/actor_g1.onnx` through the installed SDK 0.2.1 WBC
observation/decoder math. The model is supplied separately (excluded from Git)
and copied by the bootstrap to `artifacts/actor_g1/actor_g1.onnx`
and SHA256 pinned in `config/actor_pico.json`. Its metadata identifies
`vita02a_pd40_aniso_dr`, input `[1,1520]`, output `[1,28]`, and 50 Hz.
It is the Vita28 tracking actor, regardless of the `g1` filename.

The new entry supports model checks, offline Pico replay, and live-input shadow
inference. Shadow subscribes to Body31 feedback and reads the existing Pico
service. It writes computed targets to JSONL; it never creates a controller,
requests EXTERNAL or publishes a robot command. The existing P1/P2/P3 runtime,
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
placement, collision avoidance or dynamic balance, and has not been accepted
for robot actuation. Its kinematic checks are separate from training-distribution
or real-human motion quality validation.

The actor requests 10 reference frames spaced 100 ms apart. The causal input
buffer waits for the entire 900 ms horizon plus a 20 ms forward-difference tail.
The resulting reference delay is **920 ms**, in addition to capture/compute
latency. These are real past samples interpolated at 50 Hz; future motion is
not predicted or filled with repeated terminal poses. Removing that delay
requires a validated predictor or a differently trained/exported model.

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
bash scripts/start_actor_pico.sh shadow --seconds 60 --group default
```

Connect the Pico XRoboToolkit app to the S100's reachable LAN IP, enable
Full body and Send, and keep the trackers calibrated. The shadow entry starts
and owns `RoboticsServiceProcess`; do not start a second service for that run.
It needs no Xbox. Ctrl-C ends shadow and closes its reader/subscriptions/service.

Each run creates a unique `outputs/actor_pico_*.jsonl`; the Pico service writes
`outputs/pico_service_actor.log`. `--record-observations` also records all 1520
input values. Successful inference records include `actions_sent=0`, the
retarget residual, input age, the 920 ms reference delay, inference timing and
all 31 inert output slots. An overrun record includes logging time. `check`
does not open the Pico service or check hardware inputs.

The bootstrap creates the interpreter with the proper permissions and restores
executable mode on the service from the sealed archive. The actor shadow entry
requires no Xbox configuration or Wi-Fi controller key.

## Replay NPZ schema

`body_poses [T,24,7]` contains Pico world XYZ/XYZW.
`sample_times_ns [T]` is strictly increasing integer nanoseconds.
`robot_joint_names [J]` explicitly names the columns of
`robot_joint_pos [T,J]` and `robot_joint_vel [T,J]`.
`robot_root_quat_wxyz [T,4]` and `robot_root_angvel_local [T,3]` contain the
model's HIP-frame state convention. Units are radians, rad/s and seconds/ns as
named. Every one of the actor's 28 joints must be present. Replay inference
starts only after the full 920 ms window is available.

## Validation boundary

Host CPU inference and synthetic Pico replay are exercised. S100 inference
latency, live Pico/tracker quality, real robot feedback, and robot actuation
are **not_measured**. Moving the robot would additionally require validating
the retargeted reference distribution, control transitions and the actor's
robot-side virtual-ankle PD contract; this shadow entry has no actuation mode.
