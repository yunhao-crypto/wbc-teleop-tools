# Vita02A S100 deployment

This is the native VBot deployment profile in the shared HoloMotion workflow.
It reuses the common policy runtime, SMPL reference, mode state machine and
controller semantics used by G1/K1/R1; only the S100 transport, slot mapping,
joint envelope and Vita policy ABIs are robot-specific.

## Models

- Velocity: official Vbot-lab `model17000_deploy.onnx`, SHA256
  `f78fd39eb3f420675af63782d12ce2e6434f62e3159e1fd5b17b65b4fd6d7126`.
- Motion: HoloMotion `model_20000.onnx`, SHA256
  `881d936f560e22e81302483b394cd6a89ba5648e6c2893fa153172dec38033a7`.
- `Exported_policy/walk.pt` is the obsolete 20-DoF policy and is rejected.

The official Velocity model is a 28-DoF, 990-input policy. It controls no head
joints and was trained with `lin_vel_y=0`; the deployment therefore keeps the
two neck joints at q=0, KP=20, KD=1 and ignores lateral stick commands. Motion
uses the unified 30-DoF, 3126-input recurrent ABI.

## Inference backends on DPX

The normal entry and `check` use CPU for Velocity17000 and CUDA for
Motion20000, with the original ONNX weights and dtypes. Motion uses the
shared G1/R1 CUDA observation binding; TF32 is disabled. CUDA session creation
must succeed before entering the robot loop. The check receipt and session
startup event record the actual providers. SMPL reference construction remains
on CUDA. Velocity's small CPU network avoids an additional GPU round trip.

DPX's ONNX Runtime 1.23.2 requires CUDA 12 libraries, while its existing Torch
environment uses CUDA 11.8. The launcher adds only the process-local
`vendor/cuda12_ort/lib` directory (override: `HOLOMOTION_VITA_CUDA_LIB`) to its
library path. The deployed copies are CUDA runtime 12.4.127, cuBLAS 12.4.5.8,
cuFFT 11.2.1.3, cuRAND 10.3.5.147 and cuDNN 9.1.0.70; their original paths and
SHA256 hashes are in `vendor/cuda12_ort/MANIFEST.json`. No global Conda package
or GPU driver was replaced. A fresh installation needs those runtime libraries
as well as the model/source files before running `check`.

CUDA Motion retains the full observation snapshot in control logs. Its
`observation_snapshot_ms` measures the post-inference host copy separately;
that copy remains part of `policy_total_ms`. Benchmarking must include this
copy, ordinary control logging and state subscriptions, not just GPU kernels.

## Controller

- `START`: acquire EXTERNAL, reread measured joint positions and send them with
  model gains; interpolate positions over 3 seconds to Velocity defaults with
  constant gains, then immediately run zero-command Velocity;
- `A`: enable joystick speed commands without resetting the running policy;
- left stick vertical: forward/backward speed;
- right stick horizontal: yaw rate;
- `B`: Velocity to live Pico Motion;
- `Y`: Motion back to Velocity;
- `X`: leave EXTERNAL and return to Idle/PASSIVE; process keeps running;
- `BACK/View`: leave EXTERNAL and end;
- `Ctrl-C`: leave EXTERNAL and end.

Keep external support through preparation and policy entry. `START` now runs
the zero-speed policy continuously, not a permanently fixed pose. Sticks are
ignored until `A`; `A` requires completed entry and centered sticks. Logs label
the phases `prepare` and `zero_velocity`. These transitions
do not introduce a new automatic session timeout. This change does not by
itself establish unsupported hardware standing or ground-release approval.

The controller is attached directly to DPX. There is no Apex-to-DPX UDP input,
fake Pico source or application test timer in the product entry. State and
reference freshness remain safety guards. Each active validation session uses
an operator-approved temporary 7200-second (2-hour) S100 control request.
This is not unlimited: expiry still ends control, and BACK/Ctrl-C can end it
earlier. The installed rc.2 SDK has no public renewal operation. Acceptance
of the requested duration by the robot remains a hardware-entry check.

The client performs one model-specific position clamp (with float32 rounding
only), but no extra target slew limiting or silent PD-gain clipping. Gains
are validated before entry against the recorded robot configuration. An
out-of-range initial pose reports the joints and leaves the process in IDLE.
Fresh duplicate state IDs are accepted; real state expiry and ownership
loss remain errors. Raw targets, sent targets and SDK rejection telemetry
are recorded for diagnosis.

Velocity and Motion share a latest-sample LowState reader: it drains raw
queued messages without decoding intermediate frames, then decodes the last
sample once. Each drain is bounded by the 256-message queue capacity and the
read deadline. This avoids spending a policy cycle decoding discarded states.
Both policies wait for a 50 Hz cycle deadline before reading fresh state,
then infer and publish without a second wait in the command adapter. Missed
deadlines are skipped rather than replayed; overruns are logged, not hidden
by lowering the configured frequency. Command intervals can still vary
with per-cycle computation time.

Control logs include `state_read_ms`, `publish_period_s`, `timing_ms`,
`cycle_skipped_ticks`, source state timestamps, local receive time and SDK
publish start/end times. `timing_ms` separates waiting, state mapping, Pico
polling, controller input, mode handling, observation, ONNX inference,
action mapping, conditioning, status/guard checks, command construction
and SDK publication. Policy total includes its observation/ONNX/mapping
sub-stages; do not sum both. A separate `cycle_timing` record with the same
command ID measures the completed main JSON write and flush.
`lowstate_age_ms` measures time since local dequeue, not network delay.
Source robot timestamps and DPX monotonic timestamps belong to different
clocks and cannot be directly subtracted to claim one-way network latency.

The DPX inference environment is Python 3.11, while the pre-existing
XRoboToolkit binding is Python 3.10-only. `scripts/build_xrobotoolkit_sdk.sh`
builds the same local SDK source for Python 3.11 into the Vita release vendor
directory; the launcher does not modify the Conda environment.

The current S100 software has a vendor-confirmed token-propagation
incompatibility with the rc.2 peer path. Until the matching S100 software
upgrade is installed, deployment uses a private client derivative at
`/home/dpx/holomotion_vita/config/pc_session_client.json5`; it differs from the
validated peer profile only in `mode`. The original private peer profile is
retained for rollback. Set `HOLOMOTION_VITA_AORTA_PROFILE` explicitly to select
another reviewed profile after an upgrade. Credentials remain mode `0600` on
DPX and are never copied into Git.

## Commands on DPX

```bash
# Byte/ABI check only; never opens EXTERNAL.
bash /home/dpx/start_vita_local_pico.sh --check-only

# Direct controller display; Ctrl-C ends it.
bash /home/dpx/start_holomotion_vita.sh controller-test

# Read-only LowState recording; Ctrl-C ends it.
bash /home/dpx/start_holomotion_vita.sh diagnostic-monitor \
  --output /home/dpx/holomotion_vita/logs/diagnostics

# Fixture-only START -> A -> B validation; Back or Ctrl-C ends it.
bash /home/dpx/start_vita_suspended_validation.sh

# Same supported validation through the normal current-release entry.
bash /home/dpx/start_vita_local_pico.sh
```

Every teleop session automatically writes a line-buffered control trace under
`/home/dpx/holomotion_vita/logs/sessions/`.

The loop schedules fresh-state cycles at 50 Hz; the command publisher does
not add a second sleep after inference. Missed ticks are not replayed. An SDK
single-frame reject is recorded with the command ID and available status
details; the client continues only while it still owns EXTERNAL and the server
remains in `RUNNING` or its documented one-frame-recovery `TIMEOUT_HOLD` phase.
State validity and ownership loss still terminate the command path.

The command-side Aorta node is created before `READY`. Cold discovery may take
up to 30 seconds; the launcher waits for both its own LowState subscription and
`control/status`, rejects an existing EXTERNAL owner and only then accepts
`START`. A single PREPARE/COMMIT or exit RPC retains a separate 5-second
timeout.

## Current release and validation scope

The current package is `vita_velocity17000_motion20000`. The normal local Pico
launcher and the suspended-validation launcher both select `release/current`.
The old model-directory arguments and old bundle generation fallback are removed.
The current v4 Motion artifact is from `tcloud-bj-053f8222e684`; a filename ending
in `20000` alone does not identify the correct training run.

The bundle retains its existing `suspended_validation` scope and reviewed HIP
IMU contract. The normal launcher explicitly selects that existing validation
path. `check` separately reports ground-release eligibility and the supported
validation contract; neither report proves physical stability.

START and Velocity use all 28 model17000 defaults/gains, plus neck q=0/KP=20/KD=1
and official open fingers. Motion uses all 30 model20000 defaults/gains. A keeps
Velocity history and clock. B resets Motion state and the SMPL root anchor,
aligns yaw and uses the current robot/reference directly. Y resets the policy
state and returns to zero-command Velocity. There is no policy target/gain blend.

Velocity bounds come from its ONNX metadata (ankle pitch approximately
[-0.960, 0.611] rad). Motion bounds come from the named joints of the same-run,
hash-checked URDF. They replace the old client joint limits; model observation
and action clips remain. No added slew limits, gain scaling or ankle compensation
are applied. Velocity commands span vx [-0.7, 0.9] m/s, vy=0, yaw [-0.8, 0.8] rad/s.
Command IDs wrap from 65535 to 1. Traces include the ONNX action, model target,
conditioned SDK target, gains, state age, send period and rejection status.

# Self-trained HoloMotion Velocity bundles

The runtime supports both the official `990 -> 28` Velocity ABI and the
HoloMotion `752 -> 28` ABI. The latter requires a checksum-pinned
`velocity_config` artifact in `bundle.yaml`: actor term order, joint selection,
scaling and history come from that training config, using the shared deployment
observation builder. It has no legacy gait clock or deployment-only action clip.
KP/KD, default angles and action scales come from each policy's ONNX metadata;
the two omitted neck actions use the Velocity model's own exported defaults.

A bundle can carry `deployment_config/config.yaml`; the launcher uses it unless
`HOLOMOTION_VITA_CONFIG` is explicitly set. Zero joystick-axis limits explicitly
disable commands absent from an acquisition model's training range. This does
not change existing bundles or the shared START/A/B/Y/X/BACK mode meanings.

Copying a v4 URDF into the bundle records the trained asset; the SDK does not
load that URDF. In S100 position mode, the firmware's parallel-ankle motor PD
remains a separate contract. Matching exported virtual KP/KD is not proof of
global equivalence to the firmware's motor-space PD. No-action checks are not
ground-standing, walking or Motion acceptance.
