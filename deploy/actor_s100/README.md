# S100 actor/Pico offline delivery

This directory versions dependency locks and the inputs for S100 offline packages. The separate `live` entry can
move the robot only after explicit on-site confirmation; see
[supported trial instructions](../../README_ACTOR_PICO.md#supported-external-trial).
Installation and validation never run that entry. Model weights are supplied
separately and are excluded from Git. The model identity,
size, SHA256 and installation path are pinned in `config/actor_pico.json`
(`artifacts.model`) and `manifest.json` (`external_resources`). Supply the separately delivered
model matching those records; the local filename does not select its version.
The optional SuperMimic actor has its own pinned model in
`config/actor_pico_supermimic.json`; install it with
`--policy supermimic --model /path/to/supermimic_vita02a.onnx` (see
[policy selection](../../README_ACTOR_PICO.md#policy-selection)).
Use a freshly extracted package directory for a changed model delivery. A different existing
model/environment is preserved rather than overwritten.
The URDF, source code, tests and configuration are tracked. Tests and developer
documentation stay in the source checkout; packages contain runtime inputs only. No original `venv`, `reference`,
`model_dependencies_readonly`, `vendor_packages` or `build` directory is needed
from the developer's machine.

The versioned [PICO Android setup guide and APK](../pico_viewer/README.md)
cover a new headset, both network ports and view switching. On an installed
robot, `bash scripts/start_actor_pico.sh connect` starts a connection/video-only
session without loading a policy. Live/auto/shadow entry points now own their
optional camera relay and clean it up after the actor exits; `--no-camera`
disables video. Offline validation modes do not start it.

## Included dependencies

`wheelhouse/` contains 21 pinned wheels: locomotion-aorta 0.3.0,
aorta-sdk/aorta-msgs 0.0.0+g7649a87ebb03, vbot-humanoid-msgs 0.0.0+g35ce9eb82d5b,
FlatBuffers 25.12.19, NumPy 1.26.4, PyYAML 6.0.3 and pybind11 2.13.6, plus the complete pinned GMR dependency closure
(mink 1.3.0, MuJoCo 3.11.0, SciPy 1.15.3, DAQP 0.10.3 and qpsolvers 4.13.0).
The optional [SONIC ONNX planner and map bridge](SONIC_ONNX_PLANNER.md) add seven
pinned ARM64 wheels, bringing the bundled closure to 28 wheels.
`requirements.txt` selects the 21-wheel base runtime;
`requirements-planner.txt` includes it and adds the seven planner wheels.
The package variant selects the requirements automatically. The base installer
does not install or import Python ONNX Runtime.
The bootstrap uses pip's full dependency resolver with `--no-index` and runs
`pip check`; there is no network or missing-dependency fallback.

The Pico service archive contains the supplied ARM64 service, its SDK, shared
libraries and Qt plugins. It includes fresh local-service settings and a new
empty database with the vendor factory schema, not saved device/connection
records. The binding archive contains the existing atomic-snapshot CPython 3.10
ARM64 extension and double-conversion dependency. Their C++ source is tracked
under `third_party/`, so the extension can also be built on S100.
The dual-trigger revision additionally exports an atomic controller-packet
snapshot with native receipt age. Its prebuilt extension was cross-compiled
with aarch64 GCC 11 and the target's Python 3.10 headers, then imported and
queried on S100 without initializing the SDK. The SDK/service binaries and
double-conversion dependency are unchanged; actual controller input is a
separate hardware check.
The binding source also exports a same-packet side-grip snapshot used by the
opt-in OP3 gripper control; see `PICO_OP3_GRIPPER.md`. That needs a rebuilt
binding and an SDK with the end-effector client.
Retain the original license/notices associated with each dependency.

## Target prerequisites

- Linux aarch64 with glibc >= 2.35 and Python 3.10 with venv/ensurepip.
- The target OS C/C++ runtime and Qt platform dependencies. Installation checks
  `ldd` and reports missing libraries rather than assuming their presence.
  The bundled Qt 6 runtime expects the Ubuntu 22.04 ARM64 system library ABI:
  ICU 70, GLib 2, PCRE2-16, Brotli, zstd and zlib, in addition to glibc and
  libstdc++. These are target OS prerequisites, not replaced by this repository.
- For `--build-pico` only: g++, Python 3.10 development headers. pybind11 headers
  are installed from the included wheel.
- For feedback, shadow and live: matching robot firmware and its reviewed
  `/app_param/zenoh/s100_session_peer.json5`, plus `/app/script/env.sh`.
  Each device keeps its own private credentials. No robot credentials are in Git.
- A Pico headset client with full-body tracking, calibrated trackers and Send
  enabled. The headset client and robot firmware are separate installed products.

The pinned SDK build follows official main `a0b50cef329181b138239f5576238fcb9e0c02ca`
(merged PR #23, which delegates BodyCmd device ranges to robot EXTERNAL).
This is a local development repack, not an official SDK release: all 22 Python
modules match that commit; `controller.py` and `_policy_state.py` replace the
previous build's modules. Native build inputs, dependency pins and native library
bytes are unchanged; no new native build or ARM execution is claimed. Repack
provenance and the wheel and native library identities are
sealed in `config/sdk021-wheel-guard.json` (file name kept from 0.2.1); update the
per-deployment venv as well as the source/config files. Changing `/userdata/sdk-demo`
alone does not update a WBC trial environment.

The actor uses SDK `0.3.0`: control protocol 4 with numeric client/control IDs
and robot runtime binding, Body31 protocol 3 / roster 1, RobotInfo protocol 2,
28 policy joints and two neck joints; waist-roll slot 14 is MOCK and every
command leaves it zero. It needs robot firmware that speaks these protocols
(vita-robot master `896385d5b` or later). The earlier tested baseline
`1.6.0-200` (`c9b4e80c`, Aorta/messages `2026.10.20`) speaks only the 0.2.1
protocols and cannot run this delivery. No robot run with SDK 0.3.0 has been made.
A different firmware requires an interface/configuration review before live use.

**Out-of-range PD targets for the motor-theta / unclipped policies (SONIC,
SuperMimic, MorphMimic).** These policies send unclipped PD setpoints that leave
the advertised RobotInfo q range (on the robot-42 SONIC log about a fifth of
steps, mainly the ankles). Official SDK #23 does not enforce RobotInfo device
ranges for q/dq/tau/kp/kd and has no per-call exemption API. It still rejects
non-numeric or non-finite-float32 values and checks target/field declarations,
MOCK, capabilities and control identity. WTT retains its policy envelope, gain,
fixed-joint and measured-pose checks. Unclipped policies still require the robot
YAML to set `allow_position_targets_outside_mechanical_range: true` together
with a finite `position_target_margin_rad` in (0, pi]; firmware #13164 preserves
q within travel plus that margin. SDK submission alone does not prove acceptance
or unclipped output: the actor monitors EXTERNAL ownership and rejection/drop
counters, while firmware clipping can occur without a rejection.
SDK PR #22 is retired; firmware PR #13164 remains unmerged and HIL-unaccepted. Device-owned peer
credentials and the active effective A-sample YAML are external prerequisites,
not files to copy from the developer's machine. Live scheduling requires
permission for SCHED_RR priority 1.

## Build offline packages

On a build host with Python and pip, from a committed source checkout:

```bash
python3 scripts/build_actor_delivery.py --variant base
python3 scripts/build_actor_delivery.py --variant planner
```

Outputs go to `outputs/actor_delivery/`: a `.tar.gz` and its `.sha256` sidecar.
Each variant is a complete package; planner includes the base runtime. The
builder resolves only the selected ARM64/Python 3.10 wheel closure, offline.
It requires no installed SDK, external model weights or robot connection.
`--allow-dirty` creates an explicitly marked development package; release builds
require a clean checkout. `--output-dir` changes the output location.

Each package has one generated `deploy/actor_s100/manifest.json`: source commit,
development status, target ABI, selected requirements, runtime file sizes/modes
and hashes, and separately provisioned model identities. Its `payload_sha256`
identifies the runtime file list. Stable archive headers allow identical inputs
to reproduce the archive. Tests and developer documentation are excluded, so
editing them does not change that runtime file list. A new source commit still
changes provenance. Do not edit or stage generated manifests.

## Install and verify

Copy the archive and checksum to S100 and extract into a new directory:

```bash
sha256sum -c actor-s100-base-<commit>-<payload>.tar.gz.sha256
tar -xzf actor-s100-base-<commit>-<payload>.tar.gz -C /userdata
cd /userdata/actor-s100-base-<commit>-<payload>
```

Then run from that package directory (use the planner package for planner models):

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --model /path/to/policy.onnx
# Optional alternative, compiling the binding locally:
# python3.10 scripts/bootstrap_actor_s100.py --model /path/to/policy.onnx --build-pico
bash scripts/start_actor_pico.sh sonic check --steps 100
bash scripts/start_actor_pico.sh sonic state-check --seconds 5
bash scripts/start_actor_pico.sh sonic shadow --seconds 60 --group default
```

Supply the model from your model delivery location. The bootstrap validates its
size and SHA256 before copying it to the installation path declared in the configuration.
An already placed matching file can be reused without `--model`. Missing or
mismatched weights stop installation; they are never downloaded implicitly.
The default installation uses prebuilt libraries and needs no C++ compiler.
It never requests robot control. The first completed installation writes
`build/actor_s100_install.json`. The startup check is offline/synthetic;
`shadow` reads real input and only records outputs. Installation refuses to
overwrite an unrelated existing virtual environment. Reruns can resume an
environment created by this bootstrap, including switching to `--build-pico`.
An active operator using `build/operator.lock` blocks installation.

Archive extraction refuses different existing libraries. Device service settings
and its database are preserved after first extraction. Logs and generated
configuration are private/local outputs, not source inputs.

Any machine can inspect the sealed delivery without executing ARM code:

```bash
python3 scripts/bootstrap_actor_s100.py --verify-only
```

Run this inside an extracted package. It checks sealed runtime inputs and archives only; it does not verify the separate model.

`--unpack-only` also supports inspecting the runtime files on an x86 host, without
starting the service or creating an environment. Normal installation deliberately
rejects x86 or a different Python ABI.

## Configure this board without motion

After offline installation and read-only feedback validation:

```bash
bash scripts/start_actor_pico.sh configure \
  --robot-id '<actual local S100 IPv4>' \
  --robot-config /absolute/path/to/running/effective.yaml
```

This writes `outputs/actor_pico_local.json` with mode 0600 and manual dual-trigger
handoff, without SDK control handles. It verifies local IP, machine identity,
current end-effector selector and EXTERNAL YAML contract. It cannot determine
which YAML the running service actually loaded; the operator must supply that
path. Identical configuration is idempotent. `--replace` backs up changed defaults
before replacement. `--seconds N` chooses a client duration, default 0 until STOP.
For the reviewed no-gripper setup the effective file was
`/app/locomotion/config/locomotion_foot_humanoid_asample_none.yaml`; do not assume
that selection on another robot. No hardware configuration is changed here.

The launcher subsequently requests FIXED_STAND and EXTERNAL motion with
`bash scripts/start_actor_pico.sh sonic` or `supermimic`; it has no default policy.
That requires current on-site authorization/support, independent of configuration.
Height defaults to 1.80 m. Calibration and planted-foot targets are off for
every policy unless explicitly selected in its `retargeting` configuration.
Use `--human-height` for the actual operator; setting the height alone does
not enable calibration. An explicitly selected profile at a different height
switches both features off with `ACTOR_WARNING`.
See the [run guide](../../README_ACTOR_PICO.md#default-local-startup).

## Upgrade and rollback

Use a separate extracted directory per release; keep the old directory,
interpreter, model and device profile. The installer intentionally refuses
unrelated environments and changed existing runtime libraries. Do not run an
upgrade over an active operator, reset a running checkout, or copy private
credentials between robots. A fresh directory still needs the exact separately
delivered model, the same board's private peer configuration and newly generated
local defaults.

Finish the active session and inspect confirmed control status before selecting
an older directory. CANCEL returns PASSIVE and loaded limbs may fall; hardware
support and E-stop remain necessary. Rollback means choosing the preserved old
directory and doing its offline/read-only checks first, not automatically
restarting motion. No OTA, service restart or persistent motor change is part
of these commands.

## Maintaining the delivery

After rebuilding/changing the Pico vendor inputs, `scripts/package_pico_runtime.py`
creates archives from the unpacked vendor/runtime directories. It strips saved
database rows and replaces settings with a fresh template. Check its inputs
before publishing a new delivery. Add/update matching wheels explicitly; do not
pick a new SDK just by version date or bypass its schema guard.

Run `scripts/verify_actor_delivery.py` from the source checkout. It exports
current source into an isolated directory, builds and verifies both variants,
resolves their ARM dependencies offline, checks runtime archives, tests the
actual C++ snapshot source and runs installer/session regressions. Its receipt
is `outputs/actor_delivery_validation.json`. Use `--allow-dirty` during development.

For host numerical checks, install `requirements-host-check.txt` into a clean
Linux x86_64 Python 3.10 venv, then add
`--host-inference --model /path/to/policy.onnx` (and optionally
`--supermimic-model /path/to/supermimic_vita02a.onnx`). The verifier checks each
extracted package with synthetic host inference, then runs source regression
suites. It never borrows an existing venv or unpacked vendor runtime.

`scripts/seal_actor_delivery.py` is a compatibility alias for the builder; it
now writes a package, not tracked manifests. Wheels and prebuilt binding inputs
remain in Git for now; relocating them to release storage is a separate change.

Passing that check establishes file/resource closure and host numerical behavior.
It is not a claim of executing the ARM bootstrap, live Pico, robot feedback or
50 Hz performance on S100. Those require a separate target run.

## S100 offline validation

On 2026-11-14, the base and planner development packages were installed into
separate fresh directories on `10.100.100.67` (Ubuntu 22.04, aarch64, Python
3.10.12, glibc 2.35). Both passed offline pip resolution/check, the actual SDK
and Pico binding imports, the installer model check, and a further 100 actor
inference steps using the SDK backend. Python ONNX Runtime was absent from the
base venv and present in the planner venv. The base package rejected planner
assets, and the planner package rejected the board's different existing bridge.

The pinned planner and map bridge also passed real S100 ONNX generation and
actor inference with synthetic standing feedback for forward, turning and
backward commands. Four planner calls measured 260.7–297.7 ms; planner-reference
initialization, including asset verification and sessions, took 4.38 s. These
are small offline samples, not live scheduling or tracking measurements.
The test opened no robot control handles and sent no actions. The full local
receipt is `outputs/robot67_delivery_test/deployment_test_report.json`; raw logs
stay outside the runtime package and Git. Live Pico, physical tracking and
motion acceptance remain separate validation.
