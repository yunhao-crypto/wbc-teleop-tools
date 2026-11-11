# S100 actor/Pico offline delivery

This directory versions the actor dependencies. The separate `live` entry can
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
Use a fresh checkout for a changed model delivery. A different existing
model/environment is preserved rather than overwritten.
The URDF, source code, tests and configuration are tracked. No original `venv`, `reference`,
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

The pinned SDK build follows humanoid-sdk `535cbbc02c5baa22157520491128aa211b61c997`
(main `abe21ab` plus PR #22, which adds `publish(..., q_limit_exempt_names=...)`;
only `controller.py` differs from the main wheel; development build, not a separately published SDK release; its manifest marks the
native inputs as not ABI-validated). The wheel and native library identities are
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
steps, mainly the ankles). The SDK checks every BodyCmd target q against that
range before sending, except the joints passed in `q_limit_exempt_names`. The
actor passes the policy's unclipped target joints only when the robot YAML sets
`allow_position_targets_outside_mechanical_range: true` together with a finite
`position_target_margin_rad` in (0, pi]; any other out-of-range target raises
`InvalidCommandError`, after which the actor cancels EXTERNAL and the robot goes
to PASSIVE. The firmware must implement the same opt-in (vita-robot PR #13086).
Neither SDK PR #22 nor firmware PR #13086 is merged or HIL-accepted. Device-owned peer
credentials and the active effective A-sample YAML are external prerequisites,
not files to copy from the developer's machine. Live scheduling requires
permission for SCHED_RR priority 1.

## Install and verify

From a fresh checkout of `main` on S100:

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

This checks tracked inputs and archives only; it does not verify the separate model.

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
Height defaults to 1.75 m; use `--human-height` for the actual operator.
See the [run guide](../../README_ACTOR_PICO.md#default-local-startup).

## Upgrade and rollback

Use a separate new checkout/directory per release; keep the old checkout,
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

Stage intended changes, run `scripts/seal_actor_delivery.py`, stage the regenerated
manifest and `ACTOR-PICO-MANIFEST.json`, then run
`scripts/verify_actor_delivery.py --host-inference --model /path/to/policy.onnx`
(add `--supermimic-model /path/to/supermimic_vita02a.onnx` to also check that actor).
For host inference, first install the separate
`requirements-host-check.txt` into a clean Linux x86_64 Python 3.10 venv; it
includes NumPy, ONNX Runtime, GMR and their host dependencies, not ARM binaries.
The latter exports only the Git index into a temporary directory, explicitly
imports and verifies the supplied external model, fully resolves ARM wheel dependencies
offline, verifies/unpacks archives, tests actual C++ snapshot source and runs
real host inference. Its receipt is `outputs/actor_delivery_validation.json`.

Passing that check establishes file/resource closure and host numerical behavior.
It is not a claim of executing the ARM bootstrap, live Pico, robot feedback or
50 Hz performance on S100. Those require a separate target run.
