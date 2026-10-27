# S100 actor/Pico offline delivery

This directory versions the actor dependencies. The separate `live` entry can
move the robot only after explicit on-site confirmation; see
[supported trial instructions](../../README_ACTOR_PICO.md#supported-external-trial).
Installation and validation never run that entry. Model weights are supplied
separately and are excluded from Git. Their required SHA256 is pinned in
`config/actor_pico.json` and `manifest.json` under `external_resources`:
`a341391a346e050caf11c40c6989336c658c445e3e124f12dcbe292d91324474`.
This is `model_step_431500_g1.onnx` (10 references at 20 ms spacing), installed
under the canonical `artifacts/actor_g1/actor_g1.onnx` path. The earlier
`f0dfa046…` actor has different weights, limits and timing and is not accepted
by this delivery manifest. Use a fresh checkout for the new model; an existing
different model/environment is preserved rather than overwritten.
The URDF, source code, tests and configuration are tracked. No original `venv`, `reference`,
`model_dependencies_readonly`, `vendor_packages` or `build` directory is needed
from the developer's machine.

## Included dependencies

`wheelhouse/` contains 21 pinned wheels: locomotion-aorta 0.2.1,
aorta-sdk/aorta-msgs 2026.10.20, vbot-humanoid-msgs 0.1.5,
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

The tested robot baseline is `1.6.0-200`, commit
`c9b4e80c074a8898d1290927a1be5350bc725d4a`, with Aorta/messages `2026.10.20`.
The actor uses SDK `0.2.1`, Body31 protocol 2 / roster 1, 28 policy joints and
fixed neck targets; waist-roll slot 14 is unpopulated. A different firmware
requires an interface/configuration review before live use. Device-owned peer
credentials and the active effective A-sample YAML are external prerequisites,
not files to copy from the developer's machine. Live scheduling requires
permission for SCHED_RR priority 1.

## Install and verify

From a fresh checkout of `main` on S100:

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx
# Optional alternative, compiling the binding locally:
# python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx --build-pico
bash scripts/start_actor_pico.sh check --steps 100
bash scripts/start_actor_pico.sh state-check --seconds 5
bash scripts/start_actor_pico.sh shadow --seconds 60 --group default
```

Supply the model from your model delivery location. The bootstrap validates its
size and SHA256 before copying it to `artifacts/actor_g1/actor_g1.onnx`.
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

The no-argument launcher subsequently requests FIXED_STAND and EXTERNAL motion.
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
`scripts/verify_actor_delivery.py --host-inference --model /path/to/actor_g1.onnx`.
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
