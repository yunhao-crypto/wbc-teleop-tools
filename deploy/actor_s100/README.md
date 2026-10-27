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

`wheelhouse/` contains eight pinned wheels: locomotion-aorta 0.2.1,
aorta-sdk/aorta-msgs 2026.10.20, vbot-humanoid-msgs 0.1.5,
FlatBuffers 25.12.19, NumPy 1.26.4, PyYAML 6.0.3 and pybind11 2.13.6.
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
- For live-input shadow only: matching robot firmware and its reviewed
  `/app_param/zenoh/s100_session_peer.json5`, plus `/app/script/env.sh`.
  Each device keeps its own private credentials. No robot credentials are in Git.
- A Pico headset client with full-body tracking, calibrated trackers and Send
  enabled. The headset client and robot firmware are separate installed products.

## Install and verify

From a fresh checkout on S100:

```bash
python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx
# Optional alternative, compiling the binding locally:
# python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx --build-pico
bash scripts/start_actor_pico.sh check --steps 100
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

## Maintaining the delivery

After rebuilding/changing the Pico vendor inputs, `scripts/package_pico_runtime.py`
creates archives from the unpacked vendor/runtime directories. It strips saved
database rows and replaces settings with a fresh template. Check its inputs
before publishing a new delivery. Add/update matching wheels explicitly; do not
pick a new SDK just by version date or bypass its schema guard.

Stage intended changes, run `scripts/seal_actor_delivery.py`, stage the regenerated
manifest and `ACTOR-PICO-MANIFEST.json`, then run
`scripts/verify_actor_delivery.py --host-inference --model /path/to/actor_g1.onnx`.
The latter exports only the Git index into a temporary directory, explicitly
imports and verifies the supplied external model, fully resolves ARM wheel dependencies
offline, verifies/unpacks archives, tests actual C++ snapshot source and runs
real host inference. Its receipt is `outputs/actor_delivery_validation.json`.

Passing that check establishes file/resource closure and host numerical behavior.
It is not a claim of executing the ARM bootstrap, live Pico, robot feedback or
50 Hz performance on S100. Those require a separate target run.
