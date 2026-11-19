# MimicLite v2 native tracking

Vita-robot runs the 1224-input MimicLite v2 actor at 50 Hz. WTT provides Pico/GMR
references, the operator entry and the S100 CPU layout. Fixed NPZ playback and
stand run in locomotion without the Pico service.

## Prepare the runtime and native bundle

Build the ordinary WTT base delivery and unpack the archive on S100. Install
its SDK and Pico dependencies without provisioning a WTT ONNX actor:

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --native-reference-only
```

The base delivery uses its pinned `locomotion_aorta` wheel. The launcher reads
BodyState, RobotInfo and ControlStatus through `PullStateSource`. Body commands
are emitted by native locomotion. The WTT bootstrap verifies the existing SDK
wheel and its dependencies before importing the native reference modules.

On the build host, use the dependencies in
[requirements-host-check.txt](deploy/actor_s100/requirements-host-check.txt).
Import a matching MimicLite v2 student ONNX and training YAML into a new bundle:

```bash
python3 scripts/import_mimiclite_policy.py \
  --policy-yaml /path/to/policy.yaml --onnx /path/to/policy.onnx \
  --model student --output /path/to/native-bundle
```

The importer preserves the trained graph, wraps its two inputs as one
`[1,1224]` input and verifies action parity on the host. It creates the wrapped
ONNX, a matching contract and their SHA-256 list. From the corresponding
vita-robot revision, copy the built `humanoid_tracking_client` and the four
launcher modules listed in its `humanoid/tools/MIMICLITE_NATIVE.md` into
the bundle. Provision that vita-robot binary/profile and install the same ONNX
at `models/vita02a/actor_mimiclite.onnx` relative to the locomotion daemon's
working directory. Deploy the prepared bundle and WTT runtime to S100.

## Verify and launch

From the installed WTT runtime:

```bash
bash scripts/start_mimiclite_native.sh --model student --native-root /absolute/native-bundle --check
bash scripts/start_mimiclite_native.sh --model student --native-root /absolute/native-bundle --plan
```

`--check` validates the bundle files and checksums. `--plan` previews the
command. Both are offline. After the robot and model have been accepted for
live tracking, run the same entry without either flag. Use `--stand-only` for
stand or `--motion /absolute/clip.npz` for fixed playback. NPZ input requires
named virtual joints and a velocity channel. With Pico, release then squeeze
both triggers to toggle following. Ctrl-C requests and confirms PASSIVE.

The launcher uses the same WTT operator lock as the existing entries and checks
the selected ONNX against the daemon's working directory before activation.
It uses the device-local Zenoh peer config. Set
`MIMICLITE_ZENOH_SESSION_SOURCE` if the reviewed peer config lives elsewhere.

The launcher, bridge and receiver run on CPUs 0, 2 and 3. GMR runs on CPU 1.
After the bridge confirms this session's activation, the inference pin helper
looks for the locomotion ONNX worker and defaults to CPU 4. Use `MIMICLITE_INFERENCE_CPUS=off` to skip the helper. The helper reports
its result under `outputs/`. A successful pin changes the worker's affinity until
locomotion restarts or the affinity is changed again. The Pico worker publishes
timestamped source frames and writes its GMR log through a bounded queue.
