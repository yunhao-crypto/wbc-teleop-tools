# 27000 SMPL and stationary three-point Pico deployment

`pico_smpl` and `pico_teleop` retain their original 27000 encoders, FSQ and shared
decoder. Both use the Git actor's `run_live` function body through explicit typed
reference hooks. The existing SONIC, SuperMimic and MorphMimic choices keep their
entry, default SDK and requirements.

The new modalities require an **independent BodyCmd4 package**. The repository's
default SDK 0.3.0 is not replaced. The separate build selects the pinned
`1.0.0rc1+bodycmdv4.6863ad7` wheel: native motor A/B commands, virtual PR feedback.
It retains the original ankle IK/Jacobian, affine PD targets, model gains, command
envelope, SDK ownership and cancellation protections. This codec build is not an
official SDK release and has no new hardware acceptance claim.

## Build and installation

Weights, neutral-reference NPZ files, historical input fixtures, hardware-snapshot
fixtures and the separately supplied BodyCmd4 wheel are not published here.
`deploy/pico_27000/delivery_inputs.json` lists their required relative paths,
sizes and hashes. Provide that asset tree separately (an extracted previously
verified 27000 package can serve as the asset root). Only explicitly listed files
are read; the builder never copies its venv, logs, credentials or install receipts.

```bash
python scripts/build_pico_27000_delivery.py --assets-root /absolute/27000-assets
```

The builder resolves the ARM64 wheel closure offline, verifies all external
assets and seals code bindings and source revision. It builds two independent
archives with identical payloads and different root directories. Commit the
source before a release; `--allow-dirty` marks development builds explicitly.
Extract an archive in a new `/userdata` directory and execute each step only
after its predecessor passes:

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_pico_inputs_s100.py --verify-only
python3.10 scripts/bootstrap_pico_inputs_s100.py
bash scripts/start_actor_pico.sh configure --robot-id <local-S100-IP> --robot-config /absolute/effective.yaml
bash scripts/start_actor_pico.sh pico_smpl check --steps 240
bash scripts/start_actor_pico.sh pico_smpl state-check --seconds 5
```

Bootstrap performs ORT and native SDK checks for both modalities and an in-memory
BodyCmd4 codec check, with no Pico initialization, service or controller creation.
The live entry requires the current payload's successful installation receipt.
Use `pico_teleop` in place of `pico_smpl` for the other modality.

After confirming current firmware/configuration, fresh healthy feedback and
on-site support, first run a supported short stand/exit trial without triggers:

```bash
bash scripts/start_actor_pico.sh pico_smpl auto --robot-id <local-S100-IP> --robot-config /absolute/effective.yaml --seconds 10 --record-observations
```

After actual acceptance, local startup uses the same Git-style interface:

```bash
bash scripts/start_actor_pico.sh pico_smpl
bash scripts/start_actor_pico.sh pico_teleop
```

These commands request real motion. Exit requests CANCEL/PASSIVE, which removes
active standing support. Video is optional with `--no-camera`. The old
`start_actor_pico_inputs.sh <route> auto` remains no-action inference.

## Reference and protection boundaries

SMPL uses raw 24-joint XYZ/root references, a fully received 180ms window and
100ms replay margin; six robot wrist references stay at the model standing pose.
Teleop extracts wrists 20/21 and head 15, calibrates reference-base site poses and
keeps fixed lower-body references while retaining all leg feedback and outputs.
Both neck joints remain fixed. No zero-slip or zero-step guarantee is made.

The original Git robot-joint lead-in cannot be fed into either encoder. The
adapter instead blends neutral task references over one second, with at least
three seconds preparation. `encoded_reference` records the actual model input.
SMPL re-engagement aligns reference heading; Teleop retains its standing heading.
Input freshness stays at 100ms; per-frame measured joint bounds/runtime-id checks
are retained in addition to the shared Git lifecycle guards.

Robot EXTERNAL must already match native `motor_theta`,
`ankle_virtual_pd_intent: false`, explicit affine-target opt-in and an adequate
finite position-target margin, with command interpolation disabled. This code
does not write robot YAML, firmware or system services.

Offline validation used original model inference with simulated I/O: 36 lifecycle
and model cases, four receive-process cases and ten delivery/entry checks passed.
Each route completed 524 simulated publications, 224 tracking steps and three
engagements; full-model versus split-model absolute error was 0.0. These are local
regressions, not current Pico reception, target ABI/performance or physical
stability acceptance. Keep hardware deployment paused until explicitly resumed.
