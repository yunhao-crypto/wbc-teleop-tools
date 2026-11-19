# 35000 SMPL and stationary three-point Pico deployment

`pico_smpl` and `pico_teleop` retain their 35000 encoders, FSQ and shared
decoder. Both call the actor's single `actor_pico_live.run_live` function through
an explicit task-reference adapter. All five policies use the same actor directory, `venv`,
build/bootstrap/launch entry points and installation receipt.

Enable 35000 support in the original base build to select the shared pinned
`1.0.0rc1+bodycmdv4.6863ad7` wheel: native motor A/B commands, virtual PR feedback.
It retains the original ankle IK/Jacobian, affine PD targets, model gains, command
envelope, SDK ownership and cancellation protections. Existing policies retain
their original BodyCmd3 default; SMPL/Teleop explicitly selects BodyCmd4. One environment
supports both protocols. Builds without the Pico model option retain SDK 0.3.0.
This codec build is not an
official SDK release and has no new hardware acceptance claim.

The current SMPL, Teleop and G1 models all use checkpoint **35000**. The G1 model
selects the existing `sonic` route. Filenames and the `--pico27000-assets` alias
are retained for script compatibility; checkpoint identity comes from the pinned
model hashes and `checkpoint_iteration`, not those legacy filenames.

## Build and installation

Weights, neutral-reference NPZ files, historical input fixtures, hardware-snapshot
fixtures and the separately supplied BodyCmd4 wheel are not published here.
`deploy/pico_27000/delivery_inputs.json` lists their required relative paths,
sizes and hashes. Provide that asset tree separately (an extracted previously
verified compatible asset tree can serve as the asset root). Only explicitly listed files
are read; the builder never copies its venv, logs, credentials or install receipts.

```bash
python scripts/build_actor_delivery.py --variant base --pico-assets /absolute/35000-assets
```

The builder resolves the ARM64 wheel closure offline, verifies all external
assets and seals code bindings and source revision. It builds one standard
`actor-s100-base-*` archive containing both modalities and all original actor
entry points. The supplied 35000 G1 model is included; SuperMimic/MorphMimic weights remain separate. Commit the
source before a release; `--allow-dirty` marks development builds explicitly.
Extract an archive in a new `/userdata` directory and execute each step only
after its predecessor passes:

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --verify-only
python3.10 scripts/bootstrap_actor_s100.py --policy pico_smpl
bash scripts/start_actor_pico.sh configure --robot-id <local-S100-IP> --robot-config /absolute/effective.yaml
bash scripts/start_actor_pico.sh pico_smpl check --steps 240
bash scripts/start_actor_pico.sh pico_smpl state-check --seconds 5
```

Bootstrap performs ORT and native SDK checks for both modalities and an in-memory
BodyCmd4 codec check, with no Pico initialization, service or controller creation.
The live entry requires the current payload's successful installation receipt.
Use `pico_teleop` in place of `pico_smpl` for the other modality.
Both choices initialize the same `venv/.actor_bootstrap_owned` environment and
write `build/actor_s100_install.json`; the original policies use that same
environment. The bundled G1 model uses `--policy sonic` without `--model`;
SuperMimic/MorphMimic still require `--policy <name> --model ...`.
No second venv or SDK is installed for a policy switch. The old 27000-specific
builder/bootstrap scripts remain compatibility helpers, not a separate runtime.
The compatibility bootstrap delegates to the standard bootstrap transaction.
Both install and unpack acquire `build/operator.lock` before changing models,
libraries or receipts. An interrupted/failed transaction invalidates the old
installation receipts; success publishes the new receipt atomically.

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
The 2ms receiver latches each trigger press with the receipt time of the release
that armed it. The parent accepts it only when both occurred after the current
body-recovery boundary; a release before loss cannot arm recovery.

`JointReferences` supplies the original joint-reference behavior. `TaskReferences`
changes only input acquisition, heading alignment, neutral task references and
observation inference. The control loop owns startup, ownership checks,
publication and checked cancellation for every route. The copied lifecycle and
the two old modality-specific live runners have been removed.

Robot EXTERNAL must already match native `motor_theta`,
`ankle_virtual_pd_intent: false`, explicit affine-target opt-in and an adequate
finite position-target margin, with command interpolation disabled. This code
does not write robot YAML, firmware or system services.

The no-asset fault regressions are committed in `tests/test_pico_aligned_live.py`,
`tests/test_pico_trigger_recovery.py` and `tests/test_pico_install_modes.py`.
Run them with Python 3.10+, NumPy and PyYAML:

```bash
python -m unittest discover -s tests -p 'test_pico_aligned_live.py' -v
python -m unittest discover -s tests -p 'test_pico_trigger_recovery.py' -v
python -m unittest discover -s tests -p 'test_pico_install_modes.py' -v
```

The live tests call `run_aligned` -> `actor_pico_live.run_live`, using simulated
network/controller I/O; they cover both routes, trigger recovery, stale body
feedback, ownership loss, measured bounds, publish failures and checked exit.
The installer tests use real cross-process OS locking (Linux flock on target;
Windows byte locks for host tooling). These tests need no ONNX assets.

`check_pico_aligned.py` is separately a model/input-adapter parity check. It does
not exercise the live state machine or prove training semantics. Golden export
input/action parity, full-versus-split network parity and lifecycle tests are
separate evidence. None verifies current Pico reception, target ABI/performance
or physical stability. Keep hardware deployment paused until explicitly resumed.
