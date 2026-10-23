# Persistent Vita inference backend (no actuation)

This library contains no Aorta, robot SDK, control lease, publisher, actuator command, GPU provider or Pico dependency. It loads both original FP32 policies once and returns raw, **unclipped** network actions. It is an inference component; the existing outer runtime still owns observation history, gait phase, last actions, clipping, joint-name mapping, reference anchors, software mode switching and inert output sinks.

The authorized acceptance rule is every evaluated timestep and action coordinate `abs(candidate - original_FP32) <= 1e-4`. Host checks and compilation are not that acceptance. Root owns actual S100 builds, numerical replay, per-policy backend comparisons, and end-to-end no-action shadow tests.

## Build on the target (root schedules builds/runs)

Use the isolated ORT **1.23.2** bundle, including headers and `lib/`; the vendor ORT 1.20.1 is not the matching C API for these headers. The target's installed UCP/DNN headers/libraries are used. No cmake or system installation is required.

```sh
scripts/build_onboard_backend.sh \
  --ort-root /userdata/holomotion_vita_shadow_20260929/ort1232 \
  --ucp-root /usr --output build/libvita_backend.so
```

The build uses C++17, O3, `-ffp-contract=fast`, no fast-math, ARMv8.2-A NEON and pthreads. The explicit ORT/UCP runtime paths are embedded in rpath. The build script does not infer, warm up, publish or contact the robot SDK. `--syntax-only` also works on a host with matching headers; its non-ARM router fallback exists for host compilation only, not an ARM numerical guarantee.

## Python interface

Keep `outputs/model_runtime_manifest.json` and `../artifacts/` relative paths together when staging. Alternatively pass `artifact_root` with paths relative to that root, or an explicit artifact mapping with absolute paths. Every used artifact requires a SHA256. C++ checks all graph/router hashes and the actual complete loaded NK pool bytes before inference. Pool loading includes all 1024 experts, 1.5 GiB, without pruning/freezing/quantization.

```python
from deployment.vita_s100_aorta_30dof.onboard_backend import (
    NativeBackend, NativeMotionRunner, NativeVelocityRunner,
)

backend = NativeBackend(
    "build/libvita_backend.so", "outputs/model_runtime_manifest.json",
    velocity_backend="ort",  # or "hbm" when the HBM was provided at startup
    parallel_router=True, timeout_ms=100, deadline_ms=20.0,
)
velocity_runner = NativeVelocityRunner(backend)
motion_runner = NativeMotionRunner(backend)
# Both .run(obs) return original FP32 raw action arrays for the outer runtime.
# These must be sent only to the approved inert shadow path in this task.
result = backend.run_motion(obs_3126_float32, diagnostics=True)
# result.actions: [30]; optional result.cache/routes/logits retain ONNX shapes.
# result.timing contains total_ms, step_used/next, reset/call counts, components.
backend.run_velocity(obs_990_float32, backend="hbm")
backend.reset()
print(backend.state())
backend.close()
```

`run_velocity` and `run_motion` require FP32 shape `[D]` or `[1,D]` and finite values; no silent dtype conversion. `copy=True` returns owned arrays. `copy=False` exposes reusable output buffers valid only until the next call to that policy; callers must copy before retaining them. The diagnostic KV/routes/logits buffers are also reusable. `NativeMotionRunner.reset()` delegates exactly once to native reset; the existing `VitaDualPolicyRuntime.reset()` then separately resets velocity history/gait and both last_action buffers. `NativeVelocityRunner` has no native hidden state.

The two runner adapters expose `.session.get_providers()` for existing warmup/status logging. They are not ORT sessions and do not expose `get_inputs()`/metadata inspection; construct the existing contracts from the checked model metadata outside these adapters.

## Motion ABI and recurrence

- Frontend: `obs` FP32 `[1,3126]`, `past_key_values` FP32 `[1,2,1,32,4,64]`, `step_idx` INT64 `[1]` → original norm2 hidden, attention residual, present KV.
- MoE: NK selected weights `[2,512,512]`, `[2,512,256]`, normalized gates `[1,2,1]`, hidden `[1,1,512]` → update `[1,1,512]`.
- Backend: original residual + update → `actions` FP32 `[1,30]`.
- Router parameter format: `VROUT001` + six little-endian uint32 `(3126,3028,3040,2048,512,1024)`, then int32 observation indices `[3028]`; FP32 mean/divisor `[3126]`; row-major W0 `[2048,3040]` with exact zero padding, b0 `[2048]`, W2 `[512,2048]`, b2 `[512]`, W3 `[1024,512]`.

A single persistent router worker overlaps the independent ORT frontend. `parallel_router=False` runs the same router serially, allowing comparison against the parallel implementation. Expert rows go straight from immutable cached NK pools into the queried UCP tensor byte-stride segments, splitting segments at the logical selected-expert boundary. All input cache CLEAN and output INVALIDATE, task construction/submit/wait/release and full FP32 backend are included.

The absolute step begins at zero. At the start of a Motion call, `step>=8128` resets KV+step, matching the original default 8192−64 margin; this is not the 32-slot modulo write rule. Only a successful complete inference with finite outputs commits new KV and increments the step and successful-call count. Explicit reset clears KV/step without loading models, allocating native buffers or creating threads; lifetime counters persist. The library does not touch external last_action or reference state. Long acceptance needs at least **8129 consecutive Motion calls without mode reset**, separate from rapid bidirectional switching tests.

## Velocity backends

The original FP32 ORT MLP is always resident. If `velocity_hbm` is in the startup manifest, the complete FP32 HBM is also loaded then; both choices are available without switching-time load/initialization. The HBM may itself contain CPU/BPU mixed lowering and must expose `obs` FP32 `[1,990]` → `actions` FP32 `[1,28]`. No quantized external ABI is accepted. CPU vs HBM selection remains a board-measurement decision, not a claim that one is faster or meets precision merely because it compiles.

## Errors, ownership and timeout limits

ABI status codes: 0 success, 1 invalid argument/artifact/ABI, 2 busy/concurrent call, 3 timeout, 4 nonfinite values, 5 runtime error, 6 poisoned session. ctypes raises `BackendError.status`; invalid Python observations raise `ValueError`. C calls also validate buffer sizes and struct version/size. All entry points catch C++ exceptions; no exception crosses the C ABI.

Only one call at a time is allowed per session; the C++ API and Python wrapper reject concurrent use instead of waiting indefinitely for an application lock. External code must serialize destruction versus C calls; the Python wrapper does so. Shared observations used by the router live in native frontend storage, so a Python input going out of scope after failure cannot invalidate pending work.

UCP wait and router completion waits have a positive configured timeout. A failed execution marks the whole backend poisoned; subsequent inference/reset is rejected. Pending device tasks and their model/input/output buffers remain alive. `close()` succeeds only after outstanding work is known complete; on timeout it raises and the handle remains valid for a later close retry. Do not free it manually. The destructor deliberately retains native resources if safe closure is not established, rather than freeing active DMA memory. Process teardown/supervision is the final recovery path. This is fail-closed inference recovery, never a control command.

**CPU ORT calls and OS thread scheduling do not have a proven hard preemptive timeout.** `deadline_ms` only reports a completed call exceeding its deadline; it does not abort FP32 arithmetic or hide late samples. A process-level watchdog in the outer no-action shadow runtime is required to detect an unresponsive native call. Successful task timing does not establish a 20 ms hard real-time guarantee.

`total_ms` includes native input copying/validation, automatic KV reset when due, worker scheduling, complete inference, finite checks, successful state commit and requested output copies. It excludes startup hash/load, Python conversion/copies, sensors/Pico/SMPL, history building, mapping, logging and external scheduling. `router_ms` overlaps `frontend_ms` in parallel mode and must not be added to `parallel_ms`. `moe_ms` includes UCP task construction through release (also used for the Velocity HBM interval). The outer runtime must separately time the complete pipeline and switching first cycle.

## Checks already performed

Host `-Wall -Wextra -Werror -fsyntax-only` passed with real ORT1.23.2/UCP3.13.6 headers. `tests/test_host_contract.py` independently compiles the C header, compares actual C++ struct sizes/offsets with ctypes, checks eight expert-copy cases (contiguous segments crossing selected-row boundaries, padded stride, reversed and duplicate IDs), checks rejected overlapping layout/out-of-range expert, input validation and manifest resolution. See `outputs/backend_host_tests.json`.

These checks are not ARM execution, original-model numerical acceptance, timeout fault injection, or end-to-end latency validation. Those remain root-owned board checks.
