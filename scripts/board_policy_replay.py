#!/usr/bin/env python3
"""S100 no-action replay: compare every raw action with original FP32."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from deployment.vita_s100_aorta_30dof.onboard_backend import NativeBackend


def stats(x):
    a = np.asarray(x, np.float64)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a,50)),
                p95=float(np.percentile(a,95)), p99=float(np.percentile(a,99)),
                max=float(a.max()), over_20ms=int((a>20).sum()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--velocity-backend',choices=['ort','hbm'],default='ort')
    p.add_argument('--serial',action='store_true')
    p.add_argument('--library',type=Path,help='Explicit candidate native library for offline comparison')
    p.add_argument('--limit',type=int)
    p.add_argument('--manifest',type=Path,help='Explicit selected Motion package manifest')
    a=p.parse_args()
    manifest=json.loads((a.reference/'manifest.json').read_text())
    model_manifest=a.manifest or a.root/'outputs/model_runtime_manifest.json'
    deployed=json.loads(model_manifest.read_text())
    if manifest.get('source_sha256') is not None and manifest['source_sha256'] != deployed['source_hashes'][manifest['mode']]:
        raise ValueError('Oracle source does not match selected model package')
    motion=manifest['mode']=='motion'
    observations=np.load(a.reference/'observations.npy')
    references=np.load(a.reference/'reference_actions.npy')
    reset=np.load(a.reference/'reset_before.npy')
    step_indices=np.load(a.reference/'step_indices.npy')
    n=min(len(observations),a.limit or len(observations))
    if motion:
        ids=np.load(a.reference/'reference_expert_indices.npy')
        logits=np.load(a.reference/'reference_logits.npy')
        caches=np.load(a.reference/'reference_caches.npy',mmap_mode='r')
        cache_index_path=a.reference/'cache_indices.npy'
        cache_indices=np.load(cache_index_path) if cache_index_path.exists() else np.arange(n)
        cache_slots={int(i):j for j,i in enumerate(cache_indices)}
    started=time.monotonic()
    backend=NativeBackend(a.library or a.root/'build/libvita_backend.so',model_manifest,
                          velocity_backend=a.velocity_backend,parallel_router=not a.serial)
    load_seconds=time.monotonic()-started
    actions=[]; rows=[]; max_cache=0.; max_logits=0.; route_mismatches=[]; step_mismatches=[]
    # Warmup is followed by an explicit reset, never a hidden replay-seam reset.
    for _ in range(20):
        (backend.run_motion if motion else backend.run_velocity)(observations[0])
    backend.reset()
    started=time.monotonic()
    for i in range(n):
        # Native backend implements the original automatic8128 reset itself.
        if i and reset[i] and not (motion and step_indices[i-1]==8127):
            backend.reset()
        begin=time.monotonic_ns()
        result=backend.run_motion(observations[i],diagnostics=True) if motion else backend.run_velocity(observations[i])
        elapsed=(time.monotonic_ns()-begin)/1e6
        actions.append(result.actions.copy())
        rows.append(dict(index=i,python_inference_ms=elapsed,**result.timing))
        if motion:
            if result.timing['step_used'] != int(step_indices[i]):step_mismatches.append(i)
            if not np.array_equal(result.routes.reshape(-1),ids[i].reshape(-1)):route_mismatches.append(i)
            max_logits=max(max_logits,float(np.max(np.abs(result.logits.reshape(-1)-logits[i].reshape(-1)))))
            if i in cache_slots:
                max_cache=max(max_cache,float(np.max(np.abs(result.cache-caches[cache_slots[i]]))))
    wall=time.monotonic()-started
    state=backend.state(); backend.close()
    actions=np.stack(actions); error=np.abs(actions-references[:n]); worst=np.unravel_index(np.argmax(error),error.shape)
    finite=bool(np.isfinite(actions).all())
    report=dict(mode='motion' if motion else 'velocity',velocity_backend=a.velocity_backend,
        parallel_router=not a.serial,steps=n,source_reference=str(a.reference),load_seconds=load_seconds,
        finite=finite,max_abs_error=float(error[worst]),worst_step=int(worst[0]),worst_coordinate=int(worst[1]),
        coordinates_above_1e4=int((error>1e-4).sum()),steps_above_1e4=int(np.any(error>1e-4,axis=1).sum()),
        numerical_pass=finite and bool((error<=1e-4).all()),route_mismatch_steps=route_mismatches,
        step_mismatch_steps=step_mismatches,max_cache_abs_error=max_cache,max_logit_abs_error=max_logits,
        native_total_ms=stats([r['total_ms'] for r in rows]),
        python_inference_ms=stats([r['python_inference_ms'] for r in rows]),
        wall_seconds_with_comparison=wall,final_state=state,
        action_sent=False,scope='Policy replay, own KV; excludes state decode/Pico/observations/output IO. Recorded previous-action features remain fixed.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    np.save(a.out.with_suffix('.actions.npy'),actions)
    a.out.with_suffix('.timings.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['numerical_pass'] and not step_mismatches else 2)


if __name__=='__main__':main()
