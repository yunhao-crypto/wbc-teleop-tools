"""Original FP32 oracle on recorded policy inputs, including the RoPE reset."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from model_package_manifest import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8200)
    a = p.parse_args()
    if sha(a.source) != a.expected_source_sha256 or a.steps < 8130:
        raise ValueError('Source identity or continuous reset coverage invalid')
    a.output.mkdir(parents=True, exist_ok=False)
    source = np.load(a.observations, mmap_mode='r', allow_pickle=False).reshape(-1, 3126)
    observations = np.ascontiguousarray(source[np.arange(a.steps) % len(source)], dtype=np.float32)
    if not np.isfinite(observations).all():
        raise ValueError('Nonfinite oracle observation')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(a.source), sess_options=options,
        providers=[('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider'])
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('Original CUDA FP32 oracle unavailable')
    cache = np.zeros((1, 2, 1, 32, 4, 64), np.float32)
    cache_indices = sorted({0, 1, 31, 32, 33, 127, 1023, 8127, 8128, 8129, a.steps - 1})
    caches, actions, routes, logits = [], [], [], []
    step_indices = np.arange(a.steps, dtype=np.int64) % 8128
    reset = step_indices == 0
    started = time.monotonic()
    for i, step in enumerate(step_indices):
        if reset[i]:
            cache.fill(0)
        out = session.run(None, {'obs': observations[i:i+1],
            'past_key_values': cache, 'step_idx': np.array([step], np.int64)})
        action, cache, ids, logit = out
        if not all(np.isfinite(x).all() for x in (action, cache, logit)):
            raise ValueError('Nonfinite original oracle result')
        actions.append(action.reshape(30))
        routes.append(ids.reshape(2))
        logits.append(logit.reshape(1024))
        if i in cache_indices:
            caches.append(cache.copy())
        if i % 1000 == 0:
            print('ORACLE', i, flush=True)
    for name, value in {'observations': observations, 'reference_actions': np.stack(actions),
        'reference_expert_indices': np.stack(routes), 'reference_logits': np.stack(logits),
        'reference_caches': np.stack(caches), 'cache_indices': np.array(cache_indices),
        'reset_before': reset, 'step_indices': step_indices}.items():
        np.save(a.output / (name + '.npy'), value)
    report = {'mode': 'motion', 'source_sha256': a.expected_source_sha256,
        'observation_source': str(a.observations), 'observation_source_sha256': sha(a.observations),
        'steps': a.steps, 'providers': session.get_providers(), 'tf32': False,
        'seconds': time.monotonic() - started,
        'scope': 'Recorded policy inputs repeated at source end; independent original KV; reset at 8128; not closed-loop robot simulation'}
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ORACLE_READY', report, flush=True)


if __name__ == '__main__':
    main()
