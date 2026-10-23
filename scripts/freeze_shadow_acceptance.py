#!/usr/bin/env python3
"""Freeze a passed no-action experiment's reports, sources and binary identities."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    names = {
        'live': 'outputs/live_hostfake_atomic60.summary.json',
        'oracle': 'outputs/oracle_atomic60/summary.json',
        'output': 'outputs/oracle_atomic60/output_audit.json',
        'provenance': 'outputs/atomic60_capture_audit.json',
    }
    data = {k: json.loads((ROOT/v).read_text()) for k,v in names.items()}
    live, oracle, output, audit = (data[k] for k in names)
    assert live['failure'] is None and live['action_sent'] is False
    assert oracle['passed'] and output['passed']
    assert oracle['raw_action']['max_abs'] <= 1e-4
    assert oracle['raw_action']['coordinates_exceeding_1e4'] == 0
    assert oracle['input_sha256'] == audit['capture_sha256']
    assert len({x['frames'] for x in (live, oracle, output, audit)}) == 1
    assert audit['pico']['exact_after_declared_conversion']
    assert not audit['pico']['body_mismatch_capture_indices']
    assert not audit['pico']['unknown_source_timestamps']
    assert audit['no_action_trace']['all_action_sent_explicit_false']
    assert audit['no_action_trace']['all_guards_allow']
    assert live['inert_serialized_frames'] == live['frames']
    assert live['compute_ms']['over20ms'] == 0
    assert live['actual_cadence']['intervals_over21ms'] == 0
    sources = [p for p in (ROOT/'src').rglob('*')
               if p.is_file() and p.suffix in ('.py','.cpp','.h','.hpp','.json','.yaml','.yml')]
    sources += [p for p in (ROOT/'scripts').glob('*') if p.suffix in ('.py','.sh')]
    sources += [ROOT/'packages/XRoboToolkit-PC-Service-Pybind/bindings'/name
                for name in ('py_bindings.cpp','body_pose_snapshot.h')]
    extra = [
        'artifacts/preprocessing_native_arm_v2.so',
        'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so',
        'build/pico/libdouble-conversion.so.3',
        'artifacts/policy_contracts.json','artifacts/preprocessing_skeleton.npz',
        'artifacts/output_envelopes.json','artifacts/output_envelopes.motion.urdf',
        'artifacts/output_envelopes.interface.json','outputs/model_runtime_manifest.json',
        'IMPLEMENTATION_RESULTS.md','REPRODUCE.md',
    ]
    records = {}
    for path in sorted(set(sources + [ROOT/x for x in extra] + [ROOT/x for x in names.values()])):
        records[str(path.relative_to(ROOT))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    result = {
        'schema':'vita_no_action_acceptance_v1', 'date':'2026-09-29',
        'accepted_scope':'60-second onboard shadow computation with real LowState, Codex-host fake Pico and scripted controller; no actuator transmission',
        'passed':True,'action_sent':False,'frames':live['frames'],
        'criterion':'Every evaluated raw-action coordinate absolute error <=1e-4 versus original FP32 policies with independent recurrent/previous-action state',
        'max_raw_action_error':oracle['raw_action']['max_abs'],
        'compute_ms':live['compute_ms'],'callback_to_output_ms':live['callback_to_output_ms'],
        'actual_cadence':live['actual_cadence'],
        'settings':{'policy_hz':50,'velocity_backend':'ort_fp32_cpu',
            'motion_backend':'fp32_ort_neon_router_bpu_moe','sched_rr_priority':1,
            'startup_warmup_per_policy':20,'cyclic_gc_disabled_only_bounded_loop':True,
            'preprocess_stages':'pico,features,gather,build','pico_snapshot':'atomic_same_packet_root_timestamp'},
        'board_verified_native_binaries':{
            'build/libvita_backend.so':'484041506b0d5a9e0e5f7cc2ecaac9f53cf47491c62c398ea44f718cb5c4ec83',
            'build/state/libvita_state.so':'0cbb0e14c9f7961a9ed110a897dc9745bb7cb4b7241f82adb4a7885598b04fe0'},
        'model_artifact_integrity':'Runtime verifies every manifest-listed artifact and expert-pool loaded byte by SHA256 at startup',
        'reports':names,'files':records,
        'limitations':['Finite replay and bounded timing coverage; not a universal numerical or hard-real-time guarantee',
            'Fake Pico transport uses SSH byte relay; no calibrated physical sensor-to-actuator latency',
            'No physical controller/headset/robot-dynamics test; actual ownership and actuation entrypoint remain a later phase',
            'No-action evidence includes code and capture fields, not independent network packet capture'],
    }
    (ROOT/'outputs/ACCEPTANCE_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','frames','max_raw_action_error','action_sent')},indent=2))
if __name__=='__main__':main()
