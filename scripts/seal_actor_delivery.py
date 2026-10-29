#!/usr/bin/env python3
"""Seal staged delivery file identities before committing; no robot operations."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = ROOT
    actor_path = root/'ACTOR-PICO-MANIFEST.json'
    actor = json.loads(actor_path.read_text())
    configs = {'sonic': 'config/actor_pico.json', 'supermimic': 'config/actor_pico_supermimic.json'}
    external = []
    for policy, config in configs.items():
        model = json.loads((root/config).read_text())['artifacts']['model']
        external.append(dict(path=model['path'], sha256=model['sha256'], bytes=model['bytes'],
                             provision=f'Supply separately using bootstrap_actor_s100.py --policy {policy} --model'))
    models = {item['path'] for item in external}
    actor['files'] = [item for item in actor['files'] if item['path'] not in models]
    actor['external_resources'] = external
    actor['scope'] = 'inference checks and explicitly armed supported EXTERNAL trials; hardware not_measured'
    for name in ('scripts/replay_actor_candidate.py', 'scripts/validate_actor_stand_mujoco.py',
                 'src/deployment/vita_s100_aorta_30dof/actor_pico_live.py',
                 'src/deployment/vita_s100_aorta_30dof/actor_pico_local.py',
                 'src/deployment/vita_s100_aorta_30dof/actor_pico_startup.py',
                 'src/deployment/vita_s100_aorta_30dof/actor_pico_worker.py', 'tests/test_actor_pico_live.py'):
        if not any(item['path'] == name for item in actor['files']):
            actor['files'].append(dict(path=name))
    for name in ('src/deployment/vita_s100_aorta_30dof/actor_pico_buttons.py',
                 'third_party/pico_binding/bindings/controller_trigger_snapshot.h'):
        if not any(item['path'] == name for item in actor['files']):
            actor['files'].append(dict(path=name))
    additional = ['scripts/configure_actor_s100.py', 'tests/test_actor_local_provision.py',
                  'src/deployment/vita_s100_aorta_30dof/actor_pico_scheduling.py',
                  'src/deployment/vita_s100_aorta_30dof/actor_pico_gmr.py',
                  'tests/fixtures/actor_pico/bridge_gmr_golden.npz',
                  'tests/fixtures/actor_pico/bridge_gmr_golden.json',
                  'config/actor_pico_supermimic.json',
                  'src/deployment/vita_s100_aorta_30dof/actor_supermimic.py',
                  'tests/test_actor_supermimic.py', 'tests/fixtures/actor_pico/supermimic_fixture.npz',
                  'src/deployment/vita_s100_aorta_30dof/actor_pico_stream.py',
                  'src/deployment/vita_s100_aorta_30dof/actor_pico_session_replay.py',
                  'scripts/freeze_pico_session.py', 'scripts/replay_pico_session.py', 'scripts/bench_pico_gmr.py',
                  'tests/test_actor_pico_session.py',
                  'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz']
    additional += [str(p.relative_to(root)) for p in (root/'third_party/motion_stream_bridge').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts]
    for name in additional:
        if not any(item['path'] == name for item in actor['files']):
            actor['files'].append(dict(path=name))
    for item in actor['files']:
        p = root/item['path']
        item.update(sha256=sha(p), bytes=p.stat().st_size, mode=oct(p.stat().st_mode & 0o777))
    actor_path.write_text(json.dumps(actor, indent=2)+'\n')
    output = 'deploy/actor_s100/manifest.json'
    names = sorted(n for n in subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0') if n and n != output)
    if models & set(names):
        raise RuntimeError('Remove the external models from the Git index before sealing')
    required = ['artifacts/output_envelopes.motion.urdf',
                'config/actor_pico.json', 'config/actor_pico_supermimic.json', 'config/sdk021-wheel-guard.json',
                'scripts/bootstrap_actor_s100.py', 'scripts/configure_actor_s100.py', 'scripts/start_actor_pico.sh',
                'src/deployment/vita_s100_aorta_30dof/actor_pico_live.py',
                'src/deployment/vita_s100_aorta_30dof/actor_pico_local.py',
                'src/deployment/vita_s100_aorta_30dof/actor_pico_startup.py',
                'src/deployment/vita_s100_aorta_30dof/actor_pico_worker.py',
                'src/deployment/vita_s100_aorta_30dof/actor_pico_buttons.py',
                'src/deployment/vita_s100_aorta_30dof/actor_supermimic.py',
                'src/deployment/vita_s100_aorta_30dof/actor_pico_stream.py',
                'tests/test_actor_pico_live.py',
                'third_party/pico_binding/bindings/py_bindings.cpp',
                'third_party/pico_binding/bindings/body_pose_snapshot.h',
                'third_party/pico_binding/bindings/controller_trigger_snapshot.h',
                'third_party/pico_binding/include/PXREARobotSDK.h',
                'third_party/double-conversion/double-conversion/double-to-string.cc',
                'tests/fixtures/actor_pico/pico_replay.npz', 'deploy/actor_s100/requirements.txt']
    required += [str(p.relative_to(root)) for p in sorted((root/'deploy/actor_s100/wheelhouse').glob('*.whl'))]
    archives = [dict(path='deploy/actor_s100/runtime/pico_service_arm64.tar.gz', role='pico_service',
                     destination='vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice',
                     required_members=['RoboticsServiceProcess', 'SDK/arm64/libPXREARobotSDK.so',
                                       'libPXREAGRPCServer.so', 'plugins/platforms/libqoffscreen.so',
                                       'epDBBackup.s3db']),
                dict(path='deploy/actor_s100/runtime/pico_binding_cp310_aarch64.tar.gz', role='pico_binding',
                     destination='build/pico', required_members=['xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so',
                                                                 'libdouble-conversion.so.3'])]
    required += [x['path'] for x in archives]
    missing = set(required)-set(names)
    if missing:
        raise RuntimeError('Stage these required delivery files first: '+', '.join(sorted(missing)))
    record = dict(schema='vita_actor_s100_delivery_v1', version='2026.10.29-actor21-supermimic-delivery',
                  target=dict(os='Linux', architecture='aarch64', python='3.10', glibc_min='2.35'),
                  scope='offline/shadow checks and explicitly armed supported EXTERNAL trials; hardware not_measured',
                  required_inputs=required, external_resources=external, archives=archives,
                  files=[dict(path=n, bytes=(root/n).stat().st_size, sha256=sha(root/n)) for n in names])
    (root/output).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(sealed_files=len(names), required_inputs=len(required), manifest=output)))


if __name__ == '__main__':
    main()
