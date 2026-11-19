#!/usr/bin/env python3
"""Pico-only asset and no-action checks for the original actor bootstrap.

Environment creation, pip, ABI checks and runtime unpacking stay in the original
bootstrap. This module adds no installer loop, lock or controller.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from bootstrap_actor_s100 import sha, verify as verify_actor, PICO_POLICIES

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs(root):
    root=Path(root).resolve()
    manifest=verify_actor(root)
    seen={item['path'] for item in manifest['files']}
    release = manifest.get('pico_inputs_release', {})
    if (manifest.get('variant')!='base' or release.get('isolated_sdk') is not True
            or release.get('unified_environment') is not False
            or release.get('allowed_policies')!=list(PICO_POLICIES)
            or release.get('schema')!='vita_pico_inputs_s100_physical_v1'
            or release.get('physical_workflow') is not True or release.get('motion_enabled') is not True
            or release.get('body_cmd_protocol_version')!=4 or release.get('primary_route')!='pico_smpl'
            or release.get('backend') != 'ort'):
        raise ValueError('Not a sealed, model-included Pico input delivery with an explicit action capability')
    if manifest.get('external_resources')!=[]:
        raise ValueError('Isolated Pico delivery must include its two models without external legacy policies')
    names = seen
    for config in ('config/actor_smpl.json', 'config/actor_threepoint.json'):
        settings = json.loads((root/config).read_text())
        for item in settings['artifacts'].values():
            path = root/item['path']
            if item['path'] not in names or sha(path) != item['sha256']:
                raise ValueError('Model/contract not sealed in delivery: '+item['path'])
    for key in ('input', 'triggers'):
        item = release['offline_check'][key]
        if item['path'] not in names or sha(root/item['path']) != item['sha256']:
            raise ValueError('Offline check data binding failed')
    for route in ('pico_smpl','pico_teleop'):
        settings=json.loads((root/f'config/{route}_live.json').read_text())
        for path,expected in settings['bindings'].items():
            if path not in seen or sha(root/path)!=expected:raise ValueError('Unsealed Pico runtime binding')
    for path in ('scripts/pico_actor_aligned.py','scripts/check_pico_aligned.py',
                 'src/deployment/vita_s100_aorta_30dof/pico_aligned_runtime.py'):
        if path not in seen:raise ValueError('Missing Pico entry: '+path)
    return manifest


def validate_environment(root,manifest,policy):
    """Reject legacy directories before any receipt or environment mutation."""
    release=manifest.get('pico_inputs_release',{})
    if (release.get('isolated_sdk') is not True or release.get('unified_environment') is not False
            or release.get('allowed_policies')!=list(PICO_POLICIES) or policy not in PICO_POLICIES):
        raise RuntimeError('Pico policies require an isolated R4 --pico-assets package')
    venv=Path(root)/'venv'
    if venv.exists() and not all((venv/name).is_file() for name in
            ('.actor_bootstrap_owned','.pico_bodycmd4_owned')):
        raise RuntimeError('Existing legacy or unowned venv is preserved; extract Pico into a new directory')


def require_target():
    if sys.platform != 'linux' or platform.machine() != 'aarch64' or sys.version_info[:2] != (3,10):
        raise RuntimeError('Install only on Linux aarch64 Python3.10; --verify-only is portable')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int,version.split('.')[:2])) < (2,35):
        raise RuntimeError('Requires glibc>=2.35')


def check_installation(root,manifest,python,env,policy):
    """Additional no-action checks after the original install's pip/ABI/ldd steps."""
    if verify_inputs(root)['payload_sha256']!=manifest['payload_sha256']:
        raise ValueError('Pico installation payload changed')
    receipt=dict(status='PASS_PICO_S100_OFFLINE_INSTALL',policy=policy,
        environment='isolated_pico_venv',source=manifest['source'],payload_sha256=manifest['payload_sha256'],
        pip_check='PASS',real_binding_import='PASS',actions_sent=0,control_handles_created=0,
        pico_service_started=False,pico_sdk_initialized=False,hardware_inputs_checked=False)
    subprocess.run([str(python),str(root/'scripts/verify_bodycmd4_codec.py')],cwd=root,env=env,check=True)
    receipt['bodycmd4_codec_check']=json.loads((root/'build/bodycmd4_codec_check.json').read_text())
    if receipt['bodycmd4_codec_check']['status']!='PASS_NATIVE_BODYCMD4_SERIALIZATION':
        raise RuntimeError('BodyCmd4 native serialization self-check failed')
    receipt['aligned_model_checks']={}
    for route in ('pico_smpl','pico_teleop'):
        receipt['aligned_model_checks'][route]={}
        for backend in ('ort','sdk'):
            output=root/'build'/('aligned_'+route+'_'+backend+'.json')
            subprocess.run([str(python),str(root/'scripts/check_pico_aligned.py'),'--route',route,
                '--backend',backend,'--out',str(output)],cwd=root,env=env,check=True)
            proof=json.loads(output.read_text())
            if (proof['status']!='PASS_ALIGNED_OFFLINE_CHECK' or proof['actions_sent']
                    or proof['control_handles_created']):raise RuntimeError('Pico no-action model check failed')
            receipt['aligned_model_checks'][route][backend]=proof
    return receipt


@contextmanager
def installation_transaction(root):
    """Only isolated Pico bundles invalidate stale live receipts during mutation."""
    receipts=[root/'build'/name for name in ('actor_s100_install.json','pico_inputs_s100_install.json')]
    for path in receipts:path.unlink(missing_ok=True)
    try:yield
    except BaseException:
        for path in receipts:path.unlink(missing_ok=True)
        raise


def write_install_receipt(root,receipt):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=root/'build',delete=False) as stream:
        pending=Path(stream.name)
        try:
            stream.write(json.dumps(receipt,indent=2)+'\n');stream.close()
            pending.replace(root/'build/actor_s100_install.json')
        finally:pending.unlink(missing_ok=True)


def main(argv=None):
    # Compatibility spelling only: one CLI transaction, lock and policy install.
    from bootstrap_actor_s100 import main as actor_main
    return actor_main(['--policy','pico_smpl',*(sys.argv[1:] if argv is None else argv)])


if __name__=='__main__':main()
