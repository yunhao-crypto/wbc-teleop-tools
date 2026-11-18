#!/usr/bin/env python3
"""Install this isolated no-action Pico input package; never start SDK/services.

The shipped original bootstrap supplies path/hash/archive validation and runtime
unpacking. This entry never invokes its G1 model installer or motion launcher.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import hashlib

from bootstrap_actor_s100 import unpack, environment, sha, inside, SCHEMA, TARGET

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'deploy/actor_s100/manifest.json').read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('target')!=TARGET or manifest.get('variant')!='base':
        raise ValueError('Invalid S100 target/schema')
    source=manifest['source']
    if (not isinstance(source.get('commit'),str) or len(source['commit'])!=40
            or any(c not in '0123456789abcdef' for c in source['commit']) or type(source.get('dirty')) is not bool):
        raise ValueError('Invalid delivery source identity')
    seen=set()
    for item in manifest['files']:
        name=item['path'];p=inside(root,name)
        if '\\' in name or name in seen:raise ValueError('Noncanonical or duplicate delivery path')
        seen.add(name)
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise RuntimeError('Missing or changed delivery input: '+name)
    if (manifest.get('requirements')!='deploy/actor_s100/requirements.txt' or manifest['requirements'] not in seen):
        raise ValueError('Unsealed delivery requirements')
    if hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()!=manifest['payload_sha256']:
        raise ValueError('Delivery payload identity mismatch')
    archives=manifest['archives']
    if len(archives)!=2 or {x['role'] for x in archives}!={'pico_service','pico_binding'}:
        raise ValueError('Expected the two sealed Pico runtime archives')
    for archive in archives:
        if archive['path'] not in seen:raise ValueError('Unsealed runtime archive')
        inside(root,archive['destination'])
        with tarfile.open(root/archive['path'],'r:gz') as tar:
            members=set()
            for member in tar:
                inside(root/'archive_validation',member.name)
                if not member.isfile() or member.name in members:
                    raise ValueError('Runtime archive has link/special/duplicate member')
                members.add(member.name)
            if not set(archive['required_members'])<=members:raise ValueError('Incomplete runtime archive')
    release = manifest.get('pico_inputs_release', {})
    physical=release.get('physical_workflow') is True
    release_schema='vita_pico_inputs_s100_physical_v1' if physical else 'vita_pico_inputs_s100_no_action_v1'
    if (release.get('schema') != release_schema
            or release.get('primary_route') != 'pico_smpl' or release.get('motion_enabled') is not physical
            or release.get('backend') != 'ort' or manifest.get('external_resources') != []):
        raise ValueError('Not a sealed, model-included Pico input delivery with an explicit action capability')
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
    workflow=release.get('workflow_check')
    if workflow is not None:
        for key in ('config','fixture'):
            item=workflow[key]
            if item['path'] not in seen or sha(root/item['path'])!=item['sha256']:
                raise ValueError('Unsealed workflow check')
        settings=json.loads((root/workflow['config']['path']).read_text())
        if settings.get('motion_enabled') is not False:
            raise ValueError('Workflow must retain the no-action gate')
        item=settings['standing_reference']
        if item['path'] not in seen or sha(root/item['path'])!=item['sha256']:
            raise ValueError('Unsealed SMPL standing reference')
        for path,expected in settings['unchanged_guards'].items():
            if path not in seen or sha(root/path)!=expected:raise ValueError('Original workflow guard changed')
        fixture=json.loads((root/workflow['fixture']['path']).read_text())
        for item in fixture['files'].values():
            if item['path'] not in seen or sha(root/item['path'])!=item['sha256']:
                raise ValueError('Unsealed workflow fixture file')
    if release.get('physical_workflow') is True:
        settings=json.loads((root/'config/pico_smpl_live.json').read_text())
        for path,expected in settings['bindings'].items():
            if path not in seen or sha(root/path)!=expected:raise ValueError('Unsealed physical SMPL workflow binding')
        for path in ('config/pico_smpl_live.json','scripts/pico_smpl_live.py','scripts/start_actor_pico_smpl_live.sh',
                     'src/deployment/vita_s100_aorta_30dof/pico_smpl_live.py'):
            if path not in seen:raise ValueError('Missing physical SMPL workflow entry')
    return manifest


def check_model(root, manifest, python, env):
    fixture = manifest['pico_inputs_release']['offline_check']
    output = root/'outputs'/('install_smpl_check_'+str(time.time_ns()))
    command = [str(python), str(root/'scripts/pico_smpl.py'), 'replay', '--backend', 'ort',
        '--feedback', 'synthetic-stand', '--wrist-mode', 'stand', '--seconds', '0', '--require-tracking',
        '--input', str(root/fixture['input']['path']), '--sha256', fixture['input']['sha256'],
        '--trigger-script', str(root/fixture['triggers']['path']), '--out', str(output)]
    subprocess.run(command, cwd=root, env=env, check=True)
    receipt = json.loads((output/'report.json').read_text())
    if (receipt['status'] != 'PASS_NO_ACTION_INFERENCE' or receipt['inference_steps'] <= 0
            or not receipt['tracking_verified'] or receipt['actions_sent'] != 0
            or receipt['control_handles_created'] != 0 or receipt['pico_service'] is not None):
        raise RuntimeError('Offline SMPL check did not pass with zero service/control activity')
    return receipt, output


def require_target():
    if sys.platform != 'linux' or platform.machine() != 'aarch64' or sys.version_info[:2] != (3,10):
        raise RuntimeError('Install only on Linux aarch64 Python3.10; --verify-only is portable')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int,version.split('.')[:2])) < (2,35):
        raise RuntimeError('Requires glibc>=2.35')


def install(root, manifest):
    require_target()
    venv = root/'venv'; marker = venv/'.pico_inputs_bootstrap_owned'
    if venv.exists() and not marker.is_file():
        raise RuntimeError('Existing unowned venv is preserved; use a new deployment directory')
    if not venv.exists():
        import ensurepip
        import venv as venv_module
        venv_module.EnvBuilder(with_pip=True).create(venv)
        marker.write_text('Created by bootstrap_pico_inputs_s100.py; isolated no-action delivery\n')
    python = venv/'bin/python'; env = environment(root)
    pip_env = env.copy(); pip_env.pop('PYTHONPATH',None)
    subprocess.run([str(python),'-m','pip','install','--no-index','--no-cache-dir','--find-links',
        str(root/'deploy/actor_s100/wheelhouse'),'-r',str(root/manifest['requirements'])],env=pip_env,check=True)
    subprocess.run([str(python),'-m','pip','check'],env=pip_env,check=True)
    unpack(root, manifest)
    # Import ABI/API only: no x.init(), PullStateSource, robot controller or
    # vendor service process is created during installation.
    code = ('from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources; '
        'verify_vendor_sources(); import aorta,flatbuffers,onnxruntime; '
        'from locomotion_aorta import _body_cmd_native; import xrobotoolkit_sdk as x; '
        'assert all(callable(getattr(x,n,None)) for n in '
        '("init","close","get_body_pose_snapshot","get_controller_trigger_snapshot")); '
        'print("PASS SDK/ORT/Pico import only; no SDK init or service start")')
    subprocess.run([str(python),'-c',code],cwd=root,env=env,check=True)
    for p in (root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice/RoboticsServiceProcess',
              root/'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so'):
        result = subprocess.run(['ldd',str(p)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode or 'not found' in result.stdout:
            raise RuntimeError('Missing target shared-library dependency: '+str(p)+'\n'+result.stdout)
    checked, output = check_model(root,manifest,python,env)
    workflow_report=None;sdk_workflow_report=None;sdk_workflow_output=None
    if manifest['pico_inputs_release'].get('workflow_check') is not None:
        workflow_output=root/'outputs'/('install_smpl_workflow_check_'+str(time.time_ns()))
        subprocess.run([str(python),str(root/'scripts/pico_smpl_workflow.py'),'check','--out',str(workflow_output)],
            cwd=root,env=env,check=True)
        workflow_report=json.loads((workflow_output/'report.json').read_text())
        if (workflow_report['status']!='PASS_SMPL_NO_ACTION_WORKFLOW_REPLAY'
                or workflow_report['actions_sent'] or workflow_report['control_handles_created']
                or workflow_report['lifecycle_stage']!='CLOSED' or workflow_report['tracking_steps']<=0):
            raise RuntimeError('SMPL workflow installation check failed')
        if manifest['pico_inputs_release'].get('physical_workflow') is True:
            sdk_workflow_output=root/'outputs'/('install_sdk_smpl_workflow_check_'+str(time.time_ns()))
            subprocess.run([str(python),str(root/'scripts/pico_smpl_workflow.py'),'check','--backend','sdk',
                '--out',str(sdk_workflow_output)],cwd=root,env=env,check=True)
            sdk_workflow_report=json.loads((sdk_workflow_output/'report.json').read_text())
            if (sdk_workflow_report['status']!='PASS_SMPL_NO_ACTION_WORKFLOW_REPLAY'
                    or sdk_workflow_report['actions_sent'] or sdk_workflow_report['control_handles_created']
                    or sdk_workflow_report['lifecycle_stage']!='CLOSED' or sdk_workflow_report['tracking_steps']<=0):
                raise RuntimeError('Native SDK inference workflow check failed before physical deployment')
    receipt = dict(status='PASS_PICO_SMPL_S100_OFFLINE_INSTALL', route='pico_smpl', backend='ort',
        model_sha256=sha(root/'artifacts/actor_smpl/model_step_027000_smpl.onnx'),
        payload_sha256=manifest['payload_sha256'], pip_check='PASS', real_binding_import='PASS',
        model_check_steps=checked['inference_steps'], model_check_report=str(output/'report.json'),
        pico_service_started=False, pico_sdk_initialized=False, hardware_inputs_checked=False,
        actions_sent=0,control_handles_created=0,robot_configuration_modified=False,protection_relaxed=False)
    if manifest['pico_inputs_release'].get('body_cmd_protocol_version')==4:
        subprocess.run([str(python),str(root/'scripts/verify_bodycmd4_codec.py')],cwd=root,env=env,check=True)
        receipt['bodycmd4_codec_check']=json.loads((root/'build/bodycmd4_codec_check.json').read_text())
        if receipt['bodycmd4_codec_check']['status']!='PASS_NATIVE_BODYCMD4_SERIALIZATION':
            raise RuntimeError('BodyCmd4 native serialization self-check did not pass')
    if workflow_report is not None:
        receipt['workflow_check']=dict(status=workflow_report['status'],inference_steps=workflow_report['inference_steps'],
            tracking_steps=workflow_report['tracking_steps'],report=str(workflow_output/'report.json'),
            lifecycle_backend=workflow_report['lifecycle_backend'],hardware_startup_exit_validated=False)
    if sdk_workflow_report is not None:
        receipt['sdk_workflow_check']=dict(status=sdk_workflow_report['status'],
            inference_steps=sdk_workflow_report['inference_steps'],tracking_steps=sdk_workflow_report['tracking_steps'],
            report=str(sdk_workflow_output/'report.json'),backend='sdk',actions_sent=0,control_handles_created=0)
    receipt['aligned_model_checks']={}
    for route in ('pico_smpl','pico_teleop'):
        receipt['aligned_model_checks'][route]={}
        for backend in ('ort','sdk'):
            check_output=root/'build'/('aligned_'+route+'_'+backend+'.json')
            subprocess.run([str(python),str(root/'scripts/check_pico_aligned.py'),'--route',route,
                '--backend',backend,'--out',str(check_output)],cwd=root,env=env,check=True)
            proof=json.loads(check_output.read_text())
            if proof['status']!='PASS_ALIGNED_OFFLINE_CHECK' or proof['actions_sent'] or proof['control_handles_created']:
                raise RuntimeError('Aligned original-encoder check failed')
            receipt['aligned_model_checks'][route][backend]=proof
    (root/'build/pico_inputs_s100_install.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only',action='store_true')
    modes.add_argument('--unpack-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve();manifest=verify_inputs(root)
    if args.verify_only:
        result=dict(status='PASS_PICO_INPUTS_DELIVERY_FILES',files=len(manifest['files']),
            models_included=True,s100_execution='not_measured',actions_sent=0,control_handles_created=0)
    elif args.unpack_only:
        unpack(root,manifest)
        result=dict(status='PASS_PICO_INPUTS_RUNTIME_UNPACK',s100_execution='not_measured',
            actions_sent=0,control_handles_created=0)
    else:
        require_target()
        import fcntl
        (root/'build').mkdir(exist_ok=True)
        with (root/'build/pico_inputs_install.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=install(root,manifest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
