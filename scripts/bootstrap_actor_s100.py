#!/usr/bin/env python3
"""Offline, hash-verified actor/Pico installation from an extracted S100 package.

Only --verify-only and --unpack-only are architecture independent. Installation
requires Linux ARM64/Python 3.10. No robot node, control lease or publisher is
created. The final model check uses a synthetic hold reference only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = Path('deploy/actor_s100')
SCHEMA = 'vita_actor_s100_delivery_v2'
TARGET = dict(os='Linux', architecture='aarch64', python='3.10', glibc_min='2.35')
# Mirrors actor_pico_runtime.POLICY_CONFIGS; bootstrap runs before the venv exists.
POLICY_CONFIGS = {'sonic': 'config/actor_pico.json', 'supermimic': 'config/actor_pico_supermimic.json',
                  'morphmimic': 'config/actor_pico_morphmimic.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Invalid delivery path: '+str(relative))
    target = (root/relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Delivery path escapes destination: '+str(relative))
    return target


def verify(root):
    try:
        manifest = json.loads((root/DELIVERY/'manifest.json').read_text())
    except FileNotFoundError as error:
        raise RuntimeError('No built delivery manifest. Build with scripts/build_actor_delivery.py, '
                           'then extract the package and run its bootstrap.') from error
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unknown actor delivery schema')
    seen = set()
    for item in manifest['files']:
        name = item['path']
        if name in seen:
            raise ValueError('Duplicate delivery entry: '+name)
        seen.add(name)
        p = inside(root, name)
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise RuntimeError('Missing or changed delivery input: '+name)
        if 'mode' in item and p.stat().st_mode & 0o777 != int(item['mode'], 8):
            raise RuntimeError('Changed delivery file mode: '+name)
    # Validate member paths/types before extracting any file, on either host.
    for archive in manifest['archives']:
        if archive['path'] not in seen:
            raise RuntimeError('Runtime archive is absent from sealed manifest: '+archive['path'])
        inside(root, archive['destination'])
        with tarfile.open(inside(root, archive['path']), 'r:gz') as tar:
            members = set()
            for member in tar:
                inside(Path('/delivery-validation'), member.name)
                if not member.isfile() or member.name in members:
                    raise ValueError('Archive contains a link, special file or duplicate: '+member.name)
                members.add(member.name)
            if not set(archive['required_members']) <= members:
                raise RuntimeError('Incomplete runtime archive: '+archive['path'])
    if manifest['target'] != TARGET:
        raise ValueError('Delivery target must be S100 Linux aarch64 / Python 3.10 / glibc 2.35+')
    source = manifest['source']
    if (not isinstance(source['commit'], str) or len(source['commit']) != 40
            or any(c not in '0123456789abcdef' for c in source['commit'])
            or type(source['dirty']) is not bool):
        raise ValueError('Invalid delivery source identity')
    requirements = str(DELIVERY/('requirements-planner.txt' if manifest['variant'] == 'planner' else 'requirements.txt'))
    if (manifest['variant'] not in ('base', 'planner') or manifest['requirements'] != requirements
            or requirements not in seen or str(DELIVERY/'requirements.txt') not in seen):
        raise RuntimeError('Invalid or unsealed package requirements')
    payload = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if payload != manifest['payload_sha256']:
        raise RuntimeError('Delivery payload identity mismatch')
    return manifest


def unpack(root, manifest, *, use_prebuilt=True):
    for archive in manifest['archives']:
        if archive['role'] == 'pico_binding' and not use_prebuilt:
            continue
        destination = inside(root, archive['destination'])
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(root/archive['path'], 'r:gz') as tar:
            for member in tar:
                target = inside(destination, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Saved service settings are device-local after first install.
                if member.name in ('setting.ini', 'epDBBackup.s3db') and target.exists():
                    continue
                with tar.extractfile(member) as stream:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                        temporary = Path(output.name)
                        try:
                            shutil.copyfileobj(stream, output, 1024*1024)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                try:
                    if target.exists():
                        if not target.is_file() or sha(target) != sha(temporary):
                            raise RuntimeError('Preserving different existing runtime file: '+str(target))
                    else:
                        temporary.replace(target)
                    target.chmod(member.mode & 0o777)
                finally:
                    temporary.unlink(missing_ok=True)


def configured_model(root, policy='sonic'):
    return json.loads((root/POLICY_CONFIGS[policy]).read_text())['artifacts']['model']


def provision_external(root, manifest, configured, source=None, *, label='model', config_name='actor configuration'):
    """Validate an explicit external asset before atomically placing it locally."""
    matches = [item for item in manifest['external_resources'] if item['path'] == configured['path']]
    if (len(matches) != 1 or matches[0]['sha256'] != configured['sha256']
            or matches[0]['bytes'] != configured['bytes']):
        raise RuntimeError('External model manifest disagrees with '+config_name)
    item, = matches
    destination = inside(root, item['path'])
    source = Path(source).resolve() if source else destination
    if not source.is_file():
        raise RuntimeError('Model is not in Git. Supply --'+label+' /path/to/model.onnx')
    if source.stat().st_size != item['bytes'] or sha(source) != item['sha256']:
        raise RuntimeError('External model size/SHA256 does not match '+config_name)
    if destination.exists():
        if not destination.is_file() or sha(destination) != item['sha256']:
            raise RuntimeError('Preserving different existing model: '+str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, output, 1024*1024)
            output.close()
            if sha(temporary) != item['sha256']:
                raise RuntimeError('External model changed during copy')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def provision_model(root, manifest, source=None, policy='sonic'):
    return provision_external(root, manifest, configured_model(root, policy), source,
                              label='model', config_name=POLICY_CONFIGS[policy])


def provision_planner_assets(root, manifest, planner=None, bridge=None):
    if planner is None and bridge is None:
        return
    if manifest.get('variant') != 'planner':
        raise RuntimeError('Planner assets require a planner delivery package')
    config_name = 'config/actor_sonic_planner.json'
    settings = json.loads((root/config_name).read_text())
    for key, source in (('planner', planner), ('bridge', bridge)):
        if source is not None:
            provision_external(root, manifest, settings[key], source,
                               label=key+'-model', config_name=config_name)


def environment(root):
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env['MUJOCO_GL'] = 'disable'
    for name in list(env):
        if name.startswith('PIP_'):
            env.pop(name)
    sdk = root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    libs = [root/'venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs',
            root/'build/pico', sdk, sdk/'lib', sdk/'SDK/arm64']
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, libs))+':'+env.get('LD_LIBRARY_PATH', '')
    env['PYTHONPATH'] = ':'.join(map(str, [root/'src', root/'build/pico',
        root/'src/deployment/unitree_g1_ros2_29dof/src']))
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('HOLOMOTION_REAL_ROBOT', None)
    return env


def install(root, manifest, *, build_pico=False, policy='sonic'):
    if sys.platform != 'linux' or platform.machine() != 'aarch64' or sys.version_info[:2] != (3, 10):
        raise RuntimeError('Install on S100 Linux aarch64 with Python 3.10; use --verify-only on x86')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(int(v) for v in version.split('.')[:2]) < (2, 35):
        raise RuntimeError('The pinned SDK wheel requires glibc >= 2.35')
    venv = root/'venv'
    marker = venv/'.actor_bootstrap_owned'
    if venv.exists() and not marker.is_file():
        raise RuntimeError('Existing venv is preserved. Install into a fresh checkout/directory instead.')
    if not venv.exists():
        # Fail early if the OS venv/ensurepip package is unavailable.
        import ensurepip
        import venv as venv_module
        venv_module.EnvBuilder(with_pip=True).create(venv)
        marker.write_text('Created by bootstrap_actor_s100.py; safe to resume installation.\n')
    python = venv/'bin/python'
    env = environment(root)
    # Avoid the source snapshot's old .pth or caller Python paths during pip.
    pip_env = env.copy()
    pip_env.pop('PYTHONPATH', None)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
                    '--find-links', str(root/DELIVERY/'wheelhouse'),
                    '-r', str(root/manifest['requirements'])], env=pip_env, check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], env=pip_env, check=True)
    unpack(root, manifest, use_prebuilt=not build_pico)
    if build_pico:
        env['VITA_PYTHON'] = str(python)
        subprocess.run(['bash', str(root/'scripts/build_pico_arm64.sh')], env=env, check=True)
    # Validate actual installed SDK bytes/ABI and binding API without calling init().
    code = ('from deployment.vita_s100_aorta_30dof.sdk021_guard import verify_vendor_sources; '
            'verify_vendor_sources(); import aorta, flatbuffers; '
            'from locomotion_aorta import _body_cmd_native; import xrobotoolkit_sdk as x; '
            'assert callable(x.get_body_pose_snapshot); assert callable(x.get_controller_trigger_snapshot); '
            'assert callable(x.get_controller_teleop_snapshot); ')
    if manifest['variant'] == 'planner':
        code += 'import onnxruntime; '
    code += 'print("SDK and Pico binding verification passed")'
    subprocess.run([str(python), '-c', code], cwd=root, env=env, check=True)
    service = root/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice/RoboticsServiceProcess'
    for executable in [service, root/'build/pico'/f'xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so']:
        result = subprocess.run(['ldd', str(executable)], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode or 'not found' in result.stdout:
            raise RuntimeError('Target OS shared-library dependency check failed for '+str(executable)+'\n'+result.stdout)
    subprocess.run([str(python), str(root/'scripts/actor_pico.py'), 'check', '--steps', '10',
                    '--policy', policy], cwd=root, env=env, check=True)
    receipt = dict(status='PASS_S100_OFFLINE_INSTALL_AND_MODEL_CHECK',
                   source=manifest['source'], payload_sha256=manifest['payload_sha256'],
                   variant=manifest['variant'], build_pico=build_pico,
                   binding_sha256=sha(root/'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so'),
                   policy=policy, model_sha256=sha(root/configured_model(root, policy)['path']),
                   python=str(python), pip_check='PASS', real_binding_import='PASS',
                   model_check_steps=10, actions_sent=0, control_handles_created=0,
                   hardware_inputs_checked=False, robot_motion='not_measured')
    (root/'build/actor_s100_install.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--unpack-only', action='store_true')
    parser.add_argument('--build-pico', action='store_true', help='Rebuild from tracked C++ sources on S100')
    parser.add_argument('--model', type=Path, help='External model of --policy; verified before copying locally')
    parser.add_argument('--planner-model', type=Path, help='External SONIC planner ONNX; verified before copying locally')
    parser.add_argument('--bridge-model', type=Path, help='External map bridge ONNX; verified before copying locally')
    parser.add_argument('--policy', choices=tuple(POLICY_CONFIGS), default='sonic',
                        help='Actor whose pinned model --model supplies and the install check runs')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify(root)
    if manifest['variant'] != 'planner' and (args.planner_model or args.bridge_model):
        parser.error('Planner assets require a planner delivery package')
    if args.verify_only:
        if args.model or args.planner_model or args.bridge_model:
            parser.error('External models require installation or --unpack-only')
        result = dict(status='PASS_DELIVERY_FILES', files=len(manifest['files']),
                      external_model='separate asset; not checked by --verify-only',
                      s100_execution='not_measured', actions_sent=0)
    elif args.unpack_only:
        if args.model:
            provision_model(root, manifest, args.model, args.policy)
        provision_planner_assets(root, manifest, args.planner_model, args.bridge_model)
        unpack(root, manifest)
        result = dict(status='PASS_RUNTIME_UNPACK', s100_execution='not_measured', actions_sent=0)
    else:
        import fcntl
        (root/'build').mkdir(exist_ok=True)
        with (root/'build/operator.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another operator is using this deployment; installation is blocked') from exc
            provision_model(root, manifest, args.model, args.policy)
            provision_planner_assets(root, manifest, args.planner_model, args.bridge_model)
            result = install(root, manifest, build_pico=args.build_pico, policy=args.policy)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
