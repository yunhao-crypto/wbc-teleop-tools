#!/usr/bin/env python3
"""Build a self-contained S100 base or planner offline package from this checkout."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from bootstrap_actor_s100 import DELIVERY, POLICY_CONFIGS, SCHEMA, TARGET, inside, sha, verify

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = [
    dict(path='deploy/actor_s100/runtime/pico_service_arm64.tar.gz', role='pico_service',
         destination='vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice',
         required_members=['RoboticsServiceProcess', 'SDK/arm64/libPXREARobotSDK.so',
                           'libPXREAGRPCServer.so', 'plugins/platforms/libqoffscreen.so', 'epDBBackup.s3db']),
    dict(path='deploy/actor_s100/runtime/pico_binding_cp310_aarch64.tar.gz', role='pico_binding',
         destination='build/pico', required_members=['xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so',
                                                    'libdouble-conversion.so.3']),
]
SCRIPTS = ('actor_pico.py', 'bootstrap_actor_s100.py', 'configure_actor_s100.py',
           'start_actor_pico.sh', 'start_actor_motion.sh', 'prepare_zenoh_session.py',
           'pico_session_services.py', 'pico_camera_bridge.py', 'build_pico_arm64.sh',
           'prepare_external_ankle_config.py', 'import_actor_motion.py')
MODULES = ('__init__.py', 'policy_contract.py', 'state_adapter.py', 'sdkmain_adapter.py',
           'sdk021_guard.py', 'pico_input.py', 'pico_camera_bridge.py')
PLATFORMS = ('manylinux_2_35_aarch64', 'manylinux_2_28_aarch64', 'manylinux_2_17_aarch64',
             'manylinux2014_aarch64', 'linux_aarch64')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def source_revision(root, allow_dirty=False):
    source = dict(commit=git(root, 'rev-parse', 'HEAD'),
                  dirty=bool(git(root, 'status', '--porcelain', '--untracked-files=normal')))
    if source['dirty'] and not allow_dirty:
        raise RuntimeError('Commit source changes before release; use --allow-dirty for a development package')
    return source


def extract_bundle(archive, destination):
    """Extract only regular, unique files under one package directory."""
    destination = destination.resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen, roots = set(), set()
        for member in members:
            inside(destination, member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError('Package contains a link, special file or duplicate: '+member.name)
            seen.add(member.name)
            roots.add(Path(member.name).parts[0])
        if len(roots) != 1:
            raise ValueError('Package must contain one root directory')
        for member in members:
            target = inside(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output)
            target.chmod(member.mode & 0o777)
    return destination/roots.pop()


def runtime_inputs(root, variant):
    """Explicit runtime boundary; no tests, machine-local files or developer docs."""
    names = {str(DELIVERY/'requirements.txt'), 'config/sdk021-wheel-guard.json',
             'config/actor_motion.json', 'config/pico_op3_gripper.example.json'}
    names.update('scripts/'+name for name in SCRIPTS)
    module = 'src/deployment/vita_s100_aorta_30dof/'
    names.update(module+name for name in MODULES)
    names.update(str(p.relative_to(root)) for p in (root/module).glob('actor_*.py')
                 if p.name != 'actor_pico_session_replay.py')
    human = 'src/deployment/unitree_g1_ros2_29dof/src/humanoid_policy/'
    names.update(human+name for name in ('__init__.py', 'local_smpl_source.py', 'pico_body_pose_reader.py'))
    for directory in ('third_party/motion_stream_bridge', 'third_party/pico_binding',
                      'third_party/double-conversion', 'config/gmr_calibration'):
        names.update(str(p.relative_to(root)) for p in (root/directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.xml', '.h', '.hpp', '.cc', '.cpp')
                     and '__pycache__' not in p.parts)
    names.update(('third_party/motion_stream_bridge/GMR-LICENSE',
                  'third_party/pico_binding/LICENSE', 'third_party/double-conversion/LICENSE'))
    names.update(item['path'] for item in ARCHIVES)
    external = []
    for policy, name in POLICY_CONFIGS.items():
        names.add(name)
        config = json.loads((root/name).read_text())
        external.append(dict(config['artifacts']['model'], role=policy+'_actor'))
        for key, item in config['artifacts'].items():
            if key == 'model':
                continue
            path = inside(root, item['path'])
            if sha(path) != item['sha256']:
                raise RuntimeError('Actor artifact differs from config: '+item['path'])
            names.add(item['path'])
    if variant == 'planner':
        names.update(('config/actor_sonic_planner.json', str(DELIVERY/'requirements-planner.txt')))
        config = json.loads((root/'config/actor_sonic_planner.json').read_text())
        external.extend(dict(config[key], role=key) for key in ('planner', 'bridge'))
    return sorted(names), external


def resolve_wheels(root, requirements, destination):
    """Use pip's complete target resolver, with no indexes or dependency bypass."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1')
    command = [sys.executable, '-m', 'pip', 'download', '--no-index', '--no-cache-dir',
               '--find-links', str(root/DELIVERY/'wheelhouse'), '--dest', str(destination),
               '--only-binary=:all:', '--python-version', '310', '--abi', 'cp310', '--implementation', 'cp']
    for platform in PLATFORMS:
        command.extend(('--platform', platform))
    result = subprocess.run(command+['-r', str(root/requirements)], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Offline ARM64 wheel resolution failed:\n'+result.stdout)
    # pip may use a find-links location: require the bytes to match the tracked input.
    for wheel in destination.glob('*.whl'):
        if sha(wheel) != sha(root/DELIVERY/'wheelhouse'/wheel.name):
            raise RuntimeError('Resolved wheel differs from source: '+wheel.name)


def build(root, output, variant='base', *, allow_dirty=False, source=None):
    root, output = root.resolve(), output.resolve()
    if variant not in ('base', 'planner'):
        raise ValueError('Unknown delivery variant: '+variant)
    # The verifier passes provenance when building its isolated source export.
    source = source_revision(root, allow_dirty) if source is None else source
    commit, dirty = source['commit'], source['dirty']
    names, external = runtime_inputs(root, variant)
    if not allow_dirty and not set(names) <= set(git(root, 'ls-files').splitlines()):
        raise RuntimeError('Untracked runtime inputs cannot enter a release package')
    requirements = str(DELIVERY/('requirements-planner.txt' if variant == 'planner' else 'requirements.txt'))
    with tempfile.TemporaryDirectory(prefix='actor-package-') as temporary:
        stage = Path(temporary)/'runtime'
        stage.mkdir()
        for name in names:
            source = inside(root, name)
            if (root/name).is_symlink() or not source.is_file():
                raise ValueError('Runtime inputs must be regular files: '+name)
            destination = stage/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
        wheelhouse = stage/DELIVERY/'wheelhouse'
        wheelhouse.mkdir()
        resolve_wheels(root, stage/requirements, wheelhouse)
        guard = json.loads((stage/'config/sdk021-wheel-guard.json').read_text())['wheel']
        if sha(wheelhouse/guard['filename']) != guard['sha256']:
            raise RuntimeError('SDK wheel differs from reviewed guard')
        files = [dict(path=str(p.relative_to(stage)), bytes=p.stat().st_size,
                      sha256=sha(p), mode=oct(p.stat().st_mode & 0o777))
                 for p in sorted(stage.rglob('*')) if p.is_file()]
        payload = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        manifest = dict(schema=SCHEMA, variant=variant, source=dict(commit=commit, dirty=dirty),
                        target=TARGET, payload_sha256=payload, requirements=requirements,
                        archives=ARCHIVES, external_resources=external, files=files)
        (stage/DELIVERY/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        verify(stage)
        name = f'actor-s100-{variant}-{commit[:12]}-{payload[:12]}'+('-dirty' if dirty else '')
        archive = Path(temporary)/(name+'.tar.gz')
        # Stable headers and gzip timestamp make identical inputs produce identical archives.
        with archive.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for p in sorted(stage.rglob('*')):
                    if not p.is_file():
                        continue
                    info = tar.gettarinfo(str(p), arcname=name+'/'+str(p.relative_to(stage)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    with p.open('rb') as data:
                        tar.addfile(info, data)
        output.mkdir(parents=True, exist_ok=True)
        destination = output/archive.name
        digest = sha(archive)
        if destination.exists() and sha(destination) != digest:
            raise RuntimeError('Preserving different existing package: '+str(destination))
        # Copy across filesystems before the atomic rename.
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
            pending = Path(stream.name)
            try:
                with archive.open('rb') as data:
                    shutil.copyfileobj(data, stream)
                stream.close()
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)
        destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
    return dict(archive=str(destination), sha256=digest, variant=variant, source=manifest['source'],
                payload_sha256=payload, files=len(files),
                offline_wheels=sum(f['path'].endswith('.whl') for f in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/actor_delivery')
    parser.add_argument('--variant', choices=('base', 'planner'), default='base')
    parser.add_argument('--allow-dirty', action='store_true', help='Mark an uncommitted development package explicitly')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output_dir, args.variant, allow_dirty=args.allow_dirty), indent=2))


if __name__ == '__main__':
    main()
