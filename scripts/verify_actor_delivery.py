#!/usr/bin/env python3
"""Check the staged Git export without borrowing untracked deployment inputs.

--host-inference additionally needs host NumPy and ONNX Runtime. ARM wheels
are resolved offline, and only their SDK Python/schema payload is used on x86.
This is not ARM execution, live Pico reception, or robot control validation.
"""
import argparse
import json
import os
import re
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Check failed: '+str(command)+'\n'+result.stdout)
    return result.stdout


def run_tests(path, *, cwd, env):
    output = run([sys.executable, path], cwd=cwd, env=env)
    match = re.search(r'Ran (\d+) tests? in ', output)
    if match is None or int(match[1]) < 1:
        raise RuntimeError('Missing unittest execution receipt: '+path)
    return int(match[1])


def verify_doc_links(root):
    links = 0
    for name in ['README.md', 'README_ACTOR_PICO.md', 'deploy/actor_s100/README.md']:
        document = root/name
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', document.read_text()):
            if '://' in target or target.startswith('mailto:'):
                continue
            path, _, anchor = unquote(target).partition('#')
            resolved = (document.parent/path).resolve() if path else document
            if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                raise RuntimeError('Broken delivery documentation link: '+name+' -> '+target)
            if anchor and resolved.suffix == '.md':
                headings = re.findall(r'^#+\s+(.+)$', resolved.read_text(), re.MULTILINE)
                anchors = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}
                if anchor not in anchors:
                    raise RuntimeError('Missing delivery documentation anchor: '+name+' -> '+target)
            links += 1
    return links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-inference', action='store_true')
    parser.add_argument('--model', type=Path, help='Explicit external actor model for host inference')
    parser.add_argument('--supermimic-model', type=Path,
                        help='Also verify the SuperMimic actor with this explicit external model')
    args = parser.parse_args()
    if args.host_inference and not args.model:
        parser.error('--host-inference requires --model /path/to/actor_g1.onnx (not stored in Git)')
    if args.supermimic_model and not args.host_inference:
        parser.error('--supermimic-model requires --host-inference')
    if args.model:
        args.model = args.model.resolve()
    if args.supermimic_model:
        args.supermimic_model = args.supermimic_model.resolve()
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('PIP_'):
            env.pop(name)
    for key in ['PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH',
                'VITA_PREPROCESS_LIBRARY',
                'VITA_PREPROCESS_NATIVE_STAGES', 'HOLOMOTION_REAL_ROBOT']:
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PIP_CONFIG_FILE=os.devnull)
    with tempfile.TemporaryDirectory(prefix='actor-git-export-') as temporary:
        export = Path(temporary)/'checkout'
        export.mkdir()
        run(['git', 'checkout-index', '--all', '--prefix', str(export)+'/'], cwd=ROOT, env=env)
        # Neither extracted vendor runtime nor Python environment comes from
        # the source working directory; verify before materializing anything.
        for name in ['venv', 'vendor_packages', 'model_dependencies_readonly', 'reference',
                     'artifacts/actor_g1/actor_g1.onnx', 'artifacts/supermimic/supermimic_vita02a.onnx']:
            if (export/name).exists():
                raise RuntimeError('Unexpected untracked input in Git export: '+name)
        run([sys.executable, 'scripts/bootstrap_actor_s100.py', '--verify-only'], cwd=export, env=env)
        documentation_links = verify_doc_links(export)
        wheelhouse = export/'deploy/actor_s100/wheelhouse'
        resolved = Path(temporary)/'resolved_wheels'
        # Resolve the full dependency graph for the target ABI, offline.
        run([sys.executable, '-m', 'pip', 'download', '--no-index',
             '--find-links', str(wheelhouse), '--dest', str(resolved),
             '--only-binary=:all:', '--platform', 'manylinux_2_35_aarch64', '--platform', 'manylinux_2_28_aarch64',
             '--platform', 'manylinux_2_17_aarch64', '--platform', 'manylinux2014_aarch64',
             '--platform', 'linux_aarch64', '--python-version', '310', '--abi', 'cp310',
             '--implementation', 'cp', '-r', 'deploy/actor_s100/requirements.txt'], cwd=export, env=env)
        wheel_count = len(list(wheelhouse.glob('*.whl')))
        if len(list(resolved.glob('*.whl'))) != wheel_count:
            raise RuntimeError('Offline wheel closure differs from the bundled wheels')
        command = [sys.executable, 'scripts/bootstrap_actor_s100.py', '--unpack-only']
        if args.model:
            command += ['--model', str(args.model)]
        run(command, cwd=export, env=env)
        if args.supermimic_model:
            run([sys.executable, 'scripts/bootstrap_actor_s100.py', '--unpack-only', '--policy', 'supermimic',
                 '--model', str(args.supermimic_model)], cwd=export, env=env)
        installer_tests = run_tests('tests/test_actor_delivery.py', cwd=export, env=env)
        for entry in ['scripts/bootstrap_actor_s100.py', 'scripts/configure_actor_s100.py',
                      'scripts/bench_pico_gmr_live.py', 'scripts/bench_pico_full_load.py']:
            run([sys.executable, entry, '--help'], cwd=export, env=env)
        full_load_tests = run_tests('tests/test_actor_pico_full_load.py', cwd=export, env=env)
        # Compile actual callback/parser/getter source without Python or ARM
        # libraries; no vendor SDK init or service is run by this regression.
        run([sys.executable, 'scripts/state_test_pico_snapshot.py'], cwd=export, env=env)
        result = dict(status='PASS_STAGED_GIT_DELIVERY', offline_wheels=wheel_count,
                      runtime_archives_verified=True, pico_snapshot_cpp='PASS',
                      installer_tests=installer_tests, full_load_benchmark_tests=full_load_tests,
                      local_provision_tests='not_measured',
                      documentation_files=3, documentation_links=documentation_links,
                      external_model_supplied=bool(args.model),
                      delivery_manifest_sha256=hashlib.sha256((export/'deploy/actor_s100/manifest.json').read_bytes()).hexdigest(),
                      model_steps=0, replay_steps=0, actions_sent=0,
                      s100_install_execution='not_measured', live_inputs='not_measured')
        if args.host_inference:
            # The ARM NumPy wheel is deliberately not used by the host process.
            # Materialize exact SDK sources/schemas at the same deployment path
            # so the runtime's source/hash guard remains active and unmodified.
            site = export/'venv/lib/python3.10/site-packages'
            site.mkdir(parents=True)
            for wheel in wheelhouse.glob('*.whl'):
                if not wheel.name.startswith(('locomotion_aorta-', 'aorta_sdk-', 'aorta_msgs-', 'vbot_humanoid_msgs-', 'flatbuffers-')):
                    continue
                with zipfile.ZipFile(wheel) as archive:
                    archive.extractall(site)
            actor_tests = run_tests('tests/test_actor_pico.py', cwd=export, env=env)
            provision_tests = run_tests('tests/test_actor_local_provision.py', cwd=export, env=env)
            live_tests = run_tests('tests/test_actor_pico_live.py', cwd=export, env=env)
            reader_tests = run_tests('src/tests/test_pico_atomic_reader.py', cwd=export, env=env)
            run([sys.executable, 'scripts/actor_pico.py', 'check', '--steps', '100'], cwd=export, env=env)
            run([sys.executable, 'scripts/actor_pico.py', 'replay', '--input',
                 'tests/fixtures/actor_pico/pico_replay.npz', '--steps', '100'], cwd=export, env=env)
            if args.supermimic_model:
                result['supermimic_tests'] = run_tests('tests/test_actor_supermimic.py', cwd=export, env=env)
                result['pico_session_tests'] = run_tests('tests/test_actor_pico_session.py', cwd=export, env=env)
                run([sys.executable, 'scripts/actor_pico.py', 'check', '--policy', 'supermimic', '--steps', '100'],
                    cwd=export, env=env)
                run([sys.executable, 'scripts/actor_pico.py', 'replay', '--policy', 'supermimic', '--input',
                     'tests/fixtures/actor_pico/pico_replay.npz', '--steps', '100'], cwd=export, env=env)
            result.update(model_steps=100, replay_steps=100, actor_tests=actor_tests, pico_reader_tests=reader_tests,
                          local_provision_tests=provision_tests,
                          live_runner_tests=live_tests, live_runner_io='simulated; real SDK session/math',
                          robot_motion='not_measured')
        (ROOT/'outputs').mkdir(exist_ok=True)
        (ROOT/'outputs/actor_delivery_validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
