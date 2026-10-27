#!/usr/bin/env python3
"""Check the staged Git export without borrowing untracked deployment inputs.

--host-inference additionally needs host NumPy and ONNX Runtime. ARM wheels
are resolved offline, and only their SDK Python/schema payload is used on x86.
This is not ARM execution, live Pico reception, or robot control validation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Check failed: '+str(command)+'\n'+result.stdout)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-inference', action='store_true')
    parser.add_argument('--model', type=Path, help='Explicit external actor model for host inference')
    args = parser.parse_args()
    if args.host_inference and not args.model:
        parser.error('--host-inference requires --model /path/to/actor_g1.onnx (not stored in Git)')
    if args.model:
        args.model = args.model.resolve()
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('PIP_'):
            env.pop(name)
    for key in ['PYTHONPATH', 'PYTHONHOME', 'VITA_PREPROCESS_LIBRARY',
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
                     'artifacts/actor_g1/actor_g1.onnx']:
            if (export/name).exists():
                raise RuntimeError('Unexpected untracked input in Git export: '+name)
        run([sys.executable, 'scripts/bootstrap_actor_s100.py', '--verify-only'], cwd=export, env=env)
        wheelhouse = export/'deploy/actor_s100/wheelhouse'
        resolved = Path(temporary)/'resolved_wheels'
        # Resolve the full dependency graph for the target ABI, offline.
        run([sys.executable, '-m', 'pip', 'download', '--no-index',
             '--find-links', str(wheelhouse), '--dest', str(resolved),
             '--only-binary=:all:', '--platform', 'manylinux_2_35_aarch64',
             '--platform', 'manylinux_2_17_aarch64', '--platform', 'manylinux2014_aarch64',
             '--platform', 'linux_aarch64', '--python-version', '310', '--abi', 'cp310',
             '--implementation', 'cp', '-r', 'deploy/actor_s100/requirements.txt'], cwd=export, env=env)
        if len(list(resolved.glob('*.whl'))) != 8:
            raise RuntimeError('Offline wheel closure changed from the declared eight wheels')
        command = [sys.executable, 'scripts/bootstrap_actor_s100.py', '--unpack-only']
        if args.model:
            command += ['--model', str(args.model)]
        run(command, cwd=export, env=env)
        run([sys.executable, 'tests/test_actor_delivery.py'], cwd=export, env=env)
        # Compile actual callback/parser/getter source without Python or ARM
        # libraries; no vendor SDK init or service is run by this regression.
        run([sys.executable, 'scripts/state_test_pico_snapshot.py'], cwd=export, env=env)
        result = dict(status='PASS_STAGED_GIT_DELIVERY', offline_wheels=8,
                      runtime_archives_verified=True, pico_snapshot_cpp='PASS',
                      installer_tests=9, external_model_supplied=bool(args.model),
                      model_steps=0, replay_steps=0, actions_sent=0,
                      s100_install_execution='not_measured', live_inputs='not_measured')
        if args.host_inference:
            # The ARM NumPy wheel is deliberately not used by the host process.
            # Materialize exact SDK sources/schemas at the same deployment path
            # so the runtime's source/hash guard remains active and unmodified.
            site = export/'venv/lib/python3.10/site-packages'
            site.mkdir(parents=True)
            for wheel in wheelhouse.glob('*.whl'):
                if wheel.name.startswith(('numpy-', 'pybind11-', 'pyyaml-')):
                    continue
                with zipfile.ZipFile(wheel) as archive:
                    archive.extractall(site)
            run([sys.executable, 'tests/test_actor_pico.py'], cwd=export, env=env)
            run([sys.executable, 'tests/test_actor_pico_live.py'], cwd=export, env=env)
            run([sys.executable, 'src/tests/test_pico_atomic_reader.py'], cwd=export, env=env)
            run([sys.executable, 'scripts/actor_pico.py', 'check', '--steps', '100'], cwd=export, env=env)
            run([sys.executable, 'scripts/actor_pico.py', 'replay', '--input',
                 'tests/fixtures/actor_pico/pico_replay.npz', '--steps', '100'], cwd=export, env=env)
            result.update(model_steps=100, replay_steps=100, actor_tests=15, pico_reader_tests=3,
                          live_runner_tests=72, live_runner_io='simulated; real SDK session/math',
                          robot_motion='not_measured')
        (ROOT/'outputs').mkdir(exist_ok=True)
        (ROOT/'outputs/actor_delivery_validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
