"""Exercise the real native bundle preflight without SDK or robot access."""
import hashlib
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run_entry(bundle, *flags, expected=0):
    result = subprocess.run(
        ['bash', str(ROOT/'scripts/start_mimiclite_native.sh'),
         '--model', 'student', '--native-root', str(bundle), *flags],
        env={k: v for k, v in os.environ.items() if k != 'MIMICLITE_INFERENCE_CPUS'},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == expected, result.stderr
    return result.stdout


def test_native_bundle_checks_model_and_contract(tmp_path):
    inputs = {'student.onnx': b'model', 'contract_student.json': b'{}'}
    for name, data in inputs.items():
        (tmp_path/name).write_bytes(data)
    (tmp_path/'student.sha256').write_text(''.join(
        f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name, data in inputs.items()
    ))
    bridge = tmp_path/'humanoid_tracking_client'
    bridge.touch(mode=0o755)
    for name in ('mimiclite_native', 'mimiclite_native_control',
                 'mimiclite_reference_channel', 'mimiclite_reference_pipeline'):
        (tmp_path/(name+'.py')).touch()
    assert 'native bundle verified' in run_entry(tmp_path, '--check')
    (tmp_path/'contract_student.json').write_text('changed')
    run_entry(tmp_path, '--check', expected=2)


def test_plan_defaults_to_cpu4_and_selects_reference_source(tmp_path):
    pico = run_entry(tmp_path, '--plan')
    assert 'native inference thread CPUs: 4' in pico
    assert '--no-camera' in pico and '--pico' in pico
    assert '--activation-file' in pico and '--log' not in pico
    assert '--no-camera' not in run_entry(tmp_path, '--plan', '--stand-only')
    assert '--motion' in run_entry(tmp_path, '--plan', '--motion', '/example.npz')

def test_camera_reuses_pico_session_relay(tmp_path):
    plan = run_entry(tmp_path, '--plan', '--camera')
    assert 'pico_session_services.py' in plan and '--pico' in plan
    assert '--no-camera' not in plan.split()
    assert '--no-camera-logs' in plan and '--log' not in plan
    capture = run_entry(tmp_path, '--plan', '--camera', '--log')
    assert '--no-camera-logs' not in capture and '--log' in capture
    run_entry(tmp_path, '--plan', '--camera', '--stand-only', expected=2)
    run_entry(tmp_path, '--plan', '--camera', '--motion', '/example.npz', expected=2)
