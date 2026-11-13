"""Offline same-input SONIC comparison. No robot, publisher, or service.

Tolerance is fixed before the session is created. One finite observation is
fed to the pinned ONNX twice. The resulting action is decoded by the WTT
native affine formula and by the production C++ helper
DecodeRawMotorThetaTrackingAction, including ankle slots. This does not build
a live observation from measured joint state.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

MODEL = Path(__file__).resolve().parents[3] / 'artifacts/actor_g1/model_step_009700_longrun_closed_loop_g1.onnx'
MODEL_SHA256 = '2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d'
HEADER = Path('/host/vita-robot_githhub/src/application/locomotion/humanoid/models/vt_human_obs.h')
# Declared before any inference session is constructed.
ABS_TOLERANCE = 1e-5
REL_TOLERANCE = 1e-5
CONTRACT_KEY = 'groot_wbc.tracking_contract.v1'


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(width):
    index = np.arange(width, dtype=np.float32)
    return (np.sin(index * np.float32(0.01)) * np.float32(0.05)).astype(np.float32)


def _decode_python(action, clip, scale, default):
    clipped = np.clip(np.asarray(action, dtype=np.float32),
                      np.float32(-clip), np.float32(clip))
    return clipped.astype(np.float64) * np.asarray(scale, dtype=np.float64) + np.asarray(default, dtype=np.float64)


def _decode_cpp(action, clip, scale, default):
    source = r'''
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "vt_human_obs.h"
static std::vector<double> numbers(const std::string& text) {
  std::vector<double> out;
  std::stringstream stream(text);
  double value;
  while (stream >> value) out.push_back(value);
  return out;
}
int main(int argc, char** argv) {
  std::ifstream input(argv[1]);
  std::string action_text, scale_text, default_text, clip_text;
  std::getline(input, action_text);
  std::getline(input, scale_text);
  std::getline(input, default_text);
  std::getline(input, clip_text);
  auto as_float = [](const std::vector<double>& values) {
    std::vector<float> out(values.begin(), values.end());
    return out;
  };
  std::vector<float> clipped;
  auto targets = vita::humanoid::DecodeRawMotorThetaTrackingAction(
      as_float(numbers(action_text)), std::stod(clip_text), numbers(scale_text),
      numbers(default_text), &clipped);
  std::cout.setf(std::ios::scientific);
  std::cout.precision(17);
  for (double value : targets) std::cout << value << ' ';
  std::cout << '\n';
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'compare.cpp').write_text(source)
        binary = root / 'compare'
        subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(HEADER.parent),
                        str(root / 'compare.cpp'), '-o', str(binary)], check=True)
        payload = root / 'input.txt'
        lines = [' '.join(format(float(value), '.17g') for value in row) for row in (action, scale, default)]
        payload.write_text('\n'.join(lines + [format(float(clip), '.17g')]) + '\n')
        completed = subprocess.run([str(binary), str(payload)], check=True, text=True, capture_output=True)
    return np.fromstring(completed.stdout, sep=' ', dtype=np.float64)


def _within(left, right):
    delta = np.abs(left - right)
    scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
    return bool(np.all(delta <= ABS_TOLERANCE) and np.all(delta <= REL_TOLERANCE * scale)), float(np.max(delta))


def compare():
    if _sha256(MODEL) != MODEL_SHA256:
        raise SystemExit('model SHA256 does not match the pinned SONIC9700 file')
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), options, providers=['CPUExecutionProvider'])
    contract = json.loads(session.get_modelmeta().custom_metadata_map[CONTRACT_KEY])
    names = list(contract['joint_names'])
    if len(names) != 28:
        raise SystemExit('expected 28 policy joints')
    spec = session.get_inputs()[0]
    width = next(dim for dim in spec.shape if isinstance(dim, int) and dim > 1)
    vector = _observation(width)
    feed = vector.reshape(1, width) if len(spec.shape) >= 2 else vector
    first = np.asarray(session.run(None, {spec.name: feed})[0], dtype=np.float32).reshape(-1)
    second = np.asarray(session.run(None, {spec.name: feed})[0], dtype=np.float32).reshape(-1)
    if first.shape != (28,) or not np.isfinite(first).all() or not np.array_equal(first, second):
        raise SystemExit('same-session ONNX outputs are not finite and identical')
    scale = np.asarray(contract['action_scale'], dtype=np.float64)
    default = np.asarray(contract['default_joint_pos'], dtype=np.float64)
    clip = float(contract['action_clip'])
    python_q = _decode_python(first, clip, scale, default)
    cpp_q = _decode_cpp(first, clip, scale, default)
    ok, max_abs = _within(python_q, cpp_q)
    ankle = [index for index, name in enumerate(names) if 'ankle' in name]
    ankle_abs = float(np.max(np.abs(python_q[ankle] - cpp_q[ankle]))) if ankle else None
    aliases = contract.get('ankle', {}).get('policy_joint_aliases', {})
    report = dict(model_sha256=MODEL_SHA256, observation_width=width,
                  abs_tolerance=ABS_TOLERANCE, rel_tolerance=REL_TOLERANCE,
                  same_session_exact=True, max_abs_motor_target=max_abs,
                  ankle_control_space=contract.get('ankle', {}).get('control_space'),
                  ankle_names=[names[index] for index in ankle],
                  ankle_motor_joints=aliases, ankle_max_abs=ankle_abs,
                  passed=ok and ankle_abs is not None and ankle_abs <= ABS_TOLERANCE)
    print(json.dumps(report, sort_keys=True))
    if not report['passed'] or not math.isfinite(max_abs):
        raise SystemExit('motor-target comparison exceeded the declared tolerance')
    return report


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    compare()
