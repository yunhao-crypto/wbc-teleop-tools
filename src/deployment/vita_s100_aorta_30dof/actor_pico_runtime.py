"""actor_g1 inference through SDK 0.2.1 math; no transport or publication."""
import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .actor_pico_reference import CausalActorReference, PicoOrientationRetargeter


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_actor_config(root):
    root = Path(root).resolve()
    config = json.loads((root/'config/actor_pico.json').read_text())
    if config['schema'] != 'vita_actor_pico_shadow_v1':
        raise ValueError('Unsupported actor/Pico configuration')
    for item in config['artifacts'].values():
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root) or digest(path) != item['sha256']:
            raise ValueError('Actor artifact missing or hash mismatch: '+item['path'])
    return config


def sdk_math(root, config):
    """Use the shipped SDK's real WBC math, including on an x86 offline host.

    Appending the target site-packages keeps host NumPy/ORT ahead of ARM wheels.
    Live execution must use the target venv and the actual SDK native binaries.
    """
    site = Path(root).resolve()/'venv/lib/python3.10/site-packages'
    for name, expected in config['sdk_sources'].items():
        if digest(site/'locomotion_aorta'/name) != expected:
            raise ValueError('Unreviewed SDK actor dependency: '+name)
    if str(site) not in sys.path:
        sys.path.append(str(site))
    module = importlib.import_module('locomotion_aorta.wbc_tracking_policy')
    if Path(module.__file__).resolve() != site/'locomotion_aorta/wbc_tracking_policy.py':
        raise RuntimeError('Actor requires the pinned SDK from this deployment venv')
    return module


class ActorModel:
    def __init__(self, path, math_api, *, backend='auto', threads=1):
        if backend == 'auto':
            backend = 'sdk' if platform.machine() == 'aarch64' else 'ort'
        self.backend = backend
        if backend == 'sdk':
            self.model = math_api.Model(path, threads=threads)
            self.contract = self.model.contract
        elif backend == 'ort':
            import onnxruntime as ort
            if threads < 1:
                raise ValueError('ORT thread count must be positive')
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), options,
                                                providers=['CPUExecutionProvider'])
            self.contract = math_api.Contract.from_metadata(self.session.get_modelmeta().custom_metadata_map)
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if (len(inputs) != 1 or len(outputs) != 1
                    or inputs[0].shape != [1, self.contract.input_dim]
                    or outputs[0].shape != [1, 28]
                    or inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)'):
                raise ValueError('Actor ONNX tensors differ from metadata')
            self.input, self.output = inputs[0].name, outputs[0].name
        else:
            raise ValueError('Unknown actor inference backend')

    def infer(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.contract.input_dim,) or not np.isfinite(obs).all():
            raise ValueError('Invalid actor input')
        if self.backend == 'sdk':
            result = self.model.infer(obs)
        else:
            result = self.session.run([self.output], {self.input: obs[None]})[0][0]
        if result.shape != (28,) or not np.isfinite(result).all():
            raise ValueError('Invalid actor output')
        return result


class ActorPicoRuntime:
    def __init__(self, root, *, backend='auto', threads=1):
        self.root = Path(root).resolve()
        self.config = load_actor_config(root)
        self.math = sdk_math(root, self.config)
        self.model = ActorModel(self.root/self.config['artifacts']['model']['path'],
                                self.math, backend=backend, threads=threads)
        self.policy = self.math.WbcTrackingPolicy(self.model.contract, self.config['body_profile'])
        self.reset_inputs()

    def reset_inputs(self):
        """Discard shadow history, calibration and references before a new trial."""
        self.policy.reset()
        self.reset_reference()

    def reset_reference(self):
        """Reset only Pico calibration/reference; preserve running policy history."""
        self.reference = CausalActorReference(self.model.contract, self.math,
                                              maximum_gap=self.config['reference_maximum_gap_s'])
        self.retarget = PicoOrientationRetargeter(
            urdf=self.root/self.config['artifacts']['urdf']['path'],
            contract=replace(self.model.contract, lower=self.policy.lower, upper=self.policy.upper),
            mapping=self.config['segments'], math_api=self.math,
            iterations=self.config['ik_iterations'], max_residual_rad=self.config['ik_max_residual_rad'])
        self.last_residual = None
        self.last_retarget_ms = None

    def push_pico(self, body, stamp_ns, *, received_monotonic, robot_root):
        if self.reference.last_stamp_ns is not None and stamp_ns <= self.reference.last_stamp_ns:
            raise ValueError('Pico frame did not advance')
        start = time.perf_counter()
        q, root, residual = self.retarget.solve(body, robot_root=robot_root)
        self.reference.append(stamp_ns, q, root, received_monotonic=received_monotonic)
        self.last_residual = residual
        self.last_retarget_ms = (time.perf_counter()-start)*1000

    def infer(self, *, q, dq, root, gyro, now_monotonic, maximum_pico_age=.6):
        _, observation, row = self.infer_command(q=q, dq=dq, root=root, gyro=gyro,
            now_monotonic=now_monotonic, maximum_pico_age=maximum_pico_age)
        return observation, row

    def infer_command(self, *, q, dq, root, gyro, now_monotonic, maximum_pico_age=.6):
        age = (float('inf') if self.reference.last_receive_monotonic is None else
               now_monotonic-self.reference.last_receive_monotonic)
        if not 0 <= age <= maximum_pico_age:
            raise TimeoutError('Pico input is stale')
        motion = self.reference.motion()
        return self.infer_motion_command(motion, q=q, dq=dq, root=root, gyro=gyro,
                                 diagnostics={'reference_delay_s': self.reference.delay,
                                              'pico_age_s': age, 'ik_max_residual_rad': self.last_residual,
                                              'ik_iterations': self.retarget.last_iterations,
                                              'retarget_ms': self.last_retarget_ms})

    def infer_motion(self, motion, *, q, dq, root, gyro, diagnostics=None):
        _, observation, row = self.infer_motion_command(motion, q=q, dq=dq,
            root=root, gyro=gyro, diagnostics=diagnostics)
        return observation, row

    def infer_motion_command(self, motion, *, q, dq, root, gyro,
                             diagnostics=None, ref_time=0., yaw_offset=0.):
        """Return the complete inert SDK command, including ankle feedforward.

        Only the explicit live runner may publish it. Offline/shadow callers
        retain their two-value API and never construct control handles.
        """
        begin = time.perf_counter()
        observation = self.policy.observe(motion, ref_time, q, dq, root, gyro, yaw_offset)
        before_infer = time.perf_counter()
        action = self.model.infer(observation)
        after_infer = time.perf_counter()
        # This constructs an inert DTO only. No publisher/controller is owned.
        command, decoded = self.policy.decode(action)
        row = dict(record_type='actor_inference', backend=self.model.backend,
                   observation_dim=len(observation), action_dim=len(action),
                   observation_ms=(before_infer-begin)*1000,
                   inference_ms=(after_infer-before_infer)*1000,
                   total_ms=(time.perf_counter()-begin)*1000,
                   raw_action=action.tolist(), decoded=decoded,
                   body_order=list(self.policy.order),
                   body_targets=[dict(q=j.q, dq=j.dq, tau=j.tau, kp=j.kp, kd=j.kd)
                                 for j in command.joints],
                   actions_sent=0, control_handles_created=0,
                   **(diagnostics or {}))
        return command, observation, row
