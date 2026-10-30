"""Actor inference through SDK 0.2.1 math; no transport or publication.

``policy`` selects a pinned configuration: the SONIC/GR00T G1 actor or the
SuperMimic actor. Both share retargeting, feedback validation and Body31 decode.
"""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .actor_pico_gmr import create_retargeter
from .actor_pico_stream import stream_settings
from .actor_supermimic import (POLICY_KIND as SUPERMIMIC_KIND, NativeModel, SuperMimicContract,
                               create_policy, create_reference, reference_window)

POLICY_CONFIGS = {'sonic': 'config/actor_pico.json', 'supermimic': 'config/actor_pico_supermimic.json'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_actor_config(root, policy='sonic'):
    root = Path(root).resolve()
    if policy not in POLICY_CONFIGS:
        raise ValueError('Unknown actor policy: '+str(policy))
    config = json.loads((root/POLICY_CONFIGS[policy]).read_text())
    if config['schema'] != 'vita_actor_pico_shadow_v1':
        raise ValueError('Unsupported actor/Pico configuration')
    if config.get('policy_kind', 'groot_vita02a_tracking') != {
            'sonic': 'groot_vita02a_tracking', 'supermimic': SUPERMIMIC_KIND}[policy]:
        raise ValueError('Actor configuration does not match the selected policy')
    if policy == 'supermimic':
        reference_window(config)
    if not np.isfinite(config['pico_max_age_s']) or config['pico_max_age_s'] <= 0:
        raise ValueError('Invalid Pico maximum data age')
    hold = config.get('pico_hold_age_s')
    if (isinstance(hold, bool) or not isinstance(hold, (int, float)) or not np.isfinite(hold)
            or not 0 < hold < config['pico_max_age_s']):
        raise ValueError('Pico reference hold age must be positive and below pico_max_age_s')
    if stream_settings(config) is not None and policy != 'supermimic':
        raise ValueError('pico_stream playback is only implemented for the SuperMimic reference')
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
    def __init__(self, path, math_api, *, backend='auto', threads=1, policy_kind='groot_vita02a_tracking'):
        if backend == 'auto':
            backend = 'sdk' if platform.machine() == 'aarch64' else 'ort'
        self.backend = backend
        supermimic = policy_kind == SUPERMIMIC_KIND
        if backend == 'sdk':
            self.model = (NativeModel(path, math_api, threads=threads) if supermimic
                          else math_api.Model(path, threads=threads))
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
            metadata = self.session.get_modelmeta().custom_metadata_map
            self.contract = (SuperMimicContract if supermimic else math_api.Contract).from_metadata(metadata)
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
    def __init__(self, root, *, backend='auto', threads=1, human_height=None, policy='sonic'):
        self.root = Path(root).resolve()
        self.policy_name = policy
        self.config = load_actor_config(root, policy)
        if human_height is not None:
            self.config['retargeting']['human_height_m'] = float(human_height)
        self.stream = stream_settings(self.config)
        self.math = sdk_math(root, self.config)
        self.model = ActorModel(self.root/self.config['artifacts']['model']['path'],
                                self.math, backend=backend, threads=threads,
                                policy_kind=self.config.get('policy_kind', 'groot_vita02a_tracking'))
        self.policy = create_policy(self.model.contract, self.config['body_profile'], self.math, self.config)
        self.reset_inputs()

    def reset_inputs(self):
        """Discard shadow history, calibration and references before a new trial."""
        self.policy.reset()
        self.reset_reference()

    def reset_reference(self):
        """Reset only Pico calibration/reference; preserve running policy history."""
        self.reference = create_reference(self.model.contract, self.math, self.config)
        self.retarget = create_retargeter(self.root, self.config, self.model.contract, self.math)
        self.last_residual = None
        self.last_retarget_ms = None
        self.reference_yaw_offset = None
        self.held_motion = None

    def align_reference_yaw(self, robot_root):
        motion = self.reference.motion()
        current = motion.quat[self.reference.current_index]
        self.reference_yaw_offset = self.math.yaw(robot_root)-self.math.yaw(current)

    def push_pico(self, body, stamp_ns, *, received_monotonic, robot_root):
        if self.reference.last_stamp_ns is not None and stamp_ns <= self.reference.last_stamp_ns:
            raise ValueError('Pico frame did not advance')
        start = time.perf_counter()
        q, root, residual = self.retarget.solve(body, robot_root=robot_root)
        self.reference.append(stamp_ns, q, root, received_monotonic=received_monotonic,
                              base=self.retarget.last_qpos[:3])
        self.last_residual = residual
        self.last_retarget_ms = (time.perf_counter()-start)*1000

    def infer(self, *, q, dq, root, gyro, now_monotonic, maximum_pico_age=None):
        _, observation, row = self.infer_command(q=q, dq=dq, root=root, gyro=gyro,
            now_monotonic=now_monotonic, maximum_pico_age=maximum_pico_age)
        return observation, row

    def infer_command(self, *, q, dq, root, gyro, now_monotonic, maximum_pico_age=None):
        if maximum_pico_age is None:
            maximum_pico_age = self.config['pico_max_age_s']
        age = (float('inf') if self.reference.last_receive_monotonic is None else
               now_monotonic-self.reference.last_receive_monotonic)
        if not 0 <= age <= maximum_pico_age:
            raise TimeoutError('Pico input is stale')
        if self.reference_yaw_offset is None:
            self.align_reference_yaw(root)
        motion, held = self.hold_stale_reference(self.reference.motion(), age)
        return self.infer_motion_command(motion, q=q, dq=dq, root=root, gyro=gyro,
                                 yaw_offset=self.reference_yaw_offset, ref_time=self.reference.ref_time,
                                 diagnostics={'reference_delay_s': self.reference.delay,
                                              'pico_age_s': age, 'reference_hold': held,
                                              'gmr_task_error': self.last_residual,
                                              'retarget_ms': self.last_retarget_ms})

    def hold_stale_reference(self, motion, age):
        """Past pico_hold_age_s, hold the current reference frame with zero velocity.

        A stale window otherwise keeps presenting motion that is no longer
        arriving. Holding matches the training rule beyond a reference's end;
        pico_max_age_s still ends following. The frame is captured once when
        the hold starts and reused until a fresh reference ends it: control-
        clock playback resamples every tick, so its current frame would keep
        advancing toward the newest sample during the hold.
        """
        if age <= self.config['pico_hold_age_s']:
            self.held_motion = None
            return motion, False
        if self.held_motion is None:
            i, n = self.reference.current_index, len(motion.q)
            frame = lambda x: np.repeat(np.asarray(x)[i:i+1], n, axis=0)
            self.held_motion = self.math.Motion(frame(motion.q), frame(motion.base), frame(motion.quat))
        return self.held_motion, True

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
        row = dict(record_type='actor_inference', policy=self.policy_name, backend=self.model.backend,
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
