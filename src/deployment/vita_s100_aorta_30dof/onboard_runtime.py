"""Resident S100 dual-policy runtime. No robot transport or actuator API."""
from __future__ import annotations

import json
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np

from deployment.common.model_contract import (ArtifactStatus, DeploymentScope, ImmutableArtifact, ModelBundle, sha256_file)
from .model_contract import VitaBundleContracts, inspect_vita_session
from .onboard_preprocessing import CPUOnlineSmplReference, CPUSmplMotionObservationBuilder
from .robot_adapter import values_in_order
from .runtime import VitaDualPolicyRuntime
from .velocity_observation import VitaVelocityObservationBuilder


OUTPUT_ENVELOPES_SHA256 = '595b0d1fb26739e8fd3b0874dbc70bdd89ad8d772b5103adce199433dcaf03e0'
ORIGINAL_MODEL_SHA256 = {
    'velocity': 'f78fd39eb3f420675af63782d12ce2e6434f62e3159e1fd5b17b65b4fd6d7126',
    'motion': '881d936f560e22e81302483b394cd6a89ba5648e6c2893fa153172dec38033a7',
}


def native_bundle_metadata(manifest_path, *, velocity_backend='ort',
                           deployment_scope=DeploymentScope.SHADOW,
                           expected_model_hashes=None):
    """Recorder metadata for the composite deployed model, not a fictitious ONNX.

    NativeBackend verifies every loaded artifact. The Motion recorder entry is
    the composite manifest itself, with its own real file hash. Original model
    identities remain explicitly separate in the warmup report.
    """
    path = Path(manifest_path).resolve()
    record = json.loads(path.read_text())
    if record.get('source_hashes') != (expected_model_hashes or ORIGINAL_MODEL_SHA256):
        raise ValueError('Native manifest original policy identities mismatch')
    if record.get('model_contract', {}).get('policy_hz') != 50:
        raise ValueError('Native manifest must declare 50 Hz')
    if velocity_backend not in ('ort', 'hbm'):
        raise ValueError('Unsupported Velocity backend')
    native = {}
    for name, item in record['artifacts'].items():
        digest = str(item['sha256'])
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid native artifact SHA256')
        native[name] = ImmutableArtifact(name=name, path=(path.parent / item['path']).resolve(),
            sha256=digest, status=ArtifactStatus.TRAINED)
    chosen = native['velocity' if velocity_backend == 'ort' else 'velocity_hbm']
    artifacts = {'native_' + name: value for name, value in native.items()}
    artifacts['velocity'] = replace(chosen, name='velocity')
    artifacts['motion'] = ImmutableArtifact(name='motion', path=path,
        sha256=sha256_file(path), status=ArtifactStatus.TRAINED)
    return ModelBundle(manifest_path=path, schema='vita_onboard_native_bundle_v1', robot='Vita02A',
        policy_hz=50.0, deployment_scope=DeploymentScope(deployment_scope),
        artifacts=MappingProxyType(artifacts))


class _ContractView:
    """Frozen original ONNX metadata; never opens a second policy session."""

    def __init__(self, record):
        self.record = record

    def _tensors(self, key):
        types = {"FLOAT": "tensor(float)", "INT64": "tensor(int64)"}
        return [SimpleNamespace(name=x["name"], shape=x["shape"], type=types[x["dtype"]])
                for x in self.record[key]]

    def get_inputs(self):
        return self._tensors("inputs")

    def get_outputs(self):
        return self._tensors("outputs")

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.record["metadata_props"])


def load_contracts(path, expected_model_hashes=None):
    records = json.loads(Path(path).read_text())
    expected = expected_model_hashes or ORIGINAL_MODEL_SHA256
    for name, digest in expected.items():
        if records[name]["sha256"] != digest:
            raise ValueError(f"Unexpected original {name} model identity")
    return VitaBundleContracts(**{
        name: inspect_vita_session(_ContractView(records[name]), name)
        for name in expected
    })


class OnboardDualPolicyRuntime(VitaDualPolicyRuntime):
    """Reuse original observations/action mapping with injected resident runners.

    Both runners are created once by the caller. A switch resets history/KV,
    retaining all model sessions, BPU buffers and expert pools.
    """

    def __init__(self, *, contracts_path, skeleton_path, motion_runner, velocity_runner,
                 runtime_root=None, envelopes_path=None, bundle=None,
                 expected_model_hashes=None, envelopes_sha256=OUTPUT_ENVELOPES_SHA256):
        self.source_model_sha256 = dict(expected_model_hashes or ORIGINAL_MODEL_SHA256)
        self.envelopes_sha256 = envelopes_sha256
        self.contracts = load_contracts(contracts_path, self.source_model_sha256)
        self._metadata_cache = {}
        self.bundle = bundle if bundle is not None else SimpleNamespace(policy_hz=50.0, artifacts={})
        self.runtime_root = Path(runtime_root).resolve() if runtime_root is not None else Path(contracts_path).resolve().parent.parent
        self.envelopes_path = Path(envelopes_path) if envelopes_path is not None else self.runtime_root / 'artifacts/output_envelopes.json'
        self._output_envelopes = None
        self._owned_backend = None
        self.motion_runner = motion_runner
        self.velocity_runner = velocity_runner
        self.skeleton_path = Path(skeleton_path)
        self.device = "cpu"
        self.motion_policy_cuda = False
        self.unified_velocity = False
        self.velocity_observation = VitaVelocityObservationBuilder(self.contracts.velocity)
        motion = self.contracts.motion
        defaults = values_in_order(motion.joint_names, motion.default_joint_pos,
                                   motion.action_joint_names)
        self.motion_observation_builder = CPUSmplMotionObservationBuilder(
            n_future_frames=10, default_dof_pos=defaults, policy_hz=50.0)
        self.last_motion_action = np.zeros(30, np.float32)
        self.last_velocity_action = np.zeros(28, np.float32)
        self.last_observation = None
        self.last_raw_action = None
        self._velocity_metadata_template = super()._velocity_full_metadata()

    def _metadata(self, contract, names):
        key=(contract.kind,tuple(names))
        if key not in self._metadata_cache:
            self._metadata_cache[key]=VitaDualPolicyRuntime._metadata(contract,names)
        return self._metadata_cache[key]

    def _velocity_full_metadata(self):
        # Policy mapping mutates these arrays. Keep the resident templates intact.
        return tuple(value.copy() for value in self._velocity_metadata_template)

    def create_online_reference(self):
        return CPUOnlineSmplReference(n_future_frames=10, policy_hz=50.0,
                                      skeleton_path=self.skeleton_path)

    # Native backend counters use their own clock. The inherited RT runner
    # boundaries provide the latency measurement; do not mix those counters in.

    @classmethod
    def from_onboard_root(cls, root, *, velocity_backend='ort', timeout_ms=100,
                         parallel_router=True, deployment_scope=DeploymentScope.SHADOW,
                         library=None, manifest=None, motion_group=None):
        """Create one resident native backend; caller must close the runtime.

        This factory performs no SDK initialization, subscriptions, leases or
        publication. Scope is metadata only; it never grants control authority.
        """
        from .onboard_backend import NativeBackend, NativeMotionRunner, NativeVelocityRunner
        root = Path(root).resolve()
        selection = None
        expected = dict(ORIGINAL_MODEL_SHA256)
        contracts_path = root / 'artifacts/policy_contracts.json'
        envelopes_path = root / 'artifacts/output_envelopes.json'
        envelopes_hash = OUTPUT_ENVELOPES_SHA256
        if motion_group is not None:
            from .motion_groups import load_motion_group
            if manifest is not None:
                raise ValueError('Choose a Motion group or a manifest, not both')
            selection = load_motion_group(root, motion_group)
            manifest = selection['manifest']
            contracts_path = selection['contracts']
            envelopes_path = selection['envelopes']
            envelopes_hash = selection['envelopes_sha256']
            expected['motion'] = selection['source_sha256']
        manifest = Path(manifest).resolve() if manifest is not None else root / 'outputs/model_runtime_manifest.json'
        library = Path(library).resolve() if library is not None else root / 'build/libvita_backend.so'
        bundle = native_bundle_metadata(manifest, velocity_backend=velocity_backend,
                                        deployment_scope=deployment_scope,
                                        expected_model_hashes=expected)
        backend = NativeBackend(library, manifest, velocity_backend=velocity_backend,
            timeout_ms=timeout_ms, parallel_router=parallel_router, deadline_ms=20.0)
        try:
            runtime = cls(contracts_path=contracts_path,
                skeleton_path=root / 'artifacts/preprocessing_skeleton.npz',
                motion_runner=NativeMotionRunner(backend), velocity_runner=NativeVelocityRunner(backend),
                runtime_root=root, bundle=bundle, envelopes_path=envelopes_path,
                expected_model_hashes=expected, envelopes_sha256=envelopes_hash)
            runtime.motion_selection = ({key: selection[key] for key in
                ('group', 'iteration', 'source_sha256')} if selection else None)
            # Fail before any live integration if original envelope provenance is absent.
            runtime._load_output_envelopes()
            runtime._owned_backend = backend
            return runtime
        except BaseException:
            backend.close()
            raise

    def _load_output_envelopes(self):
        if self._output_envelopes is None:
            from .onboard_output import load_output_envelopes
            self._output_envelopes = load_output_envelopes(self.envelopes_path,
                expected_sha256=self.envelopes_sha256)
        return self._output_envelopes

    def command_envelope(self, base, mode):
        """Return original frozen policy limits after checking the live base contract."""
        if mode not in ('velocity', 'motion'):
            raise ValueError('Unknown policy envelope: ' + str(mode))
        envelopes = self._load_output_envelopes()
        expected = envelopes['base']
        for item in fields(expected):
            actual_value = getattr(base, item.name)
            expected_value = getattr(expected, item.name)
            equal = np.array_equal(actual_value, expected_value) if isinstance(expected_value, np.ndarray) else actual_value == expected_value
            if not equal:
                raise ValueError('Live interface differs from pinned original envelope: ' + item.name)
        envelope = envelopes[mode]
        return replace(envelope, **{item.name: getattr(envelope, item.name).copy()
            for item in fields(envelope) if isinstance(getattr(envelope, item.name), np.ndarray)})

    def warmup(self, iterations=20):
        """Warm both resident policies from recorded observations, then clear state.

        This is inference-only startup work. It performs no GPU preprocessing
        and never substitutes recorded observations into the subsequent live run.
        """
        if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError('warmup iterations must be a positive integer')
        started = time.time_ns()
        timings = {'velocity': [], 'motion': []}
        loaded = {}
        self.reset()
        try:
            for name, directory, width in (('motion', 'policy_motion_recorded_8192', 3126),
                                           ('velocity', 'policy_velocity_recorded_1536', 990)):
                path = self.runtime_root / 'reference' / directory / 'observations.npy'
                observations = np.load(path, mmap_mode='r', allow_pickle=False)
                if observations.shape[1:] not in ((width,), (1, width)) or len(observations) < iterations or observations.dtype != np.float32:
                    raise ValueError('Invalid recorded warmup observations: ' + str(path))
                loaded[name] = np.ascontiguousarray(observations[:iterations].reshape(iterations, width)).copy()
                if not np.isfinite(loaded[name]).all():
                    raise ValueError('Nonfinite recorded warmup observations: ' + str(path))
            for index in range(iterations):
                for name, width in (('velocity', 28), ('motion', 30)):
                    before = time.time_ns()
                    action = getattr(self, name + '_runner').run(loaded[name][index:index+1])
                    timings[name].append((time.time_ns() - before) / 1e6)
                    action = np.asarray(action)
                    if action.size != width or not np.isfinite(action).all():
                        raise RuntimeError('Invalid native warmup action: ' + name)
        finally:
            self.reset()
            self.last_observation = None
            self.last_raw_action = None
        providers = {}
        for name in ('velocity', 'motion'):
            runner = getattr(self, name + '_runner')
            session = getattr(runner, 'session', runner)
            providers[name] = list(session.get_providers()) if hasattr(session, 'get_providers') else ['InjectedRunner']
        return {'no_robot_action': True, 'providers': providers,
            'velocity_steps': int(iterations), 'motion_steps': int(iterations),
            'timing_ms': {name: float(max(values)) for name, values in timings.items()},
            'mean_timing_ms': {name: float(np.mean(values)) for name, values in timings.items()},
            'elapsed_ms': (time.time_ns() - started) / 1e6,
            'source_model_sha256': dict(self.source_model_sha256),
            'motion_representation': 'Composite native manifest: CPU ORT cuts/router + FP32 UCP expert operations',
            'state_reset_after_warmup': True,
            'input_kind': 'First original recorded observations; no expected actions/cache injected'}

    def close(self):
        """Close only the backend created by from_onboard_root; injected runners remain caller-owned."""
        backend, self._owned_backend = self._owned_backend, None
        if backend is not None:
            backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
