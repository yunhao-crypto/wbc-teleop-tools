"""Action-sink-free Velocity/Motion inference runtime for Vita02A."""

from __future__ import annotations
import time
from typing import Any, Sequence

import numpy as np

from deployment.common.model_contract import ModelBundle, verify_model_bundle
from deployment.common.policy_runtime import (
    FeedForwardOnnxRunner,
    PolicyStep,
    RecurrentOnnxRunner,
    create_onnx_session,
)
from deployment.common.smpl_reference import (
    create_motion_observation_builder,
    create_online_smpl_reference,
)
from deployment.vita_s100_aorta_30dof.model_contract import (
    VITA_MOTION_ACTION_DIM,
    VITA_MOTION_KV_SHAPE,
    VITA_UNIFIED_VELOCITY_OBSERVATION_DIM,
    VITA_VELOCITY_ACTION_DIM,
    VitaBundleContracts,
    VitaPolicyContract,
    inspect_vita_bundle,
)
from deployment.vita_s100_aorta_30dof.robot_adapter import (
    VITA_CONTROLLER_JOINT_NAMES,
    VitaRobotState,
    values_in_order,
)
from deployment.vita_s100_aorta_30dof.velocity_observation import (
    VitaVelocityCommand,
    VitaVelocityObservationBuilder,
)


class VitaDualPolicyRuntime:
    """Inference core with no Aorta session or command publisher."""

    def __init__(
        self,
        *,
        bundle: ModelBundle,
        contracts: VitaBundleContracts,
        motion_session: Any,
        velocity_session: Any,
        motion_observation_builder: Any,
        device: str,
        motion_policy_cuda: bool,
    ) -> None:
        self.last_observation = None
        self.last_raw_action = None
        self.bundle = bundle
        self.contracts = contracts
        self.motion_runner = RecurrentOnnxRunner(
            motion_session,
            observation_name=contracts.motion.observation_name,
            action_name=contracts.motion.action_name,
            observation_shape=contracts.motion.observation_shape,
            observation_dtype=contracts.motion.observation_dtype,
            action_size=VITA_MOTION_ACTION_DIM,
            kv_input_name=str(contracts.motion.kv_input_name),
            kv_output_name=str(contracts.motion.kv_output_name),
            kv_shape=VITA_MOTION_KV_SHAPE,
            kv_dtype=contracts.motion.observation_dtype,
            step_index_name=str(contracts.motion.step_index_name),
            rope_max_seq_len=int(contracts.motion.rope_max_seq_len or 0),
        )
        self.velocity_runner = FeedForwardOnnxRunner(
            velocity_session,
            input_name=contracts.velocity.observation_name,
            output_name=contracts.velocity.action_name,
            input_shape=contracts.velocity.observation_shape,
            input_dtype=contracts.velocity.observation_dtype,
            action_size=VITA_VELOCITY_ACTION_DIM,
            label="Vita velocity policy",
        )
        self.motion_observation_builder = motion_observation_builder
        self.device = device
        self.motion_policy_cuda = motion_policy_cuda
        self.unified_velocity = contracts.velocity.observation_shape == (
            1,
            VITA_UNIFIED_VELOCITY_OBSERVATION_DIM,
        )
        if self.unified_velocity:
            from deployment.common.unified_velocity_observation import (
                UnifiedVelocityObservationBuilder,
            )
            if "velocity_config" not in bundle.artifacts:
                raise ValueError(
                    "unified Velocity requires a hash-checked "
                    "velocity_config artifact"
                )
            self.velocity_observation = UnifiedVelocityObservationBuilder(
                contracts.velocity, bundle.artifacts["velocity_config"].path
            )
        else:
            self.velocity_observation = VitaVelocityObservationBuilder(
                contracts.velocity
            )
        self.last_motion_action = np.zeros(VITA_MOTION_ACTION_DIM, np.float32)
        self.last_velocity_action = np.zeros(
            VITA_VELOCITY_ACTION_DIM, np.float32
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: ModelBundle,
        *,
        require_trained: bool,
        providers: Sequence[Any] | None = None,
        require_cuda: bool = True,
        velocity_providers: Sequence[Any] | None = None,
        velocity_require_cuda: bool | None = None,
    ) -> "VitaDualPolicyRuntime":
        verify_model_bundle(bundle, require_trained=require_trained)
        motion_session = create_onnx_session(
            bundle.artifacts["motion"].path,
            providers=providers,
            require_cuda=require_cuda,
        )
        velocity_session = create_onnx_session(
            bundle.artifacts["velocity"].path,
            providers=(
                providers if velocity_providers is None else velocity_providers
            ),
            require_cuda=(
                require_cuda
                if velocity_require_cuda is None
                else velocity_require_cuda
            ),
        )
        contracts = inspect_vita_bundle(
            bundle,
            motion_session=motion_session,
            velocity_session=velocity_session,
            verify_files=False,
        )
        defaults = values_in_order(
            contracts.motion.joint_names,
            contracts.motion.default_joint_pos,
            contracts.motion.action_joint_names,
        )
        # The mature HoloMotion SMPL builder is a CUDA/Warp implementation.
        # Select ONNX providers independently: the small Velocity network and
        # recurrent Motion network have different CPU/GPU latency tradeoffs.
        device = "cuda:0"
        builder = create_motion_observation_builder(
            n_future_frames=10,
            default_dof_pos=defaults,
            policy_hz=bundle.policy_hz,
            device=device,
        )
        return cls(
            bundle=bundle,
            contracts=contracts,
            motion_session=motion_session,
            velocity_session=velocity_session,
            motion_observation_builder=builder,
            device=device,
            motion_policy_cuda=(
                "CUDAExecutionProvider" in motion_session.get_providers()
            ),
        )

    def reset(self) -> None:
        self.motion_runner.reset()
        self.velocity_observation.reset()
        self.last_motion_action.fill(0.0)
        self.last_velocity_action.fill(0.0)

    def command_envelope(self, base, mode):
        from deployment.vita_s100_aorta_30dof.policy_limits import (
            policy_envelope,
        )

        return policy_envelope(base, self.contracts, self.bundle, mode)

    @staticmethod
    def _metadata(
        contract: VitaPolicyContract,
        names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            values_in_order(contract.joint_names, value, names)
            for value in (
                contract.default_joint_pos,
                contract.joint_stiffness,
                contract.joint_damping,
            )
        )

    def _full_metadata(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._metadata(
            self.contracts.motion, VITA_CONTROLLER_JOINT_NAMES
        )

    def create_online_reference(self) -> Any:
        return create_online_smpl_reference(
            n_future_frames=10,
            policy_hz=self.bundle.policy_hz,
            device=self.device,
        )

    def warmup(self, iterations: int = 20) -> dict[str, Any]:
        if iterations < 1:
            raise ValueError("warmup iterations must be positive")
        reference = create_online_smpl_reference(
            n_future_frames=10,
            policy_hz=self.bundle.policy_hz,
            device=self.device,
            ground_history_frames=1,
        )
        pose = np.zeros((24, 7), dtype=np.float32)
        pose[:, 6] = 1.0
        received = time.time()
        for index in range(reference.capacity):
            if not reference.store_pico(
                pose,
                sample_time=index / self.bundle.policy_hz,
                current_time=received,
                frame_index=index,
            ):
                raise RuntimeError("failed to build warmup SMPL reference")
        default, _kp, _kd = self._full_metadata()
        state = VitaRobotState(
            joint_names=VITA_CONTROLLER_JOINT_NAMES,
            joint_pos=default,
            joint_vel=np.zeros(30, dtype=np.float32),
            root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], np.float32),
            root_ang_vel_b=np.zeros(3, dtype=np.float32),
        )
        timing = {"motion": [], "velocity": []}
        for _ in range(iterations):
            started = time.time_ns()
            self.step_velocity(state, (0.0, 0.0, 0.0))
            middle = time.time_ns()
            self.step_motion(
                state,
                reference,
                reference_yaw_alignment_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            ended = time.time_ns()
            timing["velocity"].append((middle - started) / 1e6)
            timing["motion"].append((ended - middle) / 1e6)
        self.reset()
        return {
            "no_robot_action": True,
            "providers": {
                "velocity": self.velocity_runner.session.get_providers(),
                "motion": self.motion_runner.session.get_providers(),
            },
            "velocity_steps": iterations,
            "motion_steps": iterations,
            "timing_ms": {key: max(value) for key, value in timing.items()},
        }

    def stand_step(self) -> PolicyStep:
        default, stiffness, damping = self._velocity_full_metadata()
        return PolicyStep(
            mode="stand",
            joint_names=VITA_CONTROLLER_JOINT_NAMES,
            raw_action=np.zeros(30, dtype=np.float32),
            target_joint_pos=default,
            joint_stiffness=stiffness,
            joint_damping=damping,
        )

    def step_velocity(
        self,
        state: VitaRobotState,
        command: VitaVelocityCommand | Sequence[float],
    ) -> PolicyStep:
        contract = self.contracts.velocity
        observation_started_wall_ns = time.time_ns()
        started_ns = observation_started_wall_ns
        observation = self.velocity_observation.build(
            state, command, self.last_velocity_action
        )
        self.last_observation = np.asarray(observation).copy()
        inference_started_wall_ns = time.time_ns()
        observation_done_ns = inference_started_wall_ns
        action = self.velocity_runner.run(observation)
        inference_finished_wall_ns = time.time_ns()
        inference_done_ns = inference_finished_wall_ns
        self.last_raw_action = action.copy()
        if not self.unified_velocity:
            clip_actions = float(contract.metadata["clip_actions"])
            action = np.clip(action, -clip_actions, clip_actions)
        defaults_28, kp_28, kd_28 = self._metadata(
            contract, contract.action_joint_names
        )
        target_28 = defaults_28 + contract.action_scale * action
        target, stiffness, damping = self._velocity_full_metadata()
        full_index = {
            name: index
            for index, name in enumerate(VITA_CONTROLLER_JOINT_NAMES)
        }
        for source_index, name in enumerate(contract.action_joint_names):
            target[full_index[name]] = target_28[source_index]
            stiffness[full_index[name]] = kp_28[source_index]
            damping[full_index[name]] = kd_28[source_index]
        np.copyto(self.last_velocity_action, action)
        action_mapping_finished_wall_ns = time.time_ns()
        self.last_timing_wall_ns = {
            "observation_started_wall_ns": observation_started_wall_ns,
            "inference_started_wall_ns": inference_started_wall_ns,
            "inference_finished_wall_ns": inference_finished_wall_ns,
            "action_mapping_finished_wall_ns": action_mapping_finished_wall_ns,
        }
        self.last_timing_ms = {
            "observation_ms": (observation_done_ns - started_ns) / 1e6,
            "onnx_ms": (inference_done_ns - observation_done_ns) / 1e6,
            "action_mapping_ms": (action_mapping_finished_wall_ns - inference_done_ns)
            / 1e6,
        }
        return PolicyStep(
            mode="velocity",
            joint_names=VITA_CONTROLLER_JOINT_NAMES,
            raw_action=action,
            target_joint_pos=target,
            joint_stiffness=stiffness,
            joint_damping=damping,
        )

    def _velocity_full_metadata(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The unified 28-action policy still exports all 30 physical joints.
        # Hold its omitted neck at its own asset default, not a Motion default.
        if self.unified_velocity:
            return self._metadata(
                self.contracts.velocity, VITA_CONTROLLER_JOINT_NAMES
            )
        # Official Velocity owns its 28 joints. Its fixed neck is defined by
        # the official action13 profile, never by the independent Motion model.
        target = np.zeros(30, dtype=np.float32)
        kp = np.full(30, 20.0, dtype=np.float32)
        kd = np.ones(30, dtype=np.float32)
        index = {name: i for i, name in enumerate(VITA_CONTROLLER_JOINT_NAMES)}
        contract = self.contracts.velocity
        for source, name in enumerate(contract.joint_names):
            dest = index[name]
            target[dest] = contract.default_joint_pos[source]
            kp[dest] = contract.joint_stiffness[source]
            kd[dest] = contract.joint_damping[source]
        return target, kp, kd

    def step_motion(
        self,
        state: VitaRobotState,
        reference: Any,
        *,
        reference_yaw_alignment_wxyz: Sequence[float] | None = None,
    ) -> PolicyStep:
        contract = self.contracts.motion
        observation_started_wall_ns = time.time_ns()
        started_ns = observation_started_wall_ns
        observation = self.motion_observation_builder.build(
            smpl_reference=reference,
            robot_root_quat_wxyz=state.root_quat_wxyz,
            robot_root_angvel_local=state.root_ang_vel_b,
            robot_dof_pos=state.select_joint_pos(contract.action_joint_names),
            robot_dof_vel=state.select_joint_vel(contract.action_joint_names),
            last_action=self.last_motion_action,
            reference_yaw_alignment_wxyz=reference_yaw_alignment_wxyz,
        )
        if not self.motion_policy_cuda and bool(
            getattr(observation, "is_cuda", False)
        ):
            observation.synchronize()
            if hasattr(observation, "array"):
                observation = observation.array.numpy().copy()
            else:
                observation = observation.detach().cpu().numpy()
        self.last_observation = (
            np.asarray(observation).copy()
            if not self.motion_policy_cuda
            else None
        )
        inference_started_wall_ns = time.time_ns()
        observation_done_ns = inference_started_wall_ns
        action = self.motion_runner.run(observation)
        inference_finished_wall_ns = time.time_ns()
        inference_done_ns = inference_finished_wall_ns
        # Preserve diagnostic observations on CUDA too, after inference so the
        # policy input stays on GPU. Account for this host snapshot separately.
        if self.motion_policy_cuda:
            if bool(getattr(observation, "is_cuda", False)):
                if hasattr(observation, "array"):
                    observation.synchronize()
                    self.last_observation = observation.array.numpy().copy()
                else:
                    self.last_observation = (
                        observation.detach().cpu().numpy().copy()
                    )
            else:
                self.last_observation = np.asarray(observation).copy()
        snapshot_done_ns = time.time_ns()
        self.last_raw_action = action.copy()
        action = np.clip(action, -100.0, 100.0)
        default, stiffness, damping = self._metadata(
            contract, contract.action_joint_names
        )
        target = default + contract.action_scale * action
        np.copyto(self.last_motion_action, action)
        action_mapping_finished_wall_ns = time.time_ns()
        self.last_timing_wall_ns = {
            "observation_started_wall_ns": observation_started_wall_ns,
            "inference_started_wall_ns": inference_started_wall_ns,
            "inference_finished_wall_ns": inference_finished_wall_ns,
            "action_mapping_finished_wall_ns": action_mapping_finished_wall_ns,
        }
        self.last_timing_ms = {
            "observation_ms": (observation_done_ns - started_ns) / 1e6,
            "onnx_ms": (inference_done_ns - observation_done_ns) / 1e6,
            "observation_snapshot_ms": (snapshot_done_ns - inference_done_ns)
            / 1e6,
            "action_mapping_ms": (action_mapping_finished_wall_ns - snapshot_done_ns)
            / 1e6,
        }
        return PolicyStep(
            mode="motion",
            joint_names=VITA_CONTROLLER_JOINT_NAMES,
            raw_action=action,
            target_joint_pos=target,
            joint_stiffness=stiffness,
            joint_damping=damping,
        )


__all__ = ["VitaDualPolicyRuntime"]
