#! /your_dir/miniconda3/envs/holomotion_deploy/bin/python
"""HoloMotion Policy Node

This module implements the HoloMotion policy node with formal-SMPL offline
references and local Pico-to-SMPL online references.
It handles neural network policy inference, motion sequence management, remote controller input,
and robot state coordination for humanoid behaviors including velocity tracking and motion tracking.

The policy node serves as the high-level decision maker that:
- Processes sensor observations and builds state representations
- Executes trained neural network policies for motion generation (velocity tracking and motion tracking)
- Manages multiple motion sequences (motion clips) loaded from offline files
- Handles remote controller input for motion selection
- Coordinates with the main control node for safe operation

Key Features:
- Dual policy support: velocity tracking and motion tracking
- Offline motion file loading (.npz format)
- Runtime policy switching with button controls
- Separate hyperparameters (kps, kds, action_scale, default_angles) for each model

Author: HoloMotion Team
License: See project LICENSE file
"""

import os
import time
import traceback
from collections import deque

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from omegaconf import OmegaConf
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, String
from unitree_hg.msg import LowState

from holomotion.src.motion_tracking.smpl_actor_observation import (
    SMPL_ACTOR_CURRENT_TERMS,
    SMPL_ACTOR_FUTURE_TERMS,
    smpl_motion_actor_observation_dim,
)
from holomotion.src.motion_tracking.smpl_reference import (
    SMPL_BODY_COMMAND_DIM,
)
from humanoid_policy.config import (
    DeploymentConfig,
    RobotConfig,
    format_config_for_log,
    validate_joint_name_permutation,
)
from humanoid_policy.cpu_affinity import (
    parse_cpu_affinity,
    set_thread_cpu_affinity,
)
from humanoid_policy.local_smpl_source import LocalPicoSmplSource
from humanoid_policy.motion_clip_library import (
    list_motion_clip_files,
    load_smpl_motion_clips,
    validate_loaded_smpl_motion_clip,
)
from humanoid_policy.obs_builder import PolicyObsBuilder
from humanoid_policy.observation_evaluator import PolicyObservationEvaluator
from humanoid_policy.onnx_policy import (
    load_dual_policy_bundle,
    read_onnx_metadata,
    warmup_motion_policy,
    warmup_policy_session,
)
from humanoid_policy.policy_runtime import PolicyRuntime
from humanoid_policy.reference_transport import (
    BestEffortReferencePublisher,
    decode_zmq_topic,
)
from humanoid_policy.shadow_transport import ShadowActionStateBridge
from humanoid_policy.smpl_motion_observation import (
    GpuSmplMotionObservationBuilder,
)
from humanoid_policy.smpl_reference import (
    GpuOfflineSmplReference,
    GpuOnlineSmplReference,
)
from humanoid_policy.utils.remote_controller_filter import RemoteController


def _coerce_config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


class HoloMotionPolicyNode(Node):
    """Main policy execution node for HoloMotion humanoid robot control with dual policy support.

    This node implements the high-level control logic for a humanoid robot capable of
    performing both velocity tracking and motion sequence execution. It supports two
    neural network policies and allows runtime switching between them.

    Key Features:
    - Dual neural network policy inference (velocity + motion) using ONNX Runtime
    - Runtime policy switching with A/B/Y button controls
    - Velocity tracking mode with joystick control
    - Motion tracking mode with motion clip sequence selection
    - Safety-aware state machine with motion prerequisites
    - Real-time observation processing and action generation

    Policy Control:
    - A button: Enable policy (defaults to velocity mode)
    - B button: Switch from velocity to motion mode
    - Y button: Switch from motion back to velocity mode

    Input Controls:
    - Motion mode:  B button (for mode switch)
    - Velocity mode: Y button (for mode switch) + Joystick +UP/DOWN/LEFT/RIGHT (for motion selection)

    State Machine:
    - ZERO_TORQUE: Initial safe state, waiting for activation
    - MOVE_TO_DEFAULT: Ready state, allows policy operations
    - Policy execution with mode switching
    - Emergency stop handling
    """

    def __init__(self):
        """Initialize the policy node with configuration, models, and ROS2 interfaces.

        Sets up the complete policy execution pipeline including:
        - Configuration loading from YAML file
        - Neural network model initialization
        - Motion data loading for all sequences
        - ROS2 publishers, subscribers, and timers
        - State machine initialization

        The node starts in a safe state and waits for proper robot state
        before allowing motion execution.
        """
        super().__init__("policy_node")
        self.runtime = PolicyRuntime(self, num_actions=29)
        self.observation_evaluator = PolicyObservationEvaluator(self)

        self.deployment_config = DeploymentConfig.from_node(self)
        self.robot_config = RobotConfig.load(
            self.deployment_config.robot_config_path
        )
        self.action_sink = self.deployment_config.action_sink.strip().lower()
        self.robot_state_source = (
            self.deployment_config.robot_state_source.strip().lower()
        )
        self._shadow_bridge = None
        self._shadow_state_stale_logged = False
        self._shadow_auto_velocity_started = False
        self._shadow_auto_motion_started = False
        self._shadow_current_kps = np.zeros(29, dtype=np.float32)
        self._shadow_current_kds = np.zeros(29, dtype=np.float32)
        if self.action_sink == "mujoco_shadow":
            self._shadow_bridge = ShadowActionStateBridge(
                action_uri=self.deployment_config.shadow_action_uri,
                state_uri=self.deployment_config.shadow_state_uri,
                joint_names=list(self.robot_config.policy_dof_order),
                max_state_age=self.deployment_config.shadow_max_state_age,
            )
        self.get_logger().info(
            "Deployment config:\n"
            + format_config_for_log(self.deployment_config.to_log_dict())
        )
        self.get_logger().info(
            "Robot config:\n"
            + format_config_for_log(self.robot_config.to_log_dict())
        )
        policy_freq = self.robot_config.policy_freq
        self.dt = 1.0 / policy_freq
        self.get_logger().info(
            f"Policy frequency set to: {policy_freq} Hz (dt = {self.dt:.4f} s)"
        )
        # Initialize basic parameters - will be updated after config loading
        self.actions_dim = 29  # Default value, will be updated from config
        self.real_dof_names = []  # Will be loaded from config
        self.current_motion_clip_index = 0  # Current motion clip index
        # Safety check related flags
        self.policy_enabled = False  # Controls whether policy is enabled
        # Robot state related flags
        self.robot_state_ready = False  # Marks whether MOVE_TO_DEFAULT state is received, allowing key operations
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_timers()
        # Initialize variables for dual policy
        self.velocity_policy_session = None
        self.motion_policy_session = None
        self.use_kv_cache = False
        self.motion_kv_cache = None
        self.motion_kv_input_name = None
        self.motion_kv_output_name = None
        self.motion_step_idx_input_name = None
        self.current_policy_mode = "velocity"
        self.velocity_config = None
        self.motion_config = None
        self.motion_frame_idx = 0
        self.n_motion_frames = 0

        self._latest_sender_timestamp = None
        self._last_consumed_reference_sequence = None
        self.reference_stream_active = False
        self.reference_source = (
            self.deployment_config.reference_source.strip().lower()
        )
        self.max_data_age = self.deployment_config.max_data_age
        self.stale_data_warning_count = 0
        self.last_poll_time = None
        self.reference_zmq_uri = self.deployment_config.reference_zmq_uri
        self.reference_zmq_topic = self.deployment_config.reference_zmq_topic
        self.reference_zmq_mode = self.deployment_config.reference_zmq_mode
        self.reference_zmq_conflate = (
            self.deployment_config.reference_zmq_conflate
        )
        self.zmq_jitter_delay_frames = (
            self.deployment_config.zmq_jitter_delay_frames
        )
        self.enable_teleop_reference = (
            self.deployment_config.enable_teleop_reference
        )
        self.motion_rope_max_seq_len = (
            self.deployment_config.motion_rope_max_seq_len
        )
        self.motion_rope_reset_margin = (
            self.deployment_config.motion_rope_reset_margin
        )
        self._cpu_affinity_main_str = self.deployment_config.cpu_affinity_main
        self._cpu_affinity_zmq_sub_str = (
            self.deployment_config.cpu_affinity_zmq_sub
        )
        self._local_smpl_source = None
        self._offline_smpl_reference = None
        self._offline_smpl_clip_index = None
        self.motion_reference_representation = "smpl"
        self._reference_telemetry = None
        self._pending_reference_telemetry = None
        self.timing_debug_enabled = self.deployment_config.timing_debug_enabled
        self.timing_debug_log_interval_sec = (
            self.deployment_config.timing_debug_log_interval_sec
        )
        self.timing_debug_log_per_loop = (
            self.deployment_config.timing_debug_log_per_loop
        )
        self._timing_debug_last_log_time = None
        self._timing_debug_samples = deque(maxlen=500)
        self._npz_replay_frame_index = None
        if self.reference_source == "pico_local":
            self.get_logger().info(
                "SMPL reference source: local Pico/XRoboToolkit"
            )
        else:
            self.get_logger().info("SMPL reference source: offline NPZ")

        if self.deployment_config.reference_telemetry_enabled:
            telemetry_affinity = parse_cpu_affinity(
                self.deployment_config.cpu_affinity_reference_telemetry
            )
            self._reference_telemetry = BestEffortReferencePublisher(
                uri=self.deployment_config.reference_telemetry_uri,
                topic=decode_zmq_topic(
                    self.deployment_config.reference_telemetry_topic
                ),
                mode=self.deployment_config.reference_telemetry_mode,
                publish_hz=self.deployment_config.reference_telemetry_hz,
                deadline_pause_sec=(
                    self.deployment_config.reference_telemetry_deadline_pause_sec
                ),
                logger=self.get_logger(),
                cpu_affinity=(
                    telemetry_affinity if telemetry_affinity else None
                ),
            )
            self._reference_telemetry.start()

        self.dof_names_ref_motion = []
        self.num_actions = 29
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)

        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.runtime.resize_actions(self.num_actions)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)

        self._lowstate_msg = None
        self.target_dof_pos_real = None
        self.motion_in_progress = False
        self._keybody_indices_by_term_name = {}
        self._live_smpl_reference = None

    def _is_live_smpl_ready_for_motion(self) -> bool:
        """Return whether the local Pico-to-SMPL stream is policy-ready."""
        if self._live_smpl_reference is None:
            return False
        return self._live_smpl_reference.is_ready_for_motion(
            enable_teleop_reference=getattr(
                self, "enable_teleop_reference", True
            ),
            delay_frames=getattr(self, "zmq_jitter_delay_frames", 0),
        )

    def _runtime_state(self):
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return None
        return runtime.state

    @property
    def policy_enabled(self):
        state = self._runtime_state()
        return (
            state.policy_enabled
            if state is not None
            else getattr(self, "_policy_enabled", False)
        )

    @policy_enabled.setter
    def policy_enabled(self, value):
        state = self._runtime_state()
        if state is None:
            self._policy_enabled = bool(value)
        else:
            state.policy_enabled = bool(value)

    @property
    def robot_state_ready(self):
        state = self._runtime_state()
        return (
            state.robot_state_ready
            if state is not None
            else getattr(self, "_robot_state_ready", False)
        )

    @robot_state_ready.setter
    def robot_state_ready(self, value):
        state = self._runtime_state()
        if state is None:
            self._robot_state_ready = bool(value)
        else:
            state.robot_state_ready = bool(value)

    @property
    def current_policy_mode(self):
        state = self._runtime_state()
        return (
            state.current_policy_mode
            if state is not None
            else getattr(self, "_current_policy_mode", "velocity")
        )

    @current_policy_mode.setter
    def current_policy_mode(self, value):
        state = self._runtime_state()
        if state is None:
            self._current_policy_mode = str(value)
        else:
            state.current_policy_mode = str(value)

    @property
    def reference_stream_active(self):
        state = self._runtime_state()
        return (
            state.reference_stream_active
            if state is not None
            else getattr(self, "_reference_stream_active", False)
        )

    @reference_stream_active.setter
    def reference_stream_active(self, value):
        state = self._runtime_state()
        if state is None:
            self._reference_stream_active = bool(value)
        else:
            state.reference_stream_active = bool(value)

    @property
    def motion_frame_idx(self):
        state = self._runtime_state()
        return (
            state.motion_frame_idx
            if state is not None
            else getattr(self, "_motion_frame_idx", 0)
        )

    @motion_frame_idx.setter
    def motion_frame_idx(self, value):
        state = self._runtime_state()
        if state is None:
            self._motion_frame_idx = int(value)
        else:
            state.motion_frame_idx = int(value)

    @property
    def motion_step_idx(self):
        state = self._runtime_state()
        return (
            state.motion_step_idx
            if state is not None
            else getattr(self, "_motion_step_idx", 0)
        )

    @motion_step_idx.setter
    def motion_step_idx(self, value):
        state = self._runtime_state()
        if state is None:
            self._motion_step_idx = int(value)
        else:
            state.motion_step_idx = int(value)

    @property
    def motion_in_progress(self):
        state = self._runtime_state()
        return (
            state.motion_in_progress
            if state is not None
            else getattr(self, "_motion_in_progress", False)
        )

    @motion_in_progress.setter
    def motion_in_progress(self, value):
        state = self._runtime_state()
        if state is None:
            self._motion_in_progress = bool(value)
        else:
            state.motion_in_progress = bool(value)

    @property
    def current_motion_clip_index(self):
        state = self._runtime_state()
        return (
            state.current_motion_clip_index
            if state is not None
            else getattr(self, "_current_motion_clip_index", 0)
        )

    @current_motion_clip_index.setter
    def current_motion_clip_index(self, value):
        state = self._runtime_state()
        if state is None:
            self._current_motion_clip_index = int(value)
        else:
            state.current_motion_clip_index = int(value)

    @property
    def vx(self):
        state = self._runtime_state()
        return state.vx if state is not None else getattr(self, "_vx", 0.0)

    @vx.setter
    def vx(self, value):
        state = self._runtime_state()
        if state is None:
            self._vx = float(value)
        else:
            state.vx = float(value)

    @property
    def vy(self):
        state = self._runtime_state()
        return state.vy if state is not None else getattr(self, "_vy", 0.0)

    @vy.setter
    def vy(self, value):
        state = self._runtime_state()
        if state is None:
            self._vy = float(value)
        else:
            state.vy = float(value)

    @property
    def vyaw(self):
        state = self._runtime_state()
        return state.vyaw if state is not None else getattr(self, "_vyaw", 0.0)

    @vyaw.setter
    def vyaw(self, value):
        state = self._runtime_state()
        if state is None:
            self._vyaw = float(value)
        else:
            state.vyaw = float(value)

    @property
    def actions_onnx(self):
        state = self._runtime_state()
        return (
            state.actions_onnx
            if state is not None
            else getattr(self, "_actions_onnx", None)
        )

    @actions_onnx.setter
    def actions_onnx(self, value):
        state = self._runtime_state()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if state is None:
            self._actions_onnx = arr.copy()
        else:
            state.actions_onnx = arr.copy()

    @property
    def target_dof_pos_onnx(self):
        state = self._runtime_state()
        return (
            state.target_dof_pos_onnx
            if state is not None
            else getattr(self, "_target_dof_pos_onnx", None)
        )

    @target_dof_pos_onnx.setter
    def target_dof_pos_onnx(self, value):
        state = self._runtime_state()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if state is None:
            self._target_dof_pos_onnx = arr.copy()
        else:
            state.target_dof_pos_onnx = arr.copy()

    @property
    def target_dof_pos_real(self):
        state = self._runtime_state()
        return (
            state.target_dof_pos_real
            if state is not None
            else getattr(self, "_target_dof_pos_real", None)
        )

    @target_dof_pos_real.setter
    def target_dof_pos_real(self, value):
        state = self._runtime_state()
        arr = (
            None
            if value is None
            else np.asarray(value, dtype=np.float32).reshape(-1)
        )
        if state is None:
            self._target_dof_pos_real = None if arr is None else arr.copy()
        else:
            state.target_dof_pos_real = None if arr is None else arr.copy()

    def _init_obs_buffers(self):
        """Initialize velocity observation and validate the SMPL Actor schema."""
        # Use velocity model's parameters for velocity obs_builder
        self.velocity_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.velocity_dof_names_onnx,
            default_angles_onnx=self.velocity_default_angles_onnx,
            evaluator=self.observation_evaluator,
            obs_policy_cfg=self.observation_evaluator._get_policy_atomic_obs_list(
                self.velocity_config
            ),
        )

        motion_obs_cfg = (
            self.observation_evaluator._get_policy_atomic_obs_list(
                self.motion_config
            )
        )
        motion_term_names = [
            str(next(iter(term))) for term in motion_obs_cfg["atomic_obs_list"]
        ]
        has_smpl_terms = any(
            name.startswith("actor_smpl_") for name in motion_term_names
        )
        has_robot_ref_terms = any(
            name.startswith("actor_ref_") for name in motion_term_names
        )
        if has_robot_ref_terms or not has_smpl_terms:
            raise ValueError(
                "Released deployment requires a pure-SMPL Actor schema; "
                "robot reference terms are unsupported"
            )
        expected_smpl_terms = [
            *SMPL_ACTOR_CURRENT_TERMS,
            "actor_projected_gravity",
            "actor_rel_robot_root_ang_vel",
            "actor_dof_pos",
            "actor_dof_vel",
            "actor_last_action",
            *SMPL_ACTOR_FUTURE_TERMS,
        ]
        if motion_term_names != expected_smpl_terms:
            raise ValueError(
                "Motion Actor schema order does not match the released "
                f"SMPL CUDA observation: expected {expected_smpl_terms}, "
                f"got {motion_term_names}"
            )
        self.motion_reference_representation = "smpl"

        # Motion observation is CUDA-only in the released path. Do not create
        # a second CPU observation builder with representation-specific
        # getters; the legacy robot builder remains available in its own code.
        self.motion_obs_builder = None

        # The released node installs its SMPL provider in
        # _init_gpu_motion_observation().  Do not instantiate the legacy
        # robot-qpos queue here: keeping that queue active would create a
        # second reference representation inside the policy process.
        # Set default obs_builder to velocity mode
        self.obs_builder = self.velocity_obs_builder

    def _init_gpu_motion_observation(self) -> None:
        policy_fps = float(1.0 / self.dt)
        self.gpu_motion_obs_builder = None
        requested = (
            self.deployment_config.motion_observation_backend.strip().lower()
        )
        configured_device = str(self.robot_config.device)
        if requested != "warp":
            raise RuntimeError(
                "pure-SMPL deployment has one observation path: Warp CUDA"
            )
        try:
            self.gpu_motion_obs_builder = GpuSmplMotionObservationBuilder(
                n_future_frames=int(self.n_fut_frames),
                default_dof_pos=self.motion_default_angles_onnx,
                policy_hz=policy_fps,
                device=configured_device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SMPL Warp CUDA observation backend '{requested}' failed"
            ) from exc
        self.motion_observation_backend_active = "warp-cuda"
        if self.reference_source == "pico_local":
            self._live_smpl_reference = GpuOnlineSmplReference(
                n_future_frames=int(self.n_fut_frames),
                policy_hz=policy_fps,
                device=configured_device,
            )
        else:
            self._live_smpl_reference = None
        self.get_logger().info(
            "Motion Observation backend: shared Warp CUDA kernels "
            "with direct TensorRT I/O binding"
        )

    def load_policy(self):
        """Load both velocity and motion policy models using ONNX Runtime."""
        self.get_logger().info(
            "Loading dual policies "
            f"(backend={self.deployment_config.inference_backend})..."
        )
        motion_max_context_len = int(
            self.motion_config.get("algo", {})
            .get("config", {})
            .get("num_steps_per_env", 0)
        )
        bundle = load_dual_policy_bundle(
            package_share_dir=get_package_share_directory("humanoid_control"),
            velocity_model_folder=self.robot_config.velocity_tracking_model_folder,
            motion_model_folder=self.robot_config.motion_tracking_model_folder,
            intra_op_threads=self.robot_config.onnx_intra_op_threads,
            motion_max_context_len=motion_max_context_len,
            inference_backend=self.deployment_config.inference_backend,
        )

        self.velocity_policy_session = bundle.velocity_session
        self.motion_policy_session = bundle.motion_session
        self.velocity_onnx_path = bundle.velocity_onnx_path
        self.motion_onnx_path = bundle.motion_onnx_path
        self.velocity_input_name = bundle.velocity_input_name
        self.velocity_output_name = bundle.velocity_output_name
        self.get_logger().info(
            f"Velocity policy loaded from {self.velocity_onnx_path} using: "
            f"{self.velocity_policy_session.get_providers()}"
        )
        self.get_logger().info(
            f"Motion policy loaded from {self.motion_onnx_path} using: "
            f"{self.motion_policy_session.get_providers()}"
        )

        motion_io = bundle.motion_io
        self.motion_input_name = motion_io.input_name
        self.motion_output_name = motion_io.output_name
        self.motion_kv_input_name = motion_io.kv_input_name
        self.motion_kv_output_name = motion_io.kv_output_name
        self.motion_kv_shape = motion_io.kv_shape
        self.motion_step_idx_input_name = motion_io.step_idx_input_name
        self.motion_kv_dtype = motion_io.kv_dtype

        self.get_logger().info(
            f"Velocity policy - Input: {self.velocity_input_name}, "
            f"Output: {self.velocity_output_name}"
        )
        self.get_logger().info(
            f"Motion policy - Input: {self.motion_input_name}, "
            f"Output: {self.motion_output_name}"
        )
        self.get_logger().info("Initializing KV-Cache for Motion Policy...")

        for node in self.motion_policy_session.get_inputs():
            self.get_logger().info(
                f"Motion policy input: name={node.name}, shape={node.shape}, type={node.type}"
            )
        for node in self.motion_policy_session.get_outputs():
            self.get_logger().info(
                f"Motion policy output: name={node.name}, shape={node.shape}, type={node.type}"
            )
        if (
            self.motion_kv_input_name is not None
            and self.motion_kv_output_name is None
        ):
            self.get_logger().warn(
                "Motion policy has past_key_values input but no present_key_values output was found. "
                "KV cache will not update and transformer performance will degrade."
            )

        kv_cache = bundle.motion_kv_cache
        self.motion_kv_cache = kv_cache.cache
        self.motion_max_context_len = motion_max_context_len
        self.motion_model_context_len = kv_cache.model_context_len
        self.motion_effective_context_len = kv_cache.effective_context_len
        self.use_kv_cache = kv_cache.enabled
        if self.use_kv_cache:
            self.get_logger().info(
                f"KV-Cache initialized with shape {kv_cache.shape} "
                f"(model_ctx={self.motion_model_context_len}, "
                f"effective_ctx={self.motion_effective_context_len})"
            )
        else:
            self.get_logger().warn(
                "No KV-Cache inputs found in Motion Policy model!"
            )
        self.get_logger().info("Dual policies loaded successfully")

    def load_model_config(self):
        """Load config.yaml from both velocity and motion model folders."""
        # Load velocity model config
        velocity_model_folder = (
            self.robot_config.velocity_tracking_model_folder
        )
        velocity_config_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            velocity_model_folder,
        )
        # Try different config file names for velocity model
        config_names = ["config.yaml"]
        velocity_config_path = None

        for config_name in config_names:
            potential_path = os.path.join(velocity_config_dir, config_name)
            if os.path.exists(potential_path):
                velocity_config_path = potential_path
                break

        if velocity_config_path is None:
            raise FileNotFoundError(
                f"No config file found in {velocity_config_dir}. Tried: {config_names}"
            )

        self.get_logger().info(
            f"Loading velocity model config from {velocity_config_path}"
        )
        self.velocity_config = OmegaConf.load(velocity_config_path)

        # Load motion model config
        motion_model_folder = self.robot_config.motion_tracking_model_folder
        motion_config_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            motion_model_folder,
        )
        # Try different config file names for motion model
        motion_config_path = None

        for config_name in config_names:
            potential_path = os.path.join(motion_config_dir, config_name)
            if os.path.exists(potential_path):
                motion_config_path = potential_path
                break

        if motion_config_path is None:
            raise FileNotFoundError(
                f"No config file found in {motion_config_dir}. Tried: {config_names}"
            )

        self.get_logger().info(
            f"Loading motion model config from {motion_config_path}"
        )
        self.motion_config = OmegaConf.load(motion_config_path)
        self.actor_place_holder_ndim = (
            self.observation_evaluator._find_actor_place_holder_ndim()
        )
        self.n_fut_frames = int(self.motion_config.obs.n_fut_frames)
        self.torso_body_idx = self.motion_config.robot.body_names.index(
            "torso_link"
        )
        self.get_logger().info("Both model configs loaded successfully")

    def _warmup_motion_policy(self, num_iters: int = 2) -> None:
        if self.velocity_policy_session is not None:
            velocity_obs_dim = None
            try:
                velocity_obs_dim = int(
                    self.velocity_obs_builder.build_policy_obs().shape[0]
                )
            except Exception:
                velocity_obs_dim = None
            try:
                velocity_iterations = warmup_policy_session(
                    session=self.velocity_policy_session,
                    input_name=self.velocity_input_name,
                    output_name=self.velocity_output_name,
                    obs_dim=velocity_obs_dim,
                    num_iters=num_iters,
                )
                self.get_logger().info(
                    f"[Warmup] Velocity policy warmup completed ({velocity_iterations} iterations)."
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"[Warmup] Velocity policy warmup skipped: {exc}"
                )

        if self.motion_policy_session is None:
            return

        motion_obs_dim = smpl_motion_actor_observation_dim(
            SMPL_BODY_COMMAND_DIM,
            int(self.n_fut_frames),
        )

        try:
            iterations = warmup_motion_policy(
                session=self.motion_policy_session,
                input_name=self.motion_input_name,
                output_name=self.motion_output_name,
                kv_input_name=self.motion_kv_input_name,
                kv_output_name=self.motion_kv_output_name,
                step_idx_input_name=self.motion_step_idx_input_name,
                use_kv_cache=self.use_kv_cache,
                kv_cache=self.motion_kv_cache,
                kv_shape=self.motion_kv_shape,
                kv_dtype=self.motion_kv_dtype,
                obs_dim=motion_obs_dim,
                num_iters=num_iters,
            )
            if self.motion_kv_cache is not None:
                self.motion_kv_cache.fill(0)
            self.motion_step_idx = 0
            self.get_logger().info(
                f"[Warmup] Motion policy warmup completed ({iterations} iterations, KV cache kept clean)."
            )
        except Exception as exc:
            if self.motion_kv_cache is not None:
                self.motion_kv_cache.fill(0)
            self.motion_step_idx = 0
            self.get_logger().warn(
                f"[Warmup] Motion policy warmup skipped: {exc}"
            )

    def update_config_parameters(self):
        """Update configuration parameters from loaded configs."""
        # Check if both models have the same basic parameters
        velocity_actions_dim = self.velocity_config.get("robot", {}).get(
            "actions_dim", 29
        )
        motion_actions_dim = self.motion_config.get("robot", {}).get(
            "actions_dim", 29
        )

        velocity_dof_names = self.velocity_config.get("robot", {}).get(
            "dof_names", []
        )
        motion_dof_names = self.motion_config.get("robot", {}).get(
            "dof_names", []
        )

        expected_dof_names = list(self.robot_config.policy_dof_order)
        expected_actions_dim = len(expected_dof_names)
        for label, actions_dim, dof_names in (
            ("velocity config", velocity_actions_dim, velocity_dof_names),
            ("motion config", motion_actions_dim, motion_dof_names),
        ):
            if int(actions_dim) != expected_actions_dim:
                raise ValueError(
                    f"{label} actions_dim={actions_dim} does not match "
                    f"Robot policy_dof_order size {expected_actions_dim}"
                )
            validate_joint_name_permutation(
                list(dof_names),
                expected_dof_names,
                label=f"{label} dof_names",
                expected_label="Robot policy_dof_order",
            )

        # Update basic parameters
        self.actions_dim = expected_actions_dim
        self.real_dof_names = expected_dof_names
        self.dof_names_ref_motion = expected_dof_names.copy()
        self.num_actions = len(self.dof_names_ref_motion)

        # Update arrays with correct sizes
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)
        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)

        self.get_logger().info(
            f"Updated config parameters: actions_dim={self.actions_dim}, "
            f"dof_names={len(self.real_dof_names)}"
        )

    def load_motion_data(self):
        """Load motion clip data from .npz files."""
        motion_clips_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            self.robot_config.motion_clip_dir,
        )

        self.get_logger().info(
            f"Looking for motion clip data in: {motion_clips_dir}"
        )
        self.get_logger().info(
            f"Directory exists: {os.path.exists(motion_clips_dir)}"
        )

        if not os.path.exists(motion_clips_dir):
            self.get_logger().warn(
                f"Motion clips directory not found: {motion_clips_dir}"
            )
            return

        # Only collect .npz files
        motion_clip_files = list_motion_clip_files(motion_clips_dir)
        self.get_logger().info(
            f"Found {len(motion_clip_files)} motion clip files (.npz): {motion_clip_files}"
        )
        if not motion_clip_files:
            self.get_logger().warn(
                f"No motion clip files (.npz) found in directory: {motion_clips_dir}"
            )
            return

        # Load each .npz file
        self.all_motion_data = load_smpl_motion_clips(
            motion_clips_dir, motion_clip_files
        )
        self.motion_file_names = list(motion_clip_files)

        if not self.all_motion_data:
            self.get_logger().error("Failed to load any motion clip files")
            return

        # Initialize with the first motion clip
        self.current_motion_clip_index = 0
        self._load_current_motion()

        self.get_logger().info(
            f"Loaded {len(self.all_motion_data)} motion clips successfully"
        )
        self.get_logger().info(
            f"Current motion clip: {self.motion_file_names[self.current_motion_clip_index]}"
        )

    def _load_current_motion(self):
        """Load the current selected motion clip data."""
        if not self.all_motion_data:
            return

        self.motion_frame_idx = 0
        motion_data = self.all_motion_data[self.current_motion_clip_index]
        current_motion = validate_loaded_smpl_motion_clip(motion_data)
        policy_fps = float(1.0 / self.dt)
        if abs(current_motion.target_fps - policy_fps) > 1.0e-5:
            raise ValueError(
                "HoloSMPL clip must match the motion policy rate: "
                f"clip={current_motion.target_fps:g} Hz, "
                f"policy={policy_fps:g} Hz"
            )
        self._offline_smpl_reference = GpuOfflineSmplReference(
            current_motion.human_pose_aa,
            current_motion.human_root_trans,
            n_future_frames=int(self.n_fut_frames),
            policy_hz=policy_fps,
            device=str(self.robot_config.device),
        )
        self._offline_smpl_clip_index = self.current_motion_clip_index
        self.n_motion_frames = current_motion.n_frames
        self.motion_in_progress = True
        self.get_logger().info(
            f"Loaded SMPL motion clip {self.current_motion_clip_index}: "
            f"{self.motion_file_names[self.current_motion_clip_index]} "
            f"({self.n_motion_frames} frames)"
        )

    def _setup_subscribers(self):
        """Set up ROS2 subscribers for robot state and remote controller input."""
        self.remote_controller = RemoteController()
        if self.robot_state_source == "mujoco_shadow":
            self.low_state_sub = None
            self.robot_state_sub = None
            self.get_logger().warn(
                "[SHADOW] Robot ROS subscribers are disabled; proprioception "
                "comes only from workstation MuJoCo."
            )
            return
        self.low_state_sub = self.create_subscription(
            LowState,
            self.robot_config.lowstate_topic,
            self._low_state_callback,
            QoSProfile(depth=10),
        )

        # Add robot_state topic subscription
        self.robot_state_sub = self.create_subscription(
            String,
            "/robot_state",
            self._robot_state_callback,
            QoSProfile(depth=10),
        )

    def _setup_publishers(self):
        """Set up ROS2 publishers for action commands and status information."""
        if self.action_sink == "mujoco_shadow":
            self.action_pub = None
            self.kps_pub = None
            self.kds_pub = None
            self.policy_mode_pub = None
            self._action_msg = Float32MultiArray()
            self._policy_mode_msg = String()
            self._last_policy_mode_msg_data = None
            self._last_policy_mode_publish_time = 0.0
            self._policy_mode_publish_interval_sec = 1.0
            self.get_logger().warn(
                "[SHADOW] Robot ROS action publishers are disabled; policy "
                "targets go only to workstation MuJoCo."
            )
            return
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.robot_config.action_topic,
            QoSProfile(depth=10),
        )
        # Add publishers for kps and kds parameters
        self.kps_pub = self.create_publisher(
            Float32MultiArray,
            "/humanoid/kps",
            QoSProfile(depth=10),
        )
        self.kds_pub = self.create_publisher(
            Float32MultiArray,
            "/humanoid/kds",
            QoSProfile(depth=10),
        )
        # Add publisher for policy mode status
        self.policy_mode_pub = self.create_publisher(
            String,
            "policy_mode",
            QoSProfile(depth=10),
        )
        self._action_msg = Float32MultiArray()
        self._policy_mode_msg = String()
        self._last_policy_mode_msg_data = None
        self._last_policy_mode_publish_time = 0.0
        self._policy_mode_publish_interval_sec = 1.0

    def _setup_timers(self):
        """Set up ROS2 timer for main execution loop."""
        # Create a one-time timer to call setup after ROS2 initialization
        self._setup_started = False
        self._setup_timer = self.create_timer(0.1, self._delayed_setup)
        self.create_timer(self.dt, self.run)

    def _delayed_setup(self):
        """Call setup after ROS2 initialization is complete."""
        if self._setup_started:
            return
        self._setup_started = True
        self._setup_timer.cancel()
        self.get_logger().info("Starting policy node setup...")
        try:
            self.setup()
            self._setup_completed = True
            self.get_logger().info("Policy node setup completed successfully")
        except Exception as e:
            self._setup_error = e
            self.get_logger().error(f"Policy node setup failed: {e}")
            self.get_logger().error(traceback.format_exc())

    def _robot_state_callback(self, msg: String):
        """Handle robot state messages for safety coordination.

        Processes robot state updates from the main control node to ensure
        safe operation. Button operations are only allowed when the robot
        is in MOVE_TO_DEFAULT state.

        Args:
            msg: String message containing robot state information
                Valid states: ZERO_TORQUE, MOVE_TO_DEFAULT, EMERGENCY_STOP, POLICY
        """
        robot_state = msg.data
        self.runtime.set_robot_state(robot_state)

    def _low_state_callback(self, ls_msg: LowState):
        """Forward low-level state and remote-controller input to runtime."""
        self.runtime.handle_low_state(ls_msg)

    def run(self):
        """Main execution loop for policy inference and action publication."""
        if self.robot_state_source == "mujoco_shadow":
            self._poll_shadow_state()
        control_start = time.perf_counter()
        self.runtime.run()
        if self.robot_state_source == "mujoco_shadow":
            self._maybe_start_shadow_runtime()
        self._flush_reference_telemetry(
            control_elapsed=time.perf_counter() - control_start
        )

    def _poll_shadow_state(self) -> None:
        bridge = self._shadow_bridge
        if bridge is None:
            return
        try:
            sample = bridge.receive_latest()
        except Exception as exc:
            self._lowstate_msg = None
            self.robot_state_ready = False
            self.get_logger().error(f"[SHADOW] Invalid MuJoCo state: {exc}")
            return
        now = time.monotonic()
        age = (
            float("inf") if sample is None else now - sample.received_monotonic
        )
        if sample is None or age > self.deployment_config.shadow_max_state_age:
            self._lowstate_msg = None
            self.robot_state_ready = False
            if not self._shadow_state_stale_logged:
                self.get_logger().warn(
                    "[SHADOW] Workstation MuJoCo state missing/stale; policy output "
                    "is paused and no action is sent."
                )
                self._shadow_state_stale_logged = True
            return
        self._lowstate_msg = sample.lowstate
        self.robot_state_ready = True
        if self._shadow_state_stale_logged:
            self.get_logger().info(
                "[SHADOW] Workstation MuJoCo state recovered."
            )
            self._shadow_state_stale_logged = False

    def _maybe_start_shadow_runtime(self) -> None:
        # MuJoCo may begin streaming state before the delayed model setup has
        # loaded the released velocity policy and its metadata.  Do not enable
        # either policy until all model/default-angle buffers exist.
        if not getattr(self, "_setup_completed", False):
            return
        if self._lowstate_msg is None or not self.robot_state_ready:
            return
        if not self._shadow_auto_velocity_started:
            self.runtime.enable_velocity_policy()
            self._shadow_auto_velocity_started = True
            self.get_logger().info(
                "[SHADOW] Started released velocity policy against MuJoCo "
                "state; real G1 action path remains disabled."
            )
        if (
            self.deployment_config.shadow_auto_start_motion
            and self.current_policy_mode == "velocity"
            and self._is_live_smpl_ready_for_motion()
            and self.runtime.switch_to_motion_mode()
        ):
            self._shadow_auto_motion_started = True
            self.get_logger().info(
                "[SHADOW] Live Pico queue ready; switched MuJoCo twin to "
                "motion policy automatically."
            )

    def _poll_local_smpl_reference(self):
        """Consume one local Pico frame on this policy tick."""
        source = self._local_smpl_source
        if source is None:
            return
        result = source.step()
        if result is None:
            return
        reference_value, frame_index, _sample_meta = result
        self._npz_replay_frame_index = int(frame_index)
        current_time = time.time()
        stored = self._live_smpl_reference.store_pico(
            reference_value,
            sample_time=float(_sample_meta["timestamp_ns"]) * 1.0e-9,
            current_time=current_time,
            frame_index=frame_index,
        )
        if not stored:
            self.get_logger().error("Local Pico reference frame is invalid")
            return
        if self._reference_telemetry is not None:
            self._pending_reference_telemetry = (
                reference_value,
                int(frame_index),
                _sample_meta,
            )
        if self._is_live_smpl_ready_for_motion():
            self.runtime.mark_vr_queue_ready()

    def _flush_reference_telemetry(self, *, control_elapsed: float) -> None:
        """Hand the newest raw Pico frame to the best-effort publisher."""
        del control_elapsed
        publisher = self._reference_telemetry
        pending = self._pending_reference_telemetry
        self._pending_reference_telemetry = None
        if publisher is None or pending is None:
            return
        try:
            body_poses, frame_index, sample_meta = pending
            publisher.submit_pico(
                body_poses,
                frame_index=frame_index,
                sample_meta=sample_meta,
            )
        except Exception as exc:
            self._reference_telemetry = None
            self.get_logger().error(
                f"[Telemetry] Pico snapshot disabled after failure: {exc}"
            )

    def _apply_onnx_metadata(self):
        """Apply PD/scale/defaults from ONNX metadata as authoritative values.
        Load separate metadata for velocity and motion models.
        """
        # Load velocity model metadata
        velocity_meta = read_onnx_metadata(self.velocity_onnx_path)
        self.velocity_dof_names_onnx = velocity_meta["joint_names"]
        self.velocity_action_scale_onnx = velocity_meta["action_scale"].astype(
            np.float32
        )
        self.velocity_kps_onnx = velocity_meta["kps"].astype(np.float32)
        self.velocity_kds_onnx = velocity_meta["kds"].astype(np.float32)
        self.velocity_default_angles_onnx = velocity_meta[
            "default_joint_pos"
        ].astype(np.float32)

        # Load motion model metadata
        motion_meta = read_onnx_metadata(self.motion_onnx_path)
        self.motion_dof_names_onnx = motion_meta["joint_names"]
        self.motion_action_scale_onnx = motion_meta["action_scale"].astype(
            np.float32
        )
        self.motion_kps_onnx = motion_meta["kps"].astype(np.float32)
        self.motion_kds_onnx = motion_meta["kds"].astype(np.float32)
        self.motion_default_angles_onnx = motion_meta[
            "default_joint_pos"
        ].astype(np.float32)
        configured_rope_max_seq_len = int(
            getattr(self.deployment_config, "motion_rope_max_seq_len", 0) or 0
        )
        artifact_rope_max_seq_len = motion_meta.get("rope_max_seq_len", None)
        if configured_rope_max_seq_len > 0:
            if (
                artifact_rope_max_seq_len is not None
                and int(artifact_rope_max_seq_len)
                != configured_rope_max_seq_len
            ):
                self.get_logger().warn(
                    "motion_rope_max_seq_len profile override "
                    f"({configured_rope_max_seq_len}) differs from motion ONNX "
                    f"artifact metadata ({artifact_rope_max_seq_len})."
                )
            self.motion_rope_max_seq_len = configured_rope_max_seq_len
        elif artifact_rope_max_seq_len is not None:
            self.motion_rope_max_seq_len = int(artifact_rope_max_seq_len)
            self.get_logger().info(
                "Motion RoPE max sequence length loaded from ONNX artifact: "
                f"{self.motion_rope_max_seq_len}"
            )
        elif self.motion_step_idx_input_name is not None:
            raise RuntimeError(
                "Motion policy uses step_idx/KV-cache but ONNX artifact does not "
                "provide rope_max_seq_len metadata. Re-export the motion policy "
                "with current code, or set motion_rope_max_seq_len explicitly in "
                "the launch profile for a legacy artifact."
            )
        else:
            self.motion_rope_max_seq_len = 0

        # Use velocity model metadata as default (for backward compatibility)
        self.dof_names_onnx = self.velocity_dof_names_onnx
        self.action_scale_onnx = self.velocity_action_scale_onnx
        self.kps_onnx = self.velocity_kps_onnx
        self.kds_onnx = self.velocity_kds_onnx
        self.default_angles_onnx = self.velocity_default_angles_onnx
        self.default_angles_dict = {
            name: float(self.default_angles_onnx[idx])
            for idx, name in enumerate(self.dof_names_onnx)
        }

    def _build_dof_mappings(self):
        # Map ONNX <-> MJCF for control
        validate_joint_name_permutation(
            list(self.velocity_dof_names_onnx),
            list(self.real_dof_names),
            label="Velocity ONNX joint_names",
            expected_label="Robot policy_dof_order",
        )
        validate_joint_name_permutation(
            list(self.motion_dof_names_onnx),
            list(self.real_dof_names),
            label="Motion ONNX joint_names",
            expected_label="Robot policy_dof_order",
        )

        # Build mappings for velocity model
        self.velocity_onnx_to_real = [
            self.velocity_dof_names_onnx.index(name)
            for name in self.real_dof_names
        ]
        self.velocity_kps_real = self.velocity_kps_onnx[
            self.velocity_onnx_to_real
        ].astype(np.float32)
        self.velocity_kds_real = self.velocity_kds_onnx[
            self.velocity_onnx_to_real
        ].astype(np.float32)

        # Build mappings for motion model
        self.motion_onnx_to_real = [
            self.motion_dof_names_onnx.index(name)
            for name in self.real_dof_names
        ]
        self.motion_kps_real = self.motion_kps_onnx[
            self.motion_onnx_to_real
        ].astype(np.float32)
        self.motion_kds_real = self.motion_kds_onnx[
            self.motion_onnx_to_real
        ].astype(np.float32)

        # Use velocity model mappings as default (for backward compatibility)
        self.onnx_to_real = self.velocity_onnx_to_real
        self.kps_real = self.velocity_kps_real
        self.kds_real = self.velocity_kds_real
        self.default_angles_mu = self.velocity_default_angles_onnx[
            self.velocity_onnx_to_real
        ].astype(np.float32)
        self.action_scale_mu = self.velocity_action_scale_onnx[
            self.velocity_onnx_to_real
        ].astype(np.float32)

        self.observation_evaluator.initialize_observation_state()

        # Publish kps and kds parameters (use velocity as default)
        self._publish_control_params()

    def _publish_control_params(self):
        """Publish kps and kds control parameters based on current policy mode.

        Called during initialization and mode switching to ensure control node
        receives the correct parameters for the current policy mode.
        """
        try:
            # Use appropriate parameters based on current policy mode
            if self.current_policy_mode == "motion":
                current_kps = self.motion_kps_real
                current_kds = self.motion_kds_real
            else:  # velocity mode
                current_kps = self.velocity_kps_real
                current_kds = self.velocity_kds_real

            if self.action_sink == "mujoco_shadow":
                self._shadow_current_kps = np.asarray(
                    current_kps, dtype=np.float32
                ).copy()
                self._shadow_current_kds = np.asarray(
                    current_kds, dtype=np.float32
                ).copy()
                self.get_logger().debug(
                    f"[SHADOW] Selected {self.current_policy_mode} PD gains"
                )
                return

            # Publish kps
            kps_msg = Float32MultiArray()
            kps_msg.data = current_kps.tolist()
            self.kps_pub.publish(kps_msg)

            # Publish kds
            kds_msg = Float32MultiArray()
            kds_msg.data = current_kds.tolist()
            self.kds_pub.publish(kds_msg)

            self.get_logger().debug(
                f"Published control parameters ({self.current_policy_mode} mode): "
                f"kps={len(current_kps)}, kds={len(current_kds)}"
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to publish control parameters: {e}"
            )

    def _publish_policy_mode(self, force: bool = False):
        """Publish current policy mode status."""
        if self.policy_mode_pub is None:
            return
        try:
            data = f"{self.current_policy_mode}_{'enabled' if self.policy_enabled else 'disabled'}"
            now = time.time()
            if (
                not force
                and data == self._last_policy_mode_msg_data
                and now - self._last_policy_mode_publish_time
                < self._policy_mode_publish_interval_sec
            ):
                return
            self._policy_mode_msg.data = data
            self.policy_mode_pub.publish(self._policy_mode_msg)
            self._last_policy_mode_msg_data = data
            self._last_policy_mode_publish_time = now
        except Exception as e:
            self.get_logger().error(f"Failed to publish policy mode: {e}")

    def _timing_ms(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _record_timing_sample(self, sample: dict):
        if not getattr(self, "timing_debug_enabled", False):
            return
        self._timing_debug_samples.append(sample)
        if getattr(self, "timing_debug_log_per_loop", False):
            self.get_logger().info(
                "[Timing] "
                f"loop_total={sample['loop_total_ms']:.2f}ms "
                f"io={sample['io_ms']:.2f}ms "
                f"policy_total={sample['policy_total_ms']:.2f}ms "
                f"reference_kinematics={sample['reference_kinematics_ms']:.2f}ms "
                f"obs={sample['obs_ms']:.2f}ms "
                f"onnx={sample['onnx_ms']:.2f}ms "
                f"post={sample['post_ms']:.2f}ms"
            )

        now = time.time()
        last = getattr(self, "_timing_debug_last_log_time", None)
        interval = float(getattr(self, "timing_debug_log_interval_sec", 5.0))
        if last is None:
            self._timing_debug_last_log_time = now
            return
        if now - last < interval:
            return
        if len(self._timing_debug_samples) == 0:
            self._timing_debug_last_log_time = now
            return

        keys = [
            "loop_total_ms",
            "io_ms",
            "policy_total_ms",
            "reference_kinematics_ms",
            "obs_ms",
            "onnx_ms",
            "post_ms",
        ]
        stats = {}
        for key in keys:
            vals = np.array(
                [float(s.get(key, 0.0)) for s in self._timing_debug_samples],
                dtype=np.float64,
            )
            stats[key] = (float(np.mean(vals)), float(np.max(vals)))
        self.get_logger().info(
            "[Timing-Agg] "
            + " ".join(
                f"{key}=mean:{stats[key][0]:.2f}ms/max:{stats[key][1]:.2f}ms"
                for key in keys
            )
            + f" n={len(self._timing_debug_samples)}"
        )
        self._timing_debug_samples.clear()
        self._timing_debug_last_log_time = now

    def _publish_action_target(self, target_dof_pos_real):
        """Publish action commands prepared by PolicyRuntime."""
        if target_dof_pos_real is None:
            return
        target = np.asarray(target_dof_pos_real, dtype=np.float32)
        if not np.isfinite(target).all():
            self.get_logger().error(
                "Action contains NaN/Inf; refusing to publish"
            )
            return
        if self.action_sink == "mujoco_shadow":
            if self._shadow_bridge is None or self._lowstate_msg is None:
                return
            self._shadow_bridge.publish_action(
                target_q=target,
                kp=self._shadow_current_kps,
                kd=self._shadow_current_kds,
                mode=self.current_policy_mode,
            )
            return
        action_msg = self._action_msg
        action_msg.data = target.tolist()

        # Check for NaN values
        self.action_pub.publish(action_msg)

    def setup(self):
        """Set up the evaluator by loading all required components."""
        main_affinity = parse_cpu_affinity(
            getattr(self, "_cpu_affinity_main_str", "") or ""
        )
        if main_affinity and set_thread_cpu_affinity(main_affinity):
            self.get_logger().info(
                f"[Policy] main thread pinned to CPUs {main_affinity}"
            )
        self.load_model_config()  # Load config first
        self.update_config_parameters()  # Update parameters from config
        self.load_policy()  # Then load policies
        self._apply_onnx_metadata()
        self._init_obs_buffers()
        if (
            self.enable_teleop_reference
            and self.reference_source != "pico_local"
        ):
            raise ValueError(
                "Released online SMPL tracking requires "
                "reference_source=pico_local; robot "
                "reference_qpos streams are not accepted as policy input"
            )
        self._build_dof_mappings()
        self._init_gpu_motion_observation()
        self._warmup_motion_policy()
        self.observation_evaluator._init_keybody_indices_cache()
        if self.reference_source == "pico_local":
            self._local_smpl_source = LocalPicoSmplSource(
                logger=self.get_logger(),
                max_source_age=self.max_data_age,
            )
            self._local_smpl_source.start()
        # Online releases consume only the live SMPL provider. Offline clips
        # are loaded only for reference_source=offline, so online startup does
        # not parse or precompute an unrelated second reference.
        if self.reference_source == "offline":
            self.load_motion_data()
        self.get_logger().info("Synchronous root-only policy setup completed")

    def destroy_node(self):
        try:
            if getattr(self, "_local_smpl_source", None) is not None:
                self._local_smpl_source.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_reference_telemetry", None) is not None:
                self._reference_telemetry.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_shadow_bridge", None) is not None:
                self._shadow_bridge.close()
        except Exception:
            pass

        super().destroy_node()


def main():
    """Main entry point for the policy node."""
    rclpy.init()
    policy_node = HoloMotionPolicyNode()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
