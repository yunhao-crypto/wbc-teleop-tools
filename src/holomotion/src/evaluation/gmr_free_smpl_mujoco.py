"""Fail-closed pure-SMPL MuJoCo rollout for released HoloMotion policies.

This module is intentionally separate from the legacy paired Robot-reference
evaluator.  Its input contract contains only formal SMPL pose/root arrays and
its policy observation is built by the same Warp CUDA implementation used by
the released ROS2 deployment.
"""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime
import torch
import yaml
from loguru import logger

from holomotion.src.evaluation.eval_mujoco_sim2sim import MujocoEvaluator
from holomotion.src.motion_tracking.reference_observation import (
    derive_reference_kinematics_torch,
)
from holomotion.src.motion_tracking.robot_pose_grounding import (
    RobotPoseGroundingRuntime,
)
from holomotion.src.motion_tracking.robot_skeleton_adapter import (
    RobotSkeletonAdapterSpec,
)
from holomotion.src.motion_tracking.smpl_reference import (
    SMPL_BODY_COMMAND_MODE,
    SmplReferenceBuilder,
)
from holomotion.src.motion_tracking.smpl_skeleton_target import (
    SMPL_SEMANTIC_JOINT_INDICES,
    SmplSkeletonTargetBuilder,
)
from holomotion.src.training.reference_fk import URDFParser
from holomotion.src.utils.torch_utils import quat_apply

_DEPLOYMENT_SRC = (
    Path(__file__).resolve().parents[3]
    / "deployment/unitree_g1_ros2_29dof/src"
)
if str(_DEPLOYMENT_SRC) not in sys.path:
    sys.path.insert(0, str(_DEPLOYMENT_SRC))

from humanoid_policy.obs_builder.obs_builder import (  # noqa: E402
    PolicyObsBuilder,
)
from humanoid_policy.onnx_policy import (  # noqa: E402
    read_onnx_metadata,
    run_with_cuda_observation,
)
from humanoid_policy.smpl_motion_observation import (  # noqa: E402
    GpuSmplMotionObservationBuilder,
)
from humanoid_policy.smpl_reference import (  # noqa: E402
    GpuOfflineSmplReference,
)

FORBIDDEN_INPUT_PREFIXES = ("ref_", "robot_", "q_ref")
FORBIDDEN_OUTPUT_PREFIXES = ("ref_", "q_ref")
FORBIDDEN_INPUT_NAMES = frozenset(
    {
        "dof_pos",
        "dof_vel",
        "dof_vels",
        "reference_qpos",
        "global_translation",
        "global_rotation_quat",
        "global_velocity",
        "global_angular_velocity",
    }
)
REQUIRED_INPUT_NAMES = frozenset({"human_pose_aa", "human_root_trans"})
VELOCITY_ACTOR_TERMS = (
    "actor_velocity_command",
    "actor_projected_gravity",
    "actor_rel_robot_root_ang_vel",
    "actor_dof_pos",
    "actor_dof_vel",
    "actor_last_action",
)


def validate_named_dof_matrix(
    values: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
    expected_joint_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Validate the explicit joint-order contract attached to a DOF matrix."""

    array = np.asarray(values)
    names = tuple(str(name) for name in joint_names)
    expected = tuple(str(name) for name in expected_joint_names)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(
            "DOF matrix/name shape mismatch: "
            f"matrix={array.shape}, names={len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("DOF joint names contain duplicates")
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(
            "Rollout/MuJoCo joint schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return names


def load_velocity_actor_obs_config(path: str | Path) -> dict[str, Any]:
    """Load the released velocity Actor observation schema fail-closed."""

    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    obs = payload.get("obs", {})
    context_length = int(obs.get("context_length", 0))
    if context_length <= 0:
        raise ValueError("Velocity config has no positive obs.context_length")
    atomic_list = (
        obs.get("obs_groups", {}).get("unified", {}).get("atomic_obs_list", [])
    )
    by_name: dict[str, dict[str, Any]] = {}
    for entry in atomic_list:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Invalid velocity atomic observation: {entry}")
        name, term_config = next(iter(entry.items()))
        by_name[str(name)] = dict(term_config or {})

    actor_terms = (
        payload.get("modules", {})
        .get("actor", {})
        .get("obs_schema", {})
        .get("flattened_obs", {})
        .get("terms", [])
    )
    actor_terms = tuple(str(name).rsplit("/", 1)[-1] for name in actor_terms)
    if actor_terms != VELOCITY_ACTOR_TERMS:
        raise ValueError(
            "Velocity Actor schema mismatch: "
            f"expected {VELOCITY_ACTOR_TERMS}, got {actor_terms}"
        )

    resolved: list[dict[str, dict[str, Any]]] = []
    for name in actor_terms:
        if name not in by_name:
            raise KeyError(f"Velocity Actor term is missing: {name}")
        term_config = by_name[name]
        history_length = term_config.get("history_length", context_length)
        if isinstance(history_length, str):
            if history_length != "${obs.context_length}":
                raise ValueError(
                    f"Unsupported history interpolation for {name}: "
                    f"{history_length}"
                )
            history_length = context_length
        term_config["history_length"] = int(history_length)
        if int(term_config["history_length"]) != context_length:
            raise ValueError(
                f"Velocity term {name} does not use the Actor context length"
            )
        resolved.append({name: term_config})
    return {"atomic_obs_list": resolved}


@dataclass(frozen=True)
class PureSmplClip:
    human_pose_aa: np.ndarray
    human_root_trans: np.ndarray
    motion_fps: float
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.human_pose_aa.shape[0])


def _decode_scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar metadata, got {array.shape}")
    result = array.reshape(()).item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def load_pure_smpl_clip(path: str | Path) -> PureSmplClip:
    """Load a formal-SMPL clip and reject every Robot-reference field."""

    clip_path = Path(path).expanduser().resolve()
    with np.load(clip_path, allow_pickle=False) as payload:
        names = set(payload.files)
        forbidden = sorted(
            name
            for name in names
            if name in FORBIDDEN_INPUT_NAMES
            or name.startswith(FORBIDDEN_INPUT_PREFIXES)
        )
        if forbidden:
            raise ValueError(
                "GMR-free MuJoCo input contains Robot-reference fields: "
                + ", ".join(forbidden)
            )
        missing = sorted(REQUIRED_INPUT_NAMES - names)
        if missing:
            raise KeyError(f"Pure SMPL clip is missing fields: {missing}")
        pose = np.asarray(payload["human_pose_aa"], dtype=np.float32)
        root = np.asarray(payload["human_root_trans"], dtype=np.float32)
        fps = float(
            _decode_scalar(payload["motion_fps"])
            if "motion_fps" in names
            else 50.0
        )
        metadata: dict[str, Any] = {}
        for name in sorted(
            names - {"human_pose_aa", "human_root_trans", "motion_fps"}
        ):
            value = np.asarray(payload[name])
            if value.size == 1:
                metadata[name] = _decode_scalar(value)

    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"human_pose_aa must be [T,72], got {pose.shape}")
    if root.shape != (pose.shape[0], 3):
        raise ValueError(
            "human_root_trans must be [T,3] with matching T, got "
            f"{root.shape} versus {pose.shape}"
        )
    if pose.shape[0] <= 0:
        raise ValueError("Pure SMPL clip is empty")
    if not np.isfinite(pose).all() or not np.isfinite(root).all():
        raise ValueError("Pure SMPL clip contains non-finite values")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"motion_fps must be positive, got {fps}")
    return PureSmplClip(pose, root, fps, metadata)


def save_pure_smpl_clip(
    source_path: str | Path,
    output_path: str | Path,
    *,
    frame_count: int | None = None,
) -> Path:
    """Strip a paired audit bundle down to the formal-SMPL contract."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        pose = np.asarray(payload["human_pose_aa"], dtype=np.float32)
        root = np.asarray(payload["human_root_trans"], dtype=np.float32)
        fps = np.float32(
            _decode_scalar(payload["motion_fps"])
            if "motion_fps" in payload.files
            else 50.0
        )
        if frame_count is None and "effective_frame_count" in payload.files:
            frame_count = int(_decode_scalar(payload["effective_frame_count"]))
        count = pose.shape[0] if frame_count is None else int(frame_count)
        if count <= 0 or count > pose.shape[0]:
            raise ValueError(
                f"frame_count must be in [1,{pose.shape[0]}], got {count}"
            )
        scalar_names = (
            "dataset_id",
            "clip_id",
            "source_clip_id",
            "motion_key",
        )
        extras = {
            name: np.asarray(payload[name])
            for name in scalar_names
            if name in payload.files
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        human_pose_aa=pose[:count],
        human_root_trans=root[:count],
        motion_fps=fps,
        schema_version=np.asarray("gmr_free_formal_smpl_clip_v1"),
        effective_frame_count=np.asarray(count, dtype=np.int64),
        **extras,
    )
    load_pure_smpl_clip(output)
    return output


class GmrFreeSmplMujocoEvaluator(MujocoEvaluator):
    """MuJoCo evaluator driven only by formal SMPL and actual robot state."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.pure_clip: PureSmplClip | None = None
        self.smpl_reference = None
        self.smpl_reference_cpu = None
        self.smpl_target_positions = None
        self._gmr_free_obs_builder = None
        self._gmr_free_obs_to_onnx = np.zeros(0, dtype=np.int64)
        self._gmr_free_robot_scale = 1.0
        self._gmr_free_human_scale = 1.0
        self._gmr_free_root_kinematics = None
        self._gmr_free_reset_profile = str(
            config.get("gmr_free_reset_profile", "training_canonical_default")
        )
        if self._gmr_free_reset_profile not in {
            "training_canonical_default",
            "deployment_stand",
        }:
            raise ValueError(
                "Unsupported GMR-free reset profile: "
                f"{self._gmr_free_reset_profile}"
            )
        self._gmr_free_settle_seconds = float(
            config.get("gmr_free_settle_seconds", 0.0)
        )
        if self._gmr_free_settle_seconds < 0.0:
            raise ValueError("gmr_free_settle_seconds must be non-negative")
        velocity_onnx = config.get("gmr_free_velocity_onnx_path", None)
        velocity_config = config.get("gmr_free_velocity_config_path", None)
        self._gmr_free_velocity_onnx_path = (
            None
            if velocity_onnx in {None, ""}
            else Path(str(velocity_onnx)).expanduser().resolve()
        )
        self._gmr_free_velocity_config_path = (
            None
            if velocity_config in {None, ""}
            else Path(str(velocity_config)).expanduser().resolve()
        )
        self._gmr_free_velocity_warmup_seconds = float(
            config.get("gmr_free_velocity_warmup_seconds", 0.0)
        )
        if self._gmr_free_velocity_warmup_seconds < 0.0:
            raise ValueError(
                "gmr_free_velocity_warmup_seconds must be non-negative"
            )
        if (
            self._gmr_free_velocity_warmup_seconds > 0.0
            and self._gmr_free_reset_profile != "deployment_stand"
        ):
            raise ValueError(
                "Velocity warm-up requires reset profile deployment_stand"
            )
        self._reference_yaw_alignment_wxyz = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self._visual_smpl_target_positions = None
        self._settle_summary: dict[str, float | int] = {}
        self._velocity_warmup_summary: dict[str, Any] = {}
        self._deployment_default_max_abs_diff = 0.0
        self._velocity_policy_session = None
        self._velocity_obs_builder = None
        self._velocity_policy_input_name = None
        self._velocity_policy_output_name = None
        self._velocity_stage_active = False
        self._velocity_actions_onnx = np.zeros(0, dtype=np.float32)
        self._velocity_action_scale_onnx = np.zeros(0, dtype=np.float32)
        self._velocity_default_angles_onnx = np.zeros(0, dtype=np.float32)
        self._velocity_action_to_onnx = np.zeros(0, dtype=np.int64)
        self._velocity_num_actions = 0
        self._velocity_dof_names_onnx: tuple[str, ...] = ()
        self._velocity_action_dof_names_onnx: tuple[str, ...] = ()
        self._velocity_kps_mu = np.zeros(0, dtype=np.float32)
        self._velocity_kds_mu = np.zeros(0, dtype=np.float32)
        self._velocity_mu_to_onnx: list[int] = []
        self._velocity_onnx_to_mu: list[int] = []

    def load_motion_data(self) -> None:
        path = self.config.get("motion_npz_path", None)
        if path is None:
            raise ValueError("GMR-free evaluator requires motion_npz_path")
        clip = load_pure_smpl_clip(path)
        if abs(float(clip.motion_fps) - 1.0 / self.policy_dt) > 1.0e-5:
            raise ValueError(
                "SMPL clip rate must match policy rate: "
                f"clip={clip.motion_fps}, policy={1.0 / self.policy_dt}"
            )
        self.pure_clip = clip
        self.n_motion_frames = clip.frame_count
        self.smpl_reference = GpuOfflineSmplReference(
            clip.human_pose_aa,
            clip.human_root_trans,
            n_future_frames=int(self.n_fut_frames),
            policy_hz=float(clip.motion_fps),
            device=f"cuda:{int(self.config.get('gpu_id', 0))}",
        )
        builder = SmplReferenceBuilder(
            SMPL_BODY_COMMAND_MODE,
            fps=float(clip.motion_fps),
        )
        self.smpl_reference_cpu = builder.build(
            clip.human_pose_aa,
            clip.human_root_trans,
        )
        qpos = torch.zeros((clip.frame_count, 36), dtype=torch.float32)
        qpos[:, :3] = self.smpl_reference_cpu.root_pos
        qpos[:, 3:7] = self.smpl_reference_cpu.root_rot_xyzw[:, [3, 0, 1, 2]]
        self._gmr_free_root_kinematics = derive_reference_kinematics_torch(
            qpos.contiguous(), fps=float(clip.motion_fps)
        )
        skeleton = SmplSkeletonTargetBuilder(fps=float(clip.motion_fps))
        self._gmr_free_human_scale = float(skeleton.nominal_scale)
        semantic_cfg = self.config.robot.motion.semantic_skeleton
        semantic_mapping = {
            str(key): value for key, value in semantic_cfg.items()
        }
        spec = RobotSkeletonAdapterSpec.from_mapping(semantic_mapping)
        self._gmr_free_robot_scale = float(spec.nominal_scale)
        p23 = self.smpl_reference_cpu.body_command[:, :69].reshape(
            clip.frame_count, 23, 3
        )
        root_position = self.smpl_reference_cpu.root_pos
        root_rotation = self.smpl_reference_cpu.root_rot_xyzw[:, [3, 0, 1, 2]]
        all_local = torch.cat(
            (torch.zeros_like(root_position[:, None]), p23), dim=1
        )
        semantic_indices = torch.as_tensor(
            SMPL_SEMANTIC_JOINT_INDICES, dtype=torch.long
        )
        semantic_local = all_local.index_select(1, semantic_indices)
        semantic_rotation = root_rotation[:, None].expand(
            -1, semantic_local.shape[1], -1
        )
        semantic_world = root_position[:, None] + quat_apply(
            semantic_rotation.reshape(-1, 4),
            semantic_local.reshape(-1, 3),
        ).reshape_as(semantic_local)
        self.smpl_target_positions = semantic_world.numpy().astype(np.float32)
        logger.info(
            "Loaded fail-closed pure SMPL clip: "
            "path={}, frames={}, keys=no-ref",
            path,
            clip.frame_count,
        )

    def _init_obs_buffers(self) -> None:
        actor_terms = [
            str(next(iter(term)))
            for term in self._get_policy_atomic_obs_list()
        ]
        if not any(name.startswith("actor_smpl_") for name in actor_terms):
            raise ValueError("Actor schema has no actor_smpl_* terms")
        leaked = [
            name for name in actor_terms if name.startswith("actor_ref_")
        ]
        if leaked:
            raise ValueError(
                f"Actor schema still contains Robot-ref terms: {leaked}"
            )
        if self.num_actions != len(self.dof_names_onnx):
            raise ValueError(
                "Pure-SMPL motion observation requires one action per robot "
                "DoF"
            )
        # Actor q/qdot/default terms follow the configured action-joint order,
        # not Isaac/MuJoCo's state-joint order.  The two orders differ on K1.
        self._gmr_free_obs_to_onnx = self.action_to_onnx.copy()
        actor_default_dof_pos = self.default_angles_onnx[
            self._gmr_free_obs_to_onnx
        ]
        self._gmr_free_obs_builder = GpuSmplMotionObservationBuilder(
            n_future_frames=int(self.n_fut_frames),
            default_dof_pos=actor_default_dof_pos,
            policy_hz=float(1.0 / self.policy_dt),
            device=f"cuda:{int(self.config.get('gpu_id', 0))}",
        )
        if self._gmr_free_velocity_warmup_seconds > 0.0:
            self._init_velocity_policy()

    def _init_velocity_policy(self) -> None:
        if self._gmr_free_velocity_onnx_path is None:
            raise ValueError("Velocity warm-up requires a velocity ONNX path")
        if self._gmr_free_velocity_config_path is None:
            raise ValueError(
                "Velocity warm-up requires a velocity config path"
            )
        if not self._gmr_free_velocity_onnx_path.is_file():
            raise FileNotFoundError(self._gmr_free_velocity_onnx_path)
        if not self._gmr_free_velocity_config_path.is_file():
            raise FileNotFoundError(self._gmr_free_velocity_config_path)

        metadata = read_onnx_metadata(str(self._gmr_free_velocity_onnx_path))
        joint_names = tuple(str(name) for name in metadata["joint_names"])
        action_joint_names = tuple(
            str(name)
            for name in metadata.get("action_joint_names", joint_names)
        )
        if set(joint_names) != set(self.mjcf_dof_names):
            missing = sorted(set(self.mjcf_dof_names) - set(joint_names))
            extra = sorted(set(joint_names) - set(self.mjcf_dof_names))
            raise ValueError(
                "Velocity/MuJoCo joint schema mismatch: "
                f"missing={missing}, extra={extra}"
            )
        missing_action_joints = sorted(
            set(action_joint_names) - set(joint_names)
        )
        if missing_action_joints:
            raise ValueError(
                "Velocity action joints are absent from its state schema: "
                f"{missing_action_joints}"
            )
        if len(set(action_joint_names)) != len(action_joint_names):
            raise ValueError("Velocity action_joint_names contains duplicates")
        self._velocity_mu_to_onnx = [
            self.mjcf_dof_names.index(name) for name in joint_names
        ]
        self._velocity_onnx_to_mu = [
            joint_names.index(name) for name in self.mjcf_dof_names
        ]
        self._velocity_action_to_onnx = np.asarray(
            [joint_names.index(name) for name in action_joint_names],
            dtype=np.int64,
        )
        self._velocity_num_actions = len(action_joint_names)
        self._velocity_dof_names_onnx = joint_names
        self._velocity_action_dof_names_onnx = action_joint_names
        self._velocity_action_scale_onnx = np.asarray(
            metadata["action_scale"], dtype=np.float32
        )
        self._velocity_default_angles_onnx = np.asarray(
            metadata["default_joint_pos"], dtype=np.float32
        )
        velocity_kps_onnx = np.asarray(metadata["kps"], dtype=np.float32)
        velocity_kds_onnx = np.asarray(metadata["kds"], dtype=np.float32)
        state_arrays = {
            "default_joint_pos": self._velocity_default_angles_onnx,
            "kps": velocity_kps_onnx,
            "kds": velocity_kds_onnx,
        }
        for name, value in state_arrays.items():
            if value.shape != (len(joint_names),):
                raise ValueError(
                    f"Velocity metadata {name} has shape {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"Velocity metadata {name} is non-finite")
        if self._velocity_action_scale_onnx.shape != (
            self._velocity_num_actions,
        ):
            raise ValueError(
                "Velocity metadata action_scale has shape "
                f"{self._velocity_action_scale_onnx.shape}"
            )
        if not np.isfinite(self._velocity_action_scale_onnx).all():
            raise ValueError("Velocity metadata action_scale is non-finite")
        self._velocity_kps_mu = velocity_kps_onnx[self._velocity_onnx_to_mu]
        self._velocity_kds_mu = velocity_kds_onnx[self._velocity_onnx_to_mu]
        self._velocity_actions_onnx = np.zeros(
            self._velocity_num_actions, dtype=np.float32
        )
        self._velocity_obs_builder = PolicyObsBuilder(
            dof_names_onnx=joint_names,
            default_angles_onnx=self._velocity_default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=load_velocity_actor_obs_config(
                self._gmr_free_velocity_config_path
            ),
        )

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = list(self.policy_session.get_providers())
        self._velocity_policy_session = onnxruntime.InferenceSession(
            str(self._gmr_free_velocity_onnx_path),
            sess_options=options,
            providers=providers,
        )
        velocity_inputs = self._velocity_policy_session.get_inputs()
        velocity_outputs = self._velocity_policy_session.get_outputs()
        if len(velocity_inputs) != 1 or len(velocity_outputs) != 1:
            raise ValueError(
                "Velocity policy must have one input and one output, got "
                f"{len(velocity_inputs)}/{len(velocity_outputs)}"
            )
        self._velocity_policy_input_name = velocity_inputs[0].name
        self._velocity_policy_output_name = velocity_outputs[0].name
        if velocity_outputs[0].shape != [1, self._velocity_num_actions]:
            raise ValueError(
                "Velocity policy output shape is "
                f"{velocity_outputs[0].shape}, expected "
                f"[1,{self._velocity_num_actions}]"
            )
        logger.info(
            "Loaded released velocity policy: onnx={}, config={}, "
            "input_shape={}, providers={}",
            self._gmr_free_velocity_onnx_path,
            self._gmr_free_velocity_config_path,
            velocity_inputs[0].shape,
            self._velocity_policy_session.get_providers(),
        )

    def _get_obs_dof_pos(self, joint_names=None) -> np.ndarray:
        if not self._velocity_stage_active:
            return super()._get_obs_dof_pos(joint_names)
        position_onnx = self.robot_dof_pos[self._velocity_mu_to_onnx]
        relative = position_onnx - self._velocity_default_angles_onnx
        if joint_names is not None:
            indices = [
                self._velocity_dof_names_onnx.index(str(name))
                for name in joint_names
            ]
            relative = relative[indices]
        return relative.astype(np.float32)

    def _get_obs_dof_vel(self, joint_names=None) -> np.ndarray:
        if not self._velocity_stage_active:
            return super()._get_obs_dof_vel(joint_names)
        velocity = self.robot_dof_vel[self._velocity_mu_to_onnx]
        if joint_names is not None:
            indices = [
                self._velocity_dof_names_onnx.index(str(name))
                for name in joint_names
            ]
            velocity = velocity[indices]
        return velocity.astype(np.float32)

    def _get_obs_actor_dof_pos(self) -> np.ndarray:
        if not self._velocity_stage_active:
            return super()._get_obs_actor_dof_pos()
        return self._get_obs_dof_pos(self._velocity_action_dof_names_onnx)

    def _get_obs_actor_dof_vel(self) -> np.ndarray:
        if not self._velocity_stage_active:
            return super()._get_obs_actor_dof_vel()
        return self._get_obs_dof_vel(self._velocity_action_dof_names_onnx)

    def _get_obs_last_action(self) -> np.ndarray:
        if not self._velocity_stage_active:
            return super()._get_obs_last_action()
        return self._velocity_actions_onnx

    def _get_obs_velocity_command(self) -> np.ndarray:
        if not self._velocity_stage_active:
            return super()._get_obs_velocity_command()
        return np.zeros(4, dtype=np.float32)

    def _build_pose_grounding_runtime(self) -> RobotPoseGroundingRuntime:
        semantic_cfg = self.config.robot.motion.semantic_skeleton
        semantic_mapping = {
            str(key): value for key, value in semantic_cfg.items()
        }
        spec = RobotSkeletonAdapterSpec.from_mapping(semantic_mapping)
        repo_root = Path(__file__).resolve().parents[3]
        urdf_path = repo_root / str(self.config.robot.asset.urdf_file)
        urdf_dof_names = tuple(URDFParser(str(urdf_path)).parse()[7])
        onnx_name_to_index = {
            str(name): index for index, name in enumerate(self.dof_names_onnx)
        }
        missing = sorted(set(urdf_dof_names) - set(onnx_name_to_index))
        if missing:
            raise ValueError(
                f"ONNX metadata is missing grounding joints: {missing}"
            )
        q_default_urdf = torch.as_tensor(
            [
                self.default_angles_onnx[onnx_name_to_index[name]]
                for name in urdf_dof_names
            ],
            dtype=torch.float32,
        )
        return RobotPoseGroundingRuntime(
            robot_file_path=str(urdf_path),
            adapter_spec=spec,
            expected_dof_names=urdf_dof_names,
            q_default_urdf=q_default_urdf,
            device="cpu",
            root_clearance_m=0.03,
            max_root_height_adjustment_m=0.30,
        ).eval()

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        w, x, y, z = np.asarray(quat, dtype=np.float64)
        return float(
            np.arctan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        )

    @staticmethod
    def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
        half = 0.5 * float(yaw)
        return np.asarray(
            [np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32
        )

    def _capture_deployment_yaw_alignment(self) -> None:
        if self.smpl_reference_cpu is None:
            raise RuntimeError("Pure SMPL reference was not loaded")
        reference_wxyz = self.smpl_reference_cpu.root_rot_xyzw[0, [3, 0, 1, 2]]
        robot_wxyz = self.robot_global_bodylink_rot[self.root_body_idx]
        yaw_offset = self._yaw_from_quat_wxyz(
            robot_wxyz
        ) - self._yaw_from_quat_wxyz(reference_wxyz.numpy())
        self._reference_yaw_alignment_wxyz = self._yaw_quat_wxyz(yaw_offset)

    def _build_visual_smpl_target(self) -> np.ndarray:
        if self.smpl_target_positions is None:
            raise RuntimeError("SMPL semantic target was not built")
        target = np.asarray(self.smpl_target_positions, dtype=np.float64)
        pelvis = target[:, 0]
        scale = self._gmr_free_robot_scale / self._gmr_free_human_scale
        yaw = self._yaw_from_quat_wxyz(self._reference_yaw_alignment_wxyz)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        robot_pelvis = np.asarray(
            self.robot_global_bodylink_pos[self.root_body_idx],
            dtype=np.float64,
        )
        pelvis_motion = (pelvis - pelvis[0]) * scale
        aligned_pelvis = robot_pelvis[None] + pelvis_motion @ rotation.T
        relative = (target - pelvis[:, None]) * scale
        return (aligned_pelvis[:, None] + relative @ rotation.T).astype(
            np.float32
        )

    def _deployment_stand_parameters_mu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the exact ROS2 MoveToDefault targets/gains in MuJoCo order."""

        config_path = _DEPLOYMENT_SRC / "config/g1_29dof_holomotion.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        names = tuple(str(name) for name in payload["complete_dof_order"])
        kp = np.asarray(payload["kp"], dtype=np.float64)
        kd = np.asarray(payload["kd"], dtype=np.float64)
        defaults_by_name = payload["default_joint_angles"]
        if len(names) != len(kp) or len(names) != len(kd):
            raise ValueError(
                "Deployment MoveToDefault names/kp/kd length mismatch: "
                f"{len(names)}/{len(kp)}/{len(kd)}"
            )
        if set(names) != set(self.mjcf_dof_names):
            missing = sorted(set(self.mjcf_dof_names) - set(names))
            extra = sorted(set(names) - set(self.mjcf_dof_names))
            raise ValueError(
                "Deployment/MuJoCo joint schema mismatch: "
                f"missing={missing}, extra={extra}"
            )
        index_by_name = {name: index for index, name in enumerate(names)}
        defaults_mu = np.asarray(
            [float(defaults_by_name[name]) for name in self.mjcf_dof_names],
            dtype=np.float64,
        )
        kp_mu = np.asarray(
            [kp[index_by_name[name]] for name in self.mjcf_dof_names],
            dtype=np.float64,
        )
        kd_mu = np.asarray(
            [kd[index_by_name[name]] for name in self.mjcf_dof_names],
            dtype=np.float64,
        )
        if not (
            np.isfinite(defaults_mu).all()
            and np.isfinite(kp_mu).all()
            and np.isfinite(kd_mu).all()
        ):
            raise ValueError(
                "Deployment MoveToDefault parameters contain non-finite values"
            )
        self._deployment_default_max_abs_diff = float(
            np.max(np.abs(defaults_mu - self.default_angles_mu))
        )
        return defaults_mu, kp_mu, kd_mu

    def _settle_with_deployment_move_to_default(self) -> int:
        """Hold default pose using the real deployment controller gains."""

        simulation_steps = int(
            round(self._gmr_free_settle_seconds / self.simulation_dt)
        )
        if simulation_steps <= 0:
            self._deployment_default_max_abs_diff = 0.0
            return 0
        defaults_mu, kp_mu, kd_mu = self._deployment_stand_parameters_mu()
        for _ in range(simulation_steps):
            q = self.robot_dof_pos
            qdot = self.robot_dof_vel
            for (
                actuator_name,
                actuator_index,
            ) in self.actuator_name_to_index.items():
                mu_index = self.actuator_name_to_mu_idx[actuator_name]
                torque = (
                    kp_mu[mu_index] * (defaults_mu[mu_index] - q[mu_index])
                    - kd_mu[mu_index] * qdot[mu_index]
                )
                force_range = self.actuator_force_range.get(actuator_index)
                if force_range is not None:
                    torque = np.clip(torque, *force_range)
                self.d.ctrl[actuator_index] = torque
            mujoco.mj_step(self.m, self.d)
        return simulation_steps

    def _apply_velocity_policy_control(
        self, target_dof_pos_mu: np.ndarray
    ) -> None:
        for _ in range(self.control_decimation):
            current_dof_pos = self.robot_dof_pos
            current_dof_vel = self.robot_dof_vel
            for (
                actuator_name,
                actuator_index,
            ) in self.actuator_name_to_index.items():
                mu_index = self.actuator_name_to_mu_idx[actuator_name]
                torque = (
                    self._velocity_kps_mu[mu_index]
                    * (target_dof_pos_mu[mu_index] - current_dof_pos[mu_index])
                    - self._velocity_kds_mu[mu_index]
                    * current_dof_vel[mu_index]
                )
                force_range = self.actuator_force_range.get(actuator_index)
                if force_range is not None:
                    torque = np.clip(torque, *force_range)
                self.d.ctrl[actuator_index] = torque
            mujoco.mj_step(self.m, self.d)

    def _run_velocity_policy_warmup(self) -> int:
        if self._gmr_free_velocity_warmup_seconds <= 0.0:
            self._velocity_warmup_summary = {
                "policy_steps": 0,
                "seconds": 0.0,
            }
            return 0
        if (
            self._velocity_obs_builder is None
            or self._velocity_policy_session is None
            or self._velocity_policy_input_name is None
            or self._velocity_policy_output_name is None
        ):
            raise RuntimeError("Velocity policy warm-up was not initialized")

        policy_steps = int(
            round(self._gmr_free_velocity_warmup_seconds / self.policy_dt)
        )
        if policy_steps <= 0:
            raise ValueError(
                "Velocity warm-up is shorter than one policy timestep"
            )
        self._velocity_obs_builder.reset()
        self._velocity_actions_onnx.fill(0.0)
        expected_input_dim = int(
            self._velocity_policy_session.get_inputs()[0].shape[-1]
        )
        self._velocity_stage_active = True
        try:
            for _ in range(policy_steps):
                observation = (
                    self._velocity_obs_builder.build_policy_obs()
                    .reshape(1, -1)
                    .astype(np.float32, copy=False)
                )
                if observation.shape != (1, expected_input_dim):
                    raise ValueError(
                        "Velocity observation is "
                        f"{observation.shape}, expected "
                        f"(1,{expected_input_dim})"
                    )
                outputs = self._velocity_policy_session.run(
                    [self._velocity_policy_output_name],
                    {self._velocity_policy_input_name: observation},
                )
                self._velocity_actions_onnx = np.asarray(
                    outputs[0], dtype=np.float32
                ).reshape(-1)
                if self._velocity_actions_onnx.shape != (
                    self._velocity_num_actions,
                ):
                    raise ValueError(
                        "Velocity policy action has shape "
                        f"{self._velocity_actions_onnx.shape}"
                    )
                if not np.isfinite(self._velocity_actions_onnx).all():
                    raise ValueError(
                        "Velocity policy produced non-finite actions"
                    )
                target_onnx = self._velocity_default_angles_onnx.copy()
                target_onnx[self._velocity_action_to_onnx] = (
                    target_onnx[self._velocity_action_to_onnx]
                    + self._velocity_actions_onnx
                    * self._velocity_action_scale_onnx
                )
                target_mu = target_onnx[self._velocity_onnx_to_mu]
                self._apply_velocity_policy_control(target_mu)
        finally:
            self._velocity_stage_active = False

        root_quat = np.asarray(
            self.robot_global_bodylink_rot[self.root_body_idx],
            dtype=np.float64,
        )
        self._velocity_warmup_summary = {
            "policy_steps": policy_steps,
            "simulation_steps": policy_steps * self.control_decimation,
            "seconds": policy_steps * float(self.policy_dt),
            "command": [0.0, 0.0, 0.0, 0.0],
            "max_abs_action": float(
                np.max(np.abs(self._velocity_actions_onnx))
            ),
            "max_abs_qdot": float(np.max(np.abs(self.robot_dof_vel))),
            "root_height": float(
                self.robot_global_bodylink_pos[self.root_body_idx, 2]
            ),
            "root_gravity_projection": float(
                1.0 - 2.0 * (root_quat[1] ** 2 + root_quat[2] ** 2)
            ),
            "onnx": str(self._gmr_free_velocity_onnx_path),
            "config": str(self._gmr_free_velocity_config_path),
        }
        return policy_steps

    def reset_state_teleport(self) -> None:
        if (
            self.smpl_reference_cpu is None
            or self._gmr_free_root_kinematics is None
        ):
            raise RuntimeError("Pure SMPL reference was not loaded")
        self.counter = 0
        self.motion_frame_idx = 0
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        asset_root_height = float(self.config.robot.init_state.pos[2])

        if self._gmr_free_reset_profile == "deployment_stand":
            root_pos = torch.as_tensor(
                [0.0, 0.0, asset_root_height], dtype=torch.float32
            )
            root_quat = torch.as_tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
            )
        else:
            root_pos = self.smpl_reference_cpu.root_pos[0].clone()
            root_pos[:2] *= (
                self._gmr_free_robot_scale / self._gmr_free_human_scale
            )
            root_quat = self.smpl_reference_cpu.root_rot_xyzw[0, [3, 0, 1, 2]]
        grounding = self._build_pose_grounding_runtime()
        with torch.no_grad():
            height_adjustment = grounding(
                grounding.q_default.reshape(1, -1),
                root_quat.reshape(1, 4),
            )[0]
        root_pos[2] = asset_root_height + height_adjustment

        if self._gmr_free_reset_profile == "deployment_stand":
            root_linear_world = torch.zeros(3, dtype=torch.float32)
            root_angular_world = torch.zeros(3, dtype=torch.float32)
        else:
            linear_local = self._gmr_free_root_kinematics.root_linvel_local[0]
            linear_local = linear_local * (
                self._gmr_free_robot_scale / self._gmr_free_human_scale
            )
            angular_local = self._gmr_free_root_kinematics.root_angvel_local[0]
            root_linear_world = quat_apply(root_quat, linear_local)
            root_angular_world = quat_apply(root_quat, angular_local)

        self.d.qpos[0:3] = root_pos.numpy()
        self.d.qpos[3:7] = root_quat.numpy()
        self.d.qpos[self.actuator_qpos_indices] = self.default_angles_mu
        self.d.qvel[0:3] = root_linear_world.numpy()
        self.d.qvel[3:6] = root_angular_world.numpy()
        self.d.qvel[self.actuator_qvel_indices] = 0.0
        self.target_dof_pos_mu = self.default_angles_mu.astype(np.float32)
        self.target_dof_pos_by_name = {
            self.mjcf_dof_names[index]: float(self.target_dof_pos_mu[index])
            for index in range(self.m.nu)
        }
        mujoco.mj_forward(self.m, self.d)
        settle_simulation_steps = 0
        if self._gmr_free_reset_profile == "deployment_stand":
            settle_simulation_steps = (
                self._settle_with_deployment_move_to_default()
            )
        self._run_velocity_policy_warmup()
        self._capture_deployment_yaw_alignment()
        self._visual_smpl_target_positions = self._build_visual_smpl_target()
        root_quat_after_settle = np.asarray(
            self.robot_global_bodylink_rot[self.root_body_idx],
            dtype=np.float64,
        )
        self._settle_summary = {
            "simulation_steps": settle_simulation_steps,
            "seconds": settle_simulation_steps * float(self.simulation_dt),
            "max_abs_qdot": float(np.max(np.abs(self.robot_dof_vel))),
            "root_height": float(
                self.robot_global_bodylink_pos[self.root_body_idx, 2]
            ),
            "root_gravity_projection": float(
                1.0
                - 2.0
                * (
                    root_quat_after_settle[1] ** 2
                    + root_quat_after_settle[2] ** 2
                )
            ),
            "deployment_to_motion_default_max_abs_diff_rad": (
                self._deployment_default_max_abs_diff
            ),
            "velocity_warmup": self._velocity_warmup_summary,
        }
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self._prev_actions_onnx = None
        self._reset_action_delay_randomization()
        self._reset_rollout_buffers()
        if self.use_kv_cache and self.policy_kv_shape:
            shape = [
                dim if isinstance(dim, int) else 1
                for dim in self.policy_kv_shape
            ]
            self.policy_kv_cache = np.zeros(shape, dtype=np.float32)
        logger.info(
            "GMR-free reset: profile={}, root={}, q=default, "
            "settle_summary={}, velocity_warmup={}",
            self._gmr_free_reset_profile,
            self.d.qpos[:7].tolist(),
            self._settle_summary,
            self._velocity_warmup_summary,
        )

    def _reset_rollout_buffers(self) -> None:
        self._robot_dof_pos_seq = []
        self._robot_dof_vel_seq = []
        self._robot_global_translation_seq = []
        self._robot_global_rotation_quat_seq = []
        self._robot_global_velocity_seq = []
        self._robot_global_angular_velocity_seq = []
        self._robot_actions_seq = []

    def _record_pure_robot_state(self) -> None:
        self._robot_dof_pos_seq.append(
            self.robot_dof_pos[self.mu_to_onnx].astype(np.float32).copy()
        )
        self._robot_dof_vel_seq.append(
            self.robot_dof_vel[self.mu_to_onnx].astype(np.float32).copy()
        )
        self._robot_global_translation_seq.append(
            self.robot_global_bodylink_pos.astype(np.float32).copy()
        )
        rotation_wxyz = self.robot_global_bodylink_rot.astype(np.float32)
        self._robot_global_rotation_quat_seq.append(
            rotation_wxyz[:, [1, 2, 3, 0]].copy()
        )
        self._robot_global_velocity_seq.append(
            self.robot_global_bodylink_lin_vel.astype(np.float32).copy()
        )
        self._robot_global_angular_velocity_seq.append(
            self.robot_global_bodylink_ang_vel.astype(np.float32).copy()
        )

    def _update_policy(self) -> None:
        if self._gmr_free_obs_builder is None or self.smpl_reference is None:
            raise RuntimeError("GMR-free observation path is not initialized")
        self._record_pure_robot_state()
        robot_quat = self.robot_global_bodylink_rot[self.root_body_idx]
        robot_ang_local = self._get_obs_rel_robot_root_ang_vel()
        observation = self._gmr_free_obs_builder.build(
            smpl_reference=self.smpl_reference,
            reference_frame_index=self.motion_frame_idx,
            robot_root_quat_wxyz=robot_quat,
            robot_root_angvel_local=robot_ang_local,
            robot_dof_pos=self.robot_dof_pos[self.mu_to_onnx][
                self._gmr_free_obs_to_onnx
            ],
            robot_dof_vel=self.robot_dof_vel[self.mu_to_onnx][
                self._gmr_free_obs_to_onnx
            ],
            last_action=self.actions_onnx,
            reference_yaw_alignment_wxyz=(self._reference_yaw_alignment_wxyz),
        )
        expected_dim = int(self.policy_session.get_inputs()[0].shape[-1])
        if observation.shape != (1, expected_dim):
            raise ValueError(
                "Pure SMPL Actor observation is "
                f"{observation.shape}, expected (1,{expected_dim})"
            )
        cpu_inputs: dict[str, np.ndarray] = {}
        output_names = [self.policy_output_name]
        if self.use_kv_cache:
            cpu_inputs[self.policy_kv_input_name] = self.policy_kv_cache
            if self.policy_kv_output_name:
                output_names.append(self.policy_kv_output_name)
        if self.policy_step_input_name is not None:
            step = self.counter
            if self.policy_effective_context_len > 0:
                step %= self.policy_effective_context_len
            cpu_inputs[self.policy_step_input_name] = np.asarray(
                [step], dtype=np.int64
            )
        outputs = run_with_cuda_observation(
            self.policy_session,
            input_name=self.policy_input_name,
            observation=observation,
            output_names=output_names,
            cpu_inputs=cpu_inputs,
        )
        self.actions_onnx = np.asarray(outputs[0], dtype=np.float32).reshape(
            -1
        )
        if self.actions_onnx.shape != (self.num_actions,):
            raise ValueError(
                f"Policy action shape changed: {self.actions_onnx.shape}"
            )
        if not np.isfinite(self.actions_onnx).all():
            raise ValueError("Policy produced non-finite actions")
        if self.use_kv_cache and len(outputs) > 1:
            self.policy_kv_cache = outputs[1]
        self.target_dof_pos_onnx = self._compose_full_target_dof_pos(
            self.actions_onnx
        )
        self.target_dof_pos_mu = self.target_dof_pos_onnx[self.onnx_to_mu]
        self.target_dof_pos_by_name = {
            self.mjcf_dof_names[index]: float(self.target_dof_pos_mu[index])
            for index in range(self.m.nu)
        }
        self._robot_actions_seq.append(self.actions_onnx.copy())

    def run_pure_rollout(self, output_path: str | Path) -> Path:
        self.setup()
        self.reset_state_teleport()
        for frame in range(self.n_motion_frames):
            self.motion_frame_idx = frame
            self._update_policy()
            self.counter += 1
            self._apply_control(sleep=False)
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **(self.pure_clip.metadata if self.pure_clip is not None else {}),
            "schema_version": "gmr_free_smpl_mujoco_rollout_v2",
            "source": str(self.config.motion_npz_path),
            "onnx_model": str(self.config.ckpt_onnx_path),
            "policy_observation": "shared_warp_cuda_smpl_par258",
            "reset": self._gmr_free_reset_profile,
            "settle_summary": self._settle_summary,
            "velocity_warmup": self._velocity_warmup_summary,
            "robot_reference_reads": 0,
            "robot_dof_order": "motion_onnx",
        }
        np.savez_compressed(
            output,
            human_pose_aa=self.pure_clip.human_pose_aa,
            human_root_trans=self.pure_clip.human_root_trans,
            smpl_semantic_position=self.smpl_target_positions,
            smpl_semantic_position_aligned=(
                self._visual_smpl_target_positions
            ),
            robot_dof_pos=np.stack(self._robot_dof_pos_seq),
            robot_dof_vel=np.stack(self._robot_dof_vel_seq),
            robot_dof_names=np.asarray(self.dof_names_onnx),
            robot_body_names=np.asarray(
                [str(name) for name in self.config.robot.body_names]
            ),
            robot_global_translation=np.stack(
                self._robot_global_translation_seq
            ),
            robot_global_rotation_quat=np.stack(
                self._robot_global_rotation_quat_seq
            ),
            robot_global_velocity=np.stack(self._robot_global_velocity_seq),
            robot_global_angular_velocity=np.stack(
                self._robot_global_angular_velocity_seq
            ),
            robot_actions=np.stack(self._robot_actions_seq),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        with np.load(output, allow_pickle=False) as result:
            leaked = sorted(
                name
                for name in result.files
                if name.startswith(FORBIDDEN_OUTPUT_PREFIXES)
            )
            if leaked:
                raise AssertionError(f"Pure rollout leaked fields: {leaked}")
            for name in result.files:
                value = np.asarray(result[name])
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(
                        f"Rollout field {name} contains non-finite values"
                    )
        return output


__all__ = [
    "GmrFreeSmplMujocoEvaluator",
    "PureSmplClip",
    "load_pure_smpl_clip",
    "load_velocity_actor_obs_config",
    "save_pure_smpl_clip",
    "validate_named_dof_matrix",
]
