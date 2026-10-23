# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from dataclasses import MISSING, dataclass
from typing import Sequence
import hashlib
import time
import json

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils


import isaaclab.utils.math as isaaclab_math
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.managers import (
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from holomotion.src.training.h5_dataloader import (
    Hdf5MotionDataset,
    Hdf5RootDofDataset,
    MotionClipBatchCache,
    build_motion_datasets_from_cfg,
    normalize_hdf5_root_entries,
    weighted_bin_cfg_from_motion_cfg,
)
from holomotion.src.motion_tracking.reference_observation import (
    ReferenceKinematics,
    derive_reference_kinematics_torch,
)
from holomotion.src.motion_tracking.actor_observation import (
    MotionActorObservationInput,
    derive_motion_actor_terms_torch,
)
from holomotion.src.motion_tracking.robot_skeleton_adapter import (
    RobotSkeletonAdapterSpec,
    SemanticSkeletonState,
    build_robot_skeleton_state,
)
import os
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omegaconf import OmegaConf

from holomotion.src.utils.isaac_utils.rotations import (
    calc_heading_quat_inv,
    get_euler_xyz,
    my_quat_rotate,
    quat_inverse,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_matrix,
    wrap_to_pi,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from holomotion.src.utils.reference_prefix import (
    resolve_reference_tensor_key,
)
from loguru import logger


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from a WXYZ quaternion tensor."""
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


@dataclass(frozen=True)
class CanonicalSmplRootState:
    """One-frame SMPL root state in the exact Actor command convention."""

    position: torch.Tensor
    rotation_wxyz: torch.Tensor
    linear_velocity_local: torch.Tensor
    angular_velocity_local: torch.Tensor
    yaw_rate: torch.Tensor
    nominal_scale: torch.Tensor


TASK_SPACE_TARGET_SOURCES = frozenset(
    {"none", "robot_reference", "smpl", "smpl_skeleton"}
)
RESET_TARGET_SOURCES = frozenset(
    {
        "composed",
        "robot_reference",
        "smpl_default",
        "smpl_semantic_ik",
        "smpl_cycle_network",
    }
)
SMPL_RESET_ROOT_HEIGHT_SOURCES = frozenset(
    {"asset_default", "semantic_ik_grounded"}
)
SMPL_RESET_ROOT_LINEAR_VELOCITY_SOURCES = frozenset({"zero", "smpl_scaled_xy"})
SMPL_RESET_ROOT_ANGULAR_VELOCITY_SOURCES = frozenset({"zero", "smpl_yaw_rate"})
COMPOSED_RESET_ROOT_XY_SOURCES = frozenset(
    {"env_origin", "robot_reference", "smpl_actor_canonical_scaled"}
)
COMPOSED_RESET_ROOT_HEIGHT_SOURCES = frozenset(
    {"cycle_grounded", "pose_grounded", "robot_reference"}
)
COMPOSED_RESET_ROOT_ROLL_PITCH_SOURCES = frozenset(
    {"robot_reference", "smpl_actor_canonical", "zero"}
)
COMPOSED_RESET_ROOT_YAW_SOURCES = frozenset(
    {"robot_reference", "smpl", "smpl_actor_canonical"}
)
COMPOSED_RESET_ROOT_LINEAR_VELOCITY_SOURCES = frozenset(
    {
        "robot_reference",
        "smpl_actor_canonical_scaled_full",
        "smpl_scaled_xy",
    }
)
COMPOSED_RESET_ROOT_ANGULAR_VELOCITY_SOURCES = frozenset(
    {
        "robot_reference",
        "smpl_actor_canonical_full",
        "smpl_actor_canonical_yaw_rate",
        "smpl_yaw_rate",
    }
)
COMPOSED_RESET_JOINT_POSITION_SOURCES = frozenset(
    {"asset_default", "cycle_network", "robot_reference"}
)
COMPOSED_RESET_JOINT_VELOCITY_SOURCES = frozenset({"robot_reference", "zero"})
TASK_SPACE_NODE_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_virtual_toe",
    "right_virtual_toe",
    "torso_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
TASK_SPACE_CACHE_KEYS = {
    "position": "smpl_task_position_h",
    "rotation": "smpl_task_rotation_xyzw_h",
    "linear_velocity": "smpl_task_linear_velocity_h",
    "angular_velocity": "smpl_task_angular_velocity_h",
}
REF_TASK_SPACE_CACHE_KEYS = {
    "position": "ref_task_position",
    "rotation": "ref_task_rotation_xyzw",
    "linear_velocity": "ref_task_linear_velocity",
    "angular_velocity": "ref_task_angular_velocity",
}
SKELETON_CACHE_KEYS = {
    "position": "smpl_skeleton_position_h",
    "pelvis_rotation": "smpl_skeleton_pelvis_rotation_xyzw_h",
    "torso_rotation": "smpl_skeleton_torso_rotation_xyzw_h",
    "endpoint_rotation": "smpl_skeleton_endpoint_rotation_xyzw_h",
    "head_rotation": "smpl_skeleton_head_rotation_xyzw_h",
    "root_linear_velocity": "smpl_skeleton_root_linear_velocity_h",
    "root_angular_velocity": "smpl_skeleton_root_angular_velocity_h",
    "nominal_scale": "smpl_skeleton_nominal_scale_h",
}
VIRTUAL_TOE_OFFSET = (0.1, 0.0, -0.02)


@dataclass(frozen=True)
class TaskSpaceState:
    """Four task-space tensors in a common world frame.

    Every field uses the frozen ``TASK_SPACE_NODE_NAMES`` order.  Pose and
    velocity tensors therefore have shapes ``[B, 14, 3/4]``.
    """

    position_w: torch.Tensor
    rotation_w_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    angular_velocity_w: torch.Tensor


class RefMotionCommand(CommandTerm):
    cfg: CommandTermCfg

    def __init__(
        self,
        cfg,
        env: ManagerBasedRLEnv,
    ):
        # print(cfg)
        super().__init__(cfg, env)
        self._env = env
        self._is_evaluating = self.cfg.is_evaluating
        self._runtime_process_id = int(self.cfg.process_id)
        self._runtime_num_processes = max(1, int(self.cfg.num_processes))
        motion_cfg = OmegaConf.create(self.cfg.motion_lib_cfg)
        self._task_space_target_source = (
            str(motion_cfg.get("task_target_source", "none")).strip().lower()
        )
        if self._task_space_target_source not in TASK_SPACE_TARGET_SOURCES:
            raise ValueError(
                "robot.motion.task_target_source must be one of "
                f"{sorted(TASK_SPACE_TARGET_SOURCES)}, got "
                f"{self._task_space_target_source!r}"
            )
        self._reset_target_source = (
            str(
                motion_cfg.get(
                    "reset_target_source",
                    "robot_reference",
                )
            )
            .strip()
            .lower()
        )
        if self._reset_target_source not in RESET_TARGET_SOURCES:
            raise ValueError(
                "robot.motion.reset_target_source must be one of "
                f"{sorted(RESET_TARGET_SOURCES)}, got "
                f"{self._reset_target_source!r}"
            )
        smpl_reset_cfg = motion_cfg.get("smpl_reset", {}) or {}
        self._smpl_reset_root_height_source = (
            str(
                smpl_reset_cfg.get(
                    "root_height_source",
                    "asset_default",
                )
            )
            .strip()
            .lower()
        )
        if (
            self._smpl_reset_root_height_source
            not in SMPL_RESET_ROOT_HEIGHT_SOURCES
        ):
            raise ValueError(
                "robot.motion.smpl_reset.root_height_source must be one of "
                f"{sorted(SMPL_RESET_ROOT_HEIGHT_SOURCES)}, got "
                f"{self._smpl_reset_root_height_source!r}"
            )
        self._smpl_reset_root_height_offset_m = float(
            smpl_reset_cfg.get("root_height_offset_m", 0.0)
        )
        if not np.isfinite(self._smpl_reset_root_height_offset_m):
            raise ValueError(
                "robot.motion.smpl_reset.root_height_offset_m must be finite"
            )
        self._smpl_reset_root_linear_velocity_source = (
            str(smpl_reset_cfg.get("root_linear_velocity_source", "zero"))
            .strip()
            .lower()
        )
        if (
            self._smpl_reset_root_linear_velocity_source
            not in SMPL_RESET_ROOT_LINEAR_VELOCITY_SOURCES
        ):
            raise ValueError(
                "robot.motion.smpl_reset.root_linear_velocity_source must be "
                f"one of {sorted(SMPL_RESET_ROOT_LINEAR_VELOCITY_SOURCES)}, "
                f"got {self._smpl_reset_root_linear_velocity_source!r}"
            )
        self._smpl_reset_root_angular_velocity_source = (
            str(smpl_reset_cfg.get("root_angular_velocity_source", "zero"))
            .strip()
            .lower()
        )
        if (
            self._smpl_reset_root_angular_velocity_source
            not in SMPL_RESET_ROOT_ANGULAR_VELOCITY_SOURCES
        ):
            raise ValueError(
                "robot.motion.smpl_reset.root_angular_velocity_source must be "
                f"one of {sorted(SMPL_RESET_ROOT_ANGULAR_VELOCITY_SOURCES)}, "
                f"got {self._smpl_reset_root_angular_velocity_source!r}"
            )
        semantic_ik_cfg = smpl_reset_cfg.get("semantic_ik", {}) or {}
        semantic_ik_mapping = (
            OmegaConf.to_container(semantic_ik_cfg, resolve=True)
            if OmegaConf.is_config(semantic_ik_cfg)
            else dict(semantic_ik_cfg)
        )
        if not isinstance(semantic_ik_mapping, dict):
            raise TypeError(
                "robot.motion.smpl_reset.semantic_ik must resolve to a mapping"
            )
        self._semantic_ik_reset_config_mapping = semantic_ik_mapping
        cycle_network_cfg = smpl_reset_cfg.get("cycle_network", {}) or {}
        cycle_network_mapping = (
            OmegaConf.to_container(cycle_network_cfg, resolve=True)
            if OmegaConf.is_config(cycle_network_cfg)
            else dict(cycle_network_cfg)
        )
        if not isinstance(cycle_network_mapping, dict):
            raise TypeError(
                "robot.motion.smpl_reset.cycle_network must resolve to a "
                "mapping"
            )
        self._smpl_cycle_reset_config_mapping = cycle_network_mapping
        pose_grounding_cfg = smpl_reset_cfg.get("pose_grounding", {}) or {}
        pose_grounding_mapping = (
            OmegaConf.to_container(pose_grounding_cfg, resolve=True)
            if OmegaConf.is_config(pose_grounding_cfg)
            else dict(pose_grounding_cfg)
        )
        if not isinstance(pose_grounding_mapping, dict):
            raise TypeError(
                "robot.motion.smpl_reset.pose_grounding must resolve to a "
                "mapping"
            )
        self._smpl_pose_grounding_config_mapping = pose_grounding_mapping
        composed_cfg = smpl_reset_cfg.get("composed", {}) or {}
        composed_mapping = (
            OmegaConf.to_container(composed_cfg, resolve=True)
            if OmegaConf.is_config(composed_cfg)
            else dict(composed_cfg)
        )
        if not isinstance(composed_mapping, dict):
            raise TypeError(
                "robot.motion.smpl_reset.composed must resolve to a mapping"
            )
        composed_source_specs = (
            (
                "root_xy_source",
                "robot_reference",
                COMPOSED_RESET_ROOT_XY_SOURCES,
            ),
            (
                "root_height_source",
                "robot_reference",
                COMPOSED_RESET_ROOT_HEIGHT_SOURCES,
            ),
            (
                "root_roll_pitch_source",
                "robot_reference",
                COMPOSED_RESET_ROOT_ROLL_PITCH_SOURCES,
            ),
            (
                "root_yaw_source",
                "robot_reference",
                COMPOSED_RESET_ROOT_YAW_SOURCES,
            ),
            (
                "root_linear_velocity_source",
                "robot_reference",
                COMPOSED_RESET_ROOT_LINEAR_VELOCITY_SOURCES,
            ),
            (
                "root_angular_velocity_source",
                "robot_reference",
                COMPOSED_RESET_ROOT_ANGULAR_VELOCITY_SOURCES,
            ),
            (
                "joint_position_source",
                "robot_reference",
                COMPOSED_RESET_JOINT_POSITION_SOURCES,
            ),
            (
                "joint_velocity_source",
                "robot_reference",
                COMPOSED_RESET_JOINT_VELOCITY_SOURCES,
            ),
        )
        self._composed_reset_sources = {}
        for field_name, default_value, allowed_values in composed_source_specs:
            value = (
                str(composed_mapping.get(field_name, default_value))
                .strip()
                .lower()
            )
            if value not in allowed_values:
                raise ValueError(
                    f"robot.motion.smpl_reset.composed.{field_name} must be "
                    f"one of {sorted(allowed_values)}, got {value!r}"
                )
            self._composed_reset_sources[field_name] = value
        composed_uses_cycle_pose = (
            self._composed_reset_sources["joint_position_source"]
            == "cycle_network"
        )
        composed_uses_cycle_height = (
            self._composed_reset_sources["root_height_source"]
            == "cycle_grounded"
        )
        composed_uses_pose_grounding = (
            self._composed_reset_sources["root_height_source"]
            == "pose_grounded"
        )
        if composed_uses_cycle_height and not composed_uses_cycle_pose:
            raise ValueError(
                "root_height_source='cycle_grounded' requires "
                "joint_position_source='cycle_network'"
            )
        if composed_uses_cycle_pose and not (
            composed_uses_cycle_height or composed_uses_pose_grounding
        ):
            raise ValueError(
                "joint_position_source='cycle_network' requires a grounded "
                "height source"
            )
        if (
            self._composed_reset_sources["root_height_source"]
            == "pose_grounded"
            and self._composed_reset_sources["joint_position_source"]
            == "robot_reference"
        ):
            raise ValueError(
                "pose_grounded height is reserved for Robot-ref-free "
                "asset_default or cycle_network joint poses"
            )
        if (
            self._reset_target_source
            in {
                "composed",
                "smpl_default",
                "smpl_semantic_ik",
                "smpl_cycle_network",
            }
            and self._task_space_target_source != "smpl_skeleton"
        ):
            raise ValueError(
                "SMPL reset targets require "
                "robot.motion.task_target_source='smpl_skeleton'"
            )
        if (
            self._reset_target_source == "smpl_default"
            and self._smpl_reset_root_height_source != "asset_default"
        ):
            raise ValueError(
                "smpl_default reset requires "
                "root_height_source='asset_default'"
            )
        if (
            self._reset_target_source
            in {"smpl_semantic_ik", "smpl_cycle_network"}
            and self._smpl_reset_root_height_source != "semantic_ik_grounded"
        ):
            raise ValueError(
                "grounded SMPL pose reset requires "
                "root_height_source='semantic_ik_grounded'"
            )
        logger.info(
            "Motion reset configuration: "
            f"reset_target_source={self._reset_target_source}, "
            "smpl_reset.root_height_source="
            f"{self._smpl_reset_root_height_source}, "
            "smpl_reset.root_height_offset_m="
            f"{self._smpl_reset_root_height_offset_m:.6f}, "
            "smpl_reset.root_linear_velocity_source="
            f"{self._smpl_reset_root_linear_velocity_source}, "
            "smpl_reset.root_angular_velocity_source="
            f"{self._smpl_reset_root_angular_velocity_source}"
        )
        if self._reset_target_source == "composed":
            logger.info(
                "Composed reset sources: "
                + ", ".join(
                    f"{name}={value}"
                    for name, value in self._composed_reset_sources.items()
                )
            )
        self._skeleton_adapter_spec = None
        if self._task_space_target_source == "smpl_skeleton":
            skeleton_cfg = motion_cfg.get("semantic_skeleton")
            if skeleton_cfg is None:
                raise ValueError(
                    "robot.motion.semantic_skeleton is required when "
                    "task_target_source='smpl_skeleton'"
                )
            skeleton_mapping = OmegaConf.to_container(
                skeleton_cfg,
                resolve=True,
            )
            if not isinstance(skeleton_mapping, dict):
                raise TypeError(
                    "robot.motion.semantic_skeleton must resolve to a mapping"
                )
            self._skeleton_adapter_spec = (
                RobotSkeletonAdapterSpec.from_mapping(skeleton_mapping)
            )

        self._init_robot_handle()
        self._semantic_ik_reset_solver = None
        self._smpl_cycle_reset_runtime = None
        self._smpl_pose_grounding_runtime = None
        if self._reset_target_source == "smpl_semantic_ik":
            self._init_semantic_ik_reset_solver(motion_cfg)
        elif self._reset_target_source == "smpl_cycle_network":
            self._init_smpl_cycle_reset_runtime(motion_cfg)
        elif (
            self._reset_target_source == "composed"
            and self._composed_reset_sources["joint_position_source"]
            == "cycle_network"
        ):
            self._init_smpl_cycle_reset_runtime(motion_cfg)
        if (
            self._reset_target_source == "composed"
            and self._composed_reset_sources["root_height_source"]
            == "pose_grounded"
        ):
            self._init_smpl_pose_grounding_runtime(motion_cfg)
        self._init_buffers()
        self._init_motion_lib()

    #     # self._init_tracking_config()

    def _init_tracking_config(self, config):
        self.log_dict_holomotion = {}
        self.log_dict_nonreduced_holomotion = {}
        self.log_dict_nonreduced = {}
        self.log_dict = {}
        if "head_hand_bodies" in config:
            self.motion_tracking_id = [
                self.robot.body_names.index(link)
                for link in config.head_hand_bodies
            ]
        if "leg_body_names" in config:
            self.lower_body_id = [
                self.robot.body_names.index(link)
                for link in config.leg_body_names
            ]
        if "arm_body_names" in config:
            self.upper_body_id = [
                self.robot.body_names.index(link)
                for link in config.arm_body_names
            ]
        if "leg_dof_names" in config:
            self.lower_body_joint_ids = [
                config.dof_names.index(link) for link in config.leg_dof_names
            ]
        if "arm_dof_names" in config:
            self.upper_body_joint_ids = [
                config.dof_names.index(link) for link in config.arm_dof_names
            ]

        if "waist_dof_names" in config:
            self.waist_dof_indices = [
                config.dof_names.index(link) for link in config.waist_dof_names
            ]

    @staticmethod
    def _amp_filter_names_by_prefix(
        names: Sequence[str], prefix: str, keywords: Sequence[str]
    ) -> list[str]:
        return [
            name
            for name in names
            if name.startswith(prefix) and any(key in name for key in keywords)
        ]

    @staticmethod
    def _amp_pick_first_name(
        names: Sequence[str], patterns: Sequence[str]
    ) -> str | None:
        for pattern in patterns:
            for name in names:
                if pattern in name:
                    return name
        return None

    def _resolve_motion_cache_stage_device(
        self, cache_cfg: Dict[str, object]
    ) -> Optional[torch.device]:
        raw_stage_device = cache_cfg.get("device", "cuda")
        if isinstance(raw_stage_device, torch.device):
            if raw_stage_device.type == "cpu":
                return None
            if raw_stage_device.type != "cuda":
                raise ValueError(
                    f"Unsupported motion cache device: {raw_stage_device}"
                )
            if raw_stage_device.index is not None:
                return raw_stage_device
            if not torch.cuda.is_available():
                return None
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None:
                local_rank = int(local_rank_env)
                device_count = int(torch.cuda.device_count())
                if 0 <= local_rank < device_count:
                    return torch.device("cuda", local_rank)
            return torch.device("cuda", int(torch.cuda.current_device()))

        stage_device = str(raw_stage_device).strip().lower()
        if stage_device in ("none", "cpu"):
            return None
        if stage_device == "cuda":
            if isinstance(self.device, torch.device):
                if self.device.type == "cuda":
                    return self.device
                return None
            device_str = str(self.device).strip().lower()
            if device_str.startswith("cuda"):
                return torch.device(device_str)
            if not torch.cuda.is_available():
                return None
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None:
                local_rank = int(local_rank_env)
                device_count = int(torch.cuda.device_count())
                if 0 <= local_rank < device_count:
                    return torch.device("cuda", local_rank)
            return torch.device("cuda", int(torch.cuda.current_device()))
        if stage_device.startswith("cuda:"):
            return torch.device(stage_device)
        raise ValueError(
            f"Unsupported motion cache device config: {raw_stage_device}"
        )

    def _init_motion_lib(self):
        mcfg = OmegaConf.create(self.cfg.motion_lib_cfg)
        self.mcfg = mcfg
        backend = str(mcfg.get("backend", "hdf5")).lower()
        self._motion_cache = None
        if backend in ("hdf5", "hdf5_simple"):
            # Support multi-root configuration while keeping single-root
            # behavior fully backward compatible.
            train_hdf5_roots = mcfg.get("train_hdf5_roots", None)
            val_hdf5_roots = mcfg.get("val_hdf5_roots", None)

            if train_hdf5_roots:
                train_roots, _ = normalize_hdf5_root_entries(train_hdf5_roots)
            else:
                hdf5_root = mcfg.get("hdf5_root")
                if hdf5_root is None:
                    raise ValueError("hdf5_root is required")
                train_roots = [str(hdf5_root)]

            val_hdf5_root = mcfg.get("val_hdf5_root", None)
            if val_hdf5_roots:
                val_roots, _ = normalize_hdf5_root_entries(val_hdf5_roots)
            elif val_hdf5_root is not None and str(val_hdf5_root) != str(
                train_roots[0]
            ):
                val_roots = [str(val_hdf5_root)]
            else:
                val_roots = None

            train_manifest_paths = [
                os.path.join(root, "manifest.json") for root in train_roots
            ]
            for mp in train_manifest_paths:
                if not os.path.exists(mp):
                    raise FileNotFoundError(
                        f"HDF5 manifest not found at {mp}. "
                        "Please set robot.motion.hdf5_root/train_hdf5_roots to "
                        "the correct path!"
                    )

            max_frame_length = int(mcfg.get("max_frame_length", 500))
            min_frame_length = int(mcfg.get("min_frame_length", 1))
            world_frame_norm = bool(
                mcfg.get("world_frame_normalization", True)
            )

            cache_cfg = mcfg.get("cache", {})
            allowed_prefixes = cache_cfg.get(
                "allowed_prefixes",
                ["ref_"],
            )

            if len(train_manifest_paths) == 1:
                logger.info(
                    f"Loading HDF5 training dataset from {train_manifest_paths[0]}"
                )
            else:
                logger.info(
                    f"Loading HDF5 training dataset from manifests: "
                    f"{train_manifest_paths}"
                )
            train_dataset = Hdf5MotionDataset(
                manifest_path=train_manifest_paths
                if len(train_manifest_paths) > 1
                else train_manifest_paths[0],
                max_frame_length=max_frame_length,
                min_window_length=min_frame_length,
                handpicked_motion_names=mcfg.get(
                    "handpicked_motion_names", None
                ),
                excluded_motion_names=mcfg.get("excluded_motion_names", None),
                world_frame_normalization=world_frame_norm,
                allowed_prefixes=allowed_prefixes,
            )
            if len(train_dataset) == 0:
                raise ValueError(
                    "Training dataset is empty. Check that all manifests "
                    "contain valid clips with length "
                    f">= {min_frame_length}"
                )
            logger.info(f"Loaded {len(train_dataset)} training motion windows")
            train_num_clips = len(train_dataset.clips)
            train_total_frames = sum(
                int(meta.get("length", 0))
                for meta in train_dataset.clips.values()
            )
            fps_used = int(self.cfg.target_fps)
            train_duration_s = (
                float(train_total_frames) / float(fps_used)
                if fps_used > 0
                else 0.0
            )
            if len(train_roots) == 1:
                logger.info(
                    f"Train dataset: root={train_roots[0]}, "
                    f"manifest={train_manifest_paths[0]}"
                )
            else:
                logger.info(
                    f"Train dataset: roots={train_roots}, "
                    f"manifests={train_manifest_paths}"
                )
            logger.info(
                f"Train clips={train_num_clips}, frames={train_total_frames}, "
                f"duration={train_duration_s / 3600:.2f}h @ {fps_used} fps"
            )
            excluded_names = mcfg.get("excluded_motion_names", None)
            if excluded_names:
                excluded_set = set(excluded_names)
                excluded_clip_keys = [
                    k for k in train_dataset.clips.keys() if k in excluded_set
                ]
                excluded_num_clips = len(excluded_clip_keys)
                excluded_total_frames = sum(
                    int(train_dataset.clips[k].get("length", 0))
                    for k in excluded_clip_keys
                )
                excluded_duration_s = (
                    float(excluded_total_frames) / float(fps_used)
                    if fps_used > 0
                    else 0.0
                )
                left_num_clips = max(0, train_num_clips - excluded_num_clips)
                left_total_frames = max(
                    0, train_total_frames - excluded_total_frames
                )
                left_duration_s = (
                    float(left_total_frames) / float(fps_used)
                    if fps_used > 0
                    else 0.0
                )
                logger.info(
                    f"Excluded (by name): clips={excluded_num_clips}, "
                    f"frames={excluded_total_frames}, "
                    f"duration={excluded_duration_s / 3600:.2f}h"
                )
                logger.info(
                    f"Remaining after exclusion: clips={left_num_clips}, "
                    f"frames={left_total_frames}, "
                    f"duration={left_duration_s / 3600:.2f}h"
                )

            val_dataset = None
            if val_roots is not None:
                val_manifest_paths = [
                    os.path.join(root, "manifest.json") for root in val_roots
                ]
                for mp in val_manifest_paths:
                    if not os.path.exists(mp):
                        raise FileNotFoundError(
                            f"HDF5 validation manifest not found at {mp}. "
                            "Please set robot.motion.val_hdf5_root/"
                            "val_hdf5_roots to the correct path!"
                        )
                if len(val_manifest_paths) == 1:
                    logger.info(
                        f"Loading HDF5 validation dataset from {val_manifest_paths[0]}"
                    )
                else:
                    logger.info(
                        "Loading HDF5 validation dataset from manifests: "
                        f"{val_manifest_paths}"
                    )
                val_dataset = Hdf5MotionDataset(
                    manifest_path=val_manifest_paths
                    if len(val_manifest_paths) > 1
                    else val_manifest_paths[0],
                    max_frame_length=max_frame_length,
                    min_window_length=min_frame_length,
                    handpicked_motion_names=mcfg.get(
                        "handpicked_motion_names", None
                    ),
                    excluded_motion_names=mcfg.get(
                        "excluded_motion_names", None
                    ),
                    world_frame_normalization=world_frame_norm,
                    allowed_prefixes=allowed_prefixes,
                )
                logger.info(
                    f"Loaded {len(val_dataset)} validation motion windows"
                )
                val_num_clips = len(val_dataset.clips)
                val_total_frames = sum(
                    int(meta.get("length", 0))
                    for meta in val_dataset.clips.values()
                )
                val_duration_s = (
                    float(val_total_frames) / float(fps_used)
                    if fps_used > 0
                    else 0.0
                )
                if len(val_roots) == 1:
                    logger.info(
                        f"Val dataset: root={val_roots[0]}, "
                        f"manifest={val_manifest_paths[0]}"
                    )
                else:
                    logger.info(
                        f"Val dataset: roots={val_roots}, "
                        f"manifests={val_manifest_paths}"
                    )
                logger.info(
                    f"Val clips={val_num_clips}, frames={val_total_frames}, "
                    f"duration={val_duration_s / 3600:.1f}h @ {fps_used} fps"
                )
            else:
                logger.info(
                    "Validation dataset: using training dataset "
                    "(no separate val manifest found)"
                )

            dataloader_cfg = mcfg.get("dataloader", {})
            stage_device = self._resolve_motion_cache_stage_device(cache_cfg)

            self._motion_cache = MotionClipBatchCache(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=int(cache_cfg.get("max_num_clips", 1024)),
                stage_device=stage_device,
                num_workers=int(dataloader_cfg.get("num_workers", 4)),
                prefetch_factor=dataloader_cfg.get("prefetch_factor", None),
                pin_memory=bool(dataloader_cfg.get("pin_memory", True)),
                persistent_workers=bool(
                    dataloader_cfg.get("persistent_workers", True)
                ),
                batch_progress_bar=bool(
                    cache_cfg.get("batch_progress_bar", False)
                ),
                sampler_rank=int(self.cfg.process_id),
                sampler_world_size=int(self.cfg.num_processes),
                allowed_prefixes=allowed_prefixes,
                swap_interval_steps=int(
                    cache_cfg.get("swap_interval_steps", max_frame_length)
                ),
                seed=int(self.cfg.seed),
                loader_timeout=float(dataloader_cfg.get("timeout", 0.0)),
            )
            cache = self._motion_cache
            logger.info(
                "DataLoader params: "
                f"batch_size={cache._batch_size}, "
                f"num_workers={cache._num_workers}, "
                f"prefetch_factor={cache._prefetch_factor}, "
                f"pin_memory={cache._pin_memory}, "
                f"persistent_workers={cache._persistent_workers}"
            )
            logger.info(
                "Sampler/Cache params: "
                f"rank={cache._sampler_rank}/{cache._sampler_world_size}, "
                f"device={cache._stage_device}, "
                f"swap_interval_steps={cache.swap_interval_steps}"
            )
            self._motion_lib = None

        elif backend in (
            "hdf5_v2",
            "paired_hdf5_v2",
            "formal_smpl_hdf5",
            "smpl_hdf5_v1",
        ):
            max_frame_length = int(mcfg.get("max_frame_length", 500))
            min_frame_length = int(mcfg.get("min_frame_length", 1))
            world_frame_norm = bool(
                mcfg.get("world_frame_normalization", True)
            )
            cache_cfg = mcfg.get("cache", {})
            pure_smpl_backend = backend in {
                "formal_smpl_hdf5",
                "smpl_hdf5_v1",
            }
            allowed_prefixes = cache_cfg.get(
                "allowed_prefixes",
                ["smpl_", "frame_flag", "motion_fps"]
                if pure_smpl_backend
                else ["ref_"],
            )

            train_hdf5_roots = mcfg.get(
                "train_smpl_hdf5_roots"
                if pure_smpl_backend
                else "train_hdf5_roots",
                mcfg.get("train_hdf5_roots", None),
            )
            if train_hdf5_roots:
                train_roots, _ = normalize_hdf5_root_entries(train_hdf5_roots)
            else:
                hdf5_root = mcfg.get("hdf5_root", None)
                train_roots = [str(hdf5_root)] if hdf5_root is not None else []
            train_manifest_paths = [
                os.path.join(root, "manifest.json") for root in train_roots
            ]

            (
                train_dataset,
                val_dataset,
                cache_kwargs,
            ) = build_motion_datasets_from_cfg(
                motion_cfg=mcfg,
                max_frame_length=max_frame_length,
                min_window_length=min_frame_length,
                world_frame_normalization=world_frame_norm,
                handpicked_motion_names=mcfg.get(
                    "handpicked_motion_names", None
                ),
                excluded_motion_names=mcfg.get("excluded_motion_names", None),
                allowed_prefixes=allowed_prefixes,
            )
            if len(train_dataset) == 0:
                raise ValueError(
                    "Training dataset is empty. Check that all configured HDF5 "
                    "roots contain valid clips with length "
                    f">= {min_frame_length}"
                )

            if len(train_manifest_paths) == 1:
                logger.info(
                    f"Loading {backend} training dataset from "
                    f"{train_manifest_paths[0]}"
                )
            else:
                logger.info(
                    f"Loading {backend} training dataset from manifests: "
                    f"{train_manifest_paths}"
                )
            fps_used = int(self.cfg.target_fps)
            logger.info(f"Loaded {len(train_dataset)} training motion windows")
            train_num_clips = len(train_dataset.clips)
            train_total_frames = sum(
                int(meta.get("length", 0))
                for meta in train_dataset.clips.values()
            )
            train_duration_s = (
                float(train_total_frames) / float(fps_used)
                if fps_used > 0
                else 0.0
            )
            logger.info(
                f"Train clips={train_num_clips}, frames={train_total_frames}, "
                f"duration={train_duration_s / 3600:.2f}h @ {fps_used} fps"
            )
            if len(train_roots) == 1:
                logger.info(
                    f"Train dataset: root={train_roots[0]}, "
                    f"manifest={train_manifest_paths[0]}"
                )
            elif len(train_roots) > 1:
                logger.info(
                    f"Train dataset: roots={train_roots}, "
                    f"manifests={train_manifest_paths}"
                )
            excluded_names = mcfg.get("excluded_motion_names", None)
            if excluded_names:
                excluded_set = set(excluded_names)
                excluded_clip_keys: List[str] = []
                if hasattr(train_dataset, "_build_motion_key_aliases"):
                    for key, meta in train_dataset.clips.items():
                        aliases = train_dataset._build_motion_key_aliases(
                            key, meta
                        )
                        if any(alias in excluded_set for alias in aliases):
                            excluded_clip_keys.append(key)
                else:
                    excluded_clip_keys = [
                        k
                        for k in train_dataset.clips.keys()
                        if k in excluded_set
                    ]
                excluded_num_clips = len(excluded_clip_keys)
                excluded_total_frames = sum(
                    int(train_dataset.clips[k].get("length", 0))
                    for k in excluded_clip_keys
                )
                excluded_duration_s = (
                    float(excluded_total_frames) / float(fps_used)
                    if fps_used > 0
                    else 0.0
                )
                remaining_num_clips = train_num_clips - excluded_num_clips
                remaining_total_frames = (
                    train_total_frames - excluded_total_frames
                )
                remaining_duration_s = train_duration_s - excluded_duration_s
                logger.info(
                    "Excluded (by name): "
                    f"clips={excluded_num_clips}, frames={excluded_total_frames}, "
                    f"duration={excluded_duration_s / 3600:.2f}h"
                )
                logger.info(
                    "Remaining after exclusion: "
                    f"clips={remaining_num_clips}, frames={remaining_total_frames}, "
                    f"duration={remaining_duration_s / 3600:.2f}h"
                )
            if val_dataset is None:
                logger.info(
                    "Validation dataset: using training dataset "
                    "(no separate val HDF5 v2 roots found)"
                )

            dataloader_cfg = mcfg.get("dataloader", {})
            stage_device = self._resolve_motion_cache_stage_device(cache_cfg)

            self._motion_cache = MotionClipBatchCache(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=int(cache_cfg.get("max_num_clips", 1024)),
                stage_device=stage_device,
                num_workers=int(dataloader_cfg.get("num_workers", 4)),
                prefetch_factor=dataloader_cfg.get("prefetch_factor", None),
                pin_memory=bool(dataloader_cfg.get("pin_memory", True)),
                persistent_workers=bool(
                    dataloader_cfg.get("persistent_workers", True)
                ),
                batch_progress_bar=bool(
                    cache_cfg.get("batch_progress_bar", False)
                ),
                sampler_rank=int(self.cfg.process_id),
                sampler_world_size=int(self.cfg.num_processes),
                allowed_prefixes=allowed_prefixes,
                swap_interval_steps=int(
                    cache_cfg.get("swap_interval_steps", max_frame_length)
                ),
                seed=int(self.cfg.seed),
                loader_timeout=float(dataloader_cfg.get("timeout", 0.0)),
                **cache_kwargs,
            )
            cache = self._motion_cache
            logger.info(
                "DataLoader params: "
                f"batch_size={cache._batch_size}, "
                f"num_workers={cache._num_workers}, "
                f"prefetch_factor={cache._prefetch_factor}, "
                f"pin_memory={cache._pin_memory}, "
                f"persistent_workers={cache._persistent_workers}"
            )
            logger.info(
                "Sampler/Cache params: "
                f"rank={cache._sampler_rank}/{cache._sampler_world_size}, "
                f"device={cache._stage_device}, "
                f"swap_interval_steps={cache.swap_interval_steps}"
            )
            self._motion_lib = None

        else:
            raise ValueError(f"Unsupported motion backend: {backend}")

        sampling_strategy_cfg = mcfg.get("sampling_strategy", None)
        if sampling_strategy_cfg is None:
            sampling_strategy = "uniform"
        else:
            sampling_strategy = str(sampling_strategy_cfg).lower()
        if sampling_strategy == "weighted_bin":
            weighted_bin_cfg = weighted_bin_cfg_from_motion_cfg(mcfg)
            self._motion_cache.enable_weighted_bin_sampling(
                cfg=weighted_bin_cfg
            )
        elif sampling_strategy == "curriculum":
            curriculum_cfg = dict(mcfg.get("curriculum", {}) or {})
            self._motion_cache.enable_cache_curriculum_sampling(
                cfg=curriculum_cfg
            )
        elif sampling_strategy not in ("uniform", "curriculum"):
            raise ValueError(
                f"Invalid sampling_strategy '{sampling_strategy}'. "
                "Expected one of ['curriculum', 'uniform', 'weighted_bin']."
            )

        self._sampling_strategy = sampling_strategy

        self._init_per_env_cache()

    def setup_dumping_dir(self, log_dir: str):
        mcfg = self.mcfg
        base_log_dir = str(log_dir)

        if self._sampling_strategy == "curriculum":
            curriculum_dump_dir = os.path.join(
                base_log_dir, "cache_curriculum_window_scores"
            )
            self._motion_cache.set_cache_curriculum_dump_dir(
                curriculum_dump_dir
            )

        self._dump_sampled_motion_keys_enabled = bool(
            mcfg.get("dump_sampled_motion_keys", False)
        )
        if not self._dump_sampled_motion_keys_enabled:
            return
        self._dump_sampled_motion_keys_interval = max(
            1, int(mcfg.get("dump_sampled_motion_keys_interval", 1))
        )
        dump_dir_cfg = "sampled_motion_cache_keys"
        self._dump_sampled_motion_keys_dir = os.path.join(
            base_log_dir, dump_dir_cfg
        )
        if self._dump_sampled_motion_keys_enabled:
            os.makedirs(self._dump_sampled_motion_keys_dir, exist_ok=True)
            logger.info(
                f"Dumping sampled motion keys to {self._dump_sampled_motion_keys_dir}"
            )

    def set_runtime_distributed_context(
        self, *, process_id: int, num_processes: int
    ) -> None:
        self._runtime_process_id = int(process_id)
        self._runtime_num_processes = max(1, int(num_processes))

    def set_motion_cache_seed(
        self, seed: int, *, reinitialize: bool = True
    ) -> None:
        self._motion_cache.set_seed(int(seed), reinitialize=reinitialize)
        if reinitialize:
            self._init_per_env_cache()

    def close(self) -> None:
        """Release motion cache resources for this command term."""
        if self._motion_cache is not None:
            self._motion_cache.close()
            self._motion_cache = None

    def _init_per_env_cache(self):
        """Initialize per-env cache for motion tracking."""
        self._clip_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._frame_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._swap_pending = False
        self._swap_step_counter = 0

        # Initial assignment
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            self.num_envs,
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(self._is_evaluating),
        )
        self._clip_indices[:] = clip_idx
        self._frame_indices[:] = frame_idx
        self._start_frame_indices[:] = frame_idx
        self._reward_sum_since_assign[:] = 0.0
        self._step_count_since_assign[:] = 0.0
        self._update_ref_motion_state_from_cache()

    def _maybe_dump_sampled_motion_keys(self) -> None:
        if not self._dump_sampled_motion_keys_enabled:
            return

        swap_index = int(self._motion_cache.swap_index)
        if swap_index <= 0:
            return
        if swap_index % self._dump_sampled_motion_keys_interval != 0:
            return

        current_batch = self._motion_cache.current_batch
        window_indices = current_batch.window_indices.detach().cpu().tolist()
        cache_scores = None
        cache_selection_counts = None
        cache_in_prioritized_pool = None
        curriculum_state_step = None
        score_bundle = (
            self._motion_cache.cache_curriculum_scores_for_window_indices(
                current_batch.window_indices
            )
        )
        if score_bundle is not None:
            score_tensor, state, version = score_bundle
            cache_scores = score_tensor.detach().cpu().tolist()
            cache_selection_counts = (
                state["selection_count"].detach().cpu().tolist()
            )
            cache_in_prioritized_pool = (
                state["in_prioritized_pool"].detach().cpu().tolist()
            )
            curriculum_state_step = int(version)
        payload = {
            "swap_index": swap_index,
            "sampling_strategy": str(self._sampling_strategy),
            "num_keys": int(len(current_batch.motion_keys)),
            "motion_keys": list(current_batch.motion_keys),
            "raw_motion_keys": list(current_batch.raw_motion_keys),
            "window_indices": window_indices,
            "cache_sampling_score": cache_scores,
            "cache_sampling_count": cache_selection_counts,
            "cache_in_prioritized_pool": cache_in_prioritized_pool,
            "curriculum_state_step": curriculum_state_step,
        }
        file_name = (
            f"sampled_motion_keys_rank_{self._runtime_process_id:04d}_swap_"
            f"{swap_index:06d}.json"
        )
        output_path = os.path.join(
            self._dump_sampled_motion_keys_dir, file_name
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _init_robot_handle(self):
        self.robot: Articulation = self._env.scene[self.cfg.asset_name]
        self.anchor_bodylink_name = self.cfg.anchor_bodylink_name
        self.anchor_bodylink_idx = self.robot.body_names.index(
            self.anchor_bodylink_name
        )
        self.urdf_dof_names = self.cfg.urdf_dof_names
        self.urdf_body_names = self.cfg.urdf_body_names
        self.simulator_dof_names = self.robot.joint_names
        self.simulator_body_names = self.robot.body_names
        self.urdf2sim_dof_idx = [
            self.urdf_dof_names.index(dof) for dof in self.simulator_dof_names
        ]
        self.urdf2sim_body_idx = [
            self.urdf_body_names.index(body)
            for body in self.simulator_body_names
        ]
        self.sim2urdf_dof_idx = [
            self.simulator_dof_names.index(dof) for dof in self.urdf_dof_names
        ]
        self.sim2urdf_body_idx = [
            self.simulator_body_names.index(body)
            for body in self.urdf_body_names
        ]

        self.arm_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.arm_dof_names
        ]
        self.torso_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.waist_dof_names
        ]
        self.leg_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.leg_dof_names
        ]

        # Body indices for mpkpe metrics using unified naming
        self.arm_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.arm_body_names
        ]
        self.torso_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.torso_body_names
        ]
        self.leg_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.leg_body_names
        ]

        # Per-env world origins (translation only)
        # Shape: [num_envs, 3] on the same device as the sim
        self._env_origins = self._env.scene.env_origins.to(self.device)
        self._task_space_body_indices = {}
        self._skeleton_body_indices = {}
        if self._task_space_target_source in {"robot_reference", "smpl"}:
            required_body_names = (
                "pelvis",
                "left_hip_roll_link",
                "right_hip_roll_link",
                "left_knee_link",
                "right_knee_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_shoulder_yaw_link",
                "right_shoulder_yaw_link",
                "left_elbow_link",
                "right_elbow_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
            missing = [
                name
                for name in required_body_names
                if name not in self.simulator_body_names
            ]
            if missing:
                raise ValueError(
                    "Task-space reward requires G1 semantic bodies; "
                    f"missing {missing}"
                )
            self._task_space_body_indices = {
                name: self.simulator_body_names.index(name)
                for name in required_body_names
            }
        elif self._task_space_target_source == "smpl_skeleton":
            if self._skeleton_adapter_spec is None:
                raise AssertionError("Missing semantic skeleton adapter spec")
            required_body_names = (
                self._skeleton_adapter_spec.required_link_names
            )
            missing = [
                name
                for name in required_body_names
                if name not in self.simulator_body_names
            ]
            if missing:
                raise ValueError(
                    "Scale-invariant skeleton reward requires configured "
                    f"semantic links; missing {missing}"
                )
            self._skeleton_body_indices = {
                name: self.simulator_body_names.index(name)
                for name in required_body_names
            }
        self._task_space_toe_offset = torch.tensor(
            VIRTUAL_TOE_OFFSET,
            device=self.device,
            dtype=torch.float32,
        )

        # AMP-style observation indices (RSL reference alignment)
        urdf_dof_name_to_idx = {
            name: idx for idx, name in enumerate(self.urdf_dof_names)
        }
        sim_dof_name_to_idx = {
            name: idx for idx, name in enumerate(self.simulator_dof_names)
        }
        urdf_body_name_to_idx = {
            name: idx for idx, name in enumerate(self.urdf_body_names)
        }
        sim_body_name_to_idx = {
            name: idx for idx, name in enumerate(self.simulator_body_names)
        }

        left_arm_dof_names = list(
            getattr(self.cfg, "left_arm_dof_names", []) or []
        )
        right_arm_dof_names = list(
            getattr(self.cfg, "right_arm_dof_names", []) or []
        )
        left_leg_dof_names = list(
            getattr(self.cfg, "left_leg_dof_names", []) or []
        )
        right_leg_dof_names = list(
            getattr(self.cfg, "right_leg_dof_names", []) or []
        )
        if not left_arm_dof_names:
            left_arm_dof_names = self._amp_filter_names_by_prefix(
                self.urdf_dof_names,
                "left_",
                ("shoulder", "elbow", "wrist"),
            )
        if not right_arm_dof_names:
            right_arm_dof_names = self._amp_filter_names_by_prefix(
                self.urdf_dof_names,
                "right_",
                ("shoulder", "elbow", "wrist"),
            )
        if not left_leg_dof_names:
            left_leg_dof_names = self._amp_filter_names_by_prefix(
                self.urdf_dof_names, "left_", ("hip", "knee", "ankle")
            )
        if not right_leg_dof_names:
            right_leg_dof_names = self._amp_filter_names_by_prefix(
                self.urdf_dof_names, "right_", ("hip", "knee", "ankle")
            )

        self._amp_left_arm_urdf_dof_idx = [
            urdf_dof_name_to_idx[name] for name in left_arm_dof_names
        ]
        self._amp_right_arm_urdf_dof_idx = [
            urdf_dof_name_to_idx[name] for name in right_arm_dof_names
        ]
        self._amp_left_leg_urdf_dof_idx = [
            urdf_dof_name_to_idx[name] for name in left_leg_dof_names
        ]
        self._amp_right_leg_urdf_dof_idx = [
            urdf_dof_name_to_idx[name] for name in right_leg_dof_names
        ]
        self._amp_left_arm_sim_dof_idx = [
            sim_dof_name_to_idx[name] for name in left_arm_dof_names
        ]
        self._amp_right_arm_sim_dof_idx = [
            sim_dof_name_to_idx[name] for name in right_arm_dof_names
        ]
        self._amp_left_leg_sim_dof_idx = [
            sim_dof_name_to_idx[name] for name in left_leg_dof_names
        ]
        self._amp_right_leg_sim_dof_idx = [
            sim_dof_name_to_idx[name] for name in right_leg_dof_names
        ]

        left_arm_body_names = list(
            getattr(self.cfg, "left_arm_body_names", []) or []
        )
        right_arm_body_names = list(
            getattr(self.cfg, "right_arm_body_names", []) or []
        )
        left_leg_body_names = list(
            getattr(self.cfg, "left_leg_body_names", []) or []
        )
        right_leg_body_names = list(
            getattr(self.cfg, "right_leg_body_names", []) or []
        )
        if not left_arm_body_names:
            left_arm_body_names = self._amp_filter_names_by_prefix(
                self.urdf_body_names, "left_", ("shoulder", "elbow", "wrist")
            )
        if not right_arm_body_names:
            right_arm_body_names = self._amp_filter_names_by_prefix(
                self.urdf_body_names, "right_", ("shoulder", "elbow", "wrist")
            )
        if not left_leg_body_names:
            left_leg_body_names = self._amp_filter_names_by_prefix(
                self.urdf_body_names, "left_", ("hip", "knee", "ankle")
            )
        if not right_leg_body_names:
            right_leg_body_names = self._amp_filter_names_by_prefix(
                self.urdf_body_names, "right_", ("hip", "knee", "ankle")
            )

        left_elbow_name = self._amp_pick_first_name(
            left_arm_body_names, ("left_elbow", "elbow")
        )
        right_elbow_name = self._amp_pick_first_name(
            right_arm_body_names, ("right_elbow", "elbow")
        )
        left_foot_name = self._amp_pick_first_name(
            left_leg_body_names,
            ("left_ankle_roll", "left_ankle_pitch", "left_ankle"),
        )
        right_foot_name = self._amp_pick_first_name(
            right_leg_body_names,
            ("right_ankle_roll", "right_ankle_pitch", "right_ankle"),
        )

        self._amp_left_elbow_urdf_body_idx = (
            urdf_body_name_to_idx[left_elbow_name]
            if left_elbow_name is not None
            else None
        )
        self._amp_right_elbow_urdf_body_idx = (
            urdf_body_name_to_idx[right_elbow_name]
            if right_elbow_name is not None
            else None
        )
        self._amp_left_foot_urdf_body_idx = (
            urdf_body_name_to_idx[left_foot_name]
            if left_foot_name is not None
            else None
        )
        self._amp_right_foot_urdf_body_idx = (
            urdf_body_name_to_idx[right_foot_name]
            if right_foot_name is not None
            else None
        )
        self._amp_left_elbow_sim_body_idx = (
            sim_body_name_to_idx[left_elbow_name]
            if left_elbow_name is not None
            else None
        )
        self._amp_right_elbow_sim_body_idx = (
            sim_body_name_to_idx[right_elbow_name]
            if right_elbow_name is not None
            else None
        )
        self._amp_left_foot_sim_body_idx = (
            sim_body_name_to_idx[left_foot_name]
            if left_foot_name is not None
            else None
        )
        self._amp_right_foot_sim_body_idx = (
            sim_body_name_to_idx[right_foot_name]
            if right_foot_name is not None
            else None
        )

        self._amp_left_hand_local_vec = torch.tensor(
            [0.0, 0.0, -0.3], device=self.device, dtype=torch.float32
        )
        self._amp_right_hand_local_vec = torch.tensor(
            [0.0, 0.0, -0.3], device=self.device, dtype=torch.float32
        )

    def _init_semantic_ik_reset_solver(self, motion_cfg) -> None:
        """Create the reset-only differentiable FK/IK solver once per rank."""

        if self._skeleton_adapter_spec is None:
            raise AssertionError(
                "Semantic IK reset requires a skeleton adapter"
            )
        robot_file_path = str(motion_cfg.get("fk_robot_file_path", "")).strip()
        if not robot_file_path:
            raise ValueError(
                "smpl_semantic_ik reset requires robot.motion."
                "fk_robot_file_path"
            )
        robot_file_path = os.path.abspath(robot_file_path)
        if not os.path.isfile(robot_file_path):
            raise FileNotFoundError(
                f"Semantic IK robot file does not exist: {robot_file_path}"
            )
        from holomotion.src.motion_tracking.semantic_ik_reset import (
            SemanticIKResetConfig,
            SemanticIKResetSolver,
        )

        solver_config = SemanticIKResetConfig.from_mapping(
            self._semantic_ik_reset_config_mapping
        )
        self._semantic_ik_reset_solver = SemanticIKResetSolver(
            robot_file_path=robot_file_path,
            adapter_spec=self._skeleton_adapter_spec,
            config=solver_config,
            device=self.device,
            dtype=torch.float32,
        )
        if tuple(self._semantic_ik_reset_solver.fk.dof_names) != tuple(
            self.urdf_dof_names
        ):
            raise ValueError(
                "Semantic IK URDF DoF order does not match command "
                "URDF order: "
                f"solver={self._semantic_ik_reset_solver.fk.dof_names}, "
                f"command={tuple(self.urdf_dof_names)}"
            )

    def _init_smpl_cycle_reset_runtime(self, motion_cfg) -> None:
        """Load the frozen P+A+R-to-q29 network and reset-only FK."""

        if self._skeleton_adapter_spec is None:
            raise AssertionError(
                "SMPL cycle-network reset requires a skeleton adapter"
            )
        robot_file_path = str(motion_cfg.get("fk_robot_file_path", "")).strip()
        if not robot_file_path:
            raise ValueError(
                "smpl_cycle_network reset requires robot.motion."
                "fk_robot_file_path"
            )
        robot_file_path = os.path.abspath(robot_file_path)
        if not os.path.isfile(robot_file_path):
            raise FileNotFoundError(
                "SMPL cycle-network robot file does not exist: "
                f"{robot_file_path}"
            )
        config = self._smpl_cycle_reset_config_mapping
        checkpoint_path = os.path.abspath(
            os.path.expandvars(
                os.path.expanduser(str(config.get("checkpoint", "")))
            )
        )
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "smpl_cycle_network reset checkpoint does not exist: "
                f"{checkpoint_path!r}"
            )
        expected_sha256 = (
            str(config.get("expected_sha256", "")).strip().lower()
        )
        if expected_sha256:
            digest = hashlib.sha256()
            with open(checkpoint_path, "rb") as checkpoint_file:
                for chunk in iter(
                    lambda: checkpoint_file.read(1024 * 1024), b""
                ):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "smpl_cycle_network reset checkpoint SHA256 mismatch: "
                    f"expected={expected_sha256}, actual={actual_sha256}, "
                    f"path={checkpoint_path}"
                )
        from holomotion.src.motion_tracking.smpl_cycle_reset import (
            SmplG1CycleResetRuntime,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        expected_step = config.get("expected_step", None)
        if expected_step is not None:
            actual_step = int(checkpoint.get("step", -1))
            if actual_step != int(expected_step):
                raise ValueError(
                    "smpl_cycle_network reset checkpoint step mismatch: "
                    f"expected={int(expected_step)}, actual={actual_step}, "
                    f"path={checkpoint_path}"
                )
        self._smpl_cycle_reset_runtime = SmplG1CycleResetRuntime(
            checkpoint=checkpoint,
            robot_file_path=robot_file_path,
            adapter_spec=self._skeleton_adapter_spec,
            expected_dof_names=tuple(self.urdf_dof_names),
            device=self.device,
            root_clearance_m=float(config.get("root_clearance_m", 0.03)),
            max_root_height_adjustment_m=float(
                config.get("max_root_height_adjustment_m", 0.30)
            ),
        ).eval()
        self._smpl_cycle_reset_runtime.requires_grad_(False)
        logger.info(
            "Loaded SMPL cycle reset checkpoint: path={}, step={}, "
            "quantizer={}, teacher_q_loss=disabled",
            checkpoint_path,
            checkpoint.get("step", "unknown"),
            checkpoint.get("model_config", {}).get("quantizer", "unknown"),
        )

    def _init_smpl_pose_grounding_runtime(self, motion_cfg) -> None:
        """Create one network-independent FK grounding runtime per rank."""

        if self._skeleton_adapter_spec is None:
            raise AssertionError(
                "SMPL pose grounding requires a skeleton adapter"
            )
        robot_file_path = str(motion_cfg.get("fk_robot_file_path", "")).strip()
        if not robot_file_path:
            raise ValueError(
                "pose_grounded reset requires robot.motion.fk_robot_file_path"
            )
        robot_file_path = os.path.abspath(robot_file_path)
        if not os.path.isfile(robot_file_path):
            raise FileNotFoundError(
                f"Pose-grounding robot file does not exist: {robot_file_path}"
            )
        from holomotion.src.motion_tracking.robot_pose_grounding import (
            RobotPoseGroundingRuntime,
        )

        config = self._smpl_pose_grounding_config_mapping
        q_default_sim = self.robot.data.default_joint_pos[0]
        q_default_urdf = q_default_sim[self.sim2urdf_dof_idx]
        self._smpl_pose_grounding_runtime = RobotPoseGroundingRuntime(
            robot_file_path=robot_file_path,
            adapter_spec=self._skeleton_adapter_spec,
            expected_dof_names=tuple(self.urdf_dof_names),
            q_default_urdf=q_default_urdf,
            device=self.device,
            root_clearance_m=float(config.get("root_clearance_m", 0.03)),
            max_root_height_adjustment_m=float(
                config.get("max_root_height_adjustment_m", 0.30)
            ),
        ).eval()
        self._smpl_pose_grounding_runtime.requires_grad_(False)

    def _init_buffers(self):
        self.metrics = {}
        self.ref_motion_global_frame_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # mark envs that timed out (frame id exceeded end frame) in current step
        self._motion_end_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        # counter for number of motion ends per environment
        self.motion_end_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # per-environment cached motion indices
        self._cached_motion_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # env -> cache row indirection (starts as identity mapping)
        self._env_to_cache_row = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._start_frame_indices = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._reward_sum_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._mpjpe_sum_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._mpkpe_sum_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._step_count_since_assign = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._completion_rate_sum_by_window: Dict[int, float] = {}
        self._completion_rate_count_by_window: Dict[int, int] = {}
        self._mpkpe_signal_sum_by_window: Dict[int, float] = {}
        self._mpkpe_signal_count_by_window: Dict[int, int] = {}

        self.pos_history_buffer = None
        self.rot_history_buffer = None
        self.ref_pos_history_buffer = None
        self.current_accel = None
        self.ref_body_accel = None
        self.current_ang_accel = None  # Placeholder for angular acceleration
        self._task_space_anchor_yaw_wxyz = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.float32
        )
        self._task_space_anchor_yaw_wxyz[:, 0] = 1.0
        self._task_space_anchor_translation_w = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self._task_space_anchor_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._task_space_target_next_cache: TaskSpaceState | None = None
        self._task_space_robot_state_cache: TaskSpaceState | None = None
        self._skeleton_target_next_cache: SemanticSkeletonState | None = None
        self._skeleton_robot_state_cache: SemanticSkeletonState | None = None
        self._task_space_robot_state_physics_step: int | None = None
        self._skeleton_robot_state_physics_step: int | None = None

        self.metrics["Task/MPJPE_WholeBody"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["Task/MPKPE_WholeBody"] = torch.zeros(
            self.num_envs, device=self.device
        )
        for metric_name in (
            "Reset/RootHeight",
            "Reset/RootSpeed",
            "Reset/RootAngularSpeed",
            "Reset/JointDefaultRMS",
            "Reset/JointVelocityRMS",
            "Reset/IKAccepted",
            "Reset/IKInitialObjective",
            "Reset/IKFinalObjective",
            "Reset/IKRelativeImprovement",
            "Reset/IKRootHeightAdjustment",
            "Reset/IKFallbackReason",
            "Reset/CycleRootHeightAdjustment",
            "Reset/CycleJointFinite",
            "Reset/CycleJointLimitClipRMS",
        ):
            self.metrics[metric_name] = torch.zeros(
                self.num_envs,
                device=self.device,
            )

    def _record_completion_rate_for_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        selected_clip_indices = self._clip_indices[env_ids]
        lengths = self._motion_cache.lengths_for_indices(selected_clip_indices)
        window_indices = self._motion_cache.window_indices_for_indices(
            selected_clip_indices
        )
        available_steps = torch.clamp(
            lengths
            - int(self.cfg.n_fut_frames)
            - self._start_frame_indices[env_ids],
            min=1,
        )
        completion_rate = torch.clamp(
            self._step_count_since_assign[env_ids] / available_steps.float(),
            min=0.0,
            max=1.0,
        )
        step_den = torch.clamp(self._step_count_since_assign[env_ids], min=1.0)
        mpkpe_mean = self._mpkpe_sum_since_assign[env_ids] / step_den
        completion_values = completion_rate.detach().cpu().tolist()
        mpkpe_values = mpkpe_mean.detach().cpu().tolist()
        window_values = window_indices.detach().cpu().tolist()
        for idx, window_index_obj in enumerate(window_values):
            completion_value = float(completion_values[idx])
            mpkpe_value = float(mpkpe_values[idx])
            mpkpe_signal = -mpkpe_value
            window_index = int(window_index_obj)

            if window_index in self._completion_rate_sum_by_window:
                self._completion_rate_sum_by_window[window_index] += (
                    completion_value
                )
                self._completion_rate_count_by_window[window_index] += 1
            else:
                self._completion_rate_sum_by_window[window_index] = (
                    completion_value
                )
                self._completion_rate_count_by_window[window_index] = 1

            if window_index in self._mpkpe_signal_sum_by_window:
                self._mpkpe_signal_sum_by_window[window_index] += mpkpe_signal
                self._mpkpe_signal_count_by_window[window_index] += 1
            else:
                self._mpkpe_signal_sum_by_window[window_index] = mpkpe_signal
                self._mpkpe_signal_count_by_window[window_index] = 1

        self._reward_sum_since_assign[env_ids] = 0.0
        self._mpjpe_sum_since_assign[env_ids] = 0.0
        self._mpkpe_sum_since_assign[env_ids] = 0.0
        self._step_count_since_assign[env_ids] = 0.0

    def _reset_window_curriculum_stats(self) -> None:
        self._completion_rate_sum_by_window = {}
        self._completion_rate_count_by_window = {}
        self._mpkpe_signal_sum_by_window = {}
        self._mpkpe_signal_count_by_window = {}

    def _build_window_curriculum_stats_from_current_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_window_indices = self._motion_cache.current_batch.window_indices
        row_window_indices = batch_window_indices.detach().to(
            self.device, dtype=torch.long
        )
        count = int(row_window_indices.numel())
        row_mpkpe_signal = torch.zeros(
            count, dtype=torch.float32, device=self.device
        )
        row_completion_rate = torch.zeros(
            count, dtype=torch.float32, device=self.device
        )
        row_count = torch.zeros(count, dtype=torch.float32, device=self.device)

        window_values = row_window_indices.detach().cpu().tolist()
        for row_idx, window_index_obj in enumerate(window_values):
            window_index = int(window_index_obj)
            completion_count = int(
                self._completion_rate_count_by_window.get(window_index, 0)
            )
            mpkpe_count = int(
                self._mpkpe_signal_count_by_window.get(window_index, 0)
            )
            if completion_count > 0:
                row_completion_rate[row_idx] = float(
                    self._completion_rate_sum_by_window[window_index]
                ) / float(completion_count)
            if mpkpe_count > 0:
                row_mpkpe_signal[row_idx] = float(
                    self._mpkpe_signal_sum_by_window[window_index]
                ) / float(mpkpe_count)
            row_count[row_idx] = float(max(completion_count, mpkpe_count))

        return (
            row_window_indices,
            row_mpkpe_signal,
            row_completion_rate,
            row_count,
        )

    def _update_cache_curriculum_state(
        self,
        *,
        accelerator,
        swap_index: int,
    ) -> None:
        if self._sampling_strategy != "curriculum":
            self._reset_window_curriculum_stats()
            return

        (
            row_window_indices,
            row_mpkpe_signal,
            row_completion_rate,
            row_count,
        ) = self._build_window_curriculum_stats_from_current_batch()

        if accelerator is not None and int(accelerator.num_processes) > 1:
            gather_window_indices = accelerator.gather(row_window_indices)
            gather_mpkpe_signal = accelerator.gather(row_mpkpe_signal)
            gather_completion_rate = accelerator.gather(row_completion_rate)
            gather_count = accelerator.gather(row_count)
        else:
            gather_window_indices = row_window_indices
            gather_mpkpe_signal = row_mpkpe_signal
            gather_completion_rate = row_completion_rate
            gather_count = row_count

        self._motion_cache.update_cache_curriculum(
            window_indices=gather_window_indices,
            mpkpe_signal_means=gather_mpkpe_signal,
            completion_rate_means=gather_completion_rate,
            counts=gather_count,
            swap_index=int(swap_index),
        )
        self._reset_window_curriculum_stats()

    def update_curriculum_reward_accumulators(
        self, rewards: torch.Tensor
    ) -> None:
        reward_flat = rewards.view(-1).to(self.device, dtype=torch.float32)
        all_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        motion_ids = self._filter_env_ids_for_motion_task(all_ids)
        if motion_ids.numel() == 0:
            return
        self._reward_sum_since_assign[motion_ids] += reward_flat[motion_ids]
        if self._task_space_target_source == "smpl_skeleton":
            mpjpe = self.metrics["Task/SkeletonSI_GlobalStructure"]
            mpkpe = self.metrics["Task/SkeletonSI_NormalizedPosition"]
        else:
            mpjpe = self.metrics["Task/MPJPE_WholeBody"]
            mpkpe = self.metrics["Task/MPKPE_WholeBody"]
        self._mpjpe_sum_since_assign[motion_ids] += mpjpe[motion_ids].to(
            dtype=torch.float32
        )
        self._mpkpe_sum_since_assign[motion_ids] += mpkpe[motion_ids].to(
            dtype=torch.float32
        )
        self._step_count_since_assign[motion_ids] += 1.0

    @property
    def command(
        self,
    ) -> torch.Tensor:
        # call the corresponding method based on configured command_obs_name
        return getattr(self, f"_get_obs_{self.cfg.command_obs_name}")()

    @property
    def command_fut(
        self,
    ) -> torch.Tensor:
        # call the corresponding method based on configured command_obs_name
        return getattr(self, f"_get_obs_{self.cfg.command_obs_name}_fut")()

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device)
        self._motion_end_mask[env_ids] = False
        self.motion_end_counter[env_ids] = 0

        # Do not apply cache swap inside per-env reset; defer to PPO barrier.
        # Always resample only the requested envs here.
        motion_ids = self._filter_env_ids_for_motion_task(env_ids.view(-1))
        self._resample_command(motion_ids, eval=self._is_evaluating)

        return extras

    def apply_cache_swap_if_pending_barrier(self, accelerator=None) -> bool:
        """Apply a pending cache swap at a rollout barrier.

        Returns:
            bool: True if a swap was applied, otherwise False.
        """
        if not getattr(self, "_swap_pending", False):
            return False

        all_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        motion_ids = self._filter_env_ids_for_motion_task(all_ids)
        if motion_ids.numel() == 0:
            # No motion envs active under multi-task: keep ref motion inert.
            self._swap_pending = False
            self._swap_step_counter = 0
            return False

        self._record_completion_rate_for_envs(motion_ids)
        next_swap_index = int(self._motion_cache.swap_index) + 1
        self._update_cache_curriculum_state(
            accelerator=accelerator,
            swap_index=next_swap_index,
        )

        # Advance cache and reset counters
        self._motion_cache.advance()
        self._maybe_dump_sampled_motion_keys()
        self._swap_pending = False
        self._swap_step_counter = 0

        # Reassign motion envs to the new cache batch
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            int(motion_ids.numel()),
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(self._is_evaluating),
        )
        self._clip_indices[motion_ids] = clip_idx
        self._frame_indices[motion_ids] = frame_idx
        self._start_frame_indices[motion_ids] = frame_idx
        self._task_space_anchor_valid[motion_ids] = False
        self._reward_sum_since_assign[motion_ids] = 0.0
        self._step_count_since_assign[motion_ids] = 0.0
        self._update_ref_motion_state_from_cache(env_ids=motion_ids)

        # Realign robot states to the newly assigned motion.
        self._realign_robot_state(motion_ids, perturb=True)

        # Reset per-episode timeout bookkeeping for consistency
        self._motion_end_mask[motion_ids] = False
        self.motion_end_counter[motion_ids] = 0
        return True

    def compute(self, dt: float):
        all_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        motion_ids = self._filter_env_ids_for_motion_task(all_ids)
        if motion_ids.numel() == 0:
            return
        self._update_metrics()
        self._update_command()

    def _update_ref_motion_state(self):
        """Update reference motion state (unified API)."""
        return self._update_ref_motion_state_from_cache()

    def _update_ref_motion_state_from_cache(
        self, env_ids: torch.Tensor | None = None
    ):
        """Invalidate derived values after any clip/frame reassignment.

        Data itself is gathered lazily. The previous no-op left old Actor
        targets cached during reset and cache-swap realignment.
        """
        del env_ids
        self._invalidate_observation_state_cache()

    def _invalidate_observation_state_cache(self) -> None:
        self._reference_observation_cache = {}
        self._task_space_target_next_cache = None
        self._task_space_robot_state_cache = None
        self._skeleton_target_next_cache = None
        self._skeleton_robot_state_cache = None

    def _finish_robot_state_write(self, env_ids: torch.Tensor) -> None:
        # Composed reset obtains canonical velocities through Actor terms
        # before writing the new root pose. Robot-relative yaw/orientation
        # terms from that operation cannot be reused by the next observation.
        self._invalidate_observation_state_cache()
        self._update_reset_diagnostics_from_robot(env_ids)

    def _get_ref_state_array(
        self,
        base_key: str,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Gather a reference tensor from the current cache batch.

        Args:
            base_key: Base key in the motion cache (e.g. \"dof_pos\", \"root_pos\").
            prefix: Optional logical prefix (e.g. \"\", \"ref_\", \"robot_\").

        Returns:
            Tensor of shape ``[num_envs, 1 + n_fut_frames, ...]`` gathered for
            the envs' current clip/frame assignments.
        """
        if (
            base_key in {"dof_vel", "root_vel", "root_ang_vel"}
            and prefix == "ref_"
        ):
            kinematics = self._get_reference_observation_kinematics(prefix)
            return {
                "dof_vel": kinematics.dof_vel,
                "root_vel": kinematics.root_linvel_world,
                "root_ang_vel": kinematics.root_angvel_world,
            }[base_key]

        batch_tensors = self._motion_cache.current_batch.tensors
        tensor_key = resolve_reference_tensor_key(
            batch_tensors=batch_tensors,
            base_key=base_key,
            prefix=prefix,
        )
        return self._motion_cache.gather_tensor(
            tensor_key,
            clip_indices=self._clip_indices,
            frame_indices=self._frame_indices,
            n_future_frames=self.cfg.n_fut_frames,
        )

    def _get_reference_observation_kinematics(
        self, prefix: str
    ) -> ReferenceKinematics:
        cache = getattr(self, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            self._reference_observation_cache = cache
        cached = cache.get(prefix)
        if cached is not None:
            return cached

        reference_qpos = self._get_reference_qpos_with_context(prefix)
        full = derive_reference_kinematics_torch(
            reference_qpos,
            fps=float(self.cfg.target_fps),
        )
        end = 2 + int(self.cfg.n_fut_frames)
        result = full.sliced((slice(None), slice(1, end)))
        cache[prefix] = result
        return result

    def _get_reference_qpos_with_context(self, prefix: str) -> torch.Tensor:
        cache = getattr(self, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            self._reference_observation_cache = cache
        cache_key = f"qpos:{prefix}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        batch_tensors = self._motion_cache.current_batch.tensors
        keys = {
            name: resolve_reference_tensor_key(
                batch_tensors=batch_tensors,
                base_key=name,
                prefix=prefix,
            )
            for name in ("dof_pos", "root_pos", "root_rot")
        }
        past_frames = torch.clamp(self._frame_indices - 1, min=0)
        extra_future = int(self.cfg.n_fut_frames) + 1

        def gather_with_past(name: str) -> torch.Tensor:
            past = self._motion_cache.gather_tensor(
                keys[name],
                clip_indices=self._clip_indices,
                frame_indices=past_frames,
                n_future_frames=0,
            )
            current_future = self._motion_cache.gather_tensor(
                keys[name],
                clip_indices=self._clip_indices,
                frame_indices=self._frame_indices,
                n_future_frames=extra_future,
            )
            return torch.cat([past, current_future], dim=1)

        reference_qpos = torch.cat(
            [
                gather_with_past("root_pos"),
                gather_with_past("root_rot")[..., [3, 0, 1, 2]],
                gather_with_past("dof_pos"),
            ],
            dim=-1,
        ).contiguous()
        cache[cache_key] = reference_qpos
        return reference_qpos

    def get_motion_actor_observation_term(
        self,
        name: str,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        cache = getattr(self, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            self._reference_observation_cache = cache
        cache_key = f"actor_terms:{prefix}"
        terms = cache.get(cache_key)
        if terms is None:
            reference_indices = torch.as_tensor(
                self.urdf2sim_dof_idx,
                device=self.device,
                dtype=torch.int32,
            )
            terms = derive_motion_actor_terms_torch(
                MotionActorObservationInput(
                    reference_qpos=self._get_reference_qpos_with_context(
                        prefix
                    ),
                    robot_root_quat_wxyz=self.robot.data.root_quat_w,
                    robot_root_angvel_local=self.robot.data.root_ang_vel_b,
                    robot_dof_pos=self.robot.data.joint_pos,
                    robot_dof_vel=self.robot.data.joint_vel,
                    last_action=self._env.action_manager.action,
                    default_dof_pos=self.robot.data.default_joint_pos,
                    reference_dof_indices=reference_indices,
                ),
                fps=float(self.cfg.target_fps),
                current_index=1,
                num_future_frames=int(self.cfg.n_fut_frames),
            )
            cache[cache_key] = terms
        if name not in terms:
            raise KeyError(f"Unsupported shared motion actor term: {name}")
        return terms[name]

    def _get_smpl_root_state_with_context(self) -> torch.Tensor:
        """Gather SMPL root at [r-1, r, ..., r+n_future+1].

        The returned tensor contains only position and WXYZ orientation.  Pure
        SMPL observation construction must not synthesize a robot-specific
        q29/q22 reference pose.
        """

        cache = getattr(self, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            self._reference_observation_cache = cache
        cache_key = "smpl:root_state_with_context"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        batch_tensors = self._motion_cache.current_batch.tensors
        required = ("smpl_root_pos", "smpl_root_rot", "smpl_body_command")
        missing = [name for name in required if name not in batch_tensors]
        if missing:
            raise KeyError(
                "Pure-SMPL Actor requires SMPL cache tensors; missing "
                f"{missing}"
            )
        past_frames = torch.clamp(self._frame_indices - 1, min=0)
        extra_future = int(self.cfg.n_fut_frames) + 1

        def gather_with_past(name: str) -> torch.Tensor:
            past = self._motion_cache.gather_tensor(
                name,
                clip_indices=self._clip_indices,
                frame_indices=past_frames,
                n_future_frames=0,
            )
            current_future = self._motion_cache.gather_tensor(
                name,
                clip_indices=self._clip_indices,
                frame_indices=self._frame_indices,
                n_future_frames=extra_future,
            )
            return torch.cat((past, current_future), dim=1)

        root_pos = gather_with_past("smpl_root_pos")
        root_rot_wxyz = gather_with_past("smpl_root_rot")[..., [3, 0, 1, 2]]
        root_state = torch.cat((root_pos, root_rot_wxyz), dim=-1).contiguous()
        cache[cache_key] = root_state
        return root_state

    def _get_smpl_body_current_future(self) -> torch.Tensor:
        cache = getattr(self, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            self._reference_observation_cache = cache
        cache_key = "smpl:body_current_future"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        body = self._motion_cache.gather_tensor(
            "smpl_body_command",
            clip_indices=self._clip_indices,
            frame_indices=self._frame_indices,
            n_future_frames=int(self.cfg.n_fut_frames),
        )
        cache[cache_key] = body
        return body

    def _get_smpl_actor_canonical_root_rotation_current(self) -> torch.Tensor:
        """Return the exact segment-canonical SMPL root rotation seen by Actor.

        ``smpl_root_rot`` is constructed once by :class:`SmplReferenceBuilder`
        after applying ``SMPL_TO_COMMAND_FRAME`` and removing the current
        dataset-window's first-frame yaw.  Reset must gather that cached tensor
        rather than reimplementing the canonicalization from the raw skeleton
        target, otherwise Actor and Reset silently use different world frames.
        """

        rotation_xyzw = self._motion_cache.gather_tensor(
            "smpl_root_rot",
            clip_indices=self._clip_indices,
            frame_indices=self._frame_indices,
            n_future_frames=0,
        )
        if rotation_xyzw.ndim != 3 or rotation_xyzw.shape[1:] != (1, 4):
            raise ValueError(
                "Cached smpl_root_rot must have shape [B,1,4], got "
                f"{tuple(rotation_xyzw.shape)}"
            )
        return rotation_xyzw[:, 0, [3, 0, 1, 2]].contiguous()

    def _get_smpl_actor_canonical_root_state_current(
        self,
    ) -> CanonicalSmplRootState:
        """Return one internally coherent current SMPL root target.

        Position, orientation and local velocities all come from the exact
        canonical root cache consumed by Actor.  Only nominal morphology scale
        is read from the semantic skeleton target.
        """

        root_state = self._get_smpl_root_state_with_context()
        if (
            root_state.ndim != 3
            or root_state.shape[1] < 3
            or root_state.shape[-1] != 7
        ):
            raise ValueError(
                "Canonical SMPL root context must have shape [B,T>=3,7], "
                f"got {tuple(root_state.shape)}"
            )
        terms = {
            name: self.get_smpl_actor_observation_term(name)
            for name in (
                "actor_smpl_base_linvel_cur",
                "actor_smpl_base_angvel_cur",
            )
        }
        skeleton = self._get_smpl_skeleton_state(immediate_next=False)
        yaw_current = _yaw_from_quat_wxyz(root_state[:, 1, 3:7])
        yaw_next = _yaw_from_quat_wxyz(root_state[:, 2, 3:7])
        yaw_rate = wrap_to_pi(yaw_next - yaw_current) * float(
            self.cfg.target_fps
        )
        return CanonicalSmplRootState(
            position=root_state[:, 1, :3].contiguous(),
            rotation_wxyz=root_state[:, 1, 3:7].contiguous(),
            linear_velocity_local=terms[
                "actor_smpl_base_linvel_cur"
            ].contiguous(),
            angular_velocity_local=terms[
                "actor_smpl_base_angvel_cur"
            ].contiguous(),
            yaw_rate=yaw_rate.contiguous(),
            nominal_scale=skeleton.nominal_scale.contiguous(),
        )

    def get_smpl_actor_observation_term(self, name: str) -> torch.Tensor:
        """Return an Actor-only pure-SMPL command term.

        This intentionally has no prefix parameter, so it cannot masquerade as
        ``ref_*`` and leak into reward, critic, reset, or termination paths.
        """

        cache = getattr(self, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            self._reference_observation_cache = cache
        cache_key = "smpl:actor_terms"
        terms = cache.get(cache_key)
        if terms is None:
            from holomotion.src.motion_tracking.smpl_actor_observation import (
                SmplActorObservationInput,
                derive_smpl_actor_terms_torch,
            )

            terms = derive_smpl_actor_terms_torch(
                SmplActorObservationInput(
                    root_state_with_context=self._get_smpl_root_state_with_context(),
                    body_current_future=self._get_smpl_body_current_future(),
                    robot_root_quat_wxyz=self.robot.data.root_quat_w,
                ),
                fps=float(self.cfg.target_fps),
                current_index=1,
                num_future_frames=int(self.cfg.n_fut_frames),
            )
            cache[cache_key] = terms
        if name not in terms:
            raise KeyError(f"Unsupported pure-SMPL Actor term: {name}")
        return terms[name]

    def _uniform_sample_ref_start_frames(self, env_ids: torch.Tensor):
        """Uniformly sample start frames within cached windows for env_ids.

        Sampling range is [start, end - 1 - n_fut_frames] to ensure required
        future frames exist. If that upper bound is < start, it falls back to start.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device).long()

        starts = self.ref_motion_global_start_frame_ids[env_ids]
        ends = self.ref_motion_global_end_frame_ids[env_ids]

        # Ensure room for future frames if requested
        n_fut = (
            int(self.cfg.n_fut_frames)
            if hasattr(self.cfg, "n_fut_frames")
            else 0
        )
        max_start = ends - 1 - n_fut
        max_start = torch.maximum(max_start, starts)

        num_choices = (max_start - starts + 1).clamp(min=1)
        # Sample offsets uniformly
        rand = torch.rand_like(starts, dtype=torch.float32)
        offsets = torch.floor(rand * num_choices.float()).long()
        sampled = starts + offsets

        self.ref_motion_global_frame_ids[env_ids] = sampled

    def get_ref_motion_dof_pos_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("dof_pos", prefix)
        return base[:, 1:, ...][..., self.urdf2sim_dof_idx]

    def _get_immediate_next_ref_state_array(
        self,
        base_key: str,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array(base_key, prefix)
        if base.shape[1] < 2:
            raise ValueError(
                f"Immediate-next reference for '{base_key}' requires at least one future frame."
            )
        return base[:, 1, ...]

    def _build_task_space_state_from_body_tensors(
        self,
        *,
        body_position_w: torch.Tensor,
        body_rotation_w_wxyz: torch.Tensor,
        body_linear_velocity_w: torch.Tensor,
        body_angular_velocity_w: torch.Tensor,
    ) -> TaskSpaceState:
        """Select G1 semantic links and construct the two virtual toes."""

        indices = self._task_space_body_indices

        def select(tensor: torch.Tensor, name: str) -> torch.Tensor:
            return tensor[:, indices[name]]

        toe_offset = self._task_space_toe_offset.to(
            device=body_position_w.device,
            dtype=body_position_w.dtype,
        )

        def virtual_toe(side: str):
            ankle_name = f"{side}_ankle_roll_link"
            ankle_pos = select(body_position_w, ankle_name)
            ankle_rot = select(body_rotation_w_wxyz, ankle_name)
            ankle_lin = select(body_linear_velocity_w, ankle_name)
            ankle_ang = select(body_angular_velocity_w, ankle_name)
            world_offset = isaaclab_math.quat_apply(
                ankle_rot,
                toe_offset.unsqueeze(0).expand(ankle_rot.shape[0], -1),
            )
            return (
                ankle_pos + world_offset,
                ankle_rot,
                ankle_lin + torch.cross(ankle_ang, world_offset, dim=-1),
                ankle_ang,
            )

        left_toe = virtual_toe("left")
        right_toe = virtual_toe("right")
        body_names = (
            "pelvis",
            "left_hip_roll_link",
            "right_hip_roll_link",
            "left_knee_link",
            "right_knee_link",
        )
        trailing_names = (
            "torso_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        )

        def stack_field(
            tensor: torch.Tensor,
            toe_field: int,
        ) -> torch.Tensor:
            values = [select(tensor, name) for name in body_names]
            values.extend((left_toe[toe_field], right_toe[toe_field]))
            values.extend(select(tensor, name) for name in trailing_names)
            return torch.stack(values, dim=1)

        return TaskSpaceState(
            position_w=stack_field(body_position_w, 0),
            rotation_w_wxyz=stack_field(body_rotation_w_wxyz, 1),
            linear_velocity_w=stack_field(body_linear_velocity_w, 2),
            angular_velocity_w=stack_field(body_angular_velocity_w, 3),
        )

    def _get_robot_reference_task_space_state(
        self, *, immediate_next: bool
    ) -> TaskSpaceState:
        return self._get_cached_task_space_state(
            cache_keys=REF_TASK_SPACE_CACHE_KEYS,
            label="robot-reference",
            immediate_next=immediate_next,
        )

    def _get_cached_task_space_state(
        self,
        *,
        cache_keys: dict[str, str],
        label: str,
        immediate_next: bool,
    ) -> TaskSpaceState:
        future_frames = 1 if immediate_next else 0
        slot = 1 if immediate_next else 0

        def gather(name: str) -> torch.Tensor:
            tensor = self._motion_cache.gather_tensor(
                cache_keys[name],
                clip_indices=self._clip_indices,
                frame_indices=self._frame_indices,
                n_future_frames=future_frames,
            )
            if tensor.shape[1] <= slot:
                raise ValueError(
                    f"{label} task target '{name}' is missing slot {slot}"
                )
            return tensor[:, slot]

        rotation_xyzw = gather("rotation")
        return TaskSpaceState(
            position_w=gather("position"),
            rotation_w_wxyz=rotation_xyzw[..., [3, 0, 1, 2]],
            linear_velocity_w=gather("linear_velocity"),
            angular_velocity_w=gather("angular_velocity"),
        )

    def _get_smpl_task_space_state(
        self, *, immediate_next: bool
    ) -> TaskSpaceState:
        return self._get_cached_task_space_state(
            cache_keys=TASK_SPACE_CACHE_KEYS,
            label="SMPL",
            immediate_next=immediate_next,
        )

    def _get_smpl_skeleton_state(
        self,
        *,
        immediate_next: bool,
    ) -> SemanticSkeletonState:
        future_frames = 1 if immediate_next else 0
        slot = 1 if immediate_next else 0

        def gather(name: str) -> torch.Tensor:
            tensor = self._motion_cache.gather_tensor(
                SKELETON_CACHE_KEYS[name],
                clip_indices=self._clip_indices,
                frame_indices=self._frame_indices,
                n_future_frames=future_frames,
            )
            if tensor.shape[1] <= slot:
                raise ValueError(
                    f"SMPL skeleton target {name!r} is missing slot {slot}"
                )
            return tensor[:, slot]

        def wxyz(name: str) -> torch.Tensor:
            xyzw = gather(name)
            return xyzw[..., [3, 0, 1, 2]]

        return SemanticSkeletonState(
            position_w=gather("position"),
            pelvis_rotation_wxyz=wxyz("pelvis_rotation"),
            torso_rotation_wxyz=wxyz("torso_rotation"),
            endpoint_rotation_wxyz=wxyz("endpoint_rotation"),
            head_rotation_wxyz=wxyz("head_rotation"),
            root_linear_velocity_w=gather("root_linear_velocity"),
            root_angular_velocity_w=gather("root_angular_velocity"),
            nominal_scale=gather("nominal_scale"),
        )

    def _get_task_space_source_state(
        self, *, immediate_next: bool
    ) -> TaskSpaceState:
        if self._task_space_target_source == "robot_reference":
            return self._get_robot_reference_task_space_state(
                immediate_next=immediate_next
            )
        if self._task_space_target_source == "smpl":
            return self._get_smpl_task_space_state(
                immediate_next=immediate_next
            )
        raise RuntimeError(
            "Task-space target requested while "
            "robot.motion.task_target_source='none'"
        )

    def _apply_skeleton_anchor(
        self,
        state: SemanticSkeletonState,
    ) -> SemanticSkeletonState:
        if not bool(torch.all(self._task_space_anchor_valid)):
            invalid = torch.nonzero(
                ~self._task_space_anchor_valid,
                as_tuple=False,
            ).flatten()
            raise RuntimeError(
                "Skeleton episode anchor is not initialized for envs "
                f"{invalid[:16].detach().cpu().tolist()}"
            )
        anchor = self._task_space_anchor_yaw_wxyz
        anchor_nodes = anchor[:, None, :].expand(
            -1,
            state.position_w.shape[1],
            -1,
        )
        translation = self._task_space_anchor_translation_w[:, None, :]
        return SemanticSkeletonState(
            position_w=(
                isaaclab_math.quat_apply(anchor_nodes, state.position_w)
                + translation
            ),
            pelvis_rotation_wxyz=isaaclab_math.quat_mul(
                anchor,
                state.pelvis_rotation_wxyz,
            ),
            torso_rotation_wxyz=isaaclab_math.quat_mul(
                anchor,
                state.torso_rotation_wxyz,
            ),
            endpoint_rotation_wxyz=isaaclab_math.quat_mul(
                anchor[:, None, :].expand(
                    -1,
                    state.endpoint_rotation_wxyz.shape[1],
                    -1,
                ),
                state.endpoint_rotation_wxyz,
            ),
            head_rotation_wxyz=(
                None
                if state.head_rotation_wxyz is None
                else isaaclab_math.quat_mul(
                    anchor,
                    state.head_rotation_wxyz,
                )
            ),
            root_linear_velocity_w=isaaclab_math.quat_apply(
                anchor,
                state.root_linear_velocity_w,
            ),
            root_angular_velocity_w=isaaclab_math.quat_apply(
                anchor,
                state.root_angular_velocity_w,
            ),
            nominal_scale=state.nominal_scale,
        )

    def get_semantic_skeleton_target_immediate_next(
        self,
    ) -> SemanticSkeletonState:
        if self._task_space_target_source != "smpl_skeleton":
            raise RuntimeError(
                "Semantic skeleton target requires "
                "task_target_source='smpl_skeleton'"
            )
        if self._skeleton_target_next_cache is None:
            self._skeleton_target_next_cache = self._apply_skeleton_anchor(
                self._get_smpl_skeleton_state(immediate_next=True)
            )
        return self._skeleton_target_next_cache

    def get_semantic_skeleton_target_current(
        self,
    ) -> SemanticSkeletonState:
        """Return the anchored SMPL skeleton target at the current frame."""
        if self._task_space_target_source != "smpl_skeleton":
            raise RuntimeError(
                "Semantic skeleton target requires "
                "task_target_source='smpl_skeleton'"
            )
        return self._apply_skeleton_anchor(
            self._get_smpl_skeleton_state(immediate_next=False)
        )

    def is_semantic_skeleton_anchor_ready(self) -> bool:
        """Return whether every environment has completed episode anchoring.

        IsaacLab probes observation functions once while constructing its
        ObservationManager, before the first environment reset establishes
        the episode anchor.  Observation terms may use this read-only guard to
        return a correctly shaped bootstrap value during that probe; normal
        target access remains strict and still raises on an invalid anchor.
        """
        return bool(torch.all(self._task_space_anchor_valid))

    def get_semantic_skeleton_progress_token(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the read-only motion assignment used by temporal rewards."""
        return (
            self._clip_indices,
            self._frame_indices,
            self._start_frame_indices,
        )

    def get_semantic_skeleton_robot_state(self) -> SemanticSkeletonState:
        if self._task_space_target_source != "smpl_skeleton":
            raise RuntimeError(
                "Semantic skeleton robot state requires "
                "task_target_source='smpl_skeleton'"
            )
        # The critic also requests this state after CommandManager.compute.
        # Consequently command-update invalidation alone is insufficient:
        # next step's physics runs before rewards/terminations consume it.
        physics_step = self._env._sim_step_counter
        if (
            self._skeleton_robot_state_cache is not None
            and self._skeleton_robot_state_physics_step == physics_step
        ):
            return self._skeleton_robot_state_cache
        if self._skeleton_adapter_spec is None:
            raise AssertionError("Missing semantic skeleton adapter spec")
        data = self.robot.data
        if not hasattr(data, "body_link_pos_w") or not hasattr(
            data,
            "body_link_quat_w",
        ):
            raise AttributeError(
                "IsaacLab link-origin position/orientation tensors are required"
            )
        if not hasattr(data, "root_link_lin_vel_w") or not hasattr(
            data,
            "root_link_ang_vel_w",
        ):
            raise AttributeError("IsaacLab root link velocities are required")
        self._skeleton_robot_state_cache = build_robot_skeleton_state(
            spec=self._skeleton_adapter_spec,
            body_name_to_index=self._skeleton_body_indices,
            body_position_w=data.body_link_pos_w,
            body_rotation_wxyz=data.body_link_quat_w,
            root_linear_velocity_w=data.root_link_lin_vel_w,
            root_angular_velocity_w=data.root_link_ang_vel_w,
        )
        self._skeleton_robot_state_physics_step = physics_step
        return self._skeleton_robot_state_cache

    def _apply_task_space_anchor(
        self, state: TaskSpaceState
    ) -> TaskSpaceState:
        if not bool(torch.all(self._task_space_anchor_valid)):
            invalid = torch.nonzero(
                ~self._task_space_anchor_valid, as_tuple=False
            ).flatten()
            raise RuntimeError(
                "Task-space episode anchor is not initialized for envs "
                f"{invalid[:16].detach().cpu().tolist()}"
            )
        anchor = self._task_space_anchor_yaw_wxyz
        anchor_nodes = anchor[:, None, :].expand(
            -1, len(TASK_SPACE_NODE_NAMES), -1
        )
        translation = self._task_space_anchor_translation_w[:, None, :]
        return TaskSpaceState(
            position_w=(
                isaaclab_math.quat_apply(anchor_nodes, state.position_w)
                + translation
            ),
            rotation_w_wxyz=isaaclab_math.quat_mul(
                anchor_nodes, state.rotation_w_wxyz
            ),
            linear_velocity_w=isaaclab_math.quat_apply(
                anchor_nodes, state.linear_velocity_w
            ),
            angular_velocity_w=isaaclab_math.quat_apply(
                anchor_nodes, state.angular_velocity_w
            ),
        )

    def _refresh_task_space_anchor_from_written_root(
        self,
        env_ids: torch.Tensor,
        *,
        root_position_w: torch.Tensor,
        root_rotation_w_wxyz: torch.Tensor,
    ) -> None:
        """Freeze one XY/yaw anchor after the robot root is written."""

        if self._task_space_target_source == "none" or env_ids.numel() == 0:
            return
        if self._task_space_target_source == "smpl_skeleton":
            source_skeleton = self._get_smpl_skeleton_state(
                immediate_next=False
            )
            source_position = source_skeleton.position_w[env_ids, 0]
            source_rotation = source_skeleton.pelvis_rotation_wxyz[env_ids]
        else:
            source = self._get_task_space_source_state(immediate_next=False)
            source_position = source.position_w[env_ids, 0]
            source_rotation = source.rotation_w_wxyz[env_ids, 0]
        yaw_delta = _yaw_from_quat_wxyz(root_rotation_w_wxyz) - (
            _yaw_from_quat_wxyz(source_rotation)
        )
        half_yaw = 0.5 * yaw_delta
        anchor = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=-1,
        )
        rotated_source = isaaclab_math.quat_apply(anchor, source_position)
        translation = torch.zeros_like(root_position_w)
        translation[:, :2] = root_position_w[:, :2] - rotated_source[:, :2]
        self._task_space_anchor_yaw_wxyz[env_ids] = anchor
        self._task_space_anchor_translation_w[env_ids] = translation
        self._task_space_anchor_valid[env_ids] = True
        self._task_space_target_next_cache = None
        self._task_space_robot_state_cache = None
        self._skeleton_target_next_cache = None
        self._skeleton_robot_state_cache = None

    def get_task_space_target_current(self) -> TaskSpaceState:
        """Return the anchored task target used by the current Actor frame."""

        return self._apply_task_space_anchor(
            self._get_task_space_source_state(immediate_next=False)
        )

    def get_task_space_target_immediate_next(self) -> TaskSpaceState:
        """Return the anchored ``r+1`` target consumed by rewards."""

        if self._task_space_target_next_cache is None:
            self._task_space_target_next_cache = self._apply_task_space_anchor(
                self._get_task_space_source_state(immediate_next=True)
            )
        return self._task_space_target_next_cache

    def get_task_space_robot_state(self) -> TaskSpaceState:
        """Return the realized G1 semantic state using link-origin velocity."""

        physics_step = self._env._sim_step_counter
        if (
            self._task_space_robot_state_cache is not None
            and self._task_space_robot_state_physics_step == physics_step
        ):
            return self._task_space_robot_state_cache
        data = self.robot.data
        if not hasattr(data, "body_link_lin_vel_w"):
            raise AttributeError(
                "IsaacLab ArticulationData.body_link_lin_vel_w is required; "
                "body_lin_vel_w is a COM-velocity alias and is intentionally "
                "unsupported for task-space rewards"
            )
        state = self._build_task_space_state_from_body_tensors(
            body_position_w=data.body_link_pos_w,
            body_rotation_w_wxyz=data.body_link_quat_w,
            body_linear_velocity_w=data.body_link_lin_vel_w,
            body_angular_velocity_w=data.body_link_ang_vel_w,
        )
        if not hasattr(data, "root_link_lin_vel_w") or not hasattr(
            data, "root_link_ang_vel_w"
        ):
            raise AttributeError(
                "IsaacLab ArticulationData root link velocities are required"
            )
        linear_velocity = state.linear_velocity_w.clone()
        angular_velocity = state.angular_velocity_w.clone()
        linear_velocity[:, 0] = data.root_link_lin_vel_w
        angular_velocity[:, 0] = data.root_link_ang_vel_w
        self._task_space_robot_state_cache = TaskSpaceState(
            position_w=state.position_w,
            rotation_w_wxyz=state.rotation_w_wxyz,
            linear_velocity_w=linear_velocity,
            angular_velocity_w=angular_velocity,
        )
        self._task_space_robot_state_physics_step = physics_step
        return self._task_space_robot_state_cache

    def get_ref_motion_dof_vel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("dof_vel", prefix)
        return base[:, 1:, ...][..., self.urdf2sim_dof_idx]

    def get_ref_motion_root_global_pos_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_pos", prefix)
        return base[:, 1:, ...] + self._env_origins[:, None, :]

    def get_ref_motion_root_global_rot_quat_xyzw_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_ref_state_array("root_rot", prefix)[:, 1:, ...]

    def get_ref_motion_root_global_rot_quat_wxyz_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self.get_ref_motion_root_global_rot_quat_xyzw_fut(
            prefix=prefix
        )[..., [3, 0, 1, 2]]

    def get_ref_motion_root_global_lin_vel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_vel", prefix)
        return base[:, 1:, ...]

    def get_ref_motion_root_global_ang_vel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_ang_vel", prefix)
        return base[:, 1:, ...]

    def get_ref_motion_bodylink_global_pos_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("rg_pos", prefix)
        return (
            base[:, 1:, ...][..., self.urdf2sim_body_idx, :]
            + self._env_origins[:, None, None, :]
        )

    def get_ref_motion_bodylink_rel_pos_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        ref_body_global_pos = self.get_ref_motion_bodylink_global_pos_cur(
            prefix=prefix
        )  # [B, N, 3]
        ref_root_global_pos = self.get_ref_motion_root_global_pos_cur(
            prefix=prefix
        )  # [B, 3]
        ref_root_global_rot_wxyz = (
            self.get_ref_motion_root_global_rot_quat_wxyz_cur(prefix=prefix)
        )  # [B, 4]
        rel_pos_w = (
            ref_body_global_pos - ref_root_global_pos[:, None, :]
        )  # [B, N, 3]
        num_bodies = rel_pos_w.shape[1]
        expanded_ref_root_global_rot_wxyz = ref_root_global_rot_wxyz[
            :, None, :
        ].expand(-1, num_bodies, -1)
        return isaaclab_math.quat_apply_inverse(
            expanded_ref_root_global_rot_wxyz, rel_pos_w
        )  # [B, N, 3]

    def get_ref_motion_bodylink_rel_pos_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        ref_body_global_pos_fut = self.get_ref_motion_bodylink_global_pos_fut(
            prefix=prefix
        )  # [B, T, N, 3]
        ref_root_global_pos_fut = self.get_ref_motion_root_global_pos_fut(
            prefix=prefix
        )  # [B, T, 3]
        ref_root_global_rot_wxyz_fut = (
            self.get_ref_motion_root_global_rot_quat_wxyz_fut(prefix=prefix)
        )  # [B, T, 4]
        rel_pos_w_fut = (
            ref_body_global_pos_fut - ref_root_global_pos_fut[:, :, None, :]
        )  # [B, T, N, 3]
        num_bodies = rel_pos_w_fut.shape[2]
        expanded_ref_root_global_rot_wxyz_fut = ref_root_global_rot_wxyz_fut[
            :, :, None, :
        ].expand(-1, -1, num_bodies, -1)
        return isaaclab_math.quat_apply_inverse(
            expanded_ref_root_global_rot_wxyz_fut, rel_pos_w_fut
        )  # [B, T, N, 3]

    def get_ref_motion_bodylink_global_rot_xyzw_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("rb_rot", prefix)
        return base[:, 1:, ...][..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_lin_vel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("body_vel", prefix)
        return base[:, 1:, ...][..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_ang_vel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("body_ang_vel", prefix)
        return base[:, 1:, ...][..., self.urdf2sim_body_idx, :]

    def get_ref_motion_dof_pos_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("dof_pos", prefix)
        return base[:, 0, ...][..., self.urdf2sim_dof_idx]

    def get_ref_motion_dof_pos_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("dof_pos", prefix)
        return base[..., self.urdf2sim_dof_idx]

    def get_immediate_next_two_dof_pos(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Immediate next two DoF positions in simulator DoF order."""
        n_fut = int(self.cfg.n_fut_frames)
        if n_fut < 1:
            raise ValueError(
                "n_fut_frames must be at least 1 for immediate next two DoF positions."
            )
        base = self._get_ref_state_array("dof_pos", prefix)
        return base[:, :2, ...][..., self.urdf2sim_dof_idx]

    def get_ref_motion_dof_pos_cur_urdf_order(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("dof_pos", prefix)
        return base[:, 0, ...]

    def get_ref_motion_cur_heading_aligned_root_pos(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        # prepare current frame robot root global poses
        robot_cur_global_root_pos = self.robot.data.root_pos_w
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # wxyz
        yaw_quat = isaaclab_math.yaw_quat(robot_cur_global_root_rot)

        # transform the current goal frame root poses into the relative heading aligned frame
        global_pos_diff = (
            self.get_ref_motion_root_global_pos_cur(prefix=prefix)
            - robot_cur_global_root_pos
        )
        global_pos_diff_heading_aligned = isaaclab_math.quat_apply_inverse(
            yaw_quat, global_pos_diff
        )
        return global_pos_diff_heading_aligned

    def get_ref_motion_fut_heading_aligned_root_pos(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        # prepare current frame robot root global poses
        robot_cur_global_root_pos = self.robot.data.root_pos_w  # [B, 3]
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4]
        yaw_quat = isaaclab_math.yaw_quat(robot_cur_global_root_rot)  # [B, 4]

        # transform the current goal frame root poses into the relative heading aligned frame
        fut_root_global_pos = self.get_ref_motion_root_global_pos_fut(
            prefix=prefix
        )  # [B, T, 3]
        num_fut_frames = fut_root_global_pos.shape[1]
        global_pos_diff = (
            fut_root_global_pos - robot_cur_global_root_pos[:, None, :]
        )  # [B, T, 3]
        expanded_yaw_quat = yaw_quat[:, None, :].expand(
            -1, num_fut_frames, -1
        )  # [B, T, 4]
        fut_root_global_pos_heading_aligned = isaaclab_math.quat_apply_inverse(
            expanded_yaw_quat, global_pos_diff
        )  # [B, T, 3]
        return fut_root_global_pos_heading_aligned

    def get_ref_motion_cur_heading_aligned_root_rot6d(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference root rotation (rot6d) in heading-aligned frame.

        Returns:
            torch.Tensor: [B, 6]
        """
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4] wxyz
        heading_quat_wxyz = isaaclab_math.yaw_quat(
            robot_cur_global_root_rot
        )  # [B, 4] wxyz
        heading_quat_inv_wxyz = isaaclab_math.quat_inv(
            heading_quat_wxyz
        )  # [B, 4] wxyz

        ref_root_quat_wxyz = self.get_ref_motion_root_global_rot_quat_wxyz_cur(
            prefix=prefix
        )  # [B, 4] wxyz
        ref_root_quat_in_heading_wxyz = isaaclab_math.quat_mul(
            heading_quat_inv_wxyz, ref_root_quat_wxyz
        )  # [B, 4] wxyz

        # rot6d: first two columns of rotation matrix (flattened)
        ref_root_rot6d = isaaclab_math.matrix_from_quat(
            ref_root_quat_in_heading_wxyz
        )[..., :2].reshape(ref_root_quat_wxyz.shape[0], 6)  # [B, 6]
        return ref_root_rot6d

    def get_ref_motion_fut_heading_aligned_root_rot6d(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference root rotations (rot6d) in heading-aligned frame.

        Returns:
            torch.Tensor: [B, T, 6]
        """
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4] wxyz
        heading_quat_wxyz = isaaclab_math.yaw_quat(
            robot_cur_global_root_rot
        )  # [B, 4] wxyz
        heading_quat_inv_wxyz = isaaclab_math.quat_inv(
            heading_quat_wxyz
        )  # [B, 4] wxyz

        ref_root_quat_wxyz_fut = (
            self.get_ref_motion_root_global_rot_quat_wxyz_fut(prefix=prefix)
        )  # [B, T, 4] wxyz
        num_envs, num_fut_frames, _ = ref_root_quat_wxyz_fut.shape

        heading_quat_inv_wxyz_fut = heading_quat_inv_wxyz[:, None, :].expand(
            -1, num_fut_frames, -1
        )  # [B, T, 4]
        ref_root_quat_in_heading_wxyz_fut = isaaclab_math.quat_mul(
            heading_quat_inv_wxyz_fut, ref_root_quat_wxyz_fut
        )  # [B, T, 4] wxyz

        ref_root_rot6d_fut = isaaclab_math.matrix_from_quat(
            ref_root_quat_in_heading_wxyz_fut
        )[..., :2].reshape(num_envs, num_fut_frames, 6)  # [B, T, 6]

        return ref_root_rot6d_fut

    def get_ref_motion_future_yaw_delta_sin_cos(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference yaw deltas relative to current reference yaw.

        Returns:
            torch.Tensor: [B, T, 2] as sin/cos(ref_yaw[t+k] - ref_yaw[t]).
        """
        ref_root_quat_cur = self.get_ref_motion_root_global_rot_quat_wxyz_cur(
            prefix=prefix
        )
        ref_root_quat_fut = self.get_ref_motion_root_global_rot_quat_wxyz_fut(
            prefix=prefix
        )
        ref_yaw_cur = _yaw_from_quat_wxyz(ref_root_quat_cur)
        ref_yaw_fut = _yaw_from_quat_wxyz(ref_root_quat_fut)
        yaw_delta = ref_yaw_fut - ref_yaw_cur[:, None]
        return torch.stack(
            [torch.sin(yaw_delta), torch.cos(yaw_delta)],
            dim=-1,
        )

    def get_ref_robot_yaw_error_sin_cos(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference-vs-robot yaw error in the local world frame.

        Returns:
            torch.Tensor: [B, 2] as sin/cos(ref_yaw[t] - robot_yaw[t]).
        """
        ref_root_quat = self.get_ref_motion_root_global_rot_quat_wxyz_cur(
            prefix=prefix
        )
        robot_root_quat = self.robot.data.root_quat_w
        yaw_error = _yaw_from_quat_wxyz(ref_root_quat) - _yaw_from_quat_wxyz(
            robot_root_quat
        )
        return torch.stack(
            [torch.sin(yaw_error), torch.cos(yaw_error)],
            dim=-1,
        )

    def get_ref_future_root_ori_robot_frame_6d(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference root orientation in the current robot root frame.

        Returns:
            torch.Tensor: [B, T, 6] rot6d of inv(q_robot[t]) * q_ref[t+k].
        """
        robot_root_quat = self.robot.data.root_quat_w
        ref_root_quat_fut = self.get_ref_motion_root_global_rot_quat_wxyz_fut(
            prefix=prefix
        )
        num_envs, num_fut_frames, _ = ref_root_quat_fut.shape
        robot_root_quat_inv = isaaclab_math.quat_inv(robot_root_quat)
        robot_root_quat_inv_fut = robot_root_quat_inv[:, None, :].expand(
            -1, num_fut_frames, -1
        )
        rel_quat = isaaclab_math.quat_mul(
            robot_root_quat_inv_fut.reshape(-1, 4),
            ref_root_quat_fut.reshape(-1, 4),
        ).reshape(num_envs, num_fut_frames, 4)
        return isaaclab_math.matrix_from_quat(rel_quat)[..., :2].reshape(
            num_envs,
            num_fut_frames,
            6,
        )

    def get_ref_motion_cur_heading_aligned_root_lin_vel(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference root linear velocity in heading-aligned frame.
        Returns: [B, 3]
        """
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4] wxyz
        heading_quat_wxyz = isaaclab_math.yaw_quat(
            robot_cur_global_root_rot
        )  # [B, 4] wxyz
        ref_root_lin_vel_w = self.get_ref_motion_root_global_lin_vel_cur(
            prefix=prefix
        )  # [B, 3]
        ref_root_lin_vel_heading = isaaclab_math.quat_apply_inverse(
            heading_quat_wxyz, ref_root_lin_vel_w
        )  # [B, 3]
        return ref_root_lin_vel_heading

    def get_ref_motion_fut_heading_aligned_root_lin_vel(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference root linear velocity in heading-aligned frame.
        Returns: [B, T, 3]
        """
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4] wxyz
        heading_quat_wxyz = isaaclab_math.yaw_quat(
            robot_cur_global_root_rot
        )  # [B, 4] wxyz
        ref_root_lin_vel_w_fut = self.get_ref_motion_root_global_lin_vel_fut(
            prefix=prefix
        )  # [B, T, 3]
        num_envs, num_fut_frames, _ = ref_root_lin_vel_w_fut.shape
        heading_quat_wxyz_fut = heading_quat_wxyz[:, None, :].expand(
            -1, num_fut_frames, -1
        )  # [B, T, 4]
        ref_root_lin_vel_heading_fut = isaaclab_math.quat_apply_inverse(
            heading_quat_wxyz_fut, ref_root_lin_vel_w_fut
        )  # [B, T, 3]
        return ref_root_lin_vel_heading_fut

    def get_ref_motion_cur_heading_aligned_root_ang_vel(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference root angular velocity in heading-aligned frame.
        Returns: [B, 3]
        """
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4] wxyz
        heading_quat_wxyz = isaaclab_math.yaw_quat(
            robot_cur_global_root_rot
        )  # [B, 4] wxyz
        ref_root_ang_vel_w = self.get_ref_motion_root_global_ang_vel_cur(
            prefix=prefix
        )  # [B, 3]
        ref_root_ang_vel_heading = isaaclab_math.quat_apply_inverse(
            heading_quat_wxyz, ref_root_ang_vel_w
        )  # [B, 3]
        return ref_root_ang_vel_heading

    def get_ref_motion_fut_heading_aligned_root_ang_vel(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference root angular velocity in heading-aligned frame.
        Returns: [B, T, 3]
        """
        robot_cur_global_root_rot = self.robot.data.root_quat_w  # [B, 4] wxyz
        heading_quat_wxyz = isaaclab_math.yaw_quat(
            robot_cur_global_root_rot
        )  # [B, 4] wxyz
        ref_root_ang_vel_w_fut = self.get_ref_motion_root_global_ang_vel_fut(
            prefix=prefix
        )  # [B, T, 3]
        num_envs, num_fut_frames, _ = ref_root_ang_vel_w_fut.shape
        heading_quat_wxyz_fut = heading_quat_wxyz[:, None, :].expand(
            -1, num_fut_frames, -1
        )  # [B, T, 4]
        ref_root_ang_vel_heading_fut = isaaclab_math.quat_apply_inverse(
            heading_quat_wxyz_fut, ref_root_ang_vel_w_fut
        )  # [B, T, 3]
        return ref_root_ang_vel_heading_fut

    @property
    def robot_dof_pos_cur_urdf_order(self):
        return self.robot.data.joint_pos[..., self.sim2urdf_dof_idx]

    def get_ref_motion_dof_vel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("dof_vel", prefix)
        return base[:, 0, ...][..., self.urdf2sim_dof_idx]

    def get_ref_motion_dof_vel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("dof_vel", prefix)
        return base[..., self.urdf2sim_dof_idx]

    @property
    def robot_dof_vel_cur_urdf_order(self):
        return self.robot.data.joint_vel[..., self.sim2urdf_dof_idx]

    def get_ref_motion_dof_vel_cur_urdf_order(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("dof_vel", prefix)
        return base[:, 0, ...]

    def get_ref_motion_root_global_pos_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_pos", prefix)
        return base[:, 0, ...] + self._env_origins

    def get_ref_motion_root_global_pos_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("root_pos", prefix)
        return base + self._env_origins

    def get_ref_motion_root_global_rot_quat_xyzw_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_ref_state_array("root_rot", prefix)[:, 0, ...]

    def get_ref_motion_root_global_rot_quat_xyzw_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_immediate_next_ref_state_array("root_rot", prefix)

    def get_ref_motion_root_global_rot_quat_wxyz_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self.get_ref_motion_root_global_rot_quat_xyzw_cur(
            prefix=prefix
        )[..., [3, 0, 1, 2]]

    def get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self.get_ref_motion_root_global_rot_quat_xyzw_immediate_next(
            prefix=prefix
        )[..., [3, 0, 1, 2]]

    def get_ref_motion_root_global_lin_vel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_vel", prefix)
        return base[:, 0, ...]

    def get_ref_motion_root_global_lin_vel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_immediate_next_ref_state_array("root_vel", prefix)

    @property
    def ref_motion_root_global_lin_vel_cur(self) -> torch.Tensor:
        return self.get_ref_motion_root_global_lin_vel_cur()

    def get_ref_motion_root_global_ang_vel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("root_ang_vel", prefix)
        return base[:, 0, ...]

    def get_ref_motion_root_global_ang_vel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_immediate_next_ref_state_array("root_ang_vel", prefix)

    def get_ref_motion_gravity_projection_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference gravity projected into reference root frame."""
        return self._get_reference_observation_kinematics(
            prefix
        ).projected_gravity[:, 0]

    def get_ref_motion_gravity_projection_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_reference_observation_kinematics(
            prefix
        ).projected_gravity[:, 1]

    def get_ref_motion_gravity_projection_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference gravity projected into reference root frame."""
        return self._get_reference_observation_kinematics(
            prefix
        ).projected_gravity[:, 1:]

    def get_ref_motion_base_linvel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference base linear velocity in reference root frame."""
        return self._get_reference_observation_kinematics(
            prefix
        ).root_linvel_local[:, 0]

    def get_ref_motion_base_linvel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_reference_observation_kinematics(
            prefix
        ).root_linvel_local[:, 1]

    def get_ref_motion_base_linvel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference base linear velocity in reference root frame."""
        return self._get_reference_observation_kinematics(
            prefix
        ).root_linvel_local[:, 1:]

    def get_ref_motion_base_angvel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Current reference base angular velocity in reference root frame."""
        return self._get_reference_observation_kinematics(
            prefix
        ).root_angvel_local[:, 0]

    def get_ref_motion_base_angvel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        return self._get_reference_observation_kinematics(
            prefix
        ).root_angvel_local[:, 1]

    def get_ref_motion_base_angvel_fut(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        """Future reference base angular velocity in reference root frame."""
        return self._get_reference_observation_kinematics(
            prefix
        ).root_angvel_local[:, 1:]

    def get_ref_motion_bodylink_global_pos_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("rg_pos", prefix)
        return (
            base[:, 0, ...][..., self.urdf2sim_body_idx, :]
            + self._env_origins[:, None, :]
        )

    def get_ref_motion_bodylink_global_pos_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("rg_pos", prefix)
        return (
            base[..., self.urdf2sim_body_idx, :]
            + self._env_origins[:, None, :]
        )

    def get_ref_motion_bodylink_global_pos_cur_urdf_order(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("rg_pos", prefix)
        return base[:, 0, ...] + self._env_origins[:, None, :]

    def get_ref_motion_bodylink_global_rot_wxyz_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        rot_xyzw = self.get_ref_motion_bodylink_global_rot_xyzw_cur(
            prefix=prefix
        )
        return rot_xyzw[..., [3, 0, 1, 2]]

    def get_ref_motion_bodylink_global_rot_xyzw_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("rb_rot", prefix)
        return base[:, 0, ...][..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_rot_xyzw_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("rb_rot", prefix)
        return base[..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        rot_xyzw = self.get_ref_motion_bodylink_global_rot_xyzw_immediate_next(
            prefix=prefix
        )
        return rot_xyzw[..., [3, 0, 1, 2]]

    def get_ref_motion_bodylink_global_rot_xyzw_cur_urdf_order(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("rb_rot", prefix)
        return base[:, 0, ...]

    @property
    def robot_bodylink_global_pos_cur_urdf_order(self):
        return self.robot.data.body_pos_w[:, self.sim2urdf_body_idx]

    @property
    def robot_bodylink_global_rot_wxyz_cur_urdf_order(self):
        return self.robot.data.body_quat_w[:, self.sim2urdf_body_idx]

    @property
    def robot_bodylink_global_rot_xyzw_cur_urdf_order(self):
        return self.robot_bodylink_global_rot_wxyz_cur_urdf_order[
            ..., [1, 2, 3, 0]
        ]

    @property
    def robot_bodylink_global_lin_vel_cur_urdf_order(self):
        return self.robot.data.body_lin_vel_w[:, self.sim2urdf_body_idx]

    @property
    def robot_bodylink_global_ang_vel_cur_urdf_order(self):
        return self.robot.data.body_ang_vel_w[:, self.sim2urdf_body_idx]

    def get_ref_motion_bodylink_global_lin_vel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("body_vel", prefix)
        return base[:, 0, ...][..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_lin_vel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("body_vel", prefix)
        return base[..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_lin_vel_cur_urdf_order(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("body_vel", prefix)
        return base[:, 0, ...]

    def get_ref_motion_bodylink_global_ang_vel_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("body_ang_vel", prefix)
        return base[:, 0, ...][..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_ang_vel_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_immediate_next_ref_state_array("body_ang_vel", prefix)
        return base[..., self.urdf2sim_body_idx, :]

    def get_ref_motion_bodylink_global_ang_vel_cur_urdf_order(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        base = self._get_ref_state_array("body_ang_vel", prefix)
        return base[:, 0, ...]

    def _build_amp_obs_from_ref_state(
        self, frame_idx: int, prefix: str = "ref_"
    ) -> torch.Tensor:
        if (
            not self._amp_left_arm_urdf_dof_idx
            or not self._amp_right_arm_urdf_dof_idx
            or not self._amp_left_leg_urdf_dof_idx
            or not self._amp_right_leg_urdf_dof_idx
            or self._amp_left_elbow_urdf_body_idx is None
            or self._amp_right_elbow_urdf_body_idx is None
            or self._amp_left_foot_urdf_body_idx is None
            or self._amp_right_foot_urdf_body_idx is None
        ):
            raise ValueError(
                "AMP obs indices are not initialized for ref motion."
            )

        dof_pos = self._get_ref_state_array("dof_pos", prefix)[
            :, frame_idx, ...
        ]
        dof_vel = self._get_ref_state_array("dof_vel", prefix)[
            :, frame_idx, ...
        ]

        right_arm_pos = dof_pos[:, self._amp_right_arm_urdf_dof_idx]
        left_arm_pos = dof_pos[:, self._amp_left_arm_urdf_dof_idx]
        right_leg_pos = dof_pos[:, self._amp_right_leg_urdf_dof_idx]
        left_leg_pos = dof_pos[:, self._amp_left_leg_urdf_dof_idx]
        right_arm_vel = dof_vel[:, self._amp_right_arm_urdf_dof_idx]
        left_arm_vel = dof_vel[:, self._amp_left_arm_urdf_dof_idx]
        right_leg_vel = dof_vel[:, self._amp_right_leg_urdf_dof_idx]
        left_leg_vel = dof_vel[:, self._amp_left_leg_urdf_dof_idx]

        root_pos = self._get_ref_state_array("root_pos", prefix)[
            :, frame_idx, ...
        ]
        root_rot = self._get_ref_state_array("root_rot", prefix)[
            :, frame_idx, ...
        ]
        root_inv = quat_inverse(root_rot, w_last=True)

        rg_pos = self._get_ref_state_array("rg_pos", prefix)[:, frame_idx, ...]
        rb_rot = self._get_ref_state_array("rb_rot", prefix)[:, frame_idx, ...]

        left_elbow_pos = rg_pos[:, self._amp_left_elbow_urdf_body_idx, :]
        right_elbow_pos = rg_pos[:, self._amp_right_elbow_urdf_body_idx, :]
        left_elbow_rot = rb_rot[:, self._amp_left_elbow_urdf_body_idx, :]
        right_elbow_rot = rb_rot[:, self._amp_right_elbow_urdf_body_idx, :]

        left_hand_offset = self._amp_left_hand_local_vec.expand(
            left_elbow_pos.shape[0], -1
        )
        right_hand_offset = self._amp_right_hand_local_vec.expand(
            right_elbow_pos.shape[0], -1
        )
        left_hand_world = left_elbow_pos + quat_rotate(
            left_elbow_rot, left_hand_offset, w_last=True
        )
        right_hand_world = right_elbow_pos + quat_rotate(
            right_elbow_rot, right_hand_offset, w_last=True
        )
        left_hand_rel = quat_rotate(
            root_inv, left_hand_world - root_pos, w_last=True
        )
        right_hand_rel = quat_rotate(
            root_inv, right_hand_world - root_pos, w_last=True
        )

        left_foot_world = rg_pos[:, self._amp_left_foot_urdf_body_idx, :]
        right_foot_world = rg_pos[:, self._amp_right_foot_urdf_body_idx, :]
        left_foot_rel = quat_rotate(
            root_inv, left_foot_world - root_pos, w_last=True
        )
        right_foot_rel = quat_rotate(
            root_inv, right_foot_world - root_pos, w_last=True
        )

        return torch.cat(
            [
                right_arm_pos,
                left_arm_pos,
                right_leg_pos,
                left_leg_pos,
                right_arm_vel,
                left_arm_vel,
                right_leg_vel,
                left_leg_vel,
                left_hand_rel,
                right_hand_rel,
                left_foot_rel,
                right_foot_rel,
            ],
            dim=-1,
        )

    def get_ref_motion_amp_obs_cur(self, prefix: str = "ref_") -> torch.Tensor:
        """AMP observation aligned with RSL reference (current frame)."""
        return self._build_amp_obs_from_ref_state(0, prefix=prefix)

    @property
    def motion_end_mask(self) -> torch.Tensor:
        """[B] bool: per-step timeout mask.

        Uses the per-step `motion_end_mask` set before resampling so the
        event is observable within the same step, and falls back to a
        direct comparison if not available.
        """
        return self._motion_end_mask

    @property
    def global_robot_anchor_pos_cur(self):
        return self.robot.data.body_pos_w[:, self.anchor_bodylink_idx]

    def get_ref_motion_anchor_bodylink_global_pos_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        pos = self.get_ref_motion_bodylink_global_pos_cur(prefix=prefix)
        return pos[:, self.anchor_bodylink_idx]

    def get_ref_motion_anchor_bodylink_global_rot_wxyz_cur(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        rot = self.get_ref_motion_bodylink_global_rot_wxyz_cur(prefix=prefix)
        return rot[:, self.anchor_bodylink_idx]

    def get_ref_motion_anchor_bodylink_global_pos_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        pos = self.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=prefix
        )
        return pos[:, self.anchor_bodylink_idx]

    def get_ref_motion_anchor_bodylink_global_rot_wxyz_immediate_next(
        self,
        prefix: str = "ref_",
    ) -> torch.Tensor:
        rot = self.get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            prefix=prefix
        )
        return rot[:, self.anchor_bodylink_idx]

    def _get_obs_bydmmc_ref_motion(
        self,
        obs_prefix: str = "ref_",
    ) -> torch.Tensor:
        base_pos = self._get_ref_state_array("dof_pos", obs_prefix)[:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]
        base_vel = self._get_ref_state_array("dof_vel", obs_prefix)[:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]
        num_envs = base_pos.shape[0]
        cur_ref_dof_pos_flat = base_pos.reshape(num_envs, -1)
        cur_ref_dof_vel_flat = base_vel.reshape(num_envs, -1)
        return torch.cat([cur_ref_dof_pos_flat, cur_ref_dof_vel_flat], dim=-1)

    def _get_obs_bydmmc_ref_motion_fut(
        self,
        obs_prefix: str = "ref_",
    ) -> torch.Tensor:
        base_pos = self._get_ref_state_array("dof_pos", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_dof_idx]
        base_vel = self._get_ref_state_array("dof_vel", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_dof_idx]
        num_envs = base_pos.shape[0]
        n_fut_frames = int(self.cfg.n_fut_frames)
        fut_ref_dof_pos_flat = base_pos.reshape(num_envs, n_fut_frames, -1)
        fut_ref_dof_vel_flat = base_vel.reshape(num_envs, n_fut_frames, -1)
        rel_fut_ref_motion_state_seq = torch.cat(
            [fut_ref_dof_pos_flat, fut_ref_dof_vel_flat], dim=-1
        )
        return rel_fut_ref_motion_state_seq.reshape(num_envs, -1)

    def _get_obs_vr_ref_motion_states(
        self,
        obs_prefix: str = "ref_",
    ) -> torch.Tensor:
        base_pos = self._get_ref_state_array("dof_pos", obs_prefix)[:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]
        num_envs = base_pos.shape[0]
        cur_ref_dof_pos_flat = base_pos.reshape(num_envs, -1)
        return torch.cat(
            [
                cur_ref_dof_pos_flat,
                torch.zeros_like(
                    cur_ref_dof_pos_flat,
                    device=cur_ref_dof_pos_flat.device,
                ),
            ],
            dim=-1,
        )

    def _get_obs_vr_ref_motion_fut(
        self,
        obs_prefix: str = "ref_",
    ) -> torch.Tensor:
        base_pos = self._get_ref_state_array("dof_pos", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_dof_idx]
        num_envs = base_pos.shape[0]
        n_fut_frames = int(self.cfg.n_fut_frames)
        fut_ref_dof_pos_flat = base_pos.reshape(num_envs, n_fut_frames, -1)
        rel_fut_ref_motion_state_seq = torch.cat(
            [
                fut_ref_dof_pos_flat,
                torch.zeros_like(
                    fut_ref_dof_pos_flat, device=fut_ref_dof_pos_flat.device
                ),
            ],
            dim=-1,
        )
        return rel_fut_ref_motion_state_seq.reshape(num_envs, -1)

    def _get_obs_holomotion_rel_ref_motion_flat(
        self,
        obs_prefix: str = "ref_",
    ) -> torch.Tensor:
        # Gather all needed arrays with obs prefix
        fut_rg_pos = self._get_ref_state_array("rg_pos", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_body_idx, :]
        fut_rb_rot_xyzw = self._get_ref_state_array("rb_rot", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_body_idx, :]
        fut_root_rot_xyzw = self._get_ref_state_array("root_rot", obs_prefix)[
            :, 1:, ...
        ]
        fut_root_lin_vel = self._get_ref_state_array("root_vel", obs_prefix)[
            :, 1:, ...
        ]
        fut_root_ang_vel = self._get_ref_state_array(
            "root_ang_vel", obs_prefix
        )[:, 1:, ...]
        fut_dof_pos = self._get_ref_state_array("dof_pos", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_dof_idx]
        fut_dof_vel = self._get_ref_state_array("dof_vel", obs_prefix)[
            :, 1:, ...
        ][..., self.urdf2sim_dof_idx]

        num_envs, num_fut_timesteps, num_bodies, _ = fut_rg_pos.shape
        assert num_envs == self.num_envs
        assert num_fut_timesteps == self.cfg.n_fut_frames

        fut_ref_root_rot_quat = fut_root_rot_xyzw  # [B, T, 4]
        fut_ref_root_rot_quat_inv = quat_inverse(
            fut_ref_root_rot_quat, w_last=True
        )  # [B, T, 4]
        fut_ref_root_rot_quat_body_flat = (
            fut_ref_root_rot_quat[:, :, None, :]
            .repeat(1, 1, num_bodies, 1)
            .reshape(-1, 4)
        )
        fut_ref_root_rot_quat_body_flat_inv = quat_inverse(
            fut_ref_root_rot_quat_body_flat, w_last=True
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            fut_root_rot_xyzw.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            fut_root_rot_xyzw.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat(
            [ref_fut_roll, ref_fut_pitch], dim=-1
        )  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(num_envs, -1)  # [B, T * 2]
        # ---

        fut_ref_root_quat_inv_fut_flat = fut_ref_root_rot_quat_inv.reshape(
            -1, 4
        )
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            fut_ref_root_quat_inv_fut_flat,  # [B*T, 4]
            fut_root_lin_vel.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            fut_ref_root_quat_inv_fut_flat,  # [B*T, 4]
            fut_root_ang_vel.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_dof_pos_flat = fut_dof_pos.reshape(num_envs, -1)
        fut_ref_dof_vel_flat = fut_dof_vel.reshape(num_envs, -1)
        # ---

        # --- calculate the future per frame bodylink position and rotation ---
        fut_ref_global_bodylink_pos = fut_rg_pos  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_rot = fut_rb_rot_xyzw  # [B, T, num_bodies, 4]

        # get root-relative bodylink position
        fut_ref_root_rel_bodylink_pos = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            (
                fut_ref_global_bodylink_pos
                - fut_ref_global_bodylink_pos[:, :, 0:1, :]
            ).reshape(-1, 3),
            w_last=True,
        ).reshape(
            num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink rotation
        fut_ref_root_rel_bodylink_rot = quat_mul(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )
        fut_ref_root_rel_bodylink_rot_mat = quaternion_to_matrix(
            fut_ref_root_rel_bodylink_rot,
            w_last=True,
        )[:, :, :2].reshape(
            num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 6]

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_dof_pos_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_dof_vel_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_root_rel_bodylink_pos.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_rot_mat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*6]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies * (3 + 6)]
        return rel_fut_ref_motion_state_seq.reshape(self.num_envs, -1)

    def _resample_command(self, env_ids: Sequence[int], eval=False):
        """Resample command for specified environments."""
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if isinstance(env_ids, torch.Tensor):
            idxs = env_ids
        elif isinstance(env_ids, slice):
            idxs = torch.arange(self.num_envs, device=self.device)
        else:
            idxs = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        idxs = self._filter_env_ids_for_motion_task(idxs.view(-1))
        if idxs.numel() == 0:
            return

        self._record_completion_rate_for_envs(idxs)
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            len(idxs),
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(eval or self._is_evaluating),
        )
        self._clip_indices[idxs] = clip_idx
        self._frame_indices[idxs] = frame_idx
        self._start_frame_indices[idxs] = frame_idx
        self._task_space_anchor_valid[idxs] = False
        self._reward_sum_since_assign[idxs] = 0.0
        self._step_count_since_assign[idxs] = 0.0
        self._update_ref_motion_state_from_cache(env_ids=idxs)
        self._realign_robot_state(idxs, perturb=True)

    def _filter_env_ids_for_motion_task(
        self, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Filter env_ids to those currently assigned to motion_tracking task.

        In multi-task training, we may keep `ref_motion` registered for observation
        schemas, but we must avoid applying motion-based state alignment to envs
        that are not running motion tracking (e.g., velocity tracking only).

        Behavior:
        - If env does not expose multi-task task buffers, return env_ids (legacy).
        - If env exposes task buffers but has no "motion_tracking" task, return empty.
        - Otherwise, return env_ids where holo_task_ids == holo_task_name_to_id["motion_tracking"].
        """
        if env_ids.numel() == 0:
            return env_ids

        task_ids = getattr(self._env, "holo_task_ids", None)
        task_name_to_id = getattr(self._env, "holo_task_name_to_id", None)
        if task_ids is None or task_name_to_id is None:
            return env_ids

        motion_tid = task_name_to_id.get("motion_tracking", None)
        if motion_tid is None:
            return env_ids[:0]

        task_ids_t = task_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids_t = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        mask = task_ids_t[env_ids_t] == int(motion_tid)
        return env_ids_t[mask]

    def _align_root_to_ref(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        root_pos = self.get_ref_motion_root_global_pos_cur().clone()
        root_rot_xyzw = self.get_ref_motion_root_global_rot_quat_xyzw_cur()
        root_rot = root_rot_xyzw[..., [3, 0, 1, 2]].clone()
        root_lin_vel = self.get_ref_motion_root_global_lin_vel_cur().clone()
        root_ang_vel = self.get_ref_motion_root_global_ang_vel_cur().clone()

        pos_rot_range_list = [
            self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        pos_rot_ranges = torch.tensor(pos_rot_range_list, device=self.device)
        pos_rot_rand_deltas = isaaclab_math.sample_uniform(
            pos_rot_ranges[:, 0],
            pos_rot_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        translation_delta = pos_rot_rand_deltas[:, 0:3]
        rotation_delta = isaaclab_math.quat_from_euler_xyz(
            pos_rot_rand_deltas[:, 3],
            pos_rot_rand_deltas[:, 4],
            pos_rot_rand_deltas[:, 5],
        )

        root_pos[env_ids] += translation_delta
        root_rot[env_ids] = isaaclab_math.quat_mul(
            rotation_delta,
            root_rot[env_ids],
        )

        lin_ang_vel_range_list = [
            self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        lin_ang_vel_ranges = torch.tensor(
            lin_ang_vel_range_list, device=self.device
        )

        lin_ang_vel_rand_deltas = isaaclab_math.sample_uniform(
            lin_ang_vel_ranges[:, 0],
            lin_ang_vel_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_lin_vel[env_ids] += lin_ang_vel_rand_deltas[:, :3]
        root_ang_vel[env_ids] += lin_ang_vel_rand_deltas[:, 3:]

        written_root_state = torch.cat(
            [
                root_pos[env_ids],
                root_rot[env_ids],
                root_lin_vel[env_ids],
                root_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(
            written_root_state,
            env_ids=env_ids,
        )
        self._refresh_task_space_anchor_from_written_root(
            env_ids,
            root_position_w=written_root_state[:, :3],
            root_rotation_w_wxyz=written_root_state[:, 3:7],
        )

    def _align_dof_to_ref(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        dof_pos = self.get_ref_motion_dof_pos_cur().clone()
        dof_vel = self.get_ref_motion_dof_vel_cur().clone()

        dof_pos += isaaclab_math.sample_uniform(
            *self.cfg.dof_pos_perturb_range,
            dof_pos.shape,
            dof_pos.device,
        )
        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos[env_ids] = torch.clip(
            dof_pos[env_ids],
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )

        self.robot.write_joint_state_to_sim(
            dof_pos[env_ids],
            dof_vel[env_ids],
            env_ids=env_ids,
        )

    def _realign_robot_state(self, env_ids, *, perturb: bool) -> None:
        """Route every motion reassignment through one reset implementation."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        reset_target_source = getattr(
            self,
            "_reset_target_source",
            "robot_reference",
        )
        if reset_target_source == "robot_reference":
            if perturb:
                self._align_root_to_ref(env_ids)
                self._align_dof_to_ref(env_ids)
            else:
                self.force_realign_root_state_to_ref_no_perturb(env_ids)
                self.force_realign_dof_state_to_ref_no_perturb(env_ids)
            self._finish_robot_state_write(env_ids)
            return
        if reset_target_source == "smpl_default":
            self._align_root_to_smpl_default(env_ids, perturb=perturb)
            self._align_dof_to_default(env_ids, perturb=perturb)
            self._finish_robot_state_write(env_ids)
            return
        if reset_target_source == "smpl_semantic_ik":
            self._align_state_to_smpl_semantic_ik(env_ids, perturb=perturb)
            self._finish_robot_state_write(env_ids)
            return
        if reset_target_source == "smpl_cycle_network":
            self._align_state_to_smpl_cycle_network(env_ids, perturb=perturb)
            self._finish_robot_state_write(env_ids)
            return
        if reset_target_source == "composed":
            self._align_state_to_composed_reset(env_ids, perturb=perturb)
            self._finish_robot_state_write(env_ids)
            return
        raise RuntimeError(
            "Unsupported robot.motion.reset_target_source "
            f"{reset_target_source!r}"
        )

    def _update_reset_diagnostics_from_robot(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        """Record diagnostics from the exact post-write robot state."""

        if not hasattr(self, "metrics"):
            return
        required_metric_names = (
            "Reset/RootHeight",
            "Reset/RootSpeed",
            "Reset/RootAngularSpeed",
            "Reset/JointDefaultRMS",
            "Reset/JointVelocityRMS",
        )
        if any(name not in self.metrics for name in required_metric_names):
            return

        data = self.robot.data
        self.metrics["Reset/RootHeight"][env_ids] = (
            data.root_pos_w[env_ids, 2] - self._env_origins[env_ids, 2]
        )
        self.metrics["Reset/RootSpeed"][env_ids] = torch.linalg.vector_norm(
            data.root_lin_vel_w[env_ids],
            dim=-1,
        )
        self.metrics["Reset/RootAngularSpeed"][env_ids] = (
            torch.linalg.vector_norm(
                data.root_ang_vel_w[env_ids],
                dim=-1,
            )
        )
        self.metrics["Reset/JointDefaultRMS"][env_ids] = torch.sqrt(
            torch.mean(
                torch.square(
                    data.joint_pos[env_ids] - data.default_joint_pos[env_ids]
                ),
                dim=-1,
            )
        )
        self.metrics["Reset/JointVelocityRMS"][env_ids] = torch.sqrt(
            torch.mean(
                torch.square(data.joint_vel[env_ids]),
                dim=-1,
            )
        )

    @staticmethod
    def _select_semantic_skeleton_state(
        state: SemanticSkeletonState,
        env_ids: torch.Tensor,
    ) -> SemanticSkeletonState:
        return SemanticSkeletonState(
            position_w=state.position_w[env_ids],
            pelvis_rotation_wxyz=state.pelvis_rotation_wxyz[env_ids],
            torso_rotation_wxyz=state.torso_rotation_wxyz[env_ids],
            endpoint_rotation_wxyz=state.endpoint_rotation_wxyz[env_ids],
            head_rotation_wxyz=(
                None
                if state.head_rotation_wxyz is None
                else state.head_rotation_wxyz[env_ids]
            ),
            root_linear_velocity_w=state.root_linear_velocity_w[env_ids],
            root_angular_velocity_w=state.root_angular_velocity_w[env_ids],
            nominal_scale=state.nominal_scale[env_ids],
        )

    def _write_smpl_pose_reset_state(
        self,
        *,
        env_ids: torch.Tensor,
        source: SemanticSkeletonState,
        dof_pos: torch.Tensor,
        root_height_adjustment: torch.Tensor,
        perturb: bool,
    ) -> None:
        """Write the shared FQV root state and one G1 reset pose."""

        source_yaw = _yaw_from_quat_wxyz(source.pelvis_rotation_wxyz)
        zeros = torch.zeros_like(source_yaw)
        root_rot = isaaclab_math.quat_from_euler_xyz(
            zeros,
            zeros,
            source_yaw,
        )
        default_root_state = self.robot.data.default_root_state[env_ids]
        root_pos = self._env_origins[env_ids].clone()
        root_pos[:, 2] += default_root_state[:, 2]
        root_pos[:, 2] += root_height_adjustment
        root_pos[:, 2] += float(self._smpl_reset_root_height_offset_m)

        root_lin_vel = torch.zeros(
            (len(env_ids), 3),
            device=self.device,
            dtype=root_pos.dtype,
        )
        if self._smpl_reset_root_linear_velocity_source == "smpl_scaled_xy":
            target_local_velocity = isaaclab_math.quat_apply_inverse(
                source.pelvis_rotation_wxyz,
                source.root_linear_velocity_w,
            )
            robot_scale = float(self._skeleton_adapter_spec.nominal_scale)
            desired_local_velocity = (
                target_local_velocity
                / source.nominal_scale.clamp_min(1.0e-6)
                * robot_scale
            )
            desired_local_velocity = desired_local_velocity.clone()
            desired_local_velocity[:, 2] = 0.0
            root_lin_vel = isaaclab_math.quat_apply(
                root_rot,
                desired_local_velocity,
            )
        root_ang_vel = torch.zeros_like(root_lin_vel)
        if self._smpl_reset_root_angular_velocity_source == "smpl_yaw_rate":
            root_ang_vel[:, 2] = source.root_angular_velocity_w[:, 2]

        if perturb:
            pos_rot_range_list = [
                self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            pos_rot_ranges = torch.tensor(
                pos_rot_range_list,
                device=self.device,
                dtype=root_pos.dtype,
            )
            pos_rot_rand_deltas = isaaclab_math.sample_uniform(
                pos_rot_ranges[:, 0],
                pos_rot_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_pos += pos_rot_rand_deltas[:, :3]
            rotation_delta = isaaclab_math.quat_from_euler_xyz(
                pos_rot_rand_deltas[:, 3],
                pos_rot_rand_deltas[:, 4],
                pos_rot_rand_deltas[:, 5],
            )
            root_rot = isaaclab_math.quat_mul(rotation_delta, root_rot)
            velocity_ranges = torch.tensor(
                [
                    self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
                    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
                ],
                device=self.device,
                dtype=root_pos.dtype,
            )
            velocity_delta = isaaclab_math.sample_uniform(
                velocity_ranges[:, 0],
                velocity_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_lin_vel += velocity_delta[:, :3]
            root_ang_vel += velocity_delta[:, 3:]

        written_root_state = torch.cat(
            (root_pos, root_rot, root_lin_vel, root_ang_vel),
            dim=-1,
        )
        self.robot.write_root_state_to_sim(
            written_root_state,
            env_ids=env_ids,
        )
        self._refresh_task_space_anchor_from_written_root(
            env_ids,
            root_position_w=written_root_state[:, :3],
            root_rotation_w_wxyz=written_root_state[:, 3:7],
        )
        self.robot.write_joint_state_to_sim(
            dof_pos,
            torch.zeros_like(dof_pos),
            env_ids=env_ids,
        )

    def _decode_cycle_reset_pose(
        self,
        env_ids: torch.Tensor,
        *,
        perturb: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clipped decoder q, raw q and its grounded-height adjustment."""

        runtime = getattr(self, "_smpl_cycle_reset_runtime", None)
        if runtime is None:
            raise RuntimeError("SMPL cycle reset runtime is not initialized")
        body = self._get_smpl_body_current_future()[env_ids, 0].to(
            device=self.device,
            dtype=torch.float32,
        )
        q_urdf, root_height_adjustment = runtime(body)
        raw_dof_pos = q_urdf[..., self.urdf2sim_dof_idx]
        limits_sim = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos = torch.clip(
            raw_dof_pos,
            limits_sim[..., 0],
            limits_sim[..., 1],
        )
        if perturb:
            dof_pos += isaaclab_math.sample_uniform(
                *self.cfg.dof_pos_perturb_range,
                dof_pos.shape,
                dof_pos.device,
            )
            dof_pos = torch.clip(
                dof_pos,
                limits_sim[..., 0],
                limits_sim[..., 1],
            )
        return dof_pos, raw_dof_pos, root_height_adjustment

    def _record_cycle_reset_diagnostics(
        self,
        env_ids: torch.Tensor,
        *,
        dof_pos: torch.Tensor,
        raw_dof_pos: torch.Tensor,
        root_height_adjustment: torch.Tensor,
    ) -> None:
        """Record decoder diagnostics when the corresponding metrics exist."""

        values = {
            "Reset/CycleRootHeightAdjustment": root_height_adjustment,
            "Reset/CycleJointFinite": torch.isfinite(dof_pos)
            .all(dim=-1)
            .float(),
            "Reset/CycleJointLimitClipRMS": torch.sqrt(
                torch.mean(torch.square(dof_pos - raw_dof_pos), dim=-1)
            ),
        }
        for name, value in values.items():
            if name in self.metrics:
                self.metrics[name][env_ids] = value

    def _apply_composed_reset_perturbations(
        self,
        *,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        perturb: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the existing root perturbation contract to composed state."""

        if not perturb:
            return root_pos, root_rot, root_lin_vel, root_ang_vel
        pos_rot_ranges = torch.tensor(
            [
                self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ],
            device=self.device,
            dtype=root_pos.dtype,
        )
        pos_rot_delta = isaaclab_math.sample_uniform(
            pos_rot_ranges[:, 0],
            pos_rot_ranges[:, 1],
            (len(root_pos), 6),
            device=self.device,
        )
        root_pos = root_pos + pos_rot_delta[:, :3]
        rotation_delta = isaaclab_math.quat_from_euler_xyz(
            pos_rot_delta[:, 3],
            pos_rot_delta[:, 4],
            pos_rot_delta[:, 5],
        )
        root_rot = isaaclab_math.quat_mul(rotation_delta, root_rot)

        velocity_ranges = torch.tensor(
            [
                self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ],
            device=self.device,
            dtype=root_pos.dtype,
        )
        velocity_delta = isaaclab_math.sample_uniform(
            velocity_ranges[:, 0],
            velocity_ranges[:, 1],
            (len(root_pos), 6),
            device=self.device,
        )
        return (
            root_pos,
            root_rot,
            root_lin_vel + velocity_delta[:, :3],
            root_ang_vel + velocity_delta[:, 3:],
        )

    def _align_state_to_composed_reset(
        self,
        env_ids: torch.Tensor,
        *,
        perturb: bool,
    ) -> None:
        """Compose Reset fields from explicit Robot-ref, SMPL and decoder sources."""

        sources = self._composed_reset_sources
        needs_ref_position = any(
            sources[name] == "robot_reference"
            for name in ("root_xy_source", "root_height_source")
        )
        needs_ref_rotation = any(
            sources[name] == "robot_reference"
            for name in ("root_roll_pitch_source", "root_yaw_source")
        )
        needs_raw_smpl = any(
            (
                sources["root_yaw_source"] == "smpl",
                sources["root_linear_velocity_source"] == "smpl_scaled_xy",
                sources["root_angular_velocity_source"] == "smpl_yaw_rate",
            )
        )
        needs_actor_canonical_state = any(
            (
                sources["root_xy_source"] == "smpl_actor_canonical_scaled",
                sources["root_roll_pitch_source"] == "smpl_actor_canonical",
                sources["root_linear_velocity_source"]
                == "smpl_actor_canonical_scaled_full",
                sources["root_angular_velocity_source"]
                in {
                    "smpl_actor_canonical_full",
                    "smpl_actor_canonical_yaw_rate",
                },
            )
        )
        needs_actor_canonical_yaw = (
            sources["root_yaw_source"] == "smpl_actor_canonical"
        )
        needs_cycle = sources["joint_position_source"] == "cycle_network"

        ref_root_pos = ref_root_rot = ref_root_lin_vel = ref_root_ang_vel = (
            None
        )
        if needs_ref_position:
            ref_root_pos = self.get_ref_motion_root_global_pos_cur()[env_ids]
        if needs_ref_rotation:
            ref_root_rot_xyzw = (
                self.get_ref_motion_root_global_rot_quat_xyzw_cur()[env_ids]
            )
            ref_root_rot = ref_root_rot_xyzw[..., [3, 0, 1, 2]]
        if sources["root_linear_velocity_source"] == "robot_reference":
            ref_root_lin_vel = self.get_ref_motion_root_global_lin_vel_cur()[
                env_ids
            ]
        if sources["root_angular_velocity_source"] == "robot_reference":
            ref_root_ang_vel = self.get_ref_motion_root_global_ang_vel_cur()[
                env_ids
            ]

        source = None
        if needs_raw_smpl:
            source_all = self._get_smpl_skeleton_state(immediate_next=False)
            source = self._select_semantic_skeleton_state(source_all, env_ids)
        canonical = None
        if needs_actor_canonical_state:
            canonical_all = self._get_smpl_actor_canonical_root_state_current()
            canonical = CanonicalSmplRootState(
                position=canonical_all.position[env_ids],
                rotation_wxyz=canonical_all.rotation_wxyz[env_ids],
                linear_velocity_local=canonical_all.linear_velocity_local[
                    env_ids
                ],
                angular_velocity_local=canonical_all.angular_velocity_local[
                    env_ids
                ],
                yaw_rate=canonical_all.yaw_rate[env_ids],
                nominal_scale=canonical_all.nominal_scale[env_ids],
            )
        actor_canonical_root_rotation = None
        if needs_actor_canonical_yaw and canonical is None:
            actor_canonical_root_rotation = (
                self._get_smpl_actor_canonical_root_rotation_current()[env_ids]
            )

        cycle_dof_pos = cycle_raw_dof_pos = root_height_adjustment = None
        if needs_cycle:
            (
                cycle_dof_pos,
                cycle_raw_dof_pos,
                root_height_adjustment,
            ) = self._decode_cycle_reset_pose(env_ids, perturb=perturb)

        if sources["joint_position_source"] == "robot_reference":
            dof_pos = self.get_ref_motion_dof_pos_cur()[env_ids].clone()
            if perturb:
                dof_pos += isaaclab_math.sample_uniform(
                    *self.cfg.dof_pos_perturb_range,
                    dof_pos.shape,
                    dof_pos.device,
                )
        elif sources["joint_position_source"] == "cycle_network":
            dof_pos = cycle_dof_pos
        else:
            dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
            if perturb:
                dof_pos += isaaclab_math.sample_uniform(
                    *self.cfg.dof_pos_perturb_range,
                    dof_pos.shape,
                    dof_pos.device,
                )
        limits_sim = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos = torch.clip(
            dof_pos,
            limits_sim[..., 0],
            limits_sim[..., 1],
        )

        if sources["root_xy_source"] == "robot_reference":
            root_xy = ref_root_pos[:, :2]
        elif sources["root_xy_source"] == "smpl_actor_canonical_scaled":
            robot_scale = float(self._skeleton_adapter_spec.nominal_scale)
            root_xy = self._env_origins[env_ids, :2] + (
                canonical.position[:, :2]
                / canonical.nominal_scale.clamp_min(1.0e-6)
                * robot_scale
            )
        else:
            root_xy = self._env_origins[env_ids, :2]

        use_ref_roll_pitch = (
            sources["root_roll_pitch_source"] == "robot_reference"
        )
        use_ref_yaw = sources["root_yaw_source"] == "robot_reference"
        use_canonical_roll_pitch = (
            sources["root_roll_pitch_source"] == "smpl_actor_canonical"
        )
        use_canonical_yaw = (
            sources["root_yaw_source"] == "smpl_actor_canonical"
        )
        if use_ref_roll_pitch and use_ref_yaw:
            root_rot = ref_root_rot.clone()
        elif use_canonical_roll_pitch and use_canonical_yaw:
            root_rot = canonical.rotation_wxyz.clone()
        else:
            if use_ref_roll_pitch:
                roll, pitch, _ = isaaclab_math.euler_xyz_from_quat(
                    ref_root_rot
                )
            elif use_canonical_roll_pitch:
                roll, pitch, _ = isaaclab_math.euler_xyz_from_quat(
                    canonical.rotation_wxyz
                )
            else:
                roll = torch.zeros(len(env_ids), device=self.device)
                pitch = torch.zeros_like(roll)
            if use_ref_yaw:
                yaw = _yaw_from_quat_wxyz(ref_root_rot)
            elif use_canonical_yaw:
                canonical_rotation = (
                    canonical.rotation_wxyz
                    if canonical is not None
                    else actor_canonical_root_rotation
                )
                yaw = _yaw_from_quat_wxyz(canonical_rotation)
            else:
                yaw = _yaw_from_quat_wxyz(source.pelvis_rotation_wxyz)
            root_rot = isaaclab_math.quat_from_euler_xyz(roll, pitch, yaw)

        if sources["root_height_source"] == "robot_reference":
            root_z = ref_root_pos[:, 2]
        else:
            default_root_height = self.robot.data.default_root_state[
                env_ids, 2
            ]
            if sources["root_height_source"] == "pose_grounded":
                runtime = self._smpl_pose_grounding_runtime
                if runtime is None:
                    raise RuntimeError(
                        "pose_grounded Reset runtime was not initialized"
                    )
                root_height_adjustment = runtime(dof_pos, root_rot)
            root_z = (
                self._env_origins[env_ids, 2]
                + default_root_height
                + root_height_adjustment
                + float(self._smpl_reset_root_height_offset_m)
            )
        root_pos = torch.cat((root_xy, root_z.unsqueeze(-1)), dim=-1).clone()

        if sources["root_linear_velocity_source"] == "robot_reference":
            root_lin_vel = ref_root_lin_vel.clone()
        elif sources["root_linear_velocity_source"] == "smpl_scaled_xy":
            target_local_velocity = isaaclab_math.quat_apply_inverse(
                source.pelvis_rotation_wxyz,
                source.root_linear_velocity_w,
            )
            robot_scale = float(self._skeleton_adapter_spec.nominal_scale)
            desired_local_velocity = (
                target_local_velocity
                / source.nominal_scale.clamp_min(1.0e-6)
                * robot_scale
            ).clone()
            desired_local_velocity[:, 2] = 0.0
            root_lin_vel = isaaclab_math.quat_apply(
                root_rot,
                desired_local_velocity,
            )
        else:
            robot_scale = float(self._skeleton_adapter_spec.nominal_scale)
            desired_local_velocity = (
                canonical.linear_velocity_local
                / canonical.nominal_scale.clamp_min(1.0e-6)
                * robot_scale
            )
            root_lin_vel = isaaclab_math.quat_apply(
                root_rot,
                desired_local_velocity,
            )
        if sources["root_angular_velocity_source"] == "robot_reference":
            root_ang_vel = ref_root_ang_vel.clone()
        elif sources["root_angular_velocity_source"] == "smpl_yaw_rate":
            root_ang_vel = torch.zeros_like(root_lin_vel)
            root_ang_vel[:, 2] = source.root_angular_velocity_w[:, 2]
        elif (
            sources["root_angular_velocity_source"]
            == "smpl_actor_canonical_yaw_rate"
        ):
            root_ang_vel = torch.zeros_like(root_lin_vel)
            root_ang_vel[:, 2] = canonical.yaw_rate
        else:
            root_ang_vel = isaaclab_math.quat_apply(
                root_rot,
                canonical.angular_velocity_local,
            )

        root_pos, root_rot, root_lin_vel, root_ang_vel = (
            self._apply_composed_reset_perturbations(
                root_pos=root_pos,
                root_rot=root_rot,
                root_lin_vel=root_lin_vel,
                root_ang_vel=root_ang_vel,
                perturb=perturb,
            )
        )

        if sources["joint_velocity_source"] == "robot_reference":
            dof_vel = self.get_ref_motion_dof_vel_cur()[env_ids].clone()
        else:
            dof_vel = torch.zeros_like(dof_pos)

        written_root_state = torch.cat(
            (root_pos, root_rot, root_lin_vel, root_ang_vel),
            dim=-1,
        )
        self.robot.write_root_state_to_sim(
            written_root_state,
            env_ids=env_ids,
        )
        self._refresh_task_space_anchor_from_written_root(
            env_ids,
            root_position_w=written_root_state[:, :3],
            root_rotation_w_wxyz=written_root_state[:, 3:7],
        )
        self.robot.write_joint_state_to_sim(
            dof_pos,
            dof_vel,
            env_ids=env_ids,
        )
        if needs_cycle:
            self._record_cycle_reset_diagnostics(
                env_ids,
                dof_pos=dof_pos,
                raw_dof_pos=cycle_raw_dof_pos,
                root_height_adjustment=root_height_adjustment,
            )

    def _align_state_to_smpl_cycle_network(
        self,
        env_ids: torch.Tensor,
        *,
        perturb: bool,
    ) -> None:
        """Reset from one learned P+A+R-to-q29 forward pass."""

        source_all = self._get_smpl_skeleton_state(immediate_next=False)
        source = self._select_semantic_skeleton_state(source_all, env_ids)
        dof_pos, raw_dof_pos, root_height_adjustment = (
            self._decode_cycle_reset_pose(env_ids, perturb=perturb)
        )
        self._write_smpl_pose_reset_state(
            env_ids=env_ids,
            source=source,
            dof_pos=dof_pos,
            root_height_adjustment=root_height_adjustment,
            perturb=perturb,
        )
        self._record_cycle_reset_diagnostics(
            env_ids,
            dof_pos=dof_pos,
            raw_dof_pos=raw_dof_pos,
            root_height_adjustment=root_height_adjustment,
        )

    def _align_state_to_smpl_semantic_ik(
        self,
        env_ids,
        *,
        perturb: bool,
    ) -> None:
        """Reset from a short, batched, GMR-free semantic IK solve."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return
        solver = getattr(self, "_semantic_ik_reset_solver", None)
        if solver is None:
            raise RuntimeError("Semantic IK reset solver is not initialized")

        source_all = self._get_smpl_skeleton_state(immediate_next=False)
        source = self._select_semantic_skeleton_state(source_all, env_ids)
        default_joint_sim = self.robot.data.default_joint_pos[env_ids].clone()
        limits_sim = self.robot.data.soft_joint_pos_limits[env_ids]
        default_joint_urdf = default_joint_sim[..., self.sim2urdf_dof_idx]
        lower_urdf = limits_sim[..., 0][..., self.sim2urdf_dof_idx]
        upper_urdf = limits_sim[..., 1][..., self.sim2urdf_dof_idx]
        result = solver.solve(
            target=source,
            q_default_urdf=default_joint_urdf,
            q_lower_urdf=lower_urdf,
            q_upper_urdf=upper_urdf,
        )
        dof_pos = result.dof_pos_urdf[..., self.urdf2sim_dof_idx].clone()
        if perturb:
            dof_pos += isaaclab_math.sample_uniform(
                *self.cfg.dof_pos_perturb_range,
                dof_pos.shape,
                dof_pos.device,
            )
        dof_pos = torch.clip(
            dof_pos,
            limits_sim[..., 0],
            limits_sim[..., 1],
        )
        dof_pos_urdf = dof_pos[..., self.sim2urdf_dof_idx]
        root_height_adjustment = solver.root_height_adjustment(
            q_default_urdf=default_joint_urdf,
            q_solved_urdf=dof_pos_urdf,
        )

        self._write_smpl_pose_reset_state(
            env_ids=env_ids,
            source=source,
            dof_pos=dof_pos,
            root_height_adjustment=root_height_adjustment,
            perturb=perturb,
        )

        denominator = result.initial_total.clamp_min(1.0e-8)
        self.metrics["Reset/IKAccepted"][env_ids] = result.accepted.float()
        self.metrics["Reset/IKInitialObjective"][env_ids] = (
            result.initial_total
        )
        self.metrics["Reset/IKFinalObjective"][env_ids] = result.final_total
        self.metrics["Reset/IKRelativeImprovement"][env_ids] = (
            result.initial_total - result.final_total
        ) / denominator
        self.metrics["Reset/IKRootHeightAdjustment"][env_ids] = (
            root_height_adjustment
        )
        self.metrics["Reset/IKFallbackReason"][env_ids] = (
            result.fallback_reason_code.float()
        )

    def _align_root_to_smpl_default(
        self,
        env_ids,
        *,
        perturb: bool,
    ) -> None:
        """Reset root from SMPL heading and the robot's nominal root height."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        if (
            getattr(
                self,
                "_smpl_reset_root_height_source",
                "asset_default",
            )
            != "asset_default"
        ):
            raise RuntimeError(
                "SMPL reset currently supports only "
                "root_height_source='asset_default'"
            )

        source = self._get_smpl_skeleton_state(immediate_next=False)
        source_yaw = _yaw_from_quat_wxyz(source.pelvis_rotation_wxyz[env_ids])

        default_root_state = self.robot.data.default_root_state[env_ids]
        root_pos = self._env_origins[env_ids].clone()
        root_pos[:, 2] += default_root_state[:, 2]
        root_pos[:, 2] += float(
            getattr(self, "_smpl_reset_root_height_offset_m", 0.0)
        )
        zeros = torch.zeros_like(source_yaw)
        root_rot = isaaclab_math.quat_from_euler_xyz(
            zeros,
            zeros,
            source_yaw,
        )
        root_lin_vel = torch.zeros(
            (len(env_ids), 3),
            device=self.device,
            dtype=root_pos.dtype,
        )
        root_ang_vel = torch.zeros_like(root_lin_vel)

        if perturb:
            pos_rot_range_list = [
                self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            pos_rot_ranges = torch.tensor(
                pos_rot_range_list,
                device=self.device,
                dtype=root_pos.dtype,
            )
            pos_rot_rand_deltas = isaaclab_math.sample_uniform(
                pos_rot_ranges[:, 0],
                pos_rot_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_pos += pos_rot_rand_deltas[:, :3]
            rotation_delta = isaaclab_math.quat_from_euler_xyz(
                pos_rot_rand_deltas[:, 3],
                pos_rot_rand_deltas[:, 4],
                pos_rot_rand_deltas[:, 5],
            )
            root_rot = isaaclab_math.quat_mul(rotation_delta, root_rot)

            lin_ang_vel_range_list = [
                self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            lin_ang_vel_ranges = torch.tensor(
                lin_ang_vel_range_list,
                device=self.device,
                dtype=root_pos.dtype,
            )
            lin_ang_vel_rand_deltas = isaaclab_math.sample_uniform(
                lin_ang_vel_ranges[:, 0],
                lin_ang_vel_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_lin_vel += lin_ang_vel_rand_deltas[:, :3]
            root_ang_vel += lin_ang_vel_rand_deltas[:, 3:]

        written_root_state = torch.cat(
            [root_pos, root_rot, root_lin_vel, root_ang_vel],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(
            written_root_state,
            env_ids=env_ids,
        )
        self._refresh_task_space_anchor_from_written_root(
            env_ids,
            root_position_w=written_root_state[:, :3],
            root_rotation_w_wxyz=written_root_state[:, 3:7],
        )

    def _align_dof_to_default(
        self,
        env_ids,
        *,
        perturb: bool,
    ) -> None:
        """Reset joints from runtime asset defaults without reading q_ref."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = torch.zeros_like(dof_pos)
        if perturb:
            dof_pos_perturb = isaaclab_math.sample_uniform(
                *self.cfg.dof_pos_perturb_range,
                (self.num_envs, dof_pos.shape[-1]),
                dof_pos.device,
            )
            dof_pos += dof_pos_perturb[env_ids]

        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos = torch.clip(
            dof_pos,
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )
        self.robot.write_joint_state_to_sim(
            dof_pos,
            dof_vel,
            env_ids=env_ids,
        )

    def force_realign_root_state_to_ref_no_perturb(self, env_ids) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        root_pos = self.get_ref_motion_root_global_pos_cur().clone()
        root_rot_xyzw = self.get_ref_motion_root_global_rot_quat_xyzw_cur()
        root_rot = root_rot_xyzw[..., [3, 0, 1, 2]].clone()
        root_lin_vel = self.get_ref_motion_root_global_lin_vel_cur().clone()
        root_ang_vel = self.get_ref_motion_root_global_ang_vel_cur().clone()
        written_root_state = torch.cat(
            [
                root_pos[env_ids],
                root_rot[env_ids],
                root_lin_vel[env_ids],
                root_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(
            written_root_state,
            env_ids=env_ids,
        )
        self._refresh_task_space_anchor_from_written_root(
            env_ids,
            root_position_w=written_root_state[:, :3],
            root_rotation_w_wxyz=written_root_state[:, 3:7],
        )

    def force_realign_dof_state_to_ref_no_perturb(self, env_ids) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        env_ids = self._filter_env_ids_for_motion_task(env_ids)
        if env_ids.numel() == 0:
            return

        dof_pos = self.get_ref_motion_dof_pos_cur().clone()
        dof_vel = self.get_ref_motion_dof_vel_cur().clone()
        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos[env_ids] = torch.clip(
            dof_pos[env_ids],
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )

        self.robot.write_joint_state_to_sim(
            dof_pos[env_ids],
            dof_vel[env_ids],
            env_ids=env_ids,
        )

    def force_realign_offline_eval_no_perturb(self, env_ids) -> None:
        self._realign_robot_state(env_ids, perturb=False)

    def _update_command(self):
        self._invalidate_observation_state_cache()
        # RewardManager evaluates eight task-space terms in sequence.  Build
        # each immutable target once per command frame. Robot-state caches
        # additionally check the physics-step counter, because the critic can
        # repopulate them after this call and before the next physics step.
        all_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        motion_ids = self._filter_env_ids_for_motion_task(all_ids)
        if motion_ids.numel() == 0:
            return

        continue_ids = motion_ids
        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        if episode_length_buf is not None:
            continue_mask = episode_length_buf[motion_ids] != 0
            continue_ids = motion_ids[continue_mask]
        if continue_ids.numel() > 0:
            self._frame_indices[continue_ids] += 1
        self._swap_step_counter += 1

        if self._swap_step_counter >= self._motion_cache.swap_interval_steps:
            self._swap_pending = True

        # Resample when motion ends
        self._resample_when_motion_end_cache()
        self._update_ref_motion_state_from_cache()

    def _resample_when_motion_end_cache(self):
        """Resample environments when motion ends (simple cache mode)."""
        all_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        motion_ids = self._filter_env_ids_for_motion_task(all_ids)
        if motion_ids.numel() == 0:
            return

        lengths = self._motion_cache.lengths_for_indices(self._clip_indices)
        max_valid_frame = torch.clamp(
            lengths - 1 - self.cfg.n_fut_frames, min=0
        )
        need_resample = (
            self._frame_indices[motion_ids] > max_valid_frame[motion_ids]
        )

        if torch.any(need_resample):
            resample_ids = motion_ids[torch.nonzero(need_resample).squeeze(-1)]
            # Resample these envs
            self._record_completion_rate_for_envs(resample_ids)
            clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
                len(resample_ids),
                self.cfg.n_fut_frames,
                self.device,
                deterministic_start=self._is_evaluating,
            )
            self._clip_indices[resample_ids] = clip_idx
            self._frame_indices[resample_ids] = frame_idx
            self._start_frame_indices[resample_ids] = frame_idx
            self._task_space_anchor_valid[resample_ids] = False
            self._reward_sum_since_assign[resample_ids] = 0.0
            self._step_count_since_assign[resample_ids] = 0.0
            # Realign robot state
            self._update_ref_motion_state_from_cache(env_ids=resample_ids)
            self._realign_robot_state(resample_ids, perturb=True)
            # Mark motion end
            self._motion_end_mask[motion_ids] = False
            self._motion_end_mask[resample_ids] = True
            self.motion_end_counter[resample_ids] += 1

    def _update_metrics(self):
        """Update metrics for command progress tracking."""
        if not hasattr(self, "metrics"):
            self.metrics = {}

        if self._task_space_target_source == "smpl_skeleton":
            self._update_semantic_skeleton_metrics()
            return
        self._update_mpjpe_metrics()
        self._update_mpkpe_metrics()

    def _update_semantic_skeleton_metrics(self):
        """Update Robot-reference-free metrics for pure-SMPL training."""

        from holomotion.src.env.isaaclab_components.isaaclab_scale_invariant_skeleton_rewards import (
            global_structure_reward_and_error,
            normalized_position_reward_and_error,
        )

        target = self.get_semantic_skeleton_target_immediate_next()
        robot = self.get_semantic_skeleton_robot_state()
        _, global_error = global_structure_reward_and_error(robot, target)
        _, position_error = normalized_position_reward_and_error(robot, target)
        self.metrics["Task/SkeletonSI_GlobalStructure"] = global_error
        self.metrics["Task/SkeletonSI_NormalizedPosition"] = position_error

    def _update_mpjpe_metrics(self):
        """Update MPJPE (Mean Per Joint Position Error) metrics."""
        # Get current and reference joint positions
        current_dof_pos = self.robot.data.joint_pos  # [B, num_dofs]
        ref_dof_pos = self.get_ref_motion_dof_pos_immediate_next()

        # Compute joint position errors
        dof_pos_error = torch.abs(
            current_dof_pos - ref_dof_pos
        )  # [B, num_dofs]

        # MPJPE whole body
        mpjpe_wholebody = torch.mean(dof_pos_error, dim=-1)  # [B]

        # MPJPE arms (using unified naming)
        mpjpe_arms = torch.mean(
            dof_pos_error[:, self.arm_dof_indices], dim=-1
        )  # [B]

        # MPJPE torso (using unified naming)
        mpjpe_waist = torch.mean(
            dof_pos_error[:, self.torso_dof_indices], dim=-1
        )  # [B]

        # MPJPE legs
        mpjpe_legs = torch.mean(
            dof_pos_error[:, self.leg_dof_indices], dim=-1
        )  # [B]

        # Initialize metric tensors if needed
        for metric_name in [
            "Task/MPJPE_WholeBody",
            "Task/MPJPE_Arms",
            "Task/MPJPE_Waist",
            "Task/MPJPE_Legs",
        ]:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # Update metric values
        self.metrics["Task/MPJPE_WholeBody"][:] = mpjpe_wholebody
        self.metrics["Task/MPJPE_Arms"][:] = mpjpe_arms
        self.metrics["Task/MPJPE_Waist"][:] = mpjpe_waist
        self.metrics["Task/MPJPE_Legs"][:] = mpjpe_legs

    def _update_mpkpe_metrics(self):
        """Update MPKPE (Mean Per Keybody Position Error) metrics."""
        # Get current and reference body positions
        current_body_pos = self.robot.data.body_pos_w  # [B, num_bodies, 3]
        ref_body_pos = self.get_ref_motion_bodylink_global_pos_immediate_next()
        # [B, num_bodies, 3]

        # Compute body position errors (L2 norm)
        body_pos_error = torch.norm(
            current_body_pos - ref_body_pos, dim=-1
        )  # [B, num_bodies]

        # MPKPE whole body
        mpkpe_wholebody = torch.mean(body_pos_error, dim=-1)  # [B]

        # MPKPE arms (using unified naming)
        mpkpe_arms = torch.mean(
            body_pos_error[:, self.arm_body_indices], dim=-1
        )  # [B]

        # MPKPE torso (using unified naming)
        mpkpe_waist = torch.mean(
            body_pos_error[:, self.torso_body_indices], dim=-1
        )  # [B]

        # MPKPE legs
        mpkpe_legs = torch.mean(
            body_pos_error[:, self.leg_body_indices], dim=-1
        )  # [B]

        # Initialize metric tensors if needed
        for metric_name in [
            "Task/MPKPE_WholeBody",
            "Task/MPKPE_Arms",
            "Task/MPKPE_Waist",
            "Task/MPKPE_Legs",
        ]:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # Update metric values
        self.metrics["Task/MPKPE_WholeBody"][:] = mpkpe_wholebody
        self.metrics["Task/MPKPE_Arms"][:] = mpkpe_arms
        self.metrics["Task/MPKPE_Waist"][:] = mpkpe_waist
        self.metrics["Task/MPKPE_Legs"][:] = mpkpe_legs

    # --- Pose-error getters for curriculum (WholeBody only) ---
    def get_wholebody_mpjpe(
        self,
    ) -> torch.Tensor:
        """[B] current whole-body MPJPE (URDF joint-space abs error)."""
        if self._task_space_target_source == "smpl_skeleton":
            return self.metrics.get(
                "Task/SkeletonSI_GlobalStructure",
                torch.zeros(self.num_envs, device=self.device),
            )
        if not hasattr(self, "metrics") or (
            "Task/MPJPE_WholeBody" not in self.metrics
        ):
            return torch.zeros(self.num_envs, device=self.device)
        return self.metrics["Task/MPJPE_WholeBody"]

    def get_wholebody_mpkpe(
        self,
    ) -> torch.Tensor:
        """[B] current whole-body MPKPE (body position error)."""
        if self._task_space_target_source == "smpl_skeleton":
            return self.metrics.get(
                "Task/SkeletonSI_NormalizedPosition",
                torch.zeros(self.num_envs, device=self.device),
            )
        if not hasattr(self, "metrics") or (
            "Task/MPKPE_WholeBody" not in self.metrics
        ):
            return torch.zeros(self.num_envs, device=self.device)
        return self.metrics["Task/MPKPE_WholeBody"]

    def get_current_motion_keys(
        self,
    ) -> list[str]:
        """Return motion window keys for the envs' current cached clips."""
        try:
            if hasattr(self, "_motion_cache") and hasattr(
                self._motion_cache, "motion_keys_for_indices"
            ):
                return self._motion_cache.motion_keys_for_indices(
                    self._clip_indices
                )
        except Exception:
            pass
        return []

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Just enable debug mode - visualizers will be created lazily in callback
            self._debug_vis_enabled = True
            # Set visibility if visualizers already exist
            if hasattr(self, "ref_body_visualizers"):
                for visualizer in self.ref_body_visualizers:
                    visualizer.set_visibility(True)
        else:
            self._debug_vis_enabled = False
            # Set visibility to false
            if hasattr(self, "ref_body_visualizers"):
                for visualizer in self.ref_body_visualizers:
                    visualizer.set_visibility(False)

    def setup_offline_eval_from_frame_zero(self):
        """Setup reference frame indices for offline evaluation from frame 0."""

        self._frame_indices[:] = 0

        self._update_ref_motion_state()

        logger.info(
            f"Offline evaluation setup complete: all {self.num_envs} "
            f"environments set to frame 0 references"
        )

    def setup_offline_eval_deterministic(
        self, apply_pending_swap: bool = True
    ) -> None:
        """Deterministic multi-env setup for offline evaluation.

        - Optionally apply a pending cache swap.
        - Set env i -> cache row i mapping for active clips, frame 0.
        - Update reference state only. Robot realignment is handled by caller.
        """
        if apply_pending_swap and getattr(self, "_swap_pending", False):
            self._motion_cache.advance()
            self._swap_pending = False
            self._swap_step_counter = 0

        clip_count = int(self._motion_cache.clip_count)
        active_count = min(int(self.num_envs), clip_count)

        # Reset indices
        self._clip_indices[:] = 0
        self._frame_indices[:] = 0

        if active_count > 0:
            active_ids = torch.arange(
                active_count, dtype=torch.long, device=self.device
            )
            self._clip_indices[active_ids] = torch.arange(
                active_count, dtype=torch.long, device=self.device
            )

        self._update_ref_motion_state_from_cache()

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # Check if debug visualization is enabled
        if not getattr(self, "_debug_vis_enabled", False):
            return

        # Check if motion cache/assignments are available
        if (
            not hasattr(self, "_motion_cache")
            or self._motion_cache is None
            or not hasattr(self, "_clip_indices")
            or not hasattr(self, "_frame_indices")
        ):
            return

        # Create visualizers lazily if they don't exist
        if not hasattr(self, "ref_body_visualizers"):
            self.ref_body_visualizers = []
            # Get number of bodies from the reference motion data
            num_bodies = self.get_ref_motion_bodylink_global_pos_cur().shape[
                -2
            ]
            for i in range(num_bodies):
                # Reference bodylinks as red spheres
                self.ref_body_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.body_keypoint_visualizer_cfg.replace(
                            prim_path=f"/Visuals/Command/ref_body_{i}"
                        )
                    )
                )

        # Visualize reference body keypoints
        if len(self.ref_body_visualizers) > 0:
            ref_body_pos = self.get_ref_motion_bodylink_global_pos_cur()
            # [B, num_bodies, 3]

            num_bodies = min(
                len(self.ref_body_visualizers), ref_body_pos.shape[1]
            )

            for i in range(num_bodies):
                # Visualize reference bodylinks as spheres (position only)
                self.ref_body_visualizers[i].visualize(
                    ref_body_pos[:, i],  # [B, 3]
                )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = RefMotionCommand

    command_obs_name: str = MISSING
    urdf_dof_names: list[str] = MISSING
    urdf_body_names: list[str] = MISSING

    # DOF name groupings for mpjpe metrics (using unified naming)
    arm_dof_names: list[str] = MISSING
    waist_dof_names: list[str] = MISSING
    leg_dof_names: list[str] = MISSING

    # Body name groupings for mpkpe metrics (using unified naming)
    arm_body_names: list[str] = MISSING
    torso_body_names: list[str] = MISSING
    leg_body_names: list[str] = MISSING

    motion_lib_cfg: dict = MISSING
    seed: int = MISSING
    process_id: int = MISSING
    num_processes: int = MISSING
    is_evaluating: bool = MISSING
    resample_time_interval_s: float = MISSING

    n_fut_frames: int = MISSING
    target_fps: int = MISSING

    anchor_bodylink_name: str = "pelvis"

    asset_name: str = MISSING
    debug_vis: bool = False

    root_pose_perturb_range: dict[str, tuple[float, float]] = {}
    root_vel_perturb_range: dict[str, tuple[float, float]] = {}
    dof_pos_perturb_range: tuple[float, float] = (-0.1, 0.1)
    dof_vel_perturb_range: tuple[float, float] = (-1.0, 1.0)

    body_keypoint_visualizer_cfg: VisualizationMarkersCfg = (
        SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/ref_keypoint")
    )
    body_keypoint_visualizer_cfg.markers["sphere"].radius = 0.03
    body_keypoint_visualizer_cfg.markers[
        "sphere"
    ].visual_material = PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.0, 1.0)  # blue
    )

    resampling_time_range: tuple[float, float] = (1.0, 1.0)


@configclass
class MoTrack_CommandsCfg:
    pass


def build_motion_tracking_commands_config(command_config_dict: dict):
    """Build isaaclab-compatible CommandsCfg from a config dictionary.

    Args:
        command_config_dict: Dictionary mapping command names to command configurations.
                           Each command config should contain the type and parameters.

    Example:
        command_config_dict = {
            "ref_motion": {
                "type": "MotionCommandCfg",
                "params": {
                    "command_obs_name": "bydmmc_ref_motion",
                    "motion_lib_cfg": {...},
                    "process_id": 0,
                    "num_processes": 1,
                    # ... other parameters
                }
            }
        }
    """

    commands_cfg = MoTrack_CommandsCfg()

    # Add command terms dynamically
    for command_name, command_config in command_config_dict.items():
        command_type = command_config.get("type", "MotionCommandCfg")
        command_params = command_config.get("params", {})

        # Get the command class type
        if command_type == "MotionCommandCfg":
            command_cfg = MotionCommandCfg(**command_params)
        else:
            raise ValueError(f"Unknown command type: {command_type}")

        # Add command to config
        setattr(commands_cfg, command_name, command_cfg)

    return commands_cfg
