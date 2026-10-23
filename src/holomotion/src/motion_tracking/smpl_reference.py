"""Pure-SMPL reference features for HoloMotion actor commands.

This module intentionally contains no retargeting or IK dependency.  It loads
the fixed neutral SMPL skeleton, performs matrix forward kinematics, and emits
only the three tensors needed by the paired motion cache:

* ``smpl_body_command``: one of the Phase-2 body descriptors;
* ``smpl_root_pos``: grounded, segment-normalized root position;
* ``smpl_root_rot``: segment-normalized root orientation in XYZW order.

The robot reference remains the source of rewards, critic observations, reset
states, and terminations.  These features are exclusively an Actor input.
"""

from __future__ import annotations
import hashlib
import inspect
import os
import pickle
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from holomotion.src.utils import torch_utils

SMPL_JOINT_NAMES: Final[tuple[str, ...]] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

# Right-multiplied fixed local-basis conversion.  World coordinates are already
# canonical z-up/meters in formal HoloSMPL and are not permuted by this matrix.
SMPL_TO_COMMAND_FRAME: Final[tuple[tuple[float, ...], ...]] = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)

SMPL_BODY_COMMAND_DIMS: Final[dict[str, int]] = {
    "smpl_p23": 69,
    "smpl_a21": 63,
    "smpl_p23_a21": 132,
    "smpl_p23_r21": 195,
    "smpl_p23_a21_r21": 258,
    "smpl_p23_a6d21": 195,
    "smpl_p23_a6d_arm6": 105,
    "nmr_inspired_p23_v23": 138,
    "sonic_inspired_p23_rwrist": 81,
}
SMPL_BODY_COMMAND_MODE: Final[str] = "smpl_p23_a21_r21"
SMPL_BODY_COMMAND_DIM: Final[int] = SMPL_BODY_COMMAND_DIMS[
    SMPL_BODY_COMMAND_MODE
]

SMPL_FEATURE_SPEC_VERSION: Final[str] = "phase2_pure_smpl_v1"
SMPL_GROUND_POLICY_VERSION: Final[str] = "causal_min_j10_j11_v1"


@wp.kernel
def _pack_smpl_reference_intermediates_kernel(
    relative_world_pos: wp.array3d(dtype=wp.float32),
    global_quat_wxyz: wp.array3d(dtype=wp.float32),
    root_trans: wp.array2d(dtype=wp.float32),
    command_frame: wp.array2d(dtype=wp.float32),
    body_command: wp.array2d(dtype=wp.float32),
    foot_z: wp.array(dtype=wp.float32),
    root_command_rot: wp.array3d(dtype=wp.float32),
) -> None:
    """Pack P23 and root intermediates for the online CUDA path."""

    frame = wp.tid()
    root_quat = wp.quat(
        global_quat_wxyz[frame, 0, 1],
        global_quat_wxyz[frame, 0, 2],
        global_quat_wxyz[frame, 0, 3],
        global_quat_wxyz[frame, 0, 0],
    )
    for joint in range(1, 24):
        native_local = wp.quat_rotate_inv(
            root_quat,
            wp.vec3(
                relative_world_pos[frame, joint, 0],
                relative_world_pos[frame, joint, 1],
                relative_world_pos[frame, joint, 2],
            ),
        )
        offset = (joint - 1) * 3
        for axis in range(3):
            body_command[frame, offset + axis] = (
                command_frame[0, axis] * native_local[0]
                + command_frame[1, axis] * native_local[1]
                + command_frame[2, axis] * native_local[2]
            )
    foot_z[frame] = root_trans[frame, 2] + wp.min(
        relative_world_pos[frame, 10, 2],
        relative_world_pos[frame, 11, 2],
    )
    for column in range(3):
        world_column = wp.quat_rotate(
            root_quat,
            wp.vec3(
                command_frame[0, column],
                command_frame[1, column],
                command_frame[2, column],
            ),
        )
        for row in range(3):
            root_command_rot[frame, row, column] = world_column[row]


def launch_smpl_reference_intermediates_pack_warp(
    *,
    relative_world_pos,
    global_quat_wxyz,
    root_trans,
    command_frame,
    body_command,
    foot_z,
    root_command_rot,
    device,
) -> None:
    """Launch the training-identical P23 packer on Warp arrays."""

    wp.launch(
        _pack_smpl_reference_intermediates_kernel,
        dim=int(relative_world_pos.shape[0]),
        inputs=[
            relative_world_pos,
            global_quat_wxyz,
            root_trans,
            command_frame,
            body_command,
            foot_z,
            root_command_rot,
        ],
        device=device,
    )


def _default_smpl_model_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "thirdparties"
        / "smpl_models"
        / "SMPL_python_v.1.1.0"
        / "smpl"
        / "models"
        / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    )


def _enable_legacy_smpl_pickle_compat() -> None:
    """Provide only the compatibility shims needed to unpickle SMPL v1.1."""

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    try:
        import chumpy  # noqa: F401
    except Exception:
        chumpy_mod = types.ModuleType("chumpy")
        ch_mod = types.ModuleType("chumpy.ch")

        class Ch:
            def __setstate__(self, state: object) -> None:
                self.__dict__["_pickle_state"] = state

            def __getstate__(self) -> object:
                return self.__dict__.get("_pickle_state", {})

        ch_mod.Ch = Ch
        chumpy_mod.ch = ch_mod
        sys.modules.setdefault("chumpy", chumpy_mod)
        sys.modules.setdefault("chumpy.ch", ch_mod)


@dataclass(frozen=True)
class SmplSkeletonSpec:
    parents: torch.Tensor
    rest_parent_offsets: torch.Tensor
    skeleton_hash: str
    model_path: str


@lru_cache(maxsize=4)
def _load_neutral_smpl_skeleton_cached(path_str: str) -> SmplSkeletonSpec:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            "Licensed neutral SMPL model is required for pure-SMPL FK; "
            f"missing: {path}. Set HOLOMOTION_SMPL_MODEL_PATH if needed."
        )

    _enable_legacy_smpl_pickle_compat()
    with path.open("rb") as handle:
        model_data = pickle.load(handle, encoding="latin1")
    j_regressor = model_data["J_regressor"]
    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()
    rest_joints = (
        np.asarray(j_regressor, dtype=np.float32)
        @ np.asarray(model_data["v_template"], dtype=np.float32).reshape(-1, 3)
    ).astype(np.float32)
    parents = np.asarray(model_data["kintree_table"][0, :24], dtype=np.int64)
    parents[0] = -1
    offsets = np.zeros((24, 3), dtype=np.float32)
    for joint_id in range(1, 24):
        parent = int(parents[joint_id])
        if parent < 0 or parent >= joint_id:
            raise ValueError(
                f"Invalid SMPL parent {parent} for joint {joint_id} in {path}"
            )
        offsets[joint_id] = rest_joints[joint_id] - rest_joints[parent]

    digest = hashlib.sha256()
    digest.update(parents.tobytes())
    digest.update(offsets.tobytes())
    return SmplSkeletonSpec(
        parents=torch.from_numpy(parents.copy()),
        rest_parent_offsets=torch.from_numpy(offsets.copy()),
        skeleton_hash=digest.hexdigest(),
        model_path=str(path),
    )


def load_neutral_smpl_skeleton(
    model_path: str | os.PathLike[str] | None = None,
) -> SmplSkeletonSpec:
    """Load the 24-joint fixed-neutral SMPL skeleton once per process."""

    configured = model_path or os.environ.get("HOLOMOTION_SMPL_MODEL_PATH")
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else _default_smpl_model_path()
    )
    return _load_neutral_smpl_skeleton_cached(str(path))


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors ``[...,3]`` to matrices."""

    if rotvec.shape[-1] != 3:
        raise ValueError(f"rotvec must end in 3, got {tuple(rotvec.shape)}")
    theta2 = torch.sum(rotvec * rotvec, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1.0e-8
    # Taylor expansions avoid a zero-axis branch and preserve gradients.
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(1.0e-12),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-12),
    )
    x, y, z = torch.unbind(rotvec, dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(rotvec.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    eye = eye.expand(rotvec.shape[:-1] + (3, 3))
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return the first two columns using the project row-major layout."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end in 3x3, got {tuple(matrix.shape)}")
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def _continuous_xyzw_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    quat_wxyz = torch_utils.quat_from_matrix(matrix)
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    # Matrix-to-quaternion is sign-ambiguous.  Continuity prevents artificial
    # angular-velocity spikes while preserving the represented rotations.
    if quat_xyzw.ndim == 2:
        quat_xyzw = quat_xyzw.clone()
        for frame in range(1, quat_xyzw.shape[0]):
            if torch.dot(quat_xyzw[frame - 1], quat_xyzw[frame]) < 0.0:
                quat_xyzw[frame] = -quat_xyzw[frame]
    return quat_xyzw


@dataclass(frozen=True)
class SmplReferenceOutput:
    body_command: torch.Tensor
    root_pos: torch.Tensor
    root_rot_xyzw: torch.Tensor


class SmplReferenceBuilder:
    """Build a Phase-2 pure-SMPL Actor reference for one source slice."""

    def __init__(
        self,
        body_mode: str,
        *,
        fps: float = 50.0,
        ground_history_frames: int = 50,
        smpl_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if body_mode not in SMPL_BODY_COMMAND_DIMS:
            raise ValueError(
                f"Unsupported SMPL body mode {body_mode!r}; expected one of "
                f"{sorted(SMPL_BODY_COMMAND_DIMS)}"
            )
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if ground_history_frames <= 0:
            raise ValueError("ground_history_frames must be positive")
        self.body_mode = body_mode
        self.body_dim = SMPL_BODY_COMMAND_DIMS[body_mode]
        self.fps = float(fps)
        self.ground_history_frames = int(ground_history_frames)
        self.skeleton = load_neutral_smpl_skeleton(smpl_model_path)

    @property
    def feature_spec_hash(self) -> str:
        payload = "|".join(
            (
                SMPL_FEATURE_SPEC_VERSION,
                self.body_mode,
                str(self.body_dim),
                str(self.fps),
                SMPL_GROUND_POLICY_VERSION,
                str(self.ground_history_frames),
                self.skeleton.skeleton_hash,
                repr(SMPL_TO_COMMAND_FRAME),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _forward_kinematics(
        self, pose_aa: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_count = pose_aa.shape[0]
        local_rot = rotvec_to_matrix(pose_aa.reshape(frame_count, 24, 3))
        global_rot = torch.empty_like(local_rot)
        global_pos = torch.empty(
            (frame_count, 24, 3),
            dtype=pose_aa.dtype,
            device=pose_aa.device,
        )
        parents = self.skeleton.parents.to(device=pose_aa.device)
        offsets = self.skeleton.rest_parent_offsets.to(
            device=pose_aa.device, dtype=pose_aa.dtype
        )
        global_rot[:, 0] = local_rot[:, 0]
        # Keep joint offsets root-relative during FK.  Adding a large world
        # translation and subtracting it again would lose precision in P/V.
        global_pos[:, 0] = 0.0
        for joint_id in range(1, 24):
            parent = int(parents[joint_id].item())
            global_rot[:, joint_id] = (
                global_rot[:, parent] @ local_rot[:, joint_id]
            )
            offset = offsets[joint_id].view(1, 3, 1).expand(frame_count, 3, 1)
            global_pos[:, joint_id] = global_pos[:, parent] + (
                global_rot[:, parent] @ offset
            ).squeeze(-1)
        return global_pos, global_rot, local_rot

    def _rolling_ground(self, global_pos: torch.Tensor) -> torch.Tensor:
        foot_z = torch.minimum(global_pos[:, 10, 2], global_pos[:, 11, 2])
        history = self.ground_history_frames
        padded = F.pad(foot_z, (history - 1, 0), value=float("inf"))
        return padded.unfold(0, history, 1).amin(dim=-1)

    def build(
        self,
        human_pose_aa: torch.Tensor | np.ndarray,
        human_root_trans: torch.Tensor | np.ndarray,
        *,
        output_start: int = 0,
        output_length: int | None = None,
    ) -> SmplReferenceOutput:
        """Build features from context and crop to the requested output.

        ``output_start`` is the segment/window anchor inside the supplied
        source context. A paired Dataset may prepend up to 49 source frames so
        causal ground and the first V frame retain their real history.
        """

        pose = torch.as_tensor(
            human_pose_aa, dtype=torch.float32, device="cpu"
        )
        trans = torch.as_tensor(
            human_root_trans, dtype=torch.float32, device="cpu"
        )
        if pose.ndim != 2 or pose.shape[1] != 72:
            raise ValueError(
                f"human_pose_aa must be [T,72], got {tuple(pose.shape)}"
            )
        if trans.shape != (pose.shape[0], 3):
            raise ValueError(
                "human_root_trans must be [T,3] with the same T as pose, got "
                f"{tuple(trans.shape)} versus {tuple(pose.shape)}"
            )
        if not torch.isfinite(pose).all() or not torch.isfinite(trans).all():
            raise ValueError("SMPL pose/root input contains non-finite values")
        total = int(pose.shape[0])
        start = int(output_start)
        length = total - start if output_length is None else int(output_length)
        end = start + length
        if start < 0 or length <= 0 or end > total:
            raise ValueError(
                "Invalid output slice "
                f"start={start}, length={length}, context={total}"
            )

        relative_world_pos, global_rot, local_rot = self._forward_kinematics(
            pose
        )
        global_pos = relative_world_pos + trans[:, None]
        command_frame = torch.tensor(
            SMPL_TO_COMMAND_FRAME, dtype=pose.dtype, device=pose.device
        )

        # Root-local positions and frames do not depend on the common world-yaw
        # anchor, so compute them directly in the command local basis.
        root_inv = global_rot[:, 0].transpose(-1, -2)
        rel_pos_native = (
            root_inv[:, None] @ relative_world_pos.unsqueeze(-1)
        ).squeeze(-1)
        positions = (
            command_frame.T[None, None] @ rel_pos_native.unsqueeze(-1)
        ).squeeze(-1)
        p23 = positions[:, 1:24].reshape(total, 69)

        v23 = torch.zeros_like(p23)
        if total > 1:
            v23[1:] = (p23[1:] - p23[:-1]) * self.fps

        root_relative = (
            command_frame.T[None, None]
            @ (root_inv[:, None] @ global_rot)
            @ command_frame[None, None]
        )
        r21 = matrix_to_rot6d(root_relative[:, 1:22]).reshape(total, 126)
        r_wrist = matrix_to_rot6d(root_relative[:, 20:22]).reshape(total, 12)
        a21_native = pose.reshape(total, 24, 3)[:, 1:22].reshape(total, 63)
        # A6D retains A21's parent-relative SMPL local-rotation semantics; it
        # only replaces axis-angle with the continuous 6D representation.  It
        # is intentionally distinct from R21's root-relative global frames.
        a6d21 = matrix_to_rot6d(local_rot[:, 1:22]).reshape(total, 126)
        # Six arm joints in the canonical HoloSMPL layout: left/right
        # shoulder, elbow and wrist (indices 16..21).  These remain local,
        # parent-relative rotations; only their representation is changed
        # from axis-angle to continuous rotation-6D.
        a6d_arm6 = matrix_to_rot6d(local_rot[:, 16:22]).reshape(total, 36)

        if self.body_mode == "smpl_p23":
            body = p23
        elif self.body_mode == "smpl_a21":
            body = a21_native
        elif self.body_mode == "smpl_p23_a21":
            body = torch.cat((p23, a21_native), dim=-1)
        elif self.body_mode == "smpl_p23_r21":
            body = torch.cat((p23, r21), dim=-1)
        elif self.body_mode == "smpl_p23_a21_r21":
            body = torch.cat((p23, a21_native, r21), dim=-1)
        elif self.body_mode == "smpl_p23_a6d21":
            body = torch.cat((p23, a6d21), dim=-1)
        elif self.body_mode == "smpl_p23_a6d_arm6":
            body = torch.cat((p23, a6d_arm6), dim=-1)
        elif self.body_mode == "nmr_inspired_p23_v23":
            body = torch.cat((p23, v23), dim=-1)
        elif self.body_mode == "sonic_inspired_p23_rwrist":
            body = torch.cat((p23, r_wrist), dim=-1)
        else:  # protected by __init__, retained for static exhaustiveness
            raise AssertionError(self.body_mode)
        if body.shape != (total, self.body_dim):
            raise AssertionError(
                f"Internal body shape mismatch: {tuple(body.shape)} != "
                f"{(total, self.body_dim)}"
            )

        ground_z = self._rolling_ground(global_pos)
        grounded = trans.clone()
        grounded[:, 2] -= ground_z

        unanchored_root_rot = global_rot[:, 0] @ command_frame
        anchor_rot = unanchored_root_rot[start]
        anchor_yaw = torch.atan2(anchor_rot[1, 0], anchor_rot[0, 0])
        cos_yaw = torch.cos(-anchor_yaw)
        sin_yaw = torch.sin(-anchor_yaw)
        yaw_align = torch.stack(
            (
                torch.stack((cos_yaw, -sin_yaw, torch.zeros_like(cos_yaw))),
                torch.stack((sin_yaw, cos_yaw, torch.zeros_like(cos_yaw))),
                torch.stack(
                    (
                        torch.zeros_like(cos_yaw),
                        torch.zeros_like(cos_yaw),
                        torch.ones_like(cos_yaw),
                    )
                ),
            )
        )
        normalized_rot = yaw_align[None] @ unanchored_root_rot

        normalized_pos = grounded.clone()
        normalized_pos[:, :2] -= trans[start, :2]
        normalized_pos = (
            yaw_align[None] @ normalized_pos.unsqueeze(-1)
        ).squeeze(-1)
        root_rot_xyzw = _continuous_xyzw_from_matrix(normalized_rot)

        return SmplReferenceOutput(
            body_command=body[start:end].contiguous(),
            root_pos=normalized_pos[start:end].contiguous(),
            root_rot_xyzw=root_rot_xyzw[start:end].contiguous(),
        )


__all__ = [
    "SMPL_BODY_COMMAND_DIMS",
    "SMPL_BODY_COMMAND_DIM",
    "SMPL_BODY_COMMAND_MODE",
    "SMPL_FEATURE_SPEC_VERSION",
    "SMPL_GROUND_POLICY_VERSION",
    "SMPL_JOINT_NAMES",
    "SMPL_TO_COMMAND_FRAME",
    "SmplReferenceBuilder",
    "SmplReferenceOutput",
    "SmplSkeletonSpec",
    "load_neutral_smpl_skeleton",
    "matrix_to_rot6d",
    "rotvec_to_matrix",
    "launch_smpl_reference_intermediates_pack_warp",
]
