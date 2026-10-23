"""Shared fixed-case Motion evidence; no training or reward implementation.

Targets and robot landmarks use the existing SMPL/skeleton builders. The
manifest is checked before inference, including every source window and seed.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from holomotion.src.evaluation.offline_eval_manifest import (
    validate_fixed_case_manifest,
)

SEMANTIC_FIELDS = (
    ("semantic_position_w", "position_w"),
    ("pelvis_rotation_wxyz", "pelvis_rotation_wxyz"),
    ("torso_rotation_wxyz", "torso_rotation_wxyz"),
    ("endpoint_rotation_wxyz", "endpoint_rotation_wxyz"),
)


def validate_motion_manifest(manifest, *, profile, stage, policy_dt):
    """Reject changed windows, duplicate seeds, or a reduced test set."""
    validate_fixed_case_manifest(manifest)
    if stage not in {"pilot", "final"}:
        raise ValueError("Unknown Motion acceptance stage")
    targets = profile[stage]["motion"]
    seeds = set(profile["seeds"]["evaluation"])
    windows, loaded = {}, {}
    for case in manifest["cases"]:
        if case.get("group") not in {"basic", "challenge"}:
            raise ValueError("Motion group must be basic or challenge")
        if (
            case.get("failure_rules") != profile["behavior"]["motion_failure"]
            or case.get("nominal") != profile["behavior"]["nominal"]
            or case.get("future_boundary")
            != profile["behavior"]["motion_future_boundary"]
            or not np.isclose(case.get("policy_dt_s", 0), policy_dt)
            or not np.isclose(case.get("motion_fps", 0) * policy_dt, 1)
        ):
            raise ValueError(
                "Motion case differs from frozen evaluation rules"
            )
        path = Path(case["source_npz_path"])
        digest = case["source_npz_sha256"]
        if path not in loaded:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"Motion source SHA mismatch: {path}")
            from holomotion.src.evaluation.gmr_free_smpl_mujoco import (
                load_pure_smpl_clip,
            )

            loaded[path] = (actual, load_pure_smpl_clip(path))
        actual, clip = loaded[path]
        if (
            actual != digest
            or clip.frame_count != case["planned_num_frames"]
            or case["source_start_frame"] < 0
            or case["source_stop_frame"] - case["source_start_frame"]
            != clip.frame_count
        ):
            raise ValueError("Motion source duration or identity mismatch")
        if not np.isclose(clip.motion_fps * policy_dt, 1):
            raise ValueError("Motion source FPS differs from control rate")
        if stage == "pilot" and (
            case["group"] != "basic"
            or clip.frame_count * policy_dt < targets["clip_duration_min_s"]
        ):
            raise ValueError(
                "Pilot requires complete basic windows of minimum duration"
            )
        key = (digest, case["source_start_frame"], case["source_stop_frame"])
        entry = windows.setdefault(
            key, {"group": case["group"], "seeds": set()}
        )
        if entry["group"] != case["group"] or case["seed"] in entry["seeds"]:
            raise ValueError(
                "Conflicting group or duplicate source-window seed"
            )
        entry["seeds"].add(case["seed"])
    if len(windows) < targets["clips_min"]:
        raise ValueError("Too few distinct frozen Motion windows")
    if any(window["seeds"] != seeds for window in windows.values()):
        raise ValueError("Every Motion window requires all frozen seeds")
    if (
        stage == "final"
        and sum(window["group"] == "basic" for window in windows.values())
        < targets["basic_clips_min"]
    ):
        raise ValueError("Too few frozen basic Motion windows")
    return {path: clip for path, (_, clip) in loaded.items()}


def semantic_arrays(state, *, prefix, index=None):
    result = {}
    for name, attr in SEMANTIC_FIELDS:
        value = getattr(state, attr).detach().cpu().numpy()
        result[prefix + "_" + name] = (
            value.copy() if index is None else value[index].copy()
        )
    return result


def anchored_smpl_arrays(
    clip, *, root_position, root_quaternion_wxyz, grounded_root_height=None
):
    """Mirror the existing episode XY/yaw anchor, without morphology warping.

    This is an evaluation bridge to the shared unanchored SMPL target builder;
    it does not construct actor observations or introduce semantic nodes.
    """
    import torch

    from holomotion.src.motion_tracking.smpl_skeleton_target import (
        SmplSkeletonTargetBuilder,
    )
    from holomotion.src.utils.torch_utils import quat_apply, quat_mul

    target = SmplSkeletonTargetBuilder(fps=clip.motion_fps).build(
        clip.human_pose_aa, clip.human_root_trans
    )
    # FormalSmplHdf5Dataset translates the whole skeleton to the shared
    # causal-grounded pelvis height. Relative-angle metrics are invariant to
    # this translation, but saved world-space targets must also match.
    if grounded_root_height is not None:
        height = torch.as_tensor(grounded_root_height, dtype=torch.float32)
        if height.shape != (clip.frame_count,):
            raise ValueError("Grounded reference height shape mismatch")
        target.position_h[..., 2] += (height - target.position_h[:, 0, 2])[
            :, None
        ]
    pelvis = target.pelvis_rotation_xyzw_h[:, [3, 0, 1, 2]]

    def yaw(q):
        w, x, y, z = q
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    root_q = torch.as_tensor(root_quaternion_wxyz, dtype=torch.float32)
    delta = yaw(root_q) - yaw(pelvis[0])
    anchor = torch.stack(
        (torch.cos(delta / 2), delta * 0, delta * 0, torch.sin(delta / 2))
    )
    translation = torch.zeros(3)
    translation[:2] = (
        torch.as_tensor(root_position, dtype=torch.float32)[:2]
        - quat_apply(anchor, target.position_h[0, 0])[:2]
    )
    position = (
        quat_apply(
            anchor.expand(target.position_h.shape[:-1] + (4,)),
            target.position_h,
        )
        + translation
    )

    def rotate(q):
        return quat_mul(anchor.expand(q.shape), q).numpy()

    return {
        "target_semantic_position_w": position.numpy(),
        "target_pelvis_rotation_wxyz": rotate(pelvis),
        "target_torso_rotation_wxyz": rotate(
            target.torso_rotation_xyzw_h[:, [3, 0, 1, 2]]
        ),
        "target_endpoint_rotation_wxyz": rotate(
            target.endpoint_rotation_xyzw_h[..., [3, 0, 1, 2]]
        ),
    }


def write_motion_case(
    output, case, rows, *, completed, failure, raw=None, extra=None
):
    """Save terminal state; failed remainder stays in the denominator."""
    from holomotion.src.evaluation.gmr_free_smpl_metrics import (
        compute_clip_metrics,
    )

    if (
        not rows
        or not 0 <= completed <= len(rows) <= case["planned_num_frames"]
    ):
        raise ValueError("Invalid captured/completed Motion duration")
    if completed < case["planned_num_frames"] and not failure:
        raise ValueError(
            "Incomplete Motion case needs an explicit failure reason"
        )
    required = {
        p + "_" + name
        for p in ("target", "robot")
        for name, _ in SEMANTIC_FIELDS
    }
    if any(set(row) != required for row in rows):
        raise ValueError(
            "Motion semantic schema differs from shared 16-metric input"
        )
    payload = {
        key: np.stack([row[key] for row in rows]) for key in sorted(required)
    }
    if any(not np.isfinite(value).all() for value in payload.values()):
        raise ValueError("Nonfinite Motion semantic evidence")
    metadata = {
        "schema": "gmr_free_smpl_eval_v1",
        "motion_key": case["case_id"],
        "num_frames": len(rows),
        "clip_length": case["planned_num_frames"],
        "valid_prefix_len": completed,
        "failure_reasons": failure,
        "failure_time_s": len(rows) * case["policy_dt_s"] if failure else None,
        "robot_reference_reads": 0,
        "sample_alignment": (
            "post-control robot vs clamped immediate-next SMPL target"
        ),
        **(extra or {}),
    }
    path = Path(output) / case["output_filename"]
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(
        path, **payload, **(raw or {}), metadata=json.dumps(metadata)
    )
    return {**compute_clip_metrics(path), **metadata}
