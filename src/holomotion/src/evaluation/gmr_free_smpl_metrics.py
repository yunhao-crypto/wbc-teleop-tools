"""Pure-SMPL task-space metrics for GMR-free policy evaluation.

The input schema is emitted by :meth:`PPO.offline_evaluate_policy` when the
command target is ``smpl_skeleton``.  This module intentionally has no Robot-ref
or retargeted-q dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from holomotion.src.motion_tracking.semantic_skeleton import (
    LOCAL_CHAIN_EDGE_PAIRS,
    NODE_INDEX,
    SEMANTIC_EDGES,
)

METRIC_NAMES = (
    "pelvis_gravity_rad",
    "pelvis_yaw_rad",
    "pelvis_hip_direction_rad",
    "thigh_direction_rad",
    "shank_direction_rad",
    "knee_bend_rad",
    "leg_plane_rad",
    "foot_relative_orientation_rad",
    "pelvis_torso_direction_rad",
    "torso_shoulder_direction_rad",
    "torso_relative_orientation_rad",
    "upper_arm_direction_rad",
    "forearm_direction_rad",
    "elbow_bend_rad",
    "arm_plane_rad",
    "wrist_relative_orientation_rad",
)

PLANE_SIN_MIN = 5.0e-2


def _normalize(value: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return value / np.maximum(
        np.linalg.norm(value, axis=-1, keepdims=True), eps
    )


def _vector_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosine = np.sum(_normalize(left) * _normalize(right), axis=-1)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = _normalize(np.asarray(quaternion, dtype=np.float64))
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _so3_angle(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def _relative_rotation(root: np.ndarray, body: np.ndarray) -> np.ndarray:
    return np.einsum("tji,tnjk->tnik", root, body)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _yaw(rotation: np.ndarray) -> np.ndarray:
    return np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])


def _edge_directions(
    position: np.ndarray, pelvis_rotation: np.ndarray
) -> np.ndarray:
    parents = np.asarray([edge[0] for edge in SEMANTIC_EDGES])
    children = np.asarray([edge[1] for edge in SEMANTIC_EDGES])
    vector_w = position[:, children] - position[:, parents]
    vector_local = np.einsum("tji,tej->tei", pelvis_rotation, vector_w)
    return _normalize(vector_local)


def _chain_features(
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proximal = np.asarray([pair[0] for pair in LOCAL_CHAIN_EDGE_PAIRS])
    distal = np.asarray([pair[1] for pair in LOCAL_CHAIN_EDGE_PAIRS])
    first = direction[:, proximal]
    second = direction[:, distal]
    bend = _vector_angle(first, second)
    plane = np.cross(first, second)
    plane_norm = np.linalg.norm(plane, axis=-1)
    plane_normal = plane / np.maximum(plane_norm[..., None], PLANE_SIN_MIN)
    return bend, plane_normal, plane_norm > PLANE_SIN_MIN


def _mean_plane_error(
    target_plane: np.ndarray,
    target_valid: np.ndarray,
    robot_plane: np.ndarray,
    robot_valid: np.ndarray,
    indices: tuple[int, int],
) -> float | None:
    selected = np.asarray(indices, dtype=np.int64)
    valid = target_valid[:, selected] & robot_valid[:, selected]
    if not bool(np.any(valid)):
        return None
    cosine = np.sum(
        target_plane[:, selected] * robot_plane[:, selected], axis=-1
    )
    # Reward treats the bend plane as unoriented: n and -n are equivalent.
    error = np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))
    return float(np.mean(error[valid]))


def compute_clip_metrics(npz_path: str | Path) -> dict[str, object]:
    """Compute the 16-row human task-space table for one pure-SMPL NPZ."""

    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema") != "gmr_free_smpl_eval_v1":
            raise ValueError(f"{path} is not a gmr_free_smpl_eval_v1 file")
        target_position = data["target_semantic_position_w"].astype(np.float64)
        robot_position = data["robot_semantic_position_w"].astype(np.float64)
        target_pelvis = _quat_to_matrix_wxyz(
            data["target_pelvis_rotation_wxyz"]
        )
        robot_pelvis = _quat_to_matrix_wxyz(data["robot_pelvis_rotation_wxyz"])
        target_torso = _quat_to_matrix_wxyz(data["target_torso_rotation_wxyz"])
        robot_torso = _quat_to_matrix_wxyz(data["robot_torso_rotation_wxyz"])
        target_endpoint = _quat_to_matrix_wxyz(
            data["target_endpoint_rotation_wxyz"]
        )
        robot_endpoint = _quat_to_matrix_wxyz(
            data["robot_endpoint_rotation_wxyz"]
        )

    target_direction = _edge_directions(target_position, target_pelvis)
    robot_direction = _edge_directions(robot_position, robot_pelvis)
    direction_error = _vector_angle(target_direction, robot_direction)
    target_bend, target_plane, target_plane_valid = _chain_features(
        target_direction
    )
    robot_bend, robot_plane, robot_plane_valid = _chain_features(
        robot_direction
    )

    gravity_w = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    target_gravity = np.einsum("tji,j->ti", target_pelvis, gravity_w)
    robot_gravity = np.einsum("tji,j->ti", robot_pelvis, gravity_w)
    yaw_error = np.abs(_wrap_angle(_yaw(robot_pelvis) - _yaw(target_pelvis)))

    target_torso_rel = _relative_rotation(
        target_pelvis, target_torso[:, None]
    )[:, 0]
    robot_torso_rel = _relative_rotation(robot_pelvis, robot_torso[:, None])[
        :, 0
    ]
    target_endpoint_rel = _relative_rotation(target_pelvis, target_endpoint)
    robot_endpoint_rel = _relative_rotation(robot_pelvis, robot_endpoint)
    torso_rel_error = _so3_angle(
        np.einsum("tji,tjk->tik", target_torso_rel, robot_torso_rel)
    )
    endpoint_rel_error = _so3_angle(
        np.einsum("tnji,tnjk->tnik", target_endpoint_rel, robot_endpoint_rel)
    )

    values = {
        "pelvis_gravity_rad": float(
            np.mean(_vector_angle(target_gravity, robot_gravity))
        ),
        "pelvis_yaw_rad": float(np.mean(yaw_error)),
        "pelvis_hip_direction_rad": float(np.mean(direction_error[:, (0, 3)])),
        "thigh_direction_rad": float(np.mean(direction_error[:, (1, 4)])),
        "shank_direction_rad": float(np.mean(direction_error[:, (2, 5)])),
        "knee_bend_rad": float(
            np.mean(np.abs(target_bend[:, :2] - robot_bend[:, :2]))
        ),
        "leg_plane_rad": _mean_plane_error(
            target_plane,
            target_plane_valid,
            robot_plane,
            robot_plane_valid,
            (0, 1),
        ),
        "foot_relative_orientation_rad": float(
            np.mean(endpoint_rel_error[:, :2])
        ),
        "pelvis_torso_direction_rad": float(np.mean(direction_error[:, 6])),
        "torso_shoulder_direction_rad": float(
            np.mean(direction_error[:, (7, 10)])
        ),
        "torso_relative_orientation_rad": float(np.mean(torso_rel_error)),
        "upper_arm_direction_rad": float(np.mean(direction_error[:, (8, 11)])),
        "forearm_direction_rad": float(np.mean(direction_error[:, (9, 12)])),
        "elbow_bend_rad": float(
            np.mean(np.abs(target_bend[:, 2:] - robot_bend[:, 2:]))
        ),
        "arm_plane_rad": _mean_plane_error(
            target_plane,
            target_plane_valid,
            robot_plane,
            robot_plane_valid,
            (2, 3),
        ),
        "wrist_relative_orientation_rad": float(
            np.mean(endpoint_rel_error[:, 2:])
        ),
    }
    defined = [value for value in values.values() if value is not None]
    if (
        tuple(values) != METRIC_NAMES
        or not defined
        or not np.isfinite(defined).all()
    ):
        raise ValueError(f"invalid metrics for {path}")
    frame_count = int(target_position.shape[0])
    if frame_count == 0 or frame_count != int(metadata["num_frames"]):
        raise ValueError(f"Invalid frame count for {path}")
    coverage = {}
    for name, indices in (
        ("leg_plane_rad", (0, 1)),
        ("arm_plane_rad", (2, 3)),
    ):
        mask = target_plane_valid[:, indices] & robot_plane_valid[:, indices]
        coverage[name + "_valid_samples"] = int(mask.sum())
        coverage[name + "_total_samples"] = int(mask.size)
        coverage[name + "_coverage"] = float(mask.mean())
    planned = metadata.get("clip_length")
    valid_prefix = metadata.get("valid_prefix_len")
    completion = None
    if planned is not None and valid_prefix is not None:
        if int(planned) <= 0 or not 0 <= int(valid_prefix) <= int(planned):
            raise ValueError(
                f"Invalid planned/completed frame counts for {path}"
            )
        completion = float(valid_prefix) / float(planned)
    return {
        "motion_key": str(metadata["motion_key"]),
        "num_frames": int(metadata["num_frames"]),
        **values,
        # Preserve the existing mean exactly when all 16 metrics are defined.
        # Undefined planes never become zero; expose their changed denominator.
        "mean_tracking_error_rad": float(np.mean(defined)),
        "valid_metric_count": len(defined),
        "metric_count": len(METRIC_NAMES),
        **coverage,
        "planned_num_frames": int(planned) if planned is not None else None,
        "valid_prefix_frames": int(valid_prefix)
        if valid_prefix is not None
        else None,
        "completion_fraction": completion,
    }


def aggregate_directory(
    input_dir: str | Path, *, expected_manifest=None
) -> dict[str, object]:
    """Write per-clip CSV and macro/micro JSON for one evaluation directory."""

    root = Path(input_dir)
    expected = None
    manifest = None
    if expected_manifest is not None:
        from holomotion.src.evaluation.offline_eval_manifest import (
            validate_fixed_case_manifest,
        )

        manifest = (
            json.loads(Path(expected_manifest).read_text())
            if isinstance(expected_manifest, (str, Path))
            else expected_manifest
        )
        expected = validate_fixed_case_manifest(manifest)
    paths = sorted(root.glob("*.npz"))
    if expected is not None and set(p.name for p in paths) - set(expected):
        raise ValueError(
            "Unexpected rollout files outside the frozen case manifest"
        )
    records = [compute_clip_metrics(path) for path in paths]
    if not records and expected is None:
        raise ValueError(f"no pure-SMPL NPZ files found under {root}")
    frame_weights = np.asarray([record["num_frames"] for record in records])

    names = (*METRIC_NAMES, "mean_tracking_error_rad")

    def summarize(weights: np.ndarray) -> dict[str, float | None]:
        result = {}
        for name in names:
            valid = np.asarray(
                [record[name] is not None for record in records]
            )
            if not valid.any():
                result[name] = None
                continue
            selected_weights = weights[valid]
            result[name] = float(
                np.average(
                    [
                        record[name]
                        for record in records
                        if record[name] is not None
                    ],
                    weights=selected_weights,
                )
            )
        return result

    summary = {
        "schema": "gmr_free_smpl_metrics_v2",
        "clip_count": len(records),
        "frame_count": int(frame_weights.sum()),
        "macro": summarize(np.ones(len(records), dtype=np.float64)),
        "micro": summarize(frame_weights.astype(np.float64)),
        "coverage": {
            name: {
                "valid_clips": sum(
                    record[name] is not None for record in records
                ),
                "total_clips": len(records),
                "valid_clip_fraction": sum(
                    record[name] is not None for record in records
                )
                / len(records)
                if records
                else 0.0,
            }
            for name in names
        },
        "plane_sample_coverage": {
            name: {
                "valid_samples": sum(
                    record[name + "_valid_samples"] for record in records
                ),
                "total_samples": sum(
                    record[name + "_total_samples"] for record in records
                ),
            }
            for name in ("leg_plane_rad", "arm_plane_rad")
        },
        "completion_evidence_clips": sum(
            record["completion_fraction"] is not None for record in records
        ),
        "completion_scope": "Only files present; an externally frozen expected-clip manifest is required to count missing/failed-to-start clips.",
    }
    if expected is not None:
        by_name = {
            path.name: record
            for path, record in zip(paths, records, strict=True)
        }
        case_results = []
        for filename, case in expected.items():
            record = by_name.get(filename)
            planned = case["planned_num_frames"]
            if record is None:
                completed, fraction, available = 0, 0.0, False
            else:
                if record["motion_key"] != case["case_id"]:
                    raise ValueError(
                        f"Rollout case identity differs from frozen manifest: {filename}"
                    )
                if (
                    record["planned_num_frames"] != planned
                    or record["valid_prefix_frames"] is None
                ):
                    raise ValueError(
                        f"Missing/conflicting duration evidence for {filename}"
                    )
                completed = record["valid_prefix_frames"]
                if completed > record["num_frames"]:
                    raise ValueError(
                        f"Claimed completion exceeds captured evidence: {filename}"
                    )
                fraction, available = completed / planned, True
            case_results.append(
                {
                    **case,
                    "captured": available,
                    "completed_frames": completed,
                    "completion_fraction": fraction,
                    "full_completion": completed == planned,
                    "metrics": None
                    if record is None
                    else {name: record[name] for name in names},
                }
            )
        groups = {}
        for group in sorted(
            {case["group"] for case in case_results} | {"all"}
        ):
            selected = [
                case
                for case in case_results
                if group == "all" or case["group"] == group
            ]
            defined = [
                case["metrics"]
                for case in selected
                if case["metrics"] is not None
            ]
            groups[group] = {
                "expected_cases": len(selected),
                "captured_cases": len(defined),
                "mean_completion_fraction": float(
                    np.mean([c["completion_fraction"] for c in selected])
                ),
                "full_completion_rate": float(
                    np.mean([c["full_completion"] for c in selected])
                ),
                "all_expected_metrics_present": len(defined) == len(selected),
                "case_macro": {
                    name: float(
                        np.mean(
                            [r[name] for r in defined if r[name] is not None]
                        )
                    )
                    if any(r[name] is not None for r in defined)
                    else None
                    for name in names
                },
            }
        summary.update(
            expected_case_count=len(expected),
            missing_case_count=len(expected) - len(records),
            expected_cases_sha256=manifest["cases_sha256"],
            groups=groups,
            completion_scope="Frozen expected cases; missing outputs count as zero duration and failures. Error metrics remain null, never zero-imputed; missing metrics cannot qualify a model.",
        )
        (root / "gmr_free_smpl_cases.json").write_text(
            json.dumps(case_results, indent=2, allow_nan=False) + "\n"
        )
    csv_path = root / "gmr_free_smpl_metrics_per_clip.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=records[0].keys()
            if records
            else ["motion_key", "num_frames"],
        )
        writer.writeheader()
        writer.writerows(records)
    summary_path = root / "gmr_free_smpl_metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_directory(
                args.input_dir, expected_manifest=args.expected_manifest
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
