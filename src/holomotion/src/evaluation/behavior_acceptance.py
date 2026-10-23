"""Shared, profile-driven behavior gates; no simulator or robot-specific policy.

Completion includes every frozen case. Errors describe observed samples and carry
coverage; missing errors are never replaced with zero. Physics backends supply
terminal state *before* an automatic reset can hide a failure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from holomotion.src.evaluation.offline_eval_manifest import (
    build_fixed_case_manifest,
)
from holomotion.src.evaluation.offline_eval_manifest import (
    validate_fixed_case_manifest,
)


def heading_velocity_command_frame(
    linear_world, angular_world, quaternion_wxyz
) -> np.ndarray:
    """Match common Velocity rewards: yaw-local xy, world angular z (not body z)."""
    linear, angular, q = (
        np.asarray(x, dtype=np.float64)
        for x in (linear_world, angular_world, quaternion_wxyz)
    )
    if (
        linear.shape[-1] != 3
        or angular.shape != linear.shape
        or q.shape != linear.shape[:-1] + (4,)
    ):
        raise ValueError("Velocity/orientation shapes differ")
    if not all(np.isfinite(x).all() for x in (linear, angular, q)):
        raise ValueError("Nonfinite velocity/orientation")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero orientation")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            c * linear[..., 0] + s * linear[..., 1],
            -s * linear[..., 0] + c * linear[..., 1],
            angular[..., 2],
        ),
        axis=-1,
    )


def gravity_z_in_body(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Invalid orientation")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero orientation")
    q = q / norm
    return 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2) - 1.0


@dataclass
class VelocityFailureMonitor:
    rules: Mapping[str, Any]
    contact_duration_s: float = 0.0

    def observe(
        self,
        *,
        root_position,
        root_quaternion_wxyz,
        forbidden_ground_normal_force_n: float,
        dt_s: float,
    ) -> list[str]:
        if dt_s <= 0 or not np.isfinite(dt_s):
            raise ValueError("Contact sampling interval must be positive")
        p, q = np.asarray(root_position), np.asarray(root_quaternion_wxyz)
        if p.shape != (3,) or q.shape != (4,):
            raise ValueError("Root state shape mismatch")
        if (
            not np.isfinite(p).all()
            or not np.isfinite(q).all()
            or not np.isfinite(forbidden_ground_normal_force_n)
        ):
            return ["nonfinite_state_or_contact"]
        if forbidden_ground_normal_force_n < 0:
            raise ValueError("Normal force magnitude cannot be negative")
        reasons = []
        if p[2] < self.rules["root_height_min_m"]:
            reasons.append("root_height")
        try:
            tilt = float(np.arccos(np.clip(-gravity_z_in_body(q), -1, 1)))
        except ValueError:
            return ["invalid_orientation"]
        if tilt > self.rules["tilt_max_rad"]:
            reasons.append("root_tilt")
        self.contact_duration_s = (
            self.contact_duration_s + dt_s
            if forbidden_ground_normal_force_n
            > self.rules["forbidden_ground_normal_force_min_n"]
            else 0.0
        )
        if (
            self.contact_duration_s + 1e-12
            >= self.rules["forbidden_ground_contact_duration_s"]
        ):
            reasons.append("forbidden_ground_contact")
        return reasons


def motion_failure_reasons(
    *, target_torso_wxyz, robot_torso_wxyz, threshold: float
) -> list[str]:
    """NumPy equivalent of the existing SMPL torso-gravity termination rule."""
    try:
        difference = np.abs(
            gravity_z_in_body(target_torso_wxyz)
            - gravity_z_in_body(robot_torso_wxyz)
        )
    except ValueError:
        return ["nonfinite_or_invalid_torso_orientation"]
    if np.asarray(difference).shape != ():
        raise ValueError(
            "One Motion failure monitor expects one target/robot pose"
        )
    return (
        ["smpl_torso_gravity_projection_far"] if difference > threshold else []
    )


def velocity_case_manifest(
    profile: Mapping, *, stage: str, policy_dt: float
) -> dict:
    if stage not in {"pilot", "final"}:
        raise ValueError("Behavior stage must be pilot or final")
    targets = profile[stage]["velocity"]
    cases = []
    for seed in profile["seeds"]["evaluation"]:
        programs = [
            (
                "stand",
                "stand",
                [
                    {
                        "duration_s": targets["stand_duration_s"],
                        "command": [0.0, 0.0, 0.0],
                    }
                ],
            )
        ]
        for index, cmd in enumerate(targets["commands"]):
            if not any(cmd):
                continue  # Each final nonzero command explicitly returns to zero below.
            segments = [
                {
                    "duration_s": targets["command_duration_s"],
                    "command": list(cmd),
                }
            ]
            if stage == "final":
                segments.append(
                    {
                        "duration_s": targets["command_duration_s"],
                        "command": [0.0, 0.0, 0.0],
                    }
                )
            programs.append((f"command_{index:02d}", "command", segments))
        for name, group, segments in programs:
            case_id = f"velocity_{stage}_{name}_seed{seed}"
            steps = sum(float(s["duration_s"]) / policy_dt for s in segments)
            if abs(steps - round(steps)) > 1e-6:
                raise ValueError(
                    "Duration is not a whole number of policy intervals"
                )
            cases.append(
                {
                    "case_id": case_id,
                    "output_filename": case_id + ".npz",
                    "seed": int(seed),
                    "group": group,
                    "planned_num_frames": round(steps),
                    "policy_dt_s": policy_dt,
                    "segments": segments,
                    "failure_rules": dict(
                        profile["behavior"]["velocity_failure"]
                    ),
                    "transition_s": float(targets["transition_s"]),
                    "nominal": dict(profile["behavior"]["nominal"]),
                }
            )
    return build_fixed_case_manifest(cases)


def velocity_command_at(case: Mapping, interval_index: int) -> np.ndarray:
    if not 0 <= interval_index < int(case["planned_num_frames"]):
        raise ValueError("Velocity interval outside frozen case")
    time_s = interval_index * float(case["policy_dt_s"])
    elapsed = 0.0
    for segment in case["segments"]:
        elapsed += segment["duration_s"]
        if time_s < elapsed - 1e-10:
            return np.asarray(segment["command"], dtype=np.float32)
    raise ValueError("Velocity program duration does not cover interval")


def velocity_case_metrics(
    case: Mapping,
    *,
    root_position,
    root_velocity_local,
    completed_intervals: int,
    failure_reasons: list[str],
) -> dict:
    """Each trace row is a post-control, pre-reset state for one policy interval."""
    position = np.asarray(root_position, dtype=np.float64)
    velocity = np.asarray(root_velocity_local, dtype=np.float64)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or velocity.shape != position.shape
    ):
        raise ValueError(
            "Expected matching [T,3] position and vx/vy/yaw traces"
        )
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("Nonfinite traces cannot yield passing metrics")
    planned = int(case["planned_num_frames"])
    if not 0 <= completed_intervals <= len(position) <= planned:
        raise ValueError("Invalid completed/captured/planned interval counts")
    if (completed_intervals < planned) != bool(failure_reasons):
        raise ValueError(
            "Incomplete case needs a failure reason; complete case cannot hide one"
        )
    dt = float(case["policy_dt_s"])
    segments = []
    start = 0
    for segment in case["segments"]:
        length = round(float(segment["duration_s"]) / dt)
        end = start + length
        observed = velocity[start : min(end, len(velocity))]
        transition = round(float(case["transition_s"]) / dt)
        steady = observed[transition:]
        command = np.asarray(segment["command"], dtype=np.float64)

        def errors(samples):
            return (
                {
                    "linear_rmse_m_s": float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (samples[:, :2] - command[:2]) ** 2,
                                    axis=-1,
                                )
                            )
                        )
                    ),
                    "yaw_rmse_rad_s": float(
                        np.sqrt(np.mean((samples[:, 2] - command[2]) ** 2))
                    ),
                }
                if len(samples)
                else {"linear_rmse_m_s": None, "yaw_rmse_rad_s": None}
            )

        active = np.abs(command) > 1e-8
        response = (
            (steady.mean(axis=0)[active] / command[active]).tolist()
            if len(steady) and active.any()
            else []
        )
        segments.append(
            {
                "command": command.tolist(),
                "planned_intervals": length,
                "captured_intervals": len(observed),
                "completed_intervals": max(
                    0, min(completed_intervals, end) - start
                ),
                "steady_captured_intervals": len(steady),
                "steady_planned_intervals": length - transition,
                "steady": errors(steady),
                "transition": errors(observed[:transition]),
                "signed_response_fractions": response,
            }
        )
        start = end
    if start != planned:
        raise ValueError("Frozen command program length mismatch")
    return {
        "case_id": case["case_id"],
        "seed": case["seed"],
        "group": case["group"],
        "planned_intervals": planned,
        "captured_intervals": len(position),
        "completed_intervals": completed_intervals,
        "completion_fraction": completed_intervals / planned,
        "full_completion": completed_intervals == planned,
        "failure_reasons": failure_reasons,
        "segments": segments,
        # Caller includes the actual reset position, not the first post-step position.
        "position_trace_required_for_reset_relative_drift": True,
        "error_scope": "Observed intervals including terminal sample; missing remainder counted in completion, not zero-imputed errors.",
    }


def judge_motion_group(
    summary: Mapping, *, profile: Mapping, stage: str, group: str = "basic"
) -> dict:
    targets = profile[stage]["motion"]
    result = summary.get("groups", {}).get(group)
    if result is None:
        return {"passed": False, "failures": ["missing_frozen_group"]}
    error = result["case_macro"].get("mean_tracking_error_rad")
    conditions = {
        "all_expected_metrics_present": result["all_expected_metrics_present"],
        "duration": result["mean_completion_fraction"]
        >= targets["duration_fraction_min"],
        "completion": result["full_completion_rate"]
        >= targets["clip_success_rate_min"],
        "mean_error": error is not None
        and error <= targets["clip_macro_mean_tracking_error_max_rad"],
    }
    if stage == "final":
        wrist = result["case_macro"].get("wrist_relative_orientation_rad")
        conditions["wrist_error"] = (
            wrist is not None
            and wrist <= targets["clip_macro_wrist_orientation_max_rad"]
        )
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "failures": [
            name for name, passed in conditions.items() if not passed
        ],
    }


def judge_velocity_cases(
    manifest: Mapping, results: list[Mapping], *, profile: Mapping, stage: str
) -> dict:
    expected = {
        case["case_id"]: case
        for case in validate_fixed_case_manifest(manifest).values()
    }
    actual = {r["case_id"]: r for r in results}
    if len(actual) != len(results) or set(actual) - set(expected):
        raise ValueError("Duplicate/unexpected velocity cases")
    targets = profile[stage]["velocity"]
    stand = [case for case in expected.values() if case["group"] == "stand"]
    commands = [
        case for case in expected.values() if case["group"] == "command"
    ]
    conditions = {
        "all_expected_cases_present": set(actual) == set(expected),
        "stand": len(stand) == targets["stand_success_count"]
        and all(
            actual.get(c["case_id"], {}).get("full_completion", False)
            for c in stand
        ),
        "commands": bool(commands)
        and sum(
            actual.get(c["case_id"], {}).get("full_completion", False)
            for c in commands
        )
        / len(commands)
        >= targets["completion_rate_min"],
    }
    if stage == "final":
        conditions["stand_drift"] = all(
            actual.get(c["case_id"], {}).get(
                "reset_relative_drift_max_m", float("inf")
            )
            <= targets["stand_drift_max_m"]
            for c in stand
        )
    all_segments = [s for r in actual.values() for s in r["segments"]]
    # Per-case/segment checks avoid a good direction masking a failed direction.
    conditions["linear_error"] = bool(all_segments) and all(
        s["steady"]["linear_rmse_m_s"] is not None
        and s["steady"]["linear_rmse_m_s"] <= targets["linear_rmse_max_m_s"]
        for s in all_segments
    )
    conditions["yaw_error"] = bool(all_segments) and all(
        s["steady"]["yaw_rmse_rad_s"] is not None
        and s["steady"]["yaw_rmse_rad_s"] <= targets["yaw_rmse_max_rad_s"]
        for s in all_segments
    )
    if stage == "pilot":
        active = [s for s in all_segments if any(s["command"])]
        conditions["directional_response"] = bool(active) and all(
            s["signed_response_fractions"]
            and min(s["signed_response_fractions"])
            >= targets["signed_response_fraction_min"]
            for s in active
        )
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "expected_cases": len(expected),
        "captured_cases": len(actual),
        "missing_cases": sorted(set(expected) - set(actual)),
        "failures": [name for name, value in conditions.items() if not value],
        "error_aggregation": "Every case/segment must meet error bounds; not pooled across commands.",
    }
