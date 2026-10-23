"""Offline effort statistics in declared joint and coupled motor coordinates.

No controller mutation: saturation uses the recorded applied torque, not an
unclipped PD request. Invalid motor reconstructions retain explicit coverage.
"""

from __future__ import annotations
import re
from collections.abc import Mapping

import numpy as np


def _parameter(value, name):
    if isinstance(value, Mapping):
        matches = [
            v for pattern, v in value.items() if re.fullmatch(pattern, name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or absent effort limit for {name}")
        value = matches[0]
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Effort limits must be positive and finite")
    return value


def effort_statistics(applied_torque, limits, *, saturation_fraction=0.99):
    tau = np.asarray(applied_torque, dtype=np.float64)
    caps = np.asarray(limits, dtype=np.float64)
    if tau.ndim != 2 or len(tau) == 0 or caps.shape != (tau.shape[1],):
        raise ValueError("Effort trace/limit shapes differ")
    if (
        not np.isfinite(tau).all()
        or not np.isfinite(caps).all()
        or np.any(caps <= 0)
    ):
        raise ValueError("Nonfinite effort or invalid limits")
    if not 0 < saturation_fraction <= 1:
        raise ValueError("Saturation threshold must be in (0,1]")
    return {
        "sample_count": len(tau),
        "declared_limit_nm": caps.tolist(),
        "peak_absolute_nm": np.abs(tau).max(axis=0).tolist(),
        "saturation_fraction": (np.abs(tau) >= saturation_fraction * caps)
        .mean(axis=0)
        .tolist(),
        "peak_over_limit_nm": np.maximum(
            np.abs(tau).max(axis=0) - caps, 0
        ).tolist(),
    }


def summarize_actuator_effort(
    *,
    pre_physics_joint_pos,
    applied_torque,
    joint_names,
    actuators,
    saturation_fraction=0.99,
):
    """Use pre-physics positions for the controller's linkage Jacobian."""
    q, tau = (
        np.asarray(x, dtype=np.float64)
        for x in (pre_physics_joint_pos, applied_torque)
    )
    names = list(joint_names)
    if q.shape != tau.shape or tau.ndim != 2 or tau.shape[1] != len(names):
        raise ValueError("Position, effort and named joint shapes differ")
    if len(set(names)) != len(names) or not np.isfinite(q).all():
        raise ValueError("Invalid named position trace")
    groups = {
        name: value
        for name, value in actuators.items()
        if isinstance(value, Mapping) and "joint_names_expr" in value
    }
    caps, ownership = [], []
    for name in names:
        owners = [
            key
            for key, group in groups.items()
            if any(
                re.fullmatch(pattern, name)
                for pattern in group["joint_names_expr"]
            )
        ]
        if len(owners) != 1:
            raise ValueError(f"Ambiguous or absent actuator for {name}")
        group = groups[owners[0]]
        caps.append(_parameter(group["effort_limit_sim"], name))
        ownership.append(owners[0])
    result = {
        "joint_names": names,
        "joint_effort": effort_statistics(
            tau, caps, saturation_fraction=saturation_fraction
        ),
        "limit_scope": (
            "configured static simulation caps; not a torque-speed envelope"
        ),
        "motor_effort": {},
    }
    if actuators["actuator_type"] == "vita02a_parallel_ankle_delayed_pd":
        import torch

        from holomotion.src.utils.vita02a_parallel_ankle_math import (
            vita02a_ankle_jacobian,
        )

        for side in ("left", "right"):
            pitch, roll = (
                names.index(f"{side}_ankle_{axis}_joint")
                for axis in ("pitch", "roll")
            )
            if ownership[pitch] != ownership[roll]:
                raise ValueError("Coupled ankle crosses actuator groups")
            group = groups[ownership[pitch]]
            jac = vita02a_ankle_jacobian(
                torch.from_numpy(q[:, pitch]),
                torch.from_numpy(q[:, roll]),
                **{
                    key: float(group[key])
                    for key in ("lx", "ly", "z_top", "z_bot")
                },
            ).numpy()
            determinant = (
                jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            )
            valid = np.isfinite(jac).all(axis=(1, 2)) & (
                np.abs(determinant) > 1e-6
            )
            motors = np.linalg.solve(
                jac[valid].transpose(0, 2, 1),
                tau[valid][:, [roll, pitch], None],
            )[..., 0]
            result["motor_effort"][side] = {
                "jacobian_columns": ["roll", "pitch"],
                "valid_samples": int(valid.sum()),
                "invalid_samples": int((~valid).sum()),
                "coverage": float(valid.mean()),
                "statistics": effort_statistics(
                    motors,
                    [group["tau_motor_max"]] * 2,
                    saturation_fraction=saturation_fraction,
                )
                if valid.any()
                else None,
            }
    return result
