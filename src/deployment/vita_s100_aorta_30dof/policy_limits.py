"""Named bounds from the active model and verified training URDF."""

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np

from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_TO_COMMAND_SLOTS,
)
from deployment.vita_s100_aorta_30dof.robot_adapter import (
    VITA_CONTROLLER_JOINT_NAMES,
)


def policy_envelope(base, contracts, bundle, mode):
    if mode not in {"velocity", "motion"}:
        raise ValueError(f"unsupported limit source: {mode}")
    artifact = bundle.artifacts.get("motion_urdf")
    if artifact is None:
        raise ValueError("motion_urdf must be a hash-checked bundle artifact")
    root = ET.parse(artifact.path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if (
            limit is not None
            and "lower" in limit.attrib
            and "upper" in limit.attrib
        ):
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    if mode == "velocity":
        contract = contracts.velocity
        lower = np.fromstring(
            contract.metadata["joint_pos_limit_lower"], sep=","
        )
        upper = np.fromstring(
            contract.metadata["joint_pos_limit_upper"], sep=","
        )
        if (
            lower.shape != (len(contract.joint_names),)
            or upper.shape != lower.shape
        ):
            raise ValueError("Velocity joint limit metadata shape mismatch")
        limits.update(
            zip(
                contract.joint_names,
                zip(lower, upper, strict=True),
                strict=True,
            )
        )
    low, high = base.lower_rad.copy(), base.upper_rad.copy()
    for name, slot in zip(
        VITA_CONTROLLER_JOINT_NAMES, VITA_ACTION_TO_COMMAND_SLOTS, strict=True
    ):
        if name not in limits:
            raise ValueError(f"missing model joint bounds: {name}")
        low[slot], high[slot] = limits[name]
    if not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("policy joint bounds must be finite and ordered")
    return replace(base, lower_rad=low, upper_rad=high)
