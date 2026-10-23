"""Shared, scale-invariant semantic skeleton definitions.

This module is intentionally independent of GMR, robot IK, and IsaacLab.  It
defines the smallest common humanoid graph used by the scale-invariant reward
and provides pure tensor geometry helpers that can be tested offline.
"""

from __future__ import annotations

from typing import Final

import torch

SEMANTIC_SKELETON_VERSION: Final[str] = "semantic_skeleton_si_v1"

SEMANTIC_NODE_NAMES: Final[tuple[str, ...]] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "torso",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

NODE_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(SEMANTIC_NODE_NAMES)
}

# A 14-node tree with 13 edges.  The order is stable and part of the cache
# contract; reward aggregation is group-balanced rather than edge-count
# weighted.
SEMANTIC_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (NODE_INDEX["pelvis"], NODE_INDEX["left_hip"]),
    (NODE_INDEX["left_hip"], NODE_INDEX["left_knee"]),
    (NODE_INDEX["left_knee"], NODE_INDEX["left_foot"]),
    (NODE_INDEX["pelvis"], NODE_INDEX["right_hip"]),
    (NODE_INDEX["right_hip"], NODE_INDEX["right_knee"]),
    (NODE_INDEX["right_knee"], NODE_INDEX["right_foot"]),
    (NODE_INDEX["pelvis"], NODE_INDEX["torso"]),
    (NODE_INDEX["torso"], NODE_INDEX["left_shoulder"]),
    (NODE_INDEX["left_shoulder"], NODE_INDEX["left_elbow"]),
    (NODE_INDEX["left_elbow"], NODE_INDEX["left_wrist"]),
    (NODE_INDEX["torso"], NODE_INDEX["right_shoulder"]),
    (NODE_INDEX["right_shoulder"], NODE_INDEX["right_elbow"]),
    (NODE_INDEX["right_elbow"], NODE_INDEX["right_wrist"]),
)

STRUCTURE_GROUP_NAMES: Final[tuple[str, ...]] = (
    "torso",
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
)

EDGE_GROUPS: Final[tuple[tuple[int, ...], ...]] = (
    (6,),
    (0, 1, 2),
    (3, 4, 5),
    (7, 8, 9),
    (10, 11, 12),
)

POSITION_GROUPS: Final[tuple[tuple[int, ...], ...]] = (
    (NODE_INDEX["torso"],),
    (
        NODE_INDEX["left_hip"],
        NODE_INDEX["left_knee"],
        NODE_INDEX["left_foot"],
    ),
    (
        NODE_INDEX["right_hip"],
        NODE_INDEX["right_knee"],
        NODE_INDEX["right_foot"],
    ),
    (
        NODE_INDEX["left_shoulder"],
        NODE_INDEX["left_elbow"],
        NODE_INDEX["left_wrist"],
    ),
    (
        NODE_INDEX["right_shoulder"],
        NODE_INDEX["right_elbow"],
        NODE_INDEX["right_wrist"],
    ),
)

# (proximal edge, distal edge) for left/right knee and left/right elbow.
LOCAL_CHAIN_EDGE_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (4, 5),
    (8, 9),
    (11, 12),
)

LOCAL_CHAIN_NAMES: Final[tuple[str, ...]] = (
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
)

PELVIS_INDEX: Final[int] = NODE_INDEX["pelvis"]
TORSO_INDEX: Final[int] = NODE_INDEX["torso"]
FOOT_INDICES: Final[tuple[int, int]] = (
    NODE_INDEX["left_foot"],
    NODE_INDEX["right_foot"],
)
WRIST_INDICES: Final[tuple[int, int]] = (
    NODE_INDEX["left_wrist"],
    NODE_INDEX["right_wrist"],
)


def safe_normalize(
    vector: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize vectors and return a validity mask for non-degenerate rows."""

    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    valid = norm[..., 0] > eps
    normalized = vector / norm.clamp_min(eps)
    return normalized, valid


def _fallback_axis(
    reference: torch.Tensor,
    values: tuple[float, float, float],
) -> torch.Tensor:
    axis = torch.tensor(values, dtype=reference.dtype, device=reference.device)
    return axis.expand(reference.shape)


def canonical_frame_from_lateral_up(
    lateral: torch.Tensor,
    up_proxy: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Construct a right-handed frame with columns forward/left/up.

    Degenerate samples receive deterministic axes.  The fallback keeps the
    reward finite; diagnostics and tests still expose invalid source geometry.
    """

    left, left_valid = safe_normalize(lateral, eps=eps)
    left = torch.where(
        left_valid[..., None],
        left,
        _fallback_axis(left, (0.0, 1.0, 0.0)),
    )

    up_orthogonal = up_proxy - left * torch.sum(
        left * up_proxy, dim=-1, keepdim=True
    )
    up, up_valid = safe_normalize(up_orthogonal, eps=eps)
    fallback_up = _fallback_axis(up, (0.0, 0.0, 1.0))
    fallback_up = fallback_up - left * torch.sum(
        left * fallback_up, dim=-1, keepdim=True
    )
    fallback_up, _ = safe_normalize(fallback_up, eps=eps)
    up = torch.where(up_valid[..., None], up, fallback_up)

    forward, _ = safe_normalize(torch.cross(left, up, dim=-1), eps=eps)
    # Recompute up to make the returned matrix orthonormal after fallbacks.
    up, _ = safe_normalize(torch.cross(forward, left, dim=-1), eps=eps)
    return torch.stack((forward, left, up), dim=-1)


def canonical_pelvis_frame(position: torch.Tensor) -> torch.Tensor:
    """Return the geometry-derived pelvis frame for ``[...,14,3]`` positions."""

    if position.shape[-2:] != (len(SEMANTIC_NODE_NAMES), 3):
        raise ValueError(
            f"position must end in [14,3], got {tuple(position.shape)}"
        )
    lateral = (
        position[..., NODE_INDEX["left_hip"], :]
        - position[..., NODE_INDEX["right_hip"], :]
    )
    up_proxy = position[..., TORSO_INDEX, :] - position[..., PELVIS_INDEX, :]
    return canonical_frame_from_lateral_up(lateral, up_proxy)


def canonical_torso_frame(position: torch.Tensor) -> torch.Tensor:
    """Return a torso frame derived from shoulder width and pelvis-to-torso."""

    if position.shape[-2:] != (len(SEMANTIC_NODE_NAMES), 3):
        raise ValueError(
            f"position must end in [14,3], got {tuple(position.shape)}"
        )
    lateral = (
        position[..., NODE_INDEX["left_shoulder"], :]
        - position[..., NODE_INDEX["right_shoulder"], :]
    )
    up_proxy = position[..., TORSO_INDEX, :] - position[..., PELVIS_INDEX, :]
    return canonical_frame_from_lateral_up(lateral, up_proxy)


def pelvis_local_positions(
    position: torch.Tensor,
    pelvis_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Express semantic positions in each entity's full canonical pelvis frame."""

    frame = (
        canonical_pelvis_frame(position)
        if pelvis_frame is None
        else pelvis_frame
    )
    relative = position - position[..., PELVIS_INDEX : PELVIS_INDEX + 1, :]
    return torch.matmul(
        frame.transpose(-1, -2).unsqueeze(-3),
        relative.unsqueeze(-1),
    ).squeeze(-1)


def nominal_leg_scale(position: torch.Tensor) -> torch.Tensor:
    """Compute the mean left/right hip-knee-foot chain length.

    This function is intended for rest morphology.  Applying it to an
    articulated robot whose semantic landmarks span multiple joints would make
    the scale pose-dependent.
    """

    left = torch.linalg.vector_norm(
        position[..., NODE_INDEX["left_knee"], :]
        - position[..., NODE_INDEX["left_hip"], :],
        dim=-1,
    ) + torch.linalg.vector_norm(
        position[..., NODE_INDEX["left_foot"], :]
        - position[..., NODE_INDEX["left_knee"], :],
        dim=-1,
    )
    right = torch.linalg.vector_norm(
        position[..., NODE_INDEX["right_knee"], :]
        - position[..., NODE_INDEX["right_hip"], :],
        dim=-1,
    ) + torch.linalg.vector_norm(
        position[..., NODE_INDEX["right_foot"], :]
        - position[..., NODE_INDEX["right_knee"], :],
        dim=-1,
    )
    return 0.5 * (left + right)


def unit_edge_directions(
    position: torch.Tensor,
    pelvis_frame: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical-pelvis-frame unit directions and valid-edge masks."""

    local = pelvis_local_positions(position, pelvis_frame)
    parents = torch.tensor(
        [edge[0] for edge in SEMANTIC_EDGES],
        dtype=torch.long,
        device=position.device,
    )
    children = torch.tensor(
        [edge[1] for edge in SEMANTIC_EDGES],
        dtype=torch.long,
        device=position.device,
    )
    edge_vectors = local.index_select(-2, children) - local.index_select(
        -2, parents
    )
    return safe_normalize(edge_vectors)


def local_bend_plane_features(
    unit_directions: torch.Tensor,
    edge_valid: torch.Tensor,
    *,
    plane_eps: float = 5.0e-2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bend cosine, limb-plane normals, and stable-plane masks."""

    proximal = torch.tensor(
        [pair[0] for pair in LOCAL_CHAIN_EDGE_PAIRS],
        dtype=torch.long,
        device=unit_directions.device,
    )
    distal = torch.tensor(
        [pair[1] for pair in LOCAL_CHAIN_EDGE_PAIRS],
        dtype=torch.long,
        device=unit_directions.device,
    )
    first = unit_directions.index_select(-2, proximal)
    second = unit_directions.index_select(-2, distal)
    bend_cosine = torch.sum(first * second, dim=-1)
    plane = torch.cross(first, second, dim=-1)
    plane_normal, plane_valid = safe_normalize(plane, eps=plane_eps)
    pair_valid = edge_valid.index_select(
        -1, proximal
    ) & edge_valid.index_select(-1, distal)
    return bend_cosine, plane_normal, plane_valid & pair_valid


__all__ = [
    "EDGE_GROUPS",
    "FOOT_INDICES",
    "LOCAL_CHAIN_EDGE_PAIRS",
    "LOCAL_CHAIN_NAMES",
    "NODE_INDEX",
    "PELVIS_INDEX",
    "POSITION_GROUPS",
    "SEMANTIC_EDGES",
    "SEMANTIC_NODE_NAMES",
    "SEMANTIC_SKELETON_VERSION",
    "STRUCTURE_GROUP_NAMES",
    "TORSO_INDEX",
    "WRIST_INDICES",
    "canonical_frame_from_lateral_up",
    "canonical_pelvis_frame",
    "canonical_torso_frame",
    "local_bend_plane_features",
    "nominal_leg_scale",
    "pelvis_local_positions",
    "safe_normalize",
    "unit_edge_directions",
]
