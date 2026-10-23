"""Config-driven robot adapter for the scale-invariant semantic skeleton."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from holomotion.src.motion_tracking.semantic_skeleton import (
    FOOT_INDICES,
    SEMANTIC_NODE_NAMES,
    TORSO_INDEX,
    WRIST_INDICES,
)
from holomotion.src.utils import torch_utils


@dataclass(frozen=True)
class SemanticSkeletonState:
    """Robot-independent tensors consumed by scale-invariant rewards."""

    position_w: torch.Tensor
    pelvis_rotation_wxyz: torch.Tensor
    torso_rotation_wxyz: torch.Tensor
    endpoint_rotation_wxyz: torch.Tensor
    head_rotation_wxyz: torch.Tensor | None
    root_linear_velocity_w: torch.Tensor
    root_angular_velocity_w: torch.Tensor
    nominal_scale: torch.Tensor


@dataclass(frozen=True)
class SemanticNodeSiteSpec:
    """Config-only semantic landmark expressed from robot link origins."""

    kind: str
    link_names: tuple[str, ...]
    local_offset: tuple[float, float, float]
    orientation_link_name: str

    @property
    def primary_link_name(self) -> str:
        return self.link_names[0]


@dataclass(frozen=True)
class RobotSkeletonAdapterSpec:
    """Static morphology metadata, derived from a robot's neutral pose."""

    node_sites: tuple[SemanticNodeSiteSpec, ...]
    nominal_scale: float
    pelvis_semantic_offset_wxyz: tuple[float, float, float, float]
    torso_semantic_offset_wxyz: tuple[float, float, float, float]
    endpoint_semantic_offset_wxyz: tuple[
        tuple[float, float, float, float], ...
    ]
    head_orientation_link_name: str | None
    head_semantic_offset_wxyz: tuple[float, float, float, float] | None

    @property
    def node_link_names(self) -> tuple[str, ...]:
        """Primary link per semantic node, retained for legacy callers."""

        return tuple(site.primary_link_name for site in self.node_sites)

    @property
    def required_link_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for site in self.node_sites:
            names.extend(site.link_names)
            names.append(site.orientation_link_name)
        if self.head_orientation_link_name is not None:
            names.append(self.head_orientation_link_name)
        return tuple(dict.fromkeys(names))

    @property
    def foot_link_names(self) -> tuple[str, str]:
        return (
            self.node_sites[FOOT_INDICES[0]].primary_link_name,
            self.node_sites[FOOT_INDICES[1]].primary_link_name,
        )

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "RobotSkeletonAdapterSpec":
        try:
            nominal_scale = float(config["nominal_scale"])
            pelvis_offset_raw = config["pelvis_semantic_offset_wxyz"]
            torso_offset_raw = config["torso_semantic_offset_wxyz"]
            endpoint_offset_raw = config["endpoint_semantic_offset_wxyz"]
        except KeyError as error:
            raise ValueError(
                f"semantic_skeleton config is missing {error.args[0]!r}"
            ) from error

        node_sites = _parse_node_sites(config)
        if (
            not torch.isfinite(torch.tensor(nominal_scale))
            or nominal_scale <= 0
        ):
            raise ValueError(
                "semantic_skeleton.nominal_scale must be finite and positive"
            )

        pelvis_offset = _validated_quaternion(
            pelvis_offset_raw,
            name="pelvis_semantic_offset_wxyz",
        )
        torso_offset = _validated_quaternion(
            torso_offset_raw,
            name="torso_semantic_offset_wxyz",
        )
        endpoint_offset = tuple(
            _validated_quaternion(
                value,
                name=f"endpoint_semantic_offset_wxyz[{index}]",
            )
            for index, value in enumerate(endpoint_offset_raw)
        )
        if len(endpoint_offset) != len(FOOT_INDICES + WRIST_INDICES):
            raise ValueError(
                "endpoint_semantic_offset_wxyz must contain left/right feet "
                "and left/right wrists in that order"
            )
        head_link_raw = config.get("head_orientation_link")
        head_link = None if head_link_raw is None else str(head_link_raw)
        head_offset_raw = config.get("head_semantic_offset_wxyz")
        if (head_link is None) != (head_offset_raw is None):
            raise ValueError(
                "head_orientation_link and head_semantic_offset_wxyz must "
                "either both be set or both be omitted"
            )
        head_offset = (
            None
            if head_offset_raw is None
            else _validated_quaternion(
                head_offset_raw,
                name="head_semantic_offset_wxyz",
            )
        )
        return cls(
            node_sites=node_sites,
            nominal_scale=nominal_scale,
            pelvis_semantic_offset_wxyz=pelvis_offset,
            torso_semantic_offset_wxyz=torso_offset,
            endpoint_semantic_offset_wxyz=endpoint_offset,
            head_orientation_link_name=head_link,
            head_semantic_offset_wxyz=head_offset,
        )


def _validated_offset(
    value: Sequence[float], *, name: str
) -> tuple[float, float, float]:
    offset = tuple(float(component) for component in value)
    if len(offset) != 3 or not torch.isfinite(torch.tensor(offset)).all():
        raise ValueError(f"{name} must contain three finite values")
    return offset


def _parse_node_sites(
    config: Mapping[str, Any],
) -> tuple[SemanticNodeSiteSpec, ...]:
    raw = config.get("node_sites", config.get("node_links"))
    if raw is None:
        raise ValueError(
            "semantic_skeleton config is missing 'node_sites'/'node_links'"
        )
    if isinstance(raw, Mapping):
        missing = [name for name in SEMANTIC_NODE_NAMES if name not in raw]
        if missing:
            raise ValueError(
                "semantic_skeleton landmark mapping is missing nodes "
                f"{missing}"
            )
        values = [raw[name] for name in SEMANTIC_NODE_NAMES]
    else:
        values = list(raw)
        if len(values) != len(SEMANTIC_NODE_NAMES):
            raise ValueError(
                "semantic_skeleton landmark mapping must contain exactly "
                f"{len(SEMANTIC_NODE_NAMES)} entries"
            )

    sites = []
    for name, value in zip(SEMANTIC_NODE_NAMES, values, strict=True):
        if not isinstance(value, Mapping):
            link = str(value)
            sites.append(
                SemanticNodeSiteSpec(
                    kind="link_origin",
                    link_names=(link,),
                    local_offset=(0.0, 0.0, 0.0),
                    orientation_link_name=link,
                )
            )
            continue
        kind = str(value.get("type", "link_origin"))
        if kind in {"link_origin", "rigid_site"}:
            link_names = (str(value["link"]),)
        elif kind == "midpoint":
            link_names = tuple(str(link) for link in value.get("links", ()))
            if len(link_names) < 2:
                raise ValueError(
                    f"node_sites.{name}.links must contain at least two links"
                )
        else:
            raise ValueError(
                f"node_sites.{name}.type has unsupported value {kind!r}"
            )
        offset = _validated_offset(
            value.get("offset", (0.0, 0.0, 0.0)),
            name=f"node_sites.{name}.offset",
        )
        if kind == "midpoint" and offset != (0.0, 0.0, 0.0):
            raise ValueError("midpoint semantic sites do not accept an offset")
        orientation_link = str(value.get("orientation_link", link_names[0]))
        sites.append(
            SemanticNodeSiteSpec(
                kind=kind,
                link_names=link_names,
                local_offset=offset,
                orientation_link_name=orientation_link,
            )
        )
    return tuple(sites)


def _validated_quaternion(
    value: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float, float]:
    quaternion = tuple(float(component) for component in value)
    if len(quaternion) != 4:
        raise ValueError(f"{name} must contain four WXYZ values")
    tensor = torch.tensor(quaternion, dtype=torch.float64)
    norm = torch.linalg.vector_norm(tensor)
    if not torch.isfinite(tensor).all() or float(norm) <= 1.0e-8:
        raise ValueError(f"{name} must be a finite non-zero quaternion")
    normalized = tensor / norm
    return tuple(float(component) for component in normalized)


def _constant_quaternion(
    value: tuple[float, float, float, float],
    like: torch.Tensor,
) -> torch.Tensor:
    quaternion = torch.tensor(value, dtype=like.dtype, device=like.device)
    return quaternion.expand(like.shape[0], -1)


def build_robot_skeleton_state(
    *,
    spec: RobotSkeletonAdapterSpec,
    body_name_to_index: Mapping[str, int],
    body_position_w: torch.Tensor,
    body_rotation_wxyz: torch.Tensor,
    root_linear_velocity_w: torch.Tensor,
    root_angular_velocity_w: torch.Tensor,
) -> SemanticSkeletonState:
    """Extract one semantic skeleton state without GMR virtual geometry."""

    if body_position_w.ndim != 3 or body_position_w.shape[-1] != 3:
        raise ValueError(
            "body_position_w must be [B,N,3], got "
            f"{tuple(body_position_w.shape)}"
        )
    if body_rotation_wxyz.shape != body_position_w.shape[:-1] + (4,):
        raise ValueError(
            "body_rotation_wxyz shape must match body positions, got "
            f"{tuple(body_rotation_wxyz.shape)}"
        )
    if root_linear_velocity_w.shape != (body_position_w.shape[0], 3):
        raise ValueError("root_linear_velocity_w must be [B,3]")
    if root_angular_velocity_w.shape != (body_position_w.shape[0], 3):
        raise ValueError("root_angular_velocity_w must be [B,3]")

    try:
        required_indices = {
            name: body_name_to_index[name] for name in spec.required_link_names
        }
    except KeyError as error:
        raise ValueError(
            f"Robot skeleton link {error.args[0]!r} is unavailable"
        ) from error
    positions = []
    rotations = []
    for site in spec.node_sites:
        site_indices = torch.tensor(
            [required_indices[name] for name in site.link_names],
            dtype=torch.long,
            device=body_position_w.device,
        )
        selected = body_position_w.index_select(1, site_indices)
        if site.kind == "midpoint":
            position = selected.mean(dim=1)
        else:
            position = selected[:, 0]
            if site.kind == "rigid_site":
                link_rotation = body_rotation_wxyz[
                    :, required_indices[site.link_names[0]]
                ]
                offset = torch.tensor(
                    site.local_offset,
                    dtype=position.dtype,
                    device=position.device,
                ).expand(position.shape[0], -1)
                position = position + torch_utils.quat_apply(
                    link_rotation, offset
                )
        positions.append(position)
        rotations.append(
            body_rotation_wxyz[:, required_indices[site.orientation_link_name]]
        )
    position = torch.stack(positions, dim=1)
    rotation = torch.stack(rotations, dim=1)

    pelvis_offset = _constant_quaternion(
        spec.pelvis_semantic_offset_wxyz,
        rotation[:, 0],
    )
    torso_offset = _constant_quaternion(
        spec.torso_semantic_offset_wxyz,
        rotation[:, TORSO_INDEX],
    )
    pelvis_rotation = torch_utils.quat_mul(
        rotation[:, 0],
        pelvis_offset,
    )
    torso_rotation = torch_utils.quat_mul(
        rotation[:, TORSO_INDEX],
        torso_offset,
    )

    endpoint_indices = torch.tensor(
        FOOT_INDICES + WRIST_INDICES,
        dtype=torch.long,
        device=position.device,
    )
    endpoint_rotation = rotation.index_select(1, endpoint_indices)
    endpoint_offset = torch.tensor(
        spec.endpoint_semantic_offset_wxyz,
        dtype=rotation.dtype,
        device=rotation.device,
    )
    endpoint_offset = endpoint_offset.unsqueeze(0).expand(
        endpoint_rotation.shape[0],
        -1,
        -1,
    )
    endpoint_semantic_rotation = torch_utils.quat_mul(
        endpoint_rotation,
        endpoint_offset,
    )
    head_rotation = None
    if spec.head_orientation_link_name is not None:
        if spec.head_semantic_offset_wxyz is None:
            raise AssertionError("Head semantic rotation offset is missing")
        head_link_rotation = body_rotation_wxyz[
            :, required_indices[spec.head_orientation_link_name]
        ]
        head_rotation = torch_utils.quat_mul(
            head_link_rotation,
            _constant_quaternion(
                spec.head_semantic_offset_wxyz,
                head_link_rotation,
            ),
        )

    scale = torch.full(
        (position.shape[0], 1),
        spec.nominal_scale,
        dtype=position.dtype,
        device=position.device,
    )
    return SemanticSkeletonState(
        position_w=position,
        pelvis_rotation_wxyz=pelvis_rotation,
        torso_rotation_wxyz=torso_rotation,
        endpoint_rotation_wxyz=endpoint_semantic_rotation,
        head_rotation_wxyz=head_rotation,
        root_linear_velocity_w=root_linear_velocity_w,
        root_angular_velocity_w=root_angular_velocity_w,
        nominal_scale=scale,
    )


__all__ = [
    "RobotSkeletonAdapterSpec",
    "SemanticNodeSiteSpec",
    "SemanticSkeletonState",
    "build_robot_skeleton_state",
]
