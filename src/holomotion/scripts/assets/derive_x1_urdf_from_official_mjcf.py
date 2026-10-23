#!/usr/bin/env python3
"""Derive the X1 IsaacLab URDF from the pinned official full-body MJCF."""

from __future__ import annotations
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


# Isaac Sim's URDF importer treats '-' as an invalid USD identifier and used to
# rename this link only after building its internal parent map.  That leaves the
# generated articulation with dangling references (``map::at``).  Keep the
# official MuJoCo name in the source asset and apply this deterministic alias
# only to the IsaacLab URDF contract.
LINK_NAME_OVERRIDES = {"x1-body": "x1_body"}


def _link_name(source_name: str) -> str:
    return LINK_NAME_OVERRIDES.get(source_name, source_name)


def _vec(
    text: str | None, size: int, default: tuple[float, ...]
) -> list[float]:
    value = (
        list(default)
        if text is None
        else [float(item) for item in text.split()]
    )
    if len(value) != size:
        raise ValueError(f"expected {size} values, got {value}")
    return value


def _quat_to_rpy(wxyz: list[float]) -> tuple[float, float, float]:
    w, x, y, z = wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = (item / norm for item in (w, x, y, z))
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    del r11, r12, r21, r22
    if r20 <= -1.0 + 1.0e-10:
        pitch = math.pi / 2.0
        yaw = 0.0
        roll = math.atan2(r01, r02)
    elif r20 >= 1.0 - 1.0e-10:
        pitch = -math.pi / 2.0
        yaw = 0.0
        roll = math.atan2(-r01, -r02)
    else:
        pitch = math.asin(-r20)
        roll = math.atan2(2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))
        yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def _fmt(values) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _origin(parent: ET.Element, *, pos: str | None, quat: str | None) -> None:
    ET.SubElement(
        parent,
        "origin",
        xyz=_fmt(_vec(pos, 3, (0.0, 0.0, 0.0))),
        rpy=_fmt(_quat_to_rpy(_vec(quat, 4, (1.0, 0.0, 0.0, 0.0)))),
    )


def _add_inertial(link: ET.Element, body: ET.Element) -> None:
    source = body.find("inertial")
    if source is None:
        raise ValueError(f"body {body.get('name')} has no inertial")
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, pos=source.get("pos"), quat=source.get("quat"))
    ET.SubElement(inertial, "mass", value=source.get("mass", "0"))
    if source.get("fullinertia") is not None:
        ixx, iyy, izz, ixy, ixz, iyz = _vec(
            source.get("fullinertia"), 6, (0.0,) * 6
        )
    else:
        ixx, iyy, izz = _vec(source.get("diaginertia"), 3, (0.0,) * 3)
        ixy = ixz = iyz = 0.0
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.12g}",
        ixy=f"{ixy:.12g}",
        ixz=f"{ixz:.12g}",
        iyy=f"{iyy:.12g}",
        iyz=f"{iyz:.12g}",
        izz=f"{izz:.12g}",
    )


def _add_geometry(
    parent: ET.Element, geom: ET.Element, mesh_files: dict[str, str]
) -> None:
    wrapper = ET.SubElement(parent, "geometry")
    kind = geom.get("type", "sphere")
    if kind == "mesh":
        mesh_name = geom.get("mesh")
        if mesh_name not in mesh_files:
            raise ValueError(f"unknown mesh asset: {mesh_name}")
        ET.SubElement(
            wrapper, "mesh", filename=f"meshes/{mesh_files[mesh_name]}"
        )
    elif kind == "box":
        half = _vec(geom.get("size"), 3, (0.0, 0.0, 0.0))
        ET.SubElement(wrapper, "box", size=_fmt([2.0 * item for item in half]))
    elif kind == "sphere":
        ET.SubElement(
            wrapper,
            "sphere",
            radius=f"{_vec(geom.get('size'), 1, (0.0,))[0]:.12g}",
        )
    elif kind in {"capsule", "cylinder"}:
        size = _vec(geom.get("size"), 2, (0.0, 0.0))
        ET.SubElement(
            wrapper,
            "cylinder",
            radius=f"{size[0]:.12g}",
            length=f"{2.0 * size[1]:.12g}",
        )
    else:
        raise ValueError(f"unsupported geom type: {kind}")


def _add_geoms(
    link: ET.Element, body: ET.Element, mesh_files: dict[str, str]
) -> None:
    for geom in body.findall("geom"):
        tag = "collision" if geom.get("class", "") == "collision" else "visual"
        target = ET.SubElement(link, tag)
        _origin(target, pos=geom.get("pos"), quat=geom.get("quat"))
        _add_geometry(target, geom, mesh_files)


def derive(source: Path, output: Path) -> None:
    source_root = ET.parse(source).getroot()
    mesh_files = {
        mesh.get("name", ""): mesh.get("file", "")
        for mesh in source_root.findall(".//asset/mesh")
    }
    efforts = {}
    for actuator in source_root.findall(".//actuator/*"):
        joint = actuator.get("joint")
        lower, upper = _vec(actuator.get("ctrlrange"), 2, (-1.0e9, 1.0e9))
        if joint:
            efforts[joint] = max(abs(lower), abs(upper))
    worldbody = source_root.find("worldbody")
    root_body = None if worldbody is None else worldbody.find("body")
    if root_body is None:
        raise ValueError("MJCF has no root body")
    robot = ET.Element("robot", name="AgiBot_X1_29DoF")

    def visit(body: ET.Element, parent_name: str | None) -> None:
        source_name = body.get("name")
        if not source_name:
            raise ValueError("body without name")
        name = _link_name(source_name)
        link = ET.SubElement(robot, "link", name=name)
        _add_inertial(link, body)
        _add_geoms(link, body, mesh_files)
        joints = [
            joint
            for joint in body.findall("joint")
            if joint.get("type", "hinge") != "free"
        ]
        if parent_name is None:
            if joints:
                raise ValueError("root body cannot contain an actuated joint")
        else:
            if len(joints) != 1:
                raise ValueError(
                    f"body {name} must have exactly one hinge joint"
                )
            source_joint = joints[0]
            joint_name = source_joint.get("name")
            if not joint_name or joint_name not in efforts:
                raise ValueError(f"body {name} has unactuated/unnamed joint")
            joint = ET.SubElement(
                robot, "joint", name=joint_name, type="revolute"
            )
            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=name)
            _origin(joint, pos=body.get("pos"), quat=body.get("quat"))
            ET.SubElement(joint, "axis", xyz=source_joint.get("axis", "0 0 1"))
            lower, upper = _vec(
                source_joint.get("range"), 2, (-math.pi, math.pi)
            )
            ET.SubElement(
                joint,
                "limit",
                lower=f"{lower:.12g}",
                upper=f"{upper:.12g}",
                effort=f"{efforts[joint_name]:.12g}",
                velocity="1000",
            )
            ET.SubElement(
                joint,
                "dynamics",
                damping=source_joint.get("damping", "0"),
                friction=source_joint.get("frictionloss", "0"),
            )
        for child in body.findall("body"):
            visit(child, name)

    visit(root_body, None)
    ET.indent(robot, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    derive(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
