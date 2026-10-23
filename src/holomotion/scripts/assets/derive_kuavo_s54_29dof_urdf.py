#!/usr/bin/env python3
"""Derive the 29-DoF Kuavo S54 URDF from the pinned official train URDF."""

from __future__ import annotations
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

HEAD_JOINTS = ("zhead_1_joint", "zhead_2_joint")


def derive(source: Path, output: Path) -> None:
    """Activate only the two official head joints; preserve all other data."""
    tree = ET.parse(source)
    root = tree.getroot()
    changed: list[str] = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in HEAD_JOINTS:
            continue
        if joint.get("type") != "fixed":
            raise ValueError(f"expected official {name} to be fixed")
        if joint.find("axis") is None or joint.find("limit") is None:
            raise ValueError(f"official {name} is missing axis or limit")
        joint.set("type", "revolute")
        changed.append(name)
    if tuple(changed) != HEAD_JOINTS:
        raise ValueError(f"expected head joints {HEAD_JOINTS}, got {changed}")

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    derive(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
