#!/usr/bin/env python3
"""Build a mesh-free PM01 MuJoCo model from the pinned official SDK XML."""

from __future__ import annotations
import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def _expand_includes(element: ET.Element, source_dir: Path) -> None:
    """Recursively inline MuJoCo include files without changing child order."""
    index = 0
    while index < len(element):
        child = element[index]
        if child.tag != "include":
            _expand_includes(child, source_dir)
            index += 1
            continue

        include_path = source_dir / child.attrib["file"]
        include_root = ET.parse(include_path).getroot()
        replacements = [copy.deepcopy(item) for item in include_root]
        element.remove(child)
        for offset, replacement in enumerate(replacements):
            element.insert(index + offset, replacement)
            _expand_includes(replacement, include_path.parent)
        index += len(replacements)


def build(source: Path, output: Path) -> None:
    """Generate a deterministic collision-only PM01 MJCF."""
    tree = ET.parse(source)
    root = tree.getroot()
    _expand_includes(root, source.parent)

    asset = root.find("asset")
    if asset is not None:
        asset.clear()

    # Meshes are visual-only in the official PM01 model.  Keep primitive
    # collision geoms, all inertials, joints, actuators, sensors and sites.
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            if child.get("type") == "mesh" or child.get("mesh") is not None:
                parent.remove(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("official PM01 MJCF has no worldbody")
    floor = ET.Element(
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "pos": "0 0 0",
            "rgba": "0.8 0.8 0.8 1",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.005 0.0001",
        },
    )
    worldbody.insert(0, floor)

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
