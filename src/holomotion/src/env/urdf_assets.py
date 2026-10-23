# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import os
import hashlib
import json
import xml.etree.ElementTree as ETree
from pathlib import Path


def materialize_urdf_with_absolute_mesh_paths(
    urdf_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Resolve relative URDF mesh paths without changing the source asset.

    Isaac Sim 5 does not consistently resolve plain ``meshes/foo.STL`` URIs
    against the source URDF directory.  Official Booster K1 assets use that
    form, so training materializes a rank-local derived URDF whose mesh paths
    are absolute.  Existing absolute and package/file URIs remain unchanged.
    """

    source = Path(urdf_path).expanduser().resolve()
    tree = ETree.parse(source)
    changed = False
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith(("package://", "file://")):
            continue
        mesh_path = Path(filename)
        if mesh_path.is_absolute():
            continue
        resolved = (source.parent / mesh_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"URDF mesh does not exist: {filename!r} resolved to "
                f"{resolved}"
            )
        mesh.set("filename", str(resolved))
        changed = True

    if not changed:
        return source

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{source.stem}_resolved_meshes.urdf"
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, destination)
    return destination


def xml_asset_fingerprint(asset_path: str | Path) -> dict:
    """Hash XML plus transitive mesh/include inputs, rejecting missing LFS data.

    This is a strict file identity, not a claim of geometric equivalence. Use
    the original packaged scene for contract evaluation; a different scene or
    conversion requires an explicitly reviewed new artifact contract.
    """
    source = Path(asset_path).resolve()
    files = {}
    visiting = set()

    def add_file(path):
        path = Path(path).resolve()
        key = os.path.relpath(path, source.parent)
        if key not in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                first = stream.read(256)
                if first.startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                ):
                    raise ValueError(f"Unhydrated LFS asset: {path}")
                digest.update(first)
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[key] = digest.hexdigest()
        return path

    def reference(base, text):
        if text.startswith("package://"):
            raise ValueError(
                "Resolve package:// mesh references before binding an asset contract"
            )
        return (base / text.removeprefix("file://")).resolve()

    def visit(path):
        path = add_file(path)
        if path in visiting:
            raise ValueError(f"Cyclic XML asset include: {path}")
        visiting.add(path)
        root = ETree.parse(path).getroot()
        compiler = root.find("compiler")
        mesh_dir = path.parent / (
            compiler.get("meshdir", "") if compiler is not None else ""
        )
        for node in root.iter("mesh"):
            filename = node.get("filename", node.get("file"))
            if filename:
                add_file(
                    reference(
                        path.parent if root.tag == "robot" else mesh_dir,
                        filename,
                    )
                )
        for node in root.iter("include"):
            visit(reference(path.parent, node.attrib["file"]))
        visiting.remove(path)

    visit(source)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "files": files,
    }
