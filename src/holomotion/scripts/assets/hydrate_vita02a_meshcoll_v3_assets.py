#!/usr/bin/env python3
"""Hydrate and verify the SHA-pinned official Vita02A meshcoll v3 package.

The canonical control asset is the vendor's no-LinkerHand 30-DoF model.  The
source package is copied without physical changes.  Only ROS package mesh URLs
are made relative and the delivered no-hand scene wrapper is corrected to
include the no-hand MJCF.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET_RELATIVE_ROOT = Path("assets/robots/vbot/Vita02A/official_meshcoll_v3")
SOURCE_RELATIVE_ROOT = Path("vita02a_mesh_slided/vita02a")
PACKAGE_PREFIX = "package://vita02a/meshes/"
RELATIVE_PREFIX = "../meshes/"

SOURCE_SHA256 = {
    "scene_linkerhand_meshcoll.xml": (
        "fcda64d4e470367d58e5a5053fa0ac0ccb0d4cc87780ed46aa419271dc949d0e"
    ),
    "scene_meshcoll.xml": (
        "e17fee14a94ee69ec06d03289bea42687a49fa80f1a8394b30a1fdd3e7706a6d"
    ),
    "urdf/vita02a_linkerhand_meshcoll.urdf": (
        "91949239e91576810f0f67d536ebeb44ac921917c1f408780c45fc5c30493ca2"
    ),
    "urdf/vita02a_meshcoll.urdf": (
        "cead6e92e10a5a2e633cdb7a097cd4648079ef66c485711b2ebe29a11c9f0371"
    ),
    "vita02a_linkerhand_meshcoll.xml": (
        "2ad249ee0e45b65d819fc1bcc6218c8b4ca7a264fcd7f9fae67d92cbba18d227"
    ),
    "vita02a_meshcoll.xml": (
        "7610b81ba5f85ed2cc01fae037bf0cef37f0b9d5d4e843a92e5573fdeb3ad242"
    ),
}
HYDRATED_SHA256 = {
    "scene_linkerhand_meshcoll.xml": SOURCE_SHA256[
        "scene_linkerhand_meshcoll.xml"
    ],
    "scene_meshcoll.xml": (
        "853ea5a900bd122381a55444634f6d94308c7a432830c3bbe20487b0a7882deb"
    ),
    "urdf/vita02a_linkerhand_meshcoll.urdf": (
        "5660d18b1834c6f69aff8bc5cd9ac926c1578dac93404f630c2e800f5ba3277e"
    ),
    "urdf/vita02a_meshcoll.urdf": (
        "af2779d0a6ca6882b1fc82b5c8fdb5de0393dae46ced57b80e231c75b61cc2a4"
    ),
    "vita02a_linkerhand_meshcoll.xml": SOURCE_SHA256[
        "vita02a_linkerhand_meshcoll.xml"
    ],
    "vita02a_meshcoll.xml": SOURCE_SHA256["vita02a_meshcoll.xml"],
}
MESH_FILES = 99
MESH_BYTES = 76_166_816
MESH_TREE_SHA256 = (
    "5d348976025f96ae353d41527637b7f47383274c236088183c78ea41ac2dd0a3"
)
CANONICAL_URDF = "urdf/vita02a_meshcoll.urdf"
CANONICAL_SCENE = "scene_meshcoll.xml"
CANONICAL_LINKS = 31
CANONICAL_MOVING_JOINTS = 30
CANONICAL_MASS_KG = 54.44955488


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_tree(root: Path) -> tuple[int, int, str]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    size = 0
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
        size += path.stat().st_size
    return len(paths), size, digest.hexdigest()


def _verify_hashes(root: Path, expected: dict[str, str], label: str) -> None:
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing Vita02A {label} file: {path}")
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"Vita02A {label} digest mismatch for {path}: "
                f"expected {digest}, got {actual}"
            )


def _verify_meshes(root: Path) -> dict[str, object]:
    actual = _mesh_tree(root / "meshes")
    expected = (MESH_FILES, MESH_BYTES, MESH_TREE_SHA256)
    if actual != expected:
        raise ValueError(
            f"Vita02A mesh tree mismatch: expected {expected}, got {actual}"
        )
    return {"files": actual[0], "bytes": actual[1], "tree_sha256": actual[2]}


def _verify_resolved_meshes(target: Path) -> None:
    for name in ("vita02a_meshcoll.urdf", "vita02a_linkerhand_meshcoll.urdf"):
        path = target / "urdf" / name
        root = ET.parse(path).getroot()
        references = [mesh.get("filename") for mesh in root.findall(".//mesh")]
        if not references or any(
            reference is None or reference.startswith("package://")
            for reference in references
        ):
            raise ValueError(f"Unresolved mesh reference remains in {path}")
        for reference in references:
            assert reference is not None
            if not (path.parent / reference).resolve().is_file():
                raise FileNotFoundError(
                    f"Missing URDF mesh {reference!r} referenced by {path}"
                )


def _verify_canonical_urdf(target: Path) -> None:
    root = ET.parse(target / CANONICAL_URDF).getroot()
    links = root.findall("link")
    moving = [
        joint
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    ]
    if len(links) != CANONICAL_LINKS or len(moving) != CANONICAL_MOVING_JOINTS:
        raise ValueError(
            "Unexpected Vita02A no-hand topology: "
            f"{len(links)} links/{len(moving)} moving joints"
        )
    if any(joint.get("type") != "revolute" for joint in moving):
        raise ValueError(
            "Every canonical Vita02A moving joint must be revolute"
        )
    for joint in moving:
        dynamics = joint.find("dynamics")
        if dynamics is None or float(dynamics.get("friction", "nan")) != 0.0:
            raise ValueError(
                f"Canonical joint friction is not zero: {joint.get('name')}"
            )
    mass = sum(
        float(value.get("value"))
        for value in root.findall("link/inertial/mass")
    )
    if abs(mass - CANONICAL_MASS_KG) > 1.0e-8:
        raise ValueError(
            "Unexpected Vita02A no-hand mass: "
            f"expected {CANONICAL_MASS_KG}, got {mass}"
        )
    scene = ET.parse(target / CANONICAL_SCENE).getroot()
    include = scene.find("include")
    if include is None or include.get("file") != "vita02a_meshcoll.xml":
        raise ValueError("The no-hand scene must include vita02a_meshcoll.xml")


def _receipt(meshes: dict[str, object]) -> dict[str, object]:
    return {
        "asset_variant": "official_meshcoll_v3_slided_wrist_nohand_30dof",
        "source": {
            "relative_root": SOURCE_RELATIVE_ROOT.as_posix(),
            "files_sha256": SOURCE_SHA256,
            "meshes": meshes,
        },
        "local_transformations": [
            {
                "files": [
                    "urdf/vita02a_linkerhand_meshcoll.urdf",
                    CANONICAL_URDF,
                ],
                "replacement": f"{PACKAGE_PREFIX} -> {RELATIVE_PREFIX}",
                "reason": "Resolve meshes without ROS package registration.",
            },
            {
                "file": CANONICAL_SCENE,
                "original": "vita02a_linkerhand_meshcoll.xml",
                "replacement": "vita02a_meshcoll.xml",
                "reason": (
                    "Correct the delivered no-hand scene wrapper include."
                ),
            },
        ],
        "hydrated_files_sha256": HYDRATED_SHA256,
        "canonical_control_asset": {
            "urdf": CANONICAL_URDF,
            "scene": CANONICAL_SCENE,
            "links": CANONICAL_LINKS,
            "moving_joints": CANONICAL_MOVING_JOINTS,
            "mass_kg": CANONICAL_MASS_KG,
            "joint_friction": 0.0,
        },
        "redistribution_status": "not_verified",
    }


def _rewrite_urdf_mesh_paths(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if PACKAGE_PREFIX not in text:
        raise ValueError(f"No Vita02A package mesh paths found in {path}")
    path.write_text(
        text.replace(PACKAGE_PREFIX, RELATIVE_PREFIX), encoding="utf-8"
    )


def _fix_nohand_scene(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = 'file="vita02a_linkerhand_meshcoll.xml"'
    new = 'file="vita02a_meshcoll.xml"'
    if text.count(old) != 1:
        raise ValueError(f"Unexpected no-hand scene include in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def _verify_target(target: Path) -> dict[str, object]:
    _verify_hashes(target, HYDRATED_SHA256, "hydrated")
    meshes = _verify_meshes(target)
    _verify_resolved_meshes(target)
    _verify_canonical_urdf(target)
    return _receipt(meshes)


def _hydrate(repo_root: Path, vbot_lab: Path, *, force: bool) -> Path:
    source = vbot_lab.resolve() / SOURCE_RELATIVE_ROOT
    _verify_hashes(source, SOURCE_SHA256, "official input")
    _verify_meshes(source)
    target = repo_root / ASSET_RELATIVE_ROOT
    if target.exists():
        if not force:
            raise FileExistsError(
                f"Vita02A target exists: {target}; "
                "use --verify-only or --force"
            )
        shutil.rmtree(target)
    shutil.copytree(source, target)
    for name in ("vita02a_meshcoll.urdf", "vita02a_linkerhand_meshcoll.urdf"):
        _rewrite_urdf_mesh_paths(target / "urdf" / name)
    _fix_nohand_scene(target / CANONICAL_SCENE)
    return target


def _assert_receipt_contains(
    recorded, expected, path: str = "receipt"
) -> None:
    """Require deterministic fields while allowing validation evidence."""
    if isinstance(expected, dict):
        if not isinstance(recorded, dict):
            raise ValueError(f"{path} must be a mapping")
        for key, value in expected.items():
            if key not in recorded:
                raise ValueError(f"Missing {path}.{key}")
            _assert_receipt_contains(recorded[key], value, f"{path}.{key}")
        return
    if recorded != expected:
        raise ValueError(
            f"{path} differs: expected {expected!r}, got {recorded!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--vbot-lab", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--refresh-receipt", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.verify_only and args.refresh_receipt:
        parser.error(
            "--verify-only and --refresh-receipt are mutually exclusive"
        )

    repo_root = args.repo_root.resolve()
    target = repo_root / ASSET_RELATIVE_ROOT
    if not args.verify_only and not args.refresh_receipt:
        if args.vbot_lab is None:
            parser.error("hydration requires --vbot-lab")
        target = _hydrate(repo_root, args.vbot_lab, force=args.force)

    receipt = _verify_target(target)
    receipt_path = target / "ASSET_HYDRATION_RECEIPT.json"
    if args.verify_only:
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing Vita02A hydration receipt: {receipt_path}"
            )
        recorded = json.loads(receipt_path.read_text(encoding="utf-8"))
        _assert_receipt_contains(recorded, receipt)
    else:
        if args.refresh_receipt and receipt_path.is_file():
            recorded = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt = {**recorded, **receipt}
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
