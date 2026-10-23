#!/usr/bin/env python3
"""Derive the 30-DoF Vita02A body asset from the official LinkerHand model.

The official source stays untouched.  The derived control view keeps all
LinkerHand bodies, inertials, and visual meshes at their zero-angle pose while
welding the 22 finger joints.  Its official hand collision meshes are retained;
shoulder and wrist collision proxies are copied from the locked Vita02A shoes
asset so self-collision can be enabled without nominal wrist false contacts.
The 28 body/wrist joints and two neck joints remain active.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_RELATIVE_ROOT = Path("assets/robots/vbot/Vita02A/official_meshcoll_v2")

SOURCE_URDF_NAME = "urdf/vita02a_linkerhand_meshcoll.urdf"
SOURCE_MJCF_NAME = "vita02a_linkerhand_meshcoll.xml"
SOURCE_SCENE_NAME = "scene_linkerhand_meshcoll.xml"
COLLISION_PROXY_URDF_NAME = "../28dof/urdf/vita02a_shoes_28dof.urdf"
OUTPUT_URDF_NAME = "urdf/vita02a_linkerhand_meshcoll_30dof.urdf"
OUTPUT_MJCF_NAME = "vita02a_linkerhand_meshcoll_30dof.xml"
OUTPUT_SCENE_NAME = "scene_linkerhand_meshcoll_30dof.xml"

SOURCE_SHA256 = {
    SOURCE_URDF_NAME: (
        "a8798d82bc1d94a16fae9092c42e30ebe9d98e2dd014fb7fd5acf8aefb010f25"
    ),
    SOURCE_MJCF_NAME: (
        "7ea5c4e0990cc74db9c99e587c7b89fdf34d18d10d100032137ca819a0285518"
    ),
    SOURCE_SCENE_NAME: (
        "fcda64d4e470367d58e5a5053fa0ac0ccb0d4cc87780ed46aa419271dc949d0e"
    ),
}
COLLISION_PROXY_URDF_SHA256 = (
    "c1d8037af929f15102041d6d33ab58ca728a4d7d5a49e66b66c3b429fa8d285b"
)

# The official mesh-collision asset intentionally omits the first two shoulder
# links.  Its wrist convex hulls also overlap at q=0 once robot self-collision
# is enabled.  Reuse the corresponding Vita02A training proxies rather than
# maintaining two different fixes in Isaac Lab and MuJoCo.  The old base-link
# proxy is deliberately excluded: it penetrates both hip-roll mesh collisions
# by about 26 mm at the official zero pose.
PROXY_COLLISION_LINKS = (
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "left_wrist_roll_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "right_wrist_roll_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
)
WRIST_PROXY_LINKS = frozenset(
    name for name in PROXY_COLLISION_LINKS if "wrist" in name
)

FINGER_JOINT_NAMES = (
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_thumb_ip",
    "lh_index_mcp_pitch",
    "lh_index_dip",
    "lh_middle_mcp_pitch",
    "lh_middle_dip",
    "lh_ring_mcp_pitch",
    "lh_ring_dip",
    "lh_pinky_mcp_pitch",
    "lh_pinky_dip",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_thumb_ip",
    "rh_index_mcp_pitch",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_dip",
    "rh_ring_mcp_pitch",
    "rh_ring_dip",
    "rh_pinky_mcp_pitch",
    "rh_pinky_dip",
)
FINGER_JOINT_SET = frozenset(FINGER_JOINT_NAMES)
HEAD_JOINT_NAMES = ("neck_yaw_joint", "neck_pitch_joint")
URDF_FIXED_JOINT_TAGS = frozenset(
    {
        "axis",
        "calibration",
        "dynamics",
        "limit",
        "mimic",
        "safety_controller",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _write(tree: ET.ElementTree, path: Path, *, declaration: bool) -> None:
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=declaration,
        short_empty_elements=True,
    )
    with path.open("ab") as stream:
        stream.write(b"\n")


def _children(element: ET.Element) -> list[ET.Element]:
    """Return XML elements without comments retained by the parser."""
    return [child for child in element if isinstance(child.tag, str)]


def _verify_sources(asset_root: Path) -> dict[str, str]:
    actual = {}
    for relative_name, expected in SOURCE_SHA256.items():
        path = asset_root / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Vita02A source asset: {path}")
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(
                f"Vita02A source digest mismatch for {path}: "
                f"expected {expected}, got {digest}"
            )
        actual[relative_name] = digest
    collision_proxy = (asset_root / COLLISION_PROXY_URDF_NAME).resolve()
    digest = _sha256(collision_proxy)
    if digest != COLLISION_PROXY_URDF_SHA256:
        raise ValueError(
            f"Vita02A collision proxy digest mismatch for {collision_proxy}: "
            f"expected {COLLISION_PROXY_URDF_SHA256}, got {digest}"
        )
    actual[COLLISION_PROXY_URDF_NAME] = digest
    return actual


def _collision_proxies(proxy_urdf: Path) -> dict[str, ET.Element]:
    root = ET.parse(proxy_urdf).getroot()
    proxies = {}
    for link_name in PROXY_COLLISION_LINKS:
        link = root.find(f"link[@name='{link_name}']")
        if link is None:
            raise ValueError(f"Missing Vita02A proxy link: {link_name}")
        collisions = link.findall("collision")
        if len(collisions) != 1:
            raise ValueError(
                f"Expected one collision proxy for {link_name}, "
                f"got {len(collisions)}"
            )
        geometry = collisions[0].find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError(f"Malformed collision proxy for {link_name}")
        shape = list(geometry)[0]
        if shape.tag != "cylinder":
            raise ValueError(
                f"Expected cylinder collision proxy for {link_name}, "
                f"got {shape.tag}"
            )
        proxies[link_name] = deepcopy(collisions[0])
    return proxies


def _install_urdf_collision_proxies(
    root: ET.Element, proxies: dict[str, ET.Element]
) -> None:
    for link_name, collision in proxies.items():
        link = root.find(f"link[@name='{link_name}']")
        if link is None:
            raise ValueError(f"Missing official Vita02A link: {link_name}")
        existing = link.findall("collision")
        expected = 1 if link_name in WRIST_PROXY_LINKS else 0
        if len(existing) != expected:
            raise ValueError(
                f"Unexpected official collisions on {link_name}: "
                f"expected {expected}, got {len(existing)}"
            )
        for item in existing:
            link.remove(item)
        collision.set("name", f"collision_{link_name}_proxy")
        link.append(collision)


def _derive_urdf(source: Path, proxy_urdf: Path, output: Path) -> None:
    tree = _parse(source)
    root = tree.getroot()
    found = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in FINGER_JOINT_SET:
            continue
        if joint.get("type") != "revolute":
            raise ValueError(f"Expected revolute source finger joint: {name}")
        joint.set("type", "fixed")
        for child in list(joint):
            if child.tag in URDF_FIXED_JOINT_TAGS:
                joint.remove(child)
        found.append(name)

    if tuple(found) != FINGER_JOINT_NAMES:
        raise ValueError(
            "Unexpected Vita02A URDF finger order: "
            f"expected {FINGER_JOINT_NAMES}, got {tuple(found)}"
        )

    movable = [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type", "fixed") != "fixed"
    ]
    if len(movable) != 30 or any(name in FINGER_JOINT_SET for name in movable):
        raise ValueError(f"Derived Vita02A URDF is not 30-DoF: {movable}")
    if not set(HEAD_JOINT_NAMES).issubset(movable):
        raise ValueError("Derived Vita02A URDF must keep both neck joints")

    _install_urdf_collision_proxies(root, _collision_proxies(proxy_urdf))
    root.set("name", "vita02a_linkerhand_meshcoll_30dof")
    root.insert(
        0,
        ET.Comment(
            " GENERATED from vita02a_linkerhand_meshcoll.urdf; "
            "22 LinkerHand joints welded at q=0, neck joints remain active; "
            "Vita02A shoulder/wrist collision proxies installed. "
        ),
    )
    _write(tree, output, declaration=True)


def _remove_named_finger_joints(root: ET.Element) -> tuple[str, ...]:
    removed = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "joint":
                continue
            name = child.get("name")
            if name in FINGER_JOINT_SET:
                parent.remove(child)
                removed.append(name)
    return tuple(removed)


def _remove_unused_finger_defaults(root: ET.Element) -> None:
    removed_defaults = 0
    removed_comments = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "default" and child.get("class") == "vt_finger":
                parent.remove(child)
                removed_defaults += 1
            elif (
                child.tag is ET.Comment
                and child.text is not None
                and "LinkerHand O6 finger" in child.text
            ):
                parent.remove(child)
                removed_comments += 1
    if removed_defaults != 1 or removed_comments != 2:
        raise ValueError(
            "Unexpected Vita02A finger defaults/comments: removed "
            f"{removed_defaults} default and {removed_comments} comments"
        )


def _rpy_to_wxyz(rpy: str) -> str:
    roll, pitch, yaw = (float(value) for value in rpy.split())
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    values = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return " ".join(f"{value:.12g}" for value in values)


def _find_named_body(root: ET.Element, name: str) -> ET.Element:
    matches = [body for body in root.iter("body") if body.get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one MJCF body named {name}, got {len(matches)}"
        )
    return matches[0]


def _install_mjcf_collision_proxies(
    root: ET.Element, proxies: dict[str, ET.Element]
) -> None:
    for link_name, collision in proxies.items():
        body = _find_named_body(root, link_name)
        existing = [
            geom
            for geom in body.findall("geom")
            if geom.get("class") == "collision"
        ]
        expected = 1 if link_name in WRIST_PROXY_LINKS else 0
        if len(existing) != expected:
            raise ValueError(
                f"Unexpected MJCF collisions on {link_name}: "
                f"expected {expected}, got {len(existing)}"
            )
        for geom in existing:
            body.remove(geom)

        origin = collision.find("origin")
        cylinder = collision.find("geometry/cylinder")
        if cylinder is None:
            raise ValueError(f"Missing cylinder proxy for {link_name}")
        radius = cylinder.get("radius")
        length = cylinder.get("length")
        assert radius is not None and length is not None
        attributes = {
            "name": f"collision_{link_name}_proxy",
            "class": "collision",
            "type": "cylinder",
            "size": f"{radius} {float(length) / 2.0:.12g}",
        }
        if origin is not None:
            attributes["pos"] = origin.get("xyz", "0 0 0")
            attributes["quat"] = _rpy_to_wxyz(origin.get("rpy", "0 0 0"))
        body.append(ET.Element("geom", attributes))


def _enable_hand_self_collision(root: ET.Element) -> None:
    hand_geom = root.find(".//default[@class='collision_hand']/geom")
    if hand_geom is None:
        raise ValueError("Missing Vita02A collision_hand defaults")
    if (hand_geom.get("contype"), hand_geom.get("conaffinity")) != ("2", "1"):
        raise ValueError("Unexpected official Vita02A hand collision mask")
    hand_geom.set("contype", "1")
    hand_geom.set("conaffinity", "1")

    collision_default = root.find(".//default[@class='collision']")
    assert collision_default is not None
    for child in collision_default:
        if (
            child.tag is ET.Comment
            and child.text
            and "Hand-link collision" in child.text
        ):
            child.text = (
                " Hand-link collision uses the common robot mask so contacts "
                "between opposite hands and between hands and the body are "
                "enabled. Finger bodies are welded at q=0 in this 30-DoF "
                "control view; same-weld contacts are filtered by the "
                "engines. "
            )
            return
    raise ValueError("Missing Vita02A hand collision comment")


def _derive_mjcf(source: Path, proxy_urdf: Path, output: Path) -> None:
    tree = _parse(source)
    root = tree.getroot()
    removed_joints = _remove_named_finger_joints(root)
    if tuple(removed_joints) != FINGER_JOINT_NAMES:
        raise ValueError(
            "Unexpected Vita02A MJCF finger order: "
            f"expected {FINGER_JOINT_NAMES}, got {removed_joints}"
        )

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("Vita02A MJCF has no actuator section")
    removed_actuators = []
    for item in list(actuator):
        joint_name = item.get("joint")
        if joint_name in FINGER_JOINT_SET:
            actuator.remove(item)
            removed_actuators.append(joint_name)
    if tuple(removed_actuators) != FINGER_JOINT_NAMES:
        raise ValueError(
            "Unexpected Vita02A finger actuator order: "
            f"expected {FINGER_JOINT_NAMES}, got {tuple(removed_actuators)}"
        )

    sensor = root.find("sensor")
    if sensor is None:
        raise ValueError("Vita02A MJCF has no sensor section")
    removed_sensors = []
    for item in list(sensor):
        references = {item.get("joint"), item.get("actuator")}
        finger_references = references & FINGER_JOINT_SET
        if finger_references:
            sensor.remove(item)
            removed_sensors.append(item)
    if len(removed_sensors) != 3 * len(FINGER_JOINT_NAMES):
        raise ValueError(
            "Expected three sensors per Vita02A finger joint, removed "
            f"{len(removed_sensors)} sensors"
        )

    active_joints = {
        joint.get("name")
        for joint in root.iter("joint")
        if joint.get("name") is not None
    }
    actuator_joints = [item.get("joint") for item in _children(actuator)]
    if len(active_joints) != 30 or len(actuator_joints) != 30:
        raise ValueError(
            "Derived Vita02A MJCF must contain 30 named joints and "
            f"30 actuators, got {len(active_joints)} and "
            f"{len(actuator_joints)}"
        )
    if set(actuator_joints) != active_joints:
        raise ValueError("Every derived Vita02A joint must have one actuator")
    if not set(HEAD_JOINT_NAMES).issubset(active_joints):
        raise ValueError("Derived Vita02A MJCF must keep both neck joints")

    _remove_unused_finger_defaults(root)
    proxies = _collision_proxies(proxy_urdf)
    _install_mjcf_collision_proxies(root, proxies)
    _enable_hand_self_collision(root)
    root.set("model", "vita02a_linkerhand_meshcoll_30dof")
    root.insert(
        0,
        ET.Comment(
            " GENERATED from vita02a_linkerhand_meshcoll.xml; "
            "22 LinkerHand joints welded at q=0, neck joints remain active; "
            "self-collision-safe Vita02A proxies and hand mask installed. "
        ),
    )
    _write(tree, output, declaration=False)


def _derive_scene(source: Path, output: Path) -> None:
    tree = _parse(source)
    root = tree.getroot()
    include = root.find("include")
    if include is None or include.get("file") != SOURCE_MJCF_NAME:
        raise ValueError("Unexpected Vita02A source scene include")
    include.set("file", Path(OUTPUT_MJCF_NAME).name)
    root.set("model", "vita02a_linkerhand_meshcoll_30dof scene")
    _write(tree, output, declaration=False)


def _derived_summary(asset_root: Path) -> dict[str, object]:
    urdf = asset_root / OUTPUT_URDF_NAME
    mjcf = asset_root / OUTPUT_MJCF_NAME
    scene = asset_root / OUTPUT_SCENE_NAME
    for path in (urdf, mjcf, scene):
        if not path.is_file():
            raise FileNotFoundError(f"Missing derived Vita02A asset: {path}")

    urdf_root = ET.parse(urdf).getroot()
    movable = [
        joint.get("name")
        for joint in urdf_root.findall("joint")
        if joint.get("type", "fixed") != "fixed"
    ]
    fixed = [
        joint.get("name")
        for joint in urdf_root.findall("joint")
        if joint.get("type", "fixed") == "fixed"
    ]
    total_mass = sum(
        float(mass.get("value"))
        for mass in urdf_root.findall("link/inertial/mass")
    )
    mjcf_root = ET.parse(mjcf).getroot()
    actuator = mjcf_root.find("actuator")
    sensor = mjcf_root.find("sensor")
    assert actuator is not None and sensor is not None
    return {
        "source_sha256": _verify_sources(asset_root),
        "outputs_sha256": {
            OUTPUT_URDF_NAME: _sha256(urdf),
            OUTPUT_MJCF_NAME: _sha256(mjcf),
            OUTPUT_SCENE_NAME: _sha256(scene),
        },
        "urdf": {
            "links": len(urdf_root.findall("link")),
            "joints": len(urdf_root.findall("joint")),
            "movable_joints": len(movable),
            "fixed_joints": len(fixed),
            "mass_kg": total_mass,
        },
        "mjcf": {
            "named_joints": sum(
                joint.get("name") is not None
                for joint in mjcf_root.iter("joint")
            ),
            "actuators": len(_children(actuator)),
            "sensors": len(_children(sensor)),
        },
    }


def derive(asset_root: Path) -> dict[str, object]:
    """Generate and statically verify all three 30-DoF asset files."""
    _verify_sources(asset_root)
    _derive_urdf(
        asset_root / SOURCE_URDF_NAME,
        (asset_root / COLLISION_PROXY_URDF_NAME).resolve(),
        asset_root / OUTPUT_URDF_NAME,
    )
    _derive_mjcf(
        asset_root / SOURCE_MJCF_NAME,
        (asset_root / COLLISION_PROXY_URDF_NAME).resolve(),
        asset_root / OUTPUT_MJCF_NAME,
    )
    _derive_scene(
        asset_root / SOURCE_SCENE_NAME,
        asset_root / OUTPUT_SCENE_NAME,
    )
    return _derived_summary(asset_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing derived files without rewriting them.",
    )
    args = parser.parse_args()
    asset_root = args.repo_root.resolve() / ASSET_RELATIVE_ROOT
    summary = (
        _derived_summary(asset_root)
        if args.verify_only
        else derive(asset_root)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
