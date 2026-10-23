#!/usr/bin/env python3
"""Compare and render Vita02A meshcoll-v3 default-pose candidates.

This is an analysis tool: it never rewrites a robot or training config.  It
uses the official no-hand 30-DoF MuJoCo asset for FK/contact measurements and
renders the official zero pose, the superseded training pose, and the
configured arms-down pose for review.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from holomotion.src.utils.config import setup_hydra_resolvers

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT.parent / "_artifacts/holoretarget_visualization/"
    "vita02a_default_pose_applied_20260905"
)
VITA_CONFIG = (
    REPO_ROOT / "holomotion/config/robot/vbot/Vita02A/"
    "vita02a_meshcoll_v3_30dof_training_isaaclab.yaml"
)
VITA_MJCF = (
    REPO_ROOT / "assets/robots/vbot/Vita02A/official_meshcoll_v3/"
    "vita02a_meshcoll.xml"
)

# Retain the superseded values only as a visualization baseline.  The source
# of truth for the active pose is the robot config read below.
LEGACY_ARM_ANGLES = {
    "left_shoulder_roll_joint": 0.4226,
    "right_shoulder_roll_joint": -0.4226,
    "left_elbow_joint": 0.0,
    "right_elbow_joint": 0.0,
}

REFERENCE_ROBOTS = {
    "K1": {
        "config": REPO_ROOT
        / "holomotion/config/robot/booster/K1/k1_training_isaaclab.yaml",
        "mjcf": REPO_ROOT / "assets/robots/booster/K1/K1_22dof.xml",
        "shoulders": ["Left_Arm_1", "Right_Arm_1"],
        "hands": ["left_hand_link", "right_hand_link"],
    },
    "R1": {
        "config": REPO_ROOT / "holomotion/config/robot/unitree/R1/"
        "r1_a5_26dof_training_isaaclab.yaml",
        "mjcf": REPO_ROOT / "assets/robots/unitree/R1/26dof/r1_a5_26dof.xml",
        "shoulders": [
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
        ],
        "hands": ["left_wrist_roll_link", "right_wrist_roll_link"],
    },
}


def _load_robot_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["robot"]


def _load_vita_config() -> dict:
    """Resolve the v3 overlay together with its inherited controller config."""

    setup_hydra_resolvers()
    with initialize_config_dir(
        config_dir=str((REPO_ROOT / "holomotion/config").resolve()),
        version_base=None,
    ):
        config = compose(
            config_name=(
                "robot/vbot/Vita02A/"
                "vita02a_meshcoll_v3_30dof_training_isaaclab"
            )
        )
    # This standalone robot composition intentionally lacks the training
    # algorithm keys used by unrelated robot.motion interpolations.  Resolve
    # only the pose subtree consumed by this analysis.
    return {
        "init_state": OmegaConf.to_container(
            config.robot.init_state, resolve=True
        )
    }


def _set_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_positions: dict[str, float],
) -> None:
    for name, value in joint_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(
                f"Joint {name!r} is missing from the MuJoCo model"
            )
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _body_position(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body {name!r} is missing from the MuJoCo model")
    return data.xpos[body_id].copy()


def _collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
    ]


def _distal_body_clearance(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, object]:
    """Return the closest non-adjacent hand/wrist-to-robot collision pair."""

    has_linker_hands = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_hand_base_link")
        >= 0
    )

    def distal_side(body_name: str) -> str | None:
        if has_linker_hands:
            if body_name.startswith("lh_"):
                return "left"
            if body_name.startswith("rh_"):
                return "right"
            return None
        if "_wrist_" in body_name:
            return body_name.split("_", 1)[0]
        return None

    rows = []
    geom_ids = _collision_geom_ids(model)
    for index, geom1 in enumerate(geom_ids):
        body1 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom1]),
        )
        assert body1 is not None
        for geom2 in geom_ids[index + 1 :]:
            body2 = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom2]),
            )
            assert body2 is not None
            side1 = distal_side(body1)
            side2 = distal_side(body2)
            if side1 is None and side2 is None:
                continue
            distal_body = body1 if side1 is not None else body2
            other_body = body2 if side1 is not None else body1
            side = side1 if side1 is not None else side2
            if has_linker_hands and other_body.startswith(
                "lh_" if side == "left" else "rh_"
            ):
                continue
            if other_body.startswith(f"{side}_") and any(
                token in other_body for token in ("shoulder", "elbow", "wrist")
            ):
                continue
            distance = float(
                mujoco.mj_geomDistance(model, data, geom1, geom2, 0.5, None)
            )
            # Some mesh pairs return exactly zero outside the narrow phase.
            # Actual contacts are reported separately through data.ncon.
            if distance <= 1.0e-9:
                continue
            rows.append((distance, geom1, geom2, distal_body, other_body))

    if not rows:
        return {"distance_m": None, "distal_body": None, "other_body": None}
    distance, geom1, geom2, distal_body, other_body = min(rows)
    return {
        "distance_m": distance,
        "distal_body": distal_body,
        # Preserve the historical key for downstream reports/tests.
        "hand_body": distal_body if has_linker_hands else None,
        "wrist_body": None if has_linker_hands else distal_body,
        "other_body": other_body,
        "geom_pair": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
        ],
    }


def _visual_bend_degrees(
    shoulder: np.ndarray, elbow: np.ndarray, finger: np.ndarray
) -> float:
    upper = elbow - shoulder
    lower = finger - elbow
    cosine = np.dot(upper, lower) / (
        np.linalg.norm(upper) * np.linalg.norm(lower)
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _vita_pose_metrics(
    model: mujoco.MjModel,
    name: str,
    joint_positions: dict[str, float],
) -> tuple[mujoco.MjData, dict[str, object]]:
    data = mujoco.MjData(model)
    _set_joint_positions(model, data, joint_positions)
    mujoco.mj_comPos(model, data)

    sides = {}
    has_linker_hands = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_hand_base_link")
        >= 0
    )
    for side, hand_prefix in (("left", "lh"), ("right", "rh")):
        shoulder = _body_position(model, data, f"{side}_shoulder_pitch_link")
        elbow = _body_position(model, data, f"{side}_elbow_link")
        wrist = _body_position(model, data, f"{side}_wrist_pitch_link")
        side_metrics = {
            "shoulder_xyz_m": shoulder.tolist(),
            "elbow_xyz_m": elbow.tolist(),
            "wrist_xyz_m": wrist.tolist(),
            "shoulder_to_wrist_xyz_m": (wrist - shoulder).tolist(),
        }
        endpoint = wrist
        if has_linker_hands:
            palm = _body_position(model, data, f"{hand_prefix}_hand_base_link")
            endpoint = _body_position(
                model, data, f"{hand_prefix}_middle_distal"
            )
            side_metrics.update(
                {
                    "palm_xyz_m": palm.tolist(),
                    "middle_finger_xyz_m": endpoint.tolist(),
                    "shoulder_to_palm_xyz_m": (palm - shoulder).tolist(),
                }
            )
        side_metrics["visual_arm_bend_deg"] = _visual_bend_degrees(
            shoulder, elbow, endpoint
        )
        sides[side] = side_metrics

    clearance = _distal_body_clearance(model, data)

    metric = {
        "name": name,
        "joint_positions_rad": joint_positions,
        "self_contact_count": int(data.ncon),
        "closest_wrist_body_clearance": clearance,
        # Preserve the v2 result contract while the v3 model has no hands.
        "closest_hand_body_clearance": clearance,
        "center_of_mass_xyz_m": data.subtree_com[1].tolist(),
        "sides": sides,
    }
    return data, metric


def _reference_metrics(
    name: str, spec: dict[str, object]
) -> dict[str, object]:
    config = _load_robot_config(spec["config"])
    model = mujoco.MjModel.from_xml_path(str(spec["mjcf"].resolve()))
    data = mujoco.MjData(model)
    positions = config["init_state"]["default_joint_angles"]
    _set_joint_positions(model, data, positions)

    sides = {}
    for side, shoulder_name, hand_name in zip(
        ("left", "right"),
        spec["shoulders"],
        spec["hands"],
        strict=True,
    ):
        shoulder = _body_position(model, data, shoulder_name)
        hand = _body_position(model, data, hand_name)
        sides[side] = {
            "shoulder_xyz_m": shoulder.tolist(),
            "hand_xyz_m": hand.tolist(),
            "shoulder_to_hand_xyz_m": (hand - shoulder).tolist(),
        }
    arm_angles = {
        joint: value
        for joint, value in positions.items()
        if any(
            token in joint.lower() for token in ("shoulder", "elbow", "wrist")
        )
    }
    return {
        "name": name,
        "self_contact_count": int(data.ncon),
        "arm_joint_positions_rad": arm_angles,
        "sides": sides,
    }


def _render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    *,
    azimuth: float,
    elevation: float,
    label: str,
    collision_overlay: bool = False,
) -> np.ndarray:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.78)
    camera.distance = 2.05
    camera.azimuth = azimuth
    camera.elevation = elevation

    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1
    if collision_overlay:
        option.geomgroup[3] = 1
    renderer.update_scene(data, camera=camera, scene_option=option)
    frame = renderer.render().copy()
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    draw.rounded_rectangle((12, 12, 12 + len(label) * 12, 44), fill="white")
    draw.text((20, 18), label, fill="black", font=font)
    return np.asarray(image)


def _save_montage(images: list[np.ndarray], output: Path) -> None:
    pil_images = [Image.fromarray(image) for image in images]
    width, height = pil_images[0].size
    canvas = Image.new("RGB", (width * 3, height * 2), "white")
    for index, image in enumerate(pil_images):
        canvas.paste(image, ((index % 3) * width, (index // 3) * height))
    canvas.save(output)


def _write_turntable(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    output: Path,
    *,
    num_frames: int,
    fps: int,
) -> None:
    with imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame_index in range(num_frames):
            azimuth = 90.0 + 360.0 * frame_index / num_frames
            frame = _render_frame(
                renderer,
                data,
                azimuth=azimuth,
                elevation=-4.0,
                label="Vita02A configured default - natural arms down",
            )
            writer.append_data(frame)


def analyze(output_dir: Path, *, num_video_frames: int, fps: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_vita_config()
    configured = config["init_state"]["default_joint_angles"]
    legacy = dict(configured)
    legacy.update(LEGACY_ARM_ANGLES)
    poses = {
        "official_q0": {},
        "legacy_training_default": legacy,
        "configured_training_default": configured,
    }

    model = mujoco.MjModel.from_xml_path(str(VITA_MJCF.resolve()))
    pose_data = {}
    metrics = {}
    for name, positions in poses.items():
        data, metric = _vita_pose_metrics(model, name, positions)
        pose_data[name] = data
        metrics[name] = metric

    # The official MJCF keeps MuJoCo's 480 px default offscreen framebuffer.
    # Stay within that read-only asset contract instead of mutating its visual
    # settings solely for this review utility.
    renderer = mujoco.Renderer(model, height=480, width=480)
    montage_frames = []
    for name, data in pose_data.items():
        for view, azimuth in (("front", 0.0), ("side", 90.0)):
            frame = _render_frame(
                renderer,
                data,
                azimuth=azimuth,
                elevation=-4.0,
                label=f"{name} - {view}",
            )
            imageio.imwrite(output_dir / f"{name}_{view}.png", frame)
            montage_frames.append(frame)
    _save_montage(montage_frames, output_dir / "vita02a_pose_montage.png")

    collision_frame = _render_frame(
        renderer,
        pose_data["configured_training_default"],
        azimuth=0.0,
        elevation=-4.0,
        label="configured default - visual and collision geometry",
        collision_overlay=True,
    )
    imageio.imwrite(
        output_dir / "configured_collision_overlay_front.png",
        collision_frame,
    )
    _write_turntable(
        renderer,
        pose_data["configured_training_default"],
        output_dir / "vita02a_configured_default_turntable.mp4",
        num_frames=num_video_frames,
        fps=fps,
    )
    renderer.close()

    report = {
        "schema": "vita02a_default_pose_review_v3",
        "asset": str(VITA_MJCF.relative_to(REPO_ROOT)),
        "config_read_only": str(VITA_CONFIG.relative_to(REPO_ROOT)),
        "configured_arm_angles_rad": {
            name: configured[name] for name in LEGACY_ARM_ANGLES
        },
        "legacy_arm_angles_rad": LEGACY_ARM_ANGLES,
        "vita02a": metrics,
        "references": {
            name: _reference_metrics(name, spec)
            for name, spec in REFERENCE_ROBOTS.items()
        },
        "artifacts": {
            "montage": "vita02a_pose_montage.png",
            "collision_overlay": "configured_collision_overlay_front.png",
            "turntable": "vita02a_configured_default_turntable.mp4",
        },
    }
    (output_dir / "pose_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-video-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    report = analyze(
        args.output_dir.resolve(),
        num_video_frames=args.num_video_frames,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
