#!/usr/bin/env python3
"""Roll out and visualize one formal-SMPL clip through the GMR-free chain."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holomotion.src.evaluation.eval_mujoco_sim2sim import (  # noqa: E402
    OffscreenRenderer,
    process_config,
)
from holomotion.src.evaluation.gmr_free_smpl_mujoco import (  # noqa: E402
    GmrFreeSmplMujocoEvaluator,
    load_pure_smpl_clip,
    save_pure_smpl_clip,
    validate_named_dof_matrix,
)
from holomotion.src.utils.config import setup_hydra_resolvers  # noqa: E402


def _default_ffmpeg() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(args: argparse.Namespace):
    setup_hydra_resolvers()
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now", lambda pattern: datetime.now().strftime(pattern)
        )
    default_eval = (
        REPO_ROOT / "holomotion/config/evaluation/eval_mujoco_sim2sim.yaml"
    )
    override = OmegaConf.load(default_eval)
    with open_dict(override):
        override.ckpt_onnx_path = str(Path(args.onnx).resolve())
        override.motion_npz_path = str(Path(args.pure_clip).resolve())
        override.motion_npz_dir = None
        override.robot_xml_path = str(Path(args.robot_xml).resolve())
        override.headless = True
        override.record_video = False
        override.use_gpu = True
        override.gpu_id = int(args.gpu_id)
        override.policy_action_delay_step = 0
        override.action_delay_type = "step"
        override.model_type = "holomotion"
        override.mujoco_actuator_model = str(args.mujoco_actuator_model)
        override.gmr_free_reset_profile = str(args.reset_profile)
        override.gmr_free_settle_seconds = float(args.settle_seconds)
        override.gmr_free_velocity_onnx_path = (
            None
            if args.velocity_onnx is None
            else str(Path(args.velocity_onnx).resolve())
        )
        override.gmr_free_velocity_config_path = (
            None
            if args.velocity_config is None
            else str(Path(args.velocity_config).resolve())
        )
        override.gmr_free_velocity_warmup_seconds = float(
            args.velocity_warmup_seconds
        )
    return process_config(override)


class FfmpegWriter:
    def __init__(
        self, path: Path, *, width: int, height: int, fps: int, ffmpeg: str
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}")


def _draw_label(rgb: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.78
    thickness = 2
    lines = text.splitlines() or [""]
    line_height = 34
    widths = [
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    ]
    right = min(int(rgb.shape[1]) - 14, 28 + max(widths, default=0) + 14)
    bottom = 24 + line_height * len(lines) + 8
    cv2.rectangle(rgb, (14, 14), (right, bottom), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            rgb,
            line,
            (28, 45 + index * line_height),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _apply_unified_render_style(model: mujoco.MjModel, style: str) -> None:
    """Apply display-only styling without changing robot dynamics.

    The eight embodiments keep their own MJCF, contacts, actuators and
    inertial parameters.  This function only normalizes sky/floor pixels and
    MuJoCo visual lighting so a 4x2 comparison does not mix eight unrelated
    scene themes.
    """
    if style == "native":
        return
    if style != "g1_light":
        raise ValueError(f"Unsupported render style: {style}")

    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.40, 0.40, 0.40)
    model.vis.headlight.diffuse[:] = (0.60, 0.60, 0.60)
    model.vis.headlight.specular[:] = (0.40, 0.40, 0.40)
    model.vis.rgba.haze[:] = (0.15, 0.25, 0.35, 1.0)
    # Scene XMLs contain different numbers, colors and directions of lights.
    # Keep only the common camera headlight in comparison renders.
    if int(model.nlight) > 0:
        model.light_active[:] = 0

    # Normalize every compiled skybox to the white G1 presentation theme.
    for texture_id in range(int(model.ntex)):
        if int(model.tex_type[texture_id]) != int(
            mujoco.mjtTexture.mjTEXTURE_SKYBOX
        ):
            continue
        address = int(model.tex_adr[texture_id])
        size = (
            int(model.tex_width[texture_id])
            * int(model.tex_height[texture_id])
            * int(model.tex_nchannel[texture_id])
        )
        model.tex_data[address : address + size] = 255

    # A texture-independent light-blue floor is consistent even for PM01,
    # whose official visual MJCF has no texture/material table at all.
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        model.geom_matid[geom_id] = -1
        model.geom_rgba[geom_id] = (0.72, 0.82, 0.95, 1.0)


def rollout(args: argparse.Namespace) -> Path:
    config = build_config(args)
    evaluator = GmrFreeSmplMujocoEvaluator(config)
    output = evaluator.run_pure_rollout(args.rollout)
    print(f"[ok] pure rollout: {output}")
    return output


def render(args: argparse.Namespace) -> Path:
    rollout_path = Path(args.rollout).resolve()
    with np.load(rollout_path, allow_pickle=False) as payload:
        dof = np.asarray(payload["robot_dof_pos"], dtype=np.float64)
        if "robot_dof_names" not in payload.files:
            raise ValueError(
                "Rollout has no explicit robot_dof_names; regenerate it "
                "with the v2 GMR-free rollout schema before rendering"
            )
        rollout_dof_names = tuple(
            str(name) for name in np.asarray(payload["robot_dof_names"])
        )
        root_pos = np.asarray(
            payload["robot_global_translation"][:, 0], dtype=np.float64
        )
        root_quat = np.asarray(
            payload["robot_global_rotation_quat"][:, 0], dtype=np.float64
        )
        target_key = (
            "smpl_semantic_position_aligned"
            if "smpl_semantic_position_aligned" in payload.files
            else "smpl_semantic_position"
        )
        target = np.asarray(payload[target_key], dtype=np.float64)
        metadata = json.loads(
            str(np.asarray(payload["metadata"]).reshape(()).item())
        )
    if int(metadata.get("robot_reference_reads", -1)) != 0:
        raise ValueError("Rollout does not prove zero Robot-reference reads")

    config = build_config(args)
    dof_names = validate_named_dof_matrix(
        dof,
        rollout_dof_names,
        tuple(str(name) for name in config.robot.dof_names),
    )
    model = mujoco.MjModel.from_xml_path(str(Path(args.robot_xml).resolve()))
    _apply_unified_render_style(model, str(args.render_style))
    data = mujoco.MjData(model)
    qpos_addresses: list[int] = []
    for name in dof_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise KeyError(f"MuJoCo joint not found: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint]))
    if dof.shape != (root_pos.shape[0], len(qpos_addresses)):
        raise ValueError(f"Unexpected robot_dof_pos shape: {dof.shape}")

    width, height = int(args.width), int(args.height)
    renderer = OffscreenRenderer(
        model,
        height=height,
        width=width,
        distance=float(args.camera_distance),
        azimuth=float(args.camera_azimuth),
        elevation=float(args.camera_elevation),
    )
    current_markers = np.zeros((target.shape[1], 3), dtype=np.float32)

    def overlay(scene) -> None:
        radius = np.float32(args.marker_radius)
        size = np.asarray([radius, 0.0, 0.0], dtype=np.float32)
        rotation = np.eye(3, dtype=np.float32).reshape(-1)
        color = np.asarray([0.85, 0.05, 0.05, 1.0], dtype=np.float32)
        first_marker = int(scene.ngeom)
        for index, position in enumerate(current_markers):
            geom_index = first_marker + index
            if geom_index >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[geom_index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                position,
                rotation,
                color,
            )
            scene.ngeom = geom_index + 1

    renderer.set_overlay_callback(overlay)
    writer = FfmpegWriter(
        Path(args.raw_video).resolve(),
        width=width,
        height=height,
        fps=int(args.fps),
        ffmpeg=args.ffmpeg,
    )
    try:
        for frame in range(root_pos.shape[0]):
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0
            data.qpos[:3] = root_pos[frame]
            xyzw = root_quat[frame]
            data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
            data.qpos[qpos_addresses] = dof[frame]
            mujoco.mj_forward(model, data)
            current_markers[:] = target[frame]
            target_root = target[frame, 0]
            if args.camera_follow == "robot":
                lookat = root_pos[frame].copy()
            elif args.camera_follow == "midpoint":
                lookat = 0.5 * (root_pos[frame] + target_root)
            else:
                lookat = target_root.copy()
            lookat[2] += float(args.camera_height_offset)
            distance = float(args.camera_distance)
            if args.camera_auto_distance:
                root_target_separation = float(
                    np.linalg.norm(root_pos[frame, :2] - target_root[:2])
                )
                distance += (
                    float(args.camera_distance_per_separation)
                    * root_target_separation
                )
            renderer.set_align_view(
                lookat=lookat,
                distance=distance,
                azimuth=float(args.camera_azimuth),
                elevation=float(args.camera_elevation),
            )
            rgb = np.ascontiguousarray(renderer.render(data))
            _draw_label(
                rgb,
                str(args.video_label),
            )
            writer.write(rgb)
    finally:
        renderer.close()
        writer.close()
    output = Path(args.raw_video).resolve()
    print(f"[ok] raw video: {output}")
    return output


def assemble(args: argparse.Namespace) -> Path:
    old = Path(args.historical_video).resolve()
    new = Path(args.raw_video).resolve()
    output = Path(args.comparison_video).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "[0:v]crop=1280:720:0:0[smpl];"
        "[0:v]crop=1280:720:1280:0[rew3];"
        "[0:v]crop=1280:720:0:720[olddefault];"
        "[1:v]scale=1280:720[new];"
        "[smpl][rew3]hstack=inputs=2[top];"
        "[olddefault][new]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )
    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(old),
            "-i",
            str(new),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    manifest = {
        "schema_version": "gmr_free_c04_visual_comparison_v1",
        "layout": {
            "top_left": "SMPL",
            "top_right": "historical Rew3",
            "bottom_left": "historical Canonical Default model_22000",
            "bottom_right": "new GMR-free project chain model_22000",
        },
        "pure_clip": str(Path(args.pure_clip).resolve()),
        "pure_clip_sha256": sha256(args.pure_clip),
        "onnx": str(Path(args.onnx).resolve()),
        "onnx_sha256": sha256(args.onnx),
        "rollout": str(Path(args.rollout).resolve()),
        "rollout_sha256": sha256(args.rollout),
        "video": str(output),
        "video_sha256": sha256(output),
        "robot_reference_reads": 0,
        "reset_profile": str(args.reset_profile),
        "settle_seconds": float(args.settle_seconds),
        "velocity_warmup_seconds": float(args.velocity_warmup_seconds),
        "camera": {
            "distance": float(args.camera_distance),
            "azimuth": float(args.camera_azimuth),
            "elevation": float(args.camera_elevation),
            "height_offset": float(args.camera_height_offset),
        },
        "deployment_sequence_simulated": [
            "MoveToDefault with ROS2 deployment targets and gains",
            "run released velocity policy with a zero command",
            "capture motion-entry yaw alignment",
            "clear action and KV-cache state",
            "run pure-SMPL motion policy",
        ],
        "velocity_policy_stage": {
            "onnx": str(Path(args.velocity_onnx).resolve()),
            "onnx_sha256": sha256(args.velocity_onnx),
            "config": str(Path(args.velocity_config).resolve()),
            "config_sha256": sha256(args.velocity_config),
            "command": [0.0, 0.0, 0.0, 0.0],
            "observation": "8-frame official 776D velocity Actor schema",
        },
        "comparison_boundary": (
            "The new bottom-right rollout is fail-closed pure SMPL with "
            "a deployment-style stand/settle/motion entry. Historical "
            "quadrants come from the earlier paired evaluator and are visual "
            "context, not reset-parity evidence."
        ),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] comparison video: {output}")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "command", choices=("prepare", "rollout", "render", "assemble", "all")
    )
    result.add_argument("--source-clip", required=True)
    result.add_argument("--pure-clip", required=True)
    result.add_argument("--onnx", required=True)
    result.add_argument("--robot-xml", required=True)
    result.add_argument("--rollout", required=True)
    result.add_argument("--raw-video", required=True)
    result.add_argument("--historical-video", required=True)
    result.add_argument("--comparison-video", required=True)
    result.add_argument("--frames", type=int, default=None)
    result.add_argument("--gpu-id", type=int, default=0)
    result.add_argument("--fps", type=int, default=50)
    result.add_argument("--width", type=int, default=1280)
    result.add_argument("--height", type=int, default=720)
    result.add_argument("--camera-distance", type=float, default=5.0)
    result.add_argument("--camera-azimuth", type=float, default=150.0)
    result.add_argument("--camera-elevation", type=float, default=-20.0)
    result.add_argument("--camera-height-offset", type=float, default=0.3)
    result.add_argument(
        "--camera-follow",
        choices=("target", "robot", "midpoint"),
        default="target",
        help=(
            "Camera look-at source. Midpoint keeps the simulated robot "
            "and SMPL target visible when their global roots diverge."
        ),
    )
    result.add_argument(
        "--camera-auto-distance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Increase camera distance as robot-target XY separation grows.",
    )
    result.add_argument(
        "--camera-distance-per-separation",
        type=float,
        default=0.9,
        help="Extra camera distance per metre of robot-target XY separation.",
    )
    result.add_argument("--marker-radius", type=float, default=0.03)
    result.add_argument(
        "--mujoco-actuator-model",
        choices=("auto", "xml", "training"),
        default="auto",
        help=(
            "Explicit MuJoCo actuator contract. Use xml for legacy/plain PD "
            "or training for the configured training actuator emulation."
        ),
    )
    result.add_argument(
        "--render-style",
        choices=("native", "g1_light"),
        default="native",
        help="Display-only scene style; does not change rollout dynamics.",
    )
    result.add_argument(
        "--video-label",
        default=(
            "New GMR-free project | Stand -> Velocity -> SMPL | model_22000"
        ),
    )
    result.add_argument(
        "--reset-profile",
        choices=("training_canonical_default", "deployment_stand"),
        default="deployment_stand",
    )
    result.add_argument("--settle-seconds", type=float, default=1.0)
    result.add_argument("--velocity-onnx")
    result.add_argument("--velocity-config")
    result.add_argument("--velocity-warmup-seconds", type=float, default=1.0)
    result.add_argument("--ffmpeg", default=_default_ffmpeg())
    return result


def main() -> None:
    args = parser().parse_args()
    if args.velocity_warmup_seconds > 0.0 and (
        args.velocity_onnx is None or args.velocity_config is None
    ):
        raise SystemExit(
            "--velocity-onnx and --velocity-config are required when "
            "--velocity-warmup-seconds is positive"
        )
    if args.command in {"prepare", "all"}:
        output = save_pure_smpl_clip(
            args.source_clip, args.pure_clip, frame_count=args.frames
        )
        clip = load_pure_smpl_clip(output)
        print(f"[ok] pure clip: {output} ({clip.frame_count} frames)")
    if args.command in {"rollout", "all"}:
        rollout(args)
    if args.command in {"render", "all"}:
        render(args)
    if args.command in {"assemble", "all"}:
        assemble(args)


if __name__ == "__main__":
    main()
