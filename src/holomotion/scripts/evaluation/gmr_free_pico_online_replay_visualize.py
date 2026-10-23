#!/usr/bin/env python3
"""Replay raw Pico CSV through the GMR-free online deployment simulation.

Left: the exact best-effort SMPL telemetry conversion rendered as a neutral
SMPL mesh.  Right: raw Pico -> GPU online SMPL ring -> shared Actor observation
-> motion policy -> PD-controlled MuJoCo G1.  The script never imports or reads
Robot-reference motion and never starts ROS or a real-robot transport.
"""

from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from loguru import logger
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.holomotion_teleop.holomotion_teleop_mjviewer import (  # noqa: E402
    SmplMeshModel,
    _mesh_compile_transform,
    _smpl_obj_bytes,
)
from holomotion.src.evaluation.eval_mujoco_sim2sim import (  # noqa: E402
    OffscreenRenderer,
    process_config,
)
from holomotion.src.evaluation.gmr_free_smpl_mujoco import (  # noqa: E402
    GmrFreeSmplMujocoEvaluator,
    validate_named_dof_matrix,
)
from holomotion.src.evaluation.pico_online_replay import (  # noqa: E402
    PicoReplaySchedule,
    RawPicoClip,
    build_latest_only_schedule,
    load_raw_pico_csv,
    select_shortest_per_source,
)
from holomotion.src.utils.config import setup_hydra_resolvers  # noqa: E402
from holosmpl.converters.pico.adapter import (  # noqa: E402
    body_poses_to_smpl_pose_trans,
)
from holosmpl.converters.pico_to_smpl_gpu import (  # noqa: E402
    PicoToSmplGpuConverter,
)

DEPLOYMENT_SRC = REPO_ROOT / "deployment/unitree_g1_ros2_29dof/src"
if str(DEPLOYMENT_SRC) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_SRC))

from humanoid_policy.reference_transport import (  # noqa: E402
    BestEffortReferencePublisher,
    pack_numpy_message,
    unpack_numpy_message,
)
from humanoid_policy.smpl_reference import GpuOnlineSmplReference  # noqa: E402

TOPIC = b"smpl_reference"
VISUALIZATION_PROFILE = "two_pass_matched_scene_surface_centered_v7"


def _patch_legacy_smpl_pickle_dependencies() -> None:
    """Restore aliases needed only while loading the licensed SMPL v1.1 pkl."""

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _default_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaw_from_rotvec(rotvec: np.ndarray) -> float:
    """Return Z-up yaw from one XYZ axis-angle rotation vector."""

    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return 0.0
    xyz = vector / angle * np.sin(0.5 * angle)
    w = np.cos(0.5 * angle)
    x, y, z = xyz
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    """Return Z-up yaw from one normalized XYZW quaternion."""

    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        raise ValueError("Cannot extract yaw from a zero quaternion")
    x, y, z, w = quat / norm
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def heading_relative_camera_azimuths(
    human_root_rotvec: np.ndarray,
    robot_root_quat_xyzw: np.ndarray,
    base_azimuth_degrees: float,
) -> tuple[float, float]:
    """Give Human and G1 the same camera view relative to initial heading."""

    human_yaw = _yaw_from_rotvec(human_root_rotvec)
    robot_yaw = _yaw_from_quat_xyzw(robot_root_quat_xyzw)
    return (
        float(base_azimuth_degrees + np.degrees(human_yaw)),
        float(base_azimuth_degrees + np.degrees(robot_yaw)),
    )


def _disable_renderer_shadows(renderer: OffscreenRenderer) -> None:
    """Hide long cast shadows without changing the simulated scene."""

    renderer._scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
    renderer._scene.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0
    renderer.model.light_castshadow[:] = 0


def _disable_scene_shadows(scene: mujoco.MjvScene) -> None:
    """Apply flags after mjv_updateScene, which resets first-frame flags."""

    scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
    scene.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0


def _smpl_comparison_xml() -> str:
    """Use the production SMPL mesh contract with a complete light scene.

    The production telemetry viewer intentionally uses a minimal scene with no
    skybox.  EGL therefore renders most pixels above its dark floor as pure
    black.  For a side-by-side validation that looks like a missing image, so
    this visualization-only scene keeps the same dynamic mesh contract while
    matching the G1 panel's sky and checker floor.
    """

    return """
<mujoco model="smpl_mesh_comparison_viewer">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.8 0.8 0.8"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <mesh name="smpl_surface" file="smpl.obj"/>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.5 0.6"
             rgb2="0.9 0.95 1.0" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="1.0 1.0 1.0" rgb2="0.6 0.8 1.0" markrgb="0 0 0"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0"/>
    <material name="smpl_skin" rgba="0.72 0.48 0.34 1"
              specular="0.15" shininess="0.25"/>
  </asset>
  <worldbody>
    <light pos="0 -3 4" dir="0 1 -1" directional="true"
           castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.01"
          material="groundplane" contype="0" conaffinity="0"/>
    <geom name="smpl_mesh" type="mesh" mesh="smpl_surface"
          material="smpl_skin" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def _show_robot_visual_geoms_only(
    renderer: OffscreenRenderer, model: mujoco.MjModel
) -> None:
    """Hide duplicate collision meshes while retaining the ground plane."""

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise KeyError("MuJoCo render model has no floor geom")
    model.geom_group[floor_id] = 2
    renderer._opt.geomgroup[0] = 0
    renderer._opt.geomgroup[1] = 1
    renderer._opt.geomgroup[2] = 1


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _slug(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}"


def _save_formal_target(
    output: Path,
    pose: np.ndarray,
    trans: np.ndarray,
    source: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        human_pose_aa=np.asarray(pose, dtype=np.float32),
        human_root_trans=np.asarray(trans, dtype=np.float32),
        motion_fps=np.asarray(50.0, dtype=np.float32),
        schema_version=np.asarray("gmr_free_pico_online_policy_target_v1"),
        source_clip_id=np.asarray(_slug(source)),
    )
    return output


def build_config(args: argparse.Namespace, pure_clip: Path):
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
        override.motion_npz_path = str(pure_clip.resolve())
        override.motion_npz_dir = None
        override.robot_xml_path = str(Path(args.robot_xml).resolve())
        override.headless = True
        override.record_video = False
        override.use_gpu = True
        override.gpu_id = int(args.gpu_id)
        override.policy_action_delay_step = 0
        override.action_delay_type = "step"
        override.model_type = "holomotion"
        override.gmr_free_reset_profile = "deployment_stand"
        override.gmr_free_settle_seconds = float(args.settle_seconds)
        override.gmr_free_velocity_onnx_path = str(
            Path(args.velocity_onnx).resolve()
        )
        override.gmr_free_velocity_config_path = str(
            Path(args.velocity_config).resolve()
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
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with code {code}")


class SmplMeshOffscreenRenderer:
    """Offscreen form of the production SMPL telemetry viewer."""

    def __init__(
        self,
        smpl_model_path: Path,
        *,
        width: int,
        height: int,
        distance: float,
        azimuth: float,
        elevation: float,
    ) -> None:
        _patch_legacy_smpl_pickle_dependencies()
        self.smpl = SmplMeshModel(smpl_model_path)
        neutral = self.smpl.vertices(
            np.zeros(72, dtype=np.float32), np.zeros(3, dtype=np.float32)
        )
        self.model = mujoco.MjModel.from_xml_string(
            _smpl_comparison_xml(),
            assets={"smpl.obj": _smpl_obj_bytes(neutral, self.smpl.faces)},
        )
        # MuJoCo's default offscreen framebuffer is only 640x480.  Rendering
        # our 1280x720 panel against that buffer silently leaves the right side
        # untouched (pure black), rather than raising a size error.  The G1
        # scene already declares a large buffer; this standalone SMPL scene
        # must do the same before MjrContext is constructed.
        self.model.vis.global_.offwidth = int(width)
        self.model.vis.global_.offheight = int(height)
        self.data = mujoco.MjData(self.model)
        self.mesh_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MESH, "smpl_surface"
        )
        self.mesh_adr = int(self.model.mesh_vertadr[self.mesh_id])
        self.mesh_count = int(self.model.mesh_vertnum[self.mesh_id])
        compiled = self.model.mesh_vert[
            self.mesh_adr : self.mesh_adr + self.mesh_count
        ].copy()
        self.rotation, self.translation = _mesh_compile_transform(
            neutral, compiled
        )
        self.renderer = OffscreenRenderer(
            self.model,
            height=height,
            width=width,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
        )
        self.ground_offset: float | None = None

    def render(self, pose: np.ndarray, trans: np.ndarray) -> np.ndarray:
        # Two offscreen renderers live in the same thread. Explicitly activate
        # this context before uploading the dynamic SMPL mesh; otherwise the
        # mesh can contaminate the concurrently used G1 renderer.
        self.renderer._gl_ctx.make_current()
        vertices = self.smpl.vertices(pose, trans)
        if self.ground_offset is None:
            self.ground_offset = -float(vertices[:, 2].min())
        vertices = vertices.copy()
        vertices[:, 2] += self.ground_offset
        compiled = vertices @ self.rotation + self.translation
        self.model.mesh_vert[
            self.mesh_adr : self.mesh_adr + self.mesh_count
        ] = compiled
        mujoco.mjr_uploadMesh(self.model, self.renderer._con, self.mesh_id)
        mujoco.mj_forward(self.model, self.data)
        # Center the comparison view on the visible grounded SMPL surface.
        # These source vertices are also the rendered world coordinates after
        # MuJoCo applies its inverse mesh-compile transform.  Raw Pico root
        # translations have a source-dependent root-to-surface offset and can
        # therefore crop the human even when the underlying pose is correct.
        lookat = vertices.mean(axis=0)
        self.renderer.set_align_view(lookat=lookat)
        return np.ascontiguousarray(self.renderer.render(self.data))

    def close(self) -> None:
        self.renderer.close()


def _sample_meta(
    clip: RawPicoClip, source_index: int, sample_time: float
) -> dict:
    return {
        "timestamp_realtime": float(clip.timestamp_realtime[source_index]),
        "timestamp_monotonic": float(sample_time),
        "timestamp_ns": int(round(sample_time * 1.0e9)),
        "dt": float(clip.pico_dt[source_index]),
        "fps": float(clip.pico_fps[source_index]),
    }


def _materialize_wire_payload(
    publisher: BestEffortReferencePublisher,
    clip: RawPicoClip,
    *,
    source_index: int,
    sample_time: float,
) -> dict[str, np.ndarray]:
    publisher.submit_pico(
        clip.body_poses[source_index],
        frame_index=int(source_index),
        sample_meta=_sample_meta(clip, source_index, sample_time),
    )
    if publisher._latest is None:
        raise RuntimeError("Best-effort telemetry did not retain a Pico frame")
    _, snapshot = publisher._latest
    payload = publisher._materialize_payload(snapshot)
    packet = pack_numpy_message(payload, topic=TOPIC)
    unpacked = unpack_numpy_message(packet, expected_topic=TOPIC)
    required = {"human_pose_aa", "human_root_trans", "frame_index"}
    missing = sorted(required - set(unpacked))
    if missing:
        raise KeyError(f"SMPL telemetry wire payload is missing {missing}")
    return unpacked


def _store_online_event(
    reference: GpuOnlineSmplReference,
    clip: RawPicoClip,
    schedule: PicoReplaySchedule,
    event_index: int,
) -> None:
    source_index = int(schedule.event_source_indices[event_index])
    sample_time = float(schedule.event_sample_times[event_index])
    accepted = reference.store_pico(
        clip.body_poses[source_index],
        sample_time=sample_time,
        current_time=sample_time,
        frame_index=source_index,
    )
    if not accepted:
        raise RuntimeError(
            f"GPU online reference rejected event {event_index}"
        )


def _save_rollout(
    output: Path,
    *,
    evaluator: GmrFreeSmplMujocoEvaluator,
    clip: RawPicoClip,
    schedule: PicoReplaySchedule,
    telemetry_pose: np.ndarray,
    telemetry_trans: np.ndarray,
    telemetry_parity_max_abs: float,
) -> dict[str, Any]:
    dof = np.stack(evaluator._robot_dof_pos_seq)
    root_pos = np.stack(evaluator._robot_global_translation_seq)[:, 0]
    root_quat = np.stack(evaluator._robot_global_rotation_quat_seq)[:, 0]
    actions = np.stack(evaluator._robot_actions_seq)
    target_indices = schedule.source_indices
    latest_indices = np.asarray(
        [
            schedule.latest_source_index(i)
            for i in range(schedule.policy_frame_count)
        ],
        dtype=np.int64,
    )
    lag_seconds = (
        clip.timestamp_monotonic[latest_indices]
        - clip.timestamp_monotonic[target_indices]
    )
    root_gravity = 1.0 - 2.0 * (
        np.square(root_quat[:, 0]) + np.square(root_quat[:, 1])
    )
    metadata = {
        "schema_version": "gmr_free_pico_online_rollout_v1",
        "raw_pico_csv": str(clip.path),
        "raw_source_frames": clip.frame_count,
        "policy_frames": schedule.policy_frame_count,
        "policy_hz": schedule.policy_hz,
        "source_fps_mean": float(np.mean(clip.pico_fps)),
        "timing": "raw Pico latest-only at 50 Hz; no interpolation",
        "online_history_frames": schedule.history_frames,
        "online_future_lag_frames": schedule.future_lag_frames,
        "prefill_events": schedule.prefill_event_count,
        "tail_hold_events": schedule.future_lag_frames,
        "telemetry_wire_roundtrips": schedule.policy_frame_count,
        "telemetry_cpu_batch_parity_max_abs": telemetry_parity_max_abs,
        "robot_reference_reads": 0,
        "reset": "deployment_stand -> released velocity policy -> SMPL motion",
        "settle_summary": evaluator._settle_summary,
        "velocity_warmup": evaluator._velocity_warmup_summary,
        "target_latest_lag_seconds_mean": float(np.mean(lag_seconds)),
        "target_latest_lag_seconds_p95": float(np.percentile(lag_seconds, 95)),
        "root_height_min": float(np.min(root_pos[:, 2])),
        "root_gravity_projection_min": float(np.min(root_gravity)),
        "all_finite": bool(
            np.isfinite(dof).all()
            and np.isfinite(root_pos).all()
            and np.isfinite(root_quat).all()
            and np.isfinite(actions).all()
        ),
    }
    if not metadata["all_finite"]:
        raise ValueError("Online Pico rollout contains NaN or Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        telemetry_human_pose_aa=np.asarray(telemetry_pose, dtype=np.float32),
        telemetry_human_root_trans=np.asarray(
            telemetry_trans, dtype=np.float32
        ),
        policy_target_source_index=target_indices,
        telemetry_latest_source_index=latest_indices,
        robot_dof_pos=dof,
        robot_dof_names=np.asarray(evaluator.dof_names_onnx),
        robot_root_pos=root_pos,
        robot_root_quat_xyzw=root_quat,
        smpl_semantic_position_aligned=np.asarray(
            evaluator._visual_smpl_target_positions, dtype=np.float32
        ),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def run_online_rollout(
    args: argparse.Namespace,
    clip: RawPicoClip,
    schedule: PicoReplaySchedule,
    clip_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    # The licensed neutral asset is intentionally not committed to the clean
    # repository.  Bind the explicitly supplied validation asset before any
    # shared SMPL FK builder is constructed.
    os.environ["HOLOMOTION_SMPL_MODEL_PATH"] = str(
        Path(args.smpl_model).expanduser().resolve()
    )
    pose_all, trans_all = body_poses_to_smpl_pose_trans(clip.body_poses)
    pose_all = np.asarray(pose_all, dtype=np.float32).reshape(-1, 72)
    trans_all = np.asarray(trans_all, dtype=np.float32).reshape(-1, 3)
    target_pose = pose_all[schedule.source_indices]
    target_trans = trans_all[schedule.source_indices]
    formal = _save_formal_target(
        clip_dir / "formal_target.npz", target_pose, target_trans, clip.path
    )

    evaluator = GmrFreeSmplMujocoEvaluator(build_config(args, formal))
    evaluator.setup()
    online = GpuOnlineSmplReference(
        n_future_frames=int(evaluator.n_fut_frames),
        policy_hz=float(schedule.policy_hz),
        device=f"cuda:{int(args.gpu_id)}",
        ground_history_frames=int(schedule.history_frames),
    )
    evaluator.smpl_reference = online
    for event_index in range(schedule.prefill_event_count):
        _store_online_event(online, clip, schedule, event_index)
    if not online.has_future_sequence():
        raise RuntimeError("GPU online SMPL ring did not become ready")

    evaluator.reset_state_teleport()
    publisher = BestEffortReferencePublisher(
        uri="inproc://offline-pico-replay",
        topic=TOPIC,
        logger=logger,
        publish_hz=schedule.policy_hz,
    )
    telemetry_pose: list[np.ndarray] = []
    telemetry_trans: list[np.ndarray] = []
    max_parity = 0.0
    for policy_frame in range(schedule.policy_frame_count):
        event_index = schedule.feed_event_index(policy_frame)
        if event_index is not None:
            _store_online_event(online, clip, schedule, event_index)
        latest_source = schedule.latest_source_index(policy_frame)
        payload = _materialize_wire_payload(
            publisher,
            clip,
            source_index=latest_source,
            sample_time=float(schedule.policy_sample_times[policy_frame]),
        )
        pose = np.asarray(payload["human_pose_aa"], dtype=np.float32).reshape(
            72
        )
        trans = np.asarray(
            payload["human_root_trans"], dtype=np.float32
        ).reshape(3)
        max_parity = max(
            max_parity,
            float(np.max(np.abs(pose - pose_all[latest_source]))),
            float(np.max(np.abs(trans - trans_all[latest_source]))),
        )
        telemetry_pose.append(pose)
        telemetry_trans.append(trans)

        evaluator.motion_frame_idx = policy_frame
        evaluator._update_policy()
        evaluator.counter += 1
        evaluator._apply_control(sleep=False)

    rollout_path = clip_dir / "online_rollout.npz"
    metadata = _save_rollout(
        rollout_path,
        evaluator=evaluator,
        clip=clip,
        schedule=schedule,
        telemetry_pose=np.stack(telemetry_pose),
        telemetry_trans=np.stack(telemetry_trans),
        telemetry_parity_max_abs=max_parity,
    )
    return rollout_path, metadata


def _draw_label(image: np.ndarray, title: str, detail: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        detail,
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def render_1x2(
    args: argparse.Namespace,
    clip: RawPicoClip,
    rollout_path: Path,
    output: Path,
) -> Path:
    with np.load(rollout_path, allow_pickle=False) as payload:
        telemetry_pose = np.asarray(payload["telemetry_human_pose_aa"])
        telemetry_trans = np.asarray(payload["telemetry_human_root_trans"])
        target_indices = np.asarray(payload["policy_target_source_index"])
        latest_indices = np.asarray(payload["telemetry_latest_source_index"])
        dof = np.asarray(payload["robot_dof_pos"], dtype=np.float64)
        dof_names = tuple(str(name) for name in payload["robot_dof_names"])
        root_pos = np.asarray(payload["robot_root_pos"], dtype=np.float64)
        root_quat = np.asarray(
            payload["robot_root_quat_xyzw"], dtype=np.float64
        )
        target = np.asarray(
            payload["smpl_semantic_position_aligned"], dtype=np.float64
        )
        metadata = json.loads(
            str(np.asarray(payload["metadata"]).reshape(()).item())
        )
    if int(metadata["robot_reference_reads"]) != 0:
        raise ValueError(
            "Pico rollout does not prove zero Robot-reference reads"
        )

    config = build_config(args, rollout_path.parent / "formal_target.npz")
    dof_names = validate_named_dof_matrix(
        dof, dof_names, tuple(str(name) for name in config.robot.dof_names)
    )
    model = mujoco.MjModel.from_xml_path(str(Path(args.robot_xml).resolve()))
    data = mujoco.MjData(model)
    qpos_addresses = []
    for name in dof_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise KeyError(f"MuJoCo joint not found: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint]))

    panel_width, panel_height = int(args.panel_width), int(args.panel_height)
    left_azimuth, right_azimuth = heading_relative_camera_azimuths(
        telemetry_pose[0, :3],
        root_quat[0],
        float(args.camera_azimuth),
    )
    stride = int(round(50.0 / float(args.video_fps)))
    if stride <= 0 or abs(50.0 / stride - float(args.video_fps)) > 1.0e-6:
        raise ValueError("video_fps must divide the 50 Hz policy rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    left_path = output.with_name(f".{output.stem}.left.tmp.mp4")
    right_path = output.with_name(f".{output.stem}.right.tmp.mp4")

    try:
        # Render each panel with its own renderer lifetime. MuJoCo's OpenGL
        # resources are context-local; keeping both contexts alive caused the
        # dynamic SMPL VBO to appear as a black mesh in the G1 panel.
        left = SmplMeshOffscreenRenderer(
            Path(args.smpl_model),
            width=panel_width,
            height=panel_height,
            distance=float(args.camera_distance),
            azimuth=left_azimuth,
            elevation=float(args.camera_elevation),
        )
        left.render(telemetry_pose[0], telemetry_trans[0])
        _disable_renderer_shadows(left.renderer)
        left.renderer.set_overlay_callback(_disable_scene_shadows)
        left_writer = FfmpegWriter(
            left_path,
            width=panel_width,
            height=panel_height,
            fps=int(args.video_fps),
            ffmpeg=args.ffmpeg,
        )
        try:
            for frame in range(0, dof.shape[0], stride):
                left_rgb = left.render(
                    telemetry_pose[frame], telemetry_trans[frame]
                )
                time_seconds = frame / 50.0
                _draw_label(
                    left_rgb,
                    "Deployment SMPL online viewer | raw Pico latest-only",
                    (
                        f"t={time_seconds:6.2f}s | "
                        f"source={int(latest_indices[frame])} | "
                        "heading-relative view"
                    ),
                )
                left_writer.write(left_rgb)
        finally:
            left.close()
            left_writer.close()

        right = OffscreenRenderer(
            model,
            height=panel_height,
            width=panel_width,
            distance=float(args.camera_distance),
            azimuth=right_azimuth,
            elevation=float(args.camera_elevation),
        )
        right.render(data)
        _disable_renderer_shadows(right)
        _show_robot_visual_geoms_only(right, model)
        current_markers = np.zeros((target.shape[1], 3), dtype=np.float32)

        def overlay(scene) -> None:
            _disable_scene_shadows(scene)
            size = np.asarray([0.025, 0.0, 0.0], dtype=np.float32)
            rotation = np.eye(3, dtype=np.float32).reshape(-1)
            color = np.asarray([0.85, 0.05, 0.05, 1.0], dtype=np.float32)
            first = int(scene.ngeom)
            for index, position in enumerate(current_markers):
                geom = first + index
                if geom >= scene.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scene.geoms[geom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size,
                    position,
                    rotation,
                    color,
                )
                scene.ngeom = geom + 1

        right.set_overlay_callback(overlay)
        right_writer = FfmpegWriter(
            right_path,
            width=panel_width,
            height=panel_height,
            fps=int(args.video_fps),
            ffmpeg=args.ffmpeg,
        )
        try:
            for frame in range(0, dof.shape[0], stride):
                data.qpos[:] = model.qpos0
                data.qvel[:] = 0.0
                data.qpos[:3] = root_pos[frame]
                xyzw = root_quat[frame]
                data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
                data.qpos[qpos_addresses] = dof[frame]
                mujoco.mj_forward(model, data)
                current_markers[:] = target[frame]
                lookat = root_pos[frame].copy()
                lookat[2] += 0.65
                right.set_align_view(lookat=lookat)
                right_rgb = np.ascontiguousarray(right.render(data))

                lag = float(
                    clip.timestamp_monotonic[int(latest_indices[frame])]
                    - clip.timestamp_monotonic[int(target_indices[frame])]
                )
                _draw_label(
                    right_rgb,
                    "GMR-free MuJoCo sim2sim | Canonical Default model_22000",
                    (
                        f"policy target={int(target_indices[frame])} | "
                        f"online lag={lag:.3f}s | heading-relative view"
                    ),
                )
                right_writer.write(right_rgb)
        finally:
            right.close()
            right_writer.close()

        subprocess.run(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(left_path),
                "-i",
                str(right_path),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[v]",
                "-map",
                "[v]",
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
    finally:
        left_path.unlink(missing_ok=True)
        right_path.unlink(missing_ok=True)
    return output


def run_clip(
    args: argparse.Namespace, raw_csv: Path, output_root: Path
) -> dict[str, Any]:
    slug = _slug(raw_csv)
    clip_dir = output_root / "work" / slug
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip = load_raw_pico_csv(raw_csv)
    schedule = build_latest_only_schedule(clip.timestamp_monotonic)
    logger.info(
        "Pico replay {}: raw_frames={}, policy_frames={}, duration={:.2f}s",
        slug,
        clip.frame_count,
        schedule.policy_frame_count,
        clip.duration_seconds,
    )
    rollout_path, rollout_metadata = run_online_rollout(
        args, clip, schedule, clip_dir
    )
    video = (
        output_root
        / "clips_1x2"
        / f"{slug}_SMPLOnline_MuJoCo_model22000_1x2.mp4"
    )
    render_1x2(args, clip, rollout_path, video)
    manifest = {
        "schema_version": "gmr_free_pico_online_visual_validation_v1",
        "raw_csv": str(raw_csv.resolve()),
        "raw_csv_sha256": sha256(raw_csv),
        "selection": "shortest complete clip in its immediate source folder",
        "video": str(video.resolve()),
        "video_sha256": sha256(video),
        "rollout": str(rollout_path.resolve()),
        "rollout_metadata": rollout_metadata,
        "layout": {
            "left": (
                "production CPU SMPL telemetry conversion and wire payload"
            ),
            "right": "GPU online SMPL Actor observation and MuJoCo G1",
        },
        "visualization_profile": VISUALIZATION_PROFILE,
        "camera": {
            "alignment": "same view relative to each subject initial heading",
            "cast_shadows": False,
        },
    }
    manifest_path = video.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Completed Pico replay video: {}", video)
    return manifest


def concat_videos(
    args: argparse.Namespace, videos: list[Path], output: Path
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output.parent / ".pico_sample10_concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in videos),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output),
            ],
            check=True,
        )
    finally:
        concat_file.unlink(missing_ok=True)
    return output


def validate_cpu_gpu_pico_parity(
    selected: tuple[Path, ...], *, gpu_id: int
) -> dict[str, Any]:
    """Compare deployment CPU/GPU converters on three frames per clip."""

    converter = PicoToSmplGpuConverter(f"cuda:{gpu_id}")
    pose_max = 0.0
    trans_max = 0.0
    checked = 0
    for path in selected:
        clip = load_raw_pico_csv(path)
        for index in (0, clip.frame_count // 2, clip.frame_count - 1):
            cpu_pose, cpu_trans = body_poses_to_smpl_pose_trans(
                clip.body_poses[index : index + 1]
            )
            gpu_pose, gpu_trans = converter(clip.body_poses[index])
            gpu_pose_np = gpu_pose.detach().cpu().numpy()
            gpu_trans_np = gpu_trans.detach().cpu().numpy()
            pose_max = max(
                pose_max,
                float(np.max(np.abs(cpu_pose - gpu_pose_np))),
            )
            trans_max = max(
                trans_max,
                float(np.max(np.abs(cpu_trans - gpu_trans_np))),
            )
            checked += 1
    return {
        "checked_frames": checked,
        "pose_max_abs": pose_max,
        "trans_max_abs": trans_max,
        "all_finite": bool(np.isfinite(pose_max) and np.isfinite(trans_max)),
    }


def run_batch(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = select_shortest_per_source(args.raw_root)
    if len(selected) != int(args.sample_count):
        raise ValueError(
            f"Expected {args.sample_count} source folders, "
            f"selected {len(selected)}"
        )
    selection = []
    for path in selected:
        timestamps = np.loadtxt(
            path, delimiter=",", skiprows=1, usecols=(1,), dtype=np.float64
        )
        selection.append(
            {
                "source": path.parent.name,
                "raw_csv": str(path.resolve()),
                "duration_seconds": float(timestamps[-1] - timestamps[0]),
            }
        )
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifests = []
    videos = []
    for index, raw_csv in enumerate(selected, start=1):
        logger.info(
            "Running sampled Pico clip {}/{}: {}",
            index,
            len(selected),
            raw_csv,
        )
        expected_video = (
            output_root
            / "clips_1x2"
            / f"{_slug(raw_csv)}_SMPLOnline_MuJoCo_model22000_1x2.mp4"
        )
        expected_manifest = expected_video.with_suffix(".manifest.json")
        if expected_video.is_file() and expected_manifest.is_file():
            manifest = json.loads(
                expected_manifest.read_text(encoding="utf-8")
            )
            if int(manifest["rollout_metadata"]["robot_reference_reads"]) != 0:
                raise ValueError(
                    f"Existing Pico validation is invalid: {expected_video}"
                )
            rollout_path = Path(manifest["rollout"])
            if not rollout_path.is_file():
                raise FileNotFoundError(
                    f"Existing Pico rollout is missing: {rollout_path}"
                )
            if manifest.get("visualization_profile") != VISUALIZATION_PROFILE:
                logger.info(
                    "Re-rendering existing rollout with profile {}: {}",
                    VISUALIZATION_PROFILE,
                    expected_video,
                )
                clip = load_raw_pico_csv(raw_csv)
                render_1x2(args, clip, rollout_path, expected_video)
                manifest["video_sha256"] = sha256(expected_video)
                manifest["visualization_profile"] = VISUALIZATION_PROFILE
                manifest["camera"] = {
                    "alignment": (
                        "same view relative to each subject initial heading"
                    ),
                    "cast_shadows": False,
                }
                expected_manifest.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif manifest["video_sha256"] != sha256(expected_video):
                raise ValueError(
                    f"Existing Pico video hash mismatch: {expected_video}"
                )
            else:
                logger.info(
                    "Reusing verified completed clip: {}", expected_video
                )
        else:
            manifest = run_clip(args, raw_csv, output_root)
        manifests.append(manifest)
        videos.append(Path(manifest["video"]))

    combined = (
        output_root
        / "pico0402_sample10_SMPLOnline_MuJoCoSim2Sim_model22000_1x2.mp4"
    )
    concat_videos(args, videos, combined)
    batch_manifest = {
        "schema_version": "gmr_free_pico0402_sample10_batch_v1",
        "selection_rule": (
            "one shortest complete clip per immediate source folder"
        ),
        "clip_count": len(manifests),
        "total_raw_duration_seconds": float(
            sum(item["duration_seconds"] for item in selection)
        ),
        "combined_video": str(combined),
        "combined_video_sha256": sha256(combined),
        "assets": {
            "motion_policy_onnx": str(Path(args.onnx).resolve()),
            "motion_policy_onnx_sha256": sha256(args.onnx),
            "velocity_policy_onnx": str(Path(args.velocity_onnx).resolve()),
            "velocity_policy_onnx_sha256": sha256(args.velocity_onnx),
            "robot_xml": str(Path(args.robot_xml).resolve()),
            "robot_xml_sha256": sha256(args.robot_xml),
            "smpl_model": str(Path(args.smpl_model).resolve()),
            "smpl_model_sha256": sha256(args.smpl_model),
        },
        "robot_reference_reads": 0,
        "real_robot_commands": 0,
        "visualization_profile": VISUALIZATION_PROFILE,
        "camera": {
            "alignment": "same view relative to each subject initial heading",
            "cast_shadows": False,
        },
        "pico_cpu_gpu_converter_parity": validate_cpu_gpu_pico_parity(
            selected, gpu_id=int(args.gpu_id)
        ),
        "code": {
            "repo": str(REPO_ROOT),
            "branch": _git_value("branch", "--show-current"),
            "head": _git_value("rev-parse", "HEAD"),
            "validation_script": str(Path(__file__).resolve()),
            "validation_script_sha256": sha256(Path(__file__).resolve()),
            "scheduler": str(
                (
                    REPO_ROOT
                    / "holomotion/src/evaluation/pico_online_replay.py"
                ).resolve()
            ),
            "scheduler_sha256": sha256(
                REPO_ROOT / "holomotion/src/evaluation/pico_online_replay.py"
            ),
        },
        "clips": manifests,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Combined sampled Pico validation video: {}", combined)
    return combined


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("sample", "clip", "batch"))
    result.add_argument("--raw-root", required=True)
    result.add_argument("--raw-csv")
    result.add_argument("--output-root", required=True)
    result.add_argument("--sample-count", type=int, default=10)
    result.add_argument("--onnx", required=True)
    result.add_argument("--velocity-onnx", required=True)
    result.add_argument("--velocity-config", required=True)
    result.add_argument("--robot-xml", required=True)
    result.add_argument("--smpl-model", required=True)
    result.add_argument("--gpu-id", type=int, default=0)
    result.add_argument("--settle-seconds", type=float, default=1.0)
    result.add_argument("--velocity-warmup-seconds", type=float, default=1.0)
    result.add_argument("--video-fps", type=int, default=25)
    result.add_argument("--panel-width", type=int, default=1280)
    result.add_argument("--panel-height", type=int, default=720)
    result.add_argument("--camera-distance", type=float, default=4.5)
    result.add_argument("--camera-azimuth", type=float, default=150.0)
    result.add_argument("--camera-elevation", type=float, default=-18.0)
    result.add_argument("--ffmpeg", default=_default_ffmpeg())
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "sample":
        selected = select_shortest_per_source(args.raw_root)
        for path in selected:
            timestamps = np.loadtxt(
                path, delimiter=",", skiprows=1, usecols=(1,), dtype=np.float64
            )
            print(
                f"{path.parent.name}\t"
                f"{timestamps[-1] - timestamps[0]:.3f}s\t{path}"
            )
        return
    if args.command == "clip":
        if args.raw_csv is None:
            raise SystemExit("clip requires --raw-csv")
        run_clip(
            args,
            Path(args.raw_csv).expanduser().resolve(),
            Path(args.output_root).expanduser().resolve(),
        )
        return
    run_batch(args)


if __name__ == "__main__":
    main()
