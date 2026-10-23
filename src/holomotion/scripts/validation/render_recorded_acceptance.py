"""Render recorded Isaac states; never simulate a new policy trajectory.

Every frozen case is retained, including terminal/failure frames. Videos are
explicitly labeled as Isaac state replay rendered with shared MuJoCo assets.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from omegaconf import OmegaConf

from deployment.holomotion_shadow.offscreen_renderer import OffscreenRenderer
from holomotion.src.evaluation.acceptance_mujoco import file_sha256
from holomotion.src.evaluation.behavior_acceptance import velocity_command_at
from holomotion.src.evaluation.offline_eval_manifest import (
    validate_fixed_case_manifest,
)


def render_recording(directory: Path, output: Path):
    report = json.loads((directory / "result.json").read_text())
    if report["backend"] != "isaaclab":
        raise ValueError(
            "This replay entry point requires actual Isaac traces"
        )
    manifest = json.loads((directory / "cases_manifest.json").read_text())
    validate_fixed_case_manifest(manifest)
    cfg = OmegaConf.load(directory / "resolved_evaluation_config.yaml")
    root = Path(__file__).resolve().parents[3]
    xml = (root / cfg.robot.asset.assetFileName).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Replay requires the configured floating root")
    output.mkdir(parents=True, exist_ok=False)
    renderer = OffscreenRenderer(
        model, height=540, width=960, distance=3.2, azimuth=140, elevation=-15
    )
    results = []
    try:
        for case in manifest["cases"]:
            path = directory / case["output_filename"]
            with np.load(path, allow_pickle=False) as trace:
                names = trace["joint_names"].tolist()
                q = trace["q"]
                root_pos, root_quat = (
                    trace["root_position"],
                    trace["root_quaternion"],
                )
                metadata_key = (
                    "metadata_json" if "metadata_json" in trace else "metadata"
                )
                metadata = json.loads(str(trace[metadata_key].item()))
            if len(names) != len(set(names)) or set(names) != set(
                cfg.robot.dof_names
            ):
                raise ValueError(
                    "Recorded joint names differ from training asset"
                )
            if q.shape != (len(root_pos), len(names)) or root_quat.shape != (
                len(q),
                4,
            ):
                raise ValueError("Recorded root/joint frame counts differ")
            if (
                not len(q)
                or len(q) > case["planned_num_frames"]
                or not all(
                    np.isfinite(x).all() for x in (q, root_pos, root_quat)
                )
            ):
                raise ValueError("Recorded trajectory is invalid")
            indices = []
            for name in names:
                joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                if (
                    joint < 0
                    or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                ):
                    raise ValueError(f"Replay joint not found: {name}")
                indices.append(int(model.jnt_qposadr[joint]))
            video = output / (case["case_id"] + ".mp4")
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                1 / case["policy_dt_s"],
                (960, 540),
            )
            if not writer.isOpened():
                raise RuntimeError("Recorded-state video writer failed")
            try:
                for frame in range(len(q)):
                    data.qpos[:3], data.qpos[3:7] = (
                        root_pos[frame],
                        root_quat[frame],
                    )
                    data.qpos[indices] = q[frame]
                    # Kinematics only. mj_step/mj_forward and policy inference
                    # are intentionally absent from this replay tool.
                    mujoco.mj_kinematics(model, data)
                    renderer.set_align_view(
                        lookat=root_pos[frame] + [0, 0, 0.15]
                    )
                    image = cv2.cvtColor(
                        renderer.render(data), cv2.COLOR_RGB2BGR
                    )
                    lines = [
                        "ISAAC STATE REPLAY | MuJoCo rendering only",
                        f"seed={case['seed']}  "
                        f"t={(frame + 1) * case['policy_dt_s']:.2f}s",
                    ]
                    if "segments" in case:
                        command = velocity_command_at(case, frame).tolist()
                        lines.append(f"command vx,vy,yaw={command}")
                    if metadata.get("failure_reasons") and frame == len(q) - 1:
                        lines.append(
                            "FAILED: " + ",".join(metadata["failure_reasons"])
                        )
                    for row, line in enumerate(lines):
                        cv2.putText(
                            image,
                            line,
                            (12, 24 + row * 23),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 255, 255),
                            1,
                        )
                    writer.write(image)
            finally:
                writer.release()
            results.append(
                {
                    "case_id": case["case_id"],
                    "frames": len(q),
                    "planned_frames": case["planned_num_frames"],
                    "source_sha256": file_sha256(path),
                    "video": str(video),
                    "video_sha256": file_sha256(video),
                }
            )
            print("RENDERED_ISAAC_STATE", case["case_id"], len(q), flush=True)
    finally:
        renderer.close()
    receipt = {
        "scope": "Isaac state replay, not a MuJoCo rollout or new evaluation",
        "source_result_sha256": file_sha256(directory / "result.json"),
        "asset_xml_sha256": file_sha256(xml),
        "policy_steps": 0,
        "physics_steps": 0,
        "cases": results,
    }
    (output / "render_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_recording(args.trace_directory.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
