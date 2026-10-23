"""Fixed-case acceptance adapter around the shared MuJoCo policy/controller.

No changes to robot geometry or policy inference. Ground contacts are collected
at physics rate before any kinematic-only refresh; failures never teleport.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from holomotion.src.evaluation.behavior_acceptance import (
    VelocityFailureMonitor,
    heading_velocity_command_frame,
    judge_velocity_cases,
    velocity_case_manifest,
    velocity_case_metrics,
    velocity_command_at,
)
from holomotion.src.evaluation.eval_mujoco_sim2sim import (
    MujocoEvaluator,
    OffscreenRenderer,
    process_config,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_kinematics(model, data) -> None:
    """Refresh link frames/velocities to post-integration qpos/qvel.

    mj_step integrates qpos/qvel after computing derived link state. Do not use
    mj_forward here: it would replace the actual contact/force sample.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)


def ground_normal_forces(model, data) -> dict[int, float]:
    """Actual ground-normal force by body, excluding robot self-contact."""
    result = {}
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        left, right = (int(model.geom_bodyid[g]) for g in contact.geom)
        if (left == 0) == (right == 0):
            continue
        floor_geom = contact.geom[0 if left == 0 else 1]
        if model.geom_type[floor_geom] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError(
                "Nominal acceptance requires a world-attached plane"
            )
        body = right if left == 0 else left
        mujoco.mj_contactForce(model, data, index, force)
        result[body] = result.get(body, 0.0) + abs(float(force[0]))
    return result


def run_velocity_acceptance(
    *,
    model_path: Path,
    manifest: dict,
    profile: dict,
    stage: str,
    output: Path,
    record_video: bool = True,
) -> dict:
    root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(
        root / "holomotion/config/evaluation/eval_mujoco_sim2sim.yaml"
    )
    cfg.ckpt_onnx_path = str(model_path.resolve())
    packaged = OmegaConf.load(model_path.parent.parent / "config.yaml")
    cfg.robot_xml_path = str(
        (root / packaged.robot.asset.assetFileName).resolve()
    )
    cfg.headless = True
    cfg.use_gpu = False
    cfg.record_video = False
    cfg.velocity_reset_profile = "training_init"
    cfg.mujoco_actuator_model = "auto"
    cfg.policy_action_delay_step = 0
    cfg.mujoco_actuator_delay = True
    cfg.velocity_command = [0.0, 0.0, 0.0]
    cfg = process_config(cfg)
    ev = MujocoEvaluator(cfg)
    ev.setup()
    if ev.command_mode != "velocity_tracking":
        raise ValueError("Velocity acceptance received a non-Velocity model")
    if manifest != velocity_case_manifest(
        profile, stage=stage, policy_dt=ev.policy_dt
    ):
        raise ValueError(
            "Manifest differs from the frozen profile command program"
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_evaluation_config.yaml").write_text(
        OmegaConf.to_yaml(cfg)
    )
    (output / "cases_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    receipt = {
        "backend": "mujoco",
        "version": mujoco.__version__,
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "training_config_sha256": file_sha256(
            model_path.parent.parent / "config.yaml"
        ),
        "mjcf_sha256": file_sha256(Path(cfg.robot_xml_path)),
        "physics": ev._joint_physics_receipt,
        "policy_io": ev._policy_io_receipt,
        "state_sampling": (
            "post-integration qpos/qvel and kinematic-only link refresh"
        ),
        "contact_sampling": (
            "last physics-step normal force, before refresh, world plane only"
        ),
        "no_hardware_output": True,
    }
    renderer = (
        OffscreenRenderer(
            ev.m,
            height=540,
            width=960,
            distance=3.2,
            azimuth=140,
            elevation=-15,
        )
        if record_video
        else None
    )
    results = []
    try:
        for case in manifest["cases"]:
            ev._training_actuator_rng = np.random.default_rng(case["seed"])
            ev.reset_state_teleport()
            ev.obs_builder.reset()
            ev.actions_onnx.fill(0)
            initial = ev.d.qpos[:3].copy()
            allowed = {
                mujoco.mj_name2id(ev.m, mujoco.mjtObj.mjOBJ_BODY, str(name))
                for side in ev._normalize_foot_body_name_groups(
                    cfg.robot.feet_body_names
                )
                for name in side
            }
            if -1 in allowed or not allowed:
                raise ValueError("Every allowed ground body must exist")
            monitor = VelocityFailureMonitor(case["failure_rules"])
            physics_rows = []
            states = []
            velocities = []
            actions = []
            targets = []
            commands = []
            failure = []

            def observe(
                current,
                allowed=allowed,
                monitor=monitor,
                physics_rows=physics_rows,
            ):
                nonlocal failure
                forces = ground_normal_forces(current.m, current.d)
                forbidden = sum(
                    force
                    for body, force in forces.items()
                    if body not in allowed
                )
                reasons = monitor.observe(
                    root_position=current.d.qpos[:3],
                    root_quaternion_wxyz=current.d.qpos[3:7],
                    forbidden_ground_normal_force_n=forbidden,
                    dt_s=current.simulation_dt,
                )
                failure = list(dict.fromkeys(failure + reasons))
                physics_rows.append(
                    {
                        "qpos": current.d.qpos.copy(),
                        "qvel": current.d.qvel.copy(),
                        "torque": current.d.qfrc_actuator[
                            current.actuator_qvel_indices
                        ].copy(),
                        "forbidden_ground_normal_force_n": forbidden,
                    }
                )

            ev.physics_step_observer = observe
            writer = None
            if renderer is not None:
                import cv2

                writer = cv2.VideoWriter(
                    str(output / (case["case_id"] + ".mp4")),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1 / ev.policy_dt,
                    (960, 540),
                )
                if not writer.isOpened():
                    raise RuntimeError("Video writer failed")
            try:
                for step in range(case["planned_num_frames"]):
                    command = velocity_command_at(case, step)
                    ev.config.velocity_command = command.tolist()
                    refresh_kinematics(ev.m, ev.d)
                    ev._update_policy()
                    if not np.isfinite(ev.actions_onnx).all():
                        raise ValueError("Nonfinite policy action")
                    ev.counter += 1
                    ev._apply_control(sleep=False)
                    refresh_kinematics(ev.m, ev.d)
                    q = ev.d.qpos[3:7].copy()
                    angular_world = np.zeros(3)
                    mujoco.mju_rotVecQuat(angular_world, ev.d.qvel[3:6], q)
                    velocities.append(
                        heading_velocity_command_frame(
                            ev.d.qvel[:3], angular_world, q
                        )
                    )
                    states.append(ev.d.qpos.copy())
                    actions.append(ev.actions_onnx.copy())
                    targets.append(ev.target_dof_pos_mu.copy())
                    commands.append(command)
                    if writer is not None:
                        renderer.set_align_view(
                            lookat=ev.d.qpos[:3] + [0, 0, 0.15]
                        )
                        frame = cv2.cvtColor(
                            renderer.render(ev.d), cv2.COLOR_RGB2BGR
                        )
                        cv2.putText(
                            frame,
                            (
                                f"{case['case_id']}  "
                                f"t={(step + 1) * ev.policy_dt:.2f}s"
                            ),
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.46,
                            (255, 255, 255),
                            1,
                        )
                        if failure:
                            cv2.putText(
                                frame,
                                "FAILED: " + ",".join(failure),
                                (12, 48),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 0, 255),
                                1,
                            )
                        writer.write(frame)
                    if failure:
                        break
            finally:
                if writer is not None:
                    writer.release()
            states = np.asarray(states)
            metrics = velocity_case_metrics(
                case,
                root_position=states[:, :3],
                root_velocity_local=velocities,
                completed_intervals=len(states) - int(bool(failure)),
                failure_reasons=failure,
            )
            metrics["reset_relative_drift_max_m"] = float(
                np.linalg.norm(states[:, :2] - initial[:2], axis=1).max()
            )
            metrics["failure_time_s"] = (
                len(physics_rows) * ev.simulation_dt if failure else None
            )
            metrics["actuator"] = ev._mujoco_actuator_receipt()
            torque = np.stack([r["torque"] for r in physics_rows])
            limits = np.asarray(metrics["actuator"]["effort_limit"])
            metrics["outer_joint_torque_saturation_fraction"] = float(
                np.mean(
                    np.abs(torque)
                    >= profile["behavior"][
                        "torque_saturation_fraction_of_declared_limit"
                    ]
                    * limits
                )
            )
            metrics["motor_space_saturation_status"] = (
                "not extracted; outer joint statistic is not motor saturation"
            )
            np.savez_compressed(
                output / case["output_filename"],
                schema_version="velocity_acceptance_v1",
                qpos=states,
                velocity_command_frame=np.asarray(velocities),
                action=np.asarray(actions),
                control_target=np.asarray(targets),
                command=np.asarray(commands),
                physics_qpos=np.stack([r["qpos"] for r in physics_rows]),
                physics_qvel=np.stack([r["qvel"] for r in physics_rows]),
                physics_torque=torque,
                physics_forbidden_ground_normal_force_n=np.asarray(
                    [
                        r["forbidden_ground_normal_force_n"]
                        for r in physics_rows
                    ]
                ),
                metadata_json=json.dumps(metrics),
            )
            results.append(metrics)
            (output / "cases.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                "BEHAVIOR_CASE",
                case["case_id"],
                metrics["completion_fraction"],
                failure,
                flush=True,
            )
    finally:
        ev.physics_step_observer = None
        if renderer is not None:
            renderer.close()
    report = {
        **receipt,
        "gate": judge_velocity_cases(
            manifest, results, profile=profile, stage=stage
        ),
        "cases": results,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
