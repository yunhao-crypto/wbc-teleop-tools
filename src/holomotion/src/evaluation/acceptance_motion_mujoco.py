"""Frozen Motion cases through the existing pure-SMPL MuJoCo evaluator."""

from __future__ import annotations
import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from holomotion.src.evaluation.acceptance_motion import (
    anchored_smpl_arrays,
    semantic_arrays,
    validate_motion_manifest,
    write_motion_case,
)
from holomotion.src.evaluation.acceptance_mujoco import (
    file_sha256,
    ground_normal_forces,
    refresh_kinematics,
)
from holomotion.src.evaluation.behavior_acceptance import (
    judge_motion_group,
    motion_failure_reasons,
)
from holomotion.src.evaluation.gmr_free_smpl_metrics import aggregate_directory


def run_motion_acceptance(
    *, model_path, manifest, profile, stage, output, record_video=True
):
    import torch

    from holomotion.src.evaluation.eval_mujoco_sim2sim import (
        OffscreenRenderer,
        process_config,
    )
    from holomotion.src.evaluation.gmr_free_smpl_mujoco import (
        GmrFreeSmplMujocoEvaluator,
    )
    from holomotion.src.motion_tracking.robot_skeleton_adapter import (
        RobotSkeletonAdapterSpec,
        build_robot_skeleton_state,
    )

    root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(
        root / "holomotion/config/evaluation/eval_mujoco_sim2sim.yaml"
    )
    packaged = OmegaConf.load(model_path.parent.parent / "config.yaml")
    cfg.ckpt_onnx_path = str(model_path)
    cfg.robot_xml_path = str(
        (root / packaged.robot.asset.assetFileName).resolve()
    )
    # The existing pure-SMPL observation exposes a CUDA buffer directly.
    cfg.headless, cfg.use_gpu, cfg.record_video = True, True, False
    cfg.mujoco_actuator_model = "auto"
    cfg.policy_action_delay_step = 0
    cfg.mujoco_actuator_delay = True
    cfg.motion_npz_path = manifest["cases"][0]["source_npz_path"]
    cfg = process_config(cfg)
    cfg.gmr_free_reset_profile = "training_canonical_default"
    cfg.gmr_free_settle_seconds = 0
    cfg.gmr_free_velocity_warmup_seconds = 0
    dt = (
        cfg.env.config.simulation.control_decimation
        / cfg.env.config.simulation.sim_freq
    )
    clips = validate_motion_manifest(
        manifest, profile=profile, stage=stage, policy_dt=dt
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "resolved_evaluation_config.yaml").write_text(
        OmegaConf.to_yaml(cfg)
    )
    ev = GmrFreeSmplMujocoEvaluator(cfg)
    ev.setup()
    if "CUDAExecutionProvider" not in ev.policy_session.get_providers():
        raise RuntimeError(
            "Pure-SMPL CUDA observation needs ONNX CUDA provider"
        )
    spec = RobotSkeletonAdapterSpec.from_mapping(
        OmegaConf.to_container(cfg.robot.motion.semantic_skeleton)
    )
    names = list(cfg.robot.body_names)
    body_indices = {name: index for index, name in enumerate(names)}

    def robot_state():
        return build_robot_skeleton_state(
            spec=spec,
            body_name_to_index=body_indices,
            body_position_w=torch.tensor(
                ev.robot_global_bodylink_pos[None], dtype=torch.float32
            ),
            body_rotation_wxyz=torch.tensor(
                ev.robot_global_bodylink_rot[None], dtype=torch.float32
            ),
            root_linear_velocity_w=torch.tensor(
                ev.robot_global_bodylink_lin_vel[None, ev.root_body_idx],
                dtype=torch.float32,
            ),
            root_angular_velocity_w=torch.tensor(
                ev.robot_global_bodylink_ang_vel[None, ev.root_body_idx],
                dtype=torch.float32,
            ),
        )

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
            if str(cfg.motion_npz_path) != case["source_npz_path"]:
                cfg.motion_npz_path = case["source_npz_path"]
                ev.load_motion_data()
            ev._training_actuator_rng = np.random.default_rng(case["seed"])
            ev.reset_state_teleport()
            # Pure reset clears policy KV, not the physics-delay ring.
            # Reset the existing controller for each independent case.
            ev._reset_training_actuator_state()
            refresh_kinematics(ev.m, ev.d)
            target = anchored_smpl_arrays(
                clips[Path(case["source_npz_path"])],
                root_position=ev.d.qpos[:3],
                root_quaternion_wxyz=ev.d.qpos[3:7],
                grounded_root_height=ev.smpl_reference_cpu.root_pos[:, 2],
            )
            rows, physical, qpos, actions = [], [], [], []
            failure = []
            first_failure_s = None
            cursor = {"target_index": 0}
            pre_qpos = ev.d.qpos.copy()

            def observe(
                current,
                physical=physical,
                target=target,
                case=case,
                cursor=cursor,
            ):
                nonlocal failure, first_failure_s, pre_qpos
                forces = ground_normal_forces(current.m, current.d)
                physical.append(
                    {
                        "pre_qpos": pre_qpos.copy(),
                        "qpos": current.d.qpos.copy(),
                        "qvel": current.d.qvel.copy(),
                        "torque": current.d.qfrc_actuator[
                            current.actuator_qvel_indices
                        ].copy(),
                        "ground": sum(forces.values()),
                    }
                )
                pre_qpos = current.d.qpos.copy()
                refresh_kinematics(current.m, current.d)
                reasons = motion_failure_reasons(
                    target_torso_wxyz=target["target_torso_rotation_wxyz"][
                        cursor["target_index"]
                    ],
                    robot_torso_wxyz=robot_state()
                    .torso_rotation_wxyz[0]
                    .numpy(),
                    threshold=case["failure_rules"][
                        "torso_gravity_z_difference_max"
                    ],
                )
                if reasons and first_failure_s is None:
                    first_failure_s = len(physical) * current.simulation_dt
                failure = list(dict.fromkeys(failure + reasons))

            ev.physics_step_observer = observe
            writer = None
            if renderer is not None:
                import cv2

                writer = cv2.VideoWriter(
                    str(output / (case["case_id"] + ".mp4")),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1 / dt,
                    (960, 540),
                )
                if not writer.isOpened():
                    raise RuntimeError("Motion video writer failed")
            try:
                for step in range(case["planned_num_frames"]):
                    ev.motion_frame_idx = step
                    target_index = min(
                        step + 1, case["planned_num_frames"] - 1
                    )
                    cursor["target_index"] = target_index
                    refresh_kinematics(ev.m, ev.d)
                    ev._update_policy()
                    ev.counter += 1
                    ev._apply_control(sleep=False)
                    refresh_kinematics(ev.m, ev.d)
                    rows.append(
                        {
                            **{
                                key: value[target_index].copy()
                                for key, value in target.items()
                            },
                            **semantic_arrays(
                                robot_state(), prefix="robot", index=0
                            ),
                        }
                    )
                    qpos.append(ev.d.qpos.copy())
                    actions.append(ev.actions_onnx.copy())
                    if writer is not None:
                        renderer.set_align_view(
                            lookat=ev.d.qpos[:3] + [0, 0, 0.15]
                        )
                        frame = cv2.cvtColor(
                            renderer.render(ev.d), cv2.COLOR_RGB2BGR
                        )
                        cv2.putText(
                            frame,
                            f"Motion | t={(step + 1) * dt:.2f}s | "
                            f"seed={case['seed']}",
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                        )
                        if failure:
                            cv2.putText(
                                frame,
                                "FAILED: " + ",".join(failure),
                                (12, 48),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 0, 255),
                                1,
                            )
                        writer.write(frame)
                    if failure:
                        break
            finally:
                if writer is not None:
                    writer.release()
            result = write_motion_case(
                output,
                case,
                rows,
                completed=len(rows) - bool(failure),
                failure=failure,
                raw={
                    "robot_qpos": np.stack(qpos),
                    "robot_actions": np.stack(actions),
                    "physics_pre_qpos": np.stack(
                        [p["pre_qpos"] for p in physical]
                    ),
                    "physics_qpos": np.stack([p["qpos"] for p in physical]),
                    "physics_qvel": np.stack([p["qvel"] for p in physical]),
                    "physics_control_torque": np.stack(
                        [p["torque"] for p in physical]
                    ),
                    "physics_ground_normal_force_n": np.array(
                        [p["ground"] for p in physical]
                    ),
                    "robot_dof_names": np.array(ev.mjcf_dof_names),
                },
                extra={
                    "first_failure_physics_time_s": first_failure_s,
                    "actuator": ev._mujoco_actuator_receipt(),
                },
            )
            results.append(result)
            (output / "cases.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                "MOTION_BEHAVIOR_CASE",
                case["case_id"],
                result["completion_fraction"],
                failure,
                flush=True,
            )
    finally:
        ev.physics_step_observer = None
        if renderer is not None:
            renderer.close()
    summary = aggregate_directory(output, expected_manifest=manifest)
    report = {
        "backend": "mujoco",
        "version": mujoco.__version__,
        "model_sha256": file_sha256(model_path),
        "training_config_sha256": file_sha256(
            model_path.parent.parent / "config.yaml"
        ),
        "physics": ev._joint_physics_receipt,
        "policy_io": ev._policy_io_receipt,
        "robot_reference_reads": 0,
        "cases": results,
        "summary": summary,
        "gate": judge_motion_group(summary, profile=profile, stage=stage),
        "no_hardware_output": True,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
