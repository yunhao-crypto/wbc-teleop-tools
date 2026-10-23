"""Fixed cases using real Isaac managers, shared actor observations and ONNX.

Only this disposable evaluation process overrides commands, DR and automatic
termination. Physics-rate frozen failure rules retain terminal states.
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from holomotion.src.evaluation.acceptance_mujoco import file_sha256
from holomotion.src.evaluation.behavior_acceptance import (
    VelocityFailureMonitor,
    heading_velocity_command_frame,
    judge_velocity_cases,
    velocity_case_manifest,
    velocity_case_metrics,
    velocity_command_at,
)


def run_velocity_acceptance(
    *,
    model_path: Path,
    manifest: dict,
    profile: dict,
    stage: str,
    output: Path,
    record_video: bool = True,
):
    import holomotion.src.motion_tracking.smpl_actor_observation  # noqa: F401
    from holomotion.src.evaluation.eval_mujoco_sim2sim import (
        MujocoEvaluator,
        process_config,
    )

    cfg = process_config(
        OmegaConf.create({"ckpt_onnx_path": str(model_path.resolve())})
    )
    OmegaConf.set_struct(cfg, False)
    cases = manifest["cases"]
    count = len(cases)
    policy_dt = (
        cfg.env.config.simulation.control_decimation
        / cfg.env.config.simulation.sim_freq
    )
    if manifest != velocity_case_manifest(
        profile, stage=stage, policy_dt=policy_dt
    ):
        raise ValueError("Manifest differs from frozen command program")
    output.mkdir(parents=True, exist_ok=False)
    cfg.num_envs = cfg.env.config.num_envs = count
    cfg.env.config.seed = profile["seeds"]["train"]
    cfg.env.config.env_spacing = 10.0
    cfg.env.config.terrain.terrain_type = "plane"
    cfg.env.config.domain_rand = {}
    cfg.env.config.curriculum = {}
    cfg.env.config.obs.obs_groups.unified.enable_corruption = False
    cfg.env.config.terminations = {}
    cfg.env.config.simulation.episode_length_s = 3600
    cfg.env.config.commands.base_velocity.params.debug_vis = False
    cfg.env.config.commands.base_velocity.params.heading_command = False
    (output / "resolved_evaluation_config.yaml").write_text(
        OmegaConf.to_yaml(cfg)
    )
    (output / "cases_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    cfg.ckpt_onnx_path = str(model_path.resolve())
    cfg.use_gpu = False
    cfg.robot_xml_path = cfg.robot.asset.assetFileName
    policy = MujocoEvaluator.__new__(MujocoEvaluator)
    policy.config = cfg
    policy.max_context_len = 0
    policy.load_policy()
    policy._apply_onnx_metadata()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        device="cuda:0",
        enable_cameras=False,
        kit_args=(
            "--/renderer/multiGpu/enabled=false "
            "--/renderer/multiGpu/autoEnable=false"
        ),
    )
    wrapper = None
    try:
        import torch
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationContext
        from tensordict import TensorDict

        from holomotion.src.env import velocity_tracking as environment_module
        from holomotion.src.modules.agent_modules import TensorDictAssembler

        original_build = environment_module.build_scene_config
        body_names = list(cfg.robot.body_names)

        class ResolvedPlaneContactSensor(ContactSensor):
            """PhysX GPU filters require collider paths, not parents."""

            def _initialize_impl(self):
                if not hasattr(self, "_acceptance_ground_root"):
                    self._acceptance_ground_root = (
                        self.cfg.filter_prim_paths_expr[0]
                    )
                ground_root = self._acceptance_ground_root
                stage = SimulationContext.instance().stage
                from pxr import UsdPhysics

                planes = [
                    str(prim.GetPath())
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith(ground_root + "/")
                    and prim.GetTypeName() == "Plane"
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                if len(planes) != 1:
                    raise ValueError(
                        f"Expected exactly one ground Plane collider: {planes}"
                    )
                self.cfg.filter_prim_paths_expr = planes
                super()._initialize_impl()

        def evaluation_scene(*args, **kwargs):
            scene = original_build(*args, **kwargs)
            for index, name in enumerate(body_names):
                setattr(
                    scene,
                    f"accept_ground_{index}",
                    ContactSensorCfg(
                        class_type=ResolvedPlaneContactSensor,
                        prim_path="{ENV_REGEX_NS}/Robot/" + name,
                        filter_prim_paths_expr=[str(scene.terrain.prim_path)],
                        update_period=0,
                        history_length=0,
                        track_air_time=False,
                    ),
                )
            return scene

        environment_module.build_scene_config = evaluation_scene
        try:
            wrapper = environment_module.VelocityTrackingEnv(
                config=cfg.env.config,
                device="cuda:0",
                log_dir=str(output),
                headless=True,
            )
        finally:
            environment_module.build_scene_config = original_build
        env = wrapper._env
        robot = env.scene["robot"]
        device = env.device
        if env.termination_manager.active_terms:
            raise AssertionError("Acceptance must not auto-reset")
        wrapper.reset_all()
        action_term = env.action_manager.get_term("dof_pos")
        if list(action_term._joint_names) != list(
            policy.action_dof_names_onnx
        ):
            raise ValueError(
                "Isaac action order differs from actual ONNX metadata"
            )
        origins = env.scene.env_origins.cpu().numpy().copy()
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins
        root_state[:, 3:7] = torch.tensor([1, 0, 0, 0], device=device)
        root_state[:, 7:] = 0
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            torch.zeros_like(robot.data.default_joint_pos),
        )
        lag_receipt = {}
        rngs = [np.random.default_rng(case["seed"]) for case in cases]
        for name, actuator in robot.actuators.items():
            if not hasattr(actuator, "positions_delay_buffer"):
                raise ValueError(
                    "Acceptance requires declared delayed-PD actuators"
                )
            lags = torch.tensor(
                [
                    rng.integers(
                        actuator.cfg.min_delay, actuator.cfg.max_delay + 1
                    )
                    for rng in rngs
                ],
                device=device,
                dtype=torch.int,
            )
            for buffer_name in (
                "positions_delay_buffer",
                "velocities_delay_buffer",
                "efforts_delay_buffer",
            ):
                buffer = getattr(actuator, buffer_name)
                buffer.reset()
                buffer.set_time_lag(lags)
                torch.testing.assert_close(buffer.time_lags, lags)
            lag_receipt[name] = {
                "joint_names": list(actuator.joint_names),
                "lag_per_case": lags.cpu().tolist(),
            }
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)
        env.action_manager.reset()
        env.observation_manager.reset()
        initial = robot.data.root_link_pos_w.cpu().numpy().copy() - origins
        command = env.command_manager.get_term("base_velocity")

        def scripted_command():
            step = int(env.common_step_counter)
            value = np.stack(
                [
                    velocity_command_at(
                        case, min(step, case["planned_num_frames"] - 1)
                    )
                    for case in cases
                ]
            )
            command.vel_command_b[:] = torch.as_tensor(value, device=device)
            command.time_left.fill_(1e9)
            command.is_standing_env.fill_(False)
            command.is_yaw_env.fill_(False)
            command.is_heading_env.fill_(False)

        command._update_command = scripted_command
        scripted_command()
        obs = env.observation_manager.compute(update_history=True)
        assembler = TensorDictAssembler(
            OmegaConf.to_container(
                cfg.algo.config.module_dict.actor.obs_schema, resolve=True
            )
        )
        allowed_names = {
            name
            for side in policy._normalize_foot_body_name_groups(
                cfg.robot.feet_body_names
            )
            for name in side
        }
        if not allowed_names.issubset(body_names):
            raise ValueError(
                "Declared ground bodies missing in robot body schema"
            )
        sensors = [
            env.scene[f"accept_ground_{i}"] for i in range(len(body_names))
        ]
        if any(
            sensor.contact_physx_view.filter_count != 1 for sensor in sensors
        ):
            raise ValueError(
                "Ground contact filters did not resolve exactly one filter"
            )
        monitors = [
            VelocityFailureMonitor(case["failure_rules"]) for case in cases
        ]
        failures = [[] for _ in cases]
        completed = np.zeros(count, dtype=int)
        traces = [[] for _ in cases]
        physics = [[] for _ in cases]
        active = np.ones(count, dtype=bool)
        original_update = env.scene.update
        total_foot_force_max = 0.0

        def capture_after_physics(dt, *args, **kwargs):
            nonlocal total_foot_force_max
            value = original_update(dt, *args, **kwargs)
            if dt <= 0:
                return value
            forces = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            sensor.data.force_matrix_w.reshape(count, -1, 3),
                            dim=-1,
                        ).sum(-1)
                        for sensor in sensors
                    ],
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
            total_foot_force_max = max(
                total_foot_force_max,
                float(
                    forces[:, [n in allowed_names for n in body_names]]
                    .sum(-1)
                    .max()
                ),
            )
            data = robot.data
            pos = data.root_link_pos_w.cpu().numpy() - origins
            quat = data.root_link_quat_w.cpu().numpy()
            q = data.joint_pos.cpu().numpy()
            qd = data.joint_vel.cpu().numpy()
            tau = data.applied_torque.cpu().numpy()
            for index in np.flatnonzero(active):
                forbidden = float(
                    forces[
                        index, [n not in allowed_names for n in body_names]
                    ].sum()
                )
                reasons = monitors[index].observe(
                    root_position=pos[index],
                    root_quaternion_wxyz=quat[index],
                    forbidden_ground_normal_force_n=forbidden,
                    dt_s=dt,
                )
                failures[index] = list(
                    dict.fromkeys(failures[index] + reasons)
                )
                physics[index].append(
                    {
                        "q": q[index].copy(),
                        "qd": qd[index].copy(),
                        "tau": tau[index].copy(),
                        "forbidden": forbidden,
                    }
                )
            return value

        env.scene.update = capture_after_physics
        for step in range(max(case["planned_num_frames"] for case in cases)):
            active = np.asarray(
                [
                    not failures[i] and step < case["planned_num_frames"]
                    for i, case in enumerate(cases)
                ]
            )
            if not active.any():
                break
            inputs = (
                assembler(TensorDict(obs, batch_size=[count])).cpu().numpy()
            )
            if not np.isfinite(inputs[active]).all():
                raise ValueError("Nonfinite actor input")
            actions = np.zeros(
                (count, env.action_manager.total_action_dim), dtype=np.float32
            )
            for index in np.flatnonzero(active):
                actions[index] = policy.policy_session.run(
                    [policy.policy_output_name],
                    {
                        policy.policy_input_name: inputs[
                            index : index + 1
                        ].astype(np.float32)
                    },
                )[0][0]
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite exported policy output")
            obs, _, _, _, _ = wrapper.step(
                torch.as_tensor(actions, device=device)
            )
            data = robot.data
            pos = data.root_link_pos_w.cpu().numpy() - origins
            quat = data.root_link_quat_w.cpu().numpy()
            velocity = heading_velocity_command_frame(
                data.root_link_lin_vel_w.cpu().numpy(),
                data.root_link_ang_vel_w.cpu().numpy(),
                quat,
            )
            for i in np.flatnonzero(active):
                traces[i].append(
                    {
                        "root_position": pos[i].copy(),
                        "root_quaternion": quat[i].copy(),
                        "velocity": velocity[i].copy(),
                        "action": actions[i].copy(),
                        "q": data.joint_pos[i].cpu().numpy().copy(),
                    }
                )
                if not failures[i]:
                    completed[i] += 1
        if total_foot_force_max <= 1.0:
            raise ValueError(
                "Ground-force positive control failed: no foot support contact"
            )
        results = []
        for index, case in enumerate(cases):
            trace = traces[index]
            pos = np.stack([row["root_position"] for row in trace])
            result = velocity_case_metrics(
                case,
                root_position=pos,
                root_velocity_local=[row["velocity"] for row in trace],
                completed_intervals=int(completed[index]),
                failure_reasons=failures[index],
            )
            result["reset_relative_drift_max_m"] = float(
                np.linalg.norm(pos[:, :2] - initial[index, :2], axis=1).max()
            )
            result["failure_time_s"] = (
                len(trace) * policy_dt if failures[index] else None
            )
            result["actuator_lag_per_group"] = {
                name: item["lag_per_case"][index]
                for name, item in lag_receipt.items()
            }
            np.savez_compressed(
                output / case["output_filename"],
                schema_version="velocity_acceptance_v1",
                root_position=pos,
                root_quaternion=np.stack(
                    [row["root_quaternion"] for row in trace]
                ),
                q=np.stack([row["q"] for row in trace]),
                action=np.stack([row["action"] for row in trace]),
                velocity_command_frame=np.stack(
                    [row["velocity"] for row in trace]
                ),
                physics_q=np.stack([row["q"] for row in physics[index]]),
                physics_qd=np.stack([row["qd"] for row in physics[index]]),
                physics_torque=np.stack(
                    [row["tau"] for row in physics[index]]
                ),
                physics_forbidden_ground_normal_force_n=np.asarray(
                    [row["forbidden"] for row in physics[index]]
                ),
                joint_names=np.asarray(robot.joint_names),
                metadata_json=json.dumps(result),
            )
            results.append(result)
        report = {
            "backend": "isaaclab",
            "model_sha256": file_sha256(model_path),
            "training_config_sha256": file_sha256(
                model_path.parent.parent / "config.yaml"
            ),
            "contact_positive_control_max_foot_normal_force_n": (
                total_foot_force_max
            ),
            "contact_scope": (
                "one sensor per body; normal forces against ground only"
            ),
            "actuator_lags": lag_receipt,
            "robot_joint_names": list(robot.joint_names),
            "gate": judge_velocity_cases(
                manifest, results, profile=profile, stage=stage
            ),
            "cases": results,
            "video_status": "state traces saved; offline rendering pending"
            if record_video
            else "not requested",
        }
        (output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("ISAAC_BEHAVIOR_RESULT", json.dumps(report["gate"]), flush=True)
        return report
    except Exception:
        # A diagnostic failure is not a policy failure or a passing empty run.
        # Preserve captured states when a sensor positive control rejects it.
        if "traces" in locals():
            for index, trace in enumerate(traces):
                if trace:
                    np.savez_compressed(
                        output
                        / ("unvalidated_" + cases[index]["output_filename"]),
                        root_position=np.stack(
                            [r["root_position"] for r in trace]
                        ),
                        root_quaternion=np.stack(
                            [r["root_quaternion"] for r in trace]
                        ),
                        q=np.stack([r["q"] for r in trace]),
                        action=np.stack([r["action"] for r in trace]),
                        gate_status="not_validated",
                    )
        (output / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        sys.stdout.flush()
        os._exit(1)
    finally:
        if wrapper is not None:
            wrapper._env.sim.clear_all_callbacks()
            wrapper._env.sim.clear_instance()
        launcher.app.close(wait_for_replicator=False)
