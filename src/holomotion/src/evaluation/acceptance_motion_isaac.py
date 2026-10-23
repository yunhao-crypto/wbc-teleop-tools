"""Frozen Motion cases using real Isaac managers and exported shared policies.

Process-local evaluation overrides freeze clip assignments and clamp their
future frames. Training reset, observations, actuator and skeleton code are
reused without changing the training method.
"""

from __future__ import annotations
import json
import os
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from holomotion.src.evaluation.acceptance_motion import (
    anchored_smpl_arrays,
    semantic_arrays,
    validate_motion_manifest,
    write_motion_case,
)
from holomotion.src.evaluation.acceptance_mujoco import file_sha256
from holomotion.src.evaluation.behavior_acceptance import (
    judge_motion_group,
    motion_failure_reasons,
)
from holomotion.src.evaluation.gmr_free_smpl_metrics import aggregate_directory


def verify_formal_case_sources(formal_root, clips):
    """Prove the real Formal-H5 arrays are the frozen NPZ windows."""
    import h5py

    header = json.loads((formal_root / "manifest.json").read_text())
    if header.get("schema_version") != "formal_h5_manifest_v1":
        raise ValueError("Acceptance needs a Formal-SMPL source manifest")
    expected = {path.stem: clip for path, clip in clips.items()}
    if len(expected) != len(clips):
        raise ValueError("Ambiguous frozen source names")
    found, receipts = set(), []
    for path in sorted((formal_root / "shards").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            for index, raw in enumerate(handle["clips/motion_key_id"]):
                key = raw.decode() if isinstance(raw, bytes) else str(raw)
                if key not in expected:
                    continue
                if key in found:
                    raise ValueError("Duplicate formal source identity")
                start = int(handle["clips/start"][index])
                length = int(handle["clips/length"][index])
                clip = expected[key]
                if length != clip.frame_count:
                    raise ValueError("Formal source window length differs")
                for field in ("human_pose_aa", "human_root_trans"):
                    actual = np.asarray(handle[field][start : start + length])
                    wanted = np.asarray(getattr(clip, field))
                    if not np.array_equal(actual, wanted):
                        raise ValueError("Formal/NPZ source arrays differ")
                found.add(key)
        receipts.append({"path": str(path), "sha256": file_sha256(path)})
    if found != set(expected):
        raise ValueError("Frozen cases missing in Formal-H5 source")
    return receipts


def validate_selected_windows(dataset, clips):
    """Exactly one complete window per frozen case, not all shard inventory."""
    expected = {path.stem: clip.frame_count for path, clip in clips.items()}
    actual = {}
    for window in dataset.windows:
        meta = dataset.clips[window.raw_motion_key]
        key = meta["raw_motion_key"]
        if key in actual or window.start != meta["start"]:
            raise ValueError("Repeated or truncated selected Motion window")
        actual[key] = window.length
    if actual != expected:
        raise ValueError(
            "Selected dataset differs from complete frozen windows"
        )


def run_motion_acceptance(
    *,
    model_path,
    manifest,
    profile,
    stage,
    output,
    formal_data_root,
    record_video=True,
):
    import holomotion.src.motion_tracking.smpl_actor_observation  # noqa: F401
    from holomotion.src.evaluation.eval_mujoco_sim2sim import (
        MujocoEvaluator,
        process_config,
    )

    cfg = process_config(OmegaConf.create({"ckpt_onnx_path": str(model_path)}))
    OmegaConf.set_struct(cfg, False)
    dt = (
        cfg.env.config.simulation.control_decimation
        / cfg.env.config.simulation.sim_freq
    )
    clips = validate_motion_manifest(
        manifest, profile=profile, stage=stage, policy_dt=dt
    )
    sources = verify_formal_case_sources(formal_data_root, clips)
    cases = manifest["cases"]
    count = len(cases)
    source_names = [path.stem for path in clips]
    max_length = max(clip.frame_count for clip in clips.values())
    cfg.num_envs = cfg.env.config.num_envs = count
    # A cloud export may carry a multi-rank dataset configuration; this is a
    # single local evaluation process, not one shard of the training sampler.
    cfg.env.config.process_id = 0
    cfg.env.config.num_processes = 1
    cfg.env.config.main_process = True
    cfg.env.config.seed = profile["seeds"]["train"]
    cfg.env.config.env_spacing = 10.0
    cfg.env.config.terrain.terrain_type = "plane"
    cfg.env.config.domain_rand = {}
    cfg.env.config.curriculum = {}
    cfg.env.config.terminations = {}
    cfg.env.config.obs.obs_groups.unified.enable_corruption = False
    cfg.env.config.simulation.episode_length_s = 3600
    motion = OmegaConf.create(
        OmegaConf.to_container(cfg.robot.motion, resolve=True)
    )
    motion.train_smpl_hdf5_roots = motion.val_smpl_hdf5_roots = [
        str(formal_data_root)
    ]
    motion.train_hdf5_roots = motion.val_hdf5_roots = []
    motion.handpicked_motion_names = source_names
    motion.excluded_motion_names = []
    # Frozen evaluation assigns each case explicitly; training-only dataset
    # mixture weights do not describe this evaluation source directory.
    motion.sampling_strategy = "uniform"
    motion.max_frame_length = max_length
    motion.min_frame_length = 1
    # The legacy full-dataset guard compares against all shard inventory,
    # including clips deliberately excluded from the frozen pilot manifest.
    # Validate selected windows explicitly below, without dropping any case.
    motion.require_one_window_per_clip = False
    motion.cache.max_num_clips = len(clips)
    motion.cache.swap_interval_steps = 1000000
    motion.dataloader.num_workers = 0
    motion.dataloader.persistent_workers = False
    cfg.robot.motion = motion
    cfg.env.config.robot.motion = motion
    params = cfg.env.config.commands.ref_motion.params
    params.motion_lib_cfg = motion
    params.debug_vis = False
    params.resample_time_interval_s = 1000000
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_evaluation_config.yaml").write_text(
        OmegaConf.to_yaml(cfg)
    )
    (output / "cases_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    cfg.use_gpu = True
    cfg.robot_xml_path = cfg.robot.asset.assetFileName
    policy = MujocoEvaluator.__new__(MujocoEvaluator)
    policy.config = cfg
    policy.max_context_len = int(cfg.algo.config.num_steps_per_env)
    policy.load_policy()
    policy._apply_onnx_metadata()
    if not policy.use_kv_cache or not policy.policy_kv_output_name:
        raise ValueError(
            "Formal Motion acceptance requires exported KV contract"
        )
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

        from holomotion.src.env import motion_tracking as environment_module
        from holomotion.src.modules.agent_modules import TensorDictAssembler
        from holomotion.src.training.h5_dataloader import FormalSmplHdf5Dataset

        body_names = list(cfg.robot.body_names)
        original_build = environment_module.build_scene_config

        class ResolvedPlaneContactSensor(ContactSensor):
            def _initialize_impl(self):
                from pxr import UsdPhysics

                if not hasattr(self, "_acceptance_ground_root"):
                    self._acceptance_ground_root = (
                        self.cfg.filter_prim_paths_expr[0]
                    )
                ground = self._acceptance_ground_root
                planes = [
                    str(prim.GetPath())
                    for prim in SimulationContext.instance().stage.Traverse()
                    if str(prim.GetPath()).startswith(ground + "/")
                    and prim.GetTypeName() == "Plane"
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                if len(planes) != 1:
                    raise ValueError(
                        f"Ground Plane collider is ambiguous: {planes}"
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
            wrapper = environment_module.MotionTrackingEnv(
                config=cfg.env.config,
                device="cuda:0",
                log_dir=str(output),
                headless=True,
            )
        finally:
            environment_module.build_scene_config = original_build
        env = wrapper._env
        robot = env.scene["robot"]
        command = env.command_manager.get_term("ref_motion")
        cache = command._motion_cache
        datasets = [cache._datasets[mode] for mode in ("train", "val")]
        if not all(
            isinstance(dataset, FormalSmplHdf5Dataset) for dataset in datasets
        ):
            raise ValueError(
                "Expected real Formal-SMPL train and validation datasets"
            )
        for dataset in datasets:
            validate_selected_windows(dataset, clips)
        if env.termination_manager.active_terms:
            raise ValueError("Acceptance must not auto-reset")
        wrapper.reset_all()
        cache.set_mode("val")
        batch = cache.current_batch
        if cache.real_clip_count != len(clips):
            raise ValueError("Actual cache does not cover all frozen windows")
        key_to_index = {
            key: i for i, key in enumerate(batch.raw_motion_keys[: len(clips)])
        }
        if set(key_to_index) != set(source_names):
            raise ValueError(
                "Actual cache source names differ from frozen windows"
            )
        assignments = [
            key_to_index[Path(case["source_npz_path"]).stem] for case in cases
        ]
        command._clip_indices[:] = torch.tensor(assignments, device=env.device)
        command._frame_indices.zero_()
        command._start_frame_indices.zero_()
        command._task_space_anchor_valid.fill_(False)
        command._update_ref_motion_state_from_cache()
        command.force_realign_offline_eval_no_perturb(
            torch.arange(count, device=env.device)
        )
        command._is_evaluating = True
        command.time_left.fill_(1e9)
        lengths = cache.lengths_for_indices(command._clip_indices)
        if lengths.cpu().tolist() != [
            case["planned_num_frames"] for case in cases
        ]:
            raise ValueError("Actual cache lengths differ from frozen cases")

        def clamp_without_realign():
            command._frame_indices[:] = torch.minimum(
                command._frame_indices, lengths - 1
            )
            command._motion_end_mask.fill_(False)

        command._resample_when_motion_end_cache = clamp_without_realign
        action_term = env.action_manager.get_term("dof_pos")
        if list(action_term._joint_names) != list(
            policy.action_dof_names_onnx
        ):
            raise ValueError(
                "Isaac action order differs from exported metadata"
            )
        lag_receipt = {}
        rngs = [np.random.default_rng(case["seed"]) for case in cases]
        for name, actuator in robot.actuators.items():
            if not hasattr(actuator, "positions_delay_buffer"):
                raise ValueError(
                    "Acceptance requires declared delayed-PD control"
                )
            lags = torch.tensor(
                [
                    rng.integers(
                        actuator.cfg.min_delay, actuator.cfg.max_delay + 1
                    )
                    for rng in rngs
                ],
                device=env.device,
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
        command._invalidate_observation_state_cache()
        origins = env.scene.env_origins.cpu().numpy().copy()
        initial = robot.data.root_link_pos_w.cpu().numpy().copy()
        initial_q = robot.data.root_link_quat_w.cpu().numpy().copy()
        anchored = []
        for i, case in enumerate(cases):
            ci = assignments[i]
            height = batch.tensors["smpl_root_pos"][
                ci, : case["planned_num_frames"], 2
            ].cpu()
            anchored.append(
                anchored_smpl_arrays(
                    clips[Path(case["source_npz_path"])],
                    root_position=initial[i],
                    root_quaternion_wxyz=initial_q[i],
                    grounded_root_height=height,
                )
            )
        target_error = {"position_max_abs_m": 0.0, "quaternion_max_abs": 0.0}

        def check_target(target_state, indices):
            arrays = semantic_arrays(target_state, prefix="target")
            for i, ti in enumerate(indices):
                for key, value in arrays.items():
                    wanted = anchored[i][key][ti]
                    if "position" in key:
                        err = float(np.max(np.abs(value[i] - wanted)))
                        target_error["position_max_abs_m"] = max(
                            target_error["position_max_abs_m"], err
                        )
                    else:
                        err = float(
                            np.minimum(
                                np.max(np.abs(value[i] - wanted), axis=-1),
                                np.max(np.abs(value[i] + wanted), axis=-1),
                            ).max()
                        )
                        target_error["quaternion_max_abs"] = max(
                            target_error["quaternion_max_abs"], err
                        )
            if max(target_error.values()) > 1e-4:
                raise ValueError(
                    "Actual Isaac target differs from shared MuJoCo bridge: "
                    f"{target_error}"
                )
            return arrays

        check_target(
            command.get_semantic_skeleton_target_current(), [0] * count
        )
        assembler = TensorDictAssembler(
            OmegaConf.to_container(
                cfg.algo.config.module_dict.actor.obs_schema, resolve=True
            )
        )
        obs = env.observation_manager.compute(update_history=True)
        kv = [policy.policy_kv_cache.copy() for _ in cases]
        allowed = {
            name
            for side in policy._normalize_foot_body_name_groups(
                cfg.robot.feet_body_names
            )
            for name in side
        }
        sensors = [
            env.scene[f"accept_ground_{i}"] for i in range(len(body_names))
        ]
        if not allowed.issubset(body_names) or any(
            sensor.contact_physx_view.filter_count != 1 for sensor in sensors
        ):
            raise ValueError("Ground contact sensor coverage is invalid")
        active = np.ones(count, dtype=bool)
        failures = [[] for _ in cases]
        first_failure = [None for _ in cases]
        rows, physical, traces = ([[] for _ in cases] for _ in range(3))
        completed = np.zeros(count, dtype=int)
        pre_q = robot.data.joint_pos.cpu().numpy().copy()
        target_next = command.get_semantic_skeleton_target_immediate_next()
        original_update = env.scene.update
        foot_max = 0.0

        def capture_physics(dt_s, *args, **kwargs):
            nonlocal pre_q, foot_max
            result = original_update(dt_s, *args, **kwargs)
            if dt_s <= 0:
                return result
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
            foot_max = max(
                foot_max,
                float(
                    forces[:, [n in allowed for n in body_names]].sum(-1).max()
                ),
            )
            torso = (
                command.get_semantic_skeleton_robot_state()
                .torso_rotation_wxyz.cpu()
                .numpy()
            )
            target_q = target_next.torso_rotation_wxyz.cpu().numpy()
            q, qd, tau = (
                getattr(robot.data, attr).cpu().numpy().copy()
                for attr in ("joint_pos", "joint_vel", "applied_torque")
            )
            for i in np.flatnonzero(active):
                reasons = motion_failure_reasons(
                    target_torso_wxyz=target_q[i],
                    robot_torso_wxyz=torso[i],
                    threshold=cases[i]["failure_rules"][
                        "torso_gravity_z_difference_max"
                    ],
                )
                physical[i].append(
                    {
                        "pre_q": pre_q[i].copy(),
                        "q": q[i].copy(),
                        "qd": qd[i].copy(),
                        "tau": tau[i].copy(),
                        "ground": forces[i].copy(),
                    }
                )
                if reasons and first_failure[i] is None:
                    first_failure[i] = len(physical[i]) * dt_s
                failures[i] = list(dict.fromkeys(failures[i] + reasons))
            pre_q = q
            return result

        env.scene.update = capture_physics
        for step in range(max_length):
            active = np.asarray(
                [
                    not failures[i] and step < case["planned_num_frames"]
                    for i, case in enumerate(cases)
                ]
            )
            if not active.any():
                break
            expected_indices = [
                min(step, case["planned_num_frames"] - 1) for case in cases
            ]
            if command._frame_indices.cpu().tolist() != expected_indices:
                raise ValueError(
                    "Motion assignment advanced or resampled unexpectedly"
                )
            target_next = command.get_semantic_skeleton_target_immediate_next()
            target_arrays = check_target(
                target_next,
                [
                    min(step + 1, case["planned_num_frames"] - 1)
                    for case in cases
                ],
            )
            inputs = (
                assembler(TensorDict(obs, batch_size=[count])).cpu().numpy()
            )
            if not np.isfinite(inputs[active]).all():
                raise ValueError("Nonfinite actor input")
            actions = np.zeros(
                (count, env.action_manager.total_action_dim), dtype=np.float32
            )
            for i in np.flatnonzero(active):
                feed = {
                    policy.policy_input_name: inputs[i : i + 1].astype(
                        np.float32
                    ),
                    policy.policy_kv_input_name: kv[i],
                }
                if policy.policy_step_input_name:
                    feed[policy.policy_step_input_name] = np.array(
                        [step % policy.policy_effective_context_len],
                        dtype=np.int64,
                    )
                action, kv[i] = policy.policy_session.run(
                    [policy.policy_output_name, policy.policy_kv_output_name],
                    feed,
                )
                actions[i] = action[0]
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite policy output")
            obs, _, _, _, _ = wrapper.step(
                torch.as_tensor(actions, device=env.device)
            )
            robot_arrays = semantic_arrays(
                command.get_semantic_skeleton_robot_state(), prefix="robot"
            )
            for i in np.flatnonzero(active):
                row = {
                    key: value[i].copy()
                    for key, value in {**target_arrays, **robot_arrays}.items()
                }
                for prefix in ("target", "robot"):
                    row[prefix + "_semantic_position_w"] -= origins[i]
                rows[i].append(row)
                traces[i].append(
                    {
                        "root": robot.data.root_link_pos_w[i]
                        .cpu()
                        .numpy()
                        .copy()
                        - origins[i],
                        "quat": robot.data.root_link_quat_w[i]
                        .cpu()
                        .numpy()
                        .copy(),
                        "q": robot.data.joint_pos[i].cpu().numpy().copy(),
                        "action": actions[i].copy(),
                    }
                )
                completed[i] += not bool(failures[i])
        if foot_max <= 1.0:
            raise ValueError("Ground-force positive control failed")
        reads = [int(dataset.robot_h5_reads) for dataset in datasets]
        if reads != [0, 0]:
            raise ValueError("Robot-H5 reads must remain zero")
        results = []
        for i, case in enumerate(cases):
            results.append(
                write_motion_case(
                    output,
                    case,
                    rows[i],
                    completed=int(completed[i]),
                    failure=failures[i],
                    raw={
                        "root_position": np.stack(
                            [r["root"] for r in traces[i]]
                        ),
                        "root_quaternion": np.stack(
                            [r["quat"] for r in traces[i]]
                        ),
                        "q": np.stack([r["q"] for r in traces[i]]),
                        "action": np.stack([r["action"] for r in traces[i]]),
                        "joint_names": np.asarray(robot.joint_names),
                        "physics_pre_q": np.stack(
                            [r["pre_q"] for r in physical[i]]
                        ),
                        "physics_q": np.stack([r["q"] for r in physical[i]]),
                        "physics_qd": np.stack([r["qd"] for r in physical[i]]),
                        "physics_torque": np.stack(
                            [r["tau"] for r in physical[i]]
                        ),
                        "physics_ground_normal_force_n": np.stack(
                            [r["ground"] for r in physical[i]]
                        ),
                    },
                    extra={
                        "first_failure_physics_time_s": first_failure[i],
                        "actuator_lag_per_group": {
                            name: item["lag_per_case"][i]
                            for name, item in lag_receipt.items()
                        },
                    },
                )
            )
        summary = aggregate_directory(output, expected_manifest=manifest)
        report = {
            "backend": "isaaclab",
            "model_sha256": file_sha256(model_path),
            "training_config_sha256": file_sha256(
                model_path.parent.parent / "config.yaml"
            ),
            "formal_source_receipts": sources,
            "robot_h5_reads": reads,
            "actual_target_bridge_parity": target_error,
            "contact_positive_control_max_foot_normal_force_n": foot_max,
            "actuator_lags": lag_receipt,
            "robot_joint_names": list(robot.joint_names),
            "cases": results,
            "summary": summary,
            "gate": judge_motion_group(summary, profile=profile, stage=stage),
            "no_hardware_output": True,
            "video_status": "state traces saved; offline rendering pending"
            if record_video
            else "not requested",
        }
        (output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            "ISAAC_MOTION_BEHAVIOR_RESULT",
            json.dumps(report["gate"]),
            flush=True,
        )
        return report
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        os._exit(1)
    finally:
        if wrapper is not None:
            wrapper._env.sim.clear_all_callbacks()
            wrapper._env.sim.clear_instance()
        launcher.app.close(wait_for_replicator=False)
