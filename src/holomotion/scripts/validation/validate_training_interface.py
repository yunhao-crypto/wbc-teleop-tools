"""Exercise real training managers and same-state evaluation observation builders.

No learning and no hardware output. Config/data paths are supplied by the caller;
only diagnostic receipts/arrays and generated USD are written to the new output.
"""

from __future__ import annotations

import argparse
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceptance-profile", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    from holomotion.src.utils.acceptance_profile import load_acceptance_profile
    from holomotion.src.utils.config import (
        setup_hydra_resolvers,
        compile_config_devices,
    )
    from holomotion.src.training.train import _validate_gmr_free_contract
    from holomotion.src.training.holomotion_training_contract import (
        validate_holomotion_training_contract,
    )

    # Match the training entrypoint: load the project Warp kernels before Kit
    # adds its bundled extension search paths.
    from holomotion.src.motion_tracking.smpl_actor_observation import (
        SmplMotionActorObservationInput,
        build_smpl_motion_actor_observation_torch,
    )

    profile = load_acceptance_profile(args.acceptance_profile)
    setup_hydra_resolvers()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base=None
    ):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                f"hydra.searchpath=[file://{root / 'holomotion/config'}]",
                *args.override,
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.process_id, cfg.num_processes, cfg.main_process = 0, 1, True
    cfg.env.config.obs.obs_groups.unified.enable_corruption = False
    cfg = compile_config_devices(cfg)
    _validate_gmr_free_contract(cfg)
    validate_holomotion_training_contract(cfg)
    # Resolve all consumed subtrees, not unrelated optional ??? legacy fields.
    frozen = OmegaConf.to_container(cfg, resolve=True)
    (output / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(frozen))
    )
    (output / "acceptance_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        device="cuda:0",
        enable_cameras=False,
        kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false",
    )
    wrapper = None
    try:
        import numpy as np
        import mujoco
        import torch
        from tensordict import TensorDict
        from holomotion.src.modules.agent_modules import TensorDictAssembler
        from holomotion.src.env.isaaclab_components.isaaclab_domain_rand import (
            DomainRandFunctions,
        )
        from holomotion.src.evaluation.eval_mujoco_sim2sim import (
            MujocoEvaluator,
        )
        from holomotion.src.evaluation.obs.obs_builder import PolicyObsBuilder
        from holomotion.src.utils.joint_physics import (
            isaac_joint_physics_snapshot,
        )
        from holomotion.src.motion_tracking.smpl_actor_observation import (
            SmplMotionActorObservationInput,
            build_smpl_motion_actor_observation_torch,
        )

        module_name, class_name = str(cfg.env._target_).rsplit(".", 1)
        module = importlib.import_module(module_name)
        assert Path(module.__file__).resolve().is_relative_to(root), (
            module.__file__
        )
        events = {}
        # Instrument the existing event callables only in this disposable process.
        # wraps preserves signatures consumed by the real Isaac event manager.
        wrapped = {}
        for event_name, term in frozen["env"]["config"]["domain_rand"].items():
            if not isinstance(term, dict) or "mode" not in term:
                continue
            method_name = "_get_dr_" + term.get("func", event_name)
            events[event_name] = {
                "mode": term["mode"],
                "config": term,
                "calls": 0,
                "env_applications": 0,
                "first_step": None,
                "last_step": None,
            }
            if method_name in wrapped:
                raise ValueError(
                    "Diagnostic requires distinct DR callables; extend named event instrumentation"
                )
            original = getattr(DomainRandFunctions, method_name)

            def decorate(function, name):
                @wraps(function)
                def traced(env, env_ids, *pos, **kwargs):
                    result = function(env, env_ids, *pos, **kwargs)
                    row = events[name]
                    count = (
                        int(env.num_envs)
                        if env_ids is None
                        else int(env_ids.numel())
                    )
                    step = int(getattr(env, "common_step_counter", 0))
                    row["calls"] += 1
                    row["env_applications"] += count
                    row["first_step"] = (
                        step
                        if row["first_step"] is None
                        else row["first_step"]
                    )
                    row["last_step"] = step
                    return result

                return traced

            wrapped[method_name] = original
            setattr(
                DomainRandFunctions,
                method_name,
                staticmethod(decorate(original, event_name)),
            )
        wrapper = getattr(module, class_name)(
            config=cfg.env.config,
            device="cuda:0",
            log_dir=str(output),
            headless=True,
        )
        env = wrapper._env
        robot = env.scene["robot"]
        for name in env.event_manager.active_terms.get("reset", []):
            term_cfg = env.event_manager.get_term_cfg(name)
            events[name] = {
                "mode": "reset",
                "config": {"params": repr(term_cfg.params)},
                "calls": 0,
                "env_applications": 0,
                "first_step": None,
                "last_step": None,
            }
            term_cfg.func = decorate(term_cfg.func, name)
            env.event_manager.set_term_cfg(name, term_cfg)
        obs, _ = wrapper.reset_all()
        schema = OmegaConf.to_container(
            cfg.algo.config.module_dict.actor.obs_schema, resolve=True
        )
        assembler = TensorDictAssembler(schema)
        is_motion = "ref_motion" in env.command_manager.active_terms
        expected_dim = profile["asset"][
            "expected_motion_observations"
            if is_motion
            else "expected_velocity_observations"
        ]
        action = env.action_manager.get_term("dof_pos")
        action_names = list(action._joint_names)
        action_scale = action._scale[0].detach().cpu().numpy()
        if action_scale.shape != (len(action_names),):
            raise ValueError(
                "Runtime action scale shape does not match named actions: "
                f"{action_scale.shape} versus {len(action_names)}"
            )
        if not np.isfinite(action_scale).all() or np.any(action_scale <= 0):
            raise ValueError(
                "Runtime action scale must be positive and finite"
            )
        joint_names = list(robot.joint_names)
        action_indices = [joint_names.index(name) for name in action_names]
        max_error, count, reset_count = 0.0, 0, 1
        saved_train, saved_eval = [], []
        trace = []
        builders = []
        mj_model = (
            mujoco.MjModel.from_xml_path(
                str(root / cfg.robot.asset.assetFileName)
            )
            if not is_motion
            else None
        )
        atomic = {
            str(next(iter(item))): dict(next(iter(item.values())))
            for item in frozen["obs"]["obs_groups"]["unified"][
                "atomic_obs_list"
            ]
        }
        policy_terms = []
        for group in [] if is_motion else schema.values():
            for path in group["terms"]:
                name = path.split("/")[-1]
                policy_terms.append({name: {**atomic[name], "flatten": True}})
        for index in range(min(4, env.num_envs)):
            ev = MujocoEvaluator.__new__(MujocoEvaluator)
            ev.config = cfg
            if mj_model is not None:
                ev.m = mj_model
                ev.d = mujoco.MjData(mj_model)
                ids = np.asarray(
                    [mj_model.joint(name).id for name in joint_names]
                )
                ev.actuator_qpos_indices = mj_model.jnt_qposadr[ids]
                ev.actuator_qvel_indices = mj_model.jnt_dofadr[ids]
            ev.command_mode = "velocity_tracking"
            ev.dof_names_onnx = joint_names
            ev.mu_to_onnx = np.arange(len(joint_names))
            ev._get_policy_atomic_obs_list = lambda: policy_terms
            ev._get_scripted_velocity_command = lambda: ev._probe_command
            builder = (
                PolicyObsBuilder(
                    joint_names,
                    robot.data.default_joint_pos[index].cpu().numpy(),
                    evaluator=ev,
                    obs_policy_cfg={"atomic_obs_list": policy_terms},
                )
                if not is_motion
                else None
            )
            builders.append((ev, builder))
        command = env.command_manager.get_term(
            "ref_motion" if is_motion else "base_velocity"
        )
        for step in range(args.steps + 1):
            training = assembler(TensorDict(obs, batch_size=[env.num_envs]))
            if is_motion:
                deployed = build_smpl_motion_actor_observation_torch(
                    SmplMotionActorObservationInput(
                        root_state_with_context=command._get_smpl_root_state_with_context(),
                        body_current_future=command._get_smpl_body_current_future(),
                        robot_root_quat_wxyz=robot.data.root_quat_w,
                        robot_root_angvel_local=robot.data.root_ang_vel_b,
                        robot_dof_pos=robot.data.joint_pos[:, action_indices],
                        robot_dof_vel=robot.data.joint_vel[:, action_indices],
                        last_action=env.action_manager.action,
                        default_dof_pos=robot.data.default_joint_pos[
                            :, action_indices
                        ],
                    ),
                    fps=float(cfg.obs.target_fps),
                    num_future_frames=int(cfg.obs.n_fut_frames),
                )
            else:
                values = []
                for index, (ev, builder) in enumerate(builders):
                    ev.d.qpos[:3] = robot.data.root_pos_w[index].cpu().numpy()
                    ev.d.qpos[3:7] = (
                        robot.data.root_quat_w[index].cpu().numpy()
                    )
                    ev.d.qvel[:3] = (
                        robot.data.root_lin_vel_w[index].cpu().numpy()
                    )
                    ev.d.qvel[3:6] = (
                        robot.data.root_ang_vel_b[index].cpu().numpy()
                    )
                    ev.d.qpos[ev.actuator_qpos_indices] = (
                        robot.data.joint_pos[index].cpu().numpy()
                    )
                    ev.d.qvel[ev.actuator_qvel_indices] = (
                        robot.data.joint_vel[index].cpu().numpy()
                    )
                    mujoco.mj_forward(ev.m, ev.d)
                    ev.default_angles_onnx = (
                        robot.data.default_joint_pos[index].cpu().numpy()
                    )
                    ev.actions_onnx = (
                        env.action_manager.action[index].cpu().numpy()
                    )
                    ev._probe_command = (
                        command.command[index, :3].cpu().numpy()
                    )
                    ev._get_scripted_velocity_command = lambda current=ev: (
                        current._probe_command
                    )
                    values.append(builder.build_policy_obs())
                deployed = torch.as_tensor(
                    np.stack(values), device=training.device
                )
                training = training[: len(builders)]
            assert (
                training.shape == deployed.shape
                and training.shape[-1] == expected_dim
            ), (training.shape, deployed.shape)
            if (
                not torch.isfinite(training).all()
                or not torch.isfinite(deployed).all()
            ):
                raise ValueError("Nonfinite observation")
            error = float((training - deployed).abs().max())
            max_error = max(max_error, error)
            count += training.shape[0]
            if (
                sum(x.shape[0] for x in saved_train)
                < profile["observation"]["same_state_samples"]
            ):
                saved_train.append(training.cpu().numpy().copy())
                saved_eval.append(deployed.cpu().numpy().copy())
            trace.append({"step": step, "max_abs_observation_error": error})
            if error > profile["observation"]["max_abs_error"]:
                raise ValueError(
                    f"Same-state observation mismatch at step {step}: {error}"
                )
            if step == args.steps:
                break
            # No learned policy here: bounded diagnostic excitation only.
            excitation = 0.05 * torch.sin(
                torch.arange(
                    env.action_manager.total_action_dim, device=env.device
                )[None]
                + step * 0.15
            )
            obs, _, done, _, _ = wrapper.step(
                excitation.expand(env.num_envs, -1).clone()
            )
            for index, (_, builder) in enumerate(builders):
                if builder is not None and bool(done[index]):
                    builder.reset()
                    reset_count += 1
        if count < profile["observation"]["same_state_samples"]:
            raise ValueError(f"Only {count} same-state samples")
        (output / "after_manager_steps.json").write_text(
            json.dumps(isaac_joint_physics_snapshot(robot), indent=2) + "\n"
        )
        rollout_event_counts = {
            name: dict(row) for name, row in events.items()
        }
        # Open-loop fall/reset can restart every per-environment interval timer.
        # Separately exercise the real event-manager clock without physics or
        # termination, keeping the configured intervals and callback arguments.
        manager = env.event_manager
        manager.reset(torch.arange(env.num_envs, device=env.device))
        interval_steps = 0
        if manager.active_terms.get("interval"):
            maximum = max(
                float(manager.get_term_cfg(name).interval_range_s[1])
                for name in manager.active_terms["interval"]
            )
            interval_steps = int(np.ceil(maximum / env.step_dt)) + 1
            for _ in range(interval_steps):
                manager.apply("interval", dt=env.step_dt)
        (output / "event_phase_scope.json").write_text(
            json.dumps(
                {
                    "rollout_event_counts": rollout_event_counts,
                    "isolated_interval_clock_steps": interval_steps,
                    "isolated_interval_clock_dt_s": env.step_dt,
                    "isolated_phase_has_no_physics_step": True,
                },
                indent=2,
            )
            + "\n"
        )
        np.savez_compressed(
            output / "same_state_observations.npz",
            training=np.concatenate(saved_train),
            evaluation=np.concatenate(saved_eval),
        )
        (output / "observation_trace.json").write_text(
            json.dumps(trace, indent=2) + "\n"
        )
        (output / "event_counts.json").write_text(
            json.dumps(events, indent=2) + "\n"
        )
        for row in events.values():
            if row["mode"] == "startup" and row["calls"] != 1:
                raise ValueError(f"Startup event count: {row}")
            if row["mode"] == "interval" and row["calls"] == 0:
                raise ValueError(f"Interval event not exercised: {row}")
        actual_curriculum = list(env.curriculum_manager.active_terms)
        if (
            not cfg.env.config.curriculum.get("enabled", True)
            and actual_curriculum
        ):
            raise ValueError(
                f"Disabled curriculum still has terms: {actual_curriculum}"
            )
        report = {
            "passed": True,
            "scope": "local real manager and same-state builder check; no PPO or cloud",
            "actor_dim": expected_dim,
            "samples": count,
            "max_abs_observation_error": max_error,
            "reset_count": reset_count,
            "action_joint_names": action_names,
            "action_scale_rad": action_scale.tolist(),
            "control_joint_names": joint_names,
            "curriculum_terms": actual_curriculum,
            "events": events,
        }
        if is_motion:
            from holomotion.src.training.h5_dataloader import (
                FormalSmplHdf5Dataset,
            )

            cache = command._motion_cache
            datasets = [
                value
                for value in cache._datasets.values()
                if isinstance(value, FormalSmplHdf5Dataset)
            ]
            if not datasets or any(
                value.robot_h5_reads != 0 for value in datasets
            ):
                raise ValueError(
                    "Missing formal dataset runtime evidence or forbidden Robot-H5 reads"
                )
            report["formal_datasets"] = len(datasets)
            report["robot_h5_reads"] = [
                value.robot_h5_reads for value in datasets
            ]
        (output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("TRAINING_INTERFACE_PASSED", json.dumps(report), flush=True)
    except Exception:
        if "events" in locals():
            (output / "event_counts.json").write_text(
                json.dumps(events, indent=2) + "\n"
            )
        if "trace" in locals():
            (output / "observation_trace.json").write_text(
                json.dumps(trace, indent=2) + "\n"
            )
        if "saved_train" in locals() and saved_train:
            np.savez_compressed(
                output / "same_state_observations.npz",
                training=np.concatenate(saved_train),
                evaluation=np.concatenate(saved_eval),
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


if __name__ == "__main__":
    main()
