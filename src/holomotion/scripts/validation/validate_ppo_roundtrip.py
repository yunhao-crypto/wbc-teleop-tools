"""Real local full-network PPO, checkpoint restore, and ONNX round-trip gate.

Uses the production algorithm unchanged, with caller-selected small env count.
Only writes a fresh diagnostic run. Not a learning-quality or deployment gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf


def optimizer_digest(optimizer):
    import torch

    digest = hashlib.sha256()
    for index, state in enumerate(optimizer.state.values()):
        digest.update(str(index).encode())
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                cpu = value.detach().cpu().contiguous().reshape(-1)
                digest.update(cpu.view(torch.uint8).numpy().tobytes())
            else:
                digest.update(str(value).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--config-name")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--formal-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceptance-profile", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    from holomotion.src.utils.config import (
        setup_hydra_resolvers,
        compile_config_devices,
    )
    from holomotion.src.utils.acceptance_profile import load_acceptance_profile
    from holomotion.src.training.train import _validate_gmr_free_contract
    from holomotion.src.training.holomotion_training_contract import (
        validate_holomotion_training_contract,
    )
    from holomotion.src.utils.training_health import (
        parameter_digest,
        optimizer_snapshot,
    )

    setup_hydra_resolvers()
    profile = load_acceptance_profile(args.acceptance_profile)
    if args.resolved_config is not None:
        cfg = OmegaConf.merge(
            OmegaConf.load(args.resolved_config),
            OmegaConf.from_dotlist(args.override),
        )
        cfg.timestamp = "local_functional_regression"
    else:
        if args.config_dir is None or args.config_name is None:
            parser.error(
                "Provide resolved config, or config-dir and config-name"
            )
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
    # Match train.py's runtime-populated fields before resolving base templates.
    cfg.headless = True
    cfg.experiment_save_dir = str(output)
    if OmegaConf.is_missing(cfg, "experiment_name"):
        cfg.experiment_name = "local_functional_regression"
    cfg.robot.asset.usd_cache_dir = str(output / "usd")
    if args.formal_data_root is not None:
        roots = [{"root": str(args.formal_data_root.resolve()), "ratio": 1.0}]
        cfg.train_smpl_hdf5_roots = roots
        cfg.train_hdf5_roots = roots
        cfg.val_smpl_hdf5_roots = roots
        motion = cfg.robot.motion
        motion.train_smpl_hdf5_roots = roots
        motion.val_smpl_hdf5_roots = roots
        motion.train_hdf5_roots = roots
        motion.cache_max_num_clips = int(cfg.num_envs)
        motion.dataloader = dict(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            timeout=0,
            batch_progress_bar=False,
        )
    cfg.process_id, cfg.num_processes, cfg.main_process = 0, 1, True
    cfg = compile_config_devices(cfg)
    cfg.algo.config.num_learning_iterations = int(
        profile["functional"]["local_updates"]
    )
    cfg.algo.config.save_interval = int(profile["functional"]["local_updates"])
    cfg.algo.config.export_policy = True
    cfg.algo.config.runtime_validation = dict(
        enabled=True,
        acceptance_profile=str(args.acceptance_profile.resolve()),
        expected_world_size=1,
        expected_envs_per_rank=int(cfg.num_envs),
        expected_nodes=1,
        milestones=[2],
        optimizer_check_interval=100,
    )
    _validate_gmr_free_contract(cfg)
    validate_holomotion_training_contract(cfg)
    from holomotion.src.env.urdf_assets import xml_asset_fingerprint

    asset_identity = xml_asset_fingerprint(str(cfg.robot.asset.urdf_file))
    (output / "input_asset_identity.json").write_text(
        json.dumps(asset_identity, indent=2) + "\n"
    )
    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
        output / "config.yaml",
    )
    algo = None
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from holomotion.src.modules.agent_modules import (
            PPOActorOnnxModule,
            PPOTFActorOnnxModule,
        )

        klass = get_class(cfg.algo._target_)
        algo = klass(
            env_config=cfg.env,
            config=cfg.algo.config,
            log_dir=str(output),
            headless=True,
        )
        actor = algo.accelerator.unwrap_model(algo.actor)
        is_motion = hasattr(actor, "onnx_past_key_values_shape")
        # PPO intentionally closes/clears its motion cache in the end hook.
        # Retain dataset objects, not an assumption that the cache stays live.
        dataset_refs = (
            list(algo.command_term._motion_cache._datasets.values())
            if is_motion
            else []
        )
        result = dict(
            schema="local_ppo_roundtrip_v1",
            config_name=args.config_name,
            envs_per_rank=int(cfg.num_envs),
            world_size=1,
            actor_parameter_count=sum(p.numel() for p in actor.parameters()),
            critic_parameter_count=sum(
                p.numel() for p in algo.critic.parameters()
            ),
            shared_algorithm_source=str(
                Path(sys.modules[klass.__module__].__file__).resolve()
            ),
            is_motion=is_motion,
        )
        if args.resolved_config:
            result["source_config"] = str(args.resolved_config.resolve())
            result["source_config_sha256"] = hashlib.sha256(
                args.resolved_config.read_bytes()
            ).hexdigest()
        captured = []
        if args.observations is None:
            original_step = algo.env.step

            def capture_step(*pos, **kwargs):
                answer = original_step(*pos, **kwargs)
                if sum(len(value) for value in captured) < int(
                    profile["observation"]["same_state_samples"]
                ):
                    with torch.inference_mode():
                        flat = actor.assembler(algo._wrap_obs_dict(answer[0]))
                        captured.append(flat.detach().cpu().numpy().copy())
                return answer

            algo.env.step = capture_step
        (output / "initialization.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        algo.learn()
        iteration = int(algo.current_learning_iteration)
        checkpoint = output / f"model_{iteration}.pt"
        result["completed_updates"] = iteration
        result["actor_optimizer"] = optimizer_snapshot(algo.actor_optimizer)
        result["critic_optimizer"] = optimizer_snapshot(algo.critic_optimizer)

        def identity():
            return dict(
                actor=parameter_digest(algo.actor),
                critic=parameter_digest(algo.critic),
                actor_optimizer=optimizer_digest(algo.actor_optimizer),
                critic_optimizer=optimizer_digest(algo.critic_optimizer),
            )

        before = identity()
        # Deliberately perturb live state, then prove full checkpoint restoration.
        with torch.no_grad():
            next(algo.actor.parameters()).reshape(-1)[0] += 1
            next(algo.critic.parameters()).reshape(-1)[0] += 1
            for opt in (algo.actor_optimizer, algo.critic_optimizer):
                next(iter(opt.state.values()))["exp_avg"].zero_()
        algo.current_learning_iteration = -1
        algo.load(str(checkpoint))
        after = identity()
        if before != after or algo.current_learning_iteration != iteration:
            raise RuntimeError(
                "Full Actor/Critic/Adam/iteration restoration mismatch"
            )
        result["restore"] = dict(passed=True, before=before, after=after)
        (output / "restore.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        onnx_paths = list(
            (output / "exported").glob(f"model_{iteration}*.onnx")
        )
        if len(onnx_paths) != 1:
            raise RuntimeError(
                f"Expected one exported ONNX, found {onnx_paths}"
            )
        model_path = onnx_paths[0]
        onnx.checker.check_model(str(model_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        observations = (
            np.load(args.observations)["training"]
            if args.observations is not None
            else np.concatenate(captured)
        )
        result["observation_source"] = (
            str(args.observations)
            if args.observations is not None
            else "captured real training rollout (not cross-simulator builder proof)"
        )
        count = int(profile["observation"]["same_state_samples"])
        if observations.shape[0] < count:
            raise RuntimeError("Insufficient real same-state observations")
        # Spread samples over the trace, including later resets/moving states.
        samples = observations[
            np.linspace(0, len(observations) - 1, count, dtype=int)
        ]
        normalizer = (
            actor.obs_normalizer
            if actor.obs_norm_enabled
            else torch.nn.Identity()
        )
        wrapper_type = (
            PPOTFActorOnnxModule if is_motion else PPOActorOnnxModule
        )
        wrapper = wrapper_type(
            actor.actor_module,
            normalizer,
            actor.obs_norm_enabled,
            actor.obs_norm_clip if actor.obs_norm_enabled else 0.0,
        ).eval()
        torch_outputs, ort_outputs = [], []
        torch_kv = ort_kv = None
        with torch.inference_mode():
            for index, row in enumerate(samples):
                batch = row[None].astype(np.float32)
                inputs = {session.get_inputs()[0].name: batch}
                if is_motion:
                    # Same explicit recurrent-state sequence on both backends;
                    # sampled observations are not claimed as one physical episode.
                    if index % 32 == 0:
                        ort_kv = np.zeros(
                            actor.onnx_past_key_values_shape(batch_size=1),
                            np.float32,
                        )
                        torch_kv = torch.from_numpy(ort_kv).to(algo.device)
                    step = np.array([index % 32], dtype=np.int64)
                    inputs.update(past_key_values=ort_kv, step_idx=step)
                    expected = wrapper(
                        torch.from_numpy(batch).to(algo.device),
                        torch_kv,
                        torch.from_numpy(step).to(algo.device),
                    )
                    actual = session.run(None, inputs)
                    torch_kv = expected[1]
                    ort_kv = actual[1]
                    expected_actions = expected[0]
                else:
                    expected_actions = wrapper(
                        torch.from_numpy(batch).to(algo.device)
                    )
                    actual = session.run(None, inputs)
                torch_outputs.append(expected_actions.cpu().numpy())
                ort_outputs.append(actual[0])
        expected, actual = (
            np.concatenate(torch_outputs),
            np.concatenate(ort_outputs),
        )
        limits = profile["observation"]
        np.savez_compressed(
            output / "export_parity.npz",
            observations=samples,
            pytorch=expected,
            onnx=actual,
        )
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError("Non-finite exported actions")
        np.testing.assert_allclose(
            actual,
            expected,
            atol=limits["onnx_fp32_atol"],
            rtol=limits["onnx_fp32_rtol"],
        )
        result["onnx"] = dict(
            path=str(model_path),
            sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            samples=count,
            max_abs_error=float(np.max(np.abs(expected - actual))),
            input_dim=int(samples.shape[-1]),
            output_dim=int(actual.shape[-1]),
        )
        if is_motion:
            result["robot_h5_reads"] = [
                int(getattr(ds, "robot_h5_reads", -1)) for ds in dataset_refs
            ]
            if any(value != 0 for value in result["robot_h5_reads"]):
                raise RuntimeError("Formal SMPL-only data contract failed")
        result["status"] = "passed"
        (output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        (output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if algo is not None:
            # Avoid SDK STOP callback waiting forever for PLAY during teardown.
            algo.env._env.sim.clear_all_callbacks()
            algo.env._env.sim.clear_instance()
            algo._sim_app.close()


if __name__ == "__main__":
    main()
