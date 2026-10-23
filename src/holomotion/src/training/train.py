# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import os
import sys
import traceback
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hydra.utils import get_class
from omegaconf import ListConfig, OmegaConf

from holomotion.src.training.holomotion_training_contract import (
    validate_holomotion_training_contract,
)
from holomotion.src.utils.config import compile_config


def _write_training_success_sentinel() -> None:
    """Write rank-zero completion proof only after the entrypoint returns."""
    raw_path = os.environ.get("HOLOMOTION_TRAINING_SUCCESS_SENTINEL", "")
    if not raw_path or int(os.environ.get("RANK", "0")) != 0:
        return
    target = Path(raw_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "completed",
                "rank": 0,
                "pid": os.getpid(),
                "module_file": __file__,
            },
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, target)


def _run_main_fail_closed(entrypoint):
    """Preserve a non-zero process result across simulator shutdown hooks."""
    try:
        result = entrypoint()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    _write_training_success_sentinel()
    return result


def _mujoco_eval_config(config: OmegaConf):
    """Return the optional legacy MuJoCo post-training evaluation config."""

    return config.get("mujoco_eval", None)


def _mujoco_eval_enabled(config: OmegaConf) -> bool:
    """Treat a missing legacy MuJoCo evaluation block as disabled."""

    mujoco_eval = _mujoco_eval_config(config)
    return mujoco_eval is not None and bool(mujoco_eval.get("enabled", False))


def _validate_gmr_free_contract(config: OmegaConf) -> dict | None:
    """Fail closed when a clean SMPL experiment can read Robot-ref state."""

    contract = config.get("gmr_free_contract", None)
    if contract is None or not bool(contract.get("required", False)):
        return None
    motion = config.robot.motion
    backend = str(motion.get("backend", "")).strip().lower()
    if backend not in {"formal_smpl_hdf5", "smpl_hdf5_v1"}:
        raise ValueError(
            f"GMR-free training requires formal_smpl_hdf5, got {backend!r}"
        )
    if str(motion.get("task_target_source", "")).lower() != "smpl_skeleton":
        raise ValueError(
            "GMR-free reward/termination target must be smpl_skeleton"
        )
    if str(motion.get("reset_target_source", "")).lower() != "composed":
        raise ValueError("GMR-free Canonical Default Reset must use composed")
    composed = motion.smpl_reset.composed
    expected_reset = {
        "root_xy_source": "smpl_actor_canonical_scaled",
        "root_height_source": "pose_grounded",
        "root_roll_pitch_source": "smpl_actor_canonical",
        "root_yaw_source": "smpl_actor_canonical",
        "root_linear_velocity_source": "smpl_actor_canonical_scaled_full",
        "root_angular_velocity_source": "smpl_actor_canonical_full",
        "joint_position_source": "asset_default",
        "joint_velocity_source": "zero",
    }
    actual_reset = {key: str(composed.get(key, "")) for key in expected_reset}
    if actual_reset != expected_reset:
        raise ValueError(
            "GMR-free Canonical Default Reset contract mismatch: "
            f"actual={actual_reset}, expected={expected_reset}"
        )
    prefixes = [str(value) for value in motion.cache.allowed_prefixes]
    forbidden = [value for value in prefixes if value.startswith("ref_")]
    if forbidden:
        raise ValueError(
            f"GMR-free cache cannot stage Robot-ref fields: {forbidden}"
        )
    ref_aux_weight = float(
        config.algo.config.aux_state_pred.get("w_ref_keybody_rel_pos", 0.0)
    )
    if ref_aux_weight != 0.0:
        raise ValueError(
            "GMR-free Actor loss requires w_ref_keybody_rel_pos=0, got "
            f"{ref_aux_weight}"
        )
    if _mujoco_eval_enabled(config):
        raise ValueError(
            "Legacy Robot-reference MuJoCo evaluation is forbidden in the "
            "GMR-free training entrypoint"
        )
    return {
        "schema": "gmr_free_training_contract_v1",
        "backend": backend,
        "task_target_source": "smpl_skeleton",
        "reset": actual_reset,
        "cache_allowed_prefixes": prefixes,
        "robot_h5_reads_expected": 0,
        "ref_actor_aux_weight": ref_aux_weight,
    }


def _write_gmr_free_receipt(log_dir: str | Path, receipt: dict | None) -> None:
    if receipt is None:
        return
    path = Path(log_dir) / "gmr_free_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _write_holomotion_training_receipt(
    log_dir: str | Path, receipt: dict | None
) -> None:
    if receipt is None:
        return
    path = Path(log_dir) / "holomotion_training_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _resolve_mujoco_eval_onnx_names(
    exported_dir: Path, ckpt_onnx_names
) -> list[str]:
    if not exported_dir.is_dir():
        raise FileNotFoundError(
            f"Exported ONNX directory not found: {exported_dir}"
        )
    existing = sorted([p.name for p in exported_dir.glob("*.onnx")])
    if len(existing) == 0:
        raise FileNotFoundError(f"No .onnx files found under {exported_dir}")
    existing_set = set(existing)

    if ckpt_onnx_names is None:
        return existing
    if isinstance(ckpt_onnx_names, ListConfig):
        requested = list(ckpt_onnx_names)
    elif isinstance(ckpt_onnx_names, (list, tuple)):
        requested = list(ckpt_onnx_names)
    else:
        raise TypeError(
            "mujoco_eval.ckpt_onnx_names must be a list/tuple, "
            f"got {type(ckpt_onnx_names)}"
        )
    requested_norm = []
    for name in requested:
        name_str = str(name).strip()
        if name_str == "":
            continue
        requested_norm.append(Path(name_str).name)
    if len(requested_norm) == 0:
        return existing

    selected = [name for name in requested_norm if name in existing_set]
    if len(selected) == 0:
        raise ValueError(
            "No requested ONNX checkpoints exist under exported directory. "
            f"exported_dir={exported_dir}, requested={requested_norm}, "
            f"existing={existing}"
        )
    return selected


def _exec_mujoco_eval(eval_override_dict: dict) -> None:
    cli_args = []
    for key in sorted(eval_override_dict.keys()):
        value = eval_override_dict[key]
        if value is None:
            continue
        if isinstance(value, bool):
            cli_args.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            cli_args.append(f"{key}={value}")
        elif isinstance(value, str):
            cli_args.append(f"{key}={value}")
        elif isinstance(value, (list, tuple)):
            inner = ",".join([str(v) for v in value])
            cli_args.append(f"{key}=[{inner}]")
        else:
            cli_args.append(f"{key}={value}")

    argv = [
        sys.executable,
        "-m",
        "holomotion.src.evaluation.eval_mujoco_sim2sim",
    ] + cli_args
    os.execv(sys.executable, argv)


def _phase2_eval_enabled(config: OmegaConf) -> bool:
    """Check the opt-in Phase-2 post-training evaluation switch."""
    phase2_eval = config.get("phase2_eval", None)
    return phase2_eval is not None and bool(phase2_eval.get("enabled", False))


def _exec_phase2_eval_driver(spec_path: Path) -> None:
    """Replace the rank-0 training process with the Phase-2 eval driver."""
    argv = [
        sys.executable,
        "-m",
        "holomotion.src.evaluation.phase2_auto_eval",
        "--spec",
        str(spec_path),
    ]
    os.execv(sys.executable, argv)


def _maybe_exec_phase2_eval(
    config: OmegaConf, algo, log_dir: str | Path
) -> bool:
    """Write the Phase-2 eval spec on rank 0 and hand off to its driver.

    Returns ``True`` on non-main ranks so they can return normally. On rank 0,
    a successful hand-off does not return because ``os.execv`` replaces the
    training process.
    """
    if not _phase2_eval_enabled(config):
        return False

    if not bool(algo.is_main_process):
        return True

    from holomotion.src.evaluation.phase2_auto_eval import (
        build_phase2_eval_spec,
        write_phase2_eval_spec,
    )

    phase2_eval = OmegaConf.to_container(config.phase2_eval, resolve=True)
    onnx_name_suffix = config.algo.config.get("onnx_name_suffix", None)
    spec = build_phase2_eval_spec(
        phase2_eval,
        experiment_dir=Path(log_dir),
        final_step=int(algo.current_learning_iteration),
        onnx_name_suffix=onnx_name_suffix,
    )
    spec_path = Path(log_dir).resolve() / "phase2_eval_spec.json"
    write_phase2_eval_spec(spec_path, spec)
    _exec_phase2_eval_driver(spec_path)
    return True


@hydra.main(
    config_path="../../config",
    config_name="training/train_base",
    version_base=None,
)
def main(config: OmegaConf):
    """Train the motion tracking model.

    Args:
        config: OmegaConf object containing the configuration.

    """

    # Accelerate initializes NCCL before it calls set_device(). Bind the local
    # CUDA device first so multi-node ranks cannot initialize on the wrong GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    config = compile_config(config, accelerator=None)
    gmr_free_receipt = _validate_gmr_free_contract(config)
    holomotion_training_receipt = validate_holomotion_training_contract(config)
    dist = None

    # Hydra resolves ${now:...} independently in distributed processes, so
    # experiment_save_dir can differ by rank. Initialize the process group and
    # broadcast rank 0's path so every rank writes to the same directory.
    if getattr(config, "num_processes", 1) > 1:
        project_config = ProjectConfiguration(
            project_dir=config.experiment_save_dir,
            logging_dir=config.experiment_save_dir,
        )
        _accelerator = Accelerator(project_config=project_config)
        import torch.distributed as dist

        path_list = (
            [config.experiment_save_dir]
            if _accelerator.is_main_process
            else [None]
        )
        dist.broadcast_object_list(path_list, src=0)
        config.experiment_save_dir = path_list[0]

    log_dir = config.experiment_save_dir
    if int(getattr(config, "process_id", 0)) == 0:
        _write_gmr_free_receipt(log_dir, gmr_free_receipt)
        _write_holomotion_training_receipt(
            log_dir, holomotion_training_receipt
        )
    headless = config.headless
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
    )

    algo.load(config.checkpoint)
    algo.learn()

    if _maybe_exec_phase2_eval(config, algo, log_dir):
        return

    mujoco_eval = _mujoco_eval_config(config)
    if not _mujoco_eval_enabled(config):
        return
    if not bool(config.algo.config.get("export_policy", False)):
        msg = (
            "mujoco_eval.enabled=true requires "
            "algo.config.export_policy=true to export ONNX "
            "before post-training evaluation."
        )
        raise ValueError(msg)

    if not bool(algo.is_main_process):
        os._exit(0)

    exported_dir = Path(log_dir) / "exported"
    selected_onnx_names = _resolve_mujoco_eval_onnx_names(
        exported_dir, mujoco_eval.get("ckpt_onnx_names", None)
    )
    eval_override_dict = OmegaConf.to_container(mujoco_eval, resolve=True)
    eval_override_dict.pop("enabled", None)
    eval_override_dict["ckpt_onnx_root_dir"] = str(exported_dir)
    eval_override_dict["ckpt_onnx_names"] = selected_onnx_names
    _exec_mujoco_eval(eval_override_dict)


if __name__ == "__main__":
    _run_main_fail_closed(main)
