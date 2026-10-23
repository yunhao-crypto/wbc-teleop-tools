"""Inspect a robot's actual Isaac joint physics without a policy or training.

Accepts a robot YAML or a saved resolved run YAML. Source assets are read-only;
generated USD and before/after receipts live only in the requested output dir.
Run with the project training interpreter from the HoloMotion repository root.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def inspect_reset_contacts(
    robot,
    sim,
    robot_config,
    output,
    samples,
    radius,
    bias_radius,
    profile,
    counterexample=None,
):
    """Single-step, zero-gravity contact probe; not a closed-loop policy test."""
    import numpy as np
    import torch
    import xml.etree.ElementTree as ET

    # Match the actual merged articulation mass, not the unmerged link count.
    urdf = ET.parse(robot_config["asset"]["urdf_file"]).getroot()
    source_mass = sum(
        float(node.get("value")) for node in urdf.findall("link/inertial/mass")
    )
    actual_mass = float(robot.root_physx_view.get_masses().sum())
    mass_error = abs(actual_mass - source_mass) / source_mass
    if mass_error > profile["asset"]["mass_relative_error"]:
        raise ValueError(
            f"Merged mass mismatch: {source_mass} versus {actual_mass}"
        )
    paths = list(robot.root_physx_view.link_paths[0])
    view = robot._physics_sim_view.create_rigid_contact_view(
        robot.cfg.prim_path + "/*",
        filter_patterns=paths,
        max_contact_data_count=65536,
    )
    seed = profile["seeds"]["reset"]
    tolerance = profile["collision"]["penetration_tolerance_m"]
    rng = np.random.default_rng(seed)
    default = robot.data.default_joint_pos.clone()
    root = robot.data.default_root_state.clone()
    root[:, 2] = 2.0
    root[:, 7:] = 0.0
    limits = robot.data.soft_joint_pos_limits
    report = {
        "scope": "single physics step after written reset; zero gravity, no ground, no policy or PD force",
        "seed": seed,
        "acceptance_profile": profile["name"],
        "requested_samples": samples,
        "tested_samples": 0,
        "joint_reset_radius_rad": radius,
        "default_bias_radius_rad": bias_radius,
        "source_mass_kg": source_mass,
        "merged_mass_kg": actual_mass,
        "mass_relative_error": mass_error,
        "penetration_threshold_m": tolerance,
        "sensor_paths": view.sensor_paths,
        "filter_paths": view.filter_paths,
        "failure": None,
    }
    for sample in range(samples):
        noise = rng.uniform(-radius, radius, default.shape) + rng.uniform(
            -bias_radius, bias_radius, default.shape
        )
        q = default + torch.as_tensor(
            noise, dtype=default.dtype, device=default.device
        )
        if counterexample is not None:
            for name, angle in zip(
                counterexample["joint_names"],
                counterexample["joint_position_rad"],
                strict=True,
            ):
                q[:, robot.joint_names.index(name)] = float(angle)
        q = torch.clamp(q, limits[..., 0], limits[..., 1])
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        # Do not execute the delayed PD actuator here: inspect the written pose.
        robot.root_physx_view.set_dof_actuation_forces(
            torch.zeros_like(q),
            torch.arange(
                robot.num_instances, device=robot.device, dtype=torch.int32
            ),
        )
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        forces, points, normals, distances, counts, starts = (
            view.get_contact_data(sim.get_physics_dt())
        )
        counts_np, starts_np = counts.cpu().numpy(), starts.cpu().numpy()
        bad = []
        for sensor, target in np.argwhere(counts_np > 0):
            start, count = (
                int(starts_np[sensor, target]),
                int(counts_np[sensor, target]),
            )
            if start < 0 or start + count > view.max_contact_data_count:
                raise RuntimeError("Contact report capacity exceeded")
            minimum = float(distances[start : start + count].min())
            if minimum < -tolerance:
                bad.append(
                    {
                        "sensor": view.sensor_paths[sensor],
                        "filter": view.filter_paths[sensor][target],
                        "separation_m": minimum,
                        "max_force_n": float(
                            forces[start : start + count].abs().max()
                        ),
                    }
                )
        report["tested_samples"] += 1
        if bad:
            report["failure"] = {
                "sample": sample,
                "contacts": bad,
                "joint_names": robot.joint_names,
                "written_joint_pos": q.cpu().tolist(),
            }
            break
        if (sample + 1) % 1000 == 0:
            print(f"RESET_CONTACT_PROBE {sample + 1}/{samples}", flush=True)
    report["expected_penetration"] = counterexample is not None
    report["passed"] = bool(report["failure"]) == (counterexample is not None)
    filename = (
        "counterexample_contacts_physx.json"
        if counterexample is not None
        else "reset_contacts_physx.json"
    )
    (output / filename).write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(
            f"Contact probe did not match its declared expectation; see {output / filename}"
        )


def inspect_runtime_fk(robot, sim, robot_config, output, profile):
    import torch
    from holomotion.src.training.reference_fk import TrainingReferenceFK
    from holomotion.src.utils.torch_utils import matrix_from_quat

    robot.update(sim.get_physics_dt())
    fk = TrainingReferenceFK(
        robot_config["asset"]["urdf_file"], dtype=torch.float64
    )
    indices = [robot.joint_names.index(name) for name in fk.dof_names]
    root_pos = robot.data.root_pos_w.cpu().double()[:, None].repeat(1, 2, 1)
    root_quat = robot.data.root_quat_w.cpu().double()[:, None].repeat(1, 2, 1)
    q = (
        robot.data.joint_pos[:, indices]
        .cpu()
        .double()[:, None]
        .repeat(1, 2, 1)
    )
    result = fk(root_pos, root_quat, q, fps=50.0, quat_format="wxyz")
    bodies = [fk.body_names.index(name) for name in robot.body_names]
    expected_pos = result["global_translation"][:, 0, bodies]
    expected_quat = result["global_rotation_quat"][:, 0, bodies][
        ..., [3, 0, 1, 2]
    ]
    actual_pos = robot.data.body_pos_w.cpu().double()
    actual_quat = robot.data.body_quat_w.cpu().double()
    position_error = torch.linalg.vector_norm(
        actual_pos - expected_pos, dim=-1
    )
    relative_rotation = matrix_from_quat(actual_quat).transpose(
        -1, -2
    ) @ matrix_from_quat(expected_quat)
    rotation_error = torch.acos(
        torch.clamp(
            (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2,
            -1,
            1,
        )
    )
    report = {
        "body_names": robot.body_names,
        "position_error_m": position_error.tolist(),
        "rotation_error_rad": rotation_error.tolist(),
        "joint_names": robot.joint_names,
        "actual_joint_pos": robot.data.joint_pos.cpu().tolist(),
    }
    (output / "runtime_fk.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if (
        float(position_error.max()) > profile["asset"]["fk_position_error_m"]
        or float(rotation_error.max())
        > profile["asset"]["fk_orientation_error_rad"]
    ):
        raise ValueError("Actual PhysX/FK frame mismatch; see runtime_fk.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reset-samples", type=int, default=0)
    parser.add_argument("--joint-reset-radius", type=float, default=0.0)
    parser.add_argument("--default-bias-radius", type=float, default=0.0)
    parser.add_argument("--counterexample-json", type=Path)
    parser.add_argument("--check-fk", action="store_true")
    parser.add_argument("--acceptance-profile", type=Path)
    parser.add_argument(
        "--check-com-body",
        help="Optionally test nominal-centered CoM writes on this body",
    )
    args = parser.parse_args()
    checks_requested = (
        args.check_fk
        or args.check_com_body
        or args.reset_samples
        or args.counterexample_json
    )
    if checks_requested and args.acceptance_profile is None:
        parser.error(
            "Physical checks require an explicit --acceptance-profile"
        )
    from holomotion.src.utils.acceptance_profile import load_acceptance_profile

    profile = (
        load_acceptance_profile(args.acceptance_profile)
        if args.acceptance_profile
        else None
    )
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite diagnostic output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    if profile is not None:
        (output / "acceptance_profile.json").write_text(
            json.dumps(profile, indent=2) + "\n"
        )
    source = OmegaConf.load(args.config)
    # Motion data/model fields are not needed for a physical articulation.
    # Resolve the remaining fields in the original root interpolation context.
    robot_config = {}
    for key in source.robot:
        if key == "motion":
            continue
        value = source.robot[key]
        robot_config[key] = (
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else value
        )
    robot_config["asset"]["usd_cache_dir"] = str(output / "usd")

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True,
        device=args.device,
        enable_cameras=False,
        kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false",
    )
    try:
        from easydict import EasyDict
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from holomotion.src.env.isaaclab_components.isaaclab_scene import (
            SceneFunctions,
        )
        from holomotion.src.utils.joint_physics import (
            initialize_joint_physics,
            isaac_joint_physics_snapshot,
        )
        import holomotion.src.utils.joint_physics as implementation

        expected_root = Path(__file__).resolve().parents[3]
        if (
            not Path(implementation.__file__)
            .resolve()
            .is_relative_to(expected_root)
        ):
            raise RuntimeError(
                f"Wrong implementation imported: {implementation.__file__}"
            )
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device)
        if args.reset_samples or args.counterexample_json:
            sim_cfg.gravity = (0.0, 0.0, 0.0)
        sim = sim_utils.SimulationContext(sim_cfg)
        cfg = SceneFunctions.build_robot_config(EasyDict(robot_config), {})
        cfg.prim_path = "/World/InspectedRobot"
        # Suspend the asset to separate joint-property inspection from floor contact.
        cfg.init_state.pos = (0.0, 0.0, 2.0)
        robot = Articulation(cfg)
        sim.reset()
        before = isaac_joint_physics_snapshot(robot)
        (output / "before.json").write_text(
            json.dumps(before, indent=2, allow_nan=False) + "\n"
        )
        env = SimpleNamespace(
            scene={"robot": robot},
            cfg=SimpleNamespace(sim=sim_cfg, decimation=4),
        )
        after = initialize_joint_physics(
            env,
            robot_config,
            output,
            0,
            phase="standalone_articulation_before_rollout",
        )
        if args.check_fk:
            inspect_runtime_fk(robot, sim, robot_config, output, profile)
        if args.counterexample_json:
            inspect_reset_contacts(
                robot,
                sim,
                robot_config,
                output,
                1,
                0.0,
                0.0,
                profile,
                json.loads(args.counterexample_json.read_text()),
            )
        if args.reset_samples:
            inspect_reset_contacts(
                robot,
                sim,
                robot_config,
                output,
                args.reset_samples,
                args.joint_reset_radius,
                args.default_bias_radius,
                profile,
            )
        if args.check_com_body:
            import torch
            from holomotion.src.env.isaaclab_components.isaaclab_domain_rand import (
                DomainRandFunctions,
            )

            original_com = robot.root_physx_view.get_coms().clone()
            body_ids, body_names = robot.find_bodies(args.check_com_body)
            if len(body_ids) != 1:
                raise ValueError(
                    f"Expected exactly one CoM body, got {body_names}"
                )
            measurements = []
            physics_targets = profile["physics"]
            offsets = physics_targets["com_offsets_m"]
            for offset in (
                *offsets,
                *([offsets[0]] * physics_targets["com_repeat_count"]),
                0.0,
            ):
                DomainRandFunctions._get_dr_rigid_body_com(
                    env,
                    None,
                    {"x": (offset, offset)},
                    body_names=args.check_com_body,
                )
                actual = robot.root_physx_view.get_coms().clone()
                expected = original_com.clone()
                expected[:, body_ids, 0] += offset
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=physics_targets["com_drift_atol_m"],
                    rtol=0,
                )
                measurements.append(
                    {
                        "requested_m": offset,
                        "actual_m": float(
                            (actual - original_com)[0, body_ids[0], 0]
                        ),
                    }
                )
            (output / "com_resampling.json").write_text(
                json.dumps(
                    {
                        "body_names": body_names,
                        "measurements": measurements,
                        "atol_m": physics_targets["com_drift_atol_m"],
                        "restored_nominal": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        receipt = {
            "source_config": str(args.config.resolve()),
            "implementation": implementation.__file__,
            "scope": "local articulation inspection; no PPO, cloud readback or hardware",
            "contract": after["contract"],
            "before": "before.json",
            "after": "runtime_physics_rank_0.json",
        }
        (output / "result.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(
            "JOINT_PHYSICS_INSPECTION_PASSED", json.dumps(receipt), flush=True
        )
    except Exception:
        failure = traceback.format_exc()
        (output / "error.txt").write_text(failure)
        print(failure, flush=True)
        # Kit shutdown can terminate Python with code zero while an exception
        # is unwinding. All diagnostic files are closed; preserve failure status.
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        # No renderer or replicator writer is used by this diagnostic.
        # If Kit hangs during cleanup, expose the exact stack to the supervisor.
        faulthandler.dump_traceback_later(20)
        if "sim" in locals():
            # Unsubscribe Isaac Lab's STOP callback before closing the stage;
            # that callback otherwise waits for PLAY forever in a headless CLI.
            sim.clear_all_callbacks()
            sim.clear_instance()
        launcher.app.close(wait_for_replicator=False)
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
