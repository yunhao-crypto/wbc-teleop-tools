"""Explicit joint-physics contracts shared by training, export and MuJoCo.

An absent contract preserves legacy behavior. Zero friction is an explicit
modeling choice, not a conversion between PhysX load coefficients and torque
friction. This module deliberately has no Isaac Lab import or robot-name test.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

METADATA_KEY = "joint_physics_contract"
ASSET_METADATA_KEY = "training_asset_contract"


def resolve_joint_physics(value) -> dict | None:
    """Validate an opt-in, versioned nominal physics declaration."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("robot.joint_physics must be a mapping")
    allowed = {
        "schema_version",
        "joint_friction",
        "passive_damping",
        "enforce_effort_limits",
    }
    if set(value) - allowed:
        raise ValueError(
            f"Unknown joint_physics fields: {set(value) - allowed}"
        )
    if value.get("schema_version") != 1:
        raise ValueError("joint_physics.schema_version must be 1")
    if value.get("joint_friction") != "zero":
        raise ValueError(
            "This contract supports explicit zero joint friction only; "
            "nonzero models require a separately validated, unit-aware contract"
        )
    damping = float(value["passive_damping"])
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("passive_damping must be finite and nonnegative")
    enforce = value.get("enforce_effort_limits")
    if not isinstance(enforce, bool):
        raise ValueError("enforce_effort_limits must be explicitly boolean")
    return {
        "schema_version": 1,
        "joint_friction": "zero",
        "passive_damping": damping,
        "enforce_effort_limits": enforce,
    }


def validate_physics_metadata(config_value, metadata_value) -> dict | None:
    """Fail closed when a model and evaluator declare different physics."""
    requested = resolve_joint_physics(config_value)
    if isinstance(metadata_value, str):
        metadata_value = json.loads(metadata_value)
    exported = resolve_joint_physics(metadata_value)
    if requested != exported:
        raise ValueError(
            "Joint physics mismatch between resolved config and ONNX: "
            f"config={requested}, model={exported}. Use the model's original "
            "resolved configuration; do not silently rebind a legacy policy."
        )
    return requested


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _summary(value) -> dict:
    values = _numpy(value)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Runtime physics readback is empty or non-finite")
    return {
        "shape": list(values.shape),
        "env0": values[0].tolist(),
        "min_over_envs": values.min(axis=0).tolist(),
        "max_over_envs": values.max(axis=0).tolist(),
    }


def _assert_value(label: str, actual, expected: float, *, atol=1e-8) -> None:
    values = _numpy(actual)
    if not np.isfinite(values).all() or not np.allclose(
        values, expected, atol=atol, rtol=0.0
    ):
        raise ValueError(f"Actual backend {label} differs from {expected}")


def apply_isaac_joint_physics(robot, declaration) -> dict | None:
    """Write both PhysX friction APIs and verify actual backend tensors.

    Call once after articulation/manager initialization and before the first
    policy reset/rollout. Software PD gains and default PD metadata are not
    changed. This bypasses affected SDK wrappers, not the physics backend.
    """
    import torch

    contract = resolve_joint_physics(declaration)
    if contract is None:
        return None
    if any(act.is_implicit_model for act in robot.actuators.values()):
        raise ValueError(
            "The explicit passive-damping contract cannot overwrite implicit "
            "PD drive damping; use an explicit actuator model"
        )
    view = robot.root_physx_view
    legacy = view.get_dof_friction_coefficients().clone()
    indices = torch.arange(legacy.shape[0], dtype=torch.int32, device="cpu")
    view.set_dof_friction_coefficients(torch.zeros_like(legacy), indices)
    if hasattr(view, "get_dof_friction_properties"):
        properties = view.get_dof_friction_properties().clone()
        view.set_dof_friction_properties(torch.zeros_like(properties), indices)
        _assert_value(
            "friction_properties", view.get_dof_friction_properties(), 0.0
        )
    _assert_value("legacy friction", view.get_dof_friction_coefficients(), 0.0)

    # Keep SDK diagnostic/default friction buffers consistent with the backend.
    # Do not touch default_joint_damping or actuator.damping: those store KD.
    for field in (
        "joint_friction_coeff",
        "default_joint_friction_coeff",
        "joint_dynamic_friction_coeff",
        "default_joint_dynamic_friction_coeff",
        "joint_viscous_friction_coeff",
        "default_joint_viscous_friction_coeff",
    ):
        buffer = getattr(robot.data, field, None)
        if isinstance(buffer, torch.Tensor):
            buffer.zero_()
    for actuator in robot.actuators.values():
        for field in ("friction", "dynamic_friction", "viscous_friction"):
            buffer = getattr(actuator, field, None)
            if isinstance(buffer, torch.Tensor):
                buffer.zero_()
    robot.write_joint_damping_to_sim(contract["passive_damping"])
    _assert_value(
        "passive damping",
        view.get_dof_dampings(),
        contract["passive_damping"],
        atol=1e-6,
    )
    return contract


def isaac_joint_physics_snapshot(robot) -> dict:
    """Read all environments, preserving names and explicit units."""
    view = robot.root_physx_view
    fields = {
        "legacy_joint_friction_dimensionless": "get_dof_friction_coefficients",
        "drive_damping_nm_s_per_rad": "get_dof_dampings",
        "drive_stiffness_nm_per_rad": "get_dof_stiffnesses",
        "armature_kg_m2": "get_dof_armatures",
        "effort_limit_nm": "get_dof_max_forces",
        "velocity_limit_rad_s": "get_dof_max_velocities",
        "joint_limits_rad": "get_dof_limits",
        "body_mass_kg": "get_masses",
        "body_inertia_kg_m2": "get_inertias",
        "body_com_pose": "get_coms",
    }
    result = {
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "backend": {
            key: _summary(getattr(view, method)())
            for key, method in fields.items()
        },
        "actuators": {},
    }
    if hasattr(view, "get_dof_friction_properties"):
        result["backend"]["friction_properties"] = {
            "columns": [
                "static_effort_nm",
                "dynamic_effort_nm",
                "viscous_nm_s_per_rad",
            ],
            **_summary(view.get_dof_friction_properties()),
        }
    if hasattr(view, "get_material_properties"):
        result["backend"]["contact_material_static_dynamic_restitution"] = (
            _summary(view.get_material_properties())
        )
    for name, actuator in robot.actuators.items():
        result["actuators"][name] = {
            "class": type(actuator).__name__,
            "implicit_pd": bool(actuator.is_implicit_model),
            "joint_names": list(actuator.joint_names),
            "kp_nm_per_rad": _summary(actuator.stiffness),
            "kd_nm_s_per_rad": _summary(actuator.damping),
        }
    return result


def _named_parameter(value, joint_names):
    """Resolve a declared scalar/regex mapping, rejecting gaps and ambiguity."""
    if isinstance(value, Mapping):
        resolved = []
        for name in joint_names:
            matches = [
                v
                for pattern, v in value.items()
                if re.fullmatch(pattern, name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one parameter match for {name}: {matches}"
                )
            resolved.append(matches[0])
        return np.asarray(resolved, dtype=np.float64)
    if value is None:
        raise ValueError(
            "Opt-in physics validation requires explicit actuator parameters"
        )
    return np.full(len(joint_names), float(value))


def validate_isaac_actuator_config(robot, env=None, *, atol=1e-6, rtol=1e-5):
    """Compare backend limits/armature and software PD against effective cfg.

    For uniform startup gain DR, check the declared interval on every environment.
    This does not compare software KD to the physical drive damping.
    """
    gain_event = None
    if env is not None and hasattr(env, "event_manager"):
        for mode, names in env.event_manager.active_terms.items():
            for name in names:
                term = env.event_manager.get_term_cfg(name)
                if getattr(term.func, "__name__", "") not in (
                    "_get_dr_randomize_actuator_gains",
                    "randomize_actuator_gains",
                ):
                    continue
                if gain_event is not None or mode != "startup":
                    raise ValueError(
                        "Validate multiple/non-startup gain events explicitly"
                    )
                gain_event = term.params
                if (
                    gain_event.get("body_names", ".*") != ".*"
                    or gain_event.get("distribution", "uniform") != "uniform"
                ):
                    raise ValueError(
                        "Unsupported gain DR support for runtime validation"
                    )
    backend = robot.root_physx_view
    report = {}
    for name, actuator in robot.actuators.items():
        names = list(actuator.joint_names)
        indices = [list(robot.joint_names).index(joint) for joint in names]
        checks = {}
        for field, getter in (
            ("armature", "get_dof_armatures"),
            ("effort_limit_sim", "get_dof_max_forces"),
            ("velocity_limit_sim", "get_dof_max_velocities"),
        ):
            expected = _named_parameter(getattr(actuator.cfg, field), names)
            actual = _numpy(getattr(backend, getter)())[:, indices]
            if not np.allclose(actual, expected[None], atol=atol, rtol=rtol):
                raise ValueError(
                    f"Backend {name}.{field} differs from actuator cfg"
                )
            checks[field] = {
                "max_abs_error": float(np.max(np.abs(actual - expected[None])))
            }
        for field in ("stiffness", "damping"):
            nominal = _named_parameter(getattr(actuator.cfg, field), names)
            low, high = nominal.copy(), nominal.copy()
            params = (
                None
                if gain_event is None
                else gain_event.get(field + "_distribution_params")
            )
            if params is not None:
                operation = gain_event.get("operation", "abs")
                if operation == "scale":
                    low, high = nominal * params[0], nominal * params[1]
                elif operation == "add":
                    low, high = nominal + params[0], nominal + params[1]
                elif operation == "abs":
                    low, high = (
                        np.full_like(nominal, params[0]),
                        np.full_like(nominal, params[1]),
                    )
                else:
                    raise ValueError(
                        f"Unsupported gain operation: {operation}"
                    )
            actual = _numpy(getattr(actuator, field))
            slack = atol + rtol * np.maximum(np.abs(low), np.abs(high))
            if (
                not np.isfinite(actual).all()
                or np.any(actual < low - slack)
                or np.any(actual > high + slack)
            ):
                raise ValueError(
                    f"Software PD {name}.{field} outside declared nominal/DR interval"
                )
            checks[field] = {
                "nominal": nominal.tolist(),
                "declared_low": low.tolist(),
                "declared_high": high.tolist(),
                "passed": True,
            }
        report[name] = checks
    return report


def _identity(path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def validate_policy_io_metadata(config, metadata, input_shape, output_shape):
    """Opt-in inference contract, using declared named controller/action sets.

    Controller state may use the simulator's order: it is mapped by name.
    Policy output order must match the training action term exactly. This
    distinction prevents rejecting a valid PhysX/MuJoCo order conversion.
    """
    robot = config["robot"]
    if resolve_joint_physics(robot.get("joint_physics")) is None:
        return None
    declared = list(
        config["env"]["config"]["actions"]["dof_pos"]["params"]["joint_names"]
    )
    exported = list(metadata["action_joint_names"])
    if exported != declared:
        raise ValueError(
            "Policy action joint order differs from the training action contract"
        )
    full = list(metadata["joint_names"])
    if len(full) != len(set(full)) or set(full) != set(robot["dof_names"]):
        raise ValueError(
            "Policy controller joint set differs from robot configuration"
        )
    for key in ("kps", "kds", "default_joint_pos", "action_scale"):
        if not np.isfinite(np.asarray(metadata[key], dtype=float)).all():
            raise ValueError(f"Non-finite policy metadata: {key}")
    expected_pose = np.array(
        [robot["init_state"]["default_joint_angles"][n] for n in full]
    )
    if not np.allclose(
        metadata["default_joint_pos"], expected_pose, atol=1e-6, rtol=1e-5
    ):
        raise ValueError(
            "Policy nominal joint pose differs from the training configuration"
        )
    contract = config.get("velocity_policy_contract") or config.get(
        "gmr_free_contract"
    )
    if contract is None or "actor_observation_dim" not in contract:
        raise ValueError(
            "Opt-in policy validation needs a declared actor observation dimension"
        )
    expected_input = int(contract["actor_observation_dim"])
    if input_shape[-1] != expected_input or output_shape[-1] != len(declared):
        raise ValueError(
            f"Policy IO dimensions differ from training: {input_shape}, {output_shape}"
        )
    return {
        "actor_observation_dim": expected_input,
        "action_dim": len(declared),
        "controller_dim": len(full),
        "status": "passed",
    }


def initialize_joint_physics(
    env,
    robot_config,
    log_dir,
    rank: int,
    *,
    phase="after_manager_startup_before_policy_rollout",
) -> dict:
    """Shared Velocity/Motion initialization and rank-specific receipt."""
    robot = env.scene["robot"]
    contract = apply_isaac_joint_physics(
        robot, robot_config.get("joint_physics")
    )
    env.joint_physics_contract = contract
    if contract is not None:
        from holomotion.src.env.urdf_assets import xml_asset_fingerprint

        env.training_asset_contract = {
            "urdf": xml_asset_fingerprint(robot_config["asset"]["urdf_file"]),
            "mujoco": xml_asset_fingerprint(
                robot_config["asset"]["assetFileName"]
            ),
        }
    snapshot = isaac_joint_physics_snapshot(robot)
    if contract is not None:
        snapshot["actuator_config_checks"] = validate_isaac_actuator_config(
            robot, env
        )
    spawn = robot.cfg.spawn
    asset_files = [robot_config["asset"]["urdf_file"]]
    for field in ("asset_path", "usd_path"):
        path = getattr(spawn, field, None)
        if path:
            asset_files.append(path)
    usd_dir = getattr(spawn, "usd_dir", None)
    if usd_dir:
        asset_files.extend(sorted(Path(usd_dir).rglob("*.usd")))
    snapshot.update(
        {
            "schema": "holomotion_runtime_physics_v1",
            "phase": phase,
            "rank": int(rank),
            "contract": contract,
            "training_asset_contract": getattr(
                env, "training_asset_contract", None
            ),
            "module_file": str(Path(__file__).resolve()),
            "physics_dt_s": float(env.cfg.sim.dt),
            "control_decimation": int(env.cfg.decimation),
            "assets": [
                _identity(path)
                for path in sorted({str(p) for p in asset_files})
            ],
            "spawn": {
                "merge_fixed_joints": getattr(
                    spawn, "merge_fixed_joints", None
                ),
                "replace_cylinders_with_capsules": getattr(
                    spawn, "replace_cylinders_with_capsules", None
                ),
                "enabled_self_collisions": spawn.articulation_props.enabled_self_collisions,
            },
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "isaaclab", "isaacsim")
            },
        }
    )
    if log_dir is not None:
        output = Path(log_dir) / f"runtime_physics_rank_{int(rank)}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
        )
    env.runtime_physics_receipt = snapshot
    return snapshot


def apply_mujoco_joint_physics(model, declaration, joint_names) -> dict | None:
    """Apply the same nominal model without modifying source MJCF files."""
    contract = resolve_joint_physics(declaration)
    if contract is None:
        return None
    names = list(joint_names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate joint names in joint physics contract")
    indices = []
    for name in names:
        joint = model.joint(name)
        if (
            int(joint.type[0]) != 3
        ):  # mjJNT_HINGE; contract units are rotational.
            raise ValueError(
                f"Joint physics contract requires hinge joint: {name}"
            )
        indices.append(int(joint.dofadr[0]))
    model.dof_frictionloss[indices] = 0.0
    model.dof_damping[indices] = contract["passive_damping"]
    _assert_value("MuJoCo frictionloss", model.dof_frictionloss[indices], 0.0)
    _assert_value(
        "MuJoCo passive damping",
        model.dof_damping[indices],
        contract["passive_damping"],
    )
    return {
        "contract": contract,
        "joint_names": names,
        "frictionloss_nm": model.dof_frictionloss[indices].tolist(),
        "passive_damping_nm_s_per_rad": model.dof_damping[indices].tolist(),
    }


def align_mujoco_effort_limits(model, declaration, joint_names, efforts):
    """Align all three compiled limits, only for declared unit-gear torque motors.

    Motor-space projection remains the controller's responsibility. These
    outer joint-space safety limits must not add a stale, tighter XML clamp.
    Legacy policies without the opt-in declaration are deliberately unchanged.
    """
    contract = resolve_joint_physics(declaration)
    if contract is None or not contract["enforce_effort_limits"]:
        return None
    names = list(joint_names)
    limits = np.asarray(efforts, dtype=np.float64)
    if (
        len(names) != model.nu
        or len(set(names)) != len(names)
        or limits.shape != (model.nu,)
        or not np.isfinite(limits).all()
        or np.any(limits <= 0)
    ):
        raise ValueError(
            "Effort contract requires one finite positive limit per motor"
        )
    joint_ids = []
    for motor, name in enumerate(names):
        joint = model.joint(name)
        joint_ids.append(joint.id)
        if (
            int(joint.type[0]) != 3
            or int(model.actuator_trntype[motor]) != 0
            or int(model.actuator_trnid[motor, 0]) != joint.id
            or not np.array_equal(
                model.actuator_gear[motor], [1, 0, 0, 0, 0, 0]
            )
            or int(model.actuator_dyntype[motor]) != 0
            or int(model.actuator_gaintype[motor]) != 0
            or float(model.actuator_gainprm[motor, 0]) != 1.0
            or int(model.actuator_biastype[motor]) != 0
        ):
            raise ValueError(
                f"Effort contract requires named unit-gear torque motor: {name}"
            )
    before = {
        "ctrlrange": model.actuator_ctrlrange.tolist(),
        "forcerange_nm": model.actuator_forcerange.tolist(),
        "joint_actfrcrange_nm": model.jnt_actfrcrange[joint_ids].tolist(),
    }
    ranges = np.column_stack((-limits, limits))
    model.actuator_ctrllimited[:] = 1
    model.actuator_forcelimited[:] = 1
    model.jnt_actfrclimited[joint_ids] = 1
    model.actuator_ctrlrange[:] = ranges
    model.actuator_forcerange[:] = ranges
    model.jnt_actfrcrange[joint_ids] = ranges
    return {
        "joint_names": names,
        "before": before,
        "ctrlrange_nm": model.actuator_ctrlrange.tolist(),
        "forcerange_nm": model.actuator_forcerange.tolist(),
        "joint_actfrcrange_nm": model.jnt_actfrcrange[joint_ids].tolist(),
    }
