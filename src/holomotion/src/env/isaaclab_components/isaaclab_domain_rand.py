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


from typing import Literal

import isaaclab.envs.mdp as isaaclab_mdp
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass


def _joint_id_list(
    joint_ids: slice | list[int] | tuple[int, ...] | torch.Tensor,
    *,
    num_joints: int,
) -> list[int]:
    """Materialize an Isaac Lab joint selection as asset joint indices."""
    if isinstance(joint_ids, slice):
        return list(range(num_joints))[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        values = joint_ids.detach().cpu().flatten().tolist()
    else:
        values = list(joint_ids)
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate joint ids are not supported: {result}")
    invalid = [value for value in result if not 0 <= value < num_joints]
    if invalid:
        raise ValueError(
            f"Joint ids {invalid} exceed articulation size {num_joints}."
        )
    return result


def _resolve_action_controlled_joint_mapping(
    selected_joint_ids: slice | list[int] | tuple[int, ...] | torch.Tensor,
    action_joint_ids: slice | list[int] | tuple[int, ...] | torch.Tensor,
    *,
    num_joints: int,
) -> tuple[list[int], list[int]]:
    """Map selected asset joints to columns of a possibly partial action term.

    Velocity policies may intentionally control only a subset of an
    articulation (K1: 20 body joints out of 22 total joints).  Action offsets
    are indexed in action order, not articulation order, so using asset joint
    ids directly can index past the action tensor.
    """
    selected = set(_joint_id_list(selected_joint_ids, num_joints=num_joints))
    action_ids = _joint_id_list(action_joint_ids, num_joints=num_joints)
    controlled_asset_ids: list[int] = []
    action_columns: list[int] = []
    for action_column, asset_joint_id in enumerate(action_ids):
        if asset_joint_id in selected:
            controlled_asset_ids.append(asset_joint_id)
            action_columns.append(action_column)
    if not controlled_asset_ids:
        raise ValueError(
            "default_dof_pos_bias does not select any joint controlled by "
            "the dof_pos action term."
        )
    return controlled_asset_ids, action_columns


class DomainRandFunctions:
    @staticmethod
    def _get_dr_default_dof_pos_bias(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_name: str = "robot",
        joint_names: list[str] = (".*"),
        pos_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal[
            "uniform", "log_uniform", "gaussian"
        ] = "uniform",
    ):
        asset_cfg = SceneEntityCfg(asset_name, joint_names=joint_names)
        asset_cfg.resolve(env.scene)
        asset: Articulation = env.scene[asset_name]
        # Preserve the asset default once.  Curriculum re-applications must
        # sample around the same nominal pose instead of accumulating bias on
        # top of the previous level's randomized default.
        if getattr(asset.data, "default_joint_pos_nominal", None) is None:
            asset.data.default_joint_pos_nominal = torch.clone(
                asset.data.default_joint_pos[0]
            )

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=asset.device)

        action_term = env.action_manager.get_term("dof_pos")
        controlled_asset_ids, action_columns = (
            _resolve_action_controlled_joint_mapping(
                asset_cfg.joint_ids,
                action_term._joint_ids,
                num_joints=asset.num_joints,
            )
        )
        joint_ids = torch.tensor(
            controlled_asset_ids,
            dtype=torch.long,
            device=asset.device,
        )
        action_columns_tensor = torch.tensor(
            action_columns,
            dtype=torch.long,
            device=asset.device,
        )

        if pos_distribution_params is not None:
            nominal = asset.data.default_joint_pos_nominal.to(asset.device)
            if nominal.ndim == 1:
                pos = (
                    nominal.unsqueeze(0)
                    .expand(asset.data.default_joint_pos.shape[0], -1)
                    .clone()
                )
            else:
                pos = nominal.clone()
            pos = _randomize_prop_by_op(
                pos,
                pos_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )[env_ids][:, joint_ids]

            env_grid = env_ids[:, None]
            asset.data.default_joint_pos[env_grid, joint_ids[None, :]] = pos
            if not isinstance(action_term._offset, torch.Tensor):
                raise TypeError(
                    "default_dof_pos_bias requires a tensor-valued dof_pos "
                    "offset (configure use_default_offset=True)."
                )
            action_term._offset[env_grid, action_columns_tensor[None, :]] = pos

    @staticmethod
    def _get_dr_rigid_body_com(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_name: str = "robot",
        body_names: str = "torso_link",
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        asset = env.scene[asset_name]
        current = asset.root_physx_view.get_coms().clone()
        nominal = getattr(asset.data, "default_body_com_nominal", None)
        if nominal is None:
            nominal = current.clone()
            asset.data.default_body_com_nominal = nominal
        if nominal.shape != current.shape:
            raise ValueError(
                "Nominal CoM shape changed within one articulation"
            )
        selected_envs = (
            torch.arange(current.shape[0], device="cpu")
            if env_ids is None
            else env_ids.to(device="cpu", dtype=torch.long)
        )
        body_ids = (
            torch.arange(current.shape[1], device="cpu")[asset_cfg.body_ids]
            if isinstance(asset_cfg.body_ids, slice)
            else torch.as_tensor(
                asset_cfg.body_ids, dtype=torch.long, device="cpu"
            )
        )
        bounds = torch.tensor(
            [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")],
            device="cpu",
            dtype=current.dtype,
        )
        if not torch.isfinite(bounds).all() or torch.any(
            bounds[:, 0] > bounds[:, 1]
        ):
            raise ValueError("Invalid CoM randomization bounds")
        # Same one xyz draw per environment as the SDK startup event, but
        # centered on nominal and correctly indexed for partial env selections.
        offsets = bounds[:, 0] + torch.rand(
            (len(selected_envs), 1, 3),
            device="cpu",
            dtype=current.dtype,
        ) * (bounds[:, 1] - bounds[:, 0])
        env_grid, body_grid = selected_envs[:, None], body_ids[None, :]
        current[env_grid, body_grid, :3] = (
            nominal[env_grid, body_grid, :3] + offsets
        )
        asset.root_physx_view.set_coms(current, selected_envs)

    @staticmethod
    def _get_dr_rigid_body_material(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_name: str = "robot",
        body_names: str = ".*",
        static_friction_range: tuple[float, float] | None = None,
        dynamic_friction_range: tuple[float, float] | None = None,
        restitution_range: tuple[float, float] | None = None,
        num_buckets: int = 64,
        make_consistent: bool = False,
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        eveent_cfg = EventTermCfg(
            func=isaaclab_mdp.events.randomize_rigid_body_material,
            params={
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
                "make_consistent": make_consistent,
            },
        )
        material_randomizer = (
            isaaclab_mdp.events.randomize_rigid_body_material(eveent_cfg, env)
        )
        return material_randomizer(env, env_ids, **eveent_cfg.params)

    @staticmethod
    def _get_dr_push_by_setting_velocity(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
    ):
        return isaaclab_mdp.events.push_by_setting_velocity(
            env,
            env_ids,
            velocity_range,
        )

    @staticmethod
    def _get_dr_randomize_actuator_gains(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_name: str = "robot",
        body_names: str = ".*",
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal[
            "uniform", "log_uniform", "gaussian"
        ] = "uniform",
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        return isaaclab_mdp.events.randomize_actuator_gains(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params,
            damping_distribution_params,
            operation=operation,
            distribution=distribution,
        )

    @staticmethod
    def _get_dr_randomize_mass(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_name: str = "robot",
        body_names: str = ".*",
        mass_range: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "add",
        recompute_inertia: bool = True,
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        return isaaclab_mdp.events.randomize_rigid_body_mass(
            env,
            env_ids,
            mass_distribution_params=mass_range,
            asset_cfg=asset_cfg,
            operation=operation,
            recompute_inertia=recompute_inertia,
        )

    @staticmethod
    def _get_dr_randomize_mass_by_mirror_pair(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_name: str = "robot",
        body_names: str | list[str] = ".*",
        mass_range: tuple[float, float] = (0.9, 1.1),
        recompute_inertia: bool = True,
    ):
        """Scale body mass while sharing one sample per left/right pair.

        Vita02A's reference controller randomizes bilateral link masses this
        way.  Keeping it here as a generic event avoids embedding VBot logic
        in either tracking task while retaining a physically symmetric plant.
        """
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        asset: Articulation = env.scene[asset_name]

        if env_ids is None:
            env_ids = torch.arange(
                env.scene.num_envs, dtype=torch.long, device="cpu"
            )
        else:
            env_ids = env_ids.to(dtype=torch.long, device="cpu")

        if asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(
                asset.num_bodies, dtype=torch.long, device="cpu"
            )
        else:
            body_ids = torch.as_tensor(
                asset_cfg.body_ids, dtype=torch.long, device="cpu"
            )

        masses = asset.root_physx_view.get_masses()
        device = masses.device
        tensor_env_ids = env_ids.to(device=device)
        tensor_body_ids = body_ids.to(device=device)
        scales = torch.empty(
            len(env_ids),
            len(body_ids),
            dtype=masses.dtype,
            device=device,
        ).uniform_(*mass_range)

        selected = {
            asset.body_names[int(body_id)]: local_index
            for local_index, body_id in enumerate(body_ids)
        }
        for left_name, left_local_index in selected.items():
            if not left_name.startswith("left_"):
                continue
            right_local_index = selected.get(
                "right_" + left_name.removeprefix("left_")
            )
            if right_local_index is not None:
                scales[:, right_local_index] = scales[:, left_local_index]

        defaults = asset.data.default_mass.to(device=device)
        masses[tensor_env_ids[:, None], tensor_body_ids] = (
            defaults[tensor_env_ids[:, None], tensor_body_ids] * scales
        )
        asset.root_physx_view.set_masses(masses, env_ids)

        if recompute_inertia:
            inertias = asset.root_physx_view.get_inertias()
            default_inertia = asset.data.default_inertia.to(
                device=inertias.device
            )
            inertia_env_ids = env_ids.to(device=inertias.device)
            inertia_body_ids = body_ids.to(device=inertias.device)
            inertia_scales = scales.to(device=inertias.device)
            inertias[inertia_env_ids[:, None], inertia_body_ids] = (
                default_inertia[inertia_env_ids[:, None], inertia_body_ids]
                * inertia_scales[..., None]
            )
            asset.root_physx_view.set_inertias(inertias, env_ids)


@configclass
class EventsCfg:
    pass


def build_domain_rand_config(domain_rand_config_dict: dict) -> EventsCfg:
    """Build IsaacLab-compatible EventsCfg from a config dictionary."""
    events_cfg = EventsCfg()

    for event_name, cfg in domain_rand_config_dict.items():
        # Keep non-event config under `domain_rand` available for Hydra
        # references without forcing it through the Isaac Lab event builder.
        if not (isinstance(cfg, dict) and "mode" in cfg):
            continue

        cfg = dict(cfg)
        func_name = cfg.pop("func", event_name)
        try:
            func = (
                func_name
                if callable(func_name)
                else getattr(DomainRandFunctions, f"_get_dr_{func_name}")
            )
        except AttributeError as exc:
            raise AttributeError(
                f"Unknown domain randomization function '{func_name}' "
                f"for event '{event_name}'"
            ) from exc
        term = EventTermCfg(
            func=func,
            **cfg,
        )
        setattr(events_cfg, event_name, term)

    return events_cfg
