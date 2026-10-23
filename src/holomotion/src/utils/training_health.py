"""Opt-in numerical/production-start gates; no changes to PPO objectives.

Receipts are rank-local. All ranks must pass a milestone before proceeding.
Memory is sampled after update, before checkpoint/export, at the same phase.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import torch
from loguru import logger

from holomotion.src.utils.acceptance_profile import load_acceptance_profile


def memory_growth(samples, window):
    """Compare disjoint steady windows at the caller's frozen gate point."""
    if len(samples) < 2 * window:
        raise ValueError("Not enough same-phase memory samples")
    previous = sum(samples[-2 * window : -window]) / window
    recent = sum(samples[-window:]) / window
    return 0.0 if previous == recent == 0 else (recent - previous) / previous


def normalize_memory_growth_action(value):
    """Validate how a steady-memory threshold breach is handled."""
    action = str(value).strip().lower()
    if action not in {"error", "warn"}:
        raise ValueError(
            "memory_growth_action must be either 'error' or 'warn', "
            f"got {value!r}"
        )
    return action


def optimizer_snapshot(optimizer):
    states, tensors, elements, finite, steps = 0, 0, 0, True, []
    for state in optimizer.state.values():
        if not state:
            continue
        states += 1
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError("Expected allocated Adam/AdamW moment states")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensors += 1
                elements += value.numel()
                finite = finite and bool(torch.isfinite(value).all())
                if key == "step":
                    steps.append(float(value.item()))
    if states == 0 or not finite or not steps or min(steps) <= 0:
        raise RuntimeError("Adam states absent, non-finite, or never updated")
    return dict(
        parameter_states=states,
        tensors=tensors,
        elements=elements,
        finite=finite,
        min_step=min(steps),
        max_step=max(steps),
    )


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def finite_or_raise(values, label):
    for name, value in values.items():
        if value is None:
            continue
        ok = (
            bool(torch.isfinite(value).all())
            if isinstance(value, torch.Tensor)
            else math.isfinite(float(value))
        )
        if not ok:
            raise FloatingPointError(f"{label}: non-finite {name}")


class TrainingHealthMonitor:
    """Configured explicitly under algo.config.runtime_validation."""

    def __init__(self, owner, config):
        self.owner, self.config = owner, config
        self.profile = load_acceptance_profile(
            Path(str(config["acceptance_profile"]))
        )
        self.root = Path(owner.log_dir)
        self.rank = int(owner.accelerator.process_index)
        self.world_size = int(owner.accelerator.num_processes)
        self.milestones = set(
            int(x) for x in config.get("milestones", [2, 100])
        )
        self.check_every = int(config.get("optimizer_check_interval", 100))
        functional = self.profile["functional"]
        self.memory_window = int(functional["memory_windows"])
        self.memory_growth_limit = float(
            functional["allocated_memory_growth_ratio_max"]
        )
        self.memory_growth_action = normalize_memory_growth_action(
            config.get("memory_growth_action", "error")
        )
        self.memory_growth_check_iterations = {
            int(x)
            for x in config.get(
                "memory_growth_check_iterations",
                [functional["debug_updates"]],
            )
        }
        if any(x <= 0 for x in self.memory_growth_check_iterations):
            raise ValueError(
                "memory_growth_check_iterations must contain positive integers"
            )
        self.samples, self.gradient_stats, self.sigma_stats = [], [], []
        self.last_time = time.monotonic()
        self.last_row = None
        expected_world = config.get("expected_world_size")
        expected_envs = config.get("expected_envs_per_rank")
        if expected_world is not None and self.world_size != int(
            expected_world
        ):
            raise RuntimeError(
                f"Wrong world size: {self.world_size} != {expected_world}"
            )
        if expected_envs is not None and int(owner.env.num_envs) != int(
            expected_envs
        ):
            raise RuntimeError(
                f"Wrong envs/rank: {owner.env.num_envs} != {expected_envs}"
            )
        self.identity = dict(
            rank=self.rank,
            world_size=self.world_size,
            local_rank=int(owner.local_rank),
            hostname=socket.gethostname(),
            num_envs_per_rank=int(owner.env.num_envs),
            device=str(owner.device),
            module_file=__file__,
            node_rank=os.environ.get(
                "NODE_RANK", os.environ.get("MACHINE_RANK")
            ),
        )
        config_text = str(owner.env_config) + str(owner.config)
        self.identity["runtime_config_sha256"] = hashlib.sha256(
            config_text.encode()
        ).hexdigest()
        source_root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in sorted(source_root.rglob("*.py")):
            digest.update(str(source.relative_to(source_root)).encode())
            digest.update(source.read_bytes())
        self.identity["source_tree_sha256"] = digest.hexdigest()
        physics = getattr(owner.env._env, "runtime_physics_receipt", None)
        self.identity["training_asset_contract"] = (
            None if physics is None else physics.get("training_asset_contract")
        )
        self.identity["physics_receipt_sha256"] = (
            None
            if physics is None
            else hashlib.sha256(
                json.dumps(physics, sort_keys=True).encode()
            ).hexdigest()
        )
        self._write("identity", self.identity)

    def _write(self, name, payload):
        target = (
            self.root / f"training_health_{name}_rank_{self.rank:04d}.json"
        )
        target.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )

    def before_step(
        self, *, actor_loss, critic_loss, sigma, actor_grad, critic_grad
    ):
        values = dict(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            sigma=sigma,
            actor_grad=actor_grad,
            critic_grad=critic_grad,
        )
        # Rank consensus prevents a healthy rank entering optimizer collectives
        # while another exits on a local numerical failure.
        valid = all(bool(torch.isfinite(x).all()) for x in values.values())
        valid = valid and bool((sigma > 0).all())
        flag = torch.tensor(int(valid), device=self.owner.device)
        if self.world_size > 1:
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MIN
            )
        if not int(flag.item()):
            raise FloatingPointError(
                "Non-finite loss/gradient/sigma or non-positive sigma on "
                "a rank"
            )
        self.gradient_stats.append([float(actor_grad), float(critic_grad)])
        s = sigma.detach().float()
        self.sigma_stats.append(
            [float(s.min()), float(s.mean()), float(s.max())]
        )

    def after_update(self, iteration, loss):
        owner = self.owner
        finite_or_raise(loss, "PPO update")
        if not self.gradient_stats:
            raise RuntimeError(
                "No optimizer steps recorded by the health hook"
            )
        invalid = 0
        robot = owner.env._env.scene["robot"]
        for actuator in robot.actuators.values():
            count = getattr(
                actuator, "parallel_ankle_invalid_state_count", None
            )
            if count is not None:
                invalid += int(count.sum().item())
        if invalid:
            raise RuntimeError(f"Unsafe actuator projection states: {invalid}")
        cuda = owner.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(owner.device)
        elapsed = time.monotonic() - self.last_time
        allocated = (
            int(torch.cuda.memory_allocated(owner.device)) if cuda else 0
        )
        row = dict(
            iteration=int(iteration),
            rank=self.rank,
            loss=loss,
            actor_gradient_norm_max=max(x[0] for x in self.gradient_stats),
            critic_gradient_norm_max=max(x[1] for x in self.gradient_stats),
            sigma_min=min(x[0] for x in self.sigma_stats),
            sigma_mean=sum(x[1] for x in self.sigma_stats)
            / len(self.sigma_stats),
            sigma_max=max(x[2] for x in self.sigma_stats),
            optimizer_steps=len(self.gradient_stats),
            allocated_bytes=allocated,
            reserved_bytes=int(torch.cuda.memory_reserved(owner.device))
            if cuda
            else 0,
            max_allocated_bytes=int(
                torch.cuda.max_memory_allocated(owner.device)
            )
            if cuda
            else 0,
            rollout_update_seconds=elapsed,
            invalid_actuator_states=invalid,
        )
        self.samples.append(allocated)
        if (
            iteration <= 2
            or iteration in self.milestones
            or iteration % self.check_every == 0
        ):
            row["actor_optimizer"] = optimizer_snapshot(owner.actor_optimizer)
            row["critic_optimizer"] = optimizer_snapshot(
                owner.critic_optimizer
            )
        if iteration in self.memory_growth_check_iterations:
            growth = memory_growth(self.samples, self.memory_window)
            previous_samples = self.samples[
                -2 * self.memory_window : -self.memory_window
            ]
            recent_samples = self.samples[-self.memory_window :]
            row["steady_allocated_memory_growth_ratio"] = growth
            row["steady_allocated_memory_previous_mean_bytes"] = (
                sum(previous_samples) / self.memory_window
            )
            row["steady_allocated_memory_recent_mean_bytes"] = (
                sum(recent_samples) / self.memory_window
            )
            row["steady_allocated_memory_growth_ratio_max"] = (
                self.memory_growth_limit
            )
            row["steady_allocated_memory_growth_action"] = (
                self.memory_growth_action
            )
            if growth > self.memory_growth_limit:
                message = (
                    f"Steady allocated memory grew {growth:.2%} at update "
                    f"{iteration}; configured limit is "
                    f"{self.memory_growth_limit:.2%}"
                )
                if self.memory_growth_action == "warn":
                    self._write(f"memory_warning_iteration_{iteration}", row)
                    logger.warning(message)
                else:
                    # Preserve the original strict receipt name for dedicated
                    # stability jobs and downstream consumers.
                    self._write("memory_failure", row)
                    raise RuntimeError(message)
        with (self.root / f"training_health_rank_{self.rank:04d}.jsonl").open(
            "a"
        ) as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.last_row = row
        self.gradient_stats.clear()
        self.sigma_stats.clear()

    def after_iteration(self, iteration):
        if iteration not in self.milestones:
            self.last_time = time.monotonic()
            return
        row = dict(
            self.identity,
            **{k: v for k, v in self.last_row.items() if k != "rank"},
        )
        row["actor_parameters_sha256"] = parameter_digest(self.owner.actor)
        row["critic_parameters_sha256"] = parameter_digest(self.owner.critic)
        # Buffers (e.g. normalizer running statistics) are not assumed
        # identical across ranks. Only trained parameters must match after DDP.
        rows = [None] * self.world_size
        if self.world_size > 1:
            torch.distributed.all_gather_object(rows, row)
        else:
            rows[0] = row
        for key in (
            "actor_parameters_sha256",
            "critic_parameters_sha256",
            "source_tree_sha256",
        ):
            if len({item[key] for item in rows}) != 1:
                raise RuntimeError(
                    f"DDP parameter mismatch at {iteration}: {key}"
                )
        if (
            len(
                {
                    json.dumps(item["training_asset_contract"], sort_keys=True)
                    for item in rows
                }
            )
            != 1
        ):
            raise RuntimeError("Ranks loaded different training assets")
        expected_nodes = self.config.get("expected_nodes")
        if expected_nodes is not None and len(
            {r["hostname"] for r in rows}
        ) != int(expected_nodes):
            raise RuntimeError(
                "Expected worker-host count was not observed; physical "
                "placement must also be checked from platform receipts"
            )
        self._write(
            f"milestone_{iteration}", dict(status="passed", ranks=rows)
        )
        self.last_time = time.monotonic()


def make_training_health(owner):
    config = owner.config.get("runtime_validation")
    if (
        not config
        or not bool(config.get("enabled", False))
        or owner.is_offline_eval
    ):
        return None
    return TrainingHealthMonitor(owner, config)
