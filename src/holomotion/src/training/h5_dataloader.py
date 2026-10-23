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


"""Simplified HDF5 motion cache backed by a PyTorch ``DataLoader``.

This module provides two core utilities:

* ``Hdf5MotionDataset`` – loads contiguous motion windows directly from HDF5
  shards using metadata stored in ``manifest.json``.
* ``MotionClipBatchCache`` – maintains a double-buffered cache of motion clips
  with deterministic swapping semantics suitable for high-throughput
  reinforcement learning.

Compared to the legacy slot-based prefetcher, this implementation keeps the
pipeline intentionally simple:

* A dataset-worker keeps shard handles open locally; no Ray dependency.
* Each cached batch has a fixed shape
  ``[max_num_clips, max_frame_length, feature_dims]``.
* Swapping a batch is handled via an O(1) pointer flip once the next batch is
  staged on the desired device (CPU or GPU).

The cache exposes helper methods that mirror the data access patterns required
by ``RefMotionCommand``:

* ``sample_env_assignments`` for initial clip/frame sampling.
* ``gather_tensor`` to fetch exactly one tensor field for ``1 + n_future``
  frames per environment.

All tensors returned by this module are ``torch.float32`` unless stated
otherwise; tensor shapes are noted explicitly in type annotations.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from holomotion.src.utils import torch_utils
from holomotion.src.training.reference_fk import TrainingReferenceFK

Tensor = torch.Tensor


def _cpu_only_dataloader_worker_init_fn(worker_id: int) -> None:
    """Keep cache workers lightweight without mutating CUDA visibility."""
    del worker_id
    torch.set_num_threads(1)


def _allocate_batch_counts(
    raw_counts: List[float], target_total: int
) -> List[int]:
    """Allocate integer counts that sum exactly to target_total."""
    total = int(max(0, target_total))
    if len(raw_counts) == 0:
        return []
    base_counts = [max(0, int(c)) for c in raw_counts]
    residuals = [float(c) - float(int(c)) for c in raw_counts]
    remaining = total - int(sum(base_counts))
    if remaining > 0:
        order = sorted(
            range(len(residuals)),
            key=lambda i: residuals[i],
            reverse=True,
        )
        idx_pos = 0
        while remaining > 0:
            j = order[idx_pos % len(order)]
            base_counts[j] += 1
            remaining -= 1
            idx_pos += 1
    elif remaining < 0:
        order = sorted(range(len(residuals)), key=lambda i: residuals[i])
        idx_pos = 0
        while remaining < 0:
            j = order[idx_pos % len(order)]
            if base_counts[j] > 0:
                base_counts[j] -= 1
                remaining += 1
            idx_pos += 1
    if sum(base_counts) != total:
        raise RuntimeError(
            "Internal error: integer batch-count allocation did not preserve total."
        )
    return [max(0, int(c)) for c in base_counts]


def _configure_weighted_bins(
    keys: List[str],
    cfg: Mapping[str, Any],
    batch_size_for_log: int,
) -> Tuple[List[List[int]], List[float], List[Dict[str, Any]]]:
    """Common helper to parse config, assign bins, and compute batch fractions."""
    if batch_size_for_log <= 0:
        batch_size_for_log = 1

    cfg_local: Dict[str, Any] = dict(cfg or {})

    dataset_ratios = cfg_local.get("dataset_ratios")
    normalize_dataset_ratios = dataset_ratios is not None
    if dataset_ratios is not None:
        if not isinstance(dataset_ratios, Mapping) or not dataset_ratios:
            raise ValueError(
                "weighted_bin dataset_ratios must be a non-empty mapping"
            )
        patterns_cfg = [
            {
                "name": str(dataset_name),
                "regex": rf"^{re.escape(_normalize_dataset_id(dataset_name))}::",
                "ratio": ratio,
            }
            for dataset_name, ratio in dataset_ratios.items()
        ]
    else:
        patterns_cfg = cfg_local.get("bin_regex_patterns")
        if patterns_cfg is None:
            patterns_cfg = cfg_local.get("bin_regrex_patterns")
    if not patterns_cfg:
        raise ValueError(
            "weighted_bin configuration requires 'bin_regex_patterns' "
            "(list of {regex, ratio}) to be configured"
        )

    compiled_patterns: List[Dict[str, Any]] = []
    ratios: List[float] = []
    for idx, entry in enumerate(patterns_cfg):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns must be a mapping, "
                f"got {type(entry)}"
            )
        regex_str = entry.get("regex", entry.get("regrex", None))
        if not isinstance(regex_str, str) or not regex_str:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns is missing a non-empty "
                f"'regex' field"
            )
        ratio_val = entry.get("ratio", None)
        if ratio_val is None:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns is missing 'ratio'"
            )
        ratio_f = float(ratio_val)
        if ratio_f < 0.0 or ratio_f > 1.0:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns has invalid ratio "
                f"{ratio_f:.6f}; expected in [0.0, 1.0]"
            )
        compiled_patterns.append(
            {
                "name": str(entry.get("name", f"bin_{idx}")),
                "regex": regex_str,
                "compiled": re.compile(regex_str),
            }
        )
        ratios.append(ratio_f)

    sum_explicit = float(sum(ratios))
    if normalize_dataset_ratios:
        if sum_explicit <= 0.0:
            raise ValueError(
                "weighted_bin root ratios must contain at least one positive value"
            )
        ratios = [ratio / sum_explicit for ratio in ratios]
        sum_explicit = 1.0
    elif sum_explicit > 1.0 + 1.0e-6:
        raise ValueError(
            f"Sum of weighted-bin ratios is {sum_explicit:.6f} (> 1.0). "
            "Please reduce the ratios so that their sum is <= 1.0."
        )
    if sum_explicit > 1.0:
        sum_explicit = 1.0
    others_ratio = max(0.0, 1.0 - sum_explicit)

    if len(keys) == 0:
        raise ValueError(
            "weighted_bin configuration received an empty key set"
        )

    num_items_total = float(len(keys))
    num_explicit = len(compiled_patterns)
    bin_count = num_explicit if normalize_dataset_ratios else num_explicit + 1
    bin_indices: List[List[int]] = [[] for _ in range(bin_count)]

    for idx, motion_key in enumerate(keys):
        assigned = False
        for b_idx, pat in enumerate(compiled_patterns):
            if pat["compiled"].search(motion_key):
                bin_indices[b_idx].append(idx)
                assigned = True
                break
        if not assigned and not normalize_dataset_ratios:
            bin_indices[-1].append(idx)

    # Combine explicit ratios with implicit "others" ratio
    all_ratios: List[float] = list(ratios)
    if not normalize_dataset_ratios:
        all_ratios.append(others_ratio)

    # If all motion keys are covered by explicit regex bins, but the specified
    # ratios sum to less than 1.0, linearly reweight explicit ratios so that
    # they sum to 1.0 and disable the implicit "others" bin.
    others_count = 0 if normalize_dataset_ratios else len(bin_indices[-1])
    if (
        not normalize_dataset_ratios
        and others_count == 0
        and others_ratio > 0.0
        and sum_explicit > 0.0
    ):
        scale = 1.0 / sum_explicit
        ratios = [r * scale for r in ratios]
        others_ratio = 0.0
        all_ratios = list(ratios)
        all_ratios.append(others_ratio)
        logger.info(
            "Weighted-bin: all regex bins cover the dataset; "
            "linearly reweighted explicit ratios to sum to 1.0 and disabled "
            "the implicit 'others' bin."
        )

    # Validate non-empty bins for any positive ratio (including others)
    for b_idx, r in enumerate(all_ratios):
        if r > 0.0 and len(bin_indices[b_idx]) == 0:
            if b_idx < num_explicit:
                name = compiled_patterns[b_idx]["name"]
                regex_s = compiled_patterns[b_idx]["regex"]
                raise ValueError(
                    f"Weighted-bin '{name}' (regex='{regex_s}') has ratio "
                    f"{r:.6f} but matched no motion keys"
                )
            raise ValueError(
                f"Weighted-bin 'others' has ratio {r:.6f} but matched no motion keys"
            )

    # Prepare logging summary using the configured cache batch size
    raw_counts_log = [ratio * batch_size_for_log for ratio in all_ratios]
    base_counts_log = _allocate_batch_counts(
        raw_counts=raw_counts_log,
        target_total=batch_size_for_log,
    )
    batch_fractions_log = [
        float(c) / float(batch_size_for_log) for c in base_counts_log
    ]

    # Build specs using the final, actually used batch fractions
    specs: List[Dict[str, Any]] = []
    total_items = float(max(1, num_items_total))
    for b_idx in range(num_explicit):
        name = compiled_patterns[b_idx]["name"]
        regex_s = compiled_patterns[b_idx]["regex"]
        n = len(bin_indices[b_idx])
        ds_frac = float(n) / total_items
        bf = batch_fractions_log[b_idx]
        specs.append(
            {
                "name": name,
                "regex": regex_s,
                "ratio": bf,
                "count": n,
                "dataset_fraction": ds_frac,
                "batch_fraction": bf,
            }
        )
    if not normalize_dataset_ratios:
        n_o = len(bin_indices[-1])
        ds_frac_o = float(n_o) / total_items
        bf_o = batch_fractions_log[-1]
        specs.append(
            {
                "name": "others",
                "regex": "<unmatched>",
                "ratio": bf_o,
                "count": n_o,
                "dataset_fraction": ds_frac_o,
                "batch_fraction": bf_o,
            }
        )

    return bin_indices, all_ratios, specs


def _normalize_dataset_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _manifest_dataset_id(
    manifest: Mapping[str, Any], manifest_path: str
) -> str:
    explicit = manifest.get("dataset_id")
    if explicit:
        return _normalize_dataset_id(explicit)

    path_candidates = [
        manifest.get("source_holosmpl_h5_root"),
        os.path.dirname(manifest_path),
    ]
    for path_value in path_candidates:
        if not path_value:
            continue
        for part in reversed(Path(str(path_value)).parts):
            lowered = part.lower()
            if lowered.startswith(("train_", "eval_")):
                return _normalize_dataset_id(part)

    for field in ("dataset_name", "dataset", "source_name"):
        value = manifest.get(field)
        if value:
            return _normalize_dataset_id(value)
    return "unknown"


def _manifest_dataset_id_from_path(manifest_path: str) -> str:
    """Read only the manifest header needed to identify a dataset.

    Formal HoloSMPL roots are commonly named ``formal_h5``.  Deriving the
    weighted-bin id from that directory therefore yields ``unknown`` or the
    same id for every dataset.  The canonical id lives in the manifest's
    top-level metadata and is deliberately placed before the potentially
    large clip table.
    """

    if not os.path.isfile(manifest_path):
        return _manifest_dataset_id({}, manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        header = handle.read(128 * 1024)
    for field in ("dataset_id", "dataset_name", "dataset", "source_name"):
        match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]+)"', header)
        if match:
            return _normalize_dataset_id(match.group(1))
    return _manifest_dataset_id({}, manifest_path)


def _weighted_bin_motion_key(
    motion_key: str, metadata: Mapping[str, Any] | None = None
) -> str:
    dataset_id = None if metadata is None else metadata.get("_dataset_id")
    if not dataset_id:
        return str(motion_key)
    return f"{dataset_id}::{motion_key}"


def _collect_manifest_keys(
    manifest_path: str | Sequence[str],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    if isinstance(manifest_path, (str, os.PathLike)):
        manifest_paths: List[str] = [str(manifest_path)]
    else:
        manifest_paths = [str(p) for p in manifest_path]
    if len(manifest_paths) == 0:
        raise ValueError("Expected at least one manifest path")

    key_source: Dict[str, str] = {}
    for mp in manifest_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"HDF5 manifest not found at {mp}. "
                "Please set robot.motion.hdf5_root/train_hdf5_roots "
                "to the correct path."
            )
        with open(mp, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        clips = manifest.get("clips", {})
        if not clips:
            raise ValueError(
                f"Manifest at {mp} contains no clips; cannot preview sampling."
            )
        dataset_id = _manifest_dataset_id(manifest, mp)
        for key in clips.keys():
            weighted_key = _weighted_bin_motion_key(
                key, {"_dataset_id": dataset_id}
            )
            if weighted_key in key_source:
                raise ValueError(
                    f"Duplicate weighted-bin key '{weighted_key}' found in multiple "
                    "manifests; clip keys must be globally unique."
                )
            key_source[weighted_key] = mp

    return list(key_source.keys()), key_source, manifest_paths


def preview_weighted_bin_from_manifest(
    manifest_path: str | Sequence[str],
    batch_size: int,
    cfg: Mapping[str, Any],
) -> None:
    """Lightweight preview of weighted-bin sampling using manifest.json only.

    This helper is intended to be called at configuration time before any
    MotionClipBatchCache/DataLoader is constructed, so that invalid regex or
    ratio settings can fail fast without incurring the cost of cache setup.
    """
    if batch_size <= 0:
        batch_size = 1

    keys, _, _ = _collect_manifest_keys(manifest_path=manifest_path)
    _, _, specs = _configure_weighted_bins(
        keys=keys,
        cfg=cfg,
        batch_size_for_log=batch_size,
    )

    table_rows = []
    for item in specs:
        table_rows.append(
            [
                item["name"],
                item["regex"],
                f"{item['ratio']:.4f}",
                int(item["count"]),
                f"{item['dataset_fraction']:.4f}",
                f"{item['batch_fraction']:.4f}",
            ]
        )
    headers = [
        "bin",
        "regex",
        "final_ratio",
        "num_clips",
        "clip_fraction",
        "batch_fraction",
    ]
    logger.info(
        "Weighted-bin config preview (manifest-level):\n"
        + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
    )


def preview_uniform_from_manifest(
    manifest_path: str | Sequence[str],
    batch_size: int,
    *,
    max_frame_length: int,
    min_window_length: int,
    handpicked_motion_names: Optional[Sequence[str]] = None,
    excluded_motion_names: Optional[Sequence[str]] = None,
) -> None:
    """Manifest-level preview table for uniform/curriculum sampling."""
    if batch_size <= 0:
        batch_size = 1
    if max_frame_length <= 0:
        raise ValueError("max_frame_length must be positive")
    if min_window_length <= 0:
        raise ValueError("min_window_length must be positive")

    _, _, manifest_paths = _collect_manifest_keys(manifest_path=manifest_path)
    handpicked_set = (
        set(handpicked_motion_names)
        if handpicked_motion_names is not None
        else None
    )
    excluded_set = (
        set(excluded_motion_names)
        if excluded_motion_names is not None
        else None
    )

    def _normalize_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        key = value if isinstance(value, str) else str(value)
        if not key:
            return None
        return key

    def _build_aliases(motion_key: str, meta: Mapping[str, Any]) -> List[str]:
        aliases: List[str] = []

        def _add(value: Any) -> None:
            key = _normalize_key(value)
            if key is None or key in aliases:
                return
            aliases.append(key)

        _add(motion_key)
        if isinstance(meta, Mapping):
            _add(meta.get("motion_key"))
            metadata = meta.get("metadata")
            if isinstance(metadata, Mapping):
                _add(metadata.get("motion_key"))
                _add(metadata.get("raw_motion_key"))
        return aliases

    def _count_windows(clip_length: int) -> Tuple[int, int]:
        remaining = clip_length
        offset = 0
        num_windows = 0
        num_frames = 0
        while remaining > 0:
            window_length = min(max_frame_length, remaining)
            if window_length >= min_window_length:
                num_windows += 1
                num_frames += int(window_length)
            offset += int(window_length)
            remaining = max(0, clip_length - offset)
        return num_windows, num_frames

    stats_by_manifest: Dict[str, Dict[str, float]] = {}
    for mp in manifest_paths:
        with open(mp, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        clips = manifest.get("clips", {})
        if not clips:
            raise ValueError(
                f"Manifest at {mp} contains no clips; cannot preview sampling."
            )
        num_windows = 0
        num_frames = 0
        duration_s = 0.0
        for key, meta in clips.items():
            if isinstance(meta, Mapping):
                aliases = _build_aliases(key, meta)
            else:
                aliases = [key]
            if handpicked_set is not None and not any(
                alias in handpicked_set for alias in aliases
            ):
                continue
            if excluded_set is not None and any(
                alias in excluded_set for alias in aliases
            ):
                continue
            length = (
                int(meta.get("length", 0)) if isinstance(meta, Mapping) else 0
            )
            if length <= 0:
                continue
            metadata = (
                meta.get("metadata") if isinstance(meta, Mapping) else None
            )
            motion_fps_val = None
            if isinstance(metadata, Mapping):
                motion_fps_val = metadata.get("motion_fps")
            if motion_fps_val is None and isinstance(meta, Mapping):
                motion_fps_val = meta.get("motion_fps")
            if motion_fps_val is None:
                raise ValueError(
                    f"motion_fps missing for clip {key} in manifest {mp}"
                )
            motion_fps = float(motion_fps_val)
            if motion_fps <= 0.0:
                raise ValueError(
                    f"Invalid motion_fps {motion_fps} for clip {key} in {mp}"
                )
            clip_windows, clip_frames = _count_windows(length)
            num_windows += int(clip_windows)
            num_frames += int(clip_frames)
            duration_s += float(clip_frames) / float(motion_fps)
        stats_by_manifest[mp] = {
            "num_windows": float(num_windows),
            "num_frames": float(num_frames),
            "duration_s": float(duration_s),
        }

    total_windows = int(
        sum(stats["num_windows"] for stats in stats_by_manifest.values())
    )
    if total_windows == 0:
        raise ValueError(
            "No motion windows satisfy the requested frame length constraints"
        )

    table_rows = []
    denom = float(max(1, total_windows))
    for mp in manifest_paths:
        stats = stats_by_manifest.get(mp, {})
        count = int(stats.get("num_windows", 0))
        frames = int(stats.get("num_frames", 0))
        duration_h = float(stats.get("duration_s", 0.0)) / 3600.0
        frac = float(count) / denom
        table_rows.append(
            [
                os.path.dirname(mp),
                count,
                f"{frac:.4f}",
                frames,
                f"{duration_h:.2f}",
                f"{frac:.4f}",
            ]
        )
    headers = [
        "dataset_root",
        "num_windows",
        "window_fraction",
        "num_frames",
        "duration_h",
        "batch_fraction",
    ]
    logger.info(
        "Uniform sampling preview (manifest-level):\n"
        + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
    )


def preview_sampling_from_cfg(motion_cfg: Mapping[str, Any]) -> None:
    """Preview manifest-level sampling table for uniform/weighted-bin."""
    sampling_strategy_cfg = motion_cfg.get("sampling_strategy", None)
    if sampling_strategy_cfg is None:
        sampling_strategy = "uniform"
    else:
        sampling_strategy = str(sampling_strategy_cfg).lower()
    if sampling_strategy not in ("uniform", "weighted_bin", "curriculum"):
        return

    backend = str(motion_cfg.get("backend", "hdf5")).lower()
    if backend not in ("hdf5", "hdf5_simple", "hdf5_v2"):
        return

    train_roots = _normalize_root_list(
        motion_cfg.get("train_hdf5_roots", None)
    )
    if len(train_roots) == 0:
        hdf5_root = motion_cfg.get("hdf5_root", None)
        if not hdf5_root:
            return
        train_roots = [str(hdf5_root)]
    manifest_paths = [
        os.path.join(str(root), "manifest.json") for root in train_roots
    ]
    cache_cfg = motion_cfg.get("cache", {})
    batch_size = int(cache_cfg.get("max_num_clips", 1))

    if sampling_strategy == "weighted_bin":
        weighted_bin_cfg = weighted_bin_cfg_from_motion_cfg(motion_cfg)
        preview_weighted_bin_from_manifest(
            manifest_path=manifest_paths
            if len(manifest_paths) > 1
            else manifest_paths[0],
            batch_size=batch_size,
            cfg=weighted_bin_cfg,
        )
        return

    max_frame_length = int(motion_cfg.get("max_frame_length", 1))
    min_window_length = int(motion_cfg.get("min_frame_length", 1))
    handpicked_motion_names = motion_cfg.get("handpicked_motion_names", None)
    excluded_motion_names = motion_cfg.get("excluded_motion_names", None)
    preview_uniform_from_manifest(
        manifest_path=manifest_paths
        if len(manifest_paths) > 1
        else manifest_paths[0],
        batch_size=batch_size,
        max_frame_length=max_frame_length,
        min_window_length=min_window_length,
        handpicked_motion_names=handpicked_motion_names,
        excluded_motion_names=excluded_motion_names,
    )


MANDATORY_DATASETS = {
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vel",
    "rg_pos": "global_translation",
    "rb_rot": "global_rotation_quat",
    "body_vel": "global_velocity",
    "body_ang_vel": "global_angular_velocity",
}


class _WorldFrameNormalizeTransform:
    """Normalize motion tensors into a canonical z-up world frame in-place."""

    @staticmethod
    def _apply_prefix(
        arrays: Dict[str, Tensor],
        prefix: str,
        *,
        offset_xy: Tensor,
        q_flat_wxyz: Tensor,
        ref_rg_pos_shape: torch.Size,
        ref_rb_rot_shape: torch.Size,
    ) -> None:
        pos_key = f"{prefix}rg_pos"
        rot_key = f"{prefix}rb_rot"
        vel_key = f"{prefix}body_vel"
        ang_key = f"{prefix}body_ang_vel"
        if (
            pos_key not in arrays
            or rot_key not in arrays
            or vel_key not in arrays
            or ang_key not in arrays
        ):
            return

        pos = arrays[pos_key]
        rot = arrays[rot_key]
        vel = arrays[vel_key]
        ang = arrays[ang_key]
        if pos.shape != ref_rg_pos_shape or rot.shape != ref_rb_rot_shape:
            return

        # Center XY using canonical offset.
        pos[..., 0] -= offset_xy[0]
        pos[..., 1] -= offset_xy[1]

        # Rotate vectors using shared quaternion utilities (WXYZ convention).
        pos_flat = pos.reshape(-1, 3)
        vel_flat = vel.reshape(-1, 3)
        ang_flat = ang.reshape(-1, 3)
        pos[:] = torch_utils.quat_apply(q_flat_wxyz, pos_flat).reshape_as(pos)
        vel[:] = torch_utils.quat_apply(q_flat_wxyz, vel_flat).reshape_as(vel)
        ang[:] = torch_utils.quat_apply(q_flat_wxyz, ang_flat).reshape_as(ang)

        # Rotate orientations: q' = q_heading_inv * q.
        rot_flat_xyzw = rot.reshape(-1, 4)
        rot_flat_wxyz = torch_utils.xyzw_to_wxyz(rot_flat_xyzw)
        rot_out_wxyz = torch_utils.quat_mul(q_flat_wxyz, rot_flat_wxyz)
        rot[:] = torch_utils.wxyz_to_xyzw(rot_out_wxyz).reshape_as(rot)

    def __call__(
        self, arrays: Dict[str, Tensor], *, anchor_index: int = 0
    ) -> None:
        if "ref_rg_pos" not in arrays or "ref_rb_rot" not in arrays:
            raise ValueError("ref_rg_pos and ref_rb_rot are required")
        if "ref_body_vel" not in arrays or "ref_body_ang_vel" not in arrays:
            raise ValueError("ref_body_vel and ref_body_ang_vel are required")

        rg_pos = arrays["ref_rg_pos"]
        rb_rot = arrays["ref_rb_rot"]

        anchor = int(anchor_index)
        if anchor < 0 or anchor >= rg_pos.shape[0]:
            raise ValueError(
                f"anchor_index {anchor} outside [0,{rg_pos.shape[0]})"
            )

        # Root pose at the requested output anchor, body 0 (XYZW, z-up).
        p_root0 = rg_pos[anchor, 0]  # [3]
        q_root0 = rb_rot[anchor, 0]  # [4]

        # Compute XY offset from root at frame 0 (will be applied in _apply_to_set).
        offset_xy = p_root0.clone()
        offset_xy[2] = 0.0

        # Extract yaw from q_root0 (XYZW) using z-up convention.
        x = q_root0[0]
        y = q_root0[1]
        z = q_root0[2]
        w = q_root0[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = w * w + x * x - y * y - z * z
        yaw0 = torch.atan2(siny_cosp, cosy_cosp)

        # Quaternion for rotation around +Z by -yaw0 (remove initial heading).
        half = -0.5 * yaw0
        sin_half = torch.sin(half)
        cos_half = torch.cos(half)
        q_heading_inv = torch.stack(
            [
                torch.zeros_like(sin_half),
                torch.zeros_like(sin_half),
                sin_half,
                cos_half,
            ],
            dim=-1,
        )  # [4], XYZW

        t, b, _ = rg_pos.shape
        q_flat = q_heading_inv.view(1, 1, 4).expand(t, b, 4).reshape(-1, 4)
        q_flat_wxyz = torch_utils.xyzw_to_wxyz(q_flat)

        self._apply_prefix(
            arrays,
            "ref_",
            offset_xy=offset_xy,
            q_flat_wxyz=q_flat_wxyz,
            ref_rg_pos_shape=rg_pos.shape,
            ref_rb_rot_shape=rb_rot.shape,
        )


class _CpuFKTransform:
    """Compute FK on CPU and write ref_* tensors in-place."""

    def __init__(self, robot_file_path: str) -> None:
        self._fk = TrainingReferenceFK(
            robot_file_path=str(robot_file_path), device=torch.device("cpu")
        )
        self._fk = self._fk.to(torch.device("cpu"))

    def __call__(
        self,
        arrays: Dict[str, Tensor],
        fps: float,
        prefix: str = "ref_",
    ) -> None:
        root_pos_key = f"{prefix}root_pos"
        root_rot_key = f"{prefix}root_rot"
        dof_pos_key = f"{prefix}dof_pos"
        if (
            root_pos_key not in arrays
            or root_rot_key not in arrays
            or dof_pos_key not in arrays
        ):
            raise KeyError(f"Missing {prefix}root_* or {prefix}dof_pos for FK")
        with torch.no_grad():
            fk_out = self._fk(
                root_pos=arrays[root_pos_key][None, ...],
                root_quat=arrays[root_rot_key][None, ...],
                dof_pos=arrays[dof_pos_key][None, ...],
                fps=float(fps),
                quat_format="xyzw",
            )
        arrays[f"{prefix}rg_pos"] = fk_out["global_translation"][0]
        arrays[f"{prefix}rb_rot"] = fk_out["global_rotation_quat"][0]
        arrays[f"{prefix}body_vel"] = fk_out["global_velocity"][0]
        arrays[f"{prefix}body_ang_vel"] = fk_out["global_angular_velocity"][0]
        arrays[f"{prefix}dof_vel"] = fk_out["dof_vel"][0]


@dataclass
class MotionWindow:
    """Metadata describing a contiguous motion window within an HDF5 shard."""

    motion_key: str  # unique per window
    shard_index: int
    start: int
    length: int
    raw_motion_key: str  # original clip key
    window_index: int


@dataclass
class MotionClipSample:
    """In-memory representation of a motion window.

    Attributes:
        motion_key: Unique window identifier (includes slice info).
        raw_motion_key: Original clip identifier from manifest.
        tensors: Mapping from tensor name to data tensor of shape
            ``[window_length, ...]`` (float32 unless specified otherwise).
        length: Number of valid frames contained in the sample (``<=``
            ``max_frame_length``).
    """

    motion_key: str
    raw_motion_key: str
    window_index: int
    tensors: Dict[str, Tensor]
    length: int


@dataclass
class ClipBatch:
    """Batch of motion clips ready for consumption by the environment.

    Attributes:
        tensors: Mapping from tensor name to tensor with shape
            ``[batch_size, max_frame_length, ...]`` placed on the staging
            device.
        lengths: Valid frame counts per clip ``[batch_size]``.
        motion_keys: List of motion keys corresponding to each clip.
        max_frame_length: Fixed length configured for the cache.
        real_sample_count: Number of non-padding samples in the batch.  This
            differs from the tensor batch dimension only for the final
            validation batch, which is padded to keep a fixed number of
            vectorized environments.
    """

    tensors: Dict[str, Tensor]
    lengths: Tensor
    motion_keys: List[str]
    raw_motion_keys: List[str]
    window_indices: Tensor
    max_frame_length: int
    real_sample_count: Optional[int] = None

    @staticmethod
    def collate_fn(samples: List[MotionClipSample]) -> "ClipBatch":
        if len(samples) == 0:
            raise ValueError(
                "ClipBatch collate_fn received an empty sample list"
            )

        # A pure-SMPL batch intentionally has no ``ref_dof_pos`` tensor.
        # Every tensor in a sample shares the same leading time dimension, so
        # derive the padding length from the declared sample length instead of
        # coupling the generic cache to a robot-reference field.
        max_frame_length = max(int(sample.length) for sample in samples)
        max_frame_length = int(max_frame_length)

        batched_tensors: Dict[str, Tensor] = {}
        lengths = torch.zeros(len(samples), dtype=torch.long)
        motion_keys = []
        raw_motion_keys = []
        window_indices = torch.zeros(len(samples), dtype=torch.long)

        for batch_idx, sample in enumerate(samples):
            lengths[batch_idx] = sample.length
            motion_keys.append(sample.motion_key)
            raw_motion_keys.append(sample.raw_motion_key)
            window_indices[batch_idx] = int(sample.window_index)

            for name, tensor in sample.tensors.items():
                if name not in batched_tensors:
                    pad_shape = (
                        len(samples),
                        max_frame_length,
                    ) + tensor.shape[1:]
                    batched_tensors[name] = torch.zeros(
                        pad_shape,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                target = batched_tensors[name]
                valid_frames = sample.length
                target[batch_idx, :valid_frames] = tensor

                if valid_frames < max_frame_length and valid_frames > 0:
                    target[batch_idx, valid_frames:] = tensor[valid_frames - 1]

        return ClipBatch(
            tensors=batched_tensors,
            lengths=lengths,
            motion_keys=motion_keys,
            raw_motion_keys=raw_motion_keys,
            window_indices=window_indices,
            max_frame_length=max_frame_length,
            real_sample_count=len(samples),
        )


class Hdf5RootDofDataset(Dataset[MotionClipSample]):
    """HDF5 dataset reading ref_root_* + ref_dof_pos only."""

    def __init__(
        self,
        manifest_path: str | Sequence[str],
        max_frame_length: int,
        min_window_length: int = 1,
        handpicked_motion_names: Optional[List[str]] = None,
        excluded_motion_names: Optional[List[str]] = None,
        fk_robot_file_path: Optional[str] = None,
        allowed_prefixes: Optional[Sequence[str]] = None,
        task_target_context: bool = False,
    ) -> None:
        super().__init__()
        if max_frame_length <= 0:
            raise ValueError("max_frame_length must be positive")
        if min_window_length <= 0:
            raise ValueError("min_window_length must be positive")

        self.max_frame_length = int(max_frame_length)
        self.min_window_length = int(min_window_length)
        self.handpicked_motion_names = (
            set(handpicked_motion_names)
            if handpicked_motion_names is not None
            else None
        )
        self.excluded_motion_names = (
            set(excluded_motion_names)
            if excluded_motion_names is not None
            else None
        )
        self._fk_robot_file_path = (
            str(fk_robot_file_path) if fk_robot_file_path is not None else ""
        )
        if not self._fk_robot_file_path:
            raise ValueError("fk_robot_file_path is required for hdf5_v2 FK")
        self._fk_transform = _CpuFKTransform(self._fk_robot_file_path)
        self._world_frame_transform = _WorldFrameNormalizeTransform()
        self._task_target_context = bool(task_target_context)
        if allowed_prefixes is None:
            self._allowed_prefixes = ("ref_",)
        else:
            self._allowed_prefixes = tuple(str(v) for v in allowed_prefixes)
        if "ref_" not in self._allowed_prefixes:
            raise ValueError(
                "Hdf5RootDofDataset requires 'ref_' in allowed_prefixes"
            )
        self._progress_counter: Optional[mp.Value] = None

        if isinstance(manifest_path, (str, os.PathLike)):
            manifest_paths: List[str] = [str(manifest_path)]
        else:
            manifest_paths = [str(p) for p in manifest_path]
        if len(manifest_paths) == 0:
            raise ValueError("At least one manifest_path must be provided")

        self.hdf5_root = os.path.dirname(manifest_paths[0])
        self._manifest_paths: List[str] = manifest_paths
        self._shard_paths: List[str] = []
        self.shards: List[Dict[str, Any]] = []
        self.clips: Dict[str, Dict[str, Any]] = {}

        for mp in manifest_paths:
            if not os.path.exists(mp):
                raise FileNotFoundError(
                    f"HDF5 manifest not found at {mp}. "
                    "Please set robot.motion.hdf5_root/train_hdf5_roots "
                    "to the correct path."
                )
            with open(mp, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            root = os.path.dirname(mp)
            source_holosmpl_h5_root = manifest.get(
                "source_holosmpl_h5_root", None
            )
            shards_local = list(manifest.get("hdf5_shards", []))
            clips_local = manifest.get("clips", {})
            dataset_id = _manifest_dataset_id(manifest, mp)

            shard_offset = len(self.shards)
            for shard_meta in shards_local:
                self.shards.append(shard_meta)
                rel = shard_meta.get("file", None)
                if not isinstance(rel, str) or not rel:
                    raise ValueError(
                        f"Shard entry in manifest {mp} is missing a valid 'file' field"
                    )
                self._shard_paths.append(os.path.join(root, rel))

            for key, meta in clips_local.items():
                if key in self.clips:
                    raise ValueError(
                        f"Duplicate motion clip key '{key}' found in multiple "
                        "manifests; clip keys must be globally unique."
                    )
                meta_global = dict(meta)
                meta_global["_dataset_id"] = dataset_id
                if source_holosmpl_h5_root:
                    meta_global["_source_holosmpl_h5_root"] = str(
                        source_holosmpl_h5_root
                    )
                meta_global["shard"] = (
                    int(meta_global.get("shard", 0)) + shard_offset
                )
                self.clips[key] = meta_global

        if len(self.shards) == 0:
            raise ValueError(
                f"No HDF5 shards listed in manifests: {', '.join(manifest_paths)}"
            )

        self.windows: List[MotionWindow] = self._enumerate_windows()
        if len(self.windows) == 0:
            raise ValueError(
                "No motion windows satisfy the requested frame length constraints"
            )

        # Setting up hdf5 file handles management for bounded host-memory usage
        self._file_handles: "OrderedDict[int, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        if max_open_env is None:
            self._h5_max_open_files = 16
        else:
            self._h5_max_open_files = max(1, int(max_open_env))
        self._h5_access_counter = 0
        self._h5_cleanup_interval = int(
            1.0e6
        )  # clean h5 handles every 1 million samples

    def set_progress_counter(self, counter: Optional[mp.Value]) -> None:
        self._progress_counter = counter

    @staticmethod
    def _normalize_motion_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            key = value
        else:
            key = str(value)
        if not key:
            return None
        return key

    def _build_motion_key_aliases(
        self, motion_key: str, meta: Mapping[str, Any]
    ) -> Tuple[str, ...]:
        aliases: List[str] = []

        def _add(value: Any) -> None:
            key = self._normalize_motion_key(value)
            if key is None:
                return
            if key in aliases:
                return
            aliases.append(key)

        _add(motion_key)
        if isinstance(meta, Mapping):
            _add(meta.get("motion_key"))
            metadata = meta.get("metadata")
            if isinstance(metadata, Mapping):
                _add(metadata.get("motion_key"))
                _add(metadata.get("raw_motion_key"))
        return tuple(aliases)

    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        for motion_key, meta in self.clips.items():
            aliases = self._build_motion_key_aliases(motion_key, meta)
            if self.handpicked_motion_names is not None and not any(
                alias in self.handpicked_motion_names for alias in aliases
            ):
                continue
            if self.excluded_motion_names is not None and any(
                alias in self.excluded_motion_names for alias in aliases
            ):
                continue

            shard_index = int(meta.get("shard", 0))
            start = int(meta.get("start", 0))
            length = int(meta.get("length", 0))

            if length <= 0:
                continue

            remaining = length
            offset = 0
            window_index = 0
            while remaining > 0:
                window_length = min(self.max_frame_length, remaining)
                if window_length >= self.min_window_length:
                    win_start = start + offset
                    unique_key = (
                        f"{motion_key}__start_{win_start}_len_{window_length}"
                    )
                    windows.append(
                        MotionWindow(
                            motion_key=unique_key,
                            shard_index=shard_index,
                            start=win_start,
                            length=window_length,
                            raw_motion_key=motion_key,
                            window_index=window_index,
                        )
                    )
                    window_index += 1
                offset += window_length
                remaining = max(0, length - offset)
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _cast_motion_np(np_array: np.ndarray, name: str) -> Tensor:
        if np_array.dtype == np.float32:
            pass
        elif np_array.dtype.kind == "O":
            raise ValueError(f"{name} has object dtype")
        elif np.issubdtype(np_array.dtype, np.integer):
            logger.warning(
                "Casting {} from {} to float32.", name, np_array.dtype
            )
            np_array = np_array.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"{name} has dtype {np_array.dtype}, expected float32 or integer."
            )
        return torch.from_numpy(np_array).to(torch.float32)

    @staticmethod
    def _make_scalar_metadata_tensor(value: float, length: int) -> Tensor:
        return torch.full((int(length), 1), float(value), dtype=torch.float32)

    @staticmethod
    def _derive_root_state_tensors(
        arrays: Dict[str, Tensor],
        prefix: str = "ref_",
    ) -> None:
        rg_pos_key = f"{prefix}rg_pos"
        rb_rot_key = f"{prefix}rb_rot"
        body_vel_key = f"{prefix}body_vel"
        body_ang_vel_key = f"{prefix}body_ang_vel"
        if (
            rg_pos_key not in arrays
            or rb_rot_key not in arrays
            or body_vel_key not in arrays
            or body_ang_vel_key not in arrays
        ):
            return
        # Keep root-level tensors consistent with the FK-derived body tensors.
        arrays[f"{prefix}root_pos"] = arrays[rg_pos_key][:, 0, :]
        arrays[f"{prefix}root_rot"] = arrays[rb_rot_key][:, 0, :]
        arrays[f"{prefix}root_vel"] = arrays[body_vel_key][:, 0, :]
        arrays[f"{prefix}root_ang_vel"] = arrays[body_ang_vel_key][:, 0, :]

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        shard_handle = self._get_shard_handle(window.shard_index)
        start, end = window.start, window.start + window.length
        clip_meta = self.clips.get(window.raw_motion_key, {})
        clip_start = int(clip_meta.get("start", start))
        clip_length = int(clip_meta.get("length", window.length))
        clip_end = clip_start + clip_length
        task_context_enabled = bool(
            getattr(self, "_task_target_context", False)
        )
        context_before = int(task_context_enabled and start > clip_start)
        context_after = int(task_context_enabled and end < clip_end)
        read_start = start - context_before
        read_end = end + context_after
        arrays: Dict[str, Tensor] = {}

        for dataset_name in ("ref_root_pos", "ref_root_rot", "ref_dof_pos"):
            if dataset_name not in shard_handle:
                raise KeyError(
                    f"Missing mandatory dataset '{dataset_name}' in shard index "
                    f"{window.shard_index}"
                )
            np_array = np.asarray(shard_handle[dataset_name][start:end, ...])
            arrays[dataset_name] = self._cast_motion_np(np_array, dataset_name)

        if "frame_flag" in shard_handle:
            frame_flag_np = shard_handle["frame_flag"][start:end]
            if frame_flag_np.dtype.kind == "O":
                raise ValueError("frame_flag has object dtype")
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 1:
                frame_flag[0] = 0
                frame_flag[-1] = 2
            elif window.length == 1:
                frame_flag[0] = 2
        arrays["frame_flag"] = frame_flag

        metadata = clip_meta.get("metadata", {})
        motion_fps_val = metadata.get(
            "motion_fps", clip_meta.get("motion_fps")
        )
        if motion_fps_val is None:
            raise ValueError(
                f"motion_fps missing for clip {window.raw_motion_key}"
            )
        motion_fps = float(motion_fps_val)
        if motion_fps <= 0.0:
            raise ValueError(
                f"Invalid motion_fps {motion_fps} for clip {window.raw_motion_key}"
            )
        arrays["motion_fps"] = self._make_scalar_metadata_tensor(
            motion_fps, window.length
        )
        self._fk_transform(arrays, motion_fps)
        self._world_frame_transform(arrays)

        task_target = None
        if task_context_enabled:
            from holomotion.src.motion_tracking.smpl_g1_task_target import (
                build_robot_reference_task_target,
            )

            task_arrays: Dict[str, Tensor] = {}
            for dataset_name in (
                "ref_root_pos",
                "ref_root_rot",
                "ref_dof_pos",
            ):
                context_np = np.asarray(
                    shard_handle[dataset_name][read_start:read_end, ...]
                )
                task_arrays[dataset_name] = self._cast_motion_np(
                    context_np, dataset_name
                )
            task_arrays["motion_fps"] = self._make_scalar_metadata_tensor(
                motion_fps, read_end - read_start
            )
            self._fk_transform(task_arrays, motion_fps)
            self._world_frame_transform(
                task_arrays, anchor_index=context_before
            )
            task_target = build_robot_reference_task_target(
                task_arrays["ref_rg_pos"],
                task_arrays["ref_rb_rot"],
                body_names=self._fk_transform._fk.body_names,
                fps=motion_fps,
                output_start=context_before,
                output_length=window.length,
            )

        arrays["frame_flag"] = frame_flag
        if task_target is not None:
            arrays["ref_task_position"] = task_target.position_h
            arrays["ref_task_rotation_xyzw"] = task_target.rotation_h_xyzw
            arrays["ref_task_linear_velocity"] = task_target.linear_velocity_h
            arrays["ref_task_angular_velocity"] = (
                task_target.angular_velocity_h
            )

        self._derive_root_state_tensors(arrays, prefix="ref_")

        if self._progress_counter is not None:
            with self._progress_counter.get_lock():
                self._progress_counter.value += 1

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            window_index=int(index),
            tensors=arrays,
            length=window.length,
        )

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        # periodically clean up the file handles
        self._h5_access_counter += 1
        if self._h5_access_counter >= self._h5_cleanup_interval:
            self.close()
            self._h5_access_counter = 0

        if shard_index in self._file_handles:
            handle = self._file_handles.pop(shard_index)
            if handle.id:
                self._file_handles[shard_index] = handle
                return handle

        if shard_index < 0 or shard_index >= len(self._shard_paths):
            raise IndexError(
                f"Shard index {shard_index} out of range for "
                f"{len(self._shard_paths)} available shards"
            )
        shard_path = self._shard_paths[shard_index]
        rdcc_nbytes_env = os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES")
        if rdcc_nbytes_env is None:
            rdcc_nbytes = 4 * 1024 * 1024
        else:
            rdcc_nbytes = int(rdcc_nbytes_env)
        handle = h5py.File(
            shard_path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=rdcc_nbytes,
            rdcc_w0=0.75,
        )
        if (
            self._h5_max_open_files is not None
            and len(self._file_handles) >= self._h5_max_open_files
        ):
            old_index, old_handle = self._file_handles.popitem(last=False)
            old_handle.close()
        self._file_handles[shard_index] = handle
        return handle

    def close(self) -> None:
        logger.info("Clearing HDF5 file handles ...")
        for handle in self._file_handles.values():
            if handle.id:
                handle.close()
        self._file_handles.clear()

    def __del__(self) -> None:
        self.close()


class PairedRobotSmplDataset(Dataset[MotionClipSample]):
    """Add pure-SMPL Actor tensors to an unchanged robot-reference dataset.

    Pairing follows provenance stored in each robot clip.  In particular, the
    absolute frame offsets and shard layouts of the robot and formal HoloSMPL
    files are never assumed to match.
    """

    def __init__(
        self,
        robot_dataset: Hdf5RootDofDataset,
        *,
        smpl_roots_by_dataset_id: Mapping[str, str],
        smpl_body_mode: str,
        target_fps: float = 50.0,
        smpl_model_path: str | os.PathLike[str] | None = None,
        ground_history_frames: int = 50,
        task_target_source: str = "none",
    ) -> None:
        super().__init__()
        if not smpl_roots_by_dataset_id:
            raise ValueError("paired_hdf5_v2 requires at least one smpl_root")
        # Lazy import keeps the R0 hdf5_v2 module path free from SMPL model
        # loading and any legacy pickle compatibility setup.
        from holomotion.src.motion_tracking.smpl_g1_task_target import (
            SMPL_G1_GROUND_HISTORY_FRAMES,
            SmplG1TaskTargetBuilder,
        )
        from holomotion.src.motion_tracking.smpl_skeleton_target import (
            SmplSkeletonTargetBuilder,
        )
        from holomotion.src.motion_tracking.smpl_reference import (
            SmplReferenceBuilder,
        )

        self.robot_dataset = robot_dataset
        self.windows = robot_dataset.windows
        self.clips = robot_dataset.clips
        self.max_frame_length = robot_dataset.max_frame_length
        self.min_window_length = robot_dataset.min_window_length
        self.hdf5_root = robot_dataset.hdf5_root
        self._smpl_roots_by_dataset_id = {
            _normalize_dataset_id(key): str(value)
            for key, value in smpl_roots_by_dataset_id.items()
        }
        self._smpl_shards_by_dataset_id: Dict[str, frozenset[str]] = {}
        for dataset_id, root_text in self._smpl_roots_by_dataset_id.items():
            # Keep the configured lexical path.  Published formal roots contain
            # shard symlinks into `_rank_outputs`; resolving the symlink target
            # here would incorrectly make a valid shard appear to escape the
            # configured root.
            root_path = Path(os.path.abspath(os.path.expanduser(root_text)))
            manifest_path = root_path / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Formal HoloSMPL manifest not found: {manifest_path}"
                )
            # Formal manifests may exceed 1.5 GB because they retain all sample
            # records.  Never json.load them per DDP rank merely to discover a
            # handful of shards; the schema field is at the manifest header and
            # the published `shards/` directory itself is the exact allowlist.
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_header = handle.read(64 * 1024)
            schema_match = re.search(
                r'"schema_version"\s*:\s*"([^"]+)"', manifest_header
            )
            schema_version = schema_match.group(1) if schema_match else None
            if schema_version != "formal_h5_manifest_v1":
                raise ValueError(
                    "Unsupported formal HoloSMPL manifest schema at "
                    f"{manifest_path}: {schema_version!r}"
                )
            formal_dataset_id = _manifest_dataset_id({}, str(manifest_path))
            if formal_dataset_id != dataset_id:
                raise ValueError(
                    "Configured robot/formal dataset id mismatch: "
                    f"robot={dataset_id!r}, formal={formal_dataset_id!r}, "
                    f"manifest={manifest_path}"
                )
            shards_dir = root_path / "shards"
            if not shards_dir.is_dir():
                raise FileNotFoundError(
                    f"Formal HoloSMPL shards directory not found: {shards_dir}"
                )
            allowed_shards = {
                f"shards/{entry.name}"
                for entry in os.scandir(shards_dir)
                if entry.name.endswith(".h5") and entry.is_file()
            }
            if not allowed_shards:
                raise ValueError(f"No formal shards listed in {manifest_path}")
            self._smpl_roots_by_dataset_id[dataset_id] = str(root_path)
            self._smpl_shards_by_dataset_id[dataset_id] = frozenset(
                allowed_shards
            )
        self._target_fps = float(target_fps)
        self._ground_history_frames = int(ground_history_frames)
        self.task_target_source = str(task_target_source).strip().lower()
        if self.task_target_source not in {
            "none",
            "robot_reference",
            "smpl",
            "smpl_skeleton",
        }:
            raise ValueError(
                "task_target_source must be one of "
                "{'none', 'robot_reference', 'smpl', 'smpl_skeleton'}, got "
                f"{self.task_target_source!r}"
            )
        if self.task_target_source in {"robot_reference", "smpl"}:
            self.robot_dataset._task_target_context = True
        if (
            self.task_target_source in {"robot_reference", "smpl"}
            and self._ground_history_frames != SMPL_G1_GROUND_HISTORY_FRAMES
        ):
            raise ValueError(
                "SMPL task targets require the frozen causal ground history "
                f"{SMPL_G1_GROUND_HISTORY_FRAMES}, got "
                f"{self._ground_history_frames}"
            )
        self._feature_builder = SmplReferenceBuilder(
            smpl_body_mode,
            fps=self._target_fps,
            ground_history_frames=self._ground_history_frames,
            smpl_model_path=smpl_model_path,
        )
        self._task_target_builder = (
            SmplG1TaskTargetBuilder(
                fps=self._target_fps,
                smpl_model_path=smpl_model_path,
            )
            if self.task_target_source in {"robot_reference", "smpl"}
            else None
        )
        self._skeleton_target_builder = (
            SmplSkeletonTargetBuilder(
                fps=self._target_fps,
                smpl_model_path=smpl_model_path,
            )
            if self.task_target_source == "smpl_skeleton"
            else None
        )
        self.smpl_body_mode = str(smpl_body_mode)
        self.smpl_body_dim = int(self._feature_builder.body_dim)
        self.smpl_feature_spec_hash = self._feature_builder.feature_spec_hash
        self.smpl_skeleton_hash = self._feature_builder.skeleton.skeleton_hash
        self.smpl_task_target_spec_hash = (
            self._task_target_builder.feature_spec_hash
            if self._task_target_builder is not None
            else None
        )
        self.smpl_skeleton_target_spec_hash = (
            self._skeleton_target_builder.feature_spec_hash
            if self._skeleton_target_builder is not None
            else None
        )

        self._smpl_file_handles: "OrderedDict[str, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        self._h5_max_open_files = (
            16 if max_open_env is None else max(1, int(max_open_env))
        )
        logger.info(
            "Pure-SMPL feature builder: mode={}, dim={}, spec_hash={}, "
            "skeleton_hash={}, ground_policy=causal_min_j10_j11_{}, "
            "task_target_source={}, task_target_spec_hash={}, "
            "skeleton_target_spec_hash={}",
            self.smpl_body_mode,
            self.smpl_body_dim,
            self.smpl_feature_spec_hash,
            self.smpl_skeleton_hash,
            self._ground_history_frames,
            self.task_target_source,
            self.smpl_task_target_spec_hash,
            self.smpl_skeleton_target_spec_hash,
        )

    def __len__(self) -> int:
        return len(self.robot_dataset)

    def set_progress_counter(self, counter: Optional[mp.Value]) -> None:
        self.robot_dataset.set_progress_counter(counter)

    def _build_motion_key_aliases(
        self, motion_key: str, meta: Mapping[str, Any]
    ) -> Tuple[str, ...]:
        return self.robot_dataset._build_motion_key_aliases(motion_key, meta)

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            value = value.item()
        return str(value)

    @classmethod
    def _decode_json(cls, value: Any) -> Dict[str, Any]:
        return json.loads(cls._decode_text(value))

    @staticmethod
    def _clip_metadata(clip_meta: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = clip_meta.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def _resolve_smpl_shard(
        self,
        *,
        dataset_id: str,
        recorded_root: str,
        source_shard: str,
    ) -> str:
        normalized_id = _normalize_dataset_id(dataset_id)
        configured_root = self._smpl_roots_by_dataset_id.get(normalized_id)
        if configured_root is None:
            raise KeyError(
                f"No smpl_root configured for robot dataset '{normalized_id}'"
            )
        if not recorded_root:
            raise ValueError(
                "Robot manifest is missing source_holosmpl_h5_root; refusing "
                "to guess paired formal shards from a basename"
            )
        source_path = Path(source_shard)
        recorded_path = Path(recorded_root)
        try:
            source_relative = source_path.relative_to(recorded_path)
        except ValueError as exc:
            raise ValueError(
                "source_holosmpl_shard is not under the robot manifest's "
                f"source_holosmpl_h5_root: shard={source_shard!r}, "
                f"root={recorded_root!r}"
            ) from exc
        if (
            len(source_relative.parts) != 2
            or source_relative.parts[0] != "shards"
            or not source_relative.name
        ):
            raise ValueError(
                f"Invalid source_holosmpl_shard provenance: {source_shard!r}"
            )
        relative_text = source_relative.as_posix()
        allowed_shards = self._smpl_shards_by_dataset_id[normalized_id]
        if relative_text not in allowed_shards:
            raise ValueError(
                "Paired formal shard is not listed by the configured formal "
                f"manifest: dataset={normalized_id!r}, shard={relative_text!r}"
            )

        # Relocate the exact manifest-verified relative path, without resolving
        # its final symlink target.  h5py can safely follow the published shard
        # symlink while the pairing identity remains rooted in the manifest.
        return str(Path(configured_root) / source_relative)

    @staticmethod
    def _validate_formal_contract(
        *,
        metadata: Mapping[str, Any],
        pose: np.ndarray,
        motion_key: str,
    ) -> None:
        expected_fields = {
            "canonical_coordinate_system": "canonical_z_up_xyz_m",
            "unit": "meter",
            "up_axis": "z",
        }
        for name, expected in expected_fields.items():
            actual = metadata.get(name)
            if actual != expected:
                raise ValueError(
                    f"Formal contract mismatch for {motion_key}: "
                    f"{name}={actual!r}, expected {expected!r}"
                )

        schema_version = str(metadata.get("schema_version") or "")
        if not schema_version.startswith("formal_human_"):
            raise ValueError(
                f"Unsupported formal schema for {motion_key}: {schema_version!r}"
            )

        layout = str(metadata.get("canonical_pose_body_layout") or "")
        policy = str(metadata.get("formal_pose_72_policy") or "")
        padded_dims = list(metadata.get("formal_pose_72_padded_dims") or [])
        padded_joints = list(
            metadata.get("formal_pose_72_padded_joints") or []
        )
        if layout == "smplx_21_body":
            if (
                policy != "pad_smpl_left_hand_right_hand_with_identity"
                or padded_dims != [66, 72]
                or padded_joints != ["left_hand", "right_hand"]
            ):
                raise ValueError(
                    f"Invalid smplx_21_body padding contract for {motion_key}"
                )
            if not np.allclose(pose[:, 66:72], 0.0, atol=1.0e-7, rtol=0.0):
                raise ValueError(
                    f"Non-identity padded hand rotations in {motion_key}"
                )
        elif layout == "smpl_23_body":
            if (
                policy != "concat_root_and_smpl_23_body"
                or padded_dims
                or padded_joints
            ):
                raise ValueError(
                    f"Invalid smpl_23_body pose contract for {motion_key}"
                )
        else:
            raise ValueError(
                f"Unsupported formal pose layout for {motion_key}: {layout!r}"
            )

    def _get_smpl_shard_handle(self, path: str) -> h5py.File:
        if path in self._smpl_file_handles:
            handle = self._smpl_file_handles.pop(path)
            if handle.id:
                self._smpl_file_handles[path] = handle
                return handle
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Paired formal HoloSMPL shard not found: {path}"
            )
        handle = h5py.File(
            path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=int(
                os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES", 4 * 1024 * 1024)
            ),
            rdcc_w0=0.75,
        )
        if len(self._smpl_file_handles) >= self._h5_max_open_files:
            _, old_handle = self._smpl_file_handles.popitem(last=False)
            old_handle.close()
        self._smpl_file_handles[path] = handle
        return handle

    def _read_smpl_window(
        self,
        *,
        window: MotionWindow,
        clip_meta: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor]:
        robot_metadata = self._clip_metadata(clip_meta)
        source_shard = str(robot_metadata.get("source_holosmpl_shard") or "")
        source_clip_index = robot_metadata.get("source_holosmpl_clip_index")
        if not source_shard or source_clip_index is None:
            raise ValueError(
                f"Robot clip {window.raw_motion_key!r} lacks formal HoloSMPL provenance"
            )
        dataset_id = str(clip_meta.get("_dataset_id") or "")
        shard_path = self._resolve_smpl_shard(
            dataset_id=dataset_id,
            recorded_root=str(clip_meta.get("_source_holosmpl_h5_root") or ""),
            source_shard=source_shard,
        )
        handle = self._get_smpl_shard_handle(shard_path)
        required = (
            "clips/start",
            "clips/length",
            "clips/motion_key_id",
            "clips/metadata_json",
            "human_pose_aa",
            "human_root_trans",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Formal shard {shard_path} is missing {missing}")

        clip_index = int(source_clip_index)
        clip_count = len(handle["clips/start"])
        if clip_index < 0 or clip_index >= clip_count:
            raise IndexError(
                f"Formal clip index {clip_index} outside [0,{clip_count}) in {shard_path}"
            )
        formal_start = int(handle["clips/start"][clip_index])
        formal_length = int(handle["clips/length"][clip_index])
        formal_key = self._decode_text(
            handle["clips/motion_key_id"][clip_index]
        )
        formal_metadata = self._decode_json(
            handle["clips/metadata_json"][clip_index]
        )
        formal_metadata_key = str(formal_metadata.get("clip_id") or "")
        robot_key = str(robot_metadata.get("clip_id") or window.raw_motion_key)
        if not (robot_key == formal_key == formal_metadata_key):
            raise ValueError(
                "Robot/formal clip identity mismatch: "
                f"robot={robot_key!r}, motion_key_id={formal_key!r}, "
                f"formal_metadata={formal_metadata_key!r}"
            )
        robot_clip_start = int(clip_meta.get("start", 0))
        robot_clip_length = int(clip_meta.get("length", 0))
        if formal_length != robot_clip_length:
            raise ValueError(
                f"Robot/formal length mismatch for {robot_key}: "
                f"{robot_clip_length} != {formal_length}"
            )
        robot_fps = float(
            robot_metadata.get(
                "motion_fps", clip_meta.get("motion_fps", self._target_fps)
            )
        )
        formal_fps = float(formal_metadata.get("target_fps", 0.0))
        if (
            abs(robot_fps - self._target_fps) > 1.0e-6
            or abs(formal_fps - self._target_fps) > 1.0e-6
        ):
            raise ValueError(
                f"FPS mismatch for {robot_key}: robot={robot_fps}, "
                f"formal={formal_fps}, expected={self._target_fps}"
            )

        local_offset = int(window.start) - robot_clip_start
        if local_offset < 0 or local_offset + window.length > formal_length:
            raise IndexError(
                f"Window {window.motion_key} is outside paired clip length {formal_length}"
            )
        # Grounding needs H-1 predecessors; V descriptors independently need
        # one real predecessor even when H is configured as 1.
        warmup = min(max(self._ground_history_frames - 1, 1), local_offset)
        read_start = formal_start + local_offset - warmup
        # Actor features retain their historical context exactly.  The opt-in
        # task target additionally reads one real successor when this window
        # does not end at the true clip endpoint, so its last-frame velocity
        # uses the same central-difference convention as a full clip.
        future = int(
            (
                self._task_target_builder is not None
                or self._skeleton_target_builder is not None
            )
            and local_offset + window.length < formal_length
        )
        read_end = formal_start + local_offset + window.length + future
        pose_dataset = handle["human_pose_aa"]
        trans_dataset = handle["human_root_trans"]
        if pose_dataset.dtype != np.dtype(np.float32):
            raise TypeError(
                f"Formal pose dtype for {robot_key} is {pose_dataset.dtype}, "
                "expected float32"
            )
        if trans_dataset.dtype != np.dtype(np.float32):
            raise TypeError(
                f"Formal root dtype for {robot_key} is {trans_dataset.dtype}, "
                "expected float32"
            )
        pose_np = np.asarray(pose_dataset[read_start:read_end])
        trans_np = np.asarray(trans_dataset[read_start:read_end])
        expected_context = warmup + window.length + future
        if pose_np.shape != (expected_context, 72):
            raise ValueError(
                f"Formal pose shape for {robot_key} is {pose_np.shape}, "
                f"expected {(expected_context, 72)}"
            )
        if trans_np.shape != (expected_context, 3):
            raise ValueError(
                f"Formal root shape for {robot_key} is {trans_np.shape}, "
                f"expected {(expected_context, 3)}"
            )
        if not np.isfinite(pose_np).all() or not np.isfinite(trans_np).all():
            raise ValueError(f"Non-finite formal SMPL data in {robot_key}")
        self._validate_formal_contract(
            metadata=formal_metadata,
            pose=pose_np,
            motion_key=robot_key,
        )
        return torch.from_numpy(pose_np), torch.from_numpy(trans_np)

    def __getitem__(self, index: int) -> MotionClipSample:
        robot_sample = self.robot_dataset[index]
        window = self.windows[index]
        clip_meta = self.clips[window.raw_motion_key]
        pose, trans = self._read_smpl_window(
            window=window, clip_meta=clip_meta
        )
        local_offset = int(window.start) - int(clip_meta.get("start", 0))
        warmup = min(max(self._ground_history_frames - 1, 1), local_offset)
        actor_context_end = warmup + window.length
        built = self._feature_builder.build(
            pose[:actor_context_end],
            trans[:actor_context_end],
            output_start=warmup,
            output_length=window.length,
        )
        tensors = dict(robot_sample.tensors)
        tensors["smpl_body_command"] = built.body_command
        tensors["smpl_root_pos"] = built.root_pos
        tensors["smpl_root_rot"] = built.root_rot_xyzw
        if self._task_target_builder is not None:
            task_target = self._task_target_builder.build(
                pose,
                trans,
                output_start=warmup,
                output_length=window.length,
            )
            tensors["smpl_task_position_h"] = task_target.position_h
            tensors["smpl_task_rotation_xyzw_h"] = task_target.rotation_h_xyzw
            tensors["smpl_task_linear_velocity_h"] = (
                task_target.linear_velocity_h
            )
            tensors["smpl_task_angular_velocity_h"] = (
                task_target.angular_velocity_h
            )
        if self._skeleton_target_builder is not None:
            skeleton_target = self._skeleton_target_builder.build(
                pose,
                trans,
                output_start=warmup,
                output_length=window.length,
            )
            tensors["smpl_skeleton_position_h"] = skeleton_target.position_h
            tensors["smpl_skeleton_pelvis_rotation_xyzw_h"] = (
                skeleton_target.pelvis_rotation_xyzw_h
            )
            tensors["smpl_skeleton_torso_rotation_xyzw_h"] = (
                skeleton_target.torso_rotation_xyzw_h
            )
            tensors["smpl_skeleton_endpoint_rotation_xyzw_h"] = (
                skeleton_target.endpoint_rotation_xyzw_h
            )
            tensors["smpl_skeleton_head_rotation_xyzw_h"] = (
                skeleton_target.head_rotation_xyzw_h
            )
            tensors["smpl_skeleton_root_linear_velocity_h"] = (
                skeleton_target.root_linear_velocity_h
            )
            tensors["smpl_skeleton_root_angular_velocity_h"] = (
                skeleton_target.root_angular_velocity_h
            )
            tensors["smpl_skeleton_nominal_scale_h"] = (
                skeleton_target.nominal_scale_h
            )
        return MotionClipSample(
            motion_key=robot_sample.motion_key,
            raw_motion_key=robot_sample.raw_motion_key,
            window_index=robot_sample.window_index,
            tensors=tensors,
            length=robot_sample.length,
        )

    def close(self) -> None:
        for handle in self._smpl_file_handles.values():
            if handle.id:
                handle.close()
        self._smpl_file_handles.clear()
        self.robot_dataset.close()

    def __del__(self) -> None:
        if hasattr(self, "_smpl_file_handles"):
            self.close()


class FormalSmplHdf5Dataset(Dataset[MotionClipSample]):
    """Read formal SMPL shards without opening a robot-reference dataset.

    The accepted input contract is ``formal_h5_manifest_v1`` with frame-major
    ``human_pose_aa[T,72]`` and ``human_root_trans[T,3]`` arrays.  Clip tables
    are read directly from each formal shard, so the very large ``samples``
    list in a production manifest is never loaded into memory.

    This class is deliberately fail-closed: it builds only the SMPL Actor and
    semantic-skeleton target tensors required by the GMR-free policy.  It does
    not synthesize ``ref_*`` aliases and never instantiates
    :class:`Hdf5RootDofDataset`.
    """

    _MANIFEST_SCHEMA = "formal_h5_manifest_v1"
    _SHARD_SCHEMA = "formal_h5_v1"

    def __init__(
        self,
        roots: str | os.PathLike[str] | Sequence[str],
        *,
        max_frame_length: int,
        min_window_length: int = 1,
        smpl_body_mode: str,
        target_fps: float = 50.0,
        smpl_model_path: str | os.PathLike[str] | None = None,
        ground_history_frames: int = 50,
        task_target_source: str = "smpl_skeleton",
        handpicked_motion_names: Optional[Sequence[str]] = None,
        excluded_motion_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        from holomotion.src.motion_tracking.smpl_reference import (
            SmplReferenceBuilder,
        )
        from holomotion.src.motion_tracking.smpl_skeleton_target import (
            SmplSkeletonTargetBuilder,
        )

        if max_frame_length <= 0:
            raise ValueError("max_frame_length must be positive")
        if min_window_length <= 0:
            raise ValueError("min_window_length must be positive")
        normalized_roots = (
            [str(roots)]
            if isinstance(roots, (str, os.PathLike))
            else [str(value) for value in roots]
        )
        if not normalized_roots:
            raise ValueError("At least one formal SMPL root is required")
        source = str(task_target_source).strip().lower()
        if source != "smpl_skeleton":
            raise ValueError(
                "FormalSmplHdf5Dataset supports only task_target_source="
                f"'smpl_skeleton', got {source!r}"
            )

        self.max_frame_length = int(max_frame_length)
        self.min_window_length = int(min_window_length)
        self.hdf5_root = os.path.abspath(
            os.path.expanduser(normalized_roots[0])
        )
        self._target_fps = float(target_fps)
        self._ground_history_frames = int(ground_history_frames)
        self.task_target_source = source
        self.handpicked_motion_names = (
            set(handpicked_motion_names)
            if handpicked_motion_names is not None
            else None
        )
        self.excluded_motion_names = (
            set(excluded_motion_names)
            if excluded_motion_names is not None
            else None
        )
        self._feature_builder = SmplReferenceBuilder(
            smpl_body_mode,
            fps=self._target_fps,
            ground_history_frames=self._ground_history_frames,
            smpl_model_path=smpl_model_path,
        )
        self._skeleton_target_builder = SmplSkeletonTargetBuilder(
            fps=self._target_fps,
            smpl_model_path=smpl_model_path,
        )
        self.smpl_body_mode = str(smpl_body_mode)
        self.smpl_body_dim = int(self._feature_builder.body_dim)
        self.smpl_feature_spec_hash = self._feature_builder.feature_spec_hash
        self.smpl_skeleton_hash = self._feature_builder.skeleton.skeleton_hash
        self.smpl_skeleton_target_spec_hash = (
            self._skeleton_target_builder.feature_spec_hash
        )

        self._shard_paths: List[str] = []
        self.clips: Dict[str, Dict[str, Any]] = {}
        self._load_formal_roots(normalized_roots)
        self.windows = self._enumerate_windows()
        if not self.windows:
            raise ValueError(
                "No formal SMPL windows satisfy the requested length constraints"
            )

        self._progress_counter: Optional[mp.Value] = None
        self._file_handles: "OrderedDict[int, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        self._h5_max_open_files = (
            16 if max_open_env is None else max(1, int(max_open_env))
        )
        self.robot_h5_reads = 0
        logger.info(
            "GMR-free formal-SMPL dataset: roots={}, shards={}, clips={}, "
            "windows={}, body_mode={}, body_dim={}, robot_h5_reads=0",
            len(normalized_roots),
            len(self._shard_paths),
            len(self.clips),
            len(self.windows),
            self.smpl_body_mode,
            self.smpl_body_dim,
        )

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            value = value.item()
        return str(value)

    @classmethod
    def _decode_json(cls, value: Any) -> Dict[str, Any]:
        decoded = json.loads(cls._decode_text(value))
        if not isinstance(decoded, dict):
            raise TypeError(
                "Formal clip metadata_json must decode to a mapping"
            )
        return decoded

    @staticmethod
    def _manifest_header(path: Path) -> str:
        with path.open("r", encoding="utf-8") as handle:
            return handle.read(128 * 1024)

    @staticmethod
    def _header_string(header: str, field: str) -> str | None:
        match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]+)"', header)
        return match.group(1) if match else None

    def _load_formal_roots(self, roots: Sequence[str]) -> None:
        for root_text in roots:
            root = Path(os.path.abspath(os.path.expanduser(root_text)))
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Formal HoloSMPL manifest not found: {manifest_path}"
                )
            header = self._manifest_header(manifest_path)
            schema = self._header_string(header, "schema_version")
            if schema != self._MANIFEST_SCHEMA:
                raise ValueError(
                    f"Unsupported formal manifest schema at {manifest_path}: "
                    f"{schema!r}"
                )
            dataset_name = self._header_string(header, "dataset") or root.name
            dataset_id = _normalize_dataset_id(dataset_name)
            shards_dir = root / "shards"
            shard_paths = sorted(shards_dir.glob("*.h5"))
            if not shard_paths:
                raise ValueError(
                    f"No formal HDF5 shards found under {shards_dir}"
                )

            for shard_path in shard_paths:
                shard_index = len(self._shard_paths)
                self._shard_paths.append(str(shard_path))
                with h5py.File(
                    shard_path, "r", libver="latest", swmr=True
                ) as handle:
                    shard_schema = self._decode_text(
                        handle.attrs.get("schema_version", "")
                    )
                    if shard_schema != self._SHARD_SCHEMA:
                        raise ValueError(
                            f"Unsupported formal shard schema in {shard_path}: "
                            f"{shard_schema!r}"
                        )
                    required = (
                        "clips/start",
                        "clips/length",
                        "clips/motion_key_id",
                        "clips/metadata_json",
                        "human_pose_aa",
                        "human_root_trans",
                    )
                    missing = [name for name in required if name not in handle]
                    if missing:
                        raise KeyError(
                            f"Formal shard {shard_path} is missing {missing}"
                        )
                    if handle["human_pose_aa"].dtype != np.dtype(np.float32):
                        raise TypeError(
                            f"human_pose_aa in {shard_path} must be float32"
                        )
                    if handle["human_root_trans"].dtype != np.dtype(
                        np.float32
                    ):
                        raise TypeError(
                            f"human_root_trans in {shard_path} must be float32"
                        )
                    starts = np.asarray(handle["clips/start"], dtype=np.int64)
                    lengths = np.asarray(
                        handle["clips/length"], dtype=np.int64
                    )
                    keys = handle["clips/motion_key_id"][:]
                    metadata_values = handle["clips/metadata_json"][:]
                    if not (
                        len(starts)
                        == len(lengths)
                        == len(keys)
                        == len(metadata_values)
                    ):
                        raise ValueError(
                            f"Formal clip-table length mismatch in {shard_path}"
                        )
                    for clip_index in range(len(starts)):
                        raw_key = self._decode_text(keys[clip_index])
                        metadata = self._decode_json(
                            metadata_values[clip_index]
                        )
                        metadata_key = str(metadata.get("clip_id") or "")
                        if raw_key != metadata_key:
                            raise ValueError(
                                "Formal clip identity mismatch: "
                                f"motion_key_id={raw_key!r}, metadata={metadata_key!r}"
                            )
                        fps = float(metadata.get("target_fps", 0.0))
                        if abs(fps - self._target_fps) > 1.0e-6:
                            raise ValueError(
                                f"FPS mismatch for {raw_key}: {fps}, "
                                f"expected {self._target_fps}"
                            )
                        unique_key = f"{dataset_id}::{raw_key}"
                        if unique_key in self.clips:
                            raise ValueError(
                                f"Duplicate formal motion key {unique_key!r}"
                            )
                        self.clips[unique_key] = {
                            "start": int(starts[clip_index]),
                            "length": int(lengths[clip_index]),
                            "shard": shard_index,
                            "clip_index": clip_index,
                            "raw_motion_key": raw_key,
                            "metadata": metadata,
                            "motion_fps": fps,
                            "_dataset_id": dataset_id,
                        }

    def _build_motion_key_aliases(
        self, motion_key: str, meta: Mapping[str, Any]
    ) -> Tuple[str, ...]:
        aliases = [motion_key]
        raw_key = str(meta.get("raw_motion_key") or "")
        if raw_key and raw_key not in aliases:
            aliases.append(raw_key)
        return tuple(aliases)

    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        for motion_key, meta in self.clips.items():
            aliases = self._build_motion_key_aliases(motion_key, meta)
            if self.handpicked_motion_names is not None and not any(
                alias in self.handpicked_motion_names for alias in aliases
            ):
                continue
            if self.excluded_motion_names is not None and any(
                alias in self.excluded_motion_names for alias in aliases
            ):
                continue
            clip_start = int(meta["start"])
            clip_length = int(meta["length"])
            offset = 0
            window_index = 0
            while offset < clip_length:
                length = min(self.max_frame_length, clip_length - offset)
                if length >= self.min_window_length:
                    start = clip_start + offset
                    windows.append(
                        MotionWindow(
                            motion_key=(
                                f"{motion_key}__start_{start}_len_{length}"
                            ),
                            shard_index=int(meta["shard"]),
                            start=start,
                            length=length,
                            raw_motion_key=motion_key,
                            window_index=window_index,
                        )
                    )
                    window_index += 1
                offset += length
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def set_progress_counter(self, counter: Optional[mp.Value]) -> None:
        self._progress_counter = counter

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        if shard_index in self._file_handles:
            handle = self._file_handles.pop(shard_index)
            if handle.id:
                self._file_handles[shard_index] = handle
                return handle
        path = self._shard_paths[shard_index]
        handle = h5py.File(
            path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=int(
                os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES", 4 * 1024 * 1024)
            ),
            rdcc_w0=0.75,
        )
        if len(self._file_handles) >= self._h5_max_open_files:
            _, old_handle = self._file_handles.popitem(last=False)
            old_handle.close()
        self._file_handles[shard_index] = handle
        return handle

    @staticmethod
    def _frame_flags(length: int) -> Tensor:
        flags = torch.ones(length, dtype=torch.long)
        if length == 1:
            flags[0] = 2
        elif length > 1:
            flags[0] = 0
            flags[-1] = 2
        return flags

    @staticmethod
    def _scalar_tensor(value: float, length: int) -> Tensor:
        return torch.full((length, 1), float(value), dtype=torch.float32)

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        meta = self.clips[window.raw_motion_key]
        clip_start = int(meta["start"])
        clip_length = int(meta["length"])
        local_offset = int(window.start) - clip_start
        warmup = min(max(self._ground_history_frames - 1, 1), local_offset)
        future = int(local_offset + window.length < clip_length)
        read_start = int(window.start) - warmup
        read_end = int(window.start) + int(window.length) + future
        handle = self._get_shard_handle(window.shard_index)
        pose_np = np.asarray(handle["human_pose_aa"][read_start:read_end])
        trans_np = np.asarray(handle["human_root_trans"][read_start:read_end])
        expected = warmup + int(window.length) + future
        if pose_np.shape != (expected, 72):
            raise ValueError(
                f"Formal pose shape for {window.motion_key} is {pose_np.shape}, "
                f"expected {(expected, 72)}"
            )
        if trans_np.shape != (expected, 3):
            raise ValueError(
                f"Formal root shape for {window.motion_key} is {trans_np.shape}, "
                f"expected {(expected, 3)}"
            )
        if not np.isfinite(pose_np).all() or not np.isfinite(trans_np).all():
            raise ValueError(
                f"Non-finite formal SMPL data in {window.motion_key}"
            )
        PairedRobotSmplDataset._validate_formal_contract(
            metadata=meta["metadata"],
            pose=pose_np,
            motion_key=str(meta["raw_motion_key"]),
        )
        pose = torch.from_numpy(pose_np)
        trans = torch.from_numpy(trans_np)
        output_length = int(window.length)
        built = self._feature_builder.build(
            pose[: warmup + output_length],
            trans[: warmup + output_length],
            output_start=warmup,
            output_length=output_length,
        )
        skeleton = self._skeleton_target_builder.build(
            pose,
            trans,
            output_start=warmup,
            output_length=output_length,
        )
        # The historical paired path exposed raw SMPL pelvis z to the Critic
        # while ``smpl_root_height`` used the Actor's causal grounded root.
        # Keep the semantic pose unchanged but translate every semantic node
        # vertically so both Critic terms share one grounded pelvis-height
        # target.  Pelvis-relative skeleton rewards are invariant to this
        # common translation.
        skeleton_position = skeleton.position_h.clone()
        skeleton_position[..., 2] += (
            built.root_pos[:, 2] - skeleton_position[:, 0, 2]
        )[:, None]
        tensors: Dict[str, Tensor] = {
            "frame_flag": self._frame_flags(output_length),
            "motion_fps": self._scalar_tensor(self._target_fps, output_length),
            "smpl_body_command": built.body_command,
            "smpl_root_pos": built.root_pos,
            "smpl_root_rot": built.root_rot_xyzw,
            "smpl_skeleton_position_h": skeleton_position,
            "smpl_skeleton_pelvis_rotation_xyzw_h": (
                skeleton.pelvis_rotation_xyzw_h
            ),
            "smpl_skeleton_torso_rotation_xyzw_h": (
                skeleton.torso_rotation_xyzw_h
            ),
            "smpl_skeleton_endpoint_rotation_xyzw_h": (
                skeleton.endpoint_rotation_xyzw_h
            ),
            "smpl_skeleton_head_rotation_xyzw_h": (
                skeleton.head_rotation_xyzw_h
            ),
            "smpl_skeleton_root_linear_velocity_h": (
                skeleton.root_linear_velocity_h
            ),
            "smpl_skeleton_root_angular_velocity_h": (
                skeleton.root_angular_velocity_h
            ),
            "smpl_skeleton_nominal_scale_h": skeleton.nominal_scale_h,
        }
        if self._progress_counter is not None:
            with self._progress_counter.get_lock():
                self._progress_counter.value += 1
        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=str(meta["raw_motion_key"]),
            window_index=int(index),
            tensors=tensors,
            length=output_length,
        )

    def close(self) -> None:
        for handle in self._file_handles.values():
            if handle.id:
                handle.close()
        self._file_handles.clear()

    def __del__(self) -> None:
        if hasattr(self, "_file_handles"):
            self.close()


def _normalize_root_list(value: Any) -> List[str]:
    roots, _ = normalize_hdf5_root_entries(value)
    return roots


def normalize_hdf5_root_entries(
    value: Any,
) -> Tuple[List[str], Dict[str, float]]:
    """Parse string roots or ``{root, ratio}`` entries."""

    if value is None:
        return [], {}
    if isinstance(value, (str, os.PathLike)):
        return [str(value)], {}

    roots: List[str] = []
    dataset_ratios: Dict[str, float] = {}
    for index, entry in enumerate(value):
        if isinstance(entry, Mapping):
            root = entry.get("root", entry.get("path"))
            if not root:
                raise ValueError(
                    f"train_hdf5_roots entry {index} requires 'root'"
                )
            root_str = str(root)
            ratio = entry.get("ratio")
            if ratio is not None:
                dataset_id = _manifest_dataset_id_from_path(
                    os.path.join(root_str, "manifest.json")
                )
                if dataset_id in dataset_ratios:
                    raise ValueError(
                        f"Duplicate ratio-bearing HDF5 dataset '{dataset_id}'"
                    )
                dataset_ratios[dataset_id] = float(ratio)
            roots.append(root_str)
            continue
        roots.append(str(entry))
    return roots, dataset_ratios


def paired_smpl_roots_from_entries(value: Any) -> Dict[str, str]:
    """Return ``dataset_id -> formal HoloSMPL root`` from paired entries."""

    if value is None or isinstance(value, (str, os.PathLike)):
        return {}
    result: Dict[str, str] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(
                "paired_hdf5_v2 requires mapping entries with root and smpl_root; "
                f"entry {index} is {type(entry)}"
            )
        robot_root = entry.get("root", entry.get("path"))
        smpl_root = entry.get("smpl_root", entry.get("holosmpl_root"))
        if not robot_root or not smpl_root:
            raise ValueError(
                f"paired_hdf5_v2 entry {index} requires root and smpl_root"
            )
        dataset_id = _manifest_dataset_id(
            {}, os.path.join(str(robot_root), "manifest.json")
        )
        if dataset_id in result:
            raise ValueError(f"Duplicate paired dataset id {dataset_id!r}")
        result[dataset_id] = str(smpl_root)
    return result


def weighted_bin_cfg_from_motion_cfg(
    motion_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = dict(motion_cfg.get("weighted_bin", {}) or {})
    _, root_ratios = normalize_hdf5_root_entries(
        motion_cfg.get("train_hdf5_roots")
    )
    if not root_ratios:
        return cfg
    cfg.pop("bin_regex_patterns", None)
    cfg.pop("bin_regrex_patterns", None)
    cfg.pop("dataset_ratios", None)
    cfg["dataset_ratios"] = root_ratios
    return cfg


def _assert_one_window_per_clip(
    dataset: Dataset[MotionClipSample], *, label: str
) -> None:
    """Reject an evaluation dataset that silently slices or drops clips."""

    windows = getattr(dataset, "windows", None)
    clips = getattr(dataset, "clips", None)
    if windows is None or clips is None:
        raise TypeError(f"{label} dataset does not expose clips/windows")
    window_counts: Dict[str, int] = {}
    for window in windows:
        key = str(window.raw_motion_key)
        window_counts[key] = window_counts.get(key, 0) + 1
    expected_keys = set(str(key) for key in clips)
    actual_keys = set(window_counts)
    repeated = sorted(
        key for key, count in window_counts.items() if count != 1
    )
    if actual_keys != expected_keys or repeated:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{label} must contain exactly one window per original clip: "
            f"clips={len(expected_keys)}, windows={len(windows)}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"repeated={repeated[:5]}"
        )


def build_motion_datasets_from_cfg(
    motion_cfg: Mapping[str, Any],
    *,
    max_frame_length: int,
    min_window_length: int,
    world_frame_normalization: bool = True,
    handpicked_motion_names: Optional[List[str]] = None,
    excluded_motion_names: Optional[List[str]] = None,
    allowed_prefixes: Optional[Sequence[str]] = None,
) -> Tuple[
    Dataset[MotionClipSample],
    Optional[Dataset[MotionClipSample]],
    Dict[str, Any],
]:
    preview_sampling_from_cfg(motion_cfg=motion_cfg)
    backend = str(motion_cfg.get("backend", "hdf5")).lower()
    if backend in ("hdf5", "hdf5_simple"):
        train_roots = _normalize_root_list(
            motion_cfg.get("train_hdf5_roots", None)
        )
        if len(train_roots) == 0:
            hdf5_root = motion_cfg.get("hdf5_root", None)
            if not hdf5_root:
                raise ValueError(
                    "HDF5 backend requires train_hdf5_roots or hdf5_root"
                )
            train_roots = [str(hdf5_root)]
        manifest_paths = [
            os.path.join(str(root), "manifest.json") for root in train_roots
        ]
        train_dataset = Hdf5MotionDataset(
            manifest_path=manifest_paths
            if len(manifest_paths) > 1
            else manifest_paths[0],
            max_frame_length=max_frame_length,
            min_window_length=min_window_length,
            handpicked_motion_names=handpicked_motion_names,
            excluded_motion_names=excluded_motion_names,
            world_frame_normalization=world_frame_normalization,
            allowed_prefixes=allowed_prefixes,
        )

        val_roots = _normalize_root_list(
            motion_cfg.get("val_hdf5_roots", motion_cfg.get("val_hdf5_root"))
        )
        val_dataset = None
        if len(val_roots) > 0:
            val_manifest_paths = [
                os.path.join(str(root), "manifest.json") for root in val_roots
            ]
            val_dataset = Hdf5MotionDataset(
                manifest_path=val_manifest_paths
                if len(val_manifest_paths) > 1
                else val_manifest_paths[0],
                max_frame_length=max_frame_length,
                min_window_length=min_window_length,
                handpicked_motion_names=handpicked_motion_names,
                excluded_motion_names=excluded_motion_names,
                world_frame_normalization=world_frame_normalization,
                allowed_prefixes=allowed_prefixes,
            )
        return train_dataset, val_dataset, {}

    if backend in ("formal_smpl_hdf5", "smpl_hdf5_v1"):
        train_roots = _normalize_root_list(
            motion_cfg.get(
                "train_smpl_hdf5_roots",
                motion_cfg.get("train_hdf5_roots", None),
            )
        )
        if not train_roots:
            root = motion_cfg.get(
                "smpl_hdf5_root", motion_cfg.get("hdf5_root", None)
            )
            if not root:
                raise ValueError(
                    "formal_smpl_hdf5 backend requires "
                    "train_smpl_hdf5_roots or smpl_hdf5_root"
                )
            train_roots = [str(root)]
        common_kwargs = {
            "max_frame_length": max_frame_length,
            "min_window_length": min_window_length,
            "smpl_body_mode": str(motion_cfg.get("smpl_body_mode", "")),
            "target_fps": float(motion_cfg.get("target_fps", 50.0)),
            "smpl_model_path": motion_cfg.get("smpl_model_path", None),
            "ground_history_frames": int(
                motion_cfg.get("smpl_ground_history_frames", 50)
            ),
            "task_target_source": str(
                motion_cfg.get("task_target_source", "smpl_skeleton")
            ),
            "handpicked_motion_names": handpicked_motion_names,
            "excluded_motion_names": excluded_motion_names,
        }
        train_dataset = FormalSmplHdf5Dataset(train_roots, **common_kwargs)

        val_roots = _normalize_root_list(
            motion_cfg.get(
                "val_smpl_hdf5_roots",
                motion_cfg.get(
                    "val_hdf5_roots", motion_cfg.get("val_hdf5_root")
                ),
            )
        )
        val_dataset = (
            FormalSmplHdf5Dataset(val_roots, **common_kwargs)
            if val_roots
            else None
        )
        if bool(motion_cfg.get("require_one_window_per_clip", False)):
            if val_dataset is None:
                raise ValueError(
                    "require_one_window_per_clip requires a validation dataset"
                )
            _assert_one_window_per_clip(val_dataset, label="validation")
        return (
            train_dataset,
            val_dataset,
            {
                "stage_on_swap_only": bool(
                    motion_cfg.get("stage_on_swap_only", True)
                )
            },
        )

    if backend in ("hdf5_v2", "paired_hdf5_v2"):
        fk_robot_file_path = motion_cfg.get("fk_robot_file_path")
        cache_cfg = motion_cfg.get("cache", {})
        allowed_prefixes = cache_cfg.get(
            "allowed_prefixes",
            ["ref_"],
        )
        train_roots = _normalize_root_list(
            motion_cfg.get("train_hdf5_roots", None)
        )
        if len(train_roots) == 0:
            hdf5_root = motion_cfg.get("hdf5_root", None)
            if not hdf5_root:
                raise ValueError(
                    "HDF5 v2 backend requires train_hdf5_roots or hdf5_root"
                )
            train_roots = [str(hdf5_root)]
        train_manifest_paths = [
            os.path.join(str(root), "manifest.json") for root in train_roots
        ]
        train_robot_dataset = Hdf5RootDofDataset(
            manifest_path=train_manifest_paths
            if len(train_manifest_paths) > 1
            else train_manifest_paths[0],
            max_frame_length=max_frame_length,
            min_window_length=min_window_length,
            handpicked_motion_names=handpicked_motion_names,
            excluded_motion_names=excluded_motion_names,
            fk_robot_file_path=fk_robot_file_path,
            allowed_prefixes=allowed_prefixes,
            task_target_context=(
                str(motion_cfg.get("task_target_source", "none"))
                .strip()
                .lower()
                in {"robot_reference", "smpl"}
            ),
        )
        if backend == "paired_hdf5_v2":
            train_dataset = PairedRobotSmplDataset(
                train_robot_dataset,
                smpl_roots_by_dataset_id=paired_smpl_roots_from_entries(
                    motion_cfg.get("train_hdf5_roots")
                ),
                smpl_body_mode=str(motion_cfg.get("smpl_body_mode", "")),
                target_fps=float(motion_cfg.get("target_fps", 50.0)),
                smpl_model_path=motion_cfg.get("smpl_model_path", None),
                ground_history_frames=int(
                    motion_cfg.get("smpl_ground_history_frames", 50)
                ),
                task_target_source=str(
                    motion_cfg.get("task_target_source", "none")
                ),
            )
        else:
            train_dataset = train_robot_dataset

        val_roots = _normalize_root_list(
            motion_cfg.get("val_hdf5_roots", motion_cfg.get("val_hdf5_root"))
        )
        val_dataset = None
        if len(val_roots) > 0:
            val_manifest_paths = [
                os.path.join(str(root), "manifest.json") for root in val_roots
            ]
            val_robot_dataset = Hdf5RootDofDataset(
                manifest_path=val_manifest_paths
                if len(val_manifest_paths) > 1
                else val_manifest_paths[0],
                max_frame_length=max_frame_length,
                min_window_length=min_window_length,
                handpicked_motion_names=handpicked_motion_names,
                excluded_motion_names=excluded_motion_names,
                fk_robot_file_path=fk_robot_file_path,
                allowed_prefixes=allowed_prefixes,
                task_target_context=(
                    str(motion_cfg.get("task_target_source", "none"))
                    .strip()
                    .lower()
                    in {"robot_reference", "smpl"}
                ),
            )
            if backend == "paired_hdf5_v2":
                val_dataset = PairedRobotSmplDataset(
                    val_robot_dataset,
                    smpl_roots_by_dataset_id=paired_smpl_roots_from_entries(
                        motion_cfg.get(
                            "val_hdf5_roots", motion_cfg.get("val_hdf5_root")
                        )
                    ),
                    smpl_body_mode=str(motion_cfg.get("smpl_body_mode", "")),
                    target_fps=float(motion_cfg.get("target_fps", 50.0)),
                    smpl_model_path=motion_cfg.get("smpl_model_path", None),
                    ground_history_frames=int(
                        motion_cfg.get("smpl_ground_history_frames", 50)
                    ),
                    task_target_source=str(
                        motion_cfg.get("task_target_source", "none")
                    ),
                )
            else:
                val_dataset = val_robot_dataset
        if bool(motion_cfg.get("require_one_window_per_clip", False)):
            if val_dataset is None:
                raise ValueError(
                    "require_one_window_per_clip requires a validation dataset"
                )
            _assert_one_window_per_clip(val_dataset, label="validation")
        cache_kwargs = {
            "stage_on_swap_only": bool(
                motion_cfg.get("stage_on_swap_only", True)
            )
        }
        return train_dataset, val_dataset, cache_kwargs

    raise ValueError(f"Unsupported motion backend: {backend}")


def _cache_collate_fn(
    samples: List[MotionClipSample],
    mode: str,
    batch_size: int,
) -> ClipBatch:
    """Collate function for motion cache DataLoader (supports validation padding)."""
    real_sample_count = len(samples)
    if mode == "val" and batch_size > len(samples) and len(samples) > 0:
        extra = batch_size - len(samples)
        gen = torch.Generator()
        idx = torch.randint(0, len(samples), size=(extra,), generator=gen)
        padded = list(samples)
        for i in idx.tolist():
            padded.append(samples[i])
        batch = ClipBatch.collate_fn(padded)
    else:
        batch = ClipBatch.collate_fn(samples)
    # Validation padding is an execution detail required by fixed-size
    # vectorized environments.  Preserve the pre-padding count so evaluators
    # never treat repeated padding clips as independent samples.
    batch.real_sample_count = real_sample_count
    return batch


class InfiniteDistributedSampler(DistributedSampler):
    """Distributed sampler that yields an infinite stream by cycling epochs."""

    def __iter__(self):
        # Infinite stream by cycling epochs
        while True:
            self.set_epoch(getattr(self, "_epoch", 0))
            for idx in super().__iter__():
                yield idx
            self._epoch = getattr(self, "_epoch", 0) + 1


class InfiniteRandomSampler(Sampler[int]):
    """Random sampler that yields infinite reshuffled passes over the dataset."""

    def __init__(self, data_source: Dataset, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        # Yield infinite permutations of indices
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.data_source), generator=g)
            for idx in perm.tolist():
                yield int(idx)
            self.epoch += 1

    def __len__(self) -> int:
        # Large sentinel to satisfy components that query length
        return 2**31 - 1


class WeightedBinInfiniteSampler(Sampler[int]):
    """Infinite sampler that respects regex-based weighted bins over indices."""

    def __init__(
        self,
        dataset_len: int,
        bin_indices: List[List[int]],
        ratios: List[float],
        batch_size: int,
        seed: int,
    ) -> None:
        self._ds_len = int(max(0, dataset_len))
        self._bins = [torch.tensor(b, dtype=torch.long) for b in bin_indices]
        self._ratios = list(ratios)
        self._batch_size = int(max(1, batch_size))
        self._seed = int(seed)
        self._epoch = 0

        raw_counts = [r * float(self._batch_size) for r in self._ratios]
        self._counts = _allocate_batch_counts(
            raw_counts=raw_counts,
            target_total=self._batch_size,
        )

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            batch: List[int] = []
            for bin_idx, count in zip(self._bins, self._counts):
                if count <= 0 or bin_idx.numel() == 0:
                    continue
                choice = torch.randint(
                    0,
                    int(bin_idx.numel()),
                    size=(count,),
                    generator=g,
                )
                selected = bin_idx[choice].tolist()
                batch.extend(int(x) for x in selected)

            if not batch:
                # Fallback: uniform over dataset indices
                if self._ds_len == 0:
                    raise ValueError(
                        "WeightedBinInfiniteSampler cannot sample from an empty dataset"
                    )
                all_idx = torch.randint(
                    0,
                    self._ds_len,
                    size=(self._batch_size,),
                    generator=g,
                )
                batch = [int(x) for x in all_idx.tolist()]

            if len(batch) > self._batch_size:
                batch = batch[: self._batch_size]
            elif len(batch) < self._batch_size:
                pad = self._batch_size - len(batch)
                if pad > 0:
                    batch.extend(batch[:pad])

            perm = torch.randperm(len(batch), generator=g)
            for idx in perm.tolist():
                yield int(batch[idx])
            self._epoch += 1

    def __len__(self) -> int:
        return 2**31 - 1


class PrioritizedInfiniteSampler(Sampler[int]):
    """Infinite sampler with persistent prioritized and fresh uniform pools."""

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        seed: int,
        *,
        p_a_ratio: float = 0.2,
        ema_alpha_signal: float = 0.2,
        ema_alpha_rel_improve: float = 0.2,
        relative_eps: float = 1.0e-6,
    ) -> None:
        self._ds_len = int(max(0, dataset_len))
        self._batch_size = int(max(1, batch_size))
        self._seed = int(seed)
        self._epoch = 0

        self._p_a_ratio = float(min(1.0, max(0.0, p_a_ratio)))
        self._ema_alpha_signal = float(min(1.0, max(0.0, ema_alpha_signal)))
        self._ema_alpha_rel_improve = float(
            min(1.0, max(0.0, ema_alpha_rel_improve))
        )
        self._relative_eps = float(max(1.0e-12, relative_eps))

        if self._ds_len <= 0:
            self._ema_completion_rate = torch.zeros(0, dtype=torch.float32)
            self._ema_completion_rate_sq = torch.zeros(0, dtype=torch.float32)
            self._ema_completion_rel_improve = torch.zeros(
                0, dtype=torch.float32
            )
            self._selection_counts = torch.zeros(0, dtype=torch.long)
            self._seen_mask = torch.zeros(0, dtype=torch.bool)
            self._prioritized_pool_indices = torch.zeros(0, dtype=torch.long)
            self._prioritized_pool_mask = torch.zeros(0, dtype=torch.bool)
        else:
            self._ema_completion_rate = torch.zeros(
                self._ds_len, dtype=torch.float32
            )
            self._ema_completion_rate_sq = torch.zeros(
                self._ds_len, dtype=torch.float32
            )
            self._ema_completion_rel_improve = torch.zeros(
                self._ds_len, dtype=torch.float32
            )
            self._selection_counts = torch.zeros(
                self._ds_len, dtype=torch.long
            )
            self._seen_mask = torch.zeros(self._ds_len, dtype=torch.bool)
            self._prioritized_pool_indices = torch.zeros(0, dtype=torch.long)
            self._prioritized_pool_mask = torch.zeros(
                self._ds_len, dtype=torch.bool
            )
        self._state_version = 0
        self._last_updated_swap = -1
        self._last_prioritized_pool_mean_score = 0.0
        self._last_uniform_pool_mean_score = 0.0
        self._last_entered_prioritized_pool_count = 0
        self._last_exited_prioritized_pool_count = 0
        self._uniform_cycle_start = 0
        self._uniform_cycle_step = 1
        self._uniform_cycle_offset = self._ds_len
        self._uniform_cycle_epoch = 0

    @property
    def state_version(self) -> int:
        return int(self._state_version)

    def get_pool_statistics(self) -> Optional[Dict[str, float]]:
        if self._ds_len <= 0:
            return None
        return self._pool_metric_stats()

    @staticmethod
    def _aggregate_by_index(
        window_indices: Tensor,
        values: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if window_indices.numel() == 0:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
            )
        unique_indices, inverse = torch.unique(
            window_indices.to(dtype=torch.long),
            sorted=False,
            return_inverse=True,
        )
        out_weighted_sum = torch.zeros(
            unique_indices.numel(), dtype=torch.float32
        )
        out_count = torch.zeros(unique_indices.numel(), dtype=torch.float32)
        out_weighted_sum.scatter_add_(0, inverse, values * counts)
        out_count.scatter_add_(0, inverse, counts)
        return unique_indices, out_weighted_sum, out_count

    def _pool_batch_sizes(self) -> Tuple[int, int]:
        if self._ds_len <= 0:
            return 0, 0
        uniform_count = int(round(self._p_a_ratio * float(self._batch_size)))
        uniform_count = max(0, min(self._batch_size, uniform_count))
        prioritized_count = max(0, self._batch_size - uniform_count)
        return uniform_count, prioritized_count

    def _priority_scores_for_indices(self, indices: Tensor) -> Tensor:
        if indices.numel() == 0 or self._ds_len <= 0:
            return torch.zeros(0, dtype=torch.float32)
        idx = indices.to(dtype=torch.long)
        progress = torch.clamp(
            self._ema_completion_rel_improve.index_select(0, idx),
            min=0.0,
            max=1.0,
        )
        remaining_difficulty = torch.clamp(
            1.0 - self._ema_completion_rate.index_select(0, idx),
            min=0.0,
            max=1.0,
        )
        seen = self._seen_mask.index_select(0, idx).to(dtype=torch.float32)
        return progress * remaining_difficulty * seen

    def _pool_metric_stats(self) -> Dict[str, float]:
        prioritized_pool_size = int(self._prioritized_pool_indices.numel())
        return {
            "prioritized_pool_size": float(prioritized_pool_size),
            "prioritized_pool_mean_score": float(
                self._last_prioritized_pool_mean_score
            ),
            "uniform_pool_mean_score": float(
                self._last_uniform_pool_mean_score
            ),
            "entered_prioritized_pool_count": float(
                self._last_entered_prioritized_pool_count
            ),
            "exited_prioritized_pool_count": float(
                self._last_exited_prioritized_pool_count
            ),
        }

    def get_window_state_for_indices(
        self, window_indices: Tensor
    ) -> Dict[str, Tensor]:
        if self._ds_len <= 0:
            empty_bool = torch.zeros(0, dtype=torch.bool)
            empty_float = torch.zeros(0, dtype=torch.float32)
            return {
                "ema_completion_rate": empty_float,
                "completion_rate_rel_improve": empty_float,
                "selection_count": torch.zeros(0, dtype=torch.long),
                "seen": empty_bool,
                "in_prioritized_pool": empty_bool,
            }
        idx = window_indices.detach().to(dtype=torch.long).reshape(-1).cpu()
        if idx.numel() == 0:
            empty_bool = torch.zeros(0, dtype=torch.bool)
            empty_float = torch.zeros(0, dtype=torch.float32)
            return {
                "ema_completion_rate": empty_float,
                "completion_rate_rel_improve": empty_float,
                "selection_count": torch.zeros(0, dtype=torch.long),
                "seen": empty_bool,
                "in_prioritized_pool": empty_bool,
            }
        return {
            "ema_completion_rate": self._ema_completion_rate.index_select(
                0, idx
            ).to(dtype=torch.float32),
            "completion_rate_rel_improve": (
                self._ema_completion_rel_improve.index_select(0, idx).to(
                    dtype=torch.float32
                )
            ),
            "selection_count": self._selection_counts.index_select(0, idx),
            "seen": self._seen_mask.index_select(0, idx),
            "in_prioritized_pool": self._prioritized_pool_mask.index_select(
                0, idx
            ),
        }

    def _rebuild_prioritized_pool(self, candidate_indices: Tensor) -> None:
        if self._ds_len <= 0:
            return
        _, prioritized_count = self._pool_batch_sizes()
        previous_indices = self._prioritized_pool_indices
        selected = torch.zeros(0, dtype=torch.long)
        if prioritized_count > 0:
            candidates = torch.cat(
                [
                    previous_indices.to(dtype=torch.long),
                    candidate_indices.to(dtype=torch.long).reshape(-1),
                ]
            )
            candidates = torch.unique(candidates, sorted=False)
            scores = self._priority_scores_for_indices(candidates)
            positive = scores > 0.0
            if bool(positive.any().item()):
                candidates = candidates[positive]
                scores = scores[positive]
                order = torch.argsort(scores, descending=True)
                selected = candidates.index_select(
                    0, order[: min(prioritized_count, candidates.numel())]
                )
                scores = scores.index_select(
                    0, order[: min(prioritized_count, scores.numel())]
                )
                self._last_prioritized_pool_mean_score = float(
                    scores.mean().item()
                )
            else:
                self._last_prioritized_pool_mean_score = 0.0
            if candidates.numel() > selected.numel():
                selected_mask = torch.zeros(
                    candidates.numel(), dtype=torch.bool
                )
                if selected.numel() > 0:
                    matches = candidates[:, None] == selected[None, :]
                    selected_mask = matches.any(dim=1)
                nonselected_scores = self._priority_scores_for_indices(
                    candidates[~selected_mask]
                )
                self._last_uniform_pool_mean_score = (
                    float(nonselected_scores.mean().item())
                    if nonselected_scores.numel() > 0
                    else 0.0
                )
            else:
                self._last_uniform_pool_mean_score = 0.0
        else:
            self._last_prioritized_pool_mean_score = 0.0
            self._last_uniform_pool_mean_score = 0.0
        if previous_indices.numel() > 0:
            self._prioritized_pool_mask[previous_indices] = False
        if selected.numel() > 0:
            self._prioritized_pool_mask[selected] = True
        previous_set = set(previous_indices.tolist())
        selected_set = set(selected.tolist())
        self._last_entered_prioritized_pool_count = len(
            selected_set - previous_set
        )
        self._last_exited_prioritized_pool_count = len(
            previous_set - selected_set
        )
        self._prioritized_pool_indices = selected

    def maybe_update_from_observations(
        self,
        *,
        window_indices: Tensor,
        mpkpe_signal_means: Tensor,
        completion_rate_means: Tensor,
        counts: Tensor,
        swap_index: int,
    ) -> bool:
        if self._ds_len <= 0:
            return False
        swap_idx = int(swap_index)
        if swap_idx <= 0:
            return False
        if self._last_updated_swap == swap_idx:
            return False

        indices = (
            window_indices.detach().to(dtype=torch.long).reshape(-1).cpu()
        )
        # Keep validating the MPKPE tensor shape so the command-side
        # curriculum aggregation stays aligned with completion-rate updates.
        mpkpe_signal_numel = int(mpkpe_signal_means.numel())
        completion_rate = (
            completion_rate_means.detach()
            .to(dtype=torch.float32)
            .reshape(-1)
            .cpu()
        )
        cnt = counts.detach().to(dtype=torch.float32).reshape(-1).cpu()
        if not (
            indices.numel() == mpkpe_signal_numel
            and mpkpe_signal_numel == completion_rate.numel()
            and completion_rate.numel() == cnt.numel()
        ):
            raise ValueError(
                "Prioritized sampler update tensors must have matching shape."
            )

        valid_dataset_idx = (indices >= 0) & (indices < self._ds_len)
        valid = (
            valid_dataset_idx & torch.isfinite(completion_rate) & (cnt > 0.0)
        )
        current_batch_indices = torch.unique(
            indices[valid_dataset_idx], sorted=False
        )
        if not bool(valid.any().item()):
            self._last_entered_prioritized_pool_count = 0
            self._last_exited_prioritized_pool_count = 0
            self._last_updated_swap = swap_idx
            return False

        idx_valid = indices[valid]
        completion_rate_valid = completion_rate[valid]
        cnt_valid = cnt[valid]

        touched_idx, completion_rate_sum, completion_rate_count_sum = (
            self._aggregate_by_index(
                idx_valid,
                completion_rate_valid,
                cnt_valid,
            )
        )
        if touched_idx.numel() == 0:
            self._last_entered_prioritized_pool_count = 0
            self._last_exited_prioritized_pool_count = 0
            self._last_updated_swap = swap_idx
            return False

        completion_rate_obs = (
            completion_rate_sum / completion_rate_count_sum.clamp_min(1.0e-12)
        )
        completion_rate_obs = torch.clamp(
            completion_rate_obs, min=0.0, max=1.0
        )

        prev_seen = self._seen_mask[touched_idx]
        prev_completion_rate = self._ema_completion_rate[touched_idx]
        prev_completion_rate_sq = self._ema_completion_rate_sq[touched_idx]
        prev_completion_rate_var = torch.clamp(
            prev_completion_rate_sq
            - prev_completion_rate * prev_completion_rate,
            min=1.0e-6,
        )
        prev_completion_rate_std = torch.sqrt(prev_completion_rate_var)
        next_completion_rate = torch.where(
            prev_seen,
            (1.0 - self._ema_alpha_signal) * prev_completion_rate
            + self._ema_alpha_signal * completion_rate_obs,
            completion_rate_obs,
        )
        next_completion_rate_sq = torch.where(
            prev_seen,
            (1.0 - self._ema_alpha_signal) * prev_completion_rate_sq
            + self._ema_alpha_signal
            * (completion_rate_obs * completion_rate_obs),
            completion_rate_obs * completion_rate_obs,
        )

        completion_rel_improve_obs = torch.zeros_like(next_completion_rate)
        completion_rel_improve_obs[prev_seen] = torch.tanh(
            (completion_rate_obs[prev_seen] - prev_completion_rate[prev_seen])
            / (prev_completion_rate_std[prev_seen] + self._relative_eps)
        )
        prev_completion_rel = self._ema_completion_rel_improve[touched_idx]
        next_completion_rel = torch.where(
            prev_seen,
            (1.0 - self._ema_alpha_rel_improve) * prev_completion_rel
            + self._ema_alpha_rel_improve * completion_rel_improve_obs,
            completion_rel_improve_obs,
        )

        self._ema_completion_rate[touched_idx] = next_completion_rate
        self._ema_completion_rate_sq[touched_idx] = next_completion_rate_sq
        self._ema_completion_rel_improve[touched_idx] = next_completion_rel
        self._seen_mask[touched_idx] = True

        self._rebuild_prioritized_pool(touched_idx)
        self._state_version += 1
        self._last_updated_swap = swap_idx
        return True

    def _reset_uniform_cycle(self) -> None:
        if self._ds_len <= 0:
            self._uniform_cycle_start = 0
            self._uniform_cycle_step = 1
            self._uniform_cycle_offset = 0
            return
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._uniform_cycle_epoch * 1000003)
        self._uniform_cycle_epoch += 1
        self._uniform_cycle_start = int(
            torch.randint(
                low=0,
                high=self._ds_len,
                size=(1,),
                generator=generator,
            ).item()
        )
        if self._ds_len <= 1:
            self._uniform_cycle_step = 1
        else:
            step = int(
                torch.randint(
                    low=1,
                    high=self._ds_len,
                    size=(1,),
                    generator=generator,
                ).item()
            )
            while math.gcd(step, self._ds_len) != 1:
                step += 1
                if step >= self._ds_len:
                    step = 1
            self._uniform_cycle_step = step
        self._uniform_cycle_offset = 0

    def _next_uniform_index(self) -> int:
        if self._uniform_cycle_offset >= self._ds_len:
            self._reset_uniform_cycle()
        next_index = (
            self._uniform_cycle_start
            + self._uniform_cycle_offset * self._uniform_cycle_step
        ) % self._ds_len
        self._uniform_cycle_offset += 1
        return int(next_index)

    def _sample_uniform_indices(
        self,
        generator: torch.Generator,
        count: int,
        *,
        exclude: Optional[Tensor] = None,
    ) -> Tensor:
        del generator
        if count <= 0 or self._ds_len <= 0:
            return torch.zeros(0, dtype=torch.long)
        blocked = set()
        if exclude is not None and exclude.numel() > 0:
            blocked.update(
                exclude.detach().to(dtype=torch.long).reshape(-1).tolist()
            )
        take = min(int(count), max(0, self._ds_len - len(blocked)))
        if take <= 0:
            return torch.zeros(0, dtype=torch.long)
        selected: List[int] = []
        stagnant_steps = 0
        while len(selected) < take and stagnant_steps < self._ds_len:
            next_index = self._next_uniform_index()
            if next_index in blocked:
                stagnant_steps += 1
                continue
            selected.append(next_index)
            blocked.add(next_index)
            stagnant_steps = 0
        return torch.tensor(selected, dtype=torch.long)

    def _sample_prioritized_indices(
        self, generator: torch.Generator, count: int
    ) -> Tensor:
        if count <= 0 or self._prioritized_pool_indices.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        perm = torch.randperm(
            self._prioritized_pool_indices.numel(), generator=generator
        )
        take = min(count, int(self._prioritized_pool_indices.numel()))
        return self._prioritized_pool_indices.index_select(0, perm[:take])

    def _sample_batch_indices(self, generator: torch.Generator) -> Tensor:
        uniform_count, prioritized_count = self._pool_batch_sizes()
        prioritized_indices = self._sample_prioritized_indices(
            generator, prioritized_count
        )
        uniform_indices = self._sample_uniform_indices(
            generator,
            uniform_count,
            exclude=prioritized_indices,
        )
        sampled_indices = torch.cat(
            [uniform_indices, prioritized_indices], dim=0
        )
        if sampled_indices.numel() < self._batch_size:
            extra_indices = self._sample_uniform_indices(
                generator,
                self._batch_size - int(sampled_indices.numel()),
                exclude=sampled_indices,
            )
            sampled_indices = torch.cat(
                [sampled_indices, extra_indices], dim=0
            )
        if sampled_indices.numel() != self._batch_size:
            raise ValueError(
                "Prioritized sampler failed to assemble a full cache batch."
            )

        if sampled_indices.numel() > 0:
            self._selection_counts[sampled_indices] += 1
        return sampled_indices

    def get_scores_for_indices(self, window_indices: Tensor) -> Tensor:
        if self._ds_len <= 0:
            return torch.zeros_like(window_indices, dtype=torch.float32)
        idx = window_indices.detach().to(dtype=torch.long).reshape(-1).cpu()
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.float32)
        scores = self._priority_scores_for_indices(idx)
        return scores.to(dtype=torch.float32)

    def __iter__(self):
        while True:
            if self._ds_len <= 0:
                raise ValueError(
                    "PrioritizedInfiniteSampler cannot sample from "
                    "an empty dataset."
                )
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampled_indices = self._sample_batch_indices(generator=g)
            perm = torch.randperm(sampled_indices.numel(), generator=g)
            yielded_indices = sampled_indices.index_select(0, perm)
            for idx in yielded_indices.tolist():
                yield int(idx)
            self._epoch += 1

    def __len__(self) -> int:
        return 2**31 - 1


class Hdf5MotionDataset(Dataset[MotionClipSample]):
    """Dataset that materializes fixed-length motion windows from HDF5 shards."""

    def __init__(
        self,
        manifest_path: str | Sequence[str],
        max_frame_length: int,
        min_window_length: int = 1,
        handpicked_motion_names: Optional[List[str]] = None,
        excluded_motion_names: Optional[List[str]] = None,
        world_frame_normalization: bool = True,
        allowed_prefixes: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if max_frame_length <= 0:
            raise ValueError("max_frame_length must be positive")

        self.max_frame_length = int(max_frame_length)
        self.min_window_length = int(min_window_length)
        self.handpicked_motion_names = (
            set(handpicked_motion_names)
            if handpicked_motion_names is not None
            else None
        )
        self.excluded_motion_names = (
            set(excluded_motion_names)
            if excluded_motion_names is not None
            else None
        )
        self._world_frame_transform = (
            _WorldFrameNormalizeTransform()
            if bool(world_frame_normalization)
            else None
        )
        self._allowed_prefixes: Tuple[str, ...] = ("ref_",)
        self._progress_counter: Optional[mp.Value] = None

        # Normalize manifest path(s) to a list for aggregation.
        if isinstance(manifest_path, (str, os.PathLike)):
            manifest_paths: List[str] = [str(manifest_path)]
        else:
            manifest_paths = [str(p) for p in manifest_path]
        if len(manifest_paths) == 0:
            raise ValueError("At least one manifest_path must be provided")

        # Aggregate shards and clips across one or many manifests into a single
        # logical dataset. Clip keys must be globally unique.
        self.hdf5_root = os.path.dirname(manifest_paths[0])
        self._manifest_paths: List[str] = manifest_paths
        self._shard_paths: List[str] = []
        self.shards: List[Dict[str, Any]] = []
        self.clips: Dict[str, Dict[str, Any]] = {}

        for mp in manifest_paths:
            if not os.path.exists(mp):
                raise FileNotFoundError(
                    f"HDF5 manifest not found at {mp}. "
                    "Please set robot.motion.hdf5_root/train_hdf5_roots "
                    "to the correct path."
                )
            with open(mp, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            root = os.path.dirname(mp)
            shards_local = list(manifest.get("hdf5_shards", []))
            clips_local = manifest.get("clips", {})
            dataset_id = _manifest_dataset_id(manifest, mp)

            shard_offset = len(self.shards)
            for shard_meta in shards_local:
                self.shards.append(shard_meta)
                rel = shard_meta.get("file", None)
                if not isinstance(rel, str) or not rel:
                    raise ValueError(
                        f"Shard entry in manifest {mp} is missing a valid 'file' field"
                    )
                self._shard_paths.append(os.path.join(root, rel))

            for key, meta in clips_local.items():
                if key in self.clips:
                    raise ValueError(
                        f"Duplicate motion clip key '{key}' found in multiple "
                        "manifests; clip keys must be globally unique."
                    )
                meta_global = dict(meta)
                meta_global["_dataset_id"] = dataset_id
                meta_global["shard"] = (
                    int(meta_global.get("shard", 0)) + shard_offset
                )
                self.clips[key] = meta_global

        if len(self.shards) == 0:
            raise ValueError(
                f"No HDF5 shards listed in manifests: {', '.join(manifest_paths)}"
            )

        self.windows: List[MotionWindow] = self._enumerate_windows()
        if len(self.windows) == 0:
            raise ValueError(
                "No motion windows satisfy the requested frame length constraints"
            )

        # LRU cache of open HDF5 shard handles; size is bounded to avoid
        # unbounded host-memory usage from per-file raw chunk caches.
        self._file_handles: "OrderedDict[int, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        if max_open_env is None:
            self._max_open_files = 64
        else:
            self._max_open_files = max(1, int(max_open_env))

    def set_progress_counter(self, counter: Optional[mp.Value]) -> None:
        self._progress_counter = counter

    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        for motion_key, meta in self.clips.items():
            if (
                self.handpicked_motion_names is not None
                and motion_key not in self.handpicked_motion_names
            ):
                continue
            if (
                self.excluded_motion_names is not None
                and motion_key in self.excluded_motion_names
            ):
                continue

            shard_index = int(meta.get("shard", 0))
            start = int(meta.get("start", 0))
            length = int(meta.get("length", 0))

            if length <= 0:
                continue

            remaining = length
            offset = 0
            window_index = 0
            while remaining > 0:
                window_length = min(self.max_frame_length, remaining)
                if window_length >= self.min_window_length:
                    win_start = start + offset
                    unique_key = (
                        f"{motion_key}__start_{win_start}_len_{window_length}"
                    )
                    windows.append(
                        MotionWindow(
                            motion_key=unique_key,
                            shard_index=shard_index,
                            start=win_start,
                            length=window_length,
                            raw_motion_key=motion_key,
                            window_index=window_index,
                        )
                    )
                    window_index += 1
                offset += window_length
                remaining = max(0, length - offset)

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        shard_handle = self._get_shard_handle(window.shard_index)
        start, end = window.start, window.start + window.length

        arrays: Dict[str, Tensor] = {}

        # Mandatory reference source: ref_*
        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            dname = f"ref_{dataset_name}"
            if dname not in shard_handle:
                raise KeyError(
                    f"Missing mandatory dataset '{dname}' in shard index {window.shard_index}"
                )
            np_array = shard_handle[dname][start:end]
            arrays[f"ref_{logical_name}"] = torch.from_numpy(np_array).to(
                torch.float32
            )

        if "frame_flag" in shard_handle:
            frame_flag_np = shard_handle["frame_flag"][start:end]
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 1:
                frame_flag[0] = 0
                frame_flag[-1] = 2
            elif window.length == 1:
                # Single-frame window: mark as both start and end (use 2 for end)
                frame_flag[0] = 2
        arrays["frame_flag"] = frame_flag

        if self._world_frame_transform is not None:
            self._world_frame_transform(arrays)

        # Derived root_* for ref_* (after normalization)
        arrays["ref_root_pos"] = arrays["ref_rg_pos"][:, 0, :]
        arrays["ref_root_rot"] = arrays["ref_rb_rot"][:, 0, :]
        arrays["ref_root_vel"] = arrays["ref_body_vel"][:, 0, :]
        arrays["ref_root_ang_vel"] = arrays["ref_body_ang_vel"][:, 0, :]

        if self._progress_counter is not None:
            with self._progress_counter.get_lock():
                self._progress_counter.value += 1

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            window_index=int(index),
            tensors=arrays,
            length=window.length,
        )

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        if shard_index in self._file_handles:
            handle = self._file_handles.pop(shard_index)
            if handle.id:
                # Mark as most recently used.
                self._file_handles[shard_index] = handle
                return handle

        if shard_index < 0 or shard_index >= len(self._shard_paths):
            raise IndexError(
                f"Shard index {shard_index} out of range for "
                f"{len(self._shard_paths)} available shards"
            )
        shard_path = self._shard_paths[shard_index]
        # Open with SWMR and a configurable raw chunk cache to speed up repeated reads.
        # The default cache size (in bytes) can be overridden via the
        # HOLOMOTION_HDF5_RDCC_NBYTES environment variable.
        rdcc_nbytes_env = os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES")
        if rdcc_nbytes_env is None:
            rdcc_nbytes = 256 * 1024 * 1024  # 256MB default
        else:
            rdcc_nbytes = int(rdcc_nbytes_env)
        handle = h5py.File(
            shard_path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=rdcc_nbytes,
            rdcc_w0=0.75,
        )
        # Enforce LRU limit on the number of simultaneously open shard files.
        if (
            self._max_open_files is not None
            and len(self._file_handles) >= self._max_open_files
        ):
            old_index, old_handle = self._file_handles.popitem(last=False)
            old_handle.close()
        self._file_handles[shard_index] = handle
        return handle

    def close(self) -> None:
        """Close all open HDF5 shard handles for this dataset."""
        for handle in self._file_handles.values():
            if handle.id:
                handle.close()
        self._file_handles.clear()


class MotionClipBatchCache:
    """Double-buffered motion cache for RL training and evaluation."""

    @staticmethod
    def _infer_cuda_device_index() -> int:
        device_count = int(torch.cuda.device_count())
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            if 0 <= local_rank < device_count:
                return local_rank
        return int(torch.cuda.current_device())

    @classmethod
    def _normalize_stage_device(
        cls, stage_device: Optional[object]
    ) -> Optional[torch.device]:
        if stage_device is None:
            return None

        if isinstance(stage_device, torch.device):
            if stage_device.type == "cpu":
                return None
            if stage_device.type != "cuda":
                raise ValueError(
                    f"Unsupported stage_device type: {stage_device.type}"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "stage_device requested CUDA but CUDA is not available"
                )
            if stage_device.index is not None:
                return stage_device
            return torch.device("cuda", cls._infer_cuda_device_index())

        if isinstance(stage_device, str):
            stage_device_str = stage_device.strip().lower()
            if stage_device_str in ("none", "cpu"):
                return None
            if stage_device_str == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "stage_device requested CUDA but CUDA is not available"
                    )
                return torch.device("cuda", cls._infer_cuda_device_index())
            if stage_device_str.startswith("cuda:"):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "stage_device requested CUDA but CUDA is not available"
                    )
                return torch.device(stage_device_str)
            raise ValueError(
                f"Unsupported stage_device string: {stage_device}"
            )

        raise TypeError(
            f"Unsupported stage_device value type: {type(stage_device)}"
        )

    def __init__(
        self,
        train_dataset: Dataset[MotionClipSample],
        *,
        val_dataset: Optional[Dataset[MotionClipSample]] = None,
        batch_size: int,
        stage_device: Optional[torch.device] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler_rank: int = 0,
        sampler_world_size: int = 1,
        allowed_prefixes: Optional[Sequence[str]] = None,
        swap_interval_steps: Optional[int] = None,
        force_timeout_on_swap: bool = True,
        stage_on_swap_only: bool = False,
        batch_progress_bar: bool = False,
        seed: Optional[int] = None,
        loader_timeout: float = 0.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if float(loader_timeout) < 0.0:
            raise ValueError("loader_timeout must be >= 0")

        self._datasets = {
            "train": train_dataset,
            "val": val_dataset if val_dataset is not None else train_dataset,
        }
        self._mode = "train"
        self._seed = (
            int(seed) if seed is not None else int(time.time_ns() & 0x7FFFFFFF)
        )
        self._stage_device = self._normalize_stage_device(stage_device)
        self._sampler_rank = int(sampler_rank)
        self._sampler_world_size = int(max(1, sampler_world_size))
        self._batch_size = int(batch_size)
        self._allowed_prefixes: Optional[Tuple[str, ...]] = (
            tuple(allowed_prefixes) if allowed_prefixes is not None else None
        )

        # If enabled, keep the prefetched batch on CPU (FK on CPU) and stage to GPU
        # only during cache swapping (advance).
        self._stage_on_swap_only = bool(stage_on_swap_only)
        self._batch_progress_bar = bool(batch_progress_bar)
        self._loader_timeout = float(loader_timeout)
        self.force_timeout_on_swap = bool(force_timeout_on_swap)
        self._batch_progress_counter: Optional[mp.Value] = None
        if self._should_use_batch_progress():
            ctx = mp.get_context("spawn")
            self._batch_progress_counter = ctx.Value("i", 0)

        self.swap_interval_steps = (
            swap_interval_steps
            if swap_interval_steps is not None
            else train_dataset.max_frame_length
        )

        self._num_workers = int(max(0, num_workers))
        self._prefetch_factor = (
            prefetch_factor if prefetch_factor is not None else None
        )
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers and num_workers > 0)

        self._dataloader: Optional[DataLoader] = None
        self._sampler: Optional[Sampler[int]] = None
        self._iterator: Optional[Iterator[ClipBatch]] = None

        self._current_batch: Optional[ClipBatch] = None
        self._next_batch: Optional[ClipBatch] = None
        self._swap_index = 0

        self._effective_batch_size: Optional[int] = None
        self._num_batches: Optional[int] = None

        # Weighted-bin sampling state
        self._weighted_bin_enabled: bool = False
        self._weighted_bin_bins: Optional[List[List[int]]] = None
        self._weighted_bin_ratios: Optional[List[float]] = None
        self._weighted_bin_specs: Optional[List[Dict[str, Any]]] = None
        self._cache_curriculum_enabled: bool = False
        self._cache_curriculum_cfg: Dict[str, Any] = {}
        self._cache_curriculum_sampler: Optional[
            PrioritizedInfiniteSampler
        ] = None
        self._cache_curriculum_dump_enabled: bool = False
        self._cache_curriculum_dump_every_swaps: int = 10
        self._cache_curriculum_dump_chunk_size: int = 4096
        self._cache_curriculum_dump_dir: Path = Path(
            "cache_curriculum_window_scores"
        )
        self._cache_curriculum_last_dump_swap: int = -1

        # Async GPU staging helpers
        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        self._build_dataloader()
        if (
            self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            self._copy_stream = torch.cuda.Stream(device=self._stage_device)
        self._prime_buffers()

    @property
    def current_batch(self) -> ClipBatch:
        assert self._current_batch is not None
        return self._current_batch

    @property
    def max_frame_length(self) -> int:
        return self.current_batch.max_frame_length

    @property
    def clip_count(self) -> int:
        """Padded tensor batch size used by the vectorized environment."""
        return self.current_batch.lengths.shape[0]

    @property
    def real_clip_count(self) -> int:
        """Number of non-padding clips in the current cache batch."""
        count = self.current_batch.real_sample_count
        if count is None:
            # Backward compatibility for manually constructed ClipBatch
            # objects and checkpoints/tests predating validation padding
            # metadata.
            return self.clip_count
        count = int(count)
        if count < 0 or count > self.clip_count:
            raise ValueError(
                "Invalid ClipBatch.real_sample_count: "
                f"{count} for padded batch size {self.clip_count}"
            )
        return count

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def swap_index(self) -> int:
        return self._swap_index

    @property
    def num_batches(self) -> int:
        if self._num_batches is None:
            raise RuntimeError("DataLoader is not initialised")
        return int(self._num_batches)

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        if mode not in self._datasets:
            raise ValueError(f"Unknown cache mode: {mode}")
        self._mode = mode
        self._build_dataloader()
        self._prime_buffers()

    def set_seed(self, seed: int, *, reinitialize: bool = True) -> None:
        self._seed = int(seed)
        if reinitialize:
            self._build_dataloader()
            self._prime_buffers()

    def advance(self) -> None:
        if self._stage_on_swap_only:
            if self._next_batch is None:
                self._next_batch = self._fetch_next_batch()
            old_current = self._current_batch
            self._current_batch = None
            del old_current
            # Stage the prefetched CPU batch to GPU only at swap time.
            try:
                staged = self._stage_batch_blocking(self._next_batch)
            except torch.cuda.OutOfMemoryError:
                if (
                    self._stage_device is None
                    or self._stage_device.type != "cuda"
                ):
                    raise
                torch.cuda.empty_cache()
                staged = self._stage_batch_blocking(self._next_batch)
            self._current_batch = staged
            self._next_batch = self._fetch_next_batch()
            self._swap_index += 1
            return

        if self._next_batch is None:
            self._next_batch = self._fetch_next_batch()
        # Ensure asynchronous staging finished before swapping in next batch
        if (
            self._next_ready_event is not None
            and self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            torch.cuda.current_stream(self._stage_device).wait_event(
                self._next_ready_event
            )
        self._current_batch = self._next_batch
        self._next_batch = self._fetch_next_batch()
        self._swap_index += 1

    # -------------------------
    # Weighted-bin configuration
    # -------------------------
    def enable_weighted_bin_sampling(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enable regex-based weighted-bin sampling over manifest motion keys.

        Prefer ``dataset_ratios`` to classify clips by their manifest's
        ``train_*``/``eval_*`` directory. ``bin_regex_patterns`` (or the legacy
        name ``bin_regrex_patterns``) remains available for clip-level matching.
        Each generated or explicit pattern provides:

        - ``regex`` (or ``regrex``): Python regular expression applied to the
          manifest clip key (e.g., ``AMASS_.*``, ``VR_pico_.*``).
        - ``ratio``: Target sampling ratio in [0, 1].

        The sum of explicit bin ratios must be <= 1.0. Any remaining mass is
        assigned to an implicit ``others`` bin that collects all clips not
        matched by any regex.
        """
        cfg_local: Dict[str, Any] = dict(cfg or {})
        if self._cache_curriculum_enabled:
            raise ValueError(
                "weighted-bin and cache curriculum sampling cannot be enabled together."
            )

        dataset = self._datasets.get("train")
        if dataset is None:
            raise ValueError(
                "Weighted-bin sampling requires a training dataset"
            )

        # Collect manifest-level motion keys for all windows in order
        window_keys: List[str] = []
        for window in dataset.windows:
            motion_key = getattr(window, "raw_motion_key", None)
            if motion_key is None:
                full_key = getattr(window, "motion_key", "")
                if "__start_" in full_key:
                    motion_key = full_key.split("__start_", 1)[0]
                else:
                    motion_key = full_key
            clip_metadata = dataset.clips.get(motion_key, {})
            window_keys.append(
                _weighted_bin_motion_key(motion_key, clip_metadata)
            )

        bin_indices, all_ratios, specs = _configure_weighted_bins(
            keys=window_keys,
            cfg=cfg_local,
            batch_size_for_log=int(self._batch_size),
        )

        # Log summary in terms of windows
        table_rows = []
        for item in specs:
            table_rows.append(
                [
                    item["name"],
                    item["regex"],
                    f"{item['ratio']:.4f}",
                    int(item["count"]),
                    f"{item['dataset_fraction']:.4f}",
                    f"{item['batch_fraction']:.4f}",
                ]
            )
        headers = [
            "bin",
            "regex",
            "final_ratio",
            "num_windows",
            "dataset_fraction",
            "batch_fraction",
        ]
        logger.info(
            "Motion cache weighted-bin sampling configured:\n"
            + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
        )

        # Activate weighted-bin sampling and rebuild dataloader/cache
        self._weighted_bin_enabled = True
        self._weighted_bin_bins = bin_indices
        self._weighted_bin_ratios = all_ratios
        self._weighted_bin_specs = specs
        self._build_dataloader()
        self._prime_buffers()

    def enable_cache_curriculum_sampling(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._weighted_bin_enabled:
            raise ValueError(
                "cache curriculum and weighted-bin sampling cannot be enabled together."
            )
        self._cache_curriculum_enabled = True
        self._cache_curriculum_cfg = dict(cfg or {})
        self._cache_curriculum_dump_enabled = bool(
            self._cache_curriculum_cfg.get(
                "dump_whole_window_scores_json", True
            )
        )
        self._cache_curriculum_dump_every_swaps = max(
            1,
            int(
                self._cache_curriculum_cfg.get(
                    "dump_whole_window_scores_every_swaps", 10
                )
            ),
        )
        self._cache_curriculum_dump_chunk_size = max(
            1,
            int(
                self._cache_curriculum_cfg.get(
                    "dump_whole_window_scores_chunk_size", 4096
                )
            ),
        )
        self._cache_curriculum_dump_dir = Path(
            str(
                self._cache_curriculum_cfg.get(
                    "dump_whole_window_scores_dir",
                    "cache_curriculum_window_scores",
                )
            )
        )
        self._cache_curriculum_last_dump_swap = -1
        self._prepare_cache_curriculum_dump_dir(
            self._cache_curriculum_dump_dir,
            reason="enabled",
        )
        self._build_dataloader()
        self._prime_buffers()

    def _prepare_cache_curriculum_dump_dir(
        self, dump_dir: Path, *, reason: str
    ) -> None:
        self._cache_curriculum_dump_dir = Path(str(dump_dir))
        if not self._cache_curriculum_dump_enabled:
            return
        self._cache_curriculum_dump_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Cache curriculum whole-window score dump "
            f"{reason}: dir={self._cache_curriculum_dump_dir}, "
            f"every_swaps={self._cache_curriculum_dump_every_swaps}, "
            f"rank={self._sampler_rank}"
        )

    def set_cache_curriculum_dump_dir(self, dump_dir: str) -> None:
        self._prepare_cache_curriculum_dump_dir(
            Path(str(dump_dir)),
            reason="directory set",
        )

    def update_cache_curriculum(
        self,
        *,
        window_indices: Tensor,
        mpkpe_signal_means: Tensor,
        completion_rate_means: Tensor,
        counts: Tensor,
        swap_index: int,
    ) -> bool:
        if self._cache_curriculum_sampler is None:
            return False
        updated = (
            self._cache_curriculum_sampler.maybe_update_from_observations(
                window_indices=window_indices,
                mpkpe_signal_means=mpkpe_signal_means,
                completion_rate_means=completion_rate_means,
                counts=counts,
                swap_index=swap_index,
            )
        )
        if updated:
            self._refresh_prefetched_batch()
        self._maybe_dump_cache_curriculum_scores_json(swap_index=swap_index)
        return updated

    def _refresh_prefetched_batch(self) -> None:
        if self._next_batch is None:
            return
        self._next_batch = self._fetch_next_batch()

    def _maybe_dump_cache_curriculum_scores_json(
        self, *, swap_index: int
    ) -> None:
        if not self._cache_curriculum_dump_enabled:
            return
        if self._cache_curriculum_sampler is None:
            return

        swap_idx = int(swap_index)
        if swap_idx <= 0:
            return
        if swap_idx % self._cache_curriculum_dump_every_swaps != 0:
            return
        if self._cache_curriculum_last_dump_swap == swap_idx:
            return

        dataset = self._datasets["train"]
        ds_len = int(len(dataset))
        if ds_len <= 0:
            return

        self._cache_curriculum_dump_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._cache_curriculum_dump_dir / (
            "whole_window_scores_"
            f"rank_{self._sampler_rank:04d}_swap_{swap_idx:06d}.json"
        )
        sampler_version = int(self._cache_curriculum_sampler.state_version)
        windows = dataset.windows
        score_values: List[float] = []
        completion_values: List[float] = []
        rel_improve_values: List[float] = []
        selection_count_values: List[int] = []
        seen_values: List[bool] = []
        in_pool_values: List[bool] = []
        chunk_size = max(1, int(self._cache_curriculum_dump_chunk_size))
        for chunk_start in range(0, ds_len, chunk_size):
            chunk_end = min(ds_len, chunk_start + chunk_size)
            chunk_indices = torch.arange(
                chunk_start, chunk_end, dtype=torch.long
            )
            chunk_scores = (
                self._cache_curriculum_sampler.get_scores_for_indices(
                    chunk_indices
                )
            )
            chunk_state = (
                self._cache_curriculum_sampler.get_window_state_for_indices(
                    chunk_indices
                )
            )
            if chunk_scores.numel() != chunk_indices.numel():
                raise ValueError(
                    "Whole-window score dump shape mismatch for "
                    "cache curriculum sampler."
                )
            score_values.extend(chunk_scores.tolist())
            completion_values.extend(
                chunk_state["ema_completion_rate"].tolist()
            )
            rel_improve_values.extend(
                chunk_state["completion_rate_rel_improve"].tolist()
            )
            selection_count_values.extend(
                chunk_state["selection_count"].tolist()
            )
            seen_values.extend(chunk_state["seen"].tolist())
            in_pool_values.extend(chunk_state["in_prioritized_pool"].tolist())
        rows: List[Dict[str, Any]] = []
        for window_index in range(ds_len):
            window = windows[window_index]
            rows.append(
                {
                    "swap_index": int(swap_idx),
                    "rank": int(self._sampler_rank),
                    "sampler_state_version": sampler_version,
                    "window_index": int(window_index),
                    "raw_motion_key": str(window.raw_motion_key),
                    "motion_key": str(window.motion_key),
                    "start": int(window.start),
                    "length": int(window.length),
                    "score": float(score_values[window_index]),
                    "selection_count": int(
                        selection_count_values[window_index]
                    ),
                    "ema_completion_rate": float(
                        completion_values[window_index]
                    ),
                    "completion_rate_rel_improve": float(
                        rel_improve_values[window_index]
                    ),
                    "seen": bool(seen_values[window_index]),
                    "in_prioritized_pool": bool(in_pool_values[window_index]),
                }
            )
        payload: Dict[str, Any] = {
            "swap_index": int(swap_idx),
            "rank": int(self._sampler_rank),
            "sampler_state_version": sampler_version,
            "num_windows": int(ds_len),
            "pool_metrics": self._cache_curriculum_sampler.get_pool_statistics()
            or {},
            "rows": rows,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        self._cache_curriculum_last_dump_swap = swap_idx

    def cache_curriculum_scores_for_window_indices(
        self, window_indices: Tensor
    ) -> Optional[Tuple[Tensor, Dict[str, Tensor], int]]:
        if self._cache_curriculum_sampler is None:
            return None
        scores = self._cache_curriculum_sampler.get_scores_for_indices(
            window_indices
        )
        state = self._cache_curriculum_sampler.get_window_state_for_indices(
            window_indices
        )
        version = self._cache_curriculum_sampler.state_version
        return scores, state, version

    def cache_curriculum_pool_statistics(
        self,
    ) -> Optional[Dict[str, float]]:
        if self._cache_curriculum_sampler is None:
            return None
        return self._cache_curriculum_sampler.get_pool_statistics()

    def sample_env_assignments(
        self,
        num_envs: int,
        n_future_frames: int,
        device: torch.device,
        *,
        deterministic_start: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        batch = self.current_batch
        lengths = batch.lengths.to(device)

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        total = int(lengths.shape[0])
        if total == 0:
            raise ValueError(
                "Cannot sample from an empty batch. Ensure the cache contains "
                "at least one motion clip before calling sample_env_assignments."
            )
        clip_indices = torch.randint(
            low=0, high=total, size=(num_envs,), device=device
        )

        max_start = torch.clamp(
            lengths[clip_indices] - 1 - n_future_frames, min=0
        )
        if deterministic_start:
            frame_starts = torch.zeros_like(max_start)
        else:
            rand = torch.rand_like(max_start, dtype=torch.float32)
            frame_starts = torch.floor(rand * (max_start + 1).float()).to(
                torch.long
            )

        return clip_indices, frame_starts

    def _prepare_gather_indices(
        self,
        *,
        clip_indices: Tensor,
        frame_indices: Tensor,
        n_future_frames: int,
    ) -> Tuple[Tensor, Tensor]:
        batch = self.current_batch
        staged_device = batch.lengths.device
        selected_clips = clip_indices.to(
            staged_device, dtype=torch.long
        ).clone()
        frame_indices = frame_indices.to(
            staged_device, dtype=torch.long
        ).clone()

        temporal_span = 1 + int(n_future_frames)
        time_offsets = torch.arange(
            temporal_span, device=staged_device, dtype=torch.long
        )
        gather_timesteps = frame_indices[:, None] + time_offsets[None, :]

        lengths = batch.lengths
        max_valid = torch.clamp(
            lengths.index_select(0, selected_clips) - 1, min=0
        )
        gather_timesteps = torch.minimum(
            gather_timesteps, max_valid[:, None]
        ).clone()

        return selected_clips, gather_timesteps

    def gather_tensor(
        self,
        tensor_name: str,
        *,
        clip_indices: Tensor,
        frame_indices: Tensor,
        n_future_frames: int,
    ) -> Tensor:
        batch = self.current_batch
        if tensor_name not in batch.tensors:
            raise KeyError(
                f"Tensor '{tensor_name}' is not present in current_batch"
            )
        selected_clips, gather_timesteps = self._prepare_gather_indices(
            clip_indices=clip_indices,
            frame_indices=frame_indices,
            n_future_frames=n_future_frames,
        )
        tensor = batch.tensors[tensor_name]
        return tensor[selected_clips[:, None], gather_timesteps, ...]

    def lengths_for_indices(self, clip_indices: Tensor) -> Tensor:
        lengths = self.current_batch.lengths.to(clip_indices.device)
        return lengths.index_select(0, clip_indices.long())

    def motion_keys_for_indices(self, clip_indices: Tensor) -> List[str]:
        result = []
        base_keys = self.current_batch.motion_keys
        for idx in clip_indices.tolist():
            result.append(base_keys[int(idx)])
        return result

    def window_indices_for_indices(self, clip_indices: Tensor) -> Tensor:
        base_indices = self.current_batch.window_indices.to(
            clip_indices.device
        )
        return base_indices.index_select(0, clip_indices.long())

    def _prime_buffers(self) -> None:
        if self._stage_on_swap_only:
            # Prefetch on CPU; stage to GPU only for current batch.
            cpu_current = self._fetch_next_batch()
            self._current_batch = self._stage_batch_blocking(cpu_current)
            self._next_batch = self._fetch_next_batch()
            self._pending_ready_event = None
            self._current_ready_event = None
            self._next_ready_event = None
            return

        self._current_batch = self._fetch_next_batch()
        # Ensure first staged batch is ready before consumption
        if (
            self._current_ready_event is not None
            and self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            t0 = time.time()
            torch.cuda.current_stream(self._stage_device).wait_event(
                self._current_ready_event
            )
            t1 = time.time()
            logger.info(
                f"Perf/Cache/cuda_wait_event_ms={((t1 - t0) * 1e3):.2f} (first)"
            )
        self._next_batch = self._fetch_next_batch()

    def _fetch_next_batch(self) -> ClipBatch:
        batch = self._load_next_batch()
        if self._stage_on_swap_only:
            # Prefetch raw batch on CPU.
            return batch

        staged = self._stage_batch(batch, record_event=True)
        # Move pending event into current/next slot
        if self._current_batch is None:
            self._current_ready_event = self._pending_ready_event
        else:
            self._next_ready_event = self._pending_ready_event
        self._pending_ready_event = None
        return staged

    def _load_next_batch(self) -> ClipBatch:
        if self._should_use_batch_progress():
            return self._load_next_batch_with_progress()
        return self._load_next_batch_raw()

    def _load_next_batch_raw(self) -> ClipBatch:
        if self._iterator is None:
            self._iterator = self._build_iterator()

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = self._build_iterator(reset_epoch=True)
            batch = next(self._iterator)
        return batch

    def _load_next_batch_with_progress(self) -> ClipBatch:
        if self._iterator is None:
            self._iterator = self._build_iterator()

        expected = int(self._effective_batch_size or self._batch_size)
        counter = self._batch_progress_counter
        if counter is None:
            return self._load_next_batch_raw()

        with counter.get_lock():
            counter.value = 0

        pbar = tqdm(
            total=expected,
            desc="Collecting motion batch",
            leave=False,
            dynamic_ncols=True,
        )
        last = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._load_next_batch_raw)
            while not future.done():
                with counter.get_lock():
                    value = counter.value
                if value > last:
                    step = min(value, expected) - last
                    if step > 0:
                        pbar.update(step)
                        last += step
                time.sleep(0.05)
            batch = future.result(timeout=self._result_timeout())

        with counter.get_lock():
            value = counter.value
        if value > last:
            step = min(value, expected) - last
            if step > 0:
                pbar.update(step)
        pbar.close()
        return batch

    def _stage_batch_blocking(self, batch: ClipBatch) -> ClipBatch:
        """Stage a CPU batch to the configured device on the current stream.

        This path is used when `stage_on_swap_only=True` so that only the current
        cache batch resides on GPU.
        """
        if self._stage_device is None:
            return batch
        non_blocking = bool(
            self._pin_memory and self._stage_device.type == "cuda"
        )
        tensors = {
            name: tensor.to(self._stage_device, non_blocking=non_blocking)
            for name, tensor in batch.tensors.items()
        }
        lengths = batch.lengths.to(
            self._stage_device, non_blocking=non_blocking
        )
        window_indices = batch.window_indices.to(
            self._stage_device, non_blocking=non_blocking
        )
        staged = ClipBatch(
            tensors=tensors,
            lengths=lengths,
            motion_keys=batch.motion_keys,
            raw_motion_keys=getattr(
                batch, "raw_motion_keys", batch.motion_keys
            ),
            window_indices=window_indices,
            max_frame_length=batch.max_frame_length,
            real_sample_count=batch.real_sample_count,
        )
        return staged

    def _stage_batch(
        self,
        batch: ClipBatch,
        record_event: bool = False,
    ) -> ClipBatch:
        if self._stage_device is None:
            return batch

        # If CUDA, copy on a dedicated stream and record readiness
        if self._copy_stream is None and (
            self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            self._copy_stream = torch.cuda.Stream(device=self._stage_device)
            logger.info(
                f"Perf/Cache: created CUDA copy stream lazily on {self._stage_device}"
            )

        if self._copy_stream is not None:
            # estimate payload size for logging
            try:
                total_bytes = 0
                for tensor in batch.tensors.values():
                    total_bytes += int(tensor.element_size() * tensor.numel())
                total_bytes += int(
                    batch.lengths.element_size() * batch.lengths.numel()
                )
                total_bytes += int(
                    batch.window_indices.element_size()
                    * batch.window_indices.numel()
                )
            except Exception:
                total_bytes = -1
            with torch.cuda.stream(self._copy_stream):
                tensors = {
                    name: tensor.to(self._stage_device, non_blocking=True)
                    for name, tensor in batch.tensors.items()
                }
                lengths = batch.lengths.to(
                    self._stage_device, non_blocking=True
                )
                window_indices = batch.window_indices.to(
                    self._stage_device, non_blocking=True
                )
            if record_event:
                ev = torch.cuda.Event()
                ev.record(self._copy_stream)
                self._pending_ready_event = ev

        else:
            tensors = {
                name: tensor.to(self._stage_device, non_blocking=True)
                for name, tensor in batch.tensors.items()
            }
            lengths = batch.lengths.to(self._stage_device, non_blocking=True)
            window_indices = batch.window_indices.to(
                self._stage_device, non_blocking=True
            )

        return ClipBatch(
            tensors=tensors,
            lengths=lengths,
            motion_keys=batch.motion_keys,
            raw_motion_keys=getattr(
                batch, "raw_motion_keys", batch.motion_keys
            ),
            window_indices=window_indices,
            max_frame_length=batch.max_frame_length,
            real_sample_count=batch.real_sample_count,
        )

    def _build_iterator(
        self, *, reset_epoch: bool = False
    ) -> Iterator[ClipBatch]:
        if self._dataloader is None:
            raise RuntimeError("DataLoader is not initialised")

        if isinstance(self._sampler, DistributedSampler) and reset_epoch:
            self._sampler.set_epoch(self._swap_index + 1)

        return iter(self._dataloader)

    def _build_dataloader(self) -> None:
        dataset = self._datasets[self._mode]
        dataset.set_progress_counter(self._batch_progress_counter)

        # Clamp batch size to dataset length to avoid empty iterator when drop_last is disabled
        effective_batch_size = self._batch_size
        ds_len = len(dataset)
        if isinstance(ds_len, int) and ds_len > 0:
            effective_batch_size = max(1, min(self._batch_size, ds_len))

        # Sampler selection: validation uses standard distributed/sequential samplers;
        # training can optionally use weighted-bin sampling.
        if self._mode == "val":
            if self._sampler_world_size > 1:
                self._sampler = DistributedSampler(
                    dataset,
                    num_replicas=self._sampler_world_size,
                    rank=self._sampler_rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                self._sampler = None
            self._cache_curriculum_sampler = None
        else:
            if self._cache_curriculum_enabled:
                seed = self._seed + self._sampler_rank * 100003
                cfg = dict(self._cache_curriculum_cfg)
                self._cache_curriculum_sampler = PrioritizedInfiniteSampler(
                    dataset_len=ds_len,
                    batch_size=effective_batch_size,
                    seed=seed,
                    p_a_ratio=float(cfg.get("p_a_ratio", 0.2)),
                    ema_alpha_signal=float(cfg.get("ema_alpha_signal", 0.2)),
                    ema_alpha_rel_improve=float(
                        cfg.get("ema_alpha_rel_improve", 0.2)
                    ),
                    relative_eps=float(cfg.get("relative_eps", 1.0e-6)),
                )
                self._cache_curriculum_last_dump_swap = -1
                self._sampler = self._cache_curriculum_sampler
            elif (
                self._weighted_bin_enabled
                and self._weighted_bin_bins is not None
                and self._weighted_bin_ratios is not None
            ):
                seed = self._seed + self._sampler_rank * 100003
                self._sampler = WeightedBinInfiniteSampler(
                    dataset_len=ds_len,
                    bin_indices=self._weighted_bin_bins,
                    ratios=self._weighted_bin_ratios,
                    batch_size=effective_batch_size,
                    seed=seed,
                )
                self._cache_curriculum_sampler = None
            else:
                if self._sampler_world_size > 1:
                    # Infinite sampler for training: no epoch boundaries
                    self._sampler = InfiniteDistributedSampler(
                        dataset,
                        num_replicas=self._sampler_world_size,
                        rank=self._sampler_rank,
                        shuffle=True,
                        drop_last=False,
                    )
                else:
                    # Infinite sampler for single-process training
                    self._sampler = InfiniteRandomSampler(dataset)
                self._cache_curriculum_sampler = None

        # Only pass prefetch_factor when using workers
        pf = (
            self._prefetch_factor
            if (self._num_workers and self._num_workers > 0)
            else None
        )
        pw = (
            self._persistent_workers
            if (self._num_workers and self._num_workers > 0)
            else False
        )

        # Collate wrapper: in validation, pad the batch up to cache size by
        # uniformly repeating samples when dataset is smaller than batch size.
        collate = partial(
            _cache_collate_fn,
            mode=self._mode,
            batch_size=self._batch_size,
        )

        mp_ctx = None
        if self._num_workers and self._num_workers > 0:
            mp_ctx = mp.get_context("spawn")

        worker_init_fn = None
        if (
            self._num_workers > 0
            and self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            worker_init_fn = _cpu_only_dataloader_worker_init_fn

        self._dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            sampler=self._sampler,
            shuffle=(self._sampler is None and self._mode != "val"),
            num_workers=self._num_workers,
            prefetch_factor=pf,
            pin_memory=self._pin_memory,
            timeout=self._loader_timeout_seconds(),
            persistent_workers=pw,
            collate_fn=collate,
            drop_last=False,
            multiprocessing_context=mp_ctx,
            worker_init_fn=worker_init_fn,
        )
        self._iterator = None
        self._current_batch = None
        self._next_batch = None
        self._swap_index = 0

        # Compute number of batches only for validation; training is infinite
        local_len = ds_len
        if self._mode == "val":
            if self._sampler is not None:
                local_len = (
                    ds_len + self._sampler_world_size - 1
                ) // self._sampler_world_size
            self._effective_batch_size = int(effective_batch_size)
            self._num_batches = (
                local_len + self._effective_batch_size - 1
            ) // self._effective_batch_size
        else:
            self._effective_batch_size = int(effective_batch_size)
            self._num_batches = 2**31  # effectively infinite for logging

    def close(self) -> None:
        """Release DataLoader workers and close underlying HDF5 datasets."""
        datasets = self.__dict__.get("_datasets")
        if datasets is None:
            return
        self._iterator = None
        self._current_batch = None
        self._next_batch = None
        self._dataloader = None
        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        for ds in datasets.values():
            if ds is not None:
                ds.close()

    def __del__(self) -> None:
        self.close()

    def _loader_timeout_seconds(self) -> float:
        if not self.force_timeout_on_swap:
            return 0.0
        return self._loader_timeout

    def _result_timeout(self) -> Optional[float]:
        timeout_s = self._loader_timeout_seconds()
        if timeout_s <= 0.0:
            return None
        return timeout_s + 1.0

    def _should_use_batch_progress(self) -> bool:
        if not self._batch_progress_bar:
            return False
        if self._sampler_world_size > 1:
            return False
        if self._loader_timeout_seconds() > 0.0:
            return False
        return True
