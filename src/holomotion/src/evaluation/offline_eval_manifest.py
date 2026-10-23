"""Manifest helpers for auditable IsaacLab offline evaluation clips."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

OFFLINE_EVAL_CLIP_MANIFEST_FILENAME = "offline_eval_clip_manifest.json"
OFFLINE_EVAL_CLIP_MANIFEST_SCHEMA_VERSION = 1
OFFLINE_EVAL_PROTOCOL = "isaaclab_legacy_full_future_v1"

FIXED_CASE_PROTOCOL = "holomotion_acceptance_fixed_cases_v1"


def build_fixed_case_manifest(
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze explicit case/seed/duration identities before viewing outcomes.

    This is separate from the legacy full-future protocol: no future-horizon
    trimming or a robot-specific failure rule is implied by this manifest.
    Source windows, groups, commands and termination rules can be attached to
    each record; they are all covered by the same content fingerprint.
    """
    records = [dict(case) for case in cases]
    manifest = {
        "protocol": FIXED_CASE_PROTOCOL,
        "expected_case_count": len(records),
        "cases": records,
        "cases_sha256": _clips_fingerprint(records),
    }
    validate_fixed_case_manifest(manifest)
    return manifest


def validate_fixed_case_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if manifest.get("protocol") != FIXED_CASE_PROTOCOL:
        raise ValueError("Unsupported fixed-case protocol")
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or manifest.get("expected_case_count") != len(cases)
    ):
        raise ValueError("Expected case count must match a nonempty case list")
    if manifest.get("cases_sha256") != _clips_fingerprint(cases):
        raise ValueError("Fixed-case manifest fingerprint mismatch")
    result, ids = {}, set()
    for case in cases:
        filename, case_id = case.get("output_filename"), case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Empty or duplicate case_id")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".npz")
            or filename in result
        ):
            raise ValueError("Unsafe or duplicate fixed-case output filename")
        if isinstance(case.get("seed"), bool) or not isinstance(
            case.get("seed"), int
        ):
            raise ValueError("Every fixed case requires an integer seed")
        frames = case.get("planned_num_frames")
        if (
            isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames <= 0
        ):
            raise ValueError(
                "Every fixed case requires positive planned_num_frames"
            )
        if (
            not isinstance(case.get("group"), str)
            or not case["group"]
            or case["group"] == "all"
        ):
            raise ValueError(
                "Every fixed case requires a declared group; 'all' is reserved"
            )
        ids.add(case_id)
        result[filename] = dict(case)
    return result


def sanitize_offline_eval_motion_key(motion_key: str) -> str:
    """Return the NPZ stem used by the offline evaluator."""

    key = str(motion_key)
    if not key:
        raise ValueError("offline-eval motion_key must not be empty")
    return key.replace("/", "+").replace(" ", "_").replace("\\", "+")


def _window_value(window: Any, name: str, default: Any = None) -> Any:
    if isinstance(window, Mapping):
        return window.get(name, default)
    return getattr(window, name, default)


def _clips_fingerprint(clips: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        clips,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_offline_eval_clip_manifest(
    windows: Iterable[Any],
    *,
    n_future_frames: int,
    dataset_suffix: str | None = None,
) -> dict[str, Any]:
    """Describe the exact one-window-per-clip evaluation protocol."""

    horizon = int(n_future_frames)
    if horizon < 0:
        raise ValueError("n_future_frames must be non-negative")

    clips: list[dict[str, Any]] = []
    seen_motion_keys: set[str] = set()
    output_owner: dict[str, str] = {}
    for ordinal, window in enumerate(windows):
        motion_key = str(_window_value(window, "motion_key", ""))
        raw_motion_key = str(
            _window_value(window, "raw_motion_key", motion_key)
        )
        length = int(_window_value(window, "length", 0))
        start = int(_window_value(window, "start", 0))
        window_index = int(_window_value(window, "window_index", ordinal))
        if not motion_key:
            raise ValueError(f"window {ordinal} has an empty motion_key")
        if motion_key in seen_motion_keys:
            raise ValueError(
                f"duplicate offline-eval motion_key: {motion_key!r}"
            )
        if length <= horizon:
            raise ValueError(
                f"offline-eval clip {motion_key!r} has length={length}, "
                f"which provides no scored frame for future horizon={horizon}"
            )

        metric_motion_key = sanitize_offline_eval_motion_key(motion_key)
        output_filename = f"{metric_motion_key}.npz"
        previous_owner = output_owner.get(output_filename)
        if previous_owner is not None:
            raise ValueError(
                "offline-eval filename collision after sanitization: "
                f"{previous_owner!r} and {motion_key!r} -> {output_filename!r}"
            )
        seen_motion_keys.add(motion_key)
        output_owner[output_filename] = motion_key
        clips.append(
            {
                "ordinal": ordinal,
                "motion_key": motion_key,
                "raw_motion_key": raw_motion_key,
                "metric_motion_key": metric_motion_key,
                "output_filename": output_filename,
                "window_index": window_index,
                "start": start,
                "clip_length": length,
                "max_scored_frames": length - horizon,
            }
        )

    if not clips:
        raise ValueError(
            "offline evaluation requires at least one clip window"
        )
    return {
        "schema_version": OFFLINE_EVAL_CLIP_MANIFEST_SCHEMA_VERSION,
        "protocol": OFFLINE_EVAL_PROTOCOL,
        "dataset_suffix": dataset_suffix,
        "n_future_frames": horizon,
        "expected_clip_count": len(clips),
        "clips_sha256": _clips_fingerprint(clips),
        "clips": clips,
    }


def validate_offline_eval_clip_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate a manifest and return records by metrics motion key."""

    if (
        manifest.get("schema_version")
        != OFFLINE_EVAL_CLIP_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported offline-eval clip manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("protocol") != OFFLINE_EVAL_PROTOCOL:
        raise ValueError(
            f"unsupported offline-eval protocol: {manifest.get('protocol')!r}"
        )
    horizon = manifest.get("n_future_frames")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 0
    ):
        raise ValueError(
            "manifest n_future_frames must be a non-negative integer"
        )
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("manifest clips must be a non-empty list")
    if manifest.get("expected_clip_count") != len(clips):
        raise ValueError("manifest expected_clip_count does not match clips")
    if manifest.get("clips_sha256") != _clips_fingerprint(clips):
        raise ValueError("manifest clips_sha256 does not match clips")

    by_metric_key: dict[str, dict[str, Any]] = {}
    seen_motion_keys: set[str] = set()
    for index, raw_record in enumerate(clips):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"manifest clip {index} must be a mapping")
        record = dict(raw_record)
        motion_key = record.get("motion_key")
        metric_motion_key = record.get("metric_motion_key")
        clip_length = record.get("clip_length")
        max_scored_frames = record.get("max_scored_frames")
        if not isinstance(motion_key, str) or not motion_key:
            raise ValueError(f"manifest clip {index} has invalid motion_key")
        if motion_key in seen_motion_keys:
            raise ValueError(
                f"manifest contains duplicate motion_key {motion_key!r}"
            )
        expected_metric_key = sanitize_offline_eval_motion_key(motion_key)
        if metric_motion_key != expected_metric_key:
            raise ValueError(
                f"manifest clip {motion_key!r} has invalid metric_motion_key"
            )
        if metric_motion_key in by_metric_key:
            raise ValueError(
                "manifest contains duplicate metric_motion_key "
                f"{metric_motion_key!r}"
            )
        if (
            isinstance(clip_length, bool)
            or not isinstance(clip_length, int)
            or clip_length <= horizon
        ):
            raise ValueError(
                f"manifest clip {motion_key!r} has invalid clip_length"
            )
        if max_scored_frames != clip_length - horizon:
            raise ValueError(
                f"manifest clip {motion_key!r} has inconsistent max_scored_frames"
            )
        expected_filename = f"{metric_motion_key}.npz"
        if record.get("output_filename") != expected_filename:
            raise ValueError(
                f"manifest clip {motion_key!r} has invalid output_filename"
            )
        seen_motion_keys.add(motion_key)
        by_metric_key[metric_motion_key] = record
    return by_metric_key


def write_offline_eval_clip_manifest(
    output_dir: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> Path:
    """Atomically write a validated manifest in an eval output directory."""

    validate_offline_eval_clip_manifest(manifest)
    output_path = Path(output_dir) / OFFLINE_EVAL_CLIP_MANIFEST_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest, handle, indent=2, ensure_ascii=False, sort_keys=True
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    return output_path


def load_offline_eval_clip_manifest(
    output_dir: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and validate the single required manifest in an eval output."""

    path = Path(output_dir) / OFFLINE_EVAL_CLIP_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"missing offline-eval clip manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("offline-eval clip manifest root must be a mapping")
    return manifest, validate_offline_eval_clip_manifest(manifest)


__all__ = [
    "OFFLINE_EVAL_CLIP_MANIFEST_FILENAME",
    "OFFLINE_EVAL_CLIP_MANIFEST_SCHEMA_VERSION",
    "OFFLINE_EVAL_PROTOCOL",
    "build_offline_eval_clip_manifest",
    "load_offline_eval_clip_manifest",
    "sanitize_offline_eval_motion_key",
    "validate_offline_eval_clip_manifest",
    "write_offline_eval_clip_manifest",
]
