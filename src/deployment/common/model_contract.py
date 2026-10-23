"""Immutable artifact bundle contracts shared by robot deployments."""

from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ArtifactStatus(str, Enum):
    TRAINED = "trained"
    SMOKE = "smoke"
    UNTRAINED = "untrained"
    LICENSED_ASSET = "licensed_asset"
    METADATA = "metadata"


class DeploymentScope(str, Enum):
    SHADOW = "shadow"
    SUSPENDED_VALIDATION = "suspended_validation"
    GROUND = "ground"


@dataclass(frozen=True)
class ImmutableArtifact:
    name: str
    path: Path
    sha256: str
    status: ArtifactStatus


@dataclass(frozen=True)
class ModelBundle:
    manifest_path: Path
    schema: str
    robot: str
    policy_hz: float
    deployment_scope: DeploymentScope
    artifacts: Mapping[str, ImmutableArtifact]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return _mapping(json.loads(text), "model bundle")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - image packaging guard
        raise RuntimeError(
            "PyYAML is required for YAML model bundles"
        ) from exc
    return _mapping(yaml.safe_load(text) or {}, "model bundle")


def _sha256(value: Any, label: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(
        char not in "0123456789abcdef" for char in result
    ):
        raise ValueError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return result


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a bundle and reject paths escaping its immutable directory."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"model bundle manifest is missing: {manifest_path}"
        )
    value = _load_mapping(manifest_path)
    schema = str(value.get("schema", ""))
    if schema != "holomotion_model_bundle_v1":
        raise ValueError(f"unsupported model bundle schema: {schema!r}")
    robot = str(value.get("robot", "")).strip()
    if not robot:
        raise ValueError("model bundle robot is required")
    policy_hz = float(value.get("policy_hz", 0.0))
    if not math.isfinite(policy_hz) or policy_hz <= 0.0:
        raise ValueError("model bundle policy_hz must be finite and positive")
    try:
        scope = DeploymentScope(str(value.get("deployment_scope", "")))
    except ValueError as exc:
        raise ValueError("model bundle deployment_scope is invalid") from exc

    root = manifest_path.parent
    artifact_values = _mapping(
        value.get("artifacts"), "model bundle artifacts"
    )
    required = {"motion", "velocity", "smpl"}
    missing = sorted(required - set(artifact_values))
    if missing:
        raise ValueError(f"model bundle is missing artifacts: {missing}")
    artifacts: dict[str, ImmutableArtifact] = {}
    for name, raw in artifact_values.items():
        item = _mapping(raw, f"artifact {name}")
        relative = Path(str(item.get("path", "")))
        if not str(relative) or relative.is_absolute():
            raise ValueError(f"artifact {name} path must be relative")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"artifact {name} must remain inside its bundle")
        try:
            status = ArtifactStatus(str(item.get("status", "")))
        except ValueError as exc:
            raise ValueError(f"artifact {name} status is invalid") from exc
        artifacts[str(name)] = ImmutableArtifact(
            name=str(name),
            path=resolved,
            sha256=_sha256(item.get("sha256", ""), f"artifact {name} sha256"),
            status=status,
        )
    return ModelBundle(
        manifest_path=manifest_path,
        schema=schema,
        robot=robot,
        policy_hz=policy_hz,
        deployment_scope=scope,
        artifacts=artifacts,
    )


def verify_model_bundle(
    bundle: ModelBundle, *, require_trained: bool = False
) -> ModelBundle:
    """Verify every byte hash and optional real-policy provenance."""

    for name, artifact in bundle.artifacts.items():
        if not artifact.path.is_file():
            raise FileNotFoundError(
                f"artifact {name} is missing: {artifact.path}"
            )
        actual = sha256_file(artifact.path)
        if actual != artifact.sha256:
            raise ValueError(
                f"artifact {name} SHA256 mismatch: expected "
                f"{artifact.sha256}, got {actual}"
            )
    if require_trained:
        for name in ("motion", "velocity"):
            if bundle.artifacts[name].status is not ArtifactStatus.TRAINED:
                raise ValueError(
                    f"real deployment requires trained {name} policy"
                )
    if bundle.artifacts["smpl"].status is not ArtifactStatus.LICENSED_ASSET:
        raise ValueError("SMPL artifact must be marked licensed_asset")
    return bundle


__all__ = [
    "ArtifactStatus",
    "DeploymentScope",
    "ImmutableArtifact",
    "ModelBundle",
    "load_model_bundle",
    "sha256_file",
    "verify_model_bundle",
]
