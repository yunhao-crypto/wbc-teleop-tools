"""Select a checksum-pinned Motion experiment without changing robot behavior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


MOTION_GROUPS = (*tuple("ABCDE"), "P1", "P2", "P3")


def load_motion_group(root, group):
    root = Path(root).resolve()
    if group not in MOTION_GROUPS:
        raise ValueError("Motion group must be A, B, C, D, E, P1, P2 or P3")
    index = json.loads((root / "config/motion_groups.json").read_text())
    schema = index.get("schema")
    if schema not in ("vita_motion_groups_v1", "vita_motion_groups_v2"):
        raise ValueError("Unsupported Motion group index")
    item = index["groups"][group]
    iteration = item["iteration"]
    if type(iteration) is not int or iteration <= 0:
        raise ValueError("Invalid checkpoint iteration")
    comparison = index.get("comparison", "same_iteration")
    if comparison not in ("same_iteration", "latest_per_group"):
        raise ValueError("Unknown comparison mode")
    if schema == "vita_motion_groups_v1" or comparison == "same_iteration":
        if iteration != index["iteration"]:
            raise ValueError("All Motion groups must use the same checkpoint iteration")
    elif set(index["groups"]) - {"P1", "P2", "P3"}:
        raise ValueError("latest_per_group is only supported for P1/P2/P3")
    source_hash = item["source_sha256"]
    if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("Invalid source model SHA256")
    result = {"group": group, "iteration": iteration, "source_sha256": source_hash}
    for name in ("manifest", "contracts", "envelopes"):
        entry = item[name]
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Model package path escapes deployment root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Model package SHA256 mismatch: " + name)
        result[name] = path
        result[name + "_sha256"] = entry["sha256"]
    manifest = json.loads(result["manifest"].read_text())
    if manifest.get("experiment") != {"group": group, "iteration": iteration}:
        raise ValueError("Motion manifest experiment mismatch")
    if manifest["source_hashes"]["motion"] != source_hash:
        raise ValueError("Motion manifest source identity mismatch")
    return result
