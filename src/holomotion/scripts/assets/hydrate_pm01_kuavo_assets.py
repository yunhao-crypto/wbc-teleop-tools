#!/usr/bin/env python3
"""Hydrate large pinned PM01 USD and Kuavo S54 mesh assets locally."""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PM01_COMMIT = "14ec57be718586bd0ac45375aa1115bd896fbdbc"
KUAVO_COMMIT = "845f373949d9084e59510f96cf4c82a2089f0781"
PM01_TREE_SHA256 = (
    "c19d48c4d8b146a03b518d4b4b186887df936e22245ed73abedf9200378719f6"
)
KUAVO_TREE_SHA256 = (
    "bcf9c4f7b56bbcc15ff199e4b3752ea5166bd93975fb14a15578356f0217d0f3"
)


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _tree_digest(root: Path) -> tuple[int, int, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing hydrated asset directory: {root}")
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        count += 1
        size += len(data)
    return count, size, digest.hexdigest()


def _verify_tree(root: Path, expected: str, label: str) -> dict[str, object]:
    count, size, actual = _tree_digest(root)
    if actual != expected:
        raise ValueError(
            f"{label} tree SHA256 mismatch: expected {expected}, got {actual}"
        )
    return {"files": count, "bytes": size, "tree_sha256": actual}


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Missing official asset source: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pm01-rl-lab", type=Path)
    parser.add_argument("--kuavo-train", type=Path)
    parser.add_argument(
        "--asset",
        choices=("all", "pm01", "kuavo"),
        default="all",
        help="Hydrate or verify only the selected embodiment assets.",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    pm_target = repo / "assets/robots/engineai/PM01/24dof/usd"
    kuavo_target = repo / "assets/robots/leju/Kuavo5/S54/meshes"
    if not args.verify_only:
        if args.asset in ("all", "pm01"):
            if args.pm01_rl_lab is None:
                parser.error("PM01 hydration requires --pm01-rl-lab")
            pm_source_root = args.pm01_rl_lab.resolve()
            if _git_head(pm_source_root) != PM01_COMMIT:
                raise ValueError(
                    "PM01 source checkout is not the pinned commit"
                )
            _copy_tree(
                pm_source_root
                / "source/engineai_rl_lab/engineai_rl_lab/assets/pm01",
                pm_target,
            )
        if args.asset in ("all", "kuavo"):
            if args.kuavo_train is None:
                parser.error("Kuavo hydration requires --kuavo-train")
            kuavo_source_root = args.kuavo_train.resolve()
            if _git_head(kuavo_source_root) != KUAVO_COMMIT:
                raise ValueError(
                    "Kuavo source checkout is not the pinned commit"
                )
            _copy_tree(
                kuavo_source_root
                / "source/leju_robot/leju_robot/assets/robots/kuavos54/meshes",
                kuavo_target,
            )

    receipt: dict[str, object] = {}
    receipt_roots: list[Path] = []
    if args.asset in ("all", "pm01"):
        receipt.update(
            pm01_source_commit=PM01_COMMIT,
            pm01_usd=_verify_tree(pm_target, PM01_TREE_SHA256, "PM01 USD"),
        )
        receipt_roots.append(pm_target.parent)
    if args.asset in ("all", "kuavo"):
        receipt.update(
            kuavo_source_commit=KUAVO_COMMIT,
            kuavo_meshes=_verify_tree(
                kuavo_target, KUAVO_TREE_SHA256, "Kuavo meshes"
            ),
        )
        receipt_roots.append(kuavo_target.parent)
    for root in receipt_roots:
        (root / "ASSET_HYDRATION_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
