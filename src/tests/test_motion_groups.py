"""Model selection must fail before inference if identity or iteration differs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "deployment/vita_s100_aorta_30dof/motion_groups.py"
spec = importlib.util.spec_from_file_location("motion_groups", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def package(tmp_path):
    (tmp_path / "config").mkdir()
    groups = {}
    for group in "ABCDE":
        directory = tmp_path / group
        directory.mkdir()
        source = hashlib.sha256(group.encode()).hexdigest()
        manifest = {"experiment": {"group": group, "iteration": 11000},
                    "source_hashes": {"motion": source}}
        item = {"iteration": 11000, "source_sha256": source}
        for name, data in (("manifest", manifest), ("contracts", {}), ("envelopes", {})):
            p = directory / (name + ".json")
            p.write_text(json.dumps(data))
            item[name] = {"path": str(p.relative_to(tmp_path)),
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        groups[group] = item
    index = {"schema": "vita_motion_groups_v1", "iteration": 11000, "groups": groups}
    (tmp_path / "config/motion_groups.json").write_text(json.dumps(index))
    return tmp_path, index


def save(root, index):
    (root / "config/motion_groups.json").write_text(json.dumps(index))


def test_five_groups_choose_distinct_packages(package):
    root, _ = package
    results = [module.load_motion_group(root, g) for g in "ABCDE"]
    assert len({r["manifest"] for r in results}) == 5
    assert len({r["source_sha256"] for r in results}) == 5
    assert all(r["iteration"] == 11000 for r in results)


def test_tampered_contract_rejected(package):
    root, _ = package
    (root / "B/contracts.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.load_motion_group(root, "B")


def test_wrong_iteration_rejected(package):
    root, index = package
    index["groups"]["C"]["iteration"] = 12000
    save(root, index)
    with pytest.raises(ValueError, match="same checkpoint"):
        module.load_motion_group(root, "C")


def test_swapped_manifest_rejected(package):
    root, index = package
    index["groups"]["D"]["manifest"] = index["groups"]["B"]["manifest"]
    save(root, index)
    with pytest.raises(ValueError, match="experiment mismatch"):
        module.load_motion_group(root, "D")


def test_wrong_source_identity_rejected(package):
    root, index = package
    index["groups"]["E"]["source_sha256"] = "0" * 64
    save(root, index)
    with pytest.raises(ValueError, match="source identity"):
        module.load_motion_group(root, "E")


def test_external_path_rejected(package):
    root, index = package
    index["groups"]["A"]["contracts"]["path"] = "../outside.json"
    save(root, index)
    with pytest.raises(ValueError, match="escapes"):
        module.load_motion_group(root, "A")


def test_unknown_group_rejected(package):
    root, _ = package
    with pytest.raises(ValueError, match="must be"):
        module.load_motion_group(root, "F")


def test_explicit_p123_reference_comparison(package):
    root, index = package
    index.update(schema='vita_motion_groups_v2', comparison='latest_per_group')
    groups = {}
    for old, group, iteration in [('A', 'P1', 6000), ('B', 'P2', 8000), ('C', 'P3', 6000)]:
        item = index['groups'][old]
        item['iteration'] = iteration
        path = root / item['manifest']['path']
        manifest = json.loads(path.read_text())
        manifest['experiment'] = {'group': group, 'iteration': iteration}
        path.write_text(json.dumps(manifest))
        item['manifest']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[group] = item
    index['groups'] = groups
    save(root, index)
    assert [module.load_motion_group(root, g)['iteration'] for g in groups] == [6000, 8000, 6000]
    index['comparison'] = 'same_iteration'
    save(root, index)
    with pytest.raises(ValueError, match='same checkpoint'):
        module.load_motion_group(root, 'P1')


def test_latest_mode_cannot_relax_legacy_groups(package):
    root, index = package
    index.update(schema='vita_motion_groups_v2', comparison='latest_per_group')
    save(root, index)
    with pytest.raises(ValueError, match='only supported'):
        module.load_motion_group(root, 'A')
