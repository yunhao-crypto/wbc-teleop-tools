"""Load a real export and exercise inference-contract negative controls.

No policy updates or edits to original assets/models. Deliberate corruptions
are in-memory parsed metadata/config variants, reported separately.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wrong-asset", type=Path, required=True)
    parser.add_argument("--rejected-pt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from holomotion.src.evaluation.eval_mujoco_sim2sim import MujocoEvaluator

    cfg = OmegaConf.load(args.model.parent.parent / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.ckpt_onnx_path = str(args.model.resolve())
    cfg.robot_xml_path = cfg.robot.asset.assetFileName
    cfg.use_gpu = False
    ev = MujocoEvaluator.__new__(MujocoEvaluator)
    ev.config, ev.max_context_len = cfg, 0
    ev.load_policy()
    original = ev._read_onnx_metadata()
    ev._apply_onnx_metadata()
    report = {
        "model": str(args.model.resolve()),
        "positive": ev._policy_io_receipt,
        "negative_scope": "real ONNX loaded; config/parsed-metadata corruptions in memory only",
        "negatives": {},
    }
    digest = hashlib.sha256()
    with args.model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report["model_sha256"] = digest.hexdigest()

    def reject(label, operation):
        try:
            operation()
        except (ValueError, KeyError) as exc:
            report["negatives"][label] = {"rejected": True, "error": str(exc)}
        else:
            raise AssertionError(f"Invalid input accepted: {label}")

    altered = copy.deepcopy(original)
    altered["action_joint_names"][0:2] = reversed(
        altered["action_joint_names"][0:2]
    )
    ev._read_onnx_metadata = lambda: altered
    reject("wrong_policy_action_order", ev._apply_onnx_metadata)
    ev._read_onnx_metadata = lambda: copy.deepcopy(original)
    declared = cfg.get("velocity_policy_contract") or cfg.get(
        "gmr_free_contract"
    )
    expected_dim = declared.actor_observation_dim
    declared.actor_observation_dim = expected_dim + 1
    reject("wrong_observation_dimension", ev._apply_onnx_metadata)
    declared.actor_observation_dim = expected_dim
    saved_asset = cfg.robot.asset.urdf_file
    cfg.robot.asset.urdf_file = str(args.wrong_asset.resolve())
    reject("wrong_asset", ev._apply_onnx_metadata)
    cfg.robot.asset.urdf_file = saved_asset
    assert args.rejected_pt.is_file(), args.rejected_pt
    cfg.ckpt_onnx_path = str(args.rejected_pt.resolve())
    reject("pt_is_not_an_exported_onnx_policy", ev.load_policy)
    cfg.ckpt_onnx_path = str(args.model.resolve())
    ev._apply_onnx_metadata()
    report["positive_after_negative_controls"] = ev._policy_io_receipt
    report["status"] = "passed"
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
