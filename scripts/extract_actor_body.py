#!/usr/bin/env python3
"""Offline MCAP BodyState extractor; no robot/network/control access."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Permit running directly from a source checkout.
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deployment.vita_s100_aorta_30dof.actor_body_trace import DEFAULT_TOPIC, extract_mcap, _joint_names

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mcap", nargs="+", type=Path)
    p.add_argument("--schema-sha256", required=True, help="Required SHA-256 pin for selected topic schema.data only")
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--joint-names-json", type=Path)
    p.add_argument("--actor-log", type=Path, help="JSON-lines log containing body_order")
    p.add_argument("--sdk-python-path", help="SDK site-packages/root to prepend to sys.path")
    p.add_argument("--clock-manifest", type=Path, help="JSON declarations for timestamp domains")
    p.add_argument("--unknown-default", choices=("mask","nan"), default="mask")
    p.add_argument("--validate-crcs", action=argparse.BooleanOptionalAction, default=True,
                   help="Request reader-supported CRC checking (default on; unsupported readers fail)")
    a=p.parse_args(argv)
    names=_joint_names(a.joint_names_json,a.actor_log)
    manifest=json.loads(a.clock_manifest.read_text()) if a.clock_manifest else None
    receipt=extract_mcap(a.mcap,a.output_dir,a.schema_sha256,a.topic,names,a.sdk_python_path,manifest,a.unknown_default,
                         validate_crcs=a.validate_crcs,
                         roster_sources=[x for x in (a.joint_names_json,a.actor_log,a.clock_manifest) if x])
    print(json.dumps({k: receipt[k] for k in ("body_rows", "imu_rows", "imu_location_status_counts", "protocol_versions", "crc", "extraction_valid", "errors", "imu_contradiction_rows")}, sort_keys=True))
    return 0 if receipt["extraction_valid"] else 3
if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print(f"extract_actor_body: {e}",file=sys.stderr); raise SystemExit(2)
