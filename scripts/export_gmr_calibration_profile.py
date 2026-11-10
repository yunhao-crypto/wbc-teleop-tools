"""Convert a neutral-stance GMR task fit into a hash-pinned calibration profile.

The input is the fit's patch JSON (schema ``vita_gmr_neutral_calibration_v1``):
per-task position/rotation offsets fitted so the averaged, fully processed
human neutral stance maps exactly onto the policy home pose. The profile keeps
only what the actor validates at startup (base GMR config hash, operator
height, offset_to_ground) plus provenance, and refuses to overwrite a file.

Prints the SHA256 to place in ``retargeting.calibration_profile``. Converting a
fit is not a validation of it; check held-out stances and closed-loop
simulation before enabling a profile on the robot.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT/'third_party/motion_stream_bridge/motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('patch', type=Path, help='neutral_calibration_patch.json from the stance fit')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operator', default=None, help='operator label recorded in provenance')
    args = parser.parse_args()
    patch = json.loads(args.patch.read_text())
    if patch.get('schema') != 'vita_gmr_neutral_calibration_v1':
        raise SystemExit('Expected a vita_gmr_neutral_calibration_v1 patch')
    base_sha = sha256(BASE_CONFIG)
    if patch.get('source_config_sha256') != base_sha:
        raise SystemExit('Patch was fitted against another GMR configuration than the pinned one')
    tasks = {link: dict(position_offset=[float(v) for v in update['position_offset']],
                        rotation_offset_wxyz=[float(v) for v in update['rotation_offset_wxyz']])
             for link, update in patch['updates'].items()}
    provenance = {key: patch[key] for key in ('status', 'capture', 'fit_phase', 'fit_frames', 'scope', 'note',
                                              'heldout_test_performed', 'heldout_test_passed',
                                              'heldout_conclusion', 'training_home_max_error_rad')
                  if key in patch}
    provenance.update(source_patch=args.patch.name, source_patch_sha256=sha256(args.patch))
    if args.operator is not None:
        provenance['operator'] = args.operator
    profile = dict(schema='vita_gmr_task_calibration_v1', robot='vita02a', base_config_sha256=base_sha,
                   human_height_m=float(patch['human_height_m']), offset_to_ground=True, tasks=tasks,
                   provenance=provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(profile, indent=2)+'\n')
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output),
                          human_height_m=profile['human_height_m'], tasks=sorted(tasks)), indent=2))


if __name__ == '__main__':
    main()
