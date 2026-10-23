#!/usr/bin/env python3
"""Generate sdkmain pins from a fixed SDK Git object and its reviewed wheel.

No package import, network access or compilation. Source-only pins are refused;
the final wheel must match all original guards and the new native/pull modules.
"""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ORIGINAL = (
    '__init__.py', '_body_codec.py', '_gripper_codec.py', '_gripper_lifecycle.py',
    '_hand_codec.py', '_motor_codec.py', 'body_types.py', 'brainco_capture.py',
    'controller.py', 'gripper.py', 'gripper_types.py', 'hand.py', 'hand_types.py',
    'transport.py', 'types.py',
)
RUNTIME = ('pull.py', 'state_arrays.py', '_native_body_codec.py')


def generate(args):
    if not re.fullmatch('[0-9a-f]{40}', args.sdk_commit):
        raise ValueError('A complete fixed SDK commit is required')
    if not args.sdk_wheel:
        raise ValueError('The final reviewed wheel is required; source-only guard generation is forbidden')

    def blob(name):
        return subprocess.check_output(['git', '-C', str(args.sdk_git), 'show',
                                        args.sdk_commit + ':' + name])

    compat = json.loads(blob('compatibility.json'))
    names = ORIGINAL + RUNTIME
    sources = {n: blob('python/locomotion_aorta/' + n) for n in names}
    wheel_record = None
    native_sha256 = {}
    if args.sdk_wheel:
        actual = hashlib.sha256(args.sdk_wheel.read_bytes()).hexdigest()
        if actual != args.wheel_sha256:
            raise ValueError('Reviewed wheel SHA mismatch')
        with zipfile.ZipFile(args.sdk_wheel) as wheel:
            metadata_names = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError('Ambiguous wheel metadata')
            metadata = Parser().parsestr(wheel.read(metadata_names[0]).decode())
            if metadata['Version'] != '0.2.1' or metadata['Name'].replace('_', '-') != 'locomotion-aorta':
                raise ValueError('Expected the reviewed SDK 0.2.1 wheel')
            for name, source in sources.items():
                if wheel.read('locomotion_aorta/' + name) != source:
                    raise ValueError('Wheel differs from reviewed Git source: ' + name)
            for name in wheel.namelist():
                path = PurePosixPath(name)
                if name.startswith('locomotion_aorta/') and '.so' in path.name:
                    if path.is_absolute() or '..' in path.parts or name.endswith('/'):
                        raise ValueError('Unsafe native wheel member: ' + name)
                    relative = str(path.relative_to('locomotion_aorta'))
                    native_sha256[relative] = hashlib.sha256(wheel.read(name)).hexdigest()
            if not any(Path(n).name.startswith('_body_cmd_native.') for n in native_sha256):
                raise ValueError('Reviewed wheel must contain the native BodyCmd/BodyState extension')
            required = {
                'aorta-sdk==' + compat['aorta_version'],
                'aorta-msgs==' + compat['aorta_version'],
                'vbot-humanoid-msgs==' + compat['humanoid_msgs_version'],
                'flatbuffers==' + compat['flatbuffers_version'],
            }
            if not required <= set(metadata.get_all('Requires-Dist', [])):
                raise ValueError('Wheel dependency pins differ from the reviewed compatibility manifest')
        wheel_record = {'filename': args.sdk_wheel.name, 'sha256': actual,
                        'bytes': args.sdk_wheel.stat().st_size}
    result = {
        'status': 'SEALED_FROM_REVIEWED_WHEEL',
        'sdk_commit': args.sdk_commit, 'sdk_version': '0.2.1', 'wheel': wheel_record,
        'source_sha256': {n: hashlib.sha256(sources[n]).hexdigest() for n in ORIGINAL},
        'runtime_source_sha256': {n: hashlib.sha256(sources[n]).hexdigest() for n in RUNTIME},
        'native_sha256': native_sha256,
        'schema_sha256': compat['schema_sha256'],
        'dependencies': {n: compat[n] for n in ('aorta_version', 'humanoid_msgs_version', 'flatbuffers_version')},
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-git', type=Path, required=True)
    parser.add_argument('--sdk-commit', required=True)
    parser.add_argument('--sdk-wheel', type=Path, required=True)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    result = generate(parser.parse_args())
    print(result['status'], result['sdk_commit'])
