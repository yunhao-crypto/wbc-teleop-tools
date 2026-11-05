#!/usr/bin/env python3
"""Build only the pinned SDK Body/EE codec for local source-integration tests.

No SDK installation, inference backend, robot connection or wheel publication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sdk = args.sdk_root.resolve()
    pinned = json.loads((root/'config/sdk021-wheel-guard.json').read_text())
    actual = subprocess.check_output(['git','-C',str(sdk),'rev-parse','HEAD'],text=True).strip()
    if actual != pinned['sdk_commit']:
        raise ValueError('SDK HEAD differs from pinned source integration revision')
    cpp = sdk/'demo_onnx/cpp'
    body, ee = cpp/'body_cmd_vendor', cpp/'end_effector_vendor'
    bm = json.loads((body/'provenance.json').read_text())
    em = json.loads((ee/'provenance.json').read_text())
    for directory, files in ((ee,em['headers_sha256']), (body,em['shared_body_vendor_include_sha256'])):
        for name, expected in files.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Changed generated header: '+name)
    tools = root/'build/sdk-test-tools'
    with zipfile.ZipFile(root/'deploy/actor_s100/wheelhouse/pybind11-2.13.6-py3-none-any.whl') as archive:
        archive.extractall(tools)
    defines = []
    for key, stem, actual_hash in (
        ('BODY_CMD','body_cmd',bm['body_cmd_bfbs_sha256']),
        ('BODY_STATE','body_state',bm['body_state_bfbs_sha256']),
        ('EE_STATE','end_effector_state',em['bfbs_sha256']['end_effector_state']),
        ('EE_TOUCH','end_effector_touch',em['bfbs_sha256']['end_effector_touch'])):
        if actual_hash != pinned['schema_sha256'][stem]:
            raise ValueError('Native schema differs: '+stem)
        defines.append('-D'+key+'_SCHEMA_SHA256="'+actual_hash+'"')
    output = sdk/'python/locomotion_aorta'/('_body_cmd_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    includes = [sysconfig.get_path('include'),tools/'pybind11/include',body,
                body/'message/schemas/topic/lowlevel',
                body/'declared-headers/generated/external/aorta~/message/schemas/system']
    subprocess.run(['c++','-O2','-shared','-std=c++17','-fPIC',
        *['-I'+str(p) for p in includes], *defines,
        *[str(cpp/name) for name in ('body_cmd_encoder.cpp','body_state_decoder.cpp','end_effector_decoder.cpp')],
        '-o',str(output)],check=True)
    print(output)

if __name__ == '__main__': main()
