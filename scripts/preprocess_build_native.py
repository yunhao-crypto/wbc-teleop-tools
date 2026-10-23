#!/usr/bin/env python3
"""Build the small portable CPU preprocessing library and bind its identity.

Compile on the target architecture; this script does not contact any device.
No CUDA/Torch/Warp headers or libraries are used. Architecture-specific numerical
validation against the saved original GPU oracle remains mandatory afterward.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--cxx',default='g++')
    ap.add_argument('--arch-flags',nargs='*');ap.add_argument('--force',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parents[1]
    src=root/'src/deployment/vita_s100_aorta_30dof/onboard_preprocessing_native.cpp'
    wrapper=src.with_name('onboard_preprocessing.py');out=Path(a.output).resolve()
    if out.exists() and not a.force:raise FileExistsError('Use a new output path or explicit --force for isolated candidate')
    arch=platform.machine()
    flags=a.arch_flags if a.arch_flags is not None else (['-march=armv8.2-a+simd'] if arch=='aarch64' else ['-march=native'])
    command=[a.cxx,'-O3','-std=c++17','-shared','-fPIC','-ffp-contract=fast',*flags,str(src),'-o',str(out)]
    out.parent.mkdir(parents=True,exist_ok=True);subprocess.run(command,check=True)
    meta={'schema':'vita_cpu_preprocessing_native_v1','architecture':arch,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],
          'command':command,'source_sha256':sha(src),'wrapper_sha256_at_build':sha(wrapper),'library_sha256':sha(out),
          'exports':['pp_pico','pp_features','pp_gather','pp_build'],
          'arithmetic':'float32 scalar expressions; compiler FMA contraction enabled; no fast-math/reassociation flag',
          'runtime_dependencies':'C++ standard library and libm only; Python wrapper uses NumPy/ctypes',
          'accuracy_status':'must validate this exact binary on this architecture against original GPU/action oracle',
          'license_scope':'Original project preprocessing expressions ported to portable C++; no CUDA libdevice or proprietary SDK implementation linked or copied. Skeleton asset remains separately licensed.'}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
