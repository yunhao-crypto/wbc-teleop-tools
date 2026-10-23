"""No inference or robot I/O: real C header ABI and expert-copy boundary tests."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
ROOT = BACKEND.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from deployment.vita_s100_aorta_30dof.onboard_backend.native import _Config, _State, _Timing, _observation, _artifacts


def main():
    ucp = Path(os.environ["UCP_INCLUDE"])
    with tempfile.TemporaryDirectory(prefix="backend-test-", dir=HERE) as directory:
        exe = Path(directory) / "layout_probe"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I" + str(BACKEND), "-I" + str(ucp), str(HERE / "layout_probe.cpp"), "-o", str(exe)], check=True)
        report = json.loads(subprocess.check_output([str(exe)], text=True))
    assert report["sizeof_config"] == ctypes.sizeof(_Config)
    assert report["sizeof_state"] == ctypes.sizeof(_State)
    assert report["sizeof_timing"] == ctypes.sizeof(_Timing)
    assert report["config_timeout_offset"] == _Config.timeout_ms.offset
    assert report["timing_total_offset"] == _Timing.total_ms.offset
    rejected = 0
    for value in [np.zeros(990, np.float64), np.zeros(989, np.float32), np.full(990, np.nan, np.float32)]:
        try:
            _observation(value, 990)
        except ValueError:
            rejected += 1
    assert rejected == 3
    assert _observation(np.zeros((1, 990), np.float32), 990).shape == (990,)
    assert _observation(np.zeros(6242, np.float32)[::2], 3121).flags.c_contiguous
    artifacts = _artifacts(ROOT / "outputs/model_runtime_manifest.json")
    assert len(artifacts) == 8 and all(Path(item["path"]).is_file() for item in artifacts.values())
    report.update(ctypes_abi_matches=True, invalid_observations_rejected=rejected, manifest_artifacts=len(artifacts),
        source="host C++ copy/layout and ctypes contracts only; no target inference or precision claim")
    (ROOT / "outputs/backend_host_tests.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
