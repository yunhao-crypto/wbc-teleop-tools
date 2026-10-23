"""Compile an already validated backend without altering or reconverting it."""
import argparse
import json
from pathlib import Path
import resource

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--jobs', type=int, default=6)
p.add_argument('--vm-limit-gib', type=float, default=16)
p.add_argument('--disable-mlir-threading', action='store_true',
               help='Disable nested MLIR/LLVM worker pools on this model context')
a = p.parse_args()
assert a.input.is_file() and not a.output.exists()
assert a.output.suffix == '.hbo' and a.jobs > 0 and a.vm_limit_gib > 0
memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
available = int(memory['MemAvailable'].split()[0]) * 1024
assert available >= 12 * 1024**3, f'Insufficient available memory: {available}'
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
ceiling = int(a.vm_limit_gib * 1024**3)
assert hard == resource.RLIM_INFINITY or ceiling <= hard
resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
from hbdk4.compiler import load, compile
print(json.dumps(dict(input=str(a.input), output=str(a.output), jobs=a.jobs,
                      vm_limit_gib=a.vm_limit_gib, available_bytes=available,
                      disable_mlir_threading=a.disable_mlir_threading)), flush=True)
model = load(str(a.input))
if a.disable_mlir_threading:
    model.module.context.enable_multithreading(False)
compile(model, str(a.output), march='nash-e', opt=2, jobs=a.jobs,
        progress_bar=True, balance=100)
print('HBO_READY', a.output, flush=True)
