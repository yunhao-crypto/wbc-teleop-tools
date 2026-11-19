"""Explicit synthetic-feedback shadow math loader for a non-ARM host.

Load the pinned SDK's actual Python math/DTO files without executing its native
transport package initializer. Never use this namespace for robot feedback or
actuation. No alternative math or vendor source edits are introduced.
"""
import hashlib
import json
from pathlib import Path
import sys
import types


def prepare(root, route):
    root = Path(root).resolve()
    filename = 'actor_smpl.json' if route == 'pico_smpl' else 'actor_threepoint.json'
    config = json.loads((root/'config'/filename).read_text())
    sdk = root/'venv/lib/python3.10/site-packages/locomotion_aorta'
    for name, expected in config['sdk_sources'].items():
        h = hashlib.sha256((sdk/name).read_bytes()).hexdigest()
        if h != expected: raise ValueError('Unreviewed shadow SDK math: '+name)
    package = sys.modules.get('locomotion_aorta')
    if package is None:
        package = types.ModuleType('locomotion_aorta')
        package.__path__ = [str(sdk)]
        package.__doc__ = 'Pinned Python math-only namespace for explicit no-action synthetic-feedback validation'
        sys.modules['locomotion_aorta'] = package
    elif not any(Path(p).resolve() == sdk for p in getattr(package, '__path__', [])):
        raise RuntimeError('A different locomotion_aorta package is already loaded; do not mix SDKs')
    # Expose only the pinned inert DTOs; the ARM transport initializer is not
    # executed on a host. Never use this namespace to create robot handles.
    site=str(sdk.parent)
    if site not in sys.path:sys.path.append(site)
    from locomotion_aorta import body_types
    for name in ('BodyCmd','BodyJointCmd','JointError'):
        if not hasattr(package,name):setattr(package,name,getattr(body_types,name))
