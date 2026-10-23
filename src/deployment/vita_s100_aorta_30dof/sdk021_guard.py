"""Fail-closed SDK source/BFBS pins generated offline from reviewed wheels."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

ORIGINAL_VENDOR_FILES = (
    '__init__.py', '_body_codec.py', '_gripper_codec.py', '_gripper_lifecycle.py',
    '_hand_codec.py', '_motor_codec.py', 'body_types.py', 'brainco_capture.py',
    'controller.py', 'gripper.py', 'gripper_types.py', 'hand.py', 'hand_types.py',
    'transport.py', 'types.py',
)
RUNTIME_FILES = ('pull.py', 'state_arrays.py', '_native_body_codec.py')
GUARD_PATH = Path(__file__).resolve().parents[3] / 'config/sdk021-wheel-guard.json'


def load_guard(path=GUARD_PATH):
    """The deployment must ship config/ with src/; never search another root."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError('SDK wheel guard is missing or unreadable at ' + str(path)
            + '; deploy config/sdk021-wheel-guard.json together with src/; '
              'a src-only deployment is unsupported') from exc


GUARD = load_guard()
VENDOR_SOURCE_SHA256 = GUARD['source_sha256']
BODY_STATE_SCHEMA_HASH = GUARD['schema_sha256']['body_state']
if set(VENDOR_SOURCE_SHA256) != set(ORIGINAL_VENDOR_FILES):
    raise RuntimeError('All 15 reviewed SDK source guards are required')
if (GUARD['status'] == 'SEALED_FROM_REVIEWED_WHEEL' and
        set(GUARD['runtime_source_sha256']) != set(RUNTIME_FILES)):
    raise RuntimeError('Native/pull SDK source guards are required')


def verify_vendor_sources():
    """Verify 15+3 sources and every SDK native library from the sealed wheel.

    The generator/deployer verifies the whole wheel SHA. Installed execution
    checks its actual payload, including bundled native dependencies, rather
    than assuming that recording the archive SHA proves an installed library.
    """
    require_wheel_guard()
    spec = importlib.util.find_spec('locomotion_aorta')
    if spec is None or spec.origin is None or not hasattr(spec.loader, 'get_data'):
        raise RuntimeError('Hash-verifiable locomotion_aorta sources required')
    directory = Path(spec.origin).parent
    result = {}
    for pins in (VENDOR_SOURCE_SHA256, GUARD['runtime_source_sha256']):
        for name, expected in pins.items():
            data = spec.loader.get_data(str(directory / name))
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise RuntimeError('Unreviewed locomotion_aorta source: ' + name)
            if name in VENDOR_SOURCE_SHA256:
                result[name] = actual
    native_pins = GUARD['native_sha256']
    for name, expected in native_pins.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '.so' not in path.name:
            raise RuntimeError('Invalid native SDK guard path: ' + name)
        try:
            payload = spec.loader.get_data(str(directory / name))
        except OSError as exc:
            raise RuntimeError('Missing or unreadable reviewed SDK native library: ' + name) from exc
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError('Unreviewed locomotion_aorta native library: ' + name)
    extension = importlib.util.find_spec('locomotion_aorta._body_cmd_native')
    if extension is None or extension.origin is None:
        raise RuntimeError('Hash-verifiable native BodyCmd/BodyState extension required')
    try:
        relative = str(Path(extension.origin).resolve().relative_to(directory.resolve()))
    except ValueError as exc:
        raise RuntimeError('Native BodyCmd/BodyState extension is outside the reviewed SDK') from exc
    if relative not in native_pins:
        raise RuntimeError('Unreviewed native BodyCmd/BodyState import path: ' + relative)
    return result


def require_wheel_guard():
    """A source-only development pin cannot enable this customer copy."""
    if GUARD['status'] != 'SEALED_FROM_REVIEWED_WHEEL':
        raise RuntimeError('Final reviewed SDK 0.2.1 wheel guard is pending; no handles may be opened')
    if not GUARD.get('native_sha256') or not any(
            Path(n).name.startswith('_body_cmd_native.') for n in GUARD['native_sha256']):
        raise RuntimeError('Final reviewed SDK native library guard is pending; regenerate from the final wheel')
