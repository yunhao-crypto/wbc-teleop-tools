#!/usr/bin/env bash
# Same pinned max-10s sequence as the Sonic20000 trial; operator starts motion.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
"$ROOT/venv/bin/python" - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
model=json.loads((root/'config/actor_pico.json').read_text())['artifacts']['model']
expected='2bc0ca6f0bddef1ba040a19a771181da68f5b7d2138d1e87913072def089135d'
if model['sha256']!=expected or hashlib.sha256((root/model['path']).read_bytes()).hexdigest()!=expected:
    raise SystemExit('Sonic9700 baseline model changed; review before testing.')
print('SONIC_BASELINE: Sonic9700 | same max-10s sequence | full observations in live mode')
PY
exec bash "$ROOT/scripts/start_sonic_motion_sequence.sh" "$@"
