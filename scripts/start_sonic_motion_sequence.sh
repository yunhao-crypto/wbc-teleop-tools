#!/usr/bin/env bash
# Validated 01-09 + 11 sequence, each source clip capped at 10 seconds. Start only from the operator's terminal.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-live}
case "$MODE" in
  --help|-h) echo 'Usage: start_sonic_motion_sequence.sh [check|live]'; exit 0 ;;
  check|live) if (( $# )); then shift; fi ;;
  *) echo 'Expected check or live' >&2; exit 2 ;;
esac
if (( $# )); then echo 'This pinned trial accepts no overrides; use start_actor_motion.sh for other trials.' >&2; exit 2; fi
PYTHON=${VITA_ACTOR_PYTHON:-$ROOT/venv/bin/python}
"$PYTHON" - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'reference/sequences/sonic_max10'
manifest=json.loads((folder/'sequence_manifest.json').read_text())
config=json.loads((root/'config/actor_motion_sequence_sonic.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(root/'config/actor_pico.json')!=manifest['policy_config_sha256']:
    raise SystemExit('Sonic configuration differs from the validated sequence; revalidate first.')
if sha(folder/'sequence.npz')!=manifest['sha256'] or config['sha256']!=manifest['sha256'] or (root/config['path']).resolve()!=(folder/'sequence.npz').resolve():
    raise SystemExit('Sequence reference/configuration mismatch.')
validation=json.loads((folder/'continuous_sim_validation.json').read_text())
if not validation['completed'] or validation['adapted_sha256']!=manifest['sha256']:
    raise SystemExit('Missing successful continuous simulation for this sequence.')
print('SONIC_SEQUENCE: 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 11[0:10s]')
print('SONIC_SEQUENCE: reference duration %.2fs; each source clip <=10s; skips 10; full observations recorded in live mode'%manifest['duration_s'])
PY
if [[ "$MODE" == check ]]; then
  exec bash "$ROOT/scripts/start_actor_motion.sh" sonic check \
    --motion-config "$ROOT/config/actor_motion_sequence_sonic.json" --steps 100
fi
exec bash "$ROOT/scripts/start_actor_motion.sh" sonic live \
  --motion-config "$ROOT/config/actor_motion_sequence_sonic.json" \
  --record-observations --seconds 0 \
  --log "$ROOT/outputs/sonic_sequence_$(date +%Y%m%d_%H%M%S_%N).jsonl"
