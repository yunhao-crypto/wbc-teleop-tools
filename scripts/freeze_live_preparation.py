#!/usr/bin/env python3
"""Record live-launcher preparation evidence, separately from no-action acceptance."""
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
reports=['operator_check.json','native_runtime_smoke.json','live_sdk_readonly.json',
         'state_live_io_arm.json','live_state_readonly_probe.json','vita_wifi_mock_tests.json',
         'shadow_archive_audit.json','live_final_tests.log','operator_service_tests.log',
         'vita_wifi_status.txt','vita_wifi_dry_run.json']
for name in ['operator_check.json','native_runtime_smoke.json','live_sdk_readonly.json',
             'state_live_io_arm.json','live_state_readonly_probe.json','vita_wifi_mock_tests.json']:
    assert json.loads((root/'outputs'/name).read_text())['passed'],name
archive=json.loads((root/'outputs/shadow_archive_audit.json').read_text())
assert archive['verified']==453 and archive['bad']==[]
log=(root/'outputs/live_final_tests.log').read_text()
assert re.search(r'\b78 passed\b',log)
files={}
def record(path):
    raw=path.read_bytes();files[str(path.relative_to(root))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
for tree in ['src','scripts','tests']:
    for path in sorted((root/tree).rglob('*')):
        if path.is_file() and not {'__pycache__','.pytest_cache','.git'}.intersection(path.parts) and path.suffix in ('.py','.sh','.cpp','.h','.json','.yaml','.yml'):
            record(path)
for path in [root/'LIVE_TEST.md',root/'AGENTS.md',*[root/'outputs'/x for x in reports]]:record(path)
result={'schema':'vita_live_preparation_v1','date':'2026-09-29','preparation_passed':True,
    'scope':'Prepared operator-invoked suspended action launcher, manual WiFi switch, authenticated laptop Xbox WiFi bridge',
    'agent_actions_sent':0,'agent_leases_requested':0,'agent_networks_switched':0,
    'operator_entry':'scripts/start_onboard_vita.sh suspended-test',
    'no_action_entries':['scripts/start_onboard_vita.sh check','scripts/start_onboard_vita.sh input-check'],
    'network_entry':'scripts/vita_wifi.sh connect-asus','laptop_entry':'scripts/send_xbox_wifi.sh <Vita_WiFi_IP>',
    'validation':{'offline_regressions':78,'network_offline_regressions':11,
                  'actual_arm_runtime_check':True,'arm_recorded_state_comparisons':8958,
                  'actual_readonly_state_frames':251,'state_hz':49.99926301086322,
                  'wifi_status_and_dry_run_on_robot':True},
    'not_run':['Real EXTERNAL handshake/lease and joint publication','Mechanical response',
               'Physical Xbox/Pico input check on ASUS-5G','WiFi connection switch'],
    'numerical_baseline':'outputs/ACCEPTANCE_MANIFEST.json is prior shadow evidence; its old sources are preserved in reference/no_action_accepted_sources.tar',
    'old_source_archive':archive,'private_files':{'config/xbox_wifi.key':'0600,32 bytes',
        'config/wifi/vita_wifi.private.json':'0600','config/wifi/asus.passwd':'0600; credentials omitted'},
    'files':files}
(root/'outputs/LIVE_PREPARATION_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'preparation_passed':True,'files':len(files),'actuation_tested':False}))
