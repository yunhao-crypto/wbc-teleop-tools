import json,collections,datetime
from pathlib import Path
O=Path(__file__).resolve().parents[1]/'outputs/incident_154621'
control=[];cmd=[];phases=[];last=None
for line in (O/'flatbuffer_decoded.jsonl').open():
 r=json.loads(line);t=r['topic'];d=r['data']
 if '/control/' in t:
  control.append(r)
  if t.endswith('status'):
   key=(d['currentAction'],d['actionPhase'],d['timeoutHold'],d['lastRejectCode'])
   if key!=last:phases.append(r);last=key
 elif t.endswith('external_command'):
  cmd.append({k:v for k,v in r.items() if k!='data'}|{'data':{k:v for k,v in d.items() if k!='motorCmd'},'motorCmdCount':len(d['motorCmd']), 'kp': [x['kp'] for x in d['motorCmd']], 'kd':[x['kd'] for x in d['motorCmd']]})
(O/'control_records.json').write_text(json.dumps(control,indent=2))
(O/'control_phase_changes.json').write_text(json.dumps(phases,indent=2))
(O/'external_command_summary.json').write_text(json.dumps(cmd,indent=2))
print('phase_changes',json.dumps(phases,indent=2)); print('lastcmd',json.dumps(cmd[-1],indent=2))
