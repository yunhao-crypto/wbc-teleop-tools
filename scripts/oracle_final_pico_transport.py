"""Match captured SDK Pico samples to Codex-host sender by original timestamp."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture',type=Path,required=True);p.add_argument('--sent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sent={r['timestamp_ns']*1e-9:r for r in map(json.loads,a.sent.read_text().splitlines())}
rows=list(map(json.loads,a.capture.read_text().splitlines()))
errors=[];frames=[];indices=[]
for row in rows:
    pico=row['pico']
    if pico is None:continue
    body,timestamp,local_index=pico[:3];indices.append(local_index)
    sample=sent.get(timestamp)
    if sample is None:errors.append({'tick':row['index'],'error':'No sender sample with exact timestamp'});continue
    frames.append(sample['frame_index'])
    if not np.array_equal(np.asarray(body,np.float32).view(np.uint32),np.asarray(sample['body'],np.float32).view(np.uint32)):
        errors.append({'tick':row['index'],'error':'Body differs'})
report={'capture_ticks':len(rows),'captured_pico_frames':len(indices),'unique_sender_frames':len(set(frames)),
        'sent_frames':len(sent),'first_sender_frame':min(frames),'last_sender_frame':max(frames),
        'first_local_index':min(indices),'last_local_index':max(indices),'errors':errors,'passed':not errors,
        'comparison':'Match by exact sample timestamp seconds=sender timestamp_ns*1e-9; all24x7float32 elements bitexact. LocalPicoSmplSource assigns its own contiguous index to consumed samples; it is not the sender protocol frameindex.',
        'limitations':'Sampling can skip sender frames; counts are not a packet-loss or absolute latency measurement. Headset samples are synthetic on Codexhost, transported through official native service/SDK.'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('comparison','limitations','errors')}))
raise SystemExit(0 if report['passed'] else 2)
