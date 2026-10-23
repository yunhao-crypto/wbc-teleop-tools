import json,sys,collections
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_ACTION_TO_COMMAND_SLOTS
from deployment.vita_s100_aorta_30dof.robot_adapter import VITA_CONTROLLER_JOINT_NAMES
result=[]
for p in sorted((root/'logs/live/sessions').glob('*/control.jsonl'))[-3:]:
 maxima=np.zeros(30);counts=np.zeros(30,int);examples={};clipmode=collections.Counter();policy=collections.Counter();transitions=[];prev=None;interesting=[]
 for line in p.open():
  r=json.loads(line)
  if r.get('record_type')!='control':continue
  mode=r['mode'];policy[(mode,r['policy_mode'],r.get('stand_phase'))]+=1
  if mode!=prev:
   transitions.append({k:r[k] for k in ('mode','command_id','timing_ms','publish_period_s','controller')});prev=mode
  if r['conditioning']['position_clip_max_rad']>0:clipmode[mode]+=1
  a=np.asarray(r['target_policy']);b=np.asarray(r['target_sdk'])[list(VITA_ACTION_TO_COMMAND_SLOTS)];delta=np.abs(a-b);counts+=delta>1e-6
  for i in np.where(delta>maxima)[0]:
   maxima[i]=delta[i];examples[int(i)]=dict(command_id=r['command_id'],mode=mode,requested=float(a[i]),conditioned=float(b[i]))
  if r['timing_ms']['sdk_publish_ms']>20 or r['cycle_skipped_ticks']>0:interesting.append({k:r[k] for k in ('command_id','mode','cycle_skipped_ticks','publish_period_s','timing_ms')})
 result.append(dict(session=p.parent.name,policy_counts={str(k):v for k,v in policy.items()},clipped_frames_by_mode=dict(clipmode),joints=[dict(name=VITA_CONTROLLER_JOINT_NAMES[i],count=int(counts[i]),max_delta_rad=float(maxima[i]),**examples.get(i,{})) for i in range(30) if maxima[i]>1e-6],transitions=transitions,long_publish_and_skip=interesting))
(root/'outputs/latest3_live_clipping.json').write_text(json.dumps(result,indent=2)+'\n')
