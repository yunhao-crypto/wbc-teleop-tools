#!/usr/bin/env python3
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def stats(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 if not len(a):return None
 return dict(n=len(a),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max()),over20=int(np.sum(a>20)),over40=int(np.sum(a>40)))
reports=[]
for p in sorted((ROOT/'logs/live/sessions').glob('*/control.jsonl'))[-3:]:
 size=p.stat().st_size;digest=hashlib.sha256();count=collections.Counter();groups=collections.defaultdict(lambda:collections.defaultdict(list));events=[];segments=[];controls={};slow=[];invalid=[];clip=collections.Counter();pico=collections.Counter();last=None;rej=0;drops=0;timeouts=0;beg=None;end=None;lastrecord=None
 with p.open('rb') as f:
  for lineno,line in enumerate(f,1):
   digest.update(line)
   try:r=json.loads(line)
   except Exception as exc:invalid.append([lineno,str(exc)]);continue
   typ=r.get('record_type','unknown');count[typ]+=1
   t=r.get('monotonic_ns');beg=t if beg is None else beg;end=t if t is not None else end
   lastrecord={k:v for k,v in r.items() if k in ('record_type','event','reason','mode','command_id','monotonic_ns')}
   if typ=='event':
    events.append({k:v for k,v in r.items() if k not in ('runtime_warmup','policies','target_sdk','kp_sdk','kd_sdk','initial_pose','hold_q','target','positions') and not isinstance(v,list)})
   if typ=='control':
    mode=r['mode'];cid=r['command_id'];tim=r.get('timing_ms',{});controls[cid]=(mode,tim,r.get('cycle_started_monotonic_ns'),r.get('published_monotonic_ns'))
    if not segments or segments[-1]['mode']!=mode:
     segments.append(dict(mode=mode,first_id=cid,last_id=cid,first_ns=r['published_monotonic_ns'],last_ns=r['published_monotonic_ns'],frames=1))
    else:segments[-1].update(last_id=cid,last_ns=r['published_monotonic_ns'],frames=segments[-1]['frames']+1)
    for g in ('all',mode):
     for k,v in tim.items():
      if isinstance(v,(float,int)):groups[g][k].append(v)
     groups[g]['publish_period_ms'].append(r['publish_period_s']*1000)
     groups[g]['compute_to_publish_ms'].append((r['published_monotonic_ns']-r['cycle_started_monotonic_ns'])/1e6)
     groups[g]['lowstate_age_ms'].append(r['lowstate_age_ms'])
     groups[g]['skipped_ticks'].append(r.get('cycle_skipped_ticks',0))
    if last is not None:
     groups['all']['actual_publish_interval_ms'].append((r['published_monotonic_ns']-last)/1e6)
    last=r['published_monotonic_ns']
    pico[(mode,str(r['pico_ready']))]+=1
    rej=max(rej,r.get('sdk_reject_count',0));s=r.get('sdk_status_before_publish',{});drops=max(drops,s.get('ingress_drop_count',0));timeouts+=bool(s.get('timeout_hold',False))
    for k,v in r.get('conditioning',{}).items():clip[k]=max(clip[k],v)
   if typ=='cycle_timing':
    cid=r['command_id'];mode,tim,start,pub=controls[cid]
    for g in ('all',mode):
     for k in ('cycle_through_record_ms','record_write_ms'):groups[g][k].append(r[k])
    slow.append(dict(command_id=cid,mode=mode,cycle_ms=r['cycle_through_record_ms'],write_ms=r['record_write_ms'],timing=tim))
 for s in segments:s['seconds_between_first_last']=(s['last_ns']-s['first_ns'])/1e9;s['measured_hz']=(s['frames']-1)/s['seconds_between_first_last'] if s['frames']>1 else None
 summary={g:{k:stats(v) for k,v in fields.items()} for g,fields in groups.items()}
 skipped={g:sum(fields['skipped_ticks']) for g,fields in groups.items()}
 reports.append(dict(session=p.parent.name,bytes=size,unchanged_size=size==p.stat().st_size,sha256=digest.hexdigest(),records=dict(count),malformed=invalid,duration_recorded_s=(end-beg)/1e9 if beg and end else None,events=events,segments=segments,metrics=summary,total_skipped_ticks=skipped,max_sdk_reject_count=rej,max_ingress_drop_count=drops,timeout_hold_frames=timeouts,conditioning_maxima=dict(clip),pico_ready_counts={str(k):v for k,v in pico.items()},slowest=sorted(slow,key=lambda x:x['cycle_ms'],reverse=True)[:8],last_record=lastrecord))
out=ROOT/'outputs/latest3_live_analysis.json';out.write_text(json.dumps(reports,indent=2)+'\n')
for r in reports:
 print(json.dumps({k:v for k,v in r.items() if k not in ('metrics','slowest','events')},ensure_ascii=False))
 for g,m in r['metrics'].items():print(g,json.dumps({k:v for k,v in m.items() if k in ('cycle_through_record_ms','compute_to_publish_ms','policy_total_ms','state_to_publish_ms','record_write_ms','sdk_publish_ms','publish_period_ms')},ensure_ascii=False))
 print('EVENTS',json.dumps(r['events'],ensure_ascii=False))
 print('SLOW',json.dumps(r['slowest'][:3],ensure_ascii=False))
