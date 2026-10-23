#!/usr/bin/env python3
"""Audit a saved no-action capture; standard library only, no device access.

Quantiles linearly interpolate sorted values at (n-1)*q (NumPy default). Source-host and
robot-wall clocks are not calibrated; source-to-output latency is not claimed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def quantile(x,q):
    index=(len(x)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return x[lo]+(x[hi]-x[lo])*(index-lo)


def stats(values):
    x=sorted(float(v) for v in values if v is not None)
    if not x:return {'count':0}
    if not all(math.isfinite(v) for v in x):raise ValueError('Non-finite timing')
    return {'count':len(x),'min':x[0],'mean':sum(x)/len(x),
            **{'p'+str(int(q*100)):quantile(x,q) for q in (.5,.95,.99)},
            'max':x[-1],'over_20':sum(v>20 for v in x)}


def read_jsonl(path):
    rows=[];h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for line in f:
            h.update(line)
            if line.strip():rows.append(json.loads(line))
    return rows,h.hexdigest()


def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]


def timing(rows):
    base=['compute_ms','callback_to_output_ms','state_decode_ms','input_poll_ms','scheduling_lateness_ms']
    out={k:stats(r[k] for r in rows if k in r) for k in base}
    names=sorted({k for r in rows for k in r.get('timing_ms',{})})
    out['stages_ms']={k:stats(r['timing_ms'][k] for r in rows if k in r.get('timing_ms',{})) for k in names}
    return out


def audit_pico(rows,sender):
    index={};collisions=0
    for s in sender:
        t=float(s['timestamp_ns'])*1e-9
        collisions+=t in index;index[t]=s
    if collisions:raise ValueError('Ambiguous sender timestamps after conversion')
    effective_age=[];last_capture_wall=None
    for r in rows:
        p=r.get('pico')
        if p is not None and len(p)>3:last_capture_wall=p[3]
        if last_capture_wall is not None:effective_age.append((r['tick_wall_time']-last_capture_wall)*1000)
    records=[(r,r['pico']) for r in rows if r.get('pico') is not None]
    matched=[];mismatches=[];mismatch_details=[];unknown=[];age=[];local_ids=[];source_times=[]
    for r,p in records:
        body,t,local=p[:3];s=index.get(t);local_ids.append(local);source_times.append(t)
        if len(p)>3:age.append((r['tick_wall_time']-p[3])*1000)
        if s is None:unknown.append(r['index']);continue
        matched.append(s['frame_index'])
        good=len(body)==24 and all(len(v)==7 for v in body)
        if good:good=all(v==f32(w) for a,b in zip(body,s['body']) for v,w in zip(a,b))
        if not good:
            mismatches.append(r['index'])
            flat=[v for row in body for v in row]
            expected=[f32(v) for row in s['body'] for v in row]
            nearby=[v for v in sender if abs(v['frame_index']-s['frame_index'])<=4]
            body_matches=[v['frame_index'] for v in nearby if flat==[f32(z) for row in v['body'] for z in row]]
            mismatch_details.append({'capture_index':r['index'],'timestamp_sender_frame_index':s['frame_index'],
                'nearby_sender_frame_indices_matching_entire_body':body_matches,
                'mismatched_coordinates':sum(x!=y for x,y in zip(flat,expected)),
                'max_abs_body_difference':max((abs(x-y) for x,y in zip(flat,expected)),default=None)})
    def gaps(ids):
        ds=[b-a for a,b in zip(ids,ids[1:])]
        return {'first':ids[0] if ids else None,'last':ids[-1] if ids else None,
                'unique_count':len(set(ids)),'duplicate_adjacent':sum(d==0 for d in ds),
                'backwards':sum(d<0 for d in ds),'gap_events':sum(d>1 for d in ds),
                'skipped_indices_between_samples':sum(max(0,d-1) for d in ds),
                'delta_histogram':dict(sorted(Counter(ds).items()))}
    return {'sender_frames':len(sender),'captured_samples':len(records),
            'unknown_source_timestamps':unknown,'body_mismatch_capture_indices':mismatches,'body_timestamp_mismatch_details':mismatch_details,
            'exact_after_declared_conversion':not unknown and not mismatches and bool(records),
            'comparison':'24x7 body equals sender IEEE float32 cast exactly; timestamp equals float(sender timestamp_ns)*1e-9 exactly. Capture cannot verify original double payload or integer nanoseconds bit-for-bit.',
            'sender_indices_of_captured_samples':gaps(matched),'receiver_local_indices':gaps(local_ids),
            'source_timestamp_interval_ms':stats((b-a)*1000 for a,b in zip(source_times,source_times[1:])),
            'robot_capture_to_tick_wall_ms':stats(age),
            'robot_capture_to_tick_wall_over_600ms':sum(v>600 for v in age),
            'effective_latest_sample_age_all_ticks_ms':stats(effective_age),
            'effective_latest_sample_age_over_600ms':sum(v>600 for v in effective_age),
            'freshness_clock_note':'Robot wall-clock delta only. tick_wall_time is sampled before input polling, so a sample arriving during that tick can give a small negative delta; this is not negative transport latency.',
            'source_clock_latency_claimed':False,
            'sampling_note':'Latest-sample policy polling at 50 Hz can skip sender/reader frames. Gaps here are consumption gaps, not evidence of network or SDK packet loss. Sender startup/end outside capture is excluded.',
            'pico_stored_ticks':sum(bool(r.get('pico_stored')) for r in rows),
            'reference_ready_ticks':sum(bool(r.get('reference_ready')) for r in rows)}


def telemetry(path,rows):
    if not path:return None
    raw=Path(path).read_bytes();ts=json.loads(raw);lo=rows[0]['started_monotonic_ns'];hi=rows[-1]['started_monotonic_ns']
    def subset(samples):
        freq=defaultdict(list);temp=defaultdict(list);mem=[];rss=[];load=[]
        for s in samples:
            for k,v in s.items():
                if k.endswith('/scaling_cur_freq'):
                    freq[k].append(float(v)/1000)
                if re.search(r'/thermal_zone\d+/temp$',k):temp[k].append(float(v)/1000)
                if k=='/proc/meminfo':
                    m=re.search(r'^MemAvailable:\s+(\d+)',v,re.M)
                    if m:mem.append(int(m[1])/1024)
                if k.endswith('/status') and isinstance(v,str):
                    m=re.search(r'^VmRSS:\s+(\d+)',v,re.M)
                    if m:rss.append(int(m[1])/1024)
                if k=='/proc/loadavg':load.append(float(v.split()[0]))
        bounds=lambda a:{'min':min(a),'max':max(a)} if a else None
        return {'samples':len(samples),'cpu_mhz':{k:bounds(v) for k,v in freq.items()},
                'thermal_celsius':{k:bounds(v) for k,v in temp.items()},'mem_available_mib':bounds(mem),
                'process_rss_mib':bounds(rss),'system_load_1min':bounds(load)}
    active=[s for s in ts if lo<=s['monotonic_ns']<=hi]
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
            'all_samples':subset(ts),'within_capture_start_times':subset(active),
            'sample_span_seconds':(ts[-1]['monotonic_ns']-ts[0]['monotonic_ns'])/1e9 if ts else 0,
            'sample_interval_ms':stats((b['monotonic_ns']-a['monotonic_ns'])/1e6 for a,b in zip(ts,ts[1:])),
            'limitation':'Sampled frequency/temperature/memory bounds only; no proof of absence of short throttling or load spikes, and no absolute source-clock latency.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--sender',required=True)
    p.add_argument('--telemetry');p.add_argument('--out',required=True);a=p.parse_args()
    rows,ch=read_jsonl(a.capture);sender,sh=read_jsonl(a.sender)
    if not rows:raise ValueError('Empty capture')
    modes=defaultdict(list);transitions=defaultdict(list)
    for r in rows:
        modes[r['mode']].append(r)
        t=r.get('transition')
        if t:transitions[t['event']].append(r)
    start=[r['started_monotonic_ns'] for r in rows];delta=[(b-a)/1e6 for a,b in zip(start,start[1:])]
    schedule=[r['scheduled_monotonic'] for r in rows]
    periods=[round((b-a)/.02) for a,b in zip(schedule,schedule[1:])]
    sent=[r['index'] for r in rows if r.get('action_sent') is not False]
    inert=[r['inert_output'] for r in rows if r.get('inert_output') is not None]
    out={'schema':'vita_shadow_capture_audit_v1','capture':a.capture,'capture_sha256':ch,'sender':a.sender,'sender_sha256':sh,
         'frames':len(rows),'quantiles':'linear interpolation at (n-1)*q, matching NumPy percentile default; timing units milliseconds',
         'timing_scope':'compute includes decode/input poll/preprocess/FSM/policy/inert conditioning+guard+serialization; excludes JSON capture write and actuation. callback_to_output starts at robot subscriber host receipt, not physical sensor acquisition.',
         'overall':timing(rows),'by_mode':{k:timing(v) for k,v in modes.items()},
         'transitions':{k:{'button':{'motion':'B','return_velocity':'Y','velocity':'A','start':'START'}.get(k),'count':len(v),'accepted':sum(bool(r['transition']['accepted']) for r in v),
                           'capture_indices':[r['index'] for r in v],'timing':timing(v)} for k,v in transitions.items()},
         'switch_timing_note':'fsm_switch_ms includes FSM handling, reset and anchor/yaw alignment; no separately instrumented reset-only duration.',
         'cadence':{'requested_hz':50,'start_span_seconds':(start[-1]-start[0])/1e9,
                    'observed_start_hz':(len(start)-1)*1e9/(start[-1]-start[0]) if len(start)>1 else None,
                    'start_interval_ms':stats(delta),'interval_over_30ms':sum(d>30 for d in delta),
                    'scheduled_period_delta_histogram':dict(sorted(Counter(periods).items())),
                    'skipped_scheduled_periods':sum(max(0,d-1) for d in periods)},
         'pico':audit_pico(rows,sender),
         'no_action_trace':{'all_action_sent_explicit_false':not sent,'violating_capture_indices':sent,
                            'inert_output_frames':len(inert),'ownership_scopes':dict(Counter(v.get('ownership_scope') for v in inert)),
                            'all_guards_allow':all(v.get('guard_allowed') is True for v in inert),
                            'serialized_inert_payload_frames':sum(v.get('payload_bytes',0)>0 for v in inert),
                            'limitation':'Capture-field and inert-sink evidence; not an independent network packet capture.'},
         'telemetry':telemetry(a.telemetry,rows)}
    Path(a.out).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':a.out,'frames':len(rows),'mode_counts':{k:len(v) for k,v in modes.items()},
                      'compute_ms':out['overall']['compute_ms'],'pico_exact_after_conversion':out['pico']['exact_after_declared_conversion'],
                      'no_action_trace':out['no_action_trace'],'cadence':out['cadence']},indent=2))
    if sent or out['pico']['unknown_source_timestamps'] or out['pico']['body_mismatch_capture_indices']:raise SystemExit(1)
if __name__=='__main__':main()
