#!/usr/bin/env python3
"""Offline MCAP decoder. Only generated FlatBuffers data classes; no SDK runtime."""
import sys, types, importlib, json, hashlib, datetime, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages/incident_mcap'),str(ROOT/'packages/state_schema')]
aorta=types.ModuleType('aorta'); aorta.__path__=[str(ROOT/'packages/state_schema/aorta')]; sys.modules['aorta']=aorta
from mcap.reader import make_reader
TZ=datetime.timezone(datetime.timedelta(hours=8))
def wall(ns): return datetime.datetime.fromtimestamp(ns/1e9,TZ).isoformat(timespec='microseconds')
def obj(x):
 if isinstance(x,bytes): return x.decode('utf-8','replace')
 if isinstance(x,(str,int,float,bool)) or x is None: return x
 if hasattr(x,'tolist'): return x.tolist()
 if isinstance(x,(tuple,list)): return [obj(v) for v in x]
 return {k:obj(v) for k,v in vars(x).items()}
metas={'locomotion_sdk.ControlRequest':'control_request','locomotion_sdk.ControlAck':'control_ack','locomotion_sdk.ControlStatus':'control_status','lowlevel.LowCmd':'low_cmd','lowlevel.HumanoidLowState':'humanoid_low_state'}
out=ROOT/'outputs/incident_154621'; inventory=[]; counts=collections.Counter()
with (out/'flatbuffer_decoded.jsonl').open('w') as dst:
 for p in sorted(out.glob('*.mcap')):
  with p.open('rb') as f:
   r=make_reader(f); s=r.get_summary(); topics=[]
   for ch in s.channels.values():
    sc=s.schemas[ch.schema_id]
    if sc.name not in metas: continue
    expected=importlib.import_module(metas[sc.name]+'_schema_meta').SCHEMA_HASH_HEX
    actual=hashlib.sha256(sc.data).hexdigest()
    inventory.append({'file':p.name,'topic':ch.topic,'schema':sc.name,'sha256':actual,'local_schema_matches':expected==actual})
    assert expected==actual,(sc.name,expected,actual)
    topics.append(ch.topic)
   for sc,ch,m in r.iter_messages(topics=topics):
    mod=importlib.import_module(sc.name); cls=getattr(mod,sc.name.split('.')[-1]); decoded=obj(getattr(mod,cls.__name__+'T').InitFromObj(cls.GetRootAs(m.data)))
    rec={'file':p.name,'topic':ch.topic,'log_time_ns':m.log_time,'publish_time_ns':m.publish_time,'wall':wall(m.log_time),'sequence':m.sequence,'data':decoded}
    dst.write(json.dumps(rec,separators=(',',':'))+'\n'); counts[ch.topic]+=1
(out/'flatbuffer_schema_inventory.json').write_text(json.dumps({'schemas':inventory,'counts':dict(counts)},indent=2))
print(json.dumps(dict(counts),indent=2))
