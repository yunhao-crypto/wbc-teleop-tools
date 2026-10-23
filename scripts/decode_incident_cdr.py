import sys,json,datetime,dataclasses
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'packages/incident_rosbags'),str(R/'packages/incident_mcap')]
from mcap.reader import make_reader
from rosbags.typesys import Stores,get_typestore,get_types_from_msg
O=R/'outputs/incident_154621';tz=datetime.timezone(datetime.timedelta(hours=8));store=get_typestore(Stores.EMPTY)
def simplify(x):
 if dataclasses.is_dataclass(x):return {f.name:simplify(getattr(x,f.name)) for f in dataclasses.fields(x) if f.name!='__msgtype__'}
 if hasattr(x,'tolist'):return x.tolist()
 if isinstance(x,(list,tuple)):return [simplify(i) for i in x]
 return x
start=int(datetime.datetime.fromisoformat('2026-09-29T15:46:14+08:00').timestamp()*1e9);end=start+12_000_000_000
count={}
with (O/'cdr_events_154614_154626.jsonl').open('w') as out:
 for p in sorted(O.glob('*.mcap')):
  with p.open('rb') as f:
   reader=make_reader(f);s=reader.get_summary();topics=['/locomotion/status','/locomotion/event','/locomotion/business_event','/locomotion/task_report','/rt/foot_humanoid/ankle_debug','/rosout']
   for c in s.channels.values():
    if c.topic in topics:
     z=s.schemas[c.schema_id];store.register(get_types_from_msg(z.data.decode(),z.name))
   for sc,c,m in reader.iter_messages(topics=topics,start_time=start,end_time=end):
    d=simplify(store.deserialize_cdr(m.data,sc.name));r={'topic':c.topic,'log_ns':m.log_time,'publish_ns':m.publish_time,'wall':datetime.datetime.fromtimestamp(m.log_time/1e9,tz).isoformat(timespec='microseconds'),'data':d};out.write(json.dumps(r,separators=(',',':'))+'\n');count[c.topic]=count.get(c.topic,0)+1
print(count)
