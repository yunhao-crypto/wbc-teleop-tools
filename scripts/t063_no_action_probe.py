#!/usr/bin/env python3
"""T063 on-device comparison: subscriber only; command bytes never published."""
import argparse,dataclasses,gzip,hashlib,importlib,importlib.metadata,json,math,os,sys,threading,time,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label',choices=['baseline','rc3'],required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=60);p.add_argument('--hz',type=int,default=50);p.add_argument('--replay',type=Path);a=p.parse_args()
assert (a.seconds,a.hz)==(60,50)
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
assert os.environ.get('ZENOH_SESSION_CONFIG_URI')=='/app_param/zenoh/s100_session_peer.json5'
sys.path.insert(0,str(a.source/'src'))
import aorta,flatbuffers
import locomotion_aorta
from locomotion_aorta import LowCmd,MotorCommand,ExternalController
from locomotion_aorta.transport import _AortaTransport,_fill_low_cmd,_decode_humanoid_lowstate
from lowlevel.HumanoidLowState import HumanoidLowState
from deployment.vita_s100_aorta_30dof import onboard_command_encoding
low=importlib.import_module('lowlevel.LowCmd');header_module=importlib.import_module('aorta.sys.AortaHeader')
forbidden_calls=[]
def forbidden(*args,**kwargs):
 forbidden_calls.append('publisher/client/controller/transport creation or command')
 raise RuntimeError('T063 NO ACTION: control and publication forbidden')
blocked=[]
for n in dir(aorta.Node):
 if n.startswith('create_') and any(x in n for x in ['publisher','client','action','service']):setattr(aorta.Node,n,forbidden);blocked.append(n)
ExternalController.__init__=forbidden;ExternalController.enter_external=forbidden;ExternalController.publish=forbidden;_AortaTransport.__init__=forbidden
bindings=types.SimpleNamespace(low_cmd_module=low,motor_cmd_module=onboard_command_encoding)
command=LowCmd(cmd_id=65535,motor_cmds=tuple(MotorCommand(mode=10,q=(i-20)*0.003,dq=0.0,tau=0.0,kp=30+i,kd=1.0+i/100) for i in range(42)),external_version=1,external_session_id=123456789,external_sequence=7)
def encode():
 b=flatbuffers.Builder(2048);header_module.AortaHeaderStart(b);header_module.AortaHeaderAddPublishStampNs(b,1790000000000000000);h=header_module.AortaHeaderEnd(b);b.Finish(_fill_low_cmd(bindings,command,b,h));return bytes(b.Output())
def canonical(v):return json.dumps(dataclasses.asdict(v),sort_keys=True,separators=(',',':'),allow_nan=False)
def percentile(xs,q):
 xs=sorted(xs);i=(len(xs)-1)*q;l=int(i);u=min(l+1,len(xs)-1);return xs[l]+(xs[u]-xs[l])*(i-l)
def stats(xs):
 return {'n':len(xs),'unit':'ms','mean':sum(xs)/len(xs),'p50':percentile(xs,.5),'p95':percentile(xs,.95),'p99':percentile(xs,.99),'min':min(xs),'max':max(xs),'negative':sum(x<0 for x in xs)} if xs else {'n':0}
condition=threading.Condition();latest=None;callback_count=0;errors=[];callback_costs=[];rows=[];node=None;sub=None
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();firmware=hashlib.sha256(Path('/app/version_info.txt').read_bytes()).hexdigest()
a.out.mkdir(parents=True,exist_ok=False)
profile=Path(os.environ['ZENOH_SESSION_CONFIG_URI']).read_bytes()
statefile=a.out/'first-lowstate.bin';wire=encode();(a.out/'unsent-lowcmd.bin').write_bytes(wire)
replay=a.replay.read_bytes() if a.replay else None;replay_canonical=None
started=time.time_ns();warmup=10

def callback(payload,context):
 global latest,callback_count
 begin=time.perf_counter_ns();wall=time.time_ns();mono=time.monotonic_ns();blob=bytes(payload);native=int(getattr(context,'receive_stamp_ns',0))
 with condition:
  callback_count+=1;latest=(blob,wall,mono,native,callback_count);condition.notify_all()
 callback_costs.append((time.perf_counter_ns()-begin)/1e6)
try:
 node=aorta.Node('t063-lowstate-readonly-'+a.label+'-'+str(os.getpid()),'default')
 assert node.group()=='default'
 sub=node.create_subscriber('/sdk/foot_humanoid/lowstate',callback,qos=aorta.QoS.sensor_data(),options=aorta.SubscriberOptions(include_receive_timestamp=True,enable_header_context=True))
 with condition:
  if not condition.wait_for(lambda:latest is not None,timeout=10):raise TimeoutError('No LowState within startup window')
 previous=0;period=1/a.hz;start=time.monotonic();total=a.seconds*a.hz
 for index in range(total+warmup):
  deadline=start+index*period;delay=deadline-time.monotonic()
  if delay>0:time.sleep(delay)
  with condition:
   if not condition.wait_for(lambda:latest is not None and latest[4]>previous,timeout=.2):raise TimeoutError('No advancing LowState')
   raw,cb_wall,cb_mono,native,number=latest;previous=number;selected=time.time_ns()
  if replay is None:replay=raw
  if not statefile.exists():statefile.write_bytes(raw)
  view=HumanoidLowState.GetRootAs(raw);h=view.AortaHeader();stamp=int(h.PublishStampNs()) if h else 0
  begin=time.perf_counter_ns();cpu=time.thread_time_ns();decoded=_decode_humanoid_lowstate(view);decode_ms=(time.perf_counter_ns()-begin)/1e6;decode_cpu=(time.thread_time_ns()-cpu)/1e6
  if len(decoded.motor_states)!=44:raise ValueError('LowState motor roster is not44')
  begin=time.perf_counter_ns();cpu=time.thread_time_ns();decoded_replay=_decode_humanoid_lowstate(HumanoidLowState.GetRootAs(replay));replay_ms=(time.perf_counter_ns()-begin)/1e6;replay_cpu=(time.thread_time_ns()-cpu)/1e6
  if replay_canonical is None:replay_canonical=canonical(decoded_replay)
  begin=time.perf_counter_ns();cpu=time.thread_time_ns();encoded=encode();encode_ms=(time.perf_counter_ns()-begin)/1e6;encode_cpu=(time.thread_time_ns()-cpu)/1e6
  if encoded!=wire:raise ValueError('Unsent command bytes changed')
  if index>=warmup:
   if not native or not stamp:raise ValueError('Missing native/header timestamp; no zero substitution')
   rows.append({'i':index-warmup,'state_id':int(decoded.state_id),'header_ns':stamp,'native_received_ns':native,'callback_ns':cb_wall,'selected_ns':selected,'header_to_native_ms':(native-stamp)/1e6,'native_to_callback_ms':(cb_wall-native)/1e6,'callback_to_selected_ms':(selected-cb_wall)/1e6,'live_sdk_decode_ms':decode_ms,'live_sdk_decode_cpu_ms':decode_cpu,'same_frame_replay_decode_ms':replay_ms,'same_frame_replay_decode_cpu_ms':replay_cpu,'unsent_encode_ms':encode_ms,'unsent_encode_cpu_ms':encode_cpu,'schedule_late_ms':max(0,(time.monotonic()-deadline)*1000),'motor_count':len(decoded.motor_states)})
 # Keep the defined measurement schedule at a full60seconds after ten warmups.
 remaining=start+(warmup+total)*period-time.monotonic()
 if remaining>0:time.sleep(remaining)
except BaseException as e:errors.append(type(e).__name__+': '+str(e))
finally:
 for x in [sub,node]:
  if x is not None:
   try:x.close()
   except Exception as e:errors.append('cleanup '+type(e).__name__+': '+str(e))
with gzip.open(a.out/'timings.jsonl.gz','wt') as f:
 for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
if replay is not None:(a.out/'replay-lowstate.bin').write_bytes(replay)
if replay_canonical is not None:(a.out/'replay-decoded.json').write_text(replay_canonical+'\n')
keys=['header_to_native_ms','native_to_callback_ms','callback_to_selected_ms','live_sdk_decode_ms','live_sdk_decode_cpu_ms','same_frame_replay_decode_ms','same_frame_replay_decode_cpu_ms','unsent_encode_ms','unsent_encode_cpu_ms','schedule_late_ms']
report={'label':a.label,'method':'same60s/50Hz3000frames; actual lowstate subscriber; unsent 42slot command encoding; same real frame replay decode','measurement_seconds':a.seconds,'frames':len(rows),'warmup_frames':warmup,'callback_count':callback_count,'source_root':str(a.source),'python':sys.executable,'sdk_paths':{'aorta':aorta.__file__,'locomotion_aorta':locomotion_aorta.__file__},'sdk_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(locomotion_aorta.__file__).parent.glob('*.py')},'versions':{n:importlib.metadata.version(n) for n in ['locomotion-aorta','aorta-sdk','aorta-msgs','flatbuffers']},'firmware_sha256':firmware,'firmware_unchanged':firmware==hashlib.sha256(Path('/app/version_info.txt').read_bytes()).hexdigest(),'boot_id':boot,'boot_unchanged':boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'profile_sha256':hashlib.sha256(profile).hexdigest(),'started_wall_ns':started,'finished_wall_ns':time.time_ns(),'clock':'CLOCK_REALTIME for stamps; monotonic/perf_counter durations; no independent clock-offset calibration','metrics':{k:stats([r[k] for r in rows]) for k in keys},'callback_copy_cost_ms':stats(callback_costs),'replay_sha256':hashlib.sha256(replay).hexdigest() if replay else None,'unsent_lowcmd_sha256':hashlib.sha256(wire).hexdigest(),'forbidden_factories':blocked,'forbidden_calls':forbidden_calls,'publishers_created':0,'controllers_created':0,'external_requested':False,'commands_sent':0,'errors':errors,'passed':len(rows)==a.seconds*a.hz and not errors and not forbidden_calls}
(a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
