import sys,types,json,csv,datetime,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'packages/incident_mcap'),str(R/'packages/state_schema')]
a=types.ModuleType('aorta');a.__path__=[str(R/'packages/state_schema/aorta')];sys.modules['aorta']=a
from mcap.reader import make_reader
from lowlevel.HumanoidLowState import HumanoidLowState as S
TZ=datetime.timezone(datetime.timedelta(hours=8));O=R/'outputs/incident_154621'
def ns(t):return int(datetime.datetime.fromisoformat('2026-09-29T'+t+'+08:00').timestamp()*1e9)
fields=['log_ns','publish_ns','wall','state_id','ts_state_real','ts_state_pub','cmd_id','input_cmd_id','attitude_valid','imu_index']+[f'{p}{i}' for p,n in [('quat',4),('gyro',3),('accel',3),('rpy',3)] for i in range(n)]
fields += ['imu_location','imu_status','ts_sampled_us','ts_ingested_us']
rows=[]
for p in sorted(O.glob('*.mcap')):
 with p.open('rb') as f:
  for sc,ch,m in make_reader(f).iter_messages(topics=['aorta/default/pub/sdk/foot_humanoid/lowstate'],start_time=ns('15:46:14'),end_time=ns('15:46:25')):
   x=S.GetRootAs(m.data)
   for i in range(x.ImuStatesLength()):
    sample=x.ImuStates(i);u=sample.State();d=dict(zip(fields[:10],[m.log_time,m.publish_time,datetime.datetime.fromtimestamp(m.log_time/1e9,TZ).isoformat(timespec='microseconds'),x.StateId(),x.TsStateReal(),x.TsStatePub(),x.CmdId(),x.InputCmdId(),x.AttitudeValid(),i]))
    d.update(imu_location=sample.Location(),imu_status=sample.Status(),ts_sampled_us=sample.TsSampledUs(),ts_ingested_us=sample.TsIngestedUs())
    for pr,fn,n in [('quat',u.Quaternion,4),('gyro',u.Gyroscope,3),('accel',u.Accelerometer,3),('rpy',u.Rpy,3)]:
     for j in range(n):d[pr+str(j)]=fn(j)
    rows.append(d)
with (O/'attitude_154614_154625.csv').open('w') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
print('rows',len(rows))
for sec in range(14,25):
 a=[r for r in rows if f'15:46:{sec}.' in r['wall'] and r['imu_index']==0]
 if a:
  print(sec,'first',a[0], 'maxgyro',max(math.sqrt(sum(r['gyro'+str(j)]**2 for j in range(3))) for r in a),flush=True)
