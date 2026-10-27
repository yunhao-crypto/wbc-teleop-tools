#!/usr/bin/env python3
"""Optional headless simulation using an explicitly supplied compatible scene.

MuJoCo and external scene/assets are developer validation inputs, not robot
runtime dependencies. The simulation never opens Aorta or motor control.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mujoco
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
p=argparse.ArgumentParser(description='Headless closed-loop physics validation; never connects to a robot.');p.add_argument('--scene',type=Path,required=True);p.add_argument('--incident-log',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',choices=('default','zero','incident'),default='default');p.add_argument('--seconds',type=float,default=10);p.add_argument('--lead-speed',type=float,default=1.2);p.add_argument('--fixed-base',action='store_true');a=p.parse_args()
if not 0<a.seconds<=120:raise ValueError('simulation duration must be in (0,120]')
if a.initial=='incident' and a.incident_log is None:raise ValueError('--initial incident requires --incident-log')
r=ActorPicoRuntime(root,backend='ort')
m=mujoco.MjModel.from_xml_path(str(a.scene));m.opt.timestep=.002
d=mujoco.MjData(m)
ids=np.array([m.joint(n).id for n in r.model.contract.names]);qids=m.jnt_qposadr[ids];vids=m.jnt_dofadr[ids]
from types import SimpleNamespace
from deployment.vita_s100_aorta_30dof.actor_pico_live import check_policy_step
previous=None
actids={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i):i for i in range(m.nu)}
allnames=list(actids);q0=r.model.contract.default.copy()
if a.initial=='zero':q0[:]=0
if a.initial=='incident':
 rows=[json.loads(x) for x in a.incident_log.read_text().splitlines()]
 fixed=next(x for x in rows if x.get('record_type')=='fixed_stand_confirmed')
 order=r.policy.order;q0=np.array([fixed['q'][order.index(n)] for n in r.model.contract.names])
d.qpos[qids]=q0;d.qpos[3:7]=[1,0,0,0];d.qpos[2]=1.
mujoco.mj_forward(m,d)
footids=[m.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
# Locate the support plane at the initial soles; preserve original contact geometry.
z=[]
for i in range(m.ngeom):
 if m.geom_bodyid[i] in footids and m.geom_contype[i]:
  z.append(d.geom_xpos[i,2]-m.geom_rbound[i])
if not z:raise RuntimeError('no sole collision geometry')
# Use the collision geom extents rather than mesh bounding spheres for boxes.
z=[]
for i in range(m.ngeom):
 if m.geom_bodyid[i] in footids and m.geom_contype[i]:
  rot=d.geom_xmat[i].reshape(3,3)
  extent=float(np.abs(rot[2])@m.geom_size[i]) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[i,0]
  z.append(d.geom_xpos[i,2]-extent)
d.qpos[2]-=min(z)-.001;mujoco.mj_forward(m,d)
height=float(d.qpos[2]);base=np.array(d.qpos[:7]);
config=dict(r.config['body_profile']['lead_in'],speed_q_per_s=a.lead_speed)
lead=r.math.Motion.hold(r.model.contract.default).lead_in(q0,[1,0,0,0],config)
r.reset_inputs();cmd=None;rows=[];maximum_tilt=0.;firststep=None;start=time.monotonic()
for step in range(round(a.seconds/.002)):
 if a.fixed_base:
  d.qpos[:7]=base;d.qvel[:6]=0;mujoco.mj_forward(m,d)
 if step%10==0:
  q=d.qpos[qids].copy();dq=d.qvel[vids].copy();quat=d.qpos[3:7].copy();gyro=d.qvel[3:6].copy()
  tilt=float(np.arccos(np.clip(r.math.rotation(quat)[2,2],-1,1)));maximum_tilt=max(maximum_tilt,tilt)
  if tilt>.8 or d.qpos[2]<height-.25:
   print(json.dumps(dict(status='FALL',t=step*.002,height=float(d.qpos[2]),tilt=tilt,initial=a.initial,fixed_base=a.fixed_base,first_step=firststep)));sys.exit(2)
  cmd,obs,row=r.infer_motion_command(lead,q=q,dq=dq,root=quat,gyro=gyro,ref_time=step*.002)
  body_q=np.zeros(31)
  for i,name in enumerate(r.policy.order):
   if name in actids:body_q[i]=d.qpos[m.joint(name).qposadr[0]]
  check_policy_step(cmd,SimpleNamespace(q=body_q),previous)
  previous=cmd
  targets=np.array([j.q for j in cmd.joints])[r.policy.mapping]
  if firststep is None:firststep=float(np.max(np.abs(targets-q)))
  rows.append(dict(t=step*.002,height=float(d.qpos[2]),tilt=tilt,raw_action=row['raw_action'],q=q.tolist(),target=targets.tolist()))
 for name,j in zip(r.policy.order,cmd.joints):
  if name not in actids:continue
  joint=m.joint(name);q=d.qpos[joint.qposadr[0]];v=d.qvel[joint.dofadr[0]]
  d.ctrl[actids[name]]=j.kp*(j.q-q)+j.kd*(j.dq-v)+j.tau
 mujoco.mj_step(m,d)
print(json.dumps(dict(status='PASS',seconds=a.seconds,initial=a.initial,fixed_base=a.fixed_base,first_step=firststep,max_tilt=maximum_tilt,final_height=float(d.qpos[2]),wall_seconds=time.monotonic()-start)))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as stream:json.dump(dict(scene=str(a.scene),model_sha256=r.config['artifacts']['model']['sha256'],mujoco_version=mujoco.__version__,initial=a.initial,fixed_base=a.fixed_base,seconds=a.seconds,max_tilt=maximum_tilt,first_step=firststep,steps=rows,actions_sent=0,hardware_motion='not_measured',ankle_physics='serial virtual joints; not the hardware motor-space parallel adapter'),stream)
