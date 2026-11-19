"""Synthetic MimicLite OP3 export for offline tests; no trained weights."""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from deployment.vita_s100_aorta_30dof.actor_mimiclite import (
    ACTUATOR_SCHEMA, MODEL_SCHEMA, MODEL_SHA256, DEPLOYMENT_SCHEMA, ALIASES, GEOMETRY,
    MimicLiteContract, CONTRACT_KEY, metadata_contract,
)
from deployment.vita_s100_aorta_30dof.actor_parallel_ankle import MotorThetaAnkle

NAMES = ['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint',
    'left_A_achilles_joint','left_B_achilles_joint',
    'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint',
    'right_A_achilles_joint','right_B_achilles_joint','waist_yaw_joint','waist_pitch_joint',
    'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint',
    'left_wrist_roll_joint','left_wrist_yaw_joint','left_wrist_pitch_joint',
    'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint',
    'right_wrist_roll_joint','right_wrist_yaw_joint','right_wrist_pitch_joint']


def deployment():
    names = [ALIASES.get(n,n) for n in NAMES]
    ankle = MotorThetaAnkle(names, dict(schema_version=3,control_space='motor_theta',
        reference_conversion='virtual_pitch_roll_to_motor_theta',geometry=GEOMETRY,
        motor_limits=[-.95975,.61075],roll_sign=dict(left=1,right=-1),
        slots={f'{s}_ankle_{a}_joint':dict(side=s,motor=m) for s in ('left','right')
               for a,m in (('pitch','theta_top'),('roll','theta_bottom'))}))
    default = np.zeros(28)
    for i,n in enumerate(names):
        if 'hip_pitch' in n or 'ankle_pitch' in n: default[i]=-.2
        elif 'knee' in n: default[i]=.4
        elif 'hip_roll' in n: default[i]=.088 if n.startswith('left') else -.088
        elif 'ankle_roll' in n: default[i]=-.088 if n.startswith('left') else .088
        elif 'shoulder_roll' in n: default[i]=.4226 if n.startswith('left') else -.4226
    default=ankle.to_policy(default)
    joints,scale={},[]
    for n in NAMES:
        if n in ALIASES:
            inertia,effort,s=.0192325,50.,.43857731389829635
            kp,kd=20.,1.
        else:
            if 'hip_' in n or 'knee' in n: inertia,effort,s=21.913**2*10.58e-5,120.,.14957953295027204
            elif 'waist_' in n: inertia,effort,s=35**2*1.23e-5,70.5,.29629763758025607
            elif 'wrist_' in n: inertia,effort,s=19.5278**2*1.85e-5,30.,.2692918198976216
            else: inertia,effort,s=20**2*4.38e-5,60.,.21686897183719556
            kp,kd=inertia*(20*math.pi)**2,4*inertia*20*math.pi
        p=dict(kp=kp,kd=kd,effort=effort,armature=inertia,frictionloss=1.6 if n in ALIASES else 0.,
               viscous_damping=.05 if n in ALIASES else 0.,kind='motor' if n in ALIASES else 'position')
        if n in ALIASES: p.update(dynamic_friction=1.28,friction_transition_speed=.01)
        joints[n]=p;scale.append(s)
    offsets=[0,1,2,3,4,8,16]
    policy=[]
    for kind,width in [('root_ang_vel_history',3),('projected_gravity_history',3),
                       ('joint_pos_history',28),('joint_vel_history',28)]:
        term=dict(kind=kind,size=7*width,history_steps=offsets.copy())
        if kind.startswith('joint_'): term['joint_names']=NAMES.copy()
        policy.append(term)
    policy.append(dict(kind='prev_actions',size=84,steps=3))
    command=[dict(kind=k,size=8*w,future_indices=list(range(8))) for k,w in
        [('ref_body_pos_future_local',3),('ref_root_ori_future_b',6),('ref_joint_pos_future',28)]]
    return dict(schema=DEPLOYMENT_SCHEMA,model_schema=MODEL_SCHEMA,model_sha256=MODEL_SHA256,
        actuator_schema=ACTUATOR_SCHEMA,control_space='motor_theta',control_period_s=.02,
        physics_period_s=.005,joint_names=NAMES.copy(),default_position=default.tolist(),
        action_scale=scale,joints=joints,reference_joint_names=NAMES.copy(),
        future_steps=[-8,-4,-2,0,1,2,3,4],reference_root_body='base_link',
        reference_space='motor_theta',quaternion_order='wxyz',deployment_alpha=1.,
        observations=dict(command=command,policy=policy),simulation_options={})


def contract():
    raw=metadata_contract(deployment(),'a'*64,'b'*64)
    return MimicLiteContract.from_metadata(None,{CONTRACT_KEY:json.dumps(raw)})


def source_model(d=None, *, batched=False, bias=0.):
    import onnx
    from onnx import helper as h,TensorProto as T,numpy_helper as n
    d=deployment() if d is None else d
    raw=metadata_contract(d,'a'*64,'b'*64)
    groups=raw['input_groups'];widths=raw['input_widths'];total=sum(widths)
    shapes={k:([1,w] if batched else [w]) for k,w in zip(groups,widths)}
    rng=np.random.default_rng(7)
    weights=(rng.normal(size=(total,28))*.0001).astype(np.float32)
    nodes=[h.make_node('Concat',groups,['joined'],axis=1 if batched else 0),
        h.make_node('MatMul',['joined','weights'],['unscaled']),
        h.make_node('Add',['unscaled','bias'],['action']),
        h.make_node('Identity',[groups[-1]],['priv_student'])]
    model=h.make_model(h.make_graph(nodes,'synthetic_not_trained',
        [h.make_tensor_value_info(k,T.FLOAT,shapes[k]) for k in groups],
        [h.make_tensor_value_info('action',T.FLOAT,[1,28] if batched else [28]),
         h.make_tensor_value_info('priv_student',T.FLOAT,shapes[groups[-1]])],
        [n.from_array(weights,'weights'),n.from_array(np.full(28,bias,np.float32),'bias')]),
        opset_imports=[h.make_opsetid('',18)],ir_version=9)
    return model.SerializeToString()
