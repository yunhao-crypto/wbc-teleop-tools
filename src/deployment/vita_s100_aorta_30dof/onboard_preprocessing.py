"""NumPy FP32 counterpart of the released CUDA pure-SMPL online path.

No Torch, Warp, CUDA, SMPL pickle, network or robot SDK is imported at runtime.
A licensed, hash-bound 24-joint skeleton must first be exported with
scripts/preprocess_export_skeleton.py. The original CUDA implementation remains
untouched and serves as the numerical oracle.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

F = np.float32
PICO_PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21], dtype=np.int32)
COMMAND = np.array([.5,-.5,-.5,-.5], dtype=np.float32)
FK_LEVELS = tuple(np.array(x,dtype=np.int32) for x in ((1,2,3),(4,5,6),(7,8,9),(10,11,12,13,14),(15,16,17),(18,19),(20,21),(22,23)))


_NATIVE = None
_NATIVE_TRIED = False

def _native(stage):
    global _NATIVE, _NATIVE_TRIED
    if not _NATIVE_TRIED:
        path = os.environ.get('VITA_PREPROCESS_LIBRARY')
        if path:
            _NATIVE = ctypes.CDLL(str(Path(path).resolve()))
            p = ctypes.POINTER(ctypes.c_float)
            ip = ctypes.POINTER(ctypes.c_int32)
            for name,args in {
                'pp_pico':[p,ip,p,p],
                'pp_features':[p,p,ip,p,p,p,p],
                'pp_gather':[p,p,p,p,p]+[ctypes.c_int]*6+[p,p,p,p],
                'pp_build':[p]*10+[ctypes.c_int]*2+[p],
            }.items():
                fn=getattr(_NATIVE,name);fn.argtypes=args;fn.restype=None
        _NATIVE_TRIED = True
    enabled = os.environ.get('VITA_PREPROCESS_NATIVE_STAGES','pico,features,gather,build').split(',')
    return _NATIVE if stage in enabled else None


def _ptr(a):
    if not a.flags.c_contiguous: raise ValueError('Native preprocessing requires contiguous arrays')
    typ = ctypes.c_int32 if a.dtype == np.int32 else ctypes.c_float
    if a.dtype not in (np.int32,np.float32):raise ValueError('Native preprocessing dtype mismatch')
    return a.ctypes.data_as(ctypes.POINTER(typ))


def _array(value, shape, name):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite float32 {shape}, got {a.shape}")
    return a


def _mul(a, b):
    """wxyz product; preserve scalar expression order of released Warp kernels."""
    a, b = np.broadcast_arrays(a, b)
    w,x,y,z = np.moveaxis(a, -1, 0)
    v,i,j,k = np.moveaxis(b, -1, 0)
    return np.stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v), axis=-1).astype(np.float32)


def _conj(q):
    return q * np.array([1,-1,-1,-1], dtype=np.float32)


def _norm(q, *, positive=False, reciprocal=False, epsilon=1e-9):
    q = np.asarray(q, dtype=np.float32)
    norm = np.sqrt(((q[...,0]*q[...,0]+q[...,1]*q[...,1])+q[...,2]*q[...,2])+q[...,3]*q[...,3])
    norm = np.maximum(norm, F(epsilon))[..., None]
    out = q * (F(1)/norm) if reciprocal else q / norm
    if positive:
        out = np.where(out[..., :1] < 0, -out, out)
    return out.astype(np.float32)


def _rotate(q, v):
    qv = q[..., 1:]
    t = F(2)*np.cross(qv, v)
    return (v + q[..., :1]*t + np.cross(qv, t)).astype(np.float32)


def _rotate_warp_quat_inv(q, v):
    # Warp builtin quat_rotate_inv uses its own polynomial rather than _qrot.
    qv = q[...,1:]
    dot=((qv[...,0]*v[...,0]+qv[...,1]*v[...,1])+qv[...,2]*v[...,2])[...,None]
    return (v*(F(2)*q[..., :1]*q[..., :1]-F(1)) + qv*(F(2)*dot) -
            (np.cross(qv,v)*q[..., :1])*F(2)).astype(np.float32)


def _yaw(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.arctan2(F(2)*(w*z+x*y),F(1)-F(2)*(y*y+z*z)).astype(np.float32)


def _rot6(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((F(1)-F(2)*(y*y+z*z),F(2)*(x*y-z*w),
                     F(2)*(x*y+z*w),F(1)-F(2)*(x*x+z*z),
                     F(2)*(x*z-y*w),F(2)*(y*z+x*w)),axis=-1).astype(np.float32)


def _axis_quat(pose):
    x,y,z=np.moveaxis(pose,-1,0)
    theta2=(x*x+y*y)+z*z
    theta=np.sqrt(theta2)
    half=F(.5)*theta
    scale=np.divide(np.sin(half),theta,out=np.full_like(theta,F(.5)),where=theta2>=F(1e-8))
    w=np.where(theta2<F(1e-8),F(1),np.cos(half))
    return np.stack((w,x*scale,y*scale,z*scale),axis=-1).astype(np.float32)


class CPUPicoToSmplConverter:
    """Pico world xyzw quaternions -> canonical local SMPL axis angles."""
    def to_device(self, body_poses):
        body=_array(body_poses,(24,7),'body_poses')
        if np.any(np.linalg.norm(body[:,3:7],axis=-1)<=F(1e-8)):
            raise ValueError('body_poses contains a zero quaternion')
        native=_native('pico')
        if native:
            body=np.ascontiguousarray(body);pose=np.empty((1,72),dtype=np.float32);trans=np.empty((1,3),dtype=np.float32)
            native.pp_pico(_ptr(body),_ptr(PICO_PARENTS),_ptr(pose),_ptr(trans))
            return pose,trans
        q=_norm(body[:,[6,3,4,5]],reciprocal=True,epsilon=1e-8)
        s=F(.7071067811865476)
        q=_mul(_mul(np.array([s,s,0,0],dtype=np.float32),q),np.array([0,0,1,0],dtype=np.float32))
        q=_norm(q,reciprocal=True,epsilon=1e-8)
        local=q.copy()
        local[1:]=_norm(_mul(_conj(q[PICO_PARENTS[1:]]),q[1:]),reciprocal=True,epsilon=1e-8)
        local=np.where(local[:,:1]<0,-local,local)
        v=local[:,1:]
        mag=np.sqrt((v[:,0]*v[:,0]+v[:,1]*v[:,1])+v[:,2]*v[:,2])
        angle=F(2)*np.arctan2(mag,np.maximum(local[:,0],F(0)))
        scale=np.divide(angle,mag,out=np.full_like(mag,F(2)),where=mag>F(1e-7))
        pose=(v*scale[:,None]).reshape(1,72).astype(np.float32)
        trans=body[0,[0,2,1]]*np.array([1,-1,1],dtype=np.float32)
        return pose,trans.reshape(1,3)

    __call__=to_device


def load_skeleton(path=None):
    path=Path(path or os.environ.get('VITA_SMPL_SKELETON',''))
    if not path.is_file():
        raise FileNotFoundError('Set VITA_SMPL_SKELETON or pass skeleton_path to the exported skeleton NPZ')
    metadata_path=path.with_suffix('.json')
    meta=json.loads(metadata_path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=meta['artifact_sha256']:
        raise ValueError('Skeleton artifact hash mismatch')
    with np.load(path,allow_pickle=False) as data:
        parents=data['parents'].astype(np.int32)
        offsets=_array(data['offsets'],(24,3),'offsets').copy()
    if parents.shape!=(24,) or parents[0]!=-1 or any(not 0<=int(parents[j])<j for j in range(1,24)):
        raise ValueError('Invalid topologically ordered 24-joint SMPL skeleton')
    if not np.array_equal(parents,PICO_PARENTS):
        raise ValueError('Skeleton topology differs from released Pico/SMPL mapping')
    return parents,offsets,meta


def forward_kinematics(pose, parents, offsets):
    local=_axis_quat(np.asarray(pose,dtype=np.float32).reshape(24,3))
    positions=np.zeros((24,3),dtype=np.float32)
    rotations=np.empty((24,4),dtype=np.float32)
    rotations[0]=local[0]
    for joints in FK_LEVELS:
        p=parents[joints]
        rotations[joints]=_norm(_mul(rotations[p],local[joints]),reciprocal=True,epsilon=1e-8)
        positions[joints]=positions[p]+_rotate(rotations[p],offsets[joints])
    return positions,rotations


def body_features(pose, trans, positions, rotations):
    root=rotations[0]
    native=_rotate_warp_quat_inv(root,positions[1:])
    body=np.empty(258,dtype=np.float32)
    body[:69]=native[:,[2,0,1]].reshape(-1)
    body[69:132]=np.asarray(pose,dtype=np.float32).reshape(72)[3:66]
    relative=_mul(_conj(root),rotations[1:22])
    semantic=_norm(_mul(_mul(_conj(COMMAND),relative),COMMAND),positive=True)
    body[132:]=_rot6(semantic).reshape(-1)
    foot=F(trans[2]+np.minimum(positions[10,2],positions[11,2]))
    root_command=_norm(_mul(root,COMMAND),positive=True)
    return body,foot,root_command


class CPUOnlineSmplReference:
    """Same 62-frame causal ground/future ring and anchor semantics as GPU path."""
    def __init__(self, *, n_future_frames, policy_hz, skeleton_path=None,
                 device='cpu', ground_history_frames=50):
        del device
        self.n_fut_frames=max(int(n_future_frames),0)
        self.sequence_frames=self.n_fut_frames+3
        self.policy_hz=float(policy_hz)
        self.ground_history_frames=int(ground_history_frames)
        if self.policy_hz<=0 or self.ground_history_frames<=0:
            raise ValueError('policy_hz and ground_history_frames must be positive')
        self.capacity=self.ground_history_frames-1+self.sequence_frames
        self._parents,self._offsets,self.skeleton_metadata=load_skeleton(skeleton_path)
        self.body_dim=258
        self._body_ring=np.empty((self.capacity,258),dtype=np.float32)
        self._foot_z_ring=np.empty(self.capacity,dtype=np.float32)
        self._trans_ring=np.empty((self.capacity,3),dtype=np.float32)
        self._root_quat_ring=np.empty((self.capacity,4),dtype=np.float32)
        self._time_ring=np.empty(self.capacity,dtype=np.float32)
        self._anchor=np.empty(3,dtype=np.float32)
        self._sequence_offsets=np.arange(self.ground_history_frames-1,self.capacity)
        self._ground_offsets=self._sequence_offsets[:,None]-np.arange(self.ground_history_frames)[None,:]
        self._qpos_sequence=np.empty((1,self.sequence_frames,7),dtype=np.float32)
        self._sample_time_sequence=np.empty((1,self.sequence_frames),dtype=np.float32)
        self._body_sequence=np.empty((1,self.n_fut_frames+1,258),dtype=np.float32)
        self._pico_converter=CPUPicoToSmplConverter()
        self._write_slot=0;self._version=0;self._gathered_version=-1
        self._last_sample_time=None;self._sample_time_origin=None;self._anchor_ready=False
        self.seen_frames=0;self.received=False;self.last_receive_time=None;self.latest_frame_index=None
        self.kinematics=None

    @property
    def has_reference(self): return self.received
    @property
    def latest_qpos(self): return None
    @property
    def root_rot_queue(self): return None
    def data_age(self,current_time):
        return float('inf') if self.last_receive_time is None else float(current_time)-self.last_receive_time
    def is_ready_for_motion(self,enable_teleop_reference,delay_frames):
        return bool(enable_teleop_reference and self.has_future_sequence())
    def has_future_sequence(self,n_frames=None,*,include_derivative_tail=False):
        count=self.n_fut_frames if n_frames is None else max(int(n_frames),0)
        return count==self.n_fut_frames and self.seen_frames>=self.capacity
    def reset_root_anchor(self):
        self._anchor_ready=False;self._gathered_version=-1

    def store_pico(self,body_poses,*,sample_time=None,current_time=None,frame_index=None):
        pose,trans=self._pico_converter.to_device(body_poses)
        now=float(time.time() if current_time is None else current_time)
        return self.store_smpl(pose,trans,sample_time=now if sample_time is None else sample_time,
                               current_time=now,frame_index=frame_index)

    def store_smpl(self,human_pose_aa,human_root_trans,*,sample_time,current_time=None,frame_index=None):
        pose=np.asarray(human_pose_aa,dtype=np.float32)
        trans=np.asarray(human_root_trans,dtype=np.float32)
        if pose.shape not in ((72,),(1,72)) or trans.shape not in ((3,),(1,3)):
            raise ValueError('SMPL pose/trans must have shape [72]/[3] or [1,72]/[1,3]')
        pose=_array(pose.reshape(72),(72,),'pose');trans=_array(trans.reshape(3),(3,),'trans')
        stamp=float(sample_time)
        if not np.isfinite(stamp): raise ValueError('sample_time must be finite')
        if self._last_sample_time is not None and stamp<=self._last_sample_time: return False
        if self._sample_time_origin is None: self._sample_time_origin=stamp
        native=_native('features')
        if native:
            body=np.empty(258,dtype=np.float32);foot_array=np.empty(1,dtype=np.float32);root=np.empty(4,dtype=np.float32)
            native.pp_features(_ptr(np.ascontiguousarray(pose)),_ptr(np.ascontiguousarray(trans)),_ptr(self._parents),_ptr(self._offsets),_ptr(body),_ptr(foot_array),_ptr(root))
            foot=foot_array[0]
        else:
            positions,rotations=forward_kinematics(pose,self._parents,self._offsets)
            body,foot,root=body_features(pose,trans,positions,rotations)
        slot=self._write_slot
        self._body_ring[slot]=body;self._foot_z_ring[slot]=foot
        self._trans_ring[slot]=trans;self._root_quat_ring[slot]=root
        self._time_ring[slot]=F(stamp-self._sample_time_origin)
        self._write_slot=(slot+1)%self.capacity;self._last_sample_time=stamp
        self.seen_frames+=1;self.received=True
        self.last_receive_time=float(time.time() if current_time is None else current_time)
        self.latest_frame_index=None if frame_index is None else int(frame_index)
        self._version+=1
        return True

    def device_observation_sequence(self,*,n_frames=None,include_derivative_tail=True,frame_index=None,fps=None):
        count=self.n_fut_frames if n_frames is None else max(int(n_frames),0)
        if count!=self.n_fut_frames: raise ValueError('Future frame count mismatch')
        if fps is not None and abs(float(fps)-self.policy_hz)>1e-5: raise ValueError('Reference FPS mismatch')
        if not include_derivative_tail: raise ValueError('SMPL observation requires derivative tail')
        if not self.has_future_sequence(): raise RuntimeError('SMPL reference sequence is unavailable')
        native=_native('gather')
        if self._gathered_version!=self._version and native:
            native.pp_gather(*[_ptr(x) for x in (self._body_ring,self._foot_z_ring,self._trans_ring,self._root_quat_ring,self._time_ring)],
                self._write_slot,self.capacity,self.ground_history_frames,self.sequence_frames,self.n_fut_frames,int(self._anchor_ready),
                *[_ptr(x) for x in (self._anchor,self._qpos_sequence,self._sample_time_sequence,self._body_sequence)])
            self._anchor_ready=True;self._gathered_version=self._version
        if self._gathered_version!=self._version:
            start=self.ground_history_frames-1
            slots=(self._write_slot+self._sequence_offsets)%self.capacity
            if not self._anchor_ready:
                a=slots[1]
                self._anchor[:2]=self._trans_ring[a,:2]
                self._anchor[2]=_yaw(self._root_quat_ring[a])
            yaw=-self._anchor[2];c=np.cos(yaw);s=np.sin(yaw)
            qyaw=np.array([np.cos(F(.5)*yaw),0,0,np.sin(F(.5)*yaw)],dtype=np.float32)
            out=self._qpos_sequence[0]
            xy=self._trans_ring[slots,:2]-self._anchor[:2]
            out[:,0]=c*xy[:,0]-s*xy[:,1];out[:,1]=s*xy[:,0]+c*xy[:,1]
            history=(self._write_slot+self._ground_offsets)%self.capacity
            out[:,2]=self._trans_ring[slots,2]-np.min(self._foot_z_ring[history],axis=1)
            quats=_norm(_mul(qyaw,self._root_quat_ring[slots]),positive=True)
            for f in range(1,len(quats)):
                if np.dot(quats[f-1],quats[f])<0: quats[f]=-quats[f]
            out[:,3:]=quats
            self._sample_time_sequence[0]=self._time_ring[slots]-self._time_ring[slots[0]]
            self._body_sequence[0]=self._body_ring[slots[1:1+self.n_fut_frames+1]]
            self._anchor_ready=True;self._gathered_version=self._version
        return self._qpos_sequence,self._sample_time_sequence,self._body_sequence

    def current_root_rot(self):
        if not self.has_future_sequence(): return None
        return self.device_observation_sequence()[0][0,1,3:].copy()


def root_kinematics(root,times):
    root=np.asarray(root,dtype=np.float32);times=np.asarray(times,dtype=np.float32)
    n=len(root);lo=np.maximum(np.arange(n)-1,0);hi=np.minimum(np.arange(n)+1,n-1)
    dt=np.maximum(times[hi]-times[lo],F(1e-6))
    linear=(root[hi,:3]-root[lo,:3])/dt[:,None]
    # Keep the distinct expression order from _angular_velocity_wxyz.
    q0=root[lo,3:];q1=root[hi,3:]
    a,b,c,d=np.moveaxis(q0,-1,0);w,x,y,z=np.moveaxis(q1,-1,0)
    relative=np.stack((w*a+x*b+y*c+z*d,-w*b+x*a-y*d+z*c,
                       -w*c+x*d+y*a-z*b,-w*d-x*c+y*b+z*a),axis=-1)
    relative=np.where(relative[:,:1]<0,-relative,relative)
    relative=_norm(relative,reciprocal=True)
    v=relative[:,1:];mag=np.sqrt((v[:,0]*v[:,0]+v[:,1]*v[:,1])+v[:,2]*v[:,2])
    scale=np.divide(F(2)*np.arctan2(mag,relative[:,0]),mag,out=np.full_like(mag,F(2)),where=mag>F(1e-7))
    angular=(v*scale[:,None])/dt[:,None]
    inverse=_conj(root[:,3:])
    return (_rotate(inverse,linear),_rotate(inverse,angular),
            _rotate(inverse,np.broadcast_to(np.array([0,0,-1],dtype=np.float32),(n,3))))


class CPUSmplMotionObservationBuilder:
    def __init__(self,*,n_future_frames,default_dof_pos,policy_hz,device='cpu'):
        del device
        self.n_future_frames=int(n_future_frames);self.policy_hz=float(policy_hz)
        self.defaults=np.asarray(default_dof_pos,dtype=np.float32)
        if self.defaults.ndim!=1 or self.defaults.size<=0 or not np.isfinite(self.defaults).all():
            raise ValueError('default_dof_pos must be a finite non-empty vector')
        self.robot_dof_dim=self.defaults.size
        self._output=np.empty((1,258+18+3*self.robot_dof_dim+self.n_future_frames*(258+18)),dtype=np.float32)

    def build(self,*,smpl_reference,robot_root_quat_wxyz,robot_root_angvel_local,
              robot_dof_pos,robot_dof_vel,last_action,reference_yaw_alignment_wxyz=None,
              reference_frame_index=None):
        states,times,body=smpl_reference.device_observation_sequence(n_frames=self.n_future_frames,
            fps=self.policy_hz,include_derivative_tail=True,frame_index=reference_frame_index)
        native=_native('build')
        if native:
            rq=np.ascontiguousarray(_array(robot_root_quat_wxyz,(4,),'robot_root_quat_wxyz'))
            align=np.ascontiguousarray(_array([1,0,0,0] if reference_yaw_alignment_wxyz is None else reference_yaw_alignment_wxyz,(4,),'alignment'))
            values=[states,times,body,rq]+[np.ascontiguousarray(_array(v,(d,),n)) for v,d,n in (
                (robot_root_angvel_local,3,'angvel'),(robot_dof_pos,self.robot_dof_dim,'dof_pos'),
                (robot_dof_vel,self.robot_dof_dim,'dof_vel'),(last_action,self.robot_dof_dim,'last_action'))]+[self.defaults,align]
            native.pp_build(*[_ptr(x) for x in values],self.n_future_frames,self.robot_dof_dim,_ptr(self._output))
            if not np.isfinite(self._output).all():raise ValueError('Nonfinite motion observation')
            return self._output
        root=states[0];body=body[0]
        linear,angular,gravity=root_kinematics(root,times[0])
        rq=_norm(_array(robot_root_quat_wxyz,(4,),'robot_root_quat_wxyz'))
        align=_norm(_array([1,0,0,0] if reference_yaw_alignment_wxyz is None else reference_yaw_alignment_wxyz,(4,),'alignment'))
        reference=_norm(root[1:2+self.n_future_frames,3:])
        aligned=_norm(_mul(align,reference))
        yaw_error=_yaw(aligned[0])-_yaw(rq)
        w,x,y,z=rq
        robot_gravity=np.array([F(2)*(-z*x+w*y),-F(2)*(z*y+w*x),F(1)-F(2)*(w*w+z*z)],dtype=np.float32)
        current=[gravity[1],linear[1],angular[1],body[0],root[1,2:3],
                 np.array([np.sin(yaw_error),np.cos(yaw_error)],dtype=np.float32),robot_gravity,
                 _array(robot_root_angvel_local,(3,),'robot_root_angvel_local'),
                 _array(robot_dof_pos,(self.robot_dof_dim,),'robot_dof_pos')-self.defaults,
                 _array(robot_dof_vel,(self.robot_dof_dim,),'robot_dof_vel'),
                 _array(last_action,(self.robot_dof_dim,),'last_action')]
        future=slice(2,2+self.n_future_frames)
        delta=_yaw(reference[1:])-_yaw(reference[0])
        relative=_norm(_mul(_conj(rq),aligned[1:]))
        arrays=current+[body[1:].reshape(-1),root[future,2],gravity[future].reshape(-1),
            linear[future].reshape(-1),angular[future].reshape(-1),
            np.stack((np.sin(delta),np.cos(delta)),axis=-1).reshape(-1),_rot6(relative).reshape(-1)]
        self._output[0]=np.concatenate(arrays)
        if not np.isfinite(self._output).all(): raise ValueError('Nonfinite motion observation')
        return self._output


__all__=['CPUPicoToSmplConverter','CPUOnlineSmplReference','CPUSmplMotionObservationBuilder',
         'forward_kinematics','body_features','root_kinematics','load_skeleton']
