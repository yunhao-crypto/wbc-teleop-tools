"""SONIC reference generation and arm-only Pico fusion. No robot transport.

SONIC 30->50 Hz CPU generation with configurable lookahead and replanning.
Vita adds bridge history and arm fusion.
Quaternions are wxyz throughout. The actor remains the only motor policy.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
import threading
from pathlib import Path
import time
import numpy as np

VITA_NAMES = tuple(
    [f'{s}_{j}_joint' for s in ('left','right') for j in
     ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    + ['waist_yaw_joint','waist_pitch_joint']
    + [f'{s}_{j}_joint' for s in ('left','right') for j in
       ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_yaw','wrist_pitch')])
STICK_DEADZONE = .12
G1_INDEX = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,21,20,22,23,24,25,26,28,27])


def interpolate(a, b, weight):
    a, b = np.asarray(a), np.array(b, copy=True)
    t = np.asarray(weight)[..., None]
    dot = np.sum(a[...,3:7]*b[...,3:7],axis=-1,keepdims=True)
    b[...,3:7] *= np.where(dot < 0, -1, 1)
    dot = np.clip(np.abs(dot),0,1)
    angle = np.arccos(dot)
    denom = np.maximum(np.sin(angle),1e-12)
    q = np.sin((1-t)*angle)/denom*a[...,3:7]+np.sin(t*angle)/denom*b[...,3:7]
    q = np.where(dot > .9995, (1-t)*a[...,3:7]+t*b[...,3:7], q)
    out = (1-t)*a+t*b
    out[...,3:7] = q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    return out


def sample(sequence, frames):
    frames = np.clip(frames,0,len(sequence)-1)
    lo = np.floor(frames).astype(int)
    return interpolate(sequence[lo], sequence[np.minimum(lo+1,len(sequence)-1)],frames-lo)


@dataclass(frozen=True)
class Command:
    speed: float = 0.
    heading: float = 0.
    backward: bool = False
    turning: bool = False


class StickCommand:
    """Direct translation and ramped turning; neutral immediately requests IDLE."""
    def __init__(self, speed=.25, yaw_rate=.35, backward_speed=.175, turn_speed_limit=None):
        values=(speed,backward_speed,yaw_rate)
        if not np.isfinite(values).all() or not 0<speed<=.8 or not 0<backward_speed<=.8 or not 0<yaw_rate<=.5:
            raise ValueError('Invalid walking speed/yaw limits')
        if turn_speed_limit is not None and (not np.isfinite(turn_speed_limit) or not 0<turn_speed_limit<=max(speed,backward_speed)):
            raise ValueError('Invalid turning translation speed limit')
        self.limit, self.backward_limit, self.yaw_rate = speed, backward_speed, yaw_rate
        self.turn_speed_limit=turn_speed_limit
        self.reset()

    def reset(self):
        self.armed = False
        self.speed = self.heading = self.turn_rate = 0.
        self.stamp = 0
        self.receipt = -np.inf

    def update(self, sample, now, dt=.02, *, enabled=True):
        valid = False
        try:
            receipt, stamp, forward, turn = map(float, sample)
            # Cached packets retain their earliest receipt: SDK now-age has
            # polling jitter, which must neither disarm nor extend freshness.
            if stamp == self.stamp:
                receipt = min(receipt, self.receipt)
            valid = (np.isfinite([receipt,stamp,forward,turn,now]).all()
                     and 0 <= now-receipt <= .25 and stamp > 0 and stamp >= self.stamp
                     and max(abs(forward),abs(turn)) <= 1
                     and (stamp > self.stamp or receipt <= self.receipt))
        except (TypeError,ValueError):
            pass
        if not valid:
            self.armed = False
            forward = turn = 0.
        else:
            self.stamp, self.receipt = stamp, receipt
            if abs(forward) <= STICK_DEADZONE and abs(turn) <= STICK_DEADZONE:
                self.armed = True
        def axis(x):
            return np.sign(x)*max(0.,abs(x)-STICK_DEADZONE)/(1-STICK_DEADZONE)
        direction_limit=self.backward_limit if forward<0 else self.limit
        target = axis(forward)*direction_limit if self.armed and enabled else 0.
        if self.turn_speed_limit is not None and abs(turn)>STICK_DEADZONE:
            target=float(np.clip(target,-self.turn_speed_limit,self.turn_speed_limit))
        self.speed = float(target)
        target_turn = -axis(turn)*self.yaw_rate if self.armed and enabled else 0.
        if target_turn==0.:self.turn_rate=0.
        else:self.turn_rate += float(np.clip(target_turn-self.turn_rate,-.7*dt,.7*dt))
        self.heading += self.turn_rate*dt
        output_limit=self.backward_limit if self.speed<0 else self.limit
        magnitude=round(abs(self.speed)/.025)*.025
        # Preserve exact configured maxima such as 0.43 instead of rounding to 0.425.
        if abs(self.speed)>=output_limit-1e-9:magnitude=output_limit
        return Command(min(output_limit,magnitude),round(self.heading/.025)*.025,self.speed<-.0125,abs(self.turn_rate)>1e-6)


class PlannerPipeline:
    def __init__(self, planner, bridge, names, *, threads=4, seed=1234, max_speed=.3):
        import onnxruntime as ort
        if not np.isfinite(max_speed) or not 0<max_speed<=.8:raise ValueError('Invalid planner maximum speed')
        self.max_speed=float(max_speed)
        self.names = tuple(names)
        if len(self.names)!=28 or set(self.names)!=set(VITA_NAMES):
            raise ValueError('Planner needs the Vita02A 28-joint actor contract')
        self.order = [VITA_NAMES.index(n) for n in self.names]
        self.g1 = G1_INDEX[self.order]
        def session(path, threads):
            opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
            opts.add_session_config_entry('session.intra_op.allow_spinning','0')
            return ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
        self.planner, self.bridge = session(planner,threads),session(bridge,1)
        self.seed=int(seed);self.rng=np.random.default_rng(self.seed)
        self.allowed_counts=(48,52,56,60,64)
        self.motion=None;self.origin=0;self.replans=0;self.last_inference_ms=0.
        self.moving_lookahead=22;self.idle_lookahead=2;self.blend_frames=8;self.walk_mode=1

    def infer(self, context, command):
        if not np.isfinite([command.speed,command.heading]).all() or not 0 <= command.speed <= getattr(self,'max_speed',.3):
            raise ValueError('Planner command exceeds configured walking limit')
        heading=command.heading
        facing=np.array([[np.cos(heading),np.sin(heading),0]],np.float32)
        moving=command.speed > 0
        inputs=dict(context_mujoco_qpos=np.asarray(context[None],np.float32),
            target_vel=np.array([command.speed if moving else -1],np.float32),
            mode=np.array([getattr(self,'walk_mode',1) if moving else int(command.turning)],np.int64),
            movement_direction=facing*(-1 if command.backward else 1) if moving else np.zeros((1,3),np.float32),
            facing_direction=facing,random_seed=np.array([int(self.rng.integers(0,2**31-1))],np.int64),
            has_specific_target=np.zeros((1,1),np.int64),specific_target_positions=np.zeros((1,4,3),np.float32),
            specific_target_headings=np.zeros((1,4),np.float32),
            allowed_pred_num_tokens=np.array([[int(24+4*i in self.allowed_counts) for i in range(11)]],np.int64),height=np.array([-1],np.float32))
        started=time.perf_counter()
        pred,count=self.planner.run(['mujoco_qpos','num_pred_frames'],inputs)
        self.last_inference_ms=(time.perf_counter()-started)*1000
        if pred.shape!=(1,64,36) or pred.dtype!=np.float32 or count.shape!=(1,) or count.dtype!=np.int32:
            raise ValueError('Invalid planner output signature')
        n=int(count[0])
        if n not in self.allowed_counts:
            raise ValueError('Invalid planner output signature')
        seq=pred[0,:n].astype(float)
        if not np.isfinite(seq).all() or np.any(np.linalg.norm(seq[:,3:7],axis=1)<.5):
            raise ValueError('Invalid planner pose')
        return sample(seq,np.arange(n*5//3)*.6)

    def start(self, q, quat=(1.,0.,0.,0.)):
        q=np.asarray(q,dtype=float);quat=np.asarray(quat,dtype=float)
        if q.shape!=(28,) or quat.shape!=(4,) or not np.isfinite(np.r_[q,quat]).all() or np.linalg.norm(quat)<.5:
            raise ValueError('Invalid measured planner start pose')
        # Official InitializeContext uses identity orientation and measured joints.
        frame=np.zeros(36,dtype=np.float32);frame[2]=.788740
        frame[3]=1.
        frame[7+self.g1]=q
        self.rng=np.random.default_rng(getattr(self,'seed',1234))
        self.motion=self.infer(np.repeat(frame[None],4,axis=0),Command())
        self.origin=0;self.replans=0
        return self.map(self.motion,0),0

    def map(self, motion, origin):
        features=[sample(motion,np.arange(len(motion))+offset*50/30)[:,7:] for offset in (-4,-2,0,2,4)]
        w,x,y,z=motion[:,3:7].T
        gravity=np.stack((-2*(x*z-w*y),-2*(y*z+w*x),-(1-2*(x*x+y*y))),axis=1)
        features=np.concatenate(features+[gravity],axis=1).astype(np.float32)
        residual=self.bridge.run(['joint_residual'],{'window_features':features})[0]
        if residual.shape!=(len(motion),28) or not np.isfinite(residual).all():
            raise ValueError('Invalid bridge residual')
        ramp=np.clip((origin+np.arange(len(motion)))/25,0,1)[:,None]
        q=motion[:,7+self.g1]+ramp*residual[:,self.order]
        return q,motion[:,:3].copy(),motion[:,3:7].copy()

    def predict(self, absolute_cursor, command):
        """Worker-only inference; the playing timeline is not modified here."""
        gen=absolute_cursor+(self.moving_lookahead if command.speed>0 else self.idle_lookahead)
        sequence=self.infer(sample(self.motion,gen-self.origin+np.arange(4)*50/30),command)
        return sequence,gen

    def accept(self, prediction, absolute_cursor):
        """Official arrival-time blend, retaining seven past frames for bridge."""
        sequence,absolute_gen=prediction
        cursor=absolute_cursor-self.origin;gen=absolute_gen-self.origin
        if gen+len(sequence)<=cursor:
            raise TimeoutError('Planner prediction expired before arrival')
        rebase=max(0,cursor-7);ins=gen-rebase
        n=ins+len(sequence);frames=np.arange(n)
        old=sample(self.motion,rebase+frames)
        new=sample(sequence,frames-ins)
        # Late results start their crossfade at the current playback frame.
        blend_start=max(cursor-rebase,ins)
        motion=interpolate(old,new,np.clip((frames-blend_start)/getattr(self,'blend_frames',8),0,1))
        origin=self.origin+rebase
        mapped=self.map(motion,origin)
        self.motion,self.origin=motion,origin;self.replans+=1
        return mapped,origin


class ArmFusion:
    """Bound reference speed and acceleration; legs/root are never overwritten."""
    def __init__(self, names, default):
        self.slots=np.array([i for i,n in enumerate(names) if any(s in n for s in ('shoulder','elbow','wrist'))])
        self.q=np.asarray(default)[self.slots].copy();self.velocity=np.zeros(len(self.slots))

    def apply(self, lower, upper, *, dt=.02):
        q,base,quat=lower
        result=q.copy()
        if upper is not None:
            upper=np.asarray(upper,dtype=float)
            if upper.ndim!=2 or upper.shape[1]!=28 or len(upper)==0 or not np.isfinite(upper).all():
                raise ValueError('Invalid Pico arm reference')
        # On loss, brake smoothly before holding; never zero velocity abruptly.
        def advance(position, velocity, target):
            wanted=np.zeros_like(velocity) if target is None else np.clip((target-position)/.2,-1.2,1.2)
            velocity=velocity+np.clip(wanted-velocity,-4*dt,4*dt)
            return position+velocity*dt,velocity
        desired=None if upper is None else upper[0,self.slots]
        self.q,self.velocity=advance(self.q,self.velocity,desired)
        result[0,self.slots]=self.q
        predicted,velocity=self.q.copy(),self.velocity.copy()
        for k in range(1,len(q)):
            target=None if upper is None else upper[min(k,len(upper)-1),self.slots]
            predicted,velocity=advance(predicted,velocity,target)
            result[k,self.slots]=predicted
        return result,base,quat


def _background_thread():
    # Never inherit a live SCHED_RR controller's priority for inference.
    if hasattr(os, 'sched_setscheduler'):
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    if hasattr(os, 'setpriority'):
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)


class PlannerReference:
    """One background inference at a time; bounded timeline, no blocking in tick."""
    def __init__(self, runtime, config):
        self.runtime=runtime
        root=runtime.root
        settings=json.loads((root/Path(config)).read_text())
        self.settings=settings
        self.stick=StickCommand(speed=float(settings.get('forward_speed',.25)),
            backward_speed=float(settings.get('backward_speed',.175)),
            yaw_rate=float(settings.get('yaw_rate',.35)),
            turn_speed_limit=settings.get('turn_speed_limit'))
        paths=[]
        for key in ('planner','bridge'):
            spec=settings[key];path=(root/spec['path']).resolve()
            if not path.is_relative_to(root.resolve()) or path.stat().st_size!=spec['bytes']:
                raise ValueError('Planner asset path/size mismatch: '+key)
            hasher=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                    hasher.update(chunk)
            if hasher.hexdigest()!=spec['sha256']:raise ValueError('Planner asset hash mismatch: '+key)
            paths.append(path)
        self.steps=runtime.reference.steps
        if self.steps>48 or abs(runtime.model.contract.period-.02)>1e-8:
            raise ValueError('Planner requires a 50 Hz SONIC actor with horizon <= 0.94s')
        self.pipeline=PlannerPipeline(*paths,runtime.model.contract.names,threads=settings.get('threads',4),
            seed=settings.get('seed',1234),
            max_speed=max(.3,self.stick.limit,self.stick.backward_limit))
        self.replan_ticks=round(float(settings.get('replan_interval_s',.4))*50)
        self.pipeline.moving_lookahead=round(float(settings.get('moving_lookahead_s',.44))*50)
        self.pipeline.idle_lookahead=round(float(settings.get('idle_lookahead_s',.04))*50)
        self.pipeline.blend_frames=int(settings.get('blend_frames',8))
        self.pipeline.walk_mode=int(settings.get('walk_mode',1))
        if self.pipeline.walk_mode not in (1,2):raise ValueError('Unsupported walking mode')
        if not 5<=self.replan_ticks<=50 or not 2<=self.pipeline.moving_lookahead<=30 or not 2<=self.pipeline.idle_lookahead<=30 or not 4<=self.pipeline.blend_frames<=12:
            raise ValueError('Invalid planner timing calibration')
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sonic-planner',initializer=_background_thread)
        self.pool.submit(lambda: None).result()
        self.future=None;self.active=False
        self.reset()

    def reset(self):
        # Only called while inactive; an old worker result is discarded before restart.
        self.active=False;self.clip=None;self.cursor=0;self.origin=0;self.last_request=-100
        self.stick.reset();self.fusion=ArmFusion(self.pipeline.names,self.runtime.model.contract.default)
        self.last_command=None;self.heading=None;self.stop_started=None
        self.lead=None;self.lead_cursor=0
        self.control_tick=0;self.future_initial=False;self.reference_yaw=0.

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)

    def tick(self, upper, stick_sample, now, robot_root, *, robot_q, input_enabled=True):
        r=self.runtime
        self.control_tick+=1
        if not self.active:
            if self.future is not None and not self.future.done():
                return r.hold_motion(),r.math.yaw(robot_root),{'planner_state':'waiting_previous'}
            self.future=self.pool.submit(self.pipeline.start,np.asarray(robot_q).copy(),np.asarray(robot_root).copy())
            self.future_initial=True
            self.active=True;self.heading=r.math.yaw(robot_root)
        command=self.stick.update(stick_sample if upper is not None and self.stop_started is None else None,now,enabled=input_enabled)
        if self.future is not None and self.future.done():
            try:
                result=self.future.result()
                mapped,origin=result if self.future_initial else self.pipeline.accept(result,self.cursor)
            except Exception as error:
                raise ValueError('Planner inference failed: '+str(error)) from error
            self.future=None
            # Mapping sees future neighbours, so even a zero-weight G1 frame
            # can acquire a new residual. Fade mapped references from the old
            # timeline at arrival too; this is a Vita-specific continuity layer.
            if self.clip is not None:
                indices=np.clip(origin+np.arange(len(mapped[0]))-self.origin,0,len(self.clip[0])-1)
                weight=np.clip((origin+np.arange(len(mapped[0]))-self.cursor)/getattr(self.pipeline,'blend_frames',8),0,1)
                # Correct the map residual discontinuity at arrival, without
                # crossfading the entire new walking trajectory a second time.
                at=int(np.clip(self.cursor-origin,0,len(mapped[0])-1))
                correction=self.clip[0][indices[at]]-mapped[0][at]
                mapped[0][:]+=(1-weight[:,None])*correction
            initial=self.clip is None
            self.clip,self.origin=mapped,origin
            if initial:
                self.reference_yaw=r.math.yaw(mapped[2][0])
                self.heading-=self.reference_yaw
                self.last_command=Command();self.last_request=self.control_tick
                # Match StartGenerated/BuildTrackingLeadInComposite: lead into
                # frame zero, hold the generated cursor until the lead finishes.
                hold=r.hold_motion()
                target=r.math.Motion(*(x[:1] for x in mapped))
                lead=target.lead_in(hold.q[0],hold.quat[0],r.config['body_profile']['lead_in'])
                # SDK lead_in normalizes heading; retain the planner's frame
                # while the engagement offset aligns it to the physical robot.
                rotation=r.math.yaw_quaternion(self.reference_yaw)
                lead.quat[:]=[r.math.multiply(rotation,q) for q in lead.quat]
                self.lead=(lead.q,lead.base,lead.quat)
                self.lead_cursor=0
                # Mode entry already limits all joint velocities and root rotation.
                # Do not stack a second lead-in in front of that filter.
                if getattr(r,'teleop_modes',None) is not None:self.lead=None
        if self.clip is None:
            return r.hold_motion(),self.heading,{'planner_state':'loading'}
        if self.lead is not None:
            if self.lead_cursor<len(self.lead[0])-1:
                indices=np.minimum(self.lead_cursor+np.arange(self.steps),len(self.lead[0])-1)
                lower=tuple(x[indices].copy() for x in self.lead)
                self.lead_cursor+=1
                return r.math.Motion(*self.fusion.apply(lower,upper)),self.heading,dict(
                    planner_state='lead_in',planner_cursor=self.cursor,planner_lead_frame=self.lead_cursor,
                    planner_lead_frames=len(self.lead[0])-1)
            self.lead=None
        local=self.cursor-self.origin
        remaining=len(self.clip[0])-local
        changed=(self.last_command is None or command.heading!=self.last_command.heading
            or command.turning!=self.last_command.turning
            or (command.speed>0)!=(self.last_command.speed>0)
            or (command.speed>0 and command!=self.last_command))
        periodic=self.control_tick-self.last_request>=self.replan_ticks and (command.speed>0 or command.turning)
        if self.future is None and (changed or periodic):
            self.future=self.pool.submit(self.pipeline.predict,self.cursor,command)
            self.future_initial=False
            self.last_request=self.control_tick;self.last_command=command
        indices=np.minimum(local+np.arange(self.steps),len(self.clip[0])-1)
        lower=tuple(x[indices].copy() for x in self.clip)
        fused=self.fusion.apply(lower,upper)
        self.cursor=min(self.cursor+1,self.origin+len(self.clip[0])-1)
        return r.math.Motion(*fused),self.heading,dict(planner_state='tracking',planner_cursor=self.cursor,
            planner_tail_frames=remaining,planner_command=command.__dict__,planner_replans=self.pipeline.replans,
            planner_inference_ms=self.pipeline.last_inference_ms,planner_busy=self.future is not None,planner_request_tick=self.last_request,upper_reference_held=upper is None,
            upper_reference_state=('tracking' if upper is not None else 'braking'
                if np.max(np.abs(self.fusion.velocity))>1e-6 else 'hold'))
