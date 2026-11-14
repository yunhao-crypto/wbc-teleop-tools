"""R3 mode switching through actor stand, without changing controller ownership."""
import math
import numpy as np
from .actor_sonic_planner import ArmFusion, STICK_DEADZONE


def active_planner(runtime):
    modes=getattr(runtime,'teleop_modes',None)
    return getattr(runtime,'planner',None) if modes is None or modes.mode=='joystick' else None


class EntryBlend:
    """Finite planner entry; bounded human-pose following. Outputs use world yaw."""
    def __init__(self,runtime,q,root,*,planner_entry=False):
        self.planner_entry=planner_entry;self.start_q=np.array(q,copy=True);self.start_root=np.array(root,copy=True)
        self.r=runtime;self.filter=ArmFusion(runtime.model.contract.names,q)
        self.filter.slots=np.arange(len(q));self.filter.q=np.array(q,copy=True);self.filter.velocity=np.zeros(len(q))
        self.quat=np.array(root,copy=True);self.quiet=0;self.ticks=0
        self.legs=np.array([i for i,n in enumerate(runtime.model.contract.names) if any(k in n for k in ('hip','knee','ankle','waist'))])

    def apply(self,motion,heading):
        r=self.r;dt=r.model.contract.period
        if self.planner_entry:
            # A finite C2 handover, not a low-pass on the ongoing gait. At 0.5 s
            # the lower-body trajectory and all its future frames are unmodified.
            self.ticks+=1
            t=np.clip((self.ticks+np.arange(len(motion.q)))*dt/.5,0.,1.)
            weight=t**3*(10-15*t+6*t*t)
            positions=self.start_q+weight[:,None]*(motion.q-self.start_q)
            rotation=r.math.yaw_quaternion(heading)
            targets=[r.math.multiply(rotation,q) for q in motion.quat]
            quats=[r.math.slerp(self.start_root,target,float(a)) for target,a in zip(targets,weight)]
            self.filter.q=positions[0].copy()
            return r.math.Motion(positions,motion.base,quats),0.,self.ticks*dt>=.5
        positions,base,_=self.filter.apply((motion.q,motion.base,motion.quat),motion.q,dt=dt)
        rotation=r.math.yaw_quaternion(heading)
        targets=np.array([r.math.multiply(rotation,q) for q in motion.quat]);quats=[];previous=self.quat.copy()
        for target in targets:
            angle=2*np.arccos(np.clip(abs(np.dot(previous,target)),0,1))
            previous=r.math.slerp(previous,target,min(1.,.5*dt/max(angle,1e-12)))
            quats.append(previous)
        self.quat=np.array(quats[0]);self.ticks+=1
        delta=np.max(abs(positions[:,self.legs]-motion.q[:,self.legs]))
        root_error=2*np.arccos(np.clip(abs(np.dot(self.quat,targets[0])),0,1))
        self.quiet=self.quiet+1 if delta<.002 and root_error<.005 else 0
        done=self.ticks>=25 and self.quiet>=10
        return r.math.Motion(positions,base,quats),0.,done


class TeleopModes:
    def __init__(self,runtime):
        self.r=runtime;self.mode='full_body';self.pending=None;self.stage='ready';self.last_event=0
        self.started=0.;self.quiet_since=None;self.entry=None;self.arm_follow=None;self.upper_follow=None;self.last_q=None;self.last_root=None
        self.last_stick_stamp=0.;self.rearm_after_stamp=0.;self.needs_center=True
        self.legs=np.array([i for i,n in enumerate(runtime.model.contract.names) if any(k in n for k in ('hip','knee','ankle','waist'))])

    @staticmethod
    def stick_status(snapshot,now):
        info=dict(fresh=False,centered=False,right_y=None,right_x=None,axis_age_s=None,packet_age_s=None)
        try:
            received,stamp,y,x=map(float,snapshot['planner_sticks'])
            packet_received=float(snapshot['mode_received_monotonic'])
            if not np.isfinite([received,stamp,y,x,now,packet_received]).all():return info
            age,packet_age=now-received,now-packet_received
            fresh=bool(stamp>0 and 0<=age<=.25 and 0<=packet_age<=.25 and max(abs(x),abs(y))<=1)
            info.update(fresh=fresh,centered=fresh and max(abs(x),abs(y))<=STICK_DEADZONE,
                right_y=y,right_x=x,axis_age_s=age,packet_age_s=packet_age)
        except (KeyError,TypeError,ValueError):pass
        return info

    @staticmethod
    def centered(snapshot,now):
        return TeleopModes.stick_status(snapshot,now)['centered']

    def stand_status(self,q,dq,root,gyro):
        error=np.abs(np.asarray(q)[self.legs]-self.r.model.contract.default[self.legs])
        index=int(np.argmax(error));joint_error=float(error[index])
        speed=float(np.max(abs(np.asarray(dq)[self.legs])));angular=float(np.max(abs(gyro)))
        cosine=float(self.r.math.rotation(root)[2,2])
        return dict(ready=bool(joint_error<.25 and speed<.15 and angular<.1 and cosine>math.cos(.25)),
            joint=self.r.model.contract.names[self.legs[index]],joint_error_rad=joint_error,
            joint_speed_radps=speed,gyro_radps=angular,tilt_rad=float(np.arccos(np.clip(cosine,-1,1))))

    def neutral(self,q,dq,root,gyro):
        return self.stand_status(q,dq,root,gyro)['ready']

    def request(self,snapshot,now,*,tracking,q,dq,root,gyro):
        event=int(snapshot.get('mode_event_seq',self.last_event))
        if event==self.last_event:return None
        previous=self.last_event;self.last_event=event
        reason=None;stick=self.stick_status(snapshot,now)
        if event!=previous+1 or not 0<=now-float(snapshot.get('mode_event_monotonic',-math.inf))<=.25:reason='stale_or_missed_edge'
        elif not tracking:reason='not_tracking'
        elif self.pending is not None or self.stage in ('braking','stand') or (self.mode=='joystick' and self.stage!='ready'):reason='transition_busy'
        elif not stick['fresh']:reason='controller_input_stale_or_invalid'
        elif snapshot.get('state')!='ready' or not 0<=now-float(snapshot.get('received_monotonic',-math.inf))<=self.r.config['pico_hold_age_s']:reason='pico_reference_not_fresh'
        if reason:return dict(record_type='teleop_mode_rejected',reason=reason,event_seq=event,stick=stick)
        # R3 requests a controlled return to stand. Check neutral AFTER that
        # transition, not while the robot is still following a human pose.
        self.pending='joystick' if self.mode=='full_body' else 'full_body'
        self.stage='braking';self.started=now;self.quiet_since=None
        return dict(record_type='teleop_mode_requested',target=self.pending,event_seq=event,stick=stick,
            next='松开右摇杆回中；先收步站立，再切换模式。')

    def cancel(self):
        self.pending=None;self.stage='ready';self.quiet_since=None;self.entry=None;self.arm_follow=None

    def paused(self):
        self.stage='stand' if self.pending else 'ready';self.entry=None;self.arm_follow=None;self.quiet_since=None;self.last_wait_report=None

    def ready(self,snapshot,now,*,q,dq,root,gyro,preparation_done,motion):
        if self.pending is None or self.stage!='stand':return False
        if now-self.started>10 or not self.stick_status(snapshot,now)['fresh']:
            print('TELEOP_MODE_CANCELLED: 等待回中/站立超时或手柄数据过期。',flush=True)
            self.cancel();return False
        # A thumb press can deflect the stick. Accept the stop intent first;
        # switch only after fresh centered input and stable standing feedback.
        stand=self.stand_status(q,dq,root,gyro)
        ok=self.centered(snapshot,now) and preparation_done and stand['ready']
        if motion is None:ok=False
        elif self.pending=='full_body':
            ok=ok and np.max(abs(motion.q[:,self.legs]-self.r.model.contract.default[self.legs]))<.25
        if not ok and (getattr(self,'last_wait_report',None) is None or now-self.last_wait_report>=1.):
            self.last_wait_report=now
            print('TELEOP_WAIT_STAND: '+str(dict(stand,preparation_done=preparation_done,
                stick=self.stick_status(snapshot,now),pico_reference_ready=motion is not None)),flush=True)
        self.quiet_since=(now if self.quiet_since is None else self.quiet_since) if ok else None
        if self.quiet_since is not None and now-self.quiet_since>=.5:
            self.mode=self.pending;self.pending=None;self.stage='entry';return True
        return False

    def follow_upper(self,motion,upper):
        """One arm filter for the whole Pico session, including R3 stand/entry."""
        if self.upper_follow is None:return motion
        motion=self.r.math.Motion(*self.upper_follow.apply(
            (motion.q,motion.base,motion.quat),upper,dt=self.r.model.contract.period))
        if self.last_q is not None:
            self.last_q[self.upper_follow.slots]=motion.q[0,self.upper_follow.slots]
        return motion

    def engage(self,q,root):
        if self.upper_follow is None:
            self.upper_follow=ArmFusion(self.r.model.contract.names,q)
        self.entry=EntryBlend(self.r,q,root,planner_entry=self.mode=='joystick');self.arm_follow=None;self.stage='entry'
        self.needs_center=True;self.rearm_after_stamp=self.last_stick_stamp
        if getattr(self.r,'planner',None) is not None:self.r.planner.stick.reset()

    def filter(self,motion,heading):
        if self.entry is not None:
            motion,heading,done=self.entry.apply(motion,heading)
            # Planner handover ends exactly at 0.5 s. Human-pose following
            # retains its limiter; it must never filter a generated walking gait.
            if self.stage=='entry' and self.entry.ticks*self.r.model.contract.period>=.5:
                self.stage='ready'
                print('TELEOP_READY: '+self.mode+'; 参考交接完成；右杆已回中则可直接控制。',flush=True)
            if done:
                # Arm motion must not hold the mode gate indefinitely. Keep the
                # emitted arm position/velocity when handing off the limiter.
                arms=ArmFusion(self.r.model.contract.names,self.r.model.contract.default)
                arms.q=self.entry.filter.q[arms.slots].copy()
                arms.velocity=self.entry.filter.velocity[arms.slots].copy()
                self.entry=None
                if self.mode=='full_body':self.arm_follow=arms
                else:self.r.planner.fusion=arms
        elif self.arm_follow is not None:
            motion=self.r.math.Motion(*self.arm_follow.apply((motion.q,motion.base,motion.quat),motion.q,dt=self.r.model.contract.period))
        self.last_q=motion.q[0].copy()
        self.last_root=self.r.math.multiply(self.r.math.yaw_quaternion(heading),motion.quat[0])
        return motion,heading

    def stick_sample(self,snapshot,now):
        sample=snapshot.get('planner_sticks')
        try:
            if len(sample)==4 and np.isfinite(sample).all():self.last_stick_stamp=max(self.last_stick_stamp,float(sample[1]))
        except (TypeError,ValueError):return None
        if self.needs_center:
            if self.centered(snapshot,now) and float(sample[1])>self.rearm_after_stamp:self.needs_center=False
            else:return None
        return sample
