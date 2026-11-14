"""Reference fusion invariants; no hardware or external model assets needed."""
from pathlib import Path
import json, sys, unittest
from types import SimpleNamespace as NS
from concurrent.futures import Future
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_sonic_planner import (
    ArmFusion,StickCommand,Command,PlannerPipeline,PlannerReference,VITA_NAMES,G1_INDEX,sample,interpolate)

class Tests(unittest.TestCase):
    def test_configured_speeds_exact_maximum_turn_cap_and_release(self):
        cfg=json.loads((ROOT/'config/actor_sonic_planner.json').read_text())
        stick=StickCommand(speed=cfg['forward_speed'],backward_speed=cfg['backward_speed'],
            yaw_rate=cfg['yaw_rate'],turn_speed_limit=cfg['turn_speed_limit'])
        tick=0
        def drive(y,x,n=150):
            nonlocal tick
            for _ in range(n):
                tick+=1;command=stick.update([tick*.02,tick,y,x],tick*.02)
            return command
        drive(0,0,1)
        self.assertAlmostEqual(drive(1,0).speed,.5)
        self.assertAlmostEqual(drive(1,1).speed,.5)
        self.assertEqual(drive(0,0,1).speed,0.)
        back=drive(-1,0);self.assertTrue(back.backward);self.assertAlmostEqual(back.speed,.5)
        self.assertAlmostEqual(drive(-1,-1).speed,.5)
        self.assertEqual(drive(0,0,1).speed,0.)

    def test_half_radian_yaw_limit_ramp_both_directions_and_release(self):
        s=StickCommand(yaw_rate=.5);s.update([0,1,0,0],0)
        for i in range(1,101):
            previous=s.turn_rate;s.update([i*.02,i+1,0,1],i*.02)
            self.assertLessEqual(abs(s.turn_rate-previous),.7*.02+1e-10)
        self.assertAlmostEqual(s.turn_rate,-.5)
        before=s.heading
        for i in range(101,201):command=s.update([i*.02,i+1,0,1],i*.02)
        self.assertAlmostEqual(s.heading-before,-1.)
        frozen=command.heading;stopped=s.update([4.02,202,0,0],4.02)
        self.assertEqual(s.turn_rate,0.);self.assertEqual(stopped.heading,frozen)
        for i in range(203,304):s.update([i*.02,i,0,-1],i*.02)
        self.assertAlmostEqual(s.turn_rate,.5)
        with self.assertRaises(ValueError):StickCommand(yaw_rate=.501)

    def test_release_requests_idle_and_freezes_heading_without_waiting_for_ramp(self):
        s=StickCommand();s.update([0,1,0,0],0)
        for i in range(1,100):s.update([i*.02,i+1,.8,-1],i*.02)
        self.assertGreater(s.speed,0);heading=s.heading
        stopped=s.update([2,101,0,0],2)
        self.assertEqual(stopped.speed,0);self.assertEqual(s.turn_rate,0)
        self.assertEqual(s.heading,heading)
        self.assertEqual(s.update(None,2.02).speed,0)

    def test_legacy_ramp_available_for_stop_ab_test(self):
        s=StickCommand(release_mode='ramp');s.update([0,1,0,0],0)
        for i in range(1,100):s.update([i*.02,i+1,.8,0],i*.02)
        self.assertGreater(s.update([2,101,0,0],2).speed,0)

    def test_sticks_require_center_after_start_and_loss(self):
        stick=StickCommand()
        for i in range(20):
            self.assertEqual(stick.update([i*.02,i+1,1,0],i*.02).speed,0)
        stick.update([.4,21,0,0],.4)
        for i in range(22,80):cmd=stick.update([i*.02,i,1,0],i*.02)
        self.assertEqual(cmd.speed,.25)
        for i in range(80,140):cmd=stick.update(None,i*.02)
        self.assertEqual(cmd.speed,0)
        self.assertEqual(stick.update([3,141,-1,0],3).speed,0)
        stick.update([3.02,142,0,0],3.02)
        for i in range(143,200):cmd=stick.update([i*.02,i,-1,0],i*.02)
        self.assertTrue(cmd.backward)
        self.assertLessEqual(cmd.speed,.25)

    def test_sticks_do_not_refresh_cached_or_regressed_samples(self):
        s=StickCommand();s.update([0,10,0,0],0)
        self.assertFalse(s.update([0,10,1,0],1).speed)
        s.update([1,11,0,0],1)
        s.update([1.02,9,1,1],1.02)
        self.assertFalse(s.armed)
        for malformed in (None,[],[1,1,float('nan'),0],[1,1,2,0],[2,20,0,0]):
            s.update(malformed,1.03);self.assertFalse(s.armed)

    def test_real_sdk_duplicate_receipt_jitter_does_not_disarm_or_refresh(self):
        # Three real interruption pairs, from robot173 session 1794470649001981750.
        for receipt,stamp,jitter,axes in (
            (63503.211012525,1.788668291498447e18,.000035325,(-1.,0.)),
            (63508.216893125,1.788668296522207e18,.00000165,(1.,0.)),
            (63511.4705942,1.7886682997775242e18,.000873075,(0.,-1.))):
            s=StickCommand(yaw_rate=.5)
            s.update([receipt-.02,stamp-10000000,0,0],receipt-.02)
            s.update([receipt,stamp,*axes],receipt)
            for i in range(1,11):
                s.update([receipt+jitter*i,stamp,*axes],receipt+i*.02)
                self.assertTrue(s.armed)
                self.assertEqual(s.receipt,receipt)
            s.update([receipt+.3,stamp,*axes],receipt+.3)
            self.assertFalse(s.armed)
            self.assertEqual(s.speed,0.)
            s.update([receipt+.32,stamp+10000000,*axes],receipt+.32)
            self.assertFalse(s.armed)  # Real timeout still needs neutral.

    def test_entry_gate_observes_neutral_but_never_accumulates_motion(self):
        s=StickCommand()
        s.update([0,1,0,0],0,enabled=False)
        for i in range(1,25):
            c=s.update([i*.02,i+1,-1,1],i*.02,enabled=False)
            self.assertTrue(s.armed);self.assertEqual(c.speed,0.);self.assertEqual(c.heading,0.)
        for i in range(25,35):c=s.update([i*.02,i+1,-1,0],i*.02)
        self.assertGreater(c.speed,0.);self.assertTrue(c.backward)

    def test_quaternion_resample_takes_short_arc(self):
        a=np.zeros(36);a[3]=1;b=a.copy();b[3]=-1;b[0]=1
        middle=interpolate(a,b,.5)
        self.assertAlmostEqual(middle[0],.5)
        np.testing.assert_allclose(middle[3:7],[1,0,0,0])
        np.testing.assert_allclose(sample(np.array([a,b]),[0,.5,1])[:,0],[0,.5,1])

    def test_arms_are_bounded_and_never_modify_legs_waist_or_root(self):
        fusion=ArmFusion(VITA_NAMES,np.zeros(28))
        lower=(np.full((11,28),.2),np.ones((11,3)),np.tile([1,0,0,0],(11,1)))
        upper=np.ones((11,28))*2
        last=fusion.q.copy();last_v=fusion.velocity.copy()
        for _ in range(30):
            q,base,quat=fusion.apply(lower,upper)
            np.testing.assert_array_equal(q[:,:14],lower[0][:,:14])
            self.assertIs(base,lower[1]);self.assertIs(quat,lower[2])
            self.assertLessEqual(abs(fusion.q-last).max(),1.2*.02+1e-12)
            self.assertLessEqual(abs(fusion.velocity-last_v).max(),4*.02+1e-12)
            last=fusion.q.copy();last_v=fusion.velocity.copy()
        for _ in range(30):
            before=fusion.q.copy();velocity=fusion.velocity.copy()
            window,_,_=fusion.apply(lower,None)
            self.assertLessEqual(abs(fusion.q-before).max(),.024+1e-12)
            self.assertLessEqual(abs(fusion.velocity-velocity).max(),.08+1e-12)
            self.assertLessEqual(abs(np.diff(window[:,fusion.slots],axis=0)).max(),.024+1e-12)
        held=fusion.q.copy()
        fusion.apply(lower,None)
        np.testing.assert_array_equal(fusion.q,held)
        q,_,_=fusion.apply(lower,-upper)
        self.assertLessEqual(abs(q[0,fusion.slots]-held).max(),.024+1e-12)

    def test_bridge_window_order_gravity_and_named_remap(self):
        p=PlannerPipeline.__new__(PlannerPipeline);p.order=list(reversed(range(28)));p.g1=G1_INDEX[p.order]
        class Bridge:
            def run(self, outputs, inputs):
                self.features=inputs['window_features']
                return [np.tile(np.arange(28),(len(self.features),1)).astype(np.float32)]
        p.bridge=Bridge();motion=np.zeros((30,36));motion[:,3]=1
        motion[:,7:]=np.arange(29)[None,:]+np.arange(30)[:,None]*.1
        q,_,_=p.map(motion,25)
        np.testing.assert_allclose(p.bridge.features[:,-3:],[ [0,0,-1] ]*30)
        np.testing.assert_allclose(p.bridge.features[10,:29],motion[10,7:]-2/3,atol=2e-6)
        np.testing.assert_allclose(q,motion[:,7+p.g1]+np.array(p.order))

    def test_planner_uses_official_mask_direct_speed_and_fixed_seed(self):
        p=PlannerPipeline.__new__(PlannerPipeline);p.seed=1234;p.allowed_counts=(36,40,44)
        class Model:
            def run(self,outputs,inputs):
                self.inputs=inputs
                q=np.zeros((1,64,36),np.float32);q[:,:,3]=1
                return q,np.array([44],np.int32)
        p.planner=Model();p.infer(np.tile(np.r_[0,0,.8,1,np.zeros(32)],(4,1)),Command(.25,.3,True))
        self.assertAlmostEqual(float(p.planner.inputs['target_vel'][0]),.25,places=6)
        self.assertEqual(p.planner.inputs['mode'][0],1)
        self.assertEqual(p.planner.inputs['random_seed'][0],1234)
        np.testing.assert_array_equal(p.planner.inputs['allowed_pred_num_tokens'],[[0,0,0,1,1,1,0,0,0,0,0]])
        np.testing.assert_array_equal(p.planner.inputs['movement_direction'],-p.planner.inputs['facing_direction'])
        p.infer(np.tile(np.r_[0,0,.8,1,np.zeros(32)],(4,1)),Command(.1))
        self.assertEqual(p.planner.inputs['random_seed'][0],1234)
        self.assertAlmostEqual(float(p.planner.inputs['target_vel'][0]),.1,places=6)
        with self.assertRaises(ValueError):p.infer(np.zeros((4,36)),Command(1))
        p.max_speed=.45
        p.infer(np.tile(np.r_[0,0,.8,1,np.zeros(32)],(4,1)),Command(.43))
        self.assertAlmostEqual(float(p.planner.inputs['target_vel'][0]),.43,places=6)
        p.infer(np.tile(np.r_[0,0,.8,1,np.zeros(32)],(4,1)),Command(.45,0.,True))
        with self.assertRaises(ValueError):p.infer(np.zeros((4,36)),Command(.451))

    def test_official_start_uses_measured_joints_identity_root_and_only_idle(self):
        p=PlannerPipeline.__new__(PlannerPipeline);p.g1=G1_INDEX
        contexts=[];commands=[]
        def infer(context,command):
            contexts.append(context.copy());commands.append(command)
            return np.repeat(context[:1],80,axis=0)
        p.infer=infer;p.map=lambda motion,origin:(motion[:,7:],motion[:,:3],motion[:,3:7])
        # Official startup deliberately uses identity even with measured tilt.
        quat=[np.cos(.4)*np.cos(.1),np.cos(.4)*np.sin(.1),np.sin(.4)*np.sin(.1),np.sin(.4)*np.cos(.1)]
        measured=np.arange(28)*.01;command=Command(.2)
        p.start(measured,quat,command)
        np.testing.assert_allclose(contexts[0][0,7+G1_INDEX],measured,atol=1e-8)
        np.testing.assert_allclose(contexts[0][0,3:7],[1,0,0,0],atol=1e-7)
        self.assertEqual(commands,[Command()]);self.assertEqual(p.replans,0)

    def pending_reference(self, cursor):
        p=PlannerReference.__new__(PlannerReference)
        p.active=True;p.steps=11;p.cursor=cursor;p.origin=0;p.last_request=0
        p.last_command=Command();p.heading=.7;p.stop_started=None
        p.lead=None;p.lead_cursor=0
        p.control_tick=0;p.future_initial=False;p.reference_yaw=0.
        p.pipeline=NS(replans=1,last_inference_ms=20.,accept=lambda result,cursor:result)
        p.stick=NS(update=lambda *args,**kwargs: Command())
        p.fusion=ArmFusion(VITA_NAMES,np.zeros(28))
        def clip(value):
            return (np.full((100,28),value),np.zeros((100,3)),np.tile([1.,0,0,0],(100,1)))
        p.clip=clip(0.)
        p.future=Future();p.future.set_result((clip(.5),0))
        p.runtime=NS(model=NS(contract=NS(default=np.zeros(28))),
                     math=NS(Motion=lambda *values: values))
        return p

    def test_mapped_arrival_preserves_current_reference_then_fades_in(self):
        early=self.pending_reference(1);late=self.pending_reference(5)
        for p in (early,late):p.tick(None,None,1.,[1,0,0,0],robot_q=np.zeros(28))
        for p,arrival in ((early,1),(late,5)):
            np.testing.assert_array_equal(p.clip[0][:arrival+1],0)
            np.testing.assert_array_equal(p.clip[0][arrival+8:],.5)
            self.assertAlmostEqual(p.clip[0][arrival+4,0],.25)

    def test_initial_loading_preserves_engagement_heading(self):
        p=self.pending_reference(0);p.clip=None;p.future=Future()
        p.runtime.hold_motion=lambda:'stand'
        motion,heading,diag=p.tick(None,None,1.,[1,0,0,0],robot_q=np.zeros(28))
        self.assertEqual((motion,heading,diag['planner_state']),('stand',.7,'loading'))

    def test_lead_in_holds_generated_cursor_and_targets_only_first_frame(self):
        p=self.pending_reference(0);p.clip=None
        p.future_initial=True
        from deployment.vita_s100_aorta_30dof.actor_pico_runtime import sdk_math
        p.runtime.config=json.loads((ROOT/'config/actor_pico.json').read_text())
        p.runtime.math=sdk_math(ROOT,p.runtime.config)
        p.runtime.hold_motion=lambda:p.runtime.math.Motion.hold(np.zeros(28))
        for k in range(21):  # round((.5 / 1.2) / .02)
            motion,_,diag=p.tick(None,None,k*.02,[1,0,0,0],robot_q=np.zeros(28))
            self.assertEqual(p.cursor,0);self.assertEqual(diag['planner_state'],'lead_in')
            expected=.5*(.5-.5*np.cos(np.pi*k/21))
            np.testing.assert_allclose(motion.q[0,:14],expected)
        # No new inference is needed to leave lead-in; next frame is clip frame 0.
        p.last_request=0
        motion,_,diag=p.tick(None,None,.42,[1,0,0,0],robot_q=np.zeros(28))
        self.assertEqual(p.cursor,1);self.assertEqual(diag['planner_state'],'tracking')
        np.testing.assert_allclose(motion.q[0,:14],.5)

    def test_replan_uses_two_frame_context_and_preserves_bridge_history(self):
        p=PlannerPipeline.__new__(PlannerPipeline);p.origin=0;p.replans=0
        p.motion=np.zeros((120,36));p.motion[:,3]=1;p.motion[:,0]=np.arange(120)*.01
        def infer(context,command):
            self.assertAlmostEqual(context[0,0],.32)
            seq=np.repeat(context[:1],80,axis=0);seq[:,0]+=np.arange(80)*.02
            return seq
        p.infer=infer;p.map=lambda motion,origin:(motion[:,7:],motion[:,:3],motion[:,3:7])
        previous=p.motion.copy();mapped,origin=p.replan(30,Command(.2))
        self.assertEqual(origin,23);self.assertEqual(len(p.motion),89)
        np.testing.assert_array_equal(p.motion[:10],previous[23:33])

    def test_late_prediction_blends_from_arrival_not_past_generation(self):
        p=PlannerPipeline.__new__(PlannerPipeline);p.origin=0;p.replans=0
        p.motion=np.zeros((100,36));p.motion[:,3]=1
        p.map=lambda motion,origin:(motion[:,7:],motion[:,:3],motion[:,3:7])
        new=p.motion[:60].copy();new[:,7:]=1
        mapped,origin=p.accept((new,2),6)
        np.testing.assert_array_equal(mapped[0][:7],0)
        np.testing.assert_array_equal(mapped[0][14:],1)
        self.assertAlmostEqual(mapped[0][10,0],.5)
        with self.assertRaisesRegex(TimeoutError,'expired'):p.accept((new,0),100)

    def test_idle_holds_tail_without_periodic_replanning(self):
        p=self.pending_reference(99);p.future=None
        for k in range(120):
            motion,_,diag=p.tick(None,None,k*.02,[1,0,0,0],robot_q=np.zeros(28))
            self.assertEqual(p.cursor,99);self.assertIsNone(p.future)
            self.assertEqual(diag['planner_tail_frames'],1)
            np.testing.assert_array_equal(motion[0],0)

    def test_cpu_direct_command_keeps_neutral_interlock_and_immediate_release(self):
        s=StickCommand(speed=.375,backward_speed=.4,direct_speed=True)
        self.assertEqual(s.update([0,1,1,0],0).speed,0)
        s.update([.02,2,0,0],.02)
        self.assertEqual(s.update([.04,3,1,0],.04).speed,.375)
        back=s.update([.06,4,-1,0],.06)
        self.assertTrue(back.backward);self.assertEqual(back.speed,.4)
        self.assertEqual(s.update([.08,5,0,0],.08).speed,0)
        self.assertEqual(s.update(None,.10).speed,0);self.assertFalse(s.armed)

    def test_cpu_lookahead_and_late_arrival_preserve_seam(self):
        p=PlannerPipeline.__new__(PlannerPipeline)
        p.profile='latency_reference';p.moving_lookahead=18;p.idle_lookahead=18
        p.blend_frames=8;p.origin=0;p.replans=0
        p.motion=np.zeros((120,36));p.motion[:,3]=1;p.motion[:,0]=np.arange(120)*.01
        def infer(context,command):
            self.assertAlmostEqual(context[0,0],.48)
            seq=np.repeat(context[:1],80,axis=0);seq[:,7:]=1
            return seq
        p.infer=infer;p.map=lambda motion,origin:(motion[:,7:],motion[:,:3],motion[:,3:7])
        prediction=p.predict(30,Command(.2));self.assertEqual(prediction[1],48)
        mapped,origin=p.accept(prediction,55)
        np.testing.assert_array_equal(mapped[0][:8],0)
        np.testing.assert_array_equal(mapped[0][15:],1)
        self.assertEqual(origin,48)

    def test_cpu_replan_interval_never_overlaps_pending_work(self):
        p=self.pending_reference(30);p.future=None;p.pipeline.profile='latency_reference'
        p.pipeline.blend_frames=8;p.replan_ticks=20;p.pipeline.predict=lambda *args:None
        p.last_command=Command(.2);p.stick=NS(update=lambda *args,**kwargs:Command(.2))
        submitted=[]
        def submit(*args):
            submitted.append(args);return Future()
        p.pool=NS(submit=submit)
        for k in range(19):p.tick(None,None,k*.02,[1,0,0,0],robot_q=np.zeros(28))
        self.assertEqual(len(submitted),0)
        p.tick(None,None,.38,[1,0,0,0],robot_q=np.zeros(28))
        self.assertEqual(len(submitted),1)
        p.stick=NS(update=lambda *args,**kwargs:Command())
        for k in range(30):p.tick(None,None,.4+k*.02,[1,0,0,0],robot_q=np.zeros(28))
        self.assertEqual(len(submitted),1)

if __name__=='__main__':unittest.main()
