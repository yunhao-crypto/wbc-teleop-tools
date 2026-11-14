"""Mode-switch invariants, actual SDK math and native packet parser; no robot I/O."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_pico_buttons import RightStickToggle
from deployment.vita_s100_aorta_30dof.actor_teleop_modes import TeleopModes,EntryBlend,active_planner


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.r=ActorPicoRuntime(ROOT,backend='ort',reference_source='npz')

    def setUp(self):
        self.m=TeleopModes(self.r);self.q=self.r.model.contract.default.copy();self.zero=np.zeros(28)
        self.root=np.array([1.,0,0,0]);self.gyro=np.zeros(3)

    def snapshot(self,t=1.,event=1,x=0.,stamp=100):
        return dict(state='ready',received_monotonic=t,planner_sticks=[t,stamp,0.,x],mode_event_seq=event,
            mode_event_monotonic=t,mode_received_monotonic=t)

    def request(self,s,t=1.):
        return self.m.request(s,t,tracking=True,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro)

    def test_right_click_requires_release_and_rejects_both_clicks_and_replays(self):
        toggle=RightStickToggle()
        self.assertFalse(toggle.update((True,1,0.,0.,0.,False,True),0.))
        self.assertFalse(toggle.update((True,2,0.,0.,0.,False,False),.1))
        self.assertTrue(toggle.update((True,3,0.,0.,0.,False,True),.2))
        self.assertFalse(toggle.update((True,3,0.,0.,0.,False,True),.7))
        self.assertFalse(toggle.update((True,4,0.,0.,0.,False,True),.8))
        toggle.update((True,5,0.,0.,0.,False,False),1.)
        self.assertFalse(toggle.update((True,6,0.,0.,0.,True,True),1.1))
        self.assertFalse(toggle.update((True,7,0.,0.,0.,False,True),1.2))

    def test_input_loss_disarms_click_until_fresh_release(self):
        t=RightStickToggle();t.update((True,1,0.,0.,0.,False,False),0.)
        self.assertFalse(t.update((True,2,.3,0.,0.,False,True),1.))
        self.assertFalse(t.update((True,3,0.,0.,0.,False,True),1.1))
        t.update((True,4,0.,0.,0.,False,False),1.2)
        self.assertTrue(t.update((True,5,0.,0.,0.,False,True),1.3))

    def test_default_full_body_and_switch_goes_through_stand(self):
        self.assertEqual(self.m.mode,'full_body')
        self.assertEqual(self.request(self.snapshot())['record_type'],'teleop_mode_requested')
        self.assertEqual(self.m.mode,'full_body');self.assertEqual(self.m.stage,'braking')
        self.assertEqual(self.request(self.snapshot(1.4,2),1.4)['reason'],'transition_busy')
        self.m.paused();motion=self.r.math.Motion(np.tile(self.q,(11,1)),np.zeros((11,3)),np.tile(self.root,(11,1)))
        for i in range(25):
            t=2+i*.02
            self.assertFalse(self.m.ready(self.snapshot(t,2),t,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro,preparation_done=True,motion=motion))
        self.assertTrue(self.m.ready(self.snapshot(2.52,2),2.52,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro,preparation_done=True,motion=motion))
        self.assertEqual(self.m.mode,'joystick');self.assertEqual(self.m.stage,'entry')

    def test_offcenter_and_nonneutral_stop_requests_accepted_but_stale_rejected(self):
        self.assertEqual(self.request(self.snapshot(x=.4))['record_type'],'teleop_mode_requested')
        self.assertIsNone(self.request(self.snapshot(x=0)))
        self.m=TeleopModes(self.r);s=self.snapshot();s['received_monotonic']=0
        self.assertEqual(self.request(s)['reason'],'pico_reference_not_fresh')
        self.m=TeleopModes(self.r);self.q[0]+=.5
        self.assertEqual(self.request(self.snapshot())['record_type'],'teleop_mode_requested')
        self.m.paused()
        self.assertFalse(self.m.ready(self.snapshot(1.5),1.5,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro,preparation_done=True,motion=self.r.hold_motion()))
        self.assertEqual(self.m.mode,'full_body')

    def test_offcenter_press_waits_for_release_before_selecting_mode(self):
        event=self.request(self.snapshot(x=.4))
        self.assertEqual(event['record_type'],'teleop_mode_requested')
        self.assertEqual(event['stick']['right_x'],.4);self.assertFalse(event['stick']['centered'])
        self.m.paused();motion=self.r.hold_motion()
        for i in range(50):
            now=1.02+i*.02
            self.assertFalse(self.m.ready(self.snapshot(now,x=.4),now,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro,preparation_done=True,motion=motion))
            self.assertEqual(self.m.mode,'full_body');self.assertEqual(self.m.pending,'joystick')
        for i in range(26):
            now=2.1+i*.02
            selected=self.m.ready(self.snapshot(now,x=.10),now,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro,preparation_done=True,motion=motion)
        self.assertTrue(selected);self.assertEqual(self.m.mode,'joystick')

    def test_stale_controller_diagnostics_and_invalid_numbers_remain_rejected(self):
        s=self.snapshot();s['mode_received_monotonic']=0.
        event=self.request(s)
        self.assertEqual(event['reason'],'controller_input_stale_or_invalid')
        self.assertEqual(event['stick']['packet_age_s'],1.)
        s=self.snapshot();s['planner_sticks'][2]=float('nan')
        status=self.m.stick_status(s,1.);self.assertFalse(status['fresh'])
        json.dumps(status,allow_nan=False)

    def test_real_robot_rejected_pose_can_request_stand_then_switch_on_recorded_stand(self):
        fixture=json.loads((ROOT/'tests/fixtures/actor_pico/r3_stand_rejection_20261112.json').read_text())
        self.assertEqual(fixture['joint_names'],list(self.r.model.contract.names))
        for rejected in fixture['rejected_poses']:
            self.m=TeleopModes(self.r)
            def args(row):return dict(q=np.array(row['q']),dq=np.array(row['dq']),root=np.array(row['root']),gyro=np.array(row['gyro']))
            self.assertFalse(self.m.neutral(**args(rejected)))
            result=self.m.request(self.snapshot(),1.,tracking=True,**args(rejected))
            self.assertEqual(result['record_type'],'teleop_mode_requested')
            self.assertEqual(self.m.mode,'full_body');self.m.paused()
            for i in range(50):
                now=1.02+i*.02
                self.assertFalse(self.m.ready(self.snapshot(now),now,preparation_done=True,motion=self.r.hold_motion(),**args(rejected)))
            selected=False
            for i,row in enumerate(fixture['standing_feedback']):
                now=2.5+i*.02
                selected=self.m.ready(self.snapshot(now),now,preparation_done=True,motion=self.r.hold_motion(),**args(row))
                if selected:break
            self.assertTrue(selected);self.assertEqual(self.m.mode,'joystick')

    def test_loss_during_pending_switch_cancels_auto_reengagement(self):
        self.request(self.snapshot());self.m.paused()
        self.assertFalse(self.m.ready(self.snapshot(),2.,q=self.q,dq=self.zero,root=self.root,gyro=self.gyro,preparation_done=True,motion=None))
        self.assertIsNone(self.m.pending);self.assertEqual(self.m.mode,'full_body')

    def test_new_neutral_packet_required_after_handover(self):
        self.m.mode='joystick';self.m.rearm_after_stamp=100
        self.assertIsNone(self.m.stick_sample(self.snapshot(stamp=100),1.))
        self.assertIsNone(self.m.stick_sample(self.snapshot(stamp=101,x=.5),1.))
        self.assertIsNotNone(self.m.stick_sample(self.snapshot(stamp=102),1.))
        self.assertIsNotNone(self.m.stick_sample(self.snapshot(stamp=103,x=.5),1.))

    def test_entry_bounds_joint_speed_acceleration_and_world_root(self):
        blend=EntryBlend(self.r,self.q,self.root);old=self.q.copy();velocity=np.zeros(28);root=self.root.copy()
        target=self.r.math.Motion(np.tile(self.q+1,(11,1)),np.zeros((11,3)),np.tile(self.r.math.yaw_quaternion(.7),(11,1)))
        action=self.r.policy.last_action.copy()
        for _ in range(150):
            motion,heading,done=blend.apply(target,.3)
            v=(motion.q[0]-old)/.02
            self.assertLessEqual(abs(v).max(),1.2+1e-9)
            self.assertLessEqual(abs(v-velocity).max(),.08+1e-9)
            angle=2*np.arccos(np.clip(abs(np.dot(root,motion.quat[0])),0,1))
            self.assertLessEqual(angle,.01+1e-8);self.assertEqual(heading,0.)
            old=motion.q[0];velocity=v;root=motion.quat[0]
        np.testing.assert_array_equal(self.r.policy.last_action,action)

    def test_continuously_moving_arms_do_not_block_entry_or_jump_at_handoff(self):
        self.m.engage(self.q,self.root)
        old=self.q.copy();velocity=np.zeros(28)
        arms=[i for i,n in enumerate(self.r.model.contract.names) if any(k in n for k in ('shoulder','elbow','wrist'))]
        for tick in range(120):
            target=np.tile(self.q,(11,1));target[:,arms]+=1.2*np.sin(.15*tick)+.8
            motion=self.r.math.Motion(target,np.zeros((11,3)),np.tile(self.root,(11,1)))
            motion,_=self.m.filter(motion,0.)
            v=(motion.q[0]-old)/.02
            self.assertLessEqual(abs(v[arms]).max(),1.2+1e-8)
            self.assertLessEqual(abs(v[arms]-velocity[arms]).max(),.08+1e-8)
            old=motion.q[0];velocity=v
            if tick>=25:self.assertEqual(self.m.stage,'ready')
        self.assertIsNotNone(self.m.arm_follow)
        self.assertEqual(self.request(self.snapshot())['record_type'],'teleop_mode_requested')

    def test_fullbody_standing_can_switch_while_reference_still_catching_up(self):
        self.m.engage(self.q,self.root)
        self.assertEqual(self.m.stage,'entry')
        self.assertEqual(self.request(self.snapshot())['record_type'],'teleop_mode_requested')

    def test_joystick_entry_transfers_filtered_arms_without_blocking_stick(self):
        from unittest.mock import patch
        from deployment.vita_s100_aorta_30dof.actor_sonic_planner import ArmFusion,StickCommand
        planner=NS(stick=StickCommand(),fusion=ArmFusion(self.r.model.contract.names,self.q))
        with patch.object(self.r,'planner',planner,create=True):
            self.m.mode='joystick';self.m.engage(self.q,self.root)
            for tick in range(30):
                target=np.tile(self.q,(11,1))
                target[:,planner.fusion.slots]+=np.sin(tick*.2)
                lower=planner.fusion.apply((target,np.zeros((11,3)),np.tile(self.root,(11,1))),target)
                motion,_=self.m.filter(self.r.math.Motion(*lower),0.)
                np.testing.assert_allclose(motion.q[0,self.m.legs],self.q[self.m.legs])
            self.assertEqual(self.m.stage,'ready');self.assertIsNone(self.m.entry)
            self.assertIsNotNone(self.m.stick_sample(self.snapshot(stamp=200),1.))
            self.assertIsNotNone(self.m.stick_sample(self.snapshot(stamp=201,x=.8),1.))

    def test_half_second_planner_handover_finishes_without_distorting_gait(self):
        from unittest.mock import patch
        from deployment.vita_s100_aorta_30dof.actor_sonic_planner import ArmFusion,StickCommand
        planner=NS(stick=StickCommand(),fusion=ArmFusion(self.r.model.contract.names,self.q))
        with patch.object(self.r,'planner',planner,create=True):
            self.m.mode='joystick';self.m.engage(self.q,self.root)
            target=self.r.math.Motion(np.tile(self.q+.05,(11,1)),np.zeros((11,3)),np.tile(self.root,(11,1)))
            previous=self.q.copy();velocity=np.zeros(28)
            for tick in range(80):
                motion,_=self.m.filter(target,0.)
                v=(motion.q[0]-previous)/.02
                self.assertLessEqual(abs(v).max(),1.2+1e-8)
                self.assertLessEqual(abs(v-velocity).max(),.08+1e-8)
                previous=motion.q[0];velocity=v
                if tick<24:self.assertEqual(self.m.stage,'entry')
                else:self.assertEqual(self.m.stage,'ready')
                if tick==24:
                    self.assertIsNone(self.m.entry)
                    np.testing.assert_allclose(motion.q[0,self.m.legs],target.q[0,self.m.legs])
                    self.assertIsNotNone(self.m.stick_sample(self.snapshot(stamp=200),1.))
                    self.assertIsNotNone(self.m.stick_sample(self.snapshot(stamp=201,x=.8),1.))

    def test_r3_keeps_one_upper_filter_through_stand_and_both_entries(self):
        self.m.engage(self.q,self.root)
        arms=self.m.upper_follow.slots
        target=self.r.hold_motion().q.copy();target[:,arms]+=.4
        for _ in range(200):self.m.follow_upper(self.r.hold_motion(),target)
        identity=self.m.upper_follow;old=identity.q.copy()
        self.m.pending='joystick';self.m.paused()
        for _ in range(30):
            motion=self.m.follow_upper(self.r.hold_motion(),target)
            np.testing.assert_allclose(motion.q[0,arms],old,atol=1e-7)
            np.testing.assert_array_equal(motion.q[0,self.m.legs],self.q[self.m.legs])
        self.m.mode='joystick';self.m.engage(self.q,self.root)
        self.assertIs(self.m.upper_follow,identity)
        motion=self.m.follow_upper(self.r.hold_motion(),target)
        np.testing.assert_allclose(motion.q[0,arms],old,atol=1e-7)
        self.m.pending='full_body';self.m.paused();self.m.mode='full_body'
        self.m.engage(self.q,self.root)
        self.assertIs(self.m.upper_follow,identity)
        motion=self.m.follow_upper(self.r.hold_motion(),target)
        np.testing.assert_allclose(motion.q[0,arms],old,atol=1e-7)

    def test_native_right_axes_and_clicks_are_same_packet(self):
        program=r'''#include "controller_trigger_snapshot.h"
#include <cassert>
int main(){auto j=nlohmann::json::parse(R"({"timeStampNs":123,"Controller":{"left":{"axisY":0.9,"axisClick":false},"right":{"axisY":0.2,"axisX":-0.3,"axisClick":true}}})");
auto s=parseControllerTeleopSnapshot(j);assert(s.available && s.left==.2 && s.right==-.3 && s.right_click && !s.left_click && s.packet_timestamp_ns==123);
j["Controller"]["right"]["axisClick"]=1;bool failed=false;try{parseControllerTeleopSnapshot(j);}catch(...){failed=true;}assert(failed);
j["Controller"]["right"]["axisClick"]=false;j["Controller"]["right"]["axisX"]=2;failed=false;try{parseControllerTeleopSnapshot(j);}catch(...){failed=true;}assert(failed);
}'''
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'check.cpp';source.write_text(program);binary=Path(tmp)/'check'
            subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'third_party/pico_binding/include'),'-I'+str(ROOT/'third_party/pico_binding/bindings'),str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)


if __name__=='__main__':unittest.main()
