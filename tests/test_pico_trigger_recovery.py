"""High-rate trigger evidence through the real worker and manual-input adapter."""
from pathlib import Path
from collections import deque
from types import SimpleNamespace as NS
import queue,sys,threading,unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import pico_input_process as receive
from deployment.vita_s100_aorta_30dof.pico_aligned_runtime import ManualInput
from deployment.vita_s100_aorta_30dof.pico_realtime import PicoFlow,Feedback
from deployment.vita_s100_aorta_30dof import pico_threepoint_frontend as front

class Clock:
    def __init__(self):self.ns=1_000_000_000
    def monotonic(self):return self.ns/1e9

class Flow:
    def __init__(self,clock):
        self.clock=clock;self.losses=0;self.button_not_before_ns=0;self.last_frame_ns=clock.ns
        self.source_fault=None;self.calibration=object();self.latest=object();self.events=[]
    def reference_ready(self,now):return self.source_fault is None and self.last_frame_ns is not None
    def lose(self,now,reason):
        if self.source_fault is None:
            self.losses+=1;self.source_fault=reason;self.button_not_before_ns=now;self.last_frame_ns=None
    def recover(self):
        self.source_fault=None;self.last_frame_ns=self.clock.ns;self.button_not_before_ns=self.clock.ns

class Receiver:
    def __init__(self,clock,latch=None):self.clock=clock;self.latch=latch;self.sample=(True,1,0.,0.,0.)
    def poll(self,stamp):
        state=dict(state='receiving')
        if self.latch is not None:
            state.update(trigger_event_seq=self.latch.sequence,trigger_event_monotonic_ns=self.latch.event_ns,
                trigger_event_release_monotonic_ns=self.latch.event_release_ns)
        return None,self.sample,state


def real_flow():
    """Real timing/recovery methods, excluding asset/model construction only."""
    flow=PicoFlow.__new__(PicoFlow)
    flow.route='pico_teleop';flow.gate=front.TimingGate();flow.raw=deque()
    flow.toggle=receive.DualTriggerToggle(.6);flow.window=None
    flow.calibration=object();flow.mode='stand';flow.frames=flow.losses=0;flow.events=[]
    flow.last_frame_ns=None;flow.button_not_before_ns=0;flow.source_fault=None;flow.latest=None
    flow.mapper=NS(map=lambda body:(front.anatomical_sites(body)[0],np.tile([1.,0.,0.,0.],(3,1))))
    body=np.zeros((24,7));body[:,6]=1.
    body[1,2]=-.1;body[2,2]=.1;body[15,1]=.7
    feedback=Feedback(np.zeros(28),np.zeros(28),np.array([1.,0.,0.,0.]),np.zeros(3))
    return flow,body,feedback

class TriggerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.flow=Flow(self.clock);self.latch=receive.TriggerEventLatch()
        self.receiver=Receiver(self.clock,self.latch)
        self.runtime=NS(flow=self.flow,route='pico_teleop',last_feedback=None)
        self.manual=ManualInput(self.runtime,self.receiver,clock=self.clock)
        self.packet=0
    def sample(self,seconds,left,right,*,age=0.):
        self.clock.ns=round(seconds*1e9);self.flow.last_frame_ns=self.clock.ns
        self.packet+=1;sample=(True,self.packet,age,left,right);self.receiver.sample=sample
        self.latch.update(sample,self.clock.ns)
    def test_pre_loss_release_cannot_arm_post_recovery_press(self):
        self.sample(1.,0.,0.);self.manual.snapshot()
        self.flow.lose(2_000_000_000,'body timeout');self.clock.ns=2_000_000_000
        self.manual.snapshot();self.clock.ns=3_000_000_000;self.flow.recover()
        self.sample(3.01,1.,1.)
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],0)
        self.sample(3.4,0.,0.);self.sample(3.41,1.,1.)
        result=self.manual.snapshot()
        self.assertEqual(result['trigger_event_seq'],1)
        self.assertEqual(result['trigger_event_monotonic'],3.41)
    def test_press_proof_is_not_replaced_by_later_released_packet(self):
        self.sample(1.,0.,0.);self.sample(1.02,1.,1.)
        self.clock.ns=2_000_000_000;self.flow.lose(self.clock.ns,'fault');self.flow.recover()
        self.sample(2.01,0.,0.)
        self.assertEqual(self.latch.event_release_ns,1_000_000_000)
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],0)
    def test_short_press_survives_parent_poll_coalescing(self):
        self.sample(1.,0.,0.);self.sample(1.002,1.,1.);self.sample(1.004,0.,0.)
        result=self.manual.snapshot()
        self.assertEqual(result['trigger_event_seq'],1)
        self.assertEqual(result['trigger_event_monotonic'],1.002)
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],1)
    def test_loss_invalidates_previously_accepted_event_time(self):
        self.sample(1.,0.,0.);self.sample(1.01,1.,1.);self.manual.snapshot()
        self.clock.ns=2_000_000_000;self.flow.lose(self.clock.ns,'timeout')
        self.assertEqual(self.manual.snapshot()['trigger_event_monotonic'],float('-inf'))
    def test_missing_release_proof_and_missed_edges_are_rejected(self):
        self.latch.sequence=1;self.latch.event_ns=self.clock.ns
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],0)
        self.latch.sequence=3;self.latch.event_release_ns=self.clock.ns
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],0)
        self.sample(1.4,0.,0.);self.sample(1.41,1.,1.)
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],1)
    def test_cached_release_before_recovery_cannot_rearm_fallback(self):
        self.receiver.latch=None;self.clock.ns=3_000_000_000;self.flow.recover()
        self.receiver.sample=(True,1,1.,0.,0.);self.manual.snapshot()
        self.clock.ns+=10_000_000;self.flow.last_frame_ns=self.clock.ns
        self.receiver.sample=(True,2,0.,1.,1.)
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],0)
        self.clock.ns=3_400_000_000;self.flow.last_frame_ns=self.clock.ns
        self.receiver.sample=(True,3,0.,0.,0.);self.manual.snapshot()
        self.clock.ns=3_410_000_000;self.flow.last_frame_ns=self.clock.ns
        self.receiver.sample=(True,4,0.,1.,1.)
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],1)
    def test_cached_packet_keeps_original_release_receipt(self):
        self.sample(1.,0.,0.,age=.02)
        self.latch.update(self.receiver.sample,1_100_000_000)
        self.sample(1.2,1.,1.)
        self.assertEqual(self.latch.event_release_ns,980_000_000)
    def test_worker_forwards_the_actual_arming_release(self):
        class Stop:
            ticks=0
            def is_set(self):return self.ticks>=3
            def wait(self,seconds):self.ticks+=1
        stop=Stop();samples=[(True,1,0.,0.,0.),(True,2,0.,1.,1.),(True,3,0.,0.,0.)]
        sdk=NS(init=lambda:None,close=lambda:None,get_body_pose_snapshot=lambda:(False,0,None),
            get_controller_trigger_snapshot=lambda:samples[stop.ticks])
        clocks=[0]*9;buttons=[0.,0.];status=queue.Queue()
        times=iter([1_000_000_000,1_000_000_000,1_002_000_000,1_002_000_000,1_004_000_000,1_004_000_000])
        with patch.object(receive.importlib,'import_module',return_value=sdk),patch.object(receive.time,'monotonic_ns',side_effect=lambda:next(times)):
            receive._worker(stop,threading.Lock(),bytearray(24*7*4),clocks,buttons,status,None,[])
        self.assertEqual(clocks[6:9],[1,1_002_000_000,1_000_000_000])
        self.assertEqual(buttons,[0.,0.])
        self.assertEqual(status.get()['state'],'receiving')
        self.assertTrue(status.empty())

    def test_queued_body_before_loss_cannot_recover_real_flow_or_lower_boundary(self):
        flow,body,feedback=real_flow()
        self.assertTrue(flow.push(body,1_000_000_000,1_000_000_000,feedback))
        flow.lose(1_120_000_000,'control detected reference timeout')
        self.assertFalse(flow.push(body,1_110_000_000,1_110_000_000,feedback))
        self.assertEqual(flow.button_not_before_ns,1_120_000_000)
        self.assertIsNotNone(flow.source_fault)
        self.assertFalse(flow.reference_ready(1_120_000_000))
        self.assertIsNone(flow.gate.last_stamp,'queued packet must not consume recovery timing state')
        self.assertTrue(flow.push(body,1_130_000_000,1_130_000_000,feedback))
        self.assertEqual(flow.button_not_before_ns,1_130_000_000)
        self.assertTrue(flow.reference_ready(1_130_000_000))

    def test_fault_timestamp_cannot_move_real_flow_button_boundary_backwards(self):
        flow,body,feedback=real_flow()
        flow.button_not_before_ns=2_000_000_000
        flow.lose(1_000_000_000,'queued invalid packet')
        self.assertEqual(flow.button_not_before_ns,2_000_000_000)

    def test_real_recovery_requires_a_release_after_the_new_body_boundary(self):
        flow,body,feedback=real_flow();clock=Clock()
        flow.push(body,clock.ns,clock.ns,feedback)
        flow.lose(1_120_000_000,'timeout')
        self.assertFalse(flow.push(body,1_110_000_000,1_110_000_000,feedback))
        latch=receive.TriggerEventLatch();receiver=Receiver(clock,latch)
        manual=ManualInput(NS(flow=flow,route='pico_teleop',last_feedback=feedback),receiver,clock=clock)
        # This release is newer than the queued packet but older than both
        # the fault boundary and the actual recovered body receipt.
        latch.update((True,1,0.,0.,0.),1_115_000_000)
        flow.push(body,1_130_000_000,1_130_000_000,feedback)
        clock.ns=1_140_000_000;latch.update((True,2,0.,1.,1.),clock.ns)
        self.assertEqual(manual.snapshot()['state'],'ready')
        self.assertEqual(manual.snapshot()['trigger_event_seq'],0)
        self.assertEqual(flow.button_not_before_ns,1_130_000_000)

    def test_event_received_during_poll_is_accepted_with_real_flow_freshness(self):
        flow,body,feedback=real_flow();clock=Clock()
        latch=receive.TriggerEventLatch();receiver=Receiver(clock,latch)
        runtime=NS(flow=flow,route='pico_teleop',last_feedback=feedback,
            base=NS(push=lambda pose,fb:flow.push(*pose,fb)))
        manual=ManualInput(runtime,receiver,clock=clock)
        latch.update((True,1,0.,0.,0.),clock.ns)
        def poll(start_ns):
            clock.ns+=2_000_000
            latch.update((True,2,0.,1.,1.),clock.ns)
            # Exactly what the production poll computes when its copy sees a
            # newer button packet than the caller's supplied clock instant.
            return (body,clock.ns,clock.ns),(True,2,(start_ns-clock.ns)*1e-9,1.,1.),dict(
                state='receiving',trigger_event_seq=latch.sequence,trigger_event_monotonic_ns=latch.event_ns,
                trigger_event_release_monotonic_ns=latch.event_release_ns)
        receiver.poll=poll
        result=manual.snapshot()
        self.assertEqual(result['state'],'ready')
        self.assertEqual(result['trigger_event_seq'],1)
        self.assertEqual(result['trigger_event_monotonic'],1.002)
        self.assertAlmostEqual(result['controller_received_monotonic'],1.002)
        self.assertEqual(manual.source_event_seq,1)

    def test_future_edge_is_not_consumed_and_can_be_retried(self):
        self.latch.sequence=1;self.latch.event_release_ns=999_000_000
        self.latch.event_ns=1_002_000_000
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],0)
        self.assertEqual(self.manual.source_event_seq,0)
        self.clock.ns=1_003_000_000;self.flow.last_frame_ns=self.clock.ns
        self.assertEqual(self.manual.snapshot()['trigger_event_seq'],1)
        self.assertEqual(self.manual.source_event_seq,1)

if __name__=='__main__':unittest.main()
