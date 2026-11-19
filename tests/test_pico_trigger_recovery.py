"""High-rate trigger evidence through the real worker and manual-input adapter."""
from pathlib import Path
from types import SimpleNamespace as NS
import queue,sys,threading,unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import pico_input_process as receive
from deployment.vita_s100_aorta_30dof.pico_aligned_runtime import ManualInput

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

if __name__=='__main__':unittest.main()
