import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_hands import GripGate, HandGripRunner

class GripTests(unittest.TestCase):
    def test_hold_release_and_reconnect(self):
        g = GripGate()
        self.assertFalse(g.update((1,1,1,1),1))
        self.assertFalse(g.update((1.01,2,0,0),1.01))
        self.assertFalse(g.update((1.02,3,1,1),1.02))
        self.assertTrue(g.update((1.08,4,1,1),1.08))
        self.assertFalse(g.update((1.09,5,1,0),1.09))
        self.assertFalse(g.update((1.09,5,1,1),1.4))
        self.assertFalse(g.update((1.41,6,1,1),1.41))
    def test_frozen_timestamp_and_invalid(self):
        g = GripGate()
        g.update((1,10,0,0),1)
        g.update((1.01,11,1,1),1.01)
        self.assertTrue(g.update((1.07,12,1,1),1.07))
        self.assertFalse(g.update((1.4,12,1,1),1.4))
        for sample in [(1.5,13,float('nan'),1),(1.5,13,2,1),(1.5,9,1,1)]:
            self.assertFalse(g.update(sample,1.5))
    def runner(self):
        self.clients = {}
        config = {s:dict(joint_names=list('abcdef'),open_rad=[0]*6,grasp_rad=[.3]*6,speed_rad_s=[.1]*6) for s in ('left','right')}
        def factory(side, **kw):
            c = NS(state=NS(state_id=1,output_allowed=True),sent=[],closed=False)
            c.wait_info=lambda **kw:NS(hand_type=1,joint_names=list('abcdef'),q_min=[0]*6,q_max=[1]*6,dq_max=[1]*6)
            c.publish=lambda q,dq,**kw:c.sent.append((q,dq))
            c.close=lambda:setattr(c,'closed',True)
            self.clients[side]=c
            return c
        return HandGripRunner(config,None,group='test',factory=factory)
    def test_dual_targets_and_stale_feedback(self):
        r=self.runner()
        r.publish(True,1)
        r.publish(False,1.02)
        for c in self.clients.values():
            self.assertEqual(c.sent[0][0],[.3]*6)
            self.assertEqual(c.sent[1][0],[0]*6)
        with self.assertRaises(RuntimeError):r.publish(True,1.3)
        r.close()
        self.assertTrue(all(c.closed for c in self.clients.values()))
    def test_inhibit_prevents_both_writes(self):
        r=self.runner()
        self.clients['right'].state.output_allowed=False
        with self.assertRaises(RuntimeError):r.publish(True,1)
        self.assertFalse(self.clients['left'].sent)
        r.close()

if __name__ == '__main__': unittest.main()
