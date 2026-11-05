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

if __name__ == '__main__': unittest.main()
