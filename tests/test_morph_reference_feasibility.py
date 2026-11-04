import unittest
import numpy as np
from test_native_ankle_commands import contracts

class ReferenceFeasibilityTests(unittest.TestCase):
    def test_recorded_bad_reference_becomes_feasible_and_feedback_is_not_mutated(self):
        c=contracts()[2];q=np.tile(c.default,(12,1));before=q.copy()
        for p,r,sign in c.ankle.legs:
            q[:,p]=-.88139116764;q[:,r]=sign*.3
        before=q.copy()
        out,changed=c.ankle.project_reference(q)
        self.assertTrue(changed.all());self.assertTrue(np.array_equal(before,q))
        motor=c.ankle.to_policy(out)[:,c.ankle.slots]
        self.assertGreaterEqual(motor.min(),c.ankle.motor_limits[0])
        self.assertLessEqual(motor.max(),c.ankle.motor_limits[1])
        untouched=[i for i in range(28) if i not in c.ankle.slots]
        self.assertTrue(np.array_equal(out[:,untouched],q[:,untouched]))

    def test_legal_references_unchanged_and_extremes_close(self):
        c=contracts()[2];q=np.tile(c.default,(51,1))
        q[:,c.ankle.slots]=0
        out,changed=c.ankle.project_reference(q)
        self.assertFalse(changed.any());self.assertTrue(np.array_equal(out,q))
        rng=np.random.default_rng(9);q[:,c.ankle.slots]=rng.uniform(-1.5,1.5,(51,4))
        out,_=c.ankle.project_reference(q);motors=c.ankle.to_policy(out)
        self.assertTrue(np.all(motors[:,c.ankle.slots]>=c.ankle.motor_limits[0]))
        self.assertTrue(np.all(motors[:,c.ankle.slots]<=c.ankle.motor_limits[1]))
        for pose,target in zip(out,motors):
            recovered,error=c.ankle.to_virtual(target)
            self.assertLess(error,1e-7)
            np.testing.assert_allclose(recovered[c.ankle.slots],pose[c.ankle.slots],atol=1e-7)

    def test_projected_velocity_uses_correct_trajectory(self):
        c=contracts()[2];q=np.tile(c.default,(60,1))
        for p,r,sign in c.ankle.legs:
            q[:,p]=np.linspace(-.74,-.9,60);q[:,r]=sign*.3
        out,_=c.ankle.project_reference(q)
        v=np.gradient(out,.02,axis=0)
        actual=c.ankle.velocity_to_policy(out,v)
        # On the saturated tail, virtual and motor pitch/roll rates must be zero.
        np.testing.assert_allclose(actual[-10:,c.ankle.slots],0,atol=1e-10)

if __name__=='__main__':unittest.main()
