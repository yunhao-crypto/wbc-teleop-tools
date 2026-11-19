"""Run existing SDK loopback/ownership/staleness tests with the official OP3 actor.

The inherited transport is simulated; these tests never publish to a robot.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_actor_supermimic
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime


class OP3LiveTests(test_actor_supermimic.SuperMimicLiveTests):
    @classmethod
    def setUpClass(cls):
        cls.runtime=ActorPicoRuntime(ROOT,backend='ort',policy='supermimic_op3')

    def capture_fixture(self, stamps, bodies=None):
        # Existing lifecycle tests use short G1/trackerlab windows. Supply the
        # full OP3 horizon while retaining their gap and bad-fit samples.
        stamps=list(stamps)
        bodies=None if bodies is None else list(bodies)
        if all(b>a for a,b in zip(stamps,stamps[1:])):
            while len(stamps)<self.runtime.reference.steps+5:
                stamps.append(stamps[-1]+20_000_000)
                if bodies is not None:bodies.append(bodies[-1])
        return super().capture_fixture(stamps,bodies)

    def test_online_entry_uses_measured_reference_without_clipping_observation(self):
        # OP3 has a zero knee lower limit. Give this no-clipping regression a
        # positive policy reference bound so the measured zero lies outside it.
        from unittest.mock import patch
        lower=self.runtime.policy.lower.copy()
        lower[self.runtime.model.contract.names.index('left_knee_joint')]=.01
        with patch.object(self.runtime.policy,'lower',lower):
            super().test_online_entry_uses_measured_reference_without_clipping_observation()
