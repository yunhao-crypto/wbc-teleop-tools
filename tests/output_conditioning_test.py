"""Host-only inert output checks; no SDK initialization or robot communication."""
import builtins
import dataclasses
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'packages/state_schema'))
from deployment.common.policy_runtime import PolicyStep
from deployment.vita_s100_aorta_30dof.command_guard import CommandEnvelope
from deployment.vita_s100_aorta_30dof.command_conditioning import VitaCommandAdapter, InitialPoseError
from deployment.vita_s100_aorta_30dof.onboard_output import InertCommandSink, serialize_inert_lowcmd
from deployment.vita_s100_aorta_30dof.policy_contract import VITA_ACTION_TO_COMMAND_SLOTS
from deployment.vita_s100_aorta_30dof.robot_adapter import VITA_CONTROLLER_JOINT_NAMES
from lowlevel.LowCmd import LowCmd


def envelope():
    return CommandEnvelope(tuple(f'j{i}' for i in range(42)),
                           np.full(42, -1.1), np.full(42, 1.1),
                           np.ones(42)*3, np.ones(42)*.02, .002, .1)


def step():
    return PolicyStep('motion', VITA_CONTROLLER_JOINT_NAMES, np.arange(30),
                      np.linspace(-2, 2, 30), np.arange(30)+100., np.arange(30)*.1)


def process(sink, **kwargs):
    args = dict(hold_q=np.zeros(42), period_s=.02, now_monotonic_ns=10000000,
                lowstate_received_monotonic_ns=9000000, state_id=65535,
                attitude_valid=True)
    args.update(kwargs)
    return sink.process(step(), **args)


class OutputTest(unittest.TestCase):
    def sink(self):
        obj = InertCommandSink(envelope())
        self.assertTrue(obj.arm(np.zeros(42)).allowed)
        return obj

    def test_conditioning_matches_shared_adapter_and_wire_schema(self):
        sink = self.sink()
        adapter = VitaCommandAdapter(envelope())
        adapter.arm(np.zeros(42))
        expected = adapter.condition(step(), hold_q=np.zeros(42), period_s=.02)
        result = process(sink)
        self.assertTrue(result.decision.allowed)
        self.assertFalse(result.action_sent)
        self.assertEqual(result.ownership_scope, 'simulated-local-only')
        self.assertEqual(result.conditioning, expected[3])
        for actual, reference in zip((result.target_q, result.kp, result.kd), expected[:3]):
            np.testing.assert_array_equal(actual, reference.astype(np.float32))
        msg = LowCmd.GetRootAs(result.payload)
        self.assertEqual(msg.MotorCmdLength(), 42)
        self.assertEqual(msg.CmdId(), 1)
        self.assertIsNone(msg.AortaHeader())
        self.assertEqual(msg.ExternalVersion(), 0)
        self.assertEqual(msg.ExternalSessionId(), 0)
        self.assertEqual(msg.ExternalSequence(), 0)
        slots = set(VITA_ACTION_TO_COMMAND_SLOTS)
        for i in range(42):
            m = msg.MotorCmd(i)
            self.assertEqual((m.Mode(), m.Dq(), m.Tau()), (10, 0., 0.))
            self.assertEqual(m.Q(), float(result.target_q[i]))
            self.assertEqual(m.Kp(), float(result.kp[i]))
            self.assertEqual(m.Kd(), float(result.kd[i]))
            self.assertGreaterEqual(m.Q(), -1.1)
            self.assertLessEqual(m.Q(), 1.1)
            if i not in slots:
                self.assertEqual((m.Q(), m.Kp(), m.Kd()), (0., 0., 0.))

    def test_state_rollover_and_command_rollover(self):
        sink = self.sink()
        sink.command_id = 65534
        self.assertEqual(process(sink).command_id, 65535)
        wrapped = process(sink, state_id=0)
        self.assertTrue(wrapped.decision.allowed)
        self.assertEqual(wrapped.command_id, 1)

    def test_stale_state_latches_without_serializing(self):
        sink = self.sink()
        bad = process(sink, now_monotonic_ns=200000000)
        self.assertEqual(bad.decision.reasons, ('lowstate_stale',))
        self.assertIsNone(bad.payload)
        self.assertEqual(bad.command_id, 0)
        self.assertFalse(process(sink).decision.allowed)

    def test_reversed_state_and_attitude_reject(self):
        sink = self.sink()
        process(sink, state_id=10)
        self.assertIn('lowstate_not_advancing', process(sink, state_id=9).decision.reasons)
        result = process(self.sink(), attitude_valid=False)
        self.assertIn('attitude_invalid', result.decision.reasons)
        self.assertIsNone(result.payload)

    def test_invalid_initial_pose(self):
        sink = InertCommandSink(envelope())
        with self.assertRaises(InitialPoseError):
            sink.arm(np.ones(42)*2)
        with self.assertRaises(ValueError):
            sink.arm(np.ones(30))
        with self.assertRaises(ValueError):
            sink.arm(np.full(42, np.nan))

    def test_period_gains_and_shape_reject(self):
        with self.assertRaises(RuntimeError):
            process(self.sink(), period_s=.2)
        with self.assertRaises(ValueError):
            process(self.sink(), hold_q=np.zeros(30))
        obj = step()
        obj.joint_stiffness[0] = 601
        sink = self.sink()
        with self.assertRaises(ValueError):
            sink.process(obj, hold_q=np.zeros(42), period_s=.02,
                         now_monotonic_ns=10, lowstate_received_monotonic_ns=9,
                         state_id=1, attitude_valid=True)
        with self.assertRaises(ValueError):
            serialize_inert_lowcmd(0, np.zeros(42), np.zeros(42), np.zeros(42))
        with self.assertRaises(ValueError):
            serialize_inert_lowcmd(1, np.full(42,np.nan), np.zeros(42), np.zeros(42))

    def test_envelope_switch_keeps_command_sequence(self):
        sink = self.sink()
        process(sink)
        smaller = dataclasses.replace(envelope(), lower_rad=np.full(42,-.5), upper_rad=np.full(42,.5))
        result = process(sink, state_id=0, envelope=smaller)
        self.assertTrue(result.decision.allowed)
        self.assertEqual(result.command_id, 2)
        self.assertLessEqual(float(np.max(np.abs(result.target_q))), .5)

    def test_exported_envelopes_match_original_derivation(self):
        import json
        from types import SimpleNamespace
        from deployment.vita_s100_aorta_30dof.onboard_output import load_output_envelopes
        from deployment.vita_s100_aorta_30dof.onboard_runtime import load_contracts
        from deployment.vita_s100_aorta_30dof.command_guard import command_envelope_from_contract
        from deployment.vita_s100_aorta_30dof.policy_limits import policy_envelope
        path = ROOT / 'artifacts/output_envelopes.json'
        expected = '595b0d1fb26739e8fd3b0874dbc70bdd89ad8d772b5103adce199433dcaf03e0'
        actual = load_output_envelopes(path, expected_sha256=expected)
        base = command_envelope_from_contract(json.loads(path.with_suffix('.interface.json').read_text()))
        bundle = SimpleNamespace(artifacts={'motion_urdf':SimpleNamespace(path=path.with_suffix('.motion.urdf'))})
        contracts = load_contracts(ROOT / 'artifacts/policy_contracts.json')
        direct = {'base':base, **{mode:policy_envelope(base,contracts,bundle,mode) for mode in ('velocity','motion')}}
        for name in direct:
            for key, value in dataclasses.asdict(direct[name]).items():
                np.testing.assert_array_equal(value, getattr(actual[name],key))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            load_output_envelopes(path, expected_sha256='0'*64)

    def test_module_import_has_no_sdk_dependency(self):
        original = builtins.__import__
        def reject_sdk(name, *args, **kwargs):
            if name.split('.')[0] in ('aorta', 'locomotion_aorta'):
                raise AssertionError('SDK imported by inert output module')
            return original(name, *args, **kwargs)
        builtins.__import__ = reject_sdk
        try:
            module = importlib.import_module('deployment.vita_s100_aorta_30dof.onboard_output')
            importlib.reload(module)
            self.assertIsNotNone(process(self.sink()).payload)
        finally:
            builtins.__import__ = original


if __name__ == '__main__':
    unittest.main(verbosity=2)
