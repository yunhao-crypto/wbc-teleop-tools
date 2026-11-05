"""Real SDK, native decoder and generated wire messages; fake transport only.

PYTHONPATH=$SDK/python:$SDK/tests:build/sdk-ee-bindings python tests/pico_ee_sdk_test.py
"""
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deployment.vita_s100_aorta_30dof.actor_pico_hands import HandGripRunner
import test_end_effector_client as fixtures
from test_end_effector_codec import info_object, state_object
from locomotion_aorta.end_effector import EndEffectorClient
from locomotion_aorta.end_effector_types import DeviceModel, PositionDomain, Field


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.EndEffectorClientTest()
        self.addCleanup(self.helper.doCleanups)
        self.harness, self.node = self.helper.harness()
        self.config = {s:dict(joint_names=['axis'+str(i) for i in range(6)],
                             open_rad=[0]*6, grasp_rad=[.03]*6, speed_rad_s=[.1]*6)
                       for s in ('left','right')}
        self.clients = {}

    def factory(self, side, **kwargs):
        self.assertIs(kwargs.get("subscribe_touch"), False)
        client = EndEffectorClient(side, **kwargs)
        self.addCleanup(client.close)
        self.clients[side] = client
        wire_side = 1 if side == 'left' else 2
        info = info_object(side=wire_side,n=6)
        info.deviceModel = DeviceModel.BRAINCO_REVO2
        info.positionDomain = [PositionDomain.JOINT_ANGLE]*6
        self.node.emit(side, 'info', info)
        self.node.emit(side, 'state', state_object(side=wire_side,n=6))
        return client

    def runner(self):
        r = HandGripRunner(self.config, None, connection=self.harness.controller, factory=self.factory)
        self.addCleanup(r.close)
        return r

    def test_real_wire_targets_share_session_and_sequence(self):
        r = self.runner()
        r.publish(True,time.monotonic())
        r.publish(False,time.monotonic())
        commands = [wire for _,wire in self.node.commands]
        self.assertEqual(len(commands),4)
        self.assertEqual([c.Side() for c in commands],[1,2,1,2])
        self.assertEqual(len({c.Key().SessionId() for c in commands}),1)
        sequences = [c.Key().Sequence() for c in commands]
        self.assertEqual(sequences,sorted(set(sequences)))
        for c in commands:
            self.assertEqual(c.InputObservationsLength(),1)
            self.assertEqual(c.InputObservations(0).Side(),c.Side())

    def test_body_and_both_hands_share_one_monotonic_command_stream(self):
        from test_controller import make_command, _BODY_TEST_TARGETS
        r = self.runner()
        body1 = self.harness.controller.publish(make_command(0),
            target_names=_BODY_TEST_TARGETS, required_fields=0x1f)
        r.publish(True, time.monotonic())
        body2 = self.harness.controller.publish(make_command(0),
            target_names=_BODY_TEST_TARGETS, required_fields=0x1f)
        keys = [(body1.session_id, body1.sequence)]
        keys += [(wire.Key().SessionId(), wire.Key().Sequence()) for _, wire in self.node.commands]
        keys += [(body2.session_id, body2.sequence)]
        self.assertEqual(len({session for session, _ in keys}), 1)
        self.assertEqual([seq for _, seq in keys], [1, 2, 3, 4])
        self.assertEqual(body1.control_id, self.node.commands[0][1].ControlId())

    def test_stale_original_position_rejects_even_with_new_telemetry(self):
        r=self.runner()
        position=self.clients['left'].motion_feedback()[1]
        telemetry=state_object(side=1,n=6,sample=2)
        telemetry.tsStatePubUs=120
        for axis in telemetry.axes: axis.validFields=0
        self.node.emit('left','state',telemetry)
        self.assertEqual(self.clients['left'].motion_feedback()[1],position)
        with self.assertRaisesRegex(RuntimeError,'stale'):
            r.publish(True,time.monotonic()+.3)
        self.assertFalse(self.node.commands)

    def test_feedback_arriving_during_snapshot_is_fresh(self):
        r = self.runner()
        client = self.clients['left']
        original = client.motion_feedback
        before = time.monotonic()

        def concurrent_feedback():
            update = state_object(side=1, n=6, sample=2)
            update.tsStatePubUs = 120
            self.node.emit('left', 'state', update)
            snapshot = original()
            self.assertGreater(snapshot[3], before)
            return snapshot

        with patch.object(client, 'motion_feedback', concurrent_feedback):
            r.publish(True)
        self.assertEqual(len(self.node.commands), 2)

    def test_permission_revocation_blocks_both_hands(self):
        r=self.runner()
        telemetry=state_object(side=2,n=6,sample=2)
        telemetry.tsStatePubUs=120
        telemetry.outputAllowed=False
        telemetry.inhibitReason=1
        for axis in telemetry.axes: axis.validFields=0
        self.node.emit('right','state',telemetry)
        with self.assertRaisesRegex(RuntimeError, r'permission; runtime_id=100.*output_allowed=False.*inhibit_reason=1'):
            r.publish(True,time.monotonic())
        self.assertFalse(self.node.commands)

    def test_out_of_limit_config_rejected_before_publish(self):
        self.config['left']['grasp_rad']=[5]*6
        with self.assertRaises(ValueError):self.runner()
        self.assertFalse(self.node.commands)

    def test_valid_fault_is_not_hidden_by_position_cache(self):
        r=self.runner()
        fault=state_object(side=1,n=6,sample=2)
        fault.tsStatePubUs=120
        for axis in fault.axes:
            axis.validFields=Field.ERROR_CODE
            axis.errorCode=8
        self.node.emit('left','state',fault)
        with self.assertRaisesRegex(RuntimeError,'fault'):
            r.publish(True,time.monotonic())
        self.assertFalse(self.node.commands)

    def test_runtime_restart_retires_connection(self):
        r=self.runner()
        self.helper.confirm_restart(self.harness)
        with self.assertRaises(Exception):r.publish(True,time.monotonic())
        self.assertFalse(self.node.commands)

if __name__=='__main__':unittest.main()
