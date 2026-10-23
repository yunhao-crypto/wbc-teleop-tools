"""Offline body31 boundary checks. No Node or live control is constructed."""
import dataclasses
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import flatbuffers
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import sdkmain_adapter as a
from deployment.vita_s100_aorta_30dof import onboard_live_io as io
from deployment.vita_s100_aorta_30dof import onboard_live_session as live
from deployment.vita_s100_aorta_30dof.aorta_adapter import VitaExternalSession
from locomotion_aorta._body_codec import decode_body_state
from locomotion_aorta.transport import _fill_body_cmd, BODY_CMD_TOPIC, CONTROL_REQUEST_TOPIC


def module(name):return importlib.import_module('lowlevel.'+name)

def body_buffer(f):
    b=flatbuffers.Builder(1);m=module('BodyState');im=module('BodyImuSample')
    im.BodyImuSampleStart(b)
    im.BodyImuSampleAddLocation(b,0)
    im.BodyImuSampleAddStatus(b,1)
    im.BodyImuSampleAddTsSampledUs(b,1001)
    im.BodyImuSampleAddTsAcquiredHwUs(b,1002)
    im.BodyImuSampleAddTsIngestedUs(b,1003)
    im.BodyImuSampleAddTsAssembledUs(b,1004)
    im.BodyImuSampleAddState(b,module('BodyImuState').CreateBodyImuState(
        b,f['quat'],f['gyro'],[0.,0.,9.81],[0.,0.,0.]))
    imu=im.BodyImuSampleEnd(b)
    m.BodyStateStartImuStatesVector(b,1);b.PrependUOffsetTRelative(imu);imus=b.EndVector()
    m.BodyStateStartJointsVector(b,31)
    for i,slot in reversed(list(enumerate(a.BODY_TO_LEGACY))):
        module('BodyJointState').CreateBodyJointState(b,2001,2002,2003,
            f['q44'][slot] if slot is not None else float('nan'),
            f['dq44'][slot] if slot is not None else float('nan'),0.,f.get('error_code',0) if i==0 else 0)
    joints=b.EndVector();m.BodyStateStart(b)
    m.BodyStateAddProtocolVersion(b,2);m.BodyStateAddStateId(b,f['state_id'])
    m.BodyStateAddTsStateRealUs(b,f['real']);m.BodyStateAddTsStatePubUs(b,f['pub'])
    m.BodyStateAddJoints(b,joints);m.BodyStateAddImuStates(b,imus)
    m.BodyStateAddAttitudeValid(b,True)
    m.BodyStateAddAppliedSessionId(b,f.get('applied_session_id',0))
    m.BodyStateAddAppliedSequence(b,f.get('applied_sequence',0))
    m.BodyStateAddTsFirstOutputUs(b,f.get('ts_first_output_us',0))
    b.Finish(m.BodyStateEnd(b))
    return bytes(b.Output())

def state(f):return decode_body_state(module('BodyState').BodyState.GetRootAs(body_buffer(f)))

def encode(cmd):
    b=flatbuffers.Builder(1)
    b.Finish(_fill_body_cmd(NS(body_cmd_module=module('BodyCmd')),cmd,b,0))
    return module('BodyCmd').BodyCmd.GetRootAs(bytes(b.Output()))


class Body31Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_root=Path(sys.argv[0]).resolve().parents[2]
        # Supply a host evidence path when copying this reusable test elsewhere.
        import os
        fixture_root=Path(os.environ.get('T074_FIXTURES','/tmp/tonight-T074'))
        cls.fixtures=json.loads((fixture_root/'fixtures.json').read_text())
        cls.reference=json.loads((fixture_root/'rc3-reference.json').read_text())

    def test_same_input_256_real_wire_states_match_original_rc3_exactly(self):
        for f,ref in zip(self.fixtures,self.reference,strict=True):
            with self.subTest(state_id=f['state_id']):
                raw=io.decode_live_state(body_buffer(f),alignment=f['alignment'],received_monotonic_ns=111)
                s=raw.policy_state
                for actual,name in [(s.robot_dof_pos,'q'),(s.robot_dof_vel,'dq'),
                                    (s.robot_root_quat_wxyz,'root'),(s.robot_root_angvel_local,'gyro')]:
                    np.testing.assert_array_equal(actual,np.asarray(ref[name],np.float32))
                self.assertEqual((raw.state_id,raw.ts_state_real_us,raw.ts_state_pub_us),
                                 (ref['state_id'],ref['real'],f['pub']))
                self.assertEqual(raw.received_monotonic_ns,111)
                for i,slot in enumerate(a.BODY_TO_LEGACY):
                    if slot is not None:self.assertEqual(raw.q_by_slot[slot],np.float32(f['q44'][slot]))

    def test_same_input_256_commands_preserve_every_body_joint_float32(self):
        for f,ref in zip(self.fixtures,self.reference,strict=True):
            cmd=a.make_body_command(f['target'],f['kp'],f['kd'],created_us=123)
            wire=encode(cmd)
            self.assertEqual((wire.JointsLength(),wire.Action(),wire.ProtocolVersion()),(31,18,2))
            self.assertFalse(hasattr(wire,'CmdId'));self.assertEqual(wire.TsClientCreatedUs(),123)
            for i,slot in enumerate(a.BODY_TO_LEGACY):
                joint=wire.Joints(i)
                actual=[joint.Q(),joint.Dq(),joint.Tau(),joint.Kp(),joint.Kd()]
                self.assertEqual(actual,[0.]*5 if slot is None else ref['command'][slot])

    def test_policy_joint_flags_and_slot_order_fail_closed(self):
        s=state(self.fixtures[0])
        for change in [dict(error_code=8),dict(error_code=255),dict(error_code=30),
                       dict(error_code=17),dict(error_code=0xffffff9d),dict(q=float('nan')),dict(dq=float('inf'))]:
            joints=list(s.joints);joints[0]=dataclasses.replace(joints[0],**change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                a.raw_state_from_body(dataclasses.replace(s,joints=tuple(joints)),received_monotonic_ns=1)
        for bad in [dataclasses.replace(s,protocol_version=1),dataclasses.replace(s,joints=s.joints[:-1]),
                    dataclasses.replace(s,attitude_valid=False),dataclasses.replace(s,state_id=1<<64)]:
            with self.assertRaises(ValueError):a.raw_state_from_body(bad,received_monotonic_ns=1)

    def test_imu_guards_and_zero_norm_preserved(self):
        s=state(self.fixtures[0]);imu=s.imu_states[0]
        for imus in [(),(dataclasses.replace(imu,location=1),),(imu,imu),
                     (dataclasses.replace(imu,status=0),),
                     (dataclasses.replace(imu,state=None),),
                     (dataclasses.replace(imu,state=dataclasses.replace(imu.state,quaternion=(0.,)*4)),),
                     (dataclasses.replace(imu,state=dataclasses.replace(imu.state,gyroscope=(float('nan'),0.,0.))),)]:
            with self.subTest(imus=imus),self.assertRaises(ValueError):
                raw=a.raw_state_from_body(dataclasses.replace(s,imu_states=imus),received_monotonic_ns=1)
                from deployment.vita_s100_aorta_30dof.live_shadow import policy_state_from_lowstate
                policy_state_from_lowstate(raw,base_from_hip_quaternion_wxyz=[1,0,0,0])

    def test_hands_are_excluded_and_nonzero_hand_intent_is_rejected(self):
        f=self.fixtures[0];v=list(f['target']);v[25]=1
        with self.assertRaisesRegex(ValueError,'Hand commands'):
            a.make_body_command(v,f['kp'],f['kd'],created_us=1)
        with self.assertRaises(ValueError):a.make_body_command(v[:-1],f['kp'],f['kd'],created_us=1)

    def test_sdk_native_schema_guard_and_all15_sources_preserved(self):
        from deployment.vita_s100_aorta_30dof import sdk021_guard
        a.verify_state_decoder();self.assertEqual(len(live.VENDOR_SOURCE_SHA256),15)
        if sdk021_guard.GUARD['status'] == 'SEALED_FROM_REVIEWED_WHEEL':
            extension=importlib.util.find_spec('locomotion_aorta._body_cmd_native')
            if Path(extension.origin).name in sdk021_guard.GUARD['native_sha256']:
                self.assertEqual(len(live.verify_vendor_sources()),15)
            else:
                # ARM delivery pins must reject this host's x86 native payload;
                # never weaken the production guard to run an offline oracle.
                with self.assertRaisesRegex(RuntimeError,'native library'):
                    live.verify_vendor_sources()
        else:
            with self.assertRaisesRegex(RuntimeError, 'wheel guard is pending'):
                live.verify_vendor_sources()
        with patch.dict('os.environ',{'VITA_STATE_LIBRARY':'/unused/old44.so'}):
            with self.assertRaisesRegex(RuntimeError,'unset VITA_STATE_LIBRARY'):a.verify_state_decoder()

    def test_readiness_uses_arrays_not_dto_materialization(self):
        from unittest.mock import Mock
        from aorta.services.common.ServiceStatus import ServiceStatus
        session=live.OnboardLiveSession.__new__(live.OnboardLiveSession)
        session.controller=Mock()
        session.controller.wait_body_state.side_effect=AssertionError('No lazy body DTO')
        session.controller.status=NS(current_action='PASSIVE',phase='IDLE')
        body=NS(slot_count=31,joint_names=a.BODY_NAMES,joints=tuple(NS(body_index=i,populated=i!=14) for i in range(31)))
        session.controller.get_robot_info.return_value=NS(status=ServiceStatus.SUCCESS,info=NS(protocol_version=1,body=body))
        session.wait_ready(1)
        session.controller.wait_state.assert_called_once()
        session.controller.wait_body_state.assert_not_called()

    def test_clock_override_keeps_generation_sequence_and_control_requests(self):
        obj=a.RealtimeBodyTransport.__new__(a.RealtimeBodyTransport)
        cmd=dataclasses.replace(a.make_body_command(*(self.fixtures[0][k] for k in ('target','kp','kd')),created_us=42),session_id=123,sequence=99,ts_client_send_us=7)
        with patch.object(a._AortaTransport,'publish') as publish,patch.object(a.time,'time_ns',return_value=1_792_140_000_123_456_000):
            obj.publish(BODY_CMD_TOPIC,cmd)
            sent=publish.call_args.args[1];self.assertEqual(sent.ts_client_send_us,1_792_140_000_123_456)
            self.assertEqual(dataclasses.replace(sent,ts_client_send_us=7),cmd)
            request=object();obj.publish(CONTROL_REQUEST_TOPIC,request)
            self.assertIs(publish.call_args.args[1],request)

    def test_session_build_clock_command_id_and_policy_log_timing_contract(self):
        session=VitaExternalSession.__new__(VitaExternalSession)
        session.command_id=65535;session.last_publish_monotonic_ns=None;session.last_publish_wall_ns=None
        ticks=iter([100,200]);session._monotonic_ns=lambda:next(ticks)
        from unittest.mock import Mock
        session.controller=Mock();f=self.fixtures[0]
        from locomotion_aorta import BodyCmd, BodyJointCmd
        sent=BodyCmd(joints=(BodyJointCmd(),)*31,session_id=987,sequence=123,
                     input_state_id=1<<40,input_state_ts_real_us=999)
        session.controller.publish.return_value=sent
        session.controller.transport.last_body_command=dataclasses.replace(sent,ts_client_send_us=2345)
        with patch('deployment.vita_s100_aorta_30dof.aorta_adapter.time.time_ns',side_effect=[1_000_000,2_000_000,3_000_000]):
            result=session.publish(f['target'],f['kp'],f['kd'],input_state_id=1<<40,input_state_ts_real_us=999)
        self.assertEqual(result.command_id,1);self.assertEqual(result.command_build_ms,1)
        self.assertEqual(result.sdk_publish_ms,1)
        self.assertEqual((result.command_build_started_wall_ns,result.publish_started_wall_ns,result.published_wall_ns),(1_000_000,2_000_000,3_000_000))
        cmd=session.controller.publish.call_args.args[0]
        self.assertEqual(cmd.ts_client_created_us,1000);self.assertEqual(len(cmd.joints),31)
        self.assertEqual((cmd.input_state_id,cmd.input_state_ts_real_us),(1<<40,999))
        self.assertEqual((result.session_id,result.sequence,result.ts_client_send_us),(987,123,2345))

    def test_discovery_rejects_wrong_roster_before_external(self):
        from aorta.services.common.ServiceStatus import ServiceStatus
        body=NS(slot_count=31,joint_names=a.BODY_NAMES,joints=tuple(NS(body_index=i,populated=i!=14) for i in range(31)))
        reply=NS(status=ServiceStatus.SUCCESS,info=NS(protocol_version=1,body=body))
        a.validate_robot_info(reply)
        reply.status=ServiceStatus.PARTIAL_SUCCESS
        a.validate_robot_info(reply)
        body.joint_names=tuple(reversed(a.BODY_NAMES))
        with self.assertRaises(RuntimeError):a.validate_robot_info(reply)

    def test_final_reindexed_schema_and_twenty_error_codes(self):
        self.assertEqual(module('BodyJointState').BodyJointState.SizeOf(),40)
        self.assertFalse(hasattr(module('BodyCmd'),'BodyCmdAddCmdId'))
        self.assertFalse(hasattr(module('BodyState'),'BodyStateAddAppliedCmd'))
        self.assertFalse(hasattr(module('BodyImuSample'),'BodyImuSampleAddValidFields'))
        from locomotion_aorta import JointError
        codes={v for k,v in vars(JointError).items() if k.isupper()}
        self.assertEqual(codes,set(range(18))|{30,255})
        self.assertEqual((JointError.TIMEOUT,JointError.SENSOR_FAULT),(8,17))
        import body_state_schema_meta as meta
        self.assertEqual(meta.SCHEMA_HASH_HEX,a.BODY_STATE_SCHEMA_HASH)

    def test_uint32_errors_preserve_wire_values_and_policy_rejects_faults(self):
        for code in (0x7fffffff, 0x80000000, 0xffffffff):
            with self.subTest(error_code=code):
                payload=body_buffer(dict(self.fixtures[0],error_code=code))
                wire=module('BodyState').BodyState.GetRootAs(payload)
                self.assertEqual(wire.Joints(0).ErrorCode(),code)
                decoded=decode_body_state(wire)
                self.assertEqual(decoded.joints[0].error_code,code)
                with self.assertRaisesRegex(ValueError,'Policy joint Q/Dq unavailable'):
                    a.raw_state_from_body(decoded,received_monotonic_ns=111)

    def test_uint64_state_applied_keys_and_timestamp_chains_survive_decode(self):
        f=dict(self.fixtures[0],state_id=(1<<64)-1,
               applied_session_id=(1<<63)+1,applied_sequence=(1<<40)+7,
               ts_first_output_us=1_792_140_000_123_456)
        raw=io.decode_live_state(body_buffer(f),alignment=f['alignment'],received_monotonic_ns=111)
        self.assertEqual(raw.state_id,f['state_id'])
        self.assertEqual(raw.policy_state.state_id,f['state_id'])
        self.assertEqual((raw.applied_session_id,raw.applied_sequence,raw.ts_first_output_us),
                         (f['applied_session_id'],f['applied_sequence'],f['ts_first_output_us']))
        self.assertEqual(raw.imu_timestamps_us,((0,1001,1002,1003,1004),))
        self.assertEqual(raw.joint_timestamps_us,((2001,2002,2003),)*31)
        no_applied=io.decode_live_state(body_buffer(self.fixtures[0]),alignment=f['alignment'],received_monotonic_ns=111)
        self.assertEqual((no_applied.applied_session_id,no_applied.applied_sequence,no_applied.ts_first_output_us),(0,0,0))
        body=a.raw_state_from_body(state(f),received_monotonic_ns=111)
        self.assertIsNone(body.imu_samples[0].device_timestamp_ms)
        self.assertIsNone(body.imu_samples[0].host_timestamp_us)
        for value in [-1,1<<64,True,1.5]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                dataclasses.replace(body,state_id=value)
        with self.assertRaisesRegex(ValueError,'uint16'):
            dataclasses.replace(body,state_id=65536,state_id_bits=16)

    def test_live_uint64_guards_keep_legacy_wrap_and_reject_real_backwards(self):
        from deployment.vita_s100_aorta_30dof.command_guard import (
            CommandEnvelope,FailClosedCommandGuard,SessionIdentity)
        envelope=CommandEnvelope(tuple('j'+str(i) for i in range(42)),
            np.full(42,-1.),np.full(42,1.),np.ones(42)*3,np.ones(42)*.02,.002,.1)
        def guard(bits):
            obj=FailClosedCommandGuard(envelope,state_id_bits=bits)
            self.assertTrue(obj.arm(SessionIdentity('client','lease',99,0),np.zeros(42)).allowed)
            return obj
        def check(obj,state_id,hold=False):
            args=dict(now_monotonic_ns=100,lowstate_received_monotonic_ns=99,state_id=state_id,
                attitude_valid=True,current_action='EXTERNAL',phase='RUNNING',
                client_id='client',lease_id='lease',generation=99,reject_count=0)
            if hold:return obj.evaluate_hold_last(**args)
            return obj.evaluate(target_q=np.zeros(42),period_s=.02,**args)
        for hold in (False,True):
            obj=guard(64)
            for value in [65535,65536,65536,1<<40,(1<<64)-1]:
                self.assertTrue(check(obj,value,hold).allowed)
            self.assertFalse(check(obj,0,hold).allowed)
            self.assertIn('lowstate_not_advancing',obj.fault_reasons)
            for invalid in [-1,1<<64,True,1.5]:
                self.assertFalse(check(guard(64),invalid,hold).allowed)
        legacy=guard(16)
        self.assertTrue(check(legacy,65535).allowed)
        self.assertTrue(check(legacy,0).allowed)
        self.assertFalse(check(legacy,65535).allowed)

    def test_real_controller_assigns_wire_keys_and_uses_selected_uint64_input(self):
        from locomotion_aorta import ExternalController,ControlStatus,ExternalPhase
        frames=[]
        from sdkmain_native_test_support import native_test_transport
        transport=native_test_transport(a.RealtimeBodyTransport,frames)
        import aorta
        with patch.object(aorta,'Node',side_effect=AssertionError('No real Node in offline test')):
            controller=ExternalController(client_id='test',_transport=transport)
            controller._request_id='synthetic-request';controller._lease_id='synthetic-lease'
            session=VitaExternalSession.__new__(VitaExternalSession)
            session.controller=controller;session.command_id=65535
            session.last_publish_monotonic_ns=None;session.last_publish_wall_ns=None
            session._monotonic_ns=lambda:123
            f=self.fixtures[0]
            for index,generation in enumerate((99,99,123)):
                controller._on_status(ControlStatus(status_seq=index+1,current_action='EXTERNAL',
                    phase=ExternalPhase.RUNNING,client_id='test',lease_id='synthetic-lease',generation=generation))
                selected=(1<<40)+index
                with patch.object(a.time,'time_ns',return_value=1_792_140_000_123_456_000):
                    result=session.publish(f['target'],f['kp'],f['kd'],
                        input_state_id=selected,input_state_ts_real_us=123456789)
                wire=frames[-1]
                key=(generation,2 if index==1 else 1)
                self.assertEqual((result.session_id,result.sequence),key)
                self.assertEqual((wire.SessionId(),wire.Sequence()),key)
                self.assertEqual((wire.InputStateId(),wire.InputStateTsRealUs()),(selected,123456789))
                self.assertEqual((result.input_state_id,result.input_state_ts_real_us),(selected,123456789))
                self.assertEqual(wire.TsClientSendUs(),1_792_140_000_123_456)
                self.assertEqual(result.ts_client_send_us,wire.TsClientSendUs())

    def test_failed_publish_does_not_record_a_wire_send(self):
        transport=a.RealtimeBodyTransport.__new__(a.RealtimeBodyTransport)
        cmd=a.make_body_command(*(self.fixtures[0][k] for k in ('target','kp','kd')),created_us=42)
        with patch.object(a._AortaTransport,'publish',side_effect=RuntimeError('synthetic send failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic send failure'):
                transport.publish(BODY_CMD_TOPIC,cmd)
        self.assertFalse(hasattr(transport,'last_body_command'))

if __name__=='__main__':unittest.main(verbosity=2)
