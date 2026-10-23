"""No-network codec audit of the exact delivered locomotion_aorta wheel.

Never constructs ExternalController, Node, publisher, subscriber or transport.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
WHEEL=ROOT/'packages/live_sdk_wheels/locomotion_aorta-0.1.0-py3-none-any.whl'
assert hashlib.sha256(WHEEL.read_bytes()).hexdigest()=='0e77824981b939fd0edc9348b0c5dfe440d5be26f8a79b963e5fbc6e482e4b3f'
sys.path.insert(0,str(WHEEL));sys.path.insert(0,str(ROOT/'packages/state_schema'))
# Pure generated AortaHeader namespace only; never load aorta native runtime.
schema_namespace=types.ModuleType('aorta');schema_namespace.__path__=[str(ROOT/'packages/state_schema/aorta')];sys.modules['aorta']=schema_namespace
import flatbuffers
from locomotion_aorta import ControlRequest,ControlOperation,LowCmd,MotorCommand,ExternalPhase
from locomotion_aorta.transport import _fill_control_request,_fill_low_cmd,_decode_control_status,_decode_control_ack,_decode_phase
from lowlevel.LowCmd import LowCmd as LowCmdView
from locomotion_sdk.ControlRequest import ControlRequest as RequestView
from locomotion_sdk.ControlAck import ControlAck as AckView
from locomotion_sdk.ControlStatus import ControlStatus as StatusView
bindings=types.SimpleNamespace(control_request_module=importlib.import_module('locomotion_sdk.ControlRequest'),
                               low_cmd_module=importlib.import_module('lowlevel.LowCmd'),
                               motor_cmd_module=importlib.import_module('lowlevel.MotorCmd'))

class CodecTest(unittest.TestCase):
    def test_request_roundtrip_prepare_commit_cancel(self):
        for operation,number in [(ControlOperation.PREPARE,1),(ControlOperation.COMMIT,2),(ControlOperation.CANCEL,3)]:
            request=ControlRequest('offline-request','offline-client',operation,lease_id='' if number==1 else 'offline-lease',requested_lease_ms=30000 if number==1 else 0)
            b=flatbuffers.Builder(512);b.Finish(_fill_control_request(bindings,request,b,0));view=RequestView.GetRootAs(bytes(b.Output()))
            self.assertEqual(view.ProtocolVersion(),1);self.assertEqual(view.Operation(),number)
            self.assertEqual(view.RequestId(),b'offline-request');self.assertEqual(view.ClientId(),b'offline-client')
            self.assertEqual(view.ActionName(),b'EXTERNAL');self.assertEqual(view.LeaseId(),request.lease_id.encode())
            self.assertEqual(view.RequestedLeaseMs(),request.requested_lease_ms)

    def test_lowcmd_roundtrip_real_session_fields(self):
        motors=tuple(MotorCommand(mode=10,q=i/32,kp=100+i,kd=1+i/32) for i in range(42))
        cmd=LowCmd(65535,motors,external_version=1,external_session_id=12345,external_sequence=67890)
        b=flatbuffers.Builder(2048);b.Finish(_fill_low_cmd(bindings,cmd,b,0));view=LowCmdView.GetRootAs(bytes(b.Output()))
        self.assertEqual(view.CmdId(),65535);self.assertEqual(view.ExternalVersion(),1)
        self.assertEqual(view.ExternalSessionId(),12345);self.assertEqual(view.ExternalSequence(),67890)
        self.assertEqual(view.MotorCmdLength(),42)
        for i,motor in enumerate(motors):
            v=view.MotorCmd(i)
            self.assertEqual((v.Mode(),v.Q(),v.Dq(),v.Tau(),v.Kp(),v.Kd()),(motor.mode,motor.q,0,0,motor.kp,motor.kd))

    def test_ack_decoder(self):
        module=importlib.import_module('locomotion_sdk.ControlAck')
        b=flatbuffers.Builder(512);request=b.CreateString('offline-request');lease=b.CreateString('offline-lease');reason=b.CreateString('ok')
        module.ControlAckStart(b)
        for field,value in [('ProtocolVersion',1),('RequestId',request),('Operation',2),('Accepted',True),('LeaseId',lease),('LeaseDeadlineMonotonicNs',1234567),('Reason',reason)]:
            getattr(module,'ControlAckAdd'+field)(b,value)
        b.Finish(module.ControlAckEnd(b));ack=_decode_control_ack(AckView.GetRootAs(bytes(b.Output())))
        self.assertTrue(ack.accepted);self.assertEqual(ack.operation,ControlOperation.COMMIT)
        self.assertEqual(ack.lease_id,'offline-lease');self.assertEqual(ack.lease_deadline_monotonic_ns,1234567)

    def test_status_decoder(self):
        module=importlib.import_module('locomotion_sdk.ControlStatus')
        b=flatbuffers.Builder(512);action=b.CreateString('EXTERNAL');client=b.CreateString('offline-client');lease=b.CreateString('offline-lease')
        module.ControlStatusStart(b)
        for field,value in [('ProtocolVersion',1),('StatusSeq',42),('CurrentAction',action),('ActionPhase',4),('ClientId',client),('LeaseId',lease),('Generation',12345),('StampMonotonicNs',7654321),('LastAcceptedCmdId',5),('RejectCount',2)]:
            getattr(module,'ControlStatusAdd'+field)(b,value)
        b.Finish(module.ControlStatusEnd(b));status=_decode_control_status(StatusView.GetRootAs(bytes(b.Output())))
        self.assertEqual(status.phase,ExternalPhase.RUNNING);self.assertEqual(status.generation,12345)
        self.assertEqual(status.status_seq,42);self.assertEqual(status.stamp_monotonic_ns,7654321);self.assertEqual(status.reject_count,2)
        for invalid in [-1,9,255]:
            with self.assertRaises(ValueError):_decode_phase(invalid)

    def test_no_control_objects_or_native_runtime_loaded(self):
        self.assertFalse(hasattr(sys.modules['aorta'],'Node'))
        self.assertNotIn('aorta.node',sys.modules)
        self.assertNotIn('aorta._ffi',sys.modules)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CodecTest)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests=result.testsRun,passed=result.wasSuccessful(),node_instances=0,controller_instances=0,transport_instances=0,robot_connections=0,actions_sent=0,
                scope='Pure FlatBuffers request/ack/status/LowCmd codec with delivered wheel and generated schema; no live transport, ownership handshake or robot compatibility claim',wheel_sha256=hashlib.sha256(WHEEL.read_bytes()).hexdigest())
    (ROOT/'outputs/live_protocol_offline.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
