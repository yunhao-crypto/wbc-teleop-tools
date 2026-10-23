"""No-network tests for the readiness-only transport override."""
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'packages/live_sdk_wheels/locomotion_aorta-0.1.0-py3-none-any.whl'))
from deployment.vita_s100_aorta_30dof import onboard_live_session as live
from deployment.vita_s100_aorta_30dof.aorta_adapter import VitaExternalSession
from locomotion_aorta.transport import CONTROL_ACK_TOPIC,CONTROL_STATUS_TOPIC,LOWSTATE_TOPIC,_decode_control_ack,_decode_control_status
from locomotion_aorta import ExternalController,HumanoidLowState


class View:
    def __init__(self): self.motor_reads=0
    def StateId(self): return 123
    def TsStateReal(self): return 456
    def AttitudeValid(self): return True
    def MotorStatesLength(self): return 44
    def MotorStates(self,index):
        self.motor_reads+=1
        return types.SimpleNamespace(Q=lambda:index*.01,Dq=lambda:0.,TauEst=lambda:0.,Temperature=lambda:22.)


def transport():
    # Bypass the live constructor entirely; the fake bindings have no Node.
    obj=live._ReadinessTransport.__new__(live._ReadinessTransport)
    obj._configure_readiness()
    clock=[0]
    obj._readiness_clock_ns=lambda:clock[0]
    obj._bindings=types.SimpleNamespace(
        aorta=types.SimpleNamespace(QoS=types.SimpleNamespace(sensor_data=lambda:'sensor',realtime_control=lambda:'realtime',state_update=lambda:'state')),
        humanoid_low_state_decoder=object(),control_ack_decoder=object(),control_status_decoder=object())
    return obj,clock


class ReadinessTest(unittest.TestCase):
    def test_invalid_command_schema_closes_partial_transport(self):
        def inert_init(obj,**kwargs):
            obj._bindings=types.SimpleNamespace(motor_cmd_module=types.SimpleNamespace(
                MotorCmd=types.SimpleNamespace(SizeOf=lambda:28)))
        with patch.object(live._AortaTransport,'__init__',inert_init),patch.object(live._AortaTransport,'close') as close:
            with self.assertRaisesRegex(RuntimeError,'Unexpected MotorCmd layout'):
                live._ReadinessTransport(node_name='offline',group='default')
            close.assert_called_once()

    def test_first_skip_and_boundary_use_original_full_decode(self):
        obj,clock=transport();view=View()
        decoder,convert,qos=obj._subscriber_spec(LOWSTATE_TOPIC)
        self.assertIs(decoder,obj._bindings.humanoid_low_state_decoder)
        self.assertEqual(qos,'sensor')
        value=convert(view)
        self.assertIsInstance(value,HumanoidLowState)
        self.assertEqual((value.state_id,value.stamp_ns,len(value.motor_states)),(123,456000,44))
        self.assertEqual(view.motor_reads,44)
        for t in [0,1,199999999]:
            clock[0]=t;self.assertIsNone(convert(view))
        self.assertEqual(view.motor_reads,44)
        clock[0]=200000000
        self.assertIsInstance(convert(view),HumanoidLowState)
        self.assertEqual(view.motor_reads,88)
        self.assertEqual((obj.readiness_decoded,obj.readiness_skipped),(2,3))

    def test_at_most_five_decodes_in_first_second(self):
        obj,clock=transport();view=View()
        for t in range(350):
            clock[0]=t*1_000_000_000//350
            obj._decode_readiness(view)
        self.assertEqual(obj.readiness_decoded,5)
        self.assertEqual(obj.readiness_skipped,345)
        self.assertEqual(view.motor_reads,5*44)

    def test_ack_and_status_conversion_untouched(self):
        obj,clock=transport();obj._decode_readiness(View())
        for topic,expected in [(CONTROL_ACK_TOPIC,_decode_control_ack),(CONTROL_STATUS_TOPIC,_decode_control_status)]:
            _,convert,_=obj._subscriber_spec(topic)
            self.assertIs(convert,expected)
        self.assertEqual(obj.readiness_skipped,0)
        self.assertIs(live._ReadinessTransport.publish,live._AortaTransport.publish)
        self.assertIs(live._ReadinessTransport.subscribe,live._AortaTransport.subscribe)
        self.assertIs(live._ReadinessTransport.close,live._AortaTransport.close)

    def test_vendor_none_type_guard_keeps_cached_readiness(self):
        # Call unbound pure callback on an inert stub, not a controller instance.
        stub=types.SimpleNamespace(_condition=threading.Condition(),_closed=False,_latest_lowstate='unchanged')
        ExternalController._on_lowstate(stub,None)
        self.assertEqual(stub._latest_lowstate,'unchanged')
        value=HumanoidLowState(1,2,(),True)
        ExternalController._on_lowstate(stub,value)
        self.assertIs(stub._latest_lowstate,value)

    def test_original_session_methods_inherited(self):
        for name in ['publish','enter','exit','close','wait_ready','require_available']:
            self.assertIs(getattr(live.OnboardLiveSession,name),getattr(VitaExternalSession,name))

    def test_vendor_hash_gate_precedes_any_construction(self):
        class FakeLoader:
            def get_data(self,path):return b'unreviewed'
        spec=types.SimpleNamespace(origin='/unused/locomotion_aorta/__init__.py',loader=FakeLoader())
        with patch.object(live.importlib.util,'find_spec',return_value=spec),patch.object(live,'_ReadinessTransport') as factory:
            with self.assertRaisesRegex(RuntimeError,'Unreviewed'):
                live.OnboardLiveSession('default')
            factory.assert_not_called()

    def test_constructor_failure_closes_inert_transport(self):
        closed=[]
        fake_transport=types.SimpleNamespace(close=lambda:closed.append(True))
        with patch.object(live,'_ReadinessTransport',return_value=fake_transport),patch.object(live,'ExternalController',side_effect=RuntimeError('inert failure')):
            with self.assertRaisesRegex(RuntimeError,'inert failure'):
                live.OnboardLiveSession('default')
        self.assertEqual(closed,[True])
        with patch.object(live,'_ReadinessTransport') as factory:
            with self.assertRaises(ValueError):live.OnboardLiveSession('')
            factory.assert_not_called()

    def test_initializer_uses_only_injected_fake_objects(self):
        fake_transport=object();fake_controller=object()
        with patch.object(live,'_ReadinessTransport',return_value=fake_transport) as make_transport,patch.object(live,'ExternalController',return_value=fake_controller) as make_controller:
            session=live.OnboardLiveSession(group='default')
        self.assertIs(session.controller,fake_controller)
        self.assertIs(make_controller.call_args.kwargs['_transport'],fake_transport)
        self.assertEqual(make_transport.call_args.kwargs['group'],'default')
        self.assertEqual(session.command_id,0);self.assertFalse(session.entered)
        self.assertEqual(session.vendor_source_hashes,live.VENDOR_SOURCE_SHA256)

if __name__=='__main__':unittest.main(verbosity=2)
