"""Read-only input-check factory interception and Pico freshness, all mocked."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
from deployment.common.controller_input import Button,ControllerSample
ROOT=Path(__file__).resolve().parents[2]

def load_operator():
    spec=importlib.util.spec_from_file_location('onboard_operator_under_test',ROOT/'scripts/onboard_operator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class InputCheckTests(unittest.TestCase):
    def exercise(self,pico_mode,controller_kind='wifi'):
        module=load_operator();clock=SimpleNamespace(now=10.);closed=[];calls=[]
        fake_time=SimpleNamespace(monotonic=lambda:clock.now,sleep=lambda dt:setattr(clock,'now',clock.now+dt))
        class Node:
            def create_publisher(self):raise AssertionError('Actuator-capable factory was not intercepted')
            def create_client(self):raise AssertionError('Actuator-capable factory was not intercepted')
            def create_action_client(self):raise AssertionError('Actuator-capable factory was not intercepted')
            def create_service(self):raise AssertionError('Actuator-capable factory was not intercepted')
            def create_subscriber(self):calls.append('subscriber');return object()
        class State:
            def __init__(self,**kw):
                node=Node()
                for name in ('create_publisher','create_client','create_action_client','create_service'):
                    with self_test.assertRaisesRegex(RuntimeError,'forbids control handles'):getattr(node,name)()
                node.create_subscriber()
            def __enter__(self):return self
            def __exit__(self,*args):closed.append('state')
            def read(self,timeout):return SimpleNamespace(state_id=len(calls))
        class Controller:
            def __init__(self,*args,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*args):closed.append('controller')
            def sample(self):return ControllerSample(1,clock.now,frozenset([Button.START]),{})
        class Pico:
            def __init__(self):self.count=0
            def start(self):calls.append('pico_start')
            def step(self):
                self.count+=1
                if pico_mode=='none' or (pico_mode=='stalled' and self.count>1):return None
                return ('body',self.count,{'timestamp_ns':self.count})
            def stop(self):closed.append('pico')
        self_test=self
        modules={'time':fake_time,'aorta':SimpleNamespace(Node=Node),
            'deployment.vita_s100_aorta_30dof.onboard_live_io':SimpleNamespace(OnboardLiveStateSource=State),
            'deployment.vita_s100_aorta_30dof.pico_input':SimpleNamespace(create_pico_source=lambda **kw:Pico()),
            'deployment.common.wifi_controller':SimpleNamespace(WifiControllerReceiver=Controller),
            'deployment.common.resilient_xbox':SimpleNamespace(ResilientLocalXboxController=Controller)}
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            work=Path(directory);(work/'outputs').mkdir()
            with patch.dict('sys.modules',modules):
                code=module.input_check(work,SimpleNamespace(pico_max_age=.6,state_max_age=.1,aorta_startup_timeout=30.,aorta_group='default',joystick=Path('/dev/input/js0')),None,work/'key',
                    SimpleNamespace(controller=controller_kind,seconds=1.,port=16502,peer=None))
            result=json.loads((work/'outputs/operator_input_check.json').read_text())
        self.assertEqual(closed,['state','controller','pico']);self.assertEqual(calls.count('subscriber'),1)
        self.assertEqual(result['actions_sent'],0);self.assertEqual(result['control_handles_created'],0)
        self.assertIn('start',result['buttons_seen']) # Buttons are observed, not interpreted as robot commands.
        return code,result
    def test_continuously_fresh_pico_and_state_pass_without_control_handles(self):
        code,result=self.exercise('fresh');self.assertEqual(code,0);self.assertTrue(result['pico_fresh_at_end'])
    def test_one_pico_frame_then_stall_does_not_pass(self):
        code,result=self.exercise('stalled');self.assertEqual(code,1);self.assertTrue(result['pico_seen']);self.assertFalse(result['pico_fresh_at_end'])
    def test_no_pico_does_not_pass(self):
        code,result=self.exercise('none');self.assertEqual(code,1);self.assertFalse(result['pico_seen'])
    def test_local_wireless_input_check_without_control_handles(self):
        code,result=self.exercise('fresh','local');self.assertEqual(code,0)
    def test_action_wifi_factory_enables_nonfatal_link_loss(self):
        module=load_operator()
        from unittest.mock import Mock
        receiver=Mock()
        with patch.dict('sys.modules',{'deployment.common.wifi_controller':SimpleNamespace(WifiControllerReceiver=receiver)}):
            factory=module.controller_factory(SimpleNamespace(controller='wifi',port=16502,peer=None),Path('key'))
            factory(Path('/dev/input/js0'))
        self.assertTrue(receiver.call_args.kwargs['allow_link_loss'])
        self.assertEqual(receiver.call_args.kwargs['max_age'],.2)
    def test_bluetooth_factory_uses_pinned_identity_not_config_joystick(self):
        module=load_operator()
        from unittest.mock import Mock
        receiver=Mock();loader=Mock(return_value={'address':'AA:BB:CC:DD:EE:FF','adapter':'hci0'})
        modules={'deployment.common.bluetooth_xbox':SimpleNamespace(BluetoothXboxController=receiver,load_bluetooth_config=loader)}
        with patch.dict('sys.modules',modules):
            factory=module.controller_factory(SimpleNamespace(controller='bluetooth',bluetooth_config=Path('selected.json')),None)
            factory(Path('/dev/input/js99'))
        loader.assert_called_once_with(Path('selected.json'))
        receiver.assert_called_once_with(address='AA:BB:CC:DD:EE:FF',adapter='hci0',startup_timeout=30.)
    def test_static_check_requires_atomic_pico_api(self):
        module=load_operator();module.PINNED={}
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            root=Path(directory)
            binding=root/'build/pico/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so'
            modules={'xrobotoolkit_sdk':SimpleNamespace(__file__=str(binding)),
                'deployment.vita_s100_aorta_30dof.aorta_adapter':SimpleNamespace(suspended_validation_alignment_quaternion=lambda *a,**kw:None)}
            env={'VITA_PREPROCESS_LIBRARY':str(root/'artifacts/preprocessing_native_arm_v2.so'),
                 'VITA_STATE_LIBRARY':str(root/'build/state/libvita_state.so')}
            with patch.dict('sys.modules',modules),patch.dict(os.environ,env):
                with self.assertRaisesRegex(RuntimeError,'Atomic Pico snapshot API'):
                    module.verify_static(root,SimpleNamespace(imu_alignment_contract=None,robot_serial='mock'))

if __name__=='__main__':unittest.main()
