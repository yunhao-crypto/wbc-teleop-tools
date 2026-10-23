from pathlib import Path
import json,sys,types,unittest.mock,hashlib,inspect,os
root=Path('/userdata/vita_p123_20261015_vita');sys.path.insert(0,str(root/'src'))
import aorta
from locomotion_aorta.transport import _AortaTransport,EXTERNAL_COMMAND_TOPIC
from deployment.vita_s100_aorta_30dof.onboard_live_session import _ReadinessTransport,verify_vendor_sources
from deployment.vita_s100_aorta_30dof.aorta_adapter import VitaExternalSession
results=[]
class FakePublisher:
 def publish_bytes(self,b):raise AssertionError('No bytes must be published')
 def close(self):pass
fake=FakePublisher()
def init(self,**kwargs):self._bindings=types.SimpleNamespace();self.publishers={EXTERNAL_COMMAND_TOPIC:fake}
with unittest.mock.patch.object(_AortaTransport,'__init__',init):
 transport=_ReadinessTransport(node_name='inert-adapter-check',group='default')
 assert transport.command_publisher.publisher is fake
 results.append('rc3 bindings without motor_cmd_module accepted; existing profile wrapper retained')
class FakeController:
 status=types.SimpleNamespace(current_action='PASSIVE',phase='IDLE')
 def enter_external(self,*args,**kwargs):self.called=(args,kwargs);return 'dummy-status'
session=VitaExternalSession.__new__(VitaExternalSession);session.controller=FakeController();session.entered=False
assert session.enter(1.0,lease_duration_s=7200.0)=='dummy-status'
assert session.controller.called==((1.0,),{})
results.append('adapter preserves caller API but rc3 SDK receives timeout only; fake controller only')
for rel in ['runtime.py','onboard_runtime.py','onboard_live_io.py','onboard_state.py','real_teleop.py','command_guard.py','onboard_command_encoding.py','onboard_publish_profile.py']:
 a=root/'src/deployment/vita_s100_aorta_30dof'/rel;b=Path('/userdata/vita_p123_20261015/onboard/src/deployment/vita_s100_aorta_30dof')/rel
 assert a.read_bytes()==b.read_bytes(),rel
results.append('listed model/control/state/guards/packed/profile files unchanged')
for p in (root/'scripts').glob('*.py'):compile(p.read_bytes(),str(p),'exec')
results.append('script syntax compiled in memory, no .pyc written')
report={'checks':results,'sdk_source_hashes':verify_vendor_sources(),'real_Node_instances':0,'real_controller_instances':0,'commands_sent':0,'passed':True}
(root/'outputs/t063-prepare-checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
