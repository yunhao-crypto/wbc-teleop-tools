"""Actual no-action shadow entry: SDK discovery precedes native feedback reads."""
from contextlib import redirect_stdout
import io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'src')]
import pico_inputs as entry
from test_pico_aligned_live import python_boundary,Clock,Runtime,Controller
from deployment.vita_s100_aorta_30dof import pico_realtime,pico_service,pico_input_process,actor_pico_live,sdk021_guard


class ShadowClock(Clock):
    def monotonic_ns(self):return round(self.monotonic()*1e9)
    def perf_counter(self):return self.monotonic()


class ShadowDiscoveryTests(unittest.TestCase):
    def run_shadow(self,route,*,first_frame_delay=.05,stale_first=False):
        events=[];clock=ShadowClock()
        with python_boundary() as boundary,tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=Runtime(root,boundary.BODY_NAMES,route)
            feedback=Controller(runtime,clock,boundary.BODY_NAMES)
            class Source:
                bound=False;received=False
                def get_robot_info(self,timeout):events.append(('discover',timeout));self.bound=True
                def wait_state(self,timeout):
                    events.append(('wait_state',timeout))
                    if not self.bound:raise TimeoutError('SDK rejects state before runtime discovery')
                    initial=not self.received
                    if not self.received:
                        clock.sleep(min(timeout,first_frame_delay))
                        if timeout<first_frame_delay:raise TimeoutError('first BodyState not ready within startup budget')
                        self.received=True
                    state=feedback.wait_state()
                    if initial and stale_first:state.received_monotonic_ns-=200_000_000
                    return state
                def close(self):events.append(('source_closed',None))
            source=Source()
            class Flow:
                def __init__(self,*args,**kwargs):
                    self.runtime=runtime;self.contract=runtime.model.contract
                    self.events=[];self.frames=self.track_ticks=self.inferences=0;self.calibration={}
                def step(self,stamp,measured):
                    measured.validate();self.inferences+=1
                    return dict(inferred=True,mode='stand',report={'raw_action':[0.]*28},observation=np.zeros(1167))
                def button(self,*args):pass
                def stop(self,*args):pass
                def summary(self):return dict(actions_sent=0,control_handles_created=0)
            class Service:
                def __init__(self,*args,**kwargs):self.report={'executable':'in-memory','owned':False}
                def start(self):return self.report
                def check(self):return True
                def close(self):pass
            class Receiver:
                close_report=None
                def __init__(self,*args,**kwargs):pass
                def start(self):events.append(('receiver_started',None))
                def poll(self,stamp):return None,(False,0,float('inf'),0.,0.),dict(state='receiving',binding_test_fixture=True)
                def close(self):self.close_report={'receiver_exitcode':0};return self.close_report
            original=actor_pico_live.Inputs.validate_state
            def validate(inputs,state):
                result=original(inputs,state);events.append(('state_validated',None));return result
            argv=['pico_inputs.py',route,'shadow','--root',str(root),'--feedback','body31',
                '--seconds','.04','--startup-timeout','.2','--out',str(root/'out')]
            if route=='pico_smpl':argv+=['--wrist-mode','stand']
            with patch.object(sys,'argv',argv),patch.dict(entry.os.environ,{'ZENOH_SESSION_CONFIG_URI':'in-memory-only'}), \
                    patch.object(entry,'time',clock),patch.object(pico_realtime,'PicoFlow',Flow), \
                    patch.object(pico_service,'default_sdk_directory',return_value=None), \
                    patch.object(pico_service,'library_directories',return_value=[]),patch.object(pico_service,'PicoService',Service), \
                    patch.object(pico_input_process,'PicoInputProcess',Receiver), \
                    patch.object(sys.modules['locomotion_aorta.pull'],'PullStateSource',return_value=source,create=True), \
                    patch.object(actor_pico_live,'require_peer_profile'),patch.object(sdk021_guard,'verify_vendor_sources'), \
                    patch.object(actor_pico_live.Inputs,'validate_state',validate),redirect_stdout(io.StringIO()):
                try:entry.main()
                except SystemExit as exc:code=exc.code
                except RuntimeError:code=1
                else:code=0
            report=json.loads((root/'out/report.json').read_text())
            return code,events,report

    def test_discovery_and_first_fresh_state_precede_receiver_for_both_routes(self):
        for route in ('pico_smpl','pico_teleop'):
            with self.subTest(route=route):
                code,events,report=self.run_shadow(route)
                self.assertEqual(code,0,report)
                self.assertEqual(events[:3],[('discover',.2),('wait_state',.2),('state_validated',None)])
                self.assertEqual(events[3],('receiver_started',None))
                self.assertIn(('wait_state',.02),events,'steady state must retain its short wait')
                self.assertEqual(events[-1],('source_closed',None))
                self.assertEqual(report['actions_sent'],0);self.assertEqual(report['control_handles_created'],0)

    def test_first_state_timeout_closes_readonly_source_before_pico_start(self):
        code,events,report=self.run_shadow('pico_smpl',first_frame_delay=.3)
        self.assertEqual(code,1)
        self.assertIn('first BodyState not ready',report['error'])
        self.assertNotIn(('receiver_started',None),events)
        self.assertEqual(events[-1],('source_closed',None))
        self.assertEqual(report['actions_sent'],0);self.assertEqual(report['control_handles_created'],0)

    def test_startup_wait_does_not_admit_a_stale_first_frame(self):
        code,events,report=self.run_shadow('pico_teleop',stale_first=True)
        self.assertEqual(code,1)
        self.assertIn('BodyState local receipt is stale/future',report['error'])
        self.assertNotIn(('receiver_started',None),events)
        self.assertEqual(events[-1],('source_closed',None))
        self.assertEqual(report['actions_sent'],0);self.assertEqual(report['control_handles_created'],0)


if __name__=='__main__':unittest.main()
