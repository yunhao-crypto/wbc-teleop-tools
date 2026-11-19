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
    def run_shadow(self,route,*,first_frame_delay=.05,stale_first=False,ingest_delay=0.,stale_after_ingest=False,
            timeout_after_ingest=False):
        events=[];clock=ShadowClock();ingested=False
        with python_boundary() as boundary,tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=Runtime(root,boundary.BODY_NAMES,route)
            feedback=Controller(runtime,clock,boundary.BODY_NAMES)
            class Source:
                bound=False;received=False;reads=0
                def get_robot_info(self,timeout):events.append(('discover',timeout));self.bound=True
                def wait_state(self,timeout):
                    events.append(('wait_state',timeout))
                    if not self.bound:raise TimeoutError('SDK rejects state before runtime discovery')
                    if ingested and timeout_after_ingest:raise TimeoutError('no latest BodyState after ingestion')
                    initial=not self.received
                    if not self.received:
                        clock.sleep(min(timeout,first_frame_delay))
                        if timeout<first_frame_delay:raise TimeoutError('first BodyState not ready within startup budget')
                        self.received=True
                    state=feedback.wait_state()
                    self.reads+=1;state.q[0]=self.reads/100.
                    if initial and stale_first:state.received_monotonic_ns-=200_000_000
                    if ingested and stale_after_ingest:state.received_monotonic_ns-=200_000_000
                    return state
                def close(self):events.append(('source_closed',None))
            source=Source()
            class Flow(pico_realtime.PicoFlow):
                def __init__(self,*args,**kwargs):
                    self.route=route;self.contract=runtime.model.contract;self.math=runtime.math
                    self.runtime=NS(config=runtime.config,policy=runtime.policy,
                        infer=lambda *a,**kw:(None,np.zeros(1167),{'raw_action':[0.]*28}))
                    self.events=[];self.frames=self.track_ticks=self.inferences=0;self.calibration={}
                    self.mode='stand';self.source_fault=None;self.losses=0;self.button_not_before_ns=0
                    self.last_frame_ns=self.last_step_ns=self.transition_start=None;self.last_reference=None
                    self.raw=[];self.stand_ticks=0;self.toggle=pico_realtime.DualTriggerToggle(.6)
                    self.standing=runtime.flow.latest;self.latest=self.standing
                    self.window=NS(reference=lambda stamp:self.standing,clear=lambda:None) if route=='pico_smpl' else None
                def push(self,body,stamp,received,measured):
                    nonlocal ingested
                    self.frames+=1;self.last_frame_ns=received
                    clock.sleep(ingest_delay);ingested=True
                    self.event('neutral_calibrated',clock.monotonic_ns())
                def step(self,stamp,measured):
                    events.append(('inference_feedback_q0',float(measured.q[0])))
                    return super().step(stamp,measured)
                def summary(self):return dict(actions_sent=0,control_handles_created=0,
                    pico_losses=self.losses,recovery_boundary_ns=self.button_not_before_ns)
            class Service:
                def __init__(self,*args,**kwargs):self.report={'executable':'in-memory','owned':False}
                def start(self):return self.report
                def check(self):return True
                def close(self):pass
            class Receiver:
                close_report=None
                def __init__(self,*args,**kwargs):pass
                def start(self):events.append(('receiver_started',None))
                def poll(self,stamp):
                    pose=(None,stamp,stamp) if ingest_delay and not ingested else None
                    return pose,(True,stamp,0.,0.,0.),dict(state='receiving',binding_test_fixture=True)
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
            report['test_event_log']=[json.loads(line) for line in (root/'out/inference.jsonl').read_text().splitlines()]
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

    def test_slow_ingestion_uses_new_feedback_without_refreshing_old_pico(self):
        for route in ('pico_smpl','pico_teleop'):
            with self.subTest(route=route):
                code,events,report=self.run_shadow(route,ingest_delay=.457)
                self.assertEqual(code,0,report)
                values=[value for kind,value in events if kind=='inference_feedback_q0']
                self.assertEqual(len(values),1)
                self.assertAlmostEqual(values[0],.03,places=6,msg='inference must use the post-ingestion read')
                self.assertEqual(report['pico_losses'],1,'old Pico receipt must still expire at real inference time')
                self.assertEqual(report['recovery_boundary_ns'],100_507_000_000)
                self.assertAlmostEqual(report['input_ingress_peak_ms'],457.,places=6)
                self.assertEqual(report['input_ingress_overrun_ticks'],1)

    def test_post_ingestion_feedback_must_be_fresh_and_events_survive_failure(self):
        for mode in ('stale','timeout'):
            with self.subTest(mode=mode):
                code,events,report=self.run_shadow('pico_smpl',ingest_delay=.457,
                    stale_after_ingest=mode=='stale',timeout_after_ingest=mode=='timeout')
                self.assertEqual(code,1)
                self.assertFalse(any(kind=='inference_feedback_q0' for kind,value in events))
                self.assertIn('BodyState',report['error'])
                self.assertIn('neutral_calibrated',[row['record_type'] for row in report['test_event_log']])
                self.assertEqual([value for kind,value in events if kind=='wait_state'],[.2,.02,.02],
                    'post-ingestion feedback read stays bounded; it cannot retry indefinitely')
                self.assertEqual(report['actions_sent'],0);self.assertEqual(report['control_handles_created'],0)


if __name__=='__main__':unittest.main()
