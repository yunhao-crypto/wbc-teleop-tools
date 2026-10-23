"""Exercise real startup/cleanup functions with inert resource doubles."""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def function_namespace(path,names,**values):
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    namespace=dict(__file__=str(path),**values)
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace

class CleanupTests(unittest.TestCase):
    def test_runtime_startup_failure_closes_backend_and_reports(self):
        closed=[]
        class Backend:
            def __init__(self,*a,**k):pass
            def close(self):closed.append(True)
            def state(self):return {'test':True}
        def failed_runtime(**kwargs):raise RuntimeError('runtime startup failure')
        path=ROOT/'scripts/run_onboard_shadow.py'
        ns=function_namespace(path,{'main'},argparse=argparse,Path=Path,json=json,np=np,time=time,gc=gc,
            live_alignment_quaternion=lambda x:None,NativeBackend=Backend,OnboardDualPolicyRuntime=failed_runtime,
            NativeMotionRunner=lambda x:x,NativeVelocityRunner=lambda x:x)
        before=gc.isenabled()
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            output=Path(directory)/'trace.jsonl'
            with patch.object(sys,'argv',[str(path),'--root',str(ROOT),'--out',str(output)]):
                self.assertEqual(ns['main'](),2)
            result=json.loads(output.with_suffix('.summary.json').read_text())
            self.assertIn('runtime startup failure',result['failure'])
            self.assertEqual(result['frames'],0)
        self.assertEqual(closed,[True]);self.assertEqual(gc.isenabled(),before)

    def test_loop_failure_restores_gc_and_closes_source(self):
        closed=[]
        class Backend:
            def __init__(self,*a,**k):pass
            def close(self):closed.append('backend')
            def state(self):return {}
        class Source:
            received=1
            def __init__(self):self.latest=(b'fixture',time.monotonic_ns())
            def close(self):closed.append('source')
        def failed_decode(*a,**k):
            self.assertFalse(gc.isenabled())
            raise RuntimeError('timed decode failure')
        path=ROOT/'scripts/run_onboard_shadow.py'
        ns=function_namespace(path,{'main'},argparse=argparse,Path=Path,json=json,np=np,time=time,gc=gc,
            live_alignment_quaternion=lambda x:None,NativeBackend=Backend,
            OnboardDualPolicyRuntime=lambda **k:types.SimpleNamespace(reset=lambda:None),
            NativeMotionRunner=lambda x:x,NativeVelocityRunner=lambda x:x,
            warm_native_policies=lambda *a:{},OnboardShadowFlow=lambda *a,**k:None,
            VelocityLimits=lambda **k:None,load_output_envelopes=lambda *a,**k:{'velocity':None},
            InertCommandSink=lambda *a:types.SimpleNamespace(arm=lambda q:types.SimpleNamespace(allowed=True)),
            ReadOnlyStateSource=Source,decode_policy_lowstate_buffer=failed_decode,
            decode_humanoid_lowstate_buffer=lambda *a,**k:types.SimpleNamespace(q_by_slot=np.zeros(44)))
        before=gc.isenabled()
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            output=Path(directory)/'trace.jsonl'
            with patch.object(sys,'argv',[str(path),'--root',str(ROOT),'--out',str(output),'--pico-mode','none']):
                self.assertEqual(ns['main'](),2)
            result=json.loads(output.with_suffix('.summary.json').read_text())
            self.assertIn('timed decode failure',result['failure'])
            self.assertTrue(result['cyclic_gc_disabled_during_timed_loop'])
        self.assertEqual(closed,['backend','source']);self.assertEqual(gc.isenabled(),before)

    def test_warmup_calls_both_models_twenty_times(self):
        calls=[]
        backend=types.SimpleNamespace(run_motion=lambda x:calls.append(('motion',x.shape)),
            run_velocity=lambda x:calls.append(('velocity',x.shape)))
        path=ROOT/'scripts/run_onboard_shadow.py'
        ns=function_namespace(path,{'warm_native_policies'},np=np,time=time)
        result=ns['warm_native_policies'](backend,ROOT)
        self.assertEqual(len(calls),40)
        self.assertEqual(sum(name=='motion' for name,shape in calls),20)
        self.assertEqual(result['iterations_per_policy'],20)

    def source_class(self,node):
        fake=types.SimpleNamespace(Node=node,QoS=types.SimpleNamespace(sensor_data=lambda:None),
            SubscriberOptions=lambda **k:None)
        schema=types.SimpleNamespace(SCHEMA_HASH_HEX='de514c40e02f4cd730a3154970e57f48f8f6ecbeaf0af9ed3f19b2df79bf3765')
        path=ROOT/'src/deployment/vita_s100_aorta_30dof/onboard_shadow.py'
        cls=function_namespace(path,{'ReadOnlyStateSource'},time=time)['ReadOnlyStateSource']
        return cls,{'aorta':fake,'humanoid_low_state_schema_meta':schema}

    def test_subscribe_failure_closes_node(self):
        closed=[]
        class Node:
            def __init__(self,*a):pass
            def create_subscriber(self,*a,**k):raise RuntimeError('subscribe failed')
            def close(self):closed.append(True)
        cls,modules=self.source_class(Node)
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,'subscribe failed'):cls()
        self.assertEqual(closed,[True])

    def test_subscriber_close_failure_still_closes_node(self):
        closed=[]
        class Subscriber:
            def close(self):raise RuntimeError('subscriber close failed')
        class Node:
            def __init__(self,*a):pass
            def create_subscriber(self,*a,**k):return Subscriber()
            def close(self):closed.append(True)
        cls,modules=self.source_class(Node)
        with patch.dict(sys.modules,modules):source=cls()
        with self.assertRaisesRegex(RuntimeError,'subscriber close failed'):source.close()
        self.assertEqual(closed,[True])

    def test_dead_pico_service_fails_wrapper_and_terminates_policy(self):
        class Child:
            pid=1
            def __init__(self,status):self.status=status;self.terminated=False
            def poll(self):return self.status
            def terminate(self):self.terminated=True;self.status=-15
            def wait(self,**k):return self.status
        service=Child(7);policy=Child(None)
        path=ROOT/'scripts/run_shadow_with_pico_service.py'
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            root=Path(directory);(root/'build/pico').mkdir(parents=True);(root/'outputs').mkdir()
            ns=function_namespace(path,{'main'},Path=Path,os=os,json=json,signal=signal,subprocess=subprocess,sys=sys,time=time)
            ns['__file__']=str(root/'scripts/wrapper.py')
            with patch.object(subprocess,'Popen',side_effect=[service,policy]),patch.object(signal,'signal'),patch.dict(os.environ,{'VITA_RR_PRIORITY':'1'}):
                with self.assertRaisesRegex(RuntimeError,'Pico service exited'):ns['main']()
            self.assertTrue(policy.terminated)
            self.assertTrue((root/'outputs/onboard_shadow_telemetry.json').exists())

if __name__=='__main__':unittest.main()
