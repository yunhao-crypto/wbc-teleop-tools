"""Mock-only operator subprocess lifecycle. Never starts SDK or real children."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

def load_service():
    spec=importlib.util.spec_from_file_location('operator_service_under_test',ROOT/'scripts/operator_service.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class ServiceTests(unittest.TestCase):
    def exercise(self,scenario,mode='suspended-test'):
        module=load_service();events=[];processes=[];handlers={}
        def signal(number,handler):old=handlers.get(number,0);handlers[number]=handler;return old
        class Process:
            def __init__(self,role):self.role=role;self.returncode=None;self.terminating=False;self.killed=False;self.triggered=False
            def poll(self):
                if self.role=='service' and scenario=='service_failure':self.returncode=9
                return self.returncode
            def terminate(self):events.append((self.role,'terminate'));self.terminating=True
            def kill(self):events.append((self.role,'kill'));self.killed=True
            def wait(self,timeout=None):
                if timeout is None:raise AssertionError('Unbounded child wait')
                events.append((self.role,'wait',timeout))
                if self.role=='policy' and scenario in ('sigint','sigterm','hung','unkillable') and not self.triggered and timeout==.25:
                    self.triggered=True;handlers[2 if scenario=='sigint' else 15](2 if scenario=='sigint' else 15,None)
                    raise subprocess.TimeoutExpired('mock-policy',timeout)
                if self.role=='policy' and scenario in ('hung','unkillable') and self.terminating:
                    if not self.killed or scenario=='unkillable':raise subprocess.TimeoutExpired('mock-policy',timeout)
                self.returncode=-15 if self.terminating else (7 if scenario=='child_failure' and self.role=='policy' else 0)
                events.append((self.role,'reaped'));return self.returncode
        def popen(argv,**kwargs):
            role='service' if str(argv[0]).endswith('RoboticsServiceProcess') else 'policy'
            events.append((role,'spawn',tuple(argv)))
            child=Process(role);processes.append(child)
            if role=='policy' and scenario=='signal_during_spawn':handlers[15](15,None)
            return child
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            work=Path(directory);(work/'build/pico').mkdir(parents=True);(work/'outputs').mkdir();(work/'scripts').mkdir()
            with patch.object(module,'__file__',str(work/'scripts/operator_service.py')),patch.object(module.sys,'argv',['operator_service.py',mode]),patch.object(module.signal,'signal',signal),patch.object(module.subprocess,'Popen',popen):
                if scenario in ('service_failure','unkillable'):
                    with self.assertRaisesRegex(RuntimeError,'service exited|could not confirm'):module.main()
                    result=None
                else:result=module.main()
        return result,events,processes
    def test_check_never_spawns_pico_service_or_chrt(self):
        result,events,processes=self.exercise('success','check');self.assertEqual(result,0)
        self.assertEqual([p.role for p in processes],['policy'])
        command=next(e[2] for e in events if e[:2]==('policy','spawn'))
        self.assertNotEqual(command[0],'chrt');self.assertIn('check',command)
    def test_sigint_sigterm_and_spawn_race_reap_policy_before_service(self):
        for scenario in ('sigint','sigterm','signal_during_spawn'):
            with self.subTest(scenario=scenario):
                code,events,_=self.exercise(scenario)
                self.assertEqual(code,130 if scenario=='sigint' else 143)
                self.assertLess(events.index(('policy','terminate')),events.index(('service','terminate')))
                self.assertLess(events.index(('policy','reaped')),events.index(('service','terminate')))
                self.assertEqual(sum(e==('policy','wait',15) for e in events),1)
    def test_child_failure_propagates_then_cleans_service(self):
        result,events,_=self.exercise('child_failure');self.assertEqual(result,7)
        self.assertLess(events.index(('policy','reaped')),events.index(('service','terminate')))
    def test_service_failure_cancels_policy(self):
        _,events,_=self.exercise('service_failure');self.assertIn(('policy','terminate'),events);self.assertIn(('policy','reaped'),events)
    def test_timeout_uses_one_grace_then_bounded_kill_wait(self):
        code,events,_=self.exercise('hung');self.assertEqual(code,143)
        self.assertEqual(sum(e==('policy','wait',15) for e in events),1)
        self.assertIn(('policy','wait',2),events)
        self.assertLess(events.index(('policy','kill')),events.index(('policy','reaped')))
        self.assertLess(events.index(('policy','reaped')),events.index(('service','terminate')))
    def test_unreapable_policy_reports_failure_and_still_cleans_service(self):
        _,events,_=self.exercise('unkillable');self.assertIn(('policy','wait',2),events)
        self.assertIn(('service','terminate'),events);self.assertIn(('service','reaped'),events)

if __name__=='__main__':unittest.main()
