"""Fixed NPZ references with real policy math and fake robot I/O; no motors."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from deployment.vita_s100_aorta_30dof.actor_motion import load_motion
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_pico_live import run_live
from test_actor_pico_live import Clock, Loopback, robot_config


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_runtime.create_retargeter',
                   side_effect=AssertionError('File source must not construct GMR')):
            cls.runtimes = {p: ActorPicoRuntime(ROOT, policy=p, backend='ort', reference_source='npz')
                            for p in ('sonic', 'supermimic', 'morphmimic')}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.runtime = self.runtimes['sonic']
        self.config = self.folder/'motion.json'
        self.file = self.folder/'clip.npz'
        self.config.write_text(json.dumps(dict(schema='vita_actor_motion_v1',
            path=str(self.file), end_hold_s=.04, root_body_index=0)))
        c = self.runtime.model.contract
        self.data = dict(joint_names=np.array(c.names), joint_pos=np.tile(c.default, (5, 1)),
            body_pos_w=np.tile([[[2., 3., .891088]]], (5, 1, 1)),
            body_quat_w=np.tile([[[1., 0., 0., 0.]]], (5, 1, 1)), fps=np.array(25.))

    def source(self, runtime=None):
        np.savez(self.file, **self.data)
        return load_motion(runtime or self.runtime, self.config)

    def test_named_mapping_resampling_all_policies_and_no_gmr(self):
        self.data['joint_names'] = self.data['joint_names'][::-1]
        self.data['joint_pos'] = self.data['joint_pos'][:, ::-1]
        for name, runtime in self.runtimes.items():
            with self.subTest(policy=name):
                source = self.source(runtime)
                expected = self.data['joint_pos'][0, [list(self.data['joint_names']).index(n)
                                                     for n in runtime.model.contract.names]]
                np.testing.assert_allclose(source.motion.q, np.tile(expected, (9, 1)))
                np.testing.assert_allclose(source.motion.base[:, :2], 0.)
                self.assertIsNone(runtime.retarget)
                self.assertIsNone(runtime.head)
                runtime.reset_inputs()
                source.begin(runtime.model.contract.default, [1., 0., 0., 0.])
                while not source.finished:
                    command, obs, row = source.infer(q=runtime.model.contract.default,
                        dq=np.zeros(28), root=np.array([1., 0., 0., 0.]), gyro=np.zeros(3))
                    self.assertEqual(len(obs), runtime.model.contract.input_dim)
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertEqual(len(command.joints), 31)
                    source.advance()

    def test_window_features_equal_full_motion_including_end(self):
        names = list(self.data['joint_names'])
        self.data['joint_pos'] = np.tile(self.runtime.model.contract.default, (80, 1))
        self.data['joint_pos'][:, names.index('waist_yaw_joint')] += .08*np.sin(np.arange(80)*.1)
        self.data['body_pos_w'] = np.tile([[[2., 3., .891088]]], (80, 1, 1))
        self.data['body_pos_w'][:, 0, 0] += np.arange(80)*.001
        self.data['body_quat_w'] = np.tile([[[1., 0., 0., 0.]]], (80, 1, 1))
        for name, runtime in self.runtimes.items():
            source = self.source(runtime)
            root = runtime.math.yaw_quaternion(.3)
            source.begin(runtime.model.contract.default, root)
            for tick in [0, 1, source.lead_frames, 40, source.total_ticks-1]:
                with self.subTest(policy=name, tick=tick):
                    source.tick = tick
                    window, t = source.sample()
                    self.assertLessEqual(len(window.q), runtime.reference.steps)
                    args = (runtime.model.contract.default, np.zeros(28), root, np.zeros(3), source.yaw_offset)
                    runtime.policy.reset()
                    full = runtime.policy.observe(source.prepared, tick*.02, *args)
                    runtime.policy.reset()
                    sliced = runtime.policy.observe(window, t, *args)
                    np.testing.assert_allclose(sliced, full, atol=2e-6, rtol=1e-6)

    def test_sdk_base_layout_and_opposite_quaternion_signs(self):
        self.data['base_pos_w'] = self.data.pop('body_pos_w')[:, 0]
        self.data['base_quat_w'] = self.data.pop('body_quat_w')[:, 0]
        self.data['base_quat_w'][1::2] *= -1
        self.data['framerate'] = self.data.pop('fps')
        source = self.source()
        np.testing.assert_allclose(np.abs(source.motion.quat[:, 0]), 1.)
        self.assertEqual(source.metadata['root_layout'], 'base')

    def test_reject_bad_data(self):
        changes = [
            ('missing joint', lambda d: d.update(joint_names=np.array(['bad']+list(d['joint_names'][1:])))),
            ('duplicate', lambda d: d['joint_names'].__setitem__(0, d['joint_names'][1])),
            ('nonfinite', lambda d: d['joint_pos'].__setitem__((0, 0), np.nan)),
            ('joint limit', lambda d: d['joint_pos'].__setitem__((0, 0), 99.)),
            ('quaternion', lambda d: d['body_quat_w'].__setitem__((0, 0), [0., 0., 0., 0.])),
            ('framerate', lambda d: d.update(fps=np.array(0.))),
            ('rate conflict', lambda d: d.update(framerate=np.array(50.))),
            ('motor coordinates', lambda d: d.update(joint_space=np.array('motor_theta'))),
            ('ambiguous motor metadata', lambda d: d.update(control_space=np.array('motor_theta'))),
            ('ambiguous root', lambda d: d.update(base_pos_w=np.zeros((5,3)), base_quat_w=np.tile([1,0,0,0],(5,1)))),
            ('root mismatch', lambda d: d.update(body_pos_w=np.zeros((4,1,3)))),
        ]
        original = {k:v.copy() for k,v in self.data.items()}
        for name, change in changes:
            with self.subTest(case=name):
                self.data = {k:v.copy() for k,v in original.items()}
                change(self.data)
                with self.assertRaises(ValueError): self.source()

    def test_hash_and_fixed_waist_roll(self):
        cfg = json.loads(self.config.read_text()); cfg['sha256'] = '0'*64
        self.config.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'SHA256'): self.source()
        cfg.pop('sha256'); self.config.write_text(json.dumps(cfg))
        self.data['joint_names'] = np.append(self.data['joint_names'], 'waist_roll_joint')
        self.data['joint_pos'] = np.column_stack((self.data['joint_pos'], np.full(5, .1)))
        with self.assertRaisesRegex(ValueError, 'waist_roll'): self.source()

    def test_verified_aliases_and_root_name(self):
        self.data.update(base_pos_w=self.data['body_pos_w'][:,0].copy(),
                         base_quat_w=self.data['body_quat_w'][:,0].copy(),
                         body_names=np.array(['base_link']), root_body=np.array('base_link'))
        self.assertEqual(self.source().metadata['root_layout'], 'body_with_verified_base_aliases')
        self.data['base_pos_w'][1,0] += .01
        with self.assertRaisesRegex(ValueError, 'Conflicting'): self.source()
        self.data['base_pos_w'] = self.data['body_pos_w'][:,0].copy()
        self.data['root_body'] = np.array('waist_pitch_link')
        with self.assertRaisesRegex(ValueError, 'root_body'): self.source()

    def test_explicit_bounded_reference_projection_receipt(self):
        names = list(self.data['joint_names']); j = names.index('left_ankle_roll_joint')
        upper = self.runtime.config['body_profile']['joint_limits'][names[j]][1]
        self.data['joint_pos'][2,j] = upper+8.2e-5
        with self.assertRaisesRegex(ValueError, 'outside virtual limits'): self.source()
        cfg = json.loads(self.config.read_text()); cfg['limit_projection_max_rad'] = 1e-4
        self.config.write_text(json.dumps(cfg)); source = self.source()
        changed = source.metadata['limit_corrections']
        self.assertEqual(len(changed),1); self.assertEqual(changed[0]['frame'],2)
        self.assertEqual(changed[0]['projected_rad'], upper)
        self.assertLessEqual(source.motion.q[:,j].max(),upper)
        self.data['joint_pos'][2,j] = upper+.001
        with self.assertRaisesRegex(ValueError, 'outside virtual limits'): self.source()
        cfg['limit_projection_max_rad'] = .01; self.config.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'limit_projection'): self.source()

    def test_command_path_keeps_model_inputs_and_raw_actions_identical(self):
        for name, runtime in self.runtimes.items():
            if getattr(runtime.model.contract, 'ankle', None) is None:
                with self.assertRaisesRegex(ValueError, 'motor_theta model'):
                    ActorPicoRuntime(ROOT, policy=name, backend='ort', reference_source='npz',
                                     ankle_command_space='motor_theta')
                continue
            native = ActorPicoRuntime(ROOT,policy=name,backend='ort',reference_source='npz',ankle_command_space='motor_theta')
            legacy = ActorPicoRuntime(ROOT,policy=name,backend='ort',reference_source='npz',ankle_command_space='virtual_pitch_roll')
            sources = [self.source(r) for r in (native,legacy)]
            for source,r in zip(sources,(native,legacy)):
                source.begin(r.model.contract.default,np.array([1.,0.,0.,0.]))
            while not sources[0].finished:
                results = [s.infer(q=r.model.contract.default,dq=np.zeros(28),root=np.array([1.,0.,0.,0.]),gyro=np.zeros(3))
                           for s,r in zip(sources,(native,legacy))]
                np.testing.assert_array_equal(results[0][1],results[1][1])
                np.testing.assert_array_equal(results[0][2]['raw_action'],results[1][2]['raw_action'])
                self.assertEqual(results[0][2]['decoded']['ankle_command_space'],'motor_theta')
                self.assertEqual(results[1][2]['decoded']['ankle_command_space'],'virtual_pitch_roll')
                for s in sources:s.advance()

    def setup_live(self, auto=False):
        from locomotion_aorta.pull import PullController
        self.runtime.reset_inputs()
        self.clock, self.stop, self.rows = Clock(), threading.Event(), []
        self.transport = Loopback(self.runtime, self.clock)
        self.controller = PullController(group='default', client_id=0x5eed, _transport=self.transport)
        self.addCleanup(self.controller.close)
        config = self.folder/'robot.json'
        config.write_text(json.dumps(robot_config(self.runtime)))
        self.args = NS(robot_config=config, robot_id='TEST-ONLY', startup_timeout=5.,
            seconds=20., stand_only=False, auto_start=auto, manual_pico=False)
        self.clip = self.source()
        self.startup_calls = []
        if auto:
            self.transport.state_fault = lambda s: replace(s, q=np.zeros(31,np.float32)) if not self.transport.commands else s

    def run_live(self, poll_command=lambda:None):
        def action(name):
            self.startup_calls.append(name)
            self.transport.value = replace(self.transport.value,
                current_action='FIXED_STAND' if name=='FIXED_STAND' else 'PASSIVE')
        def no_pico(): raise AssertionError('File playback must not access Pico service')
        with redirect_stdout(io.StringIO()):
            return run_live(self.runtime, self.args, lambda row, obs=None:self.rows.append(row),
                self.controller, None, self.stop, no_pico, clock=self.clock,
                motion_source=self.clip, request_action=action, poll_command=poll_command)

    def test_live_file_completion_and_existing_auto_startup(self):
        self.setup_live(auto=True)
        result = self.run_live()
        self.assertEqual(self.startup_calls, ['FIXED_STAND'])
        self.assertEqual(result['stop_reason'], 'motion_complete')
        self.assertTrue(result['exit_confirmed'])
        rows = [r for r in self.rows if r.get('stage') == 'motion_tracking']
        self.assertEqual([r['motion_tick'] for r in rows], list(range(self.clip.total_ticks)))
        self.assertEqual([r['reference_frame'] for r in rows], list(range(self.clip.total_ticks)))
        self.assertTrue(all(r['state_id'] > 0 for r in rows))
        self.assertFalse(any(r.get('stage') == 'pico_tracking' for r in self.rows))
        self.assertEqual([r.operation.value for r in self.transport.requests], ['PREPARE','COMMIT','CANCEL'])

    def test_stale_feedback_cancels_and_file_does_not_advance(self):
        self.setup_live()
        def fail(command):
            if self.clip.prepared is not None and self.clip.tick == 1:
                self.transport.state_fault = lambda s: replace(s, received_monotonic_ns=s.received_monotonic_ns-1000000000)
        self.transport.after_publish = fail
        with self.assertRaises(TimeoutError): self.run_live()
        self.assertEqual(self.clip.tick, 2)
        self.assertEqual(self.controller.status.current_action, 'PASSIVE')

    def test_duplicate_feedback_does_not_skip_a_motion_frame(self):
        self.setup_live()
        injected = []
        def duplicate(command):
            if self.clip.prepared is not None and self.clip.tick == 2 and not injected:
                cached = self.transport.wait_state()
                injected.append(True)
                def once(fresh):
                    self.transport.state_fault = None
                    return replace(cached, received_monotonic_ns=fresh.received_monotonic_ns,
                                   ts_state_pub_us=fresh.ts_state_pub_us)
                self.transport.state_fault = once
        self.transport.after_publish = duplicate
        result = self.run_live()
        self.assertEqual(result['duplicate_body_ticks'], 1)
        self.assertTrue(result['exit_confirmed'])
        rows = [r for r in self.rows if r.get('stage') == 'motion_tracking']
        self.assertEqual([r['motion_tick'] for r in rows], list(range(self.clip.total_ticks)))

    def test_operator_stop_cancels_during_file(self):
        self.setup_live()
        def stop(command):
            if self.clip.prepared is not None and self.clip.tick == 2: self.stop.set()
        self.transport.after_publish = stop
        result = self.run_live()
        self.assertEqual(result['stop_reason'], 'operator_stop')
        self.assertTrue(result['exit_confirmed'])
        self.assertEqual(self.clip.tick, 3)

    def test_terminal_stop_cancels_without_another_file_command(self):
        from deployment.vita_s100_aorta_30dof.actor_pico_live import TerminalCommands
        self.setup_live()
        terminal = TerminalCommands('TEST-ONLY', self.stop)
        def poll():
            if self.clip.prepared is not None and self.clip.tick == 2:
                with patch('sys.stdin', io.StringIO('STOP\n')), patch('select.select', return_value=([0], [], [])):
                    terminal.poll()
        result = self.run_live(poll_command=poll)
        self.assertEqual(result['stop_reason'], 'operator_stop')
        self.assertTrue(result['exit_confirmed'])
        self.assertEqual(self.clip.tick, 2)

    def test_cli_file_mode_overrides_private_pico_handoff_defaults(self):
        from deployment.vita_s100_aorta_30dof import actor_pico_local
        self.source()
        spec = importlib.util.spec_from_file_location('actor_motion_cli_test', ROOT/'scripts/actor_pico.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        defaults = dict(robot_id='TEST-ONLY', robot_config=self.folder/'robot.yaml',
                        seconds=0, wait_pico=True, auto_pico=True)
        captured = []
        def live(runtime, args, write):
            captured.append(args)
            self.assertIsNone(runtime.retarget)
            return dict(actions_sent=0)
        with (redirect_stdout(io.StringIO()),
              patch.object(actor_pico_local, 'load_local_start', return_value=defaults),
              patch.object(cli, 'live', side_effect=live),
              patch.object(sys, 'argv', ['actor_pico.py', 'live', '--auto-start', '--local-start',
                  '--confirm-motion', '--motion-config', str(self.config), '--log', str(self.folder/'cli.jsonl')])):
            cli.main()
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0].manual_pico or captured[0].wait_pico or captured[0].auto_pico)
        self.assertTrue(captured[0].auto_start)


if __name__ == '__main__':
    unittest.main()
