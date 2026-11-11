"""OP3 gripper gate, config, Info checks and publication with fake EE clients; no hardware."""
import copy
from dataclasses import dataclass, replace
import json
import multiprocessing
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import actor_pico_gripper as module
from deployment.vita_s100_aorta_30dof.actor_pico_buttons import TriggerReader
from deployment.vita_s100_aorta_30dof.actor_pico_gripper import (
    GraspMonitor, GripperRunner, SideGripGate, load_config, load_sdk, validate_config)

EXAMPLE = ROOT/'config/pico_op3_gripper.example.json'
SITE = ROOT/'venv/lib/python3.10/site-packages'
Field = NS(POS=1, VEL=2, VEL_LIMIT=4, TAU=8, CURRENT=16, TEMPERATURE=32, MODE=64, ERROR_CODE=128)
SIDE = dict(left=1, right=2)


@dataclass(frozen=True)
class Target:
    mode: tuple
    command_fields: tuple
    pos: tuple = None
    vel_limit: tuple = None
    current: tuple = None


@dataclass(frozen=True)
class State:
    side: int
    state_id: int
    pos: tuple = (.05,)
    current: tuple = (0.,)
    valid_fields: tuple = (Field.POS | Field.CURRENT,)
    error_code: tuple = (0,)
    ready: bool = True
    output_allowed: bool = True
    inhibit_reason: int = 0
    status_valid_mask: int = 7
    sample_status: int = 1

    @property
    def sample_ref(self):
        return ('ref', self.side, self.state_id)


def info(which, **changes):
    value = NS(side=SIDE[which], device_model=4, firmware='fw', serial='sn-'+which, axis_names=('opening',),
               position_domain=(2,), is_mock=(False,), supported_modes=((1 << 1) | (1 << 2),),
               supported_command_fields=(Field.POS | Field.VEL_LIMIT | Field.CURRENT,),
               supported_state_fields=(Field.POS | Field.CURRENT | Field.ERROR_CODE,),
               valid_limits=(Field.POS | Field.VEL_LIMIT | Field.CURRENT,),
               pos_min=(-.005,), pos_max=(.112,), vel_limit_max=(.2,), current_min=(0.,), current_max=(3.5,))
    for key, item in changes.items():
        setattr(value, key, item)
    return value


def prepare_target(info, target):
    if not info.supported_modes[0] & (1 << target.mode[0]):
        raise ValueError('unsupported mode')
    if not info.pos_min[0] <= target.pos[0] <= info.pos_max[0]:
        raise ValueError('pos is outside declared SI limits')
    if not 0 <= target.vel_limit[0] <= info.vel_limit_max[0]:
        raise ValueError('vel_limit is outside declared SI limits')
    if not info.current_min[0] <= target.current[0] <= info.current_max[0]:
        raise ValueError('current is outside declared SI limits')
    return target


SDK = NS(codec=NS(validate_info=lambda info: None, validate_state=lambda info, state: None,
                  prepare_target=prepare_target),
         ControlMode=NS(POSITION=1, POSITION_TORQUE=2), DeviceModel=NS(AGILINK_OP3=4, BRAINCO_REVO2=1),
         Field=Field, PositionDomain=NS(TOTAL_OPENING=2, JOINT_ANGLE=1), SampleStatus=NS(OK=1),
         InhibitReason=NS(NONE=0), SampleStream=NS(EE_STATE=2), EndEffectorSide=NS(LEFT=1, RIGHT=2),
         EndEffectorTarget=Target, STATUS_REQUIRED=7)


class FakeClient:
    infos = {}

    def __init__(self, side, *, connection, observation_sources):
        self.side, self.connection, self.observation_sources = side, connection, observation_sources
        self.info = self.infos.get(side) or info(side)
        self.state = State(SIDE[side], 1)
        self.fault = None
        self.received = time.monotonic()
        self.published, self.closed, self.publish_error = [], False, None
        self.lock = threading.Lock()
        FakeClient.created.append(self)

    def wait_info(self, timeout): return self.info
    def wait_state(self, timeout): return self.state
    def wait_output_ready(self, timeout): self.ready_waited = timeout

    def motion_feedback(self):
        with self.lock:
            return self.state, self.state, self.fault, self.received

    def set(self, now, **changes):
        with self.lock:
            self.state = replace(self.state, state_id=self.state.state_id+1, **changes)
            self.received = now

    def publish(self, target, *, input_observations):
        if self.publish_error: raise self.publish_error
        self.published.append((target, input_observations))

    def close(self): self.closed = True


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now


def shared(left=0., right=0., received=0., stamp=0.):
    value = multiprocessing.Array('d', [0.]*7)
    value[3:7] = [received, stamp, left, right]
    return value


class ConfigTests(unittest.TestCase):
    def raw(self): return json.loads(EXAMPLE.read_text())

    def test_example_is_explicit_and_valid(self):
        config = load_config(EXAMPLE)
        self.assertEqual(set(config['sides']), {'left', 'right'})
        self.assertEqual(config['control_mode'], 'POSITION_TORQUE')

    def test_rejects_missing_unknown_and_inconsistent_values(self):
        cases = []
        raw = self.raw(); del raw['rate_hz']; cases.append(raw)
        raw = self.raw(); raw['extra'] = 1; cases.append(raw)
        raw = self.raw(); del raw['sides']['left']['current_a']; cases.append(raw)
        raw = self.raw(); raw['sides']['left']['curent_a'] = 1.; cases.append(raw)
        raw = self.raw(); raw['sides']['left']['open_m'] = 0.; cases.append(raw)
        raw = self.raw(); raw['open_threshold'] = .8; cases.append(raw)
        raw = self.raw(); raw['close_threshold'] = 1.; cases.append(raw)
        raw = self.raw(); raw['rate_hz'] = True; cases.append(raw)
        raw = self.raw(); raw['sides']['right']['current_a'] = 0.; cases.append(raw)
        raw = self.raw(); raw['sides']['right']['vel_limit_m_s'] = float('nan'); cases.append(raw)
        raw = self.raw(); raw['device_model'] = 'BRAINCO_REVO2'; cases.append(raw)
        raw = self.raw(); raw['control_mode'] = 'TORQUE'; cases.append(raw)
        raw = self.raw(); raw['on_fault'] = 'ignore'; cases.append(raw)
        raw = self.raw(); raw['sides'] = {}; cases.append(raw)
        raw = self.raw(); raw['sides'] = {'middle': raw['sides']['left']}; cases.append(raw)
        raw = self.raw(); raw['schema'] = 'other'; cases.append(raw)
        for index, raw in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_config(raw)

    def test_single_side_is_allowed(self):
        raw = self.raw(); del raw['sides']['left']
        self.assertEqual(list(validate_config(raw)['sides']), ['right'])

    def test_missing_end_effector_sdk_refuses_with_clear_error(self):
        with patch.dict(sys.modules, {'locomotion_aorta.end_effector': None}):
            with self.assertRaisesRegex(RuntimeError, 'EndEffectorClient'):
                load_sdk()

    @unittest.skipUnless((SITE/'locomotion_aorta/end_effector.py').is_file(),
                         'pinned SDK is not installed under venv/')
    def test_fake_constants_match_the_pinned_sdk(self):
        # The runner tests use plain numbers; tie them to the pinned SDK enums.
        if str(SITE) not in sys.path:
            sys.path.append(str(SITE))
        real = load_sdk()
        for group in ('ControlMode', 'DeviceModel', 'Field', 'PositionDomain', 'SampleStatus',
                      'InhibitReason', 'SampleStream', 'EndEffectorSide'):
            for name, value in vars(getattr(SDK, group)).items():
                with self.subTest(constant=group+'.'+name):
                    self.assertEqual(int(getattr(getattr(real, group), name)), value)
        self.assertEqual(real.STATUS_REQUIRED, SDK.STATUS_REQUIRED)

class GateTests(unittest.TestCase):
    def setUp(self):
        self.gate = SideGripGate(load_config(EXAMPLE))
        self.stamp = 0

    def feed(self, value, now, received=None, stamp=None):
        if stamp is None:
            self.stamp += 1; stamp = self.stamp
        return self.gate.update(now if received is None else received, stamp, value, now)

    def test_held_grip_needs_release_then_debounce(self):
        self.assertFalse(self.feed(1., 1.))
        self.assertFalse(self.feed(1., 1.1))  # held since connect: never closes
        self.assertFalse(self.feed(.1, 1.2))
        self.assertFalse(self.feed(.9, 1.21))
        self.assertFalse(self.feed(.9, 1.25))
        self.assertTrue(self.feed(.9, 1.27))
        self.assertTrue(self.feed(.5, 1.3))   # hysteresis band holds closed
        self.assertFalse(self.feed(.3, 1.31))  # release opens immediately
        self.assertFalse(self.feed(.5, 1.32))

    def test_debounce_restarts_when_grip_dips(self):
        self.feed(0., 1.)
        self.feed(.9, 1.01)
        self.feed(.6, 1.04)
        self.assertFalse(self.feed(.9, 1.07))
        self.assertTrue(self.feed(.9, 1.12))

    def test_invalid_stale_frozen_and_regressed_input_opens_and_disarms(self):
        def closed():
            self.gate = SideGripGate(load_config(EXAMPLE)); self.stamp = 0
            self.feed(0., 1.); self.feed(1., 1.01)
            self.assertTrue(self.feed(1., 1.1))
        closed(); self.assertFalse(self.feed(1., 1.5, received=1.2))           # stale receipt
        closed(); self.assertFalse(self.gate.update(1.35, self.stamp, 1., 1.35))  # frozen stamp
        closed(); self.assertFalse(self.feed(1., 1.11, stamp=1))                # regressed stamp
        for bad in (float('nan'), 1.2, -.1):
            closed(); self.assertFalse(self.feed(bad, 1.11))
        closed(); self.assertFalse(self.feed(1., 1.11, received=0.))           # reader marked unavailable
        self.assertFalse(self.feed(1., 1.12)); self.assertFalse(self.feed(1., 1.3))  # still disarmed
        self.feed(0., 1.31); self.feed(1., 1.32)
        self.assertTrue(self.feed(1., 1.4))


class MonitorTests(unittest.TestCase):
    def test_grasped_closed_empty_and_current(self):
        spec = load_config(EXAMPLE)['sides']['left']
        monitor = GraspMonitor(spec)
        self.assertEqual(monitor.update(True, 0., .08, 0.), 'closing')
        for t, x in ((.1, .05), (.2, .031), (.3, .0302), (.45, .0301), (.65, .0301)):
            state = monitor.update(True, t, x, .2)
        self.assertEqual(state, 'grasped')
        self.assertEqual(monitor.update(False, .7, .0301, 0.), 'open')
        for t, x in ((1., .08), (1.2, .02), (1.4, .0), (1.6, .0), (1.8, .0)):
            state = monitor.update(True, t, x, .1)
        self.assertEqual(state, 'closed_empty')
        monitor = GraspMonitor(spec)
        for t, x in ((0., .08), (.2, .05), (.35, .04)):
            state = monitor.update(True, t, x, .9)
        self.assertEqual(state, 'grasped')  # still moving but current near the limit


class RunnerTests(unittest.TestCase):
    def setUp(self):
        FakeClient.created, FakeClient.infos = [], {}
        self.sdk = NS(**vars(SDK), EndEffectorClient=FakeClient)
        self.config = load_config(EXAMPLE)
        self.clock = Clock()
        self.shared = shared()
        self.pulses = []

    def runner(self, config=None, haptics=None):
        out = GripperRunner(config or self.config, self.shared, connection='controller', sdk=self.sdk,
                            haptics=haptics, clock=self.clock)
        self.clients = {c.side: c for c in FakeClient.created}
        return out

    def grips(self, left, right, stamp):
        self.shared[3:7] = [self.clock.now, stamp, left, right]

    def advance(self, runner, dt, left, right, stamp, **state):
        self.clock.now += dt
        for client in self.clients.values():
            client.set(self.clock.now, **state)
        self.grips(left, right, stamp)
        runner.tick()

    def test_startup_validates_both_sides_before_entry(self):
        runner = self.runner()
        ready = [r for r in runner.drain() if r['record_type'] == 'gripper_ready']
        self.assertEqual([r['side'] for r in ready], ['left', 'right'])
        self.assertEqual(ready[0]['current_limits_a'], [0., 3.5])
        for side, client in self.clients.items():
            self.assertEqual(client.connection, 'controller')
            self.assertEqual(client.observation_sources, ((2, SIDE[side]),))
            self.assertEqual(client.ready_waited, self.config['startup_timeout_s'])
            close, opened = runner.targets[side][True], runner.targets[side][False]
            self.assertEqual((close.pos, close.current, close.vel_limit), ((0.,), (1.,), (.1,)))
            self.assertEqual(opened.pos, (.08,))
            self.assertEqual(close.mode, (2,))
            self.assertEqual(close.command_fields, (Field.POS | Field.VEL_LIMIT | Field.CURRENT,))
        self.assertEqual(runner.publishes, dict(left=0, right=0))

    def test_rejects_endpoints_that_are_not_controllable_op3(self):
        cases = dict(
            model=dict(device_model=3), side=dict(side=2), axes=dict(axis_names=('a', 'b')),
            domain=dict(position_domain=(1,)), mock=dict(is_mock=(True,)),
            mode=dict(supported_modes=(1 << 1,)),
            current=dict(supported_command_fields=(Field.POS | Field.VEL_LIMIT,)),
            limits=dict(valid_limits=(Field.POS | Field.VEL_LIMIT,)),
            feedback=dict(supported_state_fields=(Field.CURRENT,)),
            current_range=dict(current_max=(.5,)), opening_range=dict(pos_max=(.05,)))
        for name, change in cases.items():
            with self.subTest(name=name):
                FakeClient.created, FakeClient.infos = [], dict(left=info('left', **change))
                with self.assertRaisesRegex(ValueError, 'left'):
                    self.runner()
                self.assertTrue(all(c.closed for c in FakeClient.created))

    def test_sides_are_independent_and_stale_input_opens(self):
        runner = self.runner(); runner.started_at = self.clock.now
        self.advance(runner, .02, 0., 0., 1)
        self.advance(runner, .02, 1., 0., 2)
        self.advance(runner, .06, 1., 0., 3)
        left, right = self.clients['left'], self.clients['right']
        self.assertEqual(left.published[-1][0], runner.targets['left'][True])
        self.assertEqual(left.published[-1][1], (('ref', 1, left.state.state_id),))
        self.assertEqual(right.published[-1][0], runner.targets['right'][False])
        self.advance(runner, .02, 1., 1., 4)  # right was armed by its release: closes after hold
        self.advance(runner, .06, 1., 1., 5)
        self.assertEqual(right.published[-1][0], runner.targets['right'][True])
        self.clock.now += .3
        for client in self.clients.values(): client.set(self.clock.now)
        runner.tick()  # PICO samples stopped arriving
        self.assertEqual(left.published[-1][0], runner.targets['left'][False])
        self.assertEqual(right.published[-1][0], runner.targets['right'][False])
        rows = [r for r in runner.drain() if r['record_type'] == 'gripper_command']
        self.assertEqual([(r['side'], r['command'], r['reason']) for r in rows],
                         [('left', 'open', 'release'), ('right', 'open', 'release'),
                          ('left', 'close', 'grip'), ('right', 'close', 'grip'),
                          ('left', 'open', 'input_invalid'), ('right', 'open', 'input_invalid')])
        self.assertEqual(rows[2]['opening_m'], .05)

    def test_waits_for_output_permission_then_faults_on_timeout_or_revocation(self):
        runner = self.runner(); runner.started_at = self.clock.now
        for client in self.clients.values(): client.set(self.clock.now, output_allowed=False, inhibit_reason=3)
        self.advance(runner, .5, 0., 0., 1, output_allowed=False, inhibit_reason=3)
        self.assertEqual([c.published for c in self.clients.values()], [[], []])
        self.assertEqual(runner.disabled, set())
        self.advance(runner, .5, 0., 0., 2, output_allowed=True, inhibit_reason=0)
        self.assertTrue(all(c.published for c in self.clients.values()))
        self.clients['left'].set(self.clock.now+.02, ready=False)
        self.clock.now += .02; self.grips(0., 0., 3); runner.tick()
        self.assertEqual(runner.disabled, {'left'})
        runner.check()  # open_and_disable_side: body keeps running
        before = len(self.clients['right'].published)
        self.advance(runner, .02, 0., 0., 4)
        self.assertEqual(len(self.clients['right'].published), before+1)
        fault = [r for r in runner.drain() if r['record_type'] == 'gripper_fault']
        self.assertEqual(fault[0]['side'], 'left'); self.assertIn('not allowed', fault[0]['error'])
        self.assertTrue(fault[0]['open_published'])  # a faulted side still opens
        self.assertEqual(self.clients['left'].published[-1][0], runner.targets['left'][False])
        runner = self.runner(); runner.started_at = self.clock.now
        self.advance(runner, 2.5, 0., 0., 5, output_allowed=False, inhibit_reason=3)
        self.assertEqual(runner.disabled, {'left', 'right'})

    def test_state_arriving_during_tick_is_not_a_negative_age(self):
        runner = self.runner(); runner.started_at = self.clock.now
        self.grips(0., 0., 1)
        for client in self.clients.values():
            client.set(self.clock.now+.001)  # callback lands after the tick started
        ticks = iter(self.clock.now+i*.002 for i in range(10))
        runner.clock = lambda: next(ticks)
        runner.tick()
        self.assertEqual(runner.disabled, set())
        self.assertTrue(all(c.published for c in self.clients.values()))

    def test_stop_session_fault_propagates_to_body_loop(self):
        config = copy.deepcopy(self.config); config['on_fault'] = 'stop_session'
        runner = self.runner(config); runner.started_at = self.clock.now
        self.advance(runner, .02, 0., 0., 1, error_code=(17,), valid_fields=(Field.POS | Field.ERROR_CODE,))
        with self.assertRaisesRegex(RuntimeError, 'gripper control failed') as caught:
            runner.check()
        self.assertIn('error_code', str(caught.exception.__cause__))

    def test_stale_or_out_of_range_feedback_faults(self):
        for change in (dict(received=-1.), dict(pos=(.5,)), dict(sample_status=2), dict(valid_fields=(Field.CURRENT,))):
            with self.subTest(change=change):
                FakeClient.created = []
                runner = self.runner(); runner.started_at = self.clock.now
                self.clock.now += .02; self.grips(0., 0., 1)
                self.clients['right'].set(self.clock.now)
                client = self.clients['left']
                received = change.pop('received', None)
                client.set(self.clock.now if received is None else self.clock.now-.5, **change)
                runner.tick()
                self.assertIn('left', runner.disabled)
                self.assertNotIn('right', runner.disabled)

    def test_grasp_detection_logs_and_pulses_haptics(self):
        class Haptics:
            def __init__(s): s.calls = []
            def pulse(s, side, event): s.calls.append((side, event))
        haptics = Haptics()
        runner = self.runner(haptics=haptics); runner.started_at = self.clock.now
        self.advance(runner, .02, 0., 0., 1)
        stamp = 2
        for opening in (.07, .05, .032, .0301, .0301, .0301, .0301, .0301):
            self.advance(runner, .1, 1., 0., stamp, pos=(opening,), current=(.5,)); stamp += 1
        self.assertEqual(haptics.calls, [('left', 'grasped')])
        rows = [r for r in runner.drain() if r['record_type'] == 'gripper_grasp' and r['side'] == 'left']
        self.assertEqual([r['state'] for r in rows], ['closing', 'grasped'])
        self.assertEqual(rows[-1]['opening_m'], .0301)

    def test_failing_haptics_never_affects_control(self):
        class Broken:
            def pulse(self, side, event): raise OSError('no vibration')
        runner = self.runner(haptics=Broken()); runner.started_at = self.clock.now
        self.advance(runner, .02, 0., 0., 1)
        stamp = 2
        for opening in (.05, .03, .03, .03, .03, .03):
            self.advance(runner, .1, 1., 0., stamp, pos=(opening,)); stamp += 1
        self.assertIsNone(runner.haptics)
        self.assertEqual(runner.disabled, set())
        self.assertTrue(any(r['record_type'] == 'gripper_haptics_failed' for r in runner.drain()))

    def test_close_keeps_the_last_command_so_a_grasp_is_held(self):
        runner = GripperRunner(copy.deepcopy(self.config), self.shared, connection='c', sdk=self.sdk)
        clients = {c.side: c for c in FakeClient.created}
        left, right = clients['left'], clients['right']
        close_left = runner.targets['left'][True]
        runner.start()
        stamp, deadline, released_until = 0, time.monotonic()+3., time.monotonic()+.2
        while time.monotonic() < deadline:
            now = time.monotonic(); stamp += 1
            for client in clients.values(): client.set(now)
            # Release first (arms the side), then hold only the left grip.
            self.shared[3:7] = [now, stamp, 0. if now < released_until else 1., 0.]
            if left.published and left.published[-1][0] == close_left:
                break
            time.sleep(.005)
        self.assertEqual(left.published[-1][0], close_left)
        runner.close()
        counts = {side: len(c.published) for side, c in clients.items()}
        time.sleep(.05)
        self.assertFalse(runner.thread.is_alive())
        for side, client in clients.items():
            self.assertTrue(client.closed)
            self.assertEqual(len(client.published), counts[side])  # nothing after close
        self.assertEqual(left.published[-1][0], close_left)
        self.assertEqual(right.published[-1][0], runner.targets['right'][False])
        closed = [r for r in runner.drain() if r['record_type'] == 'gripper_closed']
        self.assertEqual(closed[0]['disabled'], [])
        self.assertEqual(closed[0]['held_last_command'], dict(left='closed', right='open'))

    def test_thread_failure_stops_the_session_without_opening(self):
        runner = self.runner(); runner.started_at = self.clock.now
        self.advance(runner, .02, 0., 0., 1)
        self.advance(runner, .06, 1., 0., 2)
        self.advance(runner, .06, 1., 0., 3)
        published = {side: list(c.published) for side, c in self.clients.items()}
        self.assertEqual(published['left'][-1][0], runner.targets['left'][True])
        runner.tick = lambda: (_ for _ in ()).throw(OSError('shared input lost'))
        runner.run()
        with self.assertRaisesRegex(RuntimeError, 'gripper control failed'):
            runner.check()
        self.assertEqual({side: c.published for side, c in self.clients.items()}, published)


class ReaderTests(unittest.TestCase):
    def run_reader(self, grip, shared_value):
        sdk = NS(get_controller_trigger_snapshot=lambda: (True, 5, 0., 0., 0.),
                 get_controller_grip_snapshot=grip)
        reader = TriggerReader(sdk, shared_value)
        reader.start(); time.sleep(.05); reader.close()
        return shared_value

    def test_reader_publishes_same_packet_grips(self):
        value = self.run_reader(lambda: (True, 42, .01, .2, .9), multiprocessing.Array('d', [0.]*7))
        self.assertEqual(value[4:7], [42., .2, .9])
        self.assertLess(abs(time.monotonic()-value[3]), 1.)
        value = self.run_reader(lambda: (False, 0, 0., 0., 0.), multiprocessing.Array('d', [0.]*7))
        self.assertEqual(value[3:7], [0.]*4)

    def test_three_slot_array_ignores_grips(self):
        value = self.run_reader(lambda: (_ for _ in ()).throw(AssertionError('grip read')),
                                multiprocessing.Array('d', [0.]*3))
        self.assertGreater(value[2], 0.)


if __name__ == '__main__':
    unittest.main()
