"""Pure fake coverage: no SDK, Pico service, inference or robot connection."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deployment.vita_s100_aorta_30dof.sonic_action14_source import SourceFrames, reference_contract
from deployment.vita_s100_aorta_30dof.sonic_action14_orchestrator import Action14Session

PROFILE = 'actions/policy_profiles/vt_human_v2_tracking_asample_wtt_native.yaml'


class FakeSolver:
    last_qpos = [0., 0., 0.]

    def solve(self, body, *, stamp_ns):
        return [float(stamp_ns)] * 28, [1., 0., 0., 0.], 0.


class FakeControl:
    def __init__(self):
        self.owner = None
        self.events = []
        self.frames = 0
        self.fail_return = False
        self.capabilities = dict(native_raw_affine_targets=True,
                                 native_stream_auto_resume=False,
                                 native_stream_auto_start=False,
                                 native_target_margin_rad=1.5,
                                 native_runtime_config=PROFILE)

    def status(self):
        return dict(owned_req_id=self.owner, current_action='HUMANOID_TRACKING',
                    active=True, superseded=False, status_age_ms=10)

    def activate(self, request_id, profile):
        assert profile == 'profile_wtt_native'
        self.owner = request_id
        return {'ok': True, 'req_id': request_id}

    def event(self, request_id, name, params):
        self.events.append((request_id, name))
        if name == 'humanoid_tracking.query':
            data = dict(self.capabilities)
            data['stream_frames_received'] = self.frames
            data['stream_frame_age_ms'] = 20
            return {'ok': True, 'data': data}
        if name == 'humanoid_tracking.stream_start':
            if self.frames < 1:
                raise RuntimeError('no_stream')
            return {'ok': True, 'data': {'load_success': True}}
        if name == 'humanoid_tracking.return_to_stand':
            if self.fail_return:
                raise TimeoutError('lost business result')
            return {'ok': True, 'data': {'lead_in_started': True,
                                         'stay_state': 'lead_in', 'following': False}}
        raise AssertionError(name)

    def close(self):
        pass


class FakePublisher:
    def robot_info(self, timeout=2.0):
        return SimpleNamespace(runtime_id=55)

    def publish(self, cmd, *, authorized):
        assert authorized()

    def close(self):
        pass


def ready(seq, source_ns, receipt, q0=0.):
    return dict(state='ready', sequence=seq, source_ns=source_ns,
                received_monotonic=receipt, q=[q0]*28, root_wxyz=(1., 0., 0., 0.))


class SourceSessionTests(unittest.TestCase):
    def test_reference_contract(self):
        self.assertEqual(len(reference_contract().names), 28)
        self.assertNotIn('neck_pitch_joint', reference_contract().names)

    def test_head_targets_commit_only_after_publish(self):
        class Head:
            def __init__(self):
                self.commits = []

            def preview_targets(self, snapshot, now):
                return {'neck_yaw_joint': 0.2, 'neck_pitch_joint': -0.1}, snapshot['pico_timestamp_ns']

            def commit(self, candidate):
                self.commits.append(candidate)

        class RecordingPublisher(FakePublisher):
            def publish(self, cmd, *, authorized):
                super().publish(cmd, authorized=authorized)
                self.last = cmd

        head = Head()
        control = FakeControl()
        control.capabilities['native_neck_follow_reference'] = True
        publisher = RecordingPublisher()
        now = [10.]
        session = Action14Session(control, publisher, clock=lambda: now[0], head=head)
        session.initialize()
        first = ready(1, 1_000_000_000, 9.90)
        second = ready(2, 1_040_000_000, 9.94, 1.)
        for snap, stamp in ((first, 10), (second, 11)):
            snap['head_snapshot'] = dict(raw_body_poses=[], received_monotonic=snap['received_monotonic'],
                                         pico_timestamp_ns=stamp)
        session.accept(first)
        self.assertEqual(head.commits, [])
        session.accept(second)
        now[0] = 10.10
        session.tick()
        self.assertGreaterEqual(session.frames_published, 1)
        self.assertEqual(head.commits, [11])
        slots = session.encoder.slots
        self.assertAlmostEqual(publisher.last.joints[slots.index('neck_yaw_joint')].q, 0.2)
        self.assertAlmostEqual(publisher.last.joints[slots.index('neck_pitch_joint')].q, -0.1)
        session.accept(dict(state='ready', sequence=3, source_ns=1_080_000_000,
                            received_monotonic=9.98, q=[0.]*28, root_wxyz=(1., 0., 0., 0.)))
        self.assertEqual(head.commits, [11])
        self.assertEqual(session.last_posture, 'lead_in_accepted')

    def test_activation_waits_for_tracking_status(self):
        control = FakeControl()
        reads = {'n': 0}
        original = control.status

        def status():
            reads['n'] += 1
            row = original()
            if reads['n'] < 3:
                row['current_action'] = 'FIXED_STAND'
            return row

        control.status = status
        session = Action14Session(control, FakePublisher(), clock=time.monotonic)
        session.initialize()
        self.assertTrue(session.active)
        self.assertGreaterEqual(reads['n'], 3)

    def test_source_sequence_and_receipt(self):
        source = SourceFrames(FakeSolver(), {})
        with patch('deployment.vita_s100_aorta_30dof.sonic_action14_source.time.monotonic', return_value=5.):
            frame = source.push(None, dict(timestamp_ns=123, timestamp_monotonic=4.9))
            self.assertEqual(frame['sequence'], 1)
            self.assertEqual(frame['source_ns'], 123)
            with self.assertRaises(ValueError):
                source.push(None, dict(timestamp_ns=123, timestamp_monotonic=4.9))

    def test_trigger_does_not_activate_or_engage_without_accepted_reference(self):
        control = FakeControl()
        now = [10.]
        session = Action14Session(control, FakePublisher(), clock=lambda: now[0])
        self.assertEqual(session.update_trigger(1, 9.9), 'refused')
        self.assertFalse(session.active)
        self.assertEqual(control.events, [])
        session.initialize()
        self.assertTrue(session.active)
        self.assertFalse(session.following)
        self.assertEqual(session.update_trigger(2, 9.9), 'refused')
        self.assertFalse(session.following)
        self.assertNotIn('humanoid_tracking.stream_start', [name for _, name in control.events])
        session.accept(ready(1, 1_000_000_000, 9.90))
        session.accept(ready(2, 1_040_000_000, 9.94, 1.))
        now[0] = 10.10
        session.tick()
        self.assertGreaterEqual(session.frames_published, 1)
        control.frames = 0
        self.assertEqual(session.update_trigger(3, 10.05), 'refused')
        self.assertFalse(session.following)
        self.assertEqual(session.last_engage_refusal, 'no_stream')
        control.frames = session.frames_published
        self.assertEqual(session.update_trigger(4, 10.08), 'following')
        self.assertTrue(session.following)

    def test_foreign_owner_is_not_cancelled_and_loss_is_ambiguous(self):
        control = FakeControl()
        now = [10.]
        session = Action14Session(control, FakePublisher(), clock=lambda: now[0])
        session.initialize()
        owner = session.request_id
        control.owner = 'foreign'
        now[0] += session.admission_interval_s+.01
        with self.assertRaises(RuntimeError):
            session.tick()
        self.assertNotIn('humanoid_tracking.return_to_stand', [name for _, name in control.events])
        control.owner = owner
        session.active = True
        control.fail_return = True
        self.assertEqual(session.disengage(), 'ambiguous')
        self.assertEqual(session.last_posture, 'ambiguous')
        self.assertFalse(session.following)

    def test_confirmed_lead_in_is_not_a_verified_stand(self):
        control = FakeControl()
        session = Action14Session(control, FakePublisher(), clock=lambda: 10.)
        session.initialize()
        self.assertEqual(session.disengage(), 'lead_in_accepted')
        self.assertFalse(any(name == 'cancel' for _, name in control.events))

    def test_passive_close_does_not_request_a_stand_lead_in(self):
        control = FakeControl()
        session = Action14Session(control, FakePublisher(), clock=lambda: 10.)
        session.initialize()
        self.assertEqual(session.close(return_to_stand=False), 'passive_requested')
        self.assertNotIn('humanoid_tracking.return_to_stand', [name for _, name in control.events])

    def test_overdue_grid_publishes_the_newest_pose_with_the_next_sequence(self):
        control = FakeControl()
        published = []
        publisher = FakePublisher()
        publisher.publish = lambda cmd, authorized: published.append((cmd.sequence, cmd.joints[0].q))
        now = [10.]
        session = Action14Session(control, publisher, clock=lambda: now[0])
        session.initialize()
        session.accept(ready(1, 1_000_000_000, 9.80))
        session.accept(ready(2, 1_040_000_000, 9.84, .2))
        now[0] = session.pending[0]['due']
        session.tick()
        self.assertEqual([item[0] for item in published], [1])
        # Six skipped grid slots used to be published as sequence 7 and the
        # native receiver treated that as a discontinuity. Keep sequence + 1
        # and send the newest due pose.
        session.pending = [dict(grid_sequence=7, source_ns=1_160_000_000,
                                 received_monotonic=9.96, due=9.99,
                                 q=[.2]*28, root_wxyz=(1., 0., 0., 0.)),
                           dict(grid_sequence=8, source_ns=1_180_000_000,
                                 received_monotonic=9.98, due=9.99,
                                 q=[.4]*28, root_wxyz=(1., 0., 0., 0.))]
        session.next_send = 9.99
        now[0] = 9.99
        session.tick()
        self.assertEqual([item[0] for item in published], [1, 2])
        self.assertAlmostEqual(published[-1][1], .4)
        self.assertEqual(session.missed_grid, 1)
        self.assertTrue(session.active)
        self.assertNotIn('humanoid_tracking.return_to_stand', [name for _, name in control.events])

    def test_publish_loop_reuses_a_fresh_admission(self):
        control = FakeControl()
        calls = {'status': 0, 'info': 0}
        original = control.status

        def status():
            calls['status'] += 1
            return original()

        control.status = status
        publisher = FakePublisher()
        original_info = publisher.robot_info

        def robot_info(timeout=2.0):
            calls['info'] += 1
            return original_info(timeout=timeout)

        publisher.robot_info = robot_info
        now = [10.]
        session = Action14Session(control, publisher, clock=lambda: now[0])
        session.initialize()
        status_after_init = calls['status']
        info_after_init = calls['info']
        session.accept(ready(1, 1_000_000_000, 9.80))
        session.accept(ready(2, 1_040_000_000, 9.84, .2))
        now[0] = session.pending[0]['due']
        session.tick()
        now[0] += .02
        session.tick()
        self.assertEqual(calls['status'], status_after_init)
        self.assertEqual(calls['info'], info_after_init)
        now[0] = session.last_runtime_check+1.
        session.last_receipt = now[0]
        session.tick()
        self.assertEqual(calls['info'], info_after_init+1)

    def test_scheduled_gap_stays_within_cpp_sequence_limit(self):
        control = FakeControl()
        published = []
        publisher = FakePublisher()
        publisher.publish = lambda cmd, authorized: published.append(cmd.sequence)
        now = [19.12]
        session = Action14Session(control, publisher, clock=lambda: now[0])
        session.initialize()
        session.accept(ready(1, 2_000_000_000, 19.0))
        session.accept(ready(2, 2_120_000_000, 19.12, 1.))
        due = session.pending[0]['due']
        while session.pending and session.active:
            now[0] = session.pending[0]['due']
            session.tick()
            if len(published) > 10:
                break
        self.assertGreaterEqual(len(published), 2)
        self.assertEqual(published, list(range(1, len(published)+1)))
        self.assertLessEqual(published[-1] - published[0], 5)
        self.assertGreater(due, 19.0)

    def test_fail_closed_capability(self):
        control = FakeControl()
        control.capabilities['native_raw_affine_targets'] = False
        session = Action14Session(control, FakePublisher())
        with self.assertRaises(RuntimeError):
            session.initialize()
        self.assertFalse(session.active)
        self.assertNotIn('humanoid_tracking.return_to_stand', [name for _, name in control.events])


if __name__ == '__main__':
    unittest.main()
