"""Host adapter regressions only: no Aorta Node, IPC, customer model or inputs.

Run with the declared SDK source overlay; its inherited native test binary is
recorded separately. This is not a final-wheel or device acceptance run.
"""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from deployment.vita_s100_aorta_30dof import aorta_adapter as adapter
from deployment.vita_s100_aorta_30dof import onboard_live_io as live_io
from deployment.vita_s100_aorta_30dof import sdkmain_adapter as boundary
from deployment.vita_s100_aorta_30dof import sdk021_guard as guard
from deployment.vita_s100_aorta_30dof.onboard_live_session import OnboardLiveSession
from deployment.vita_s100_aorta_30dof.timing import PolicyCycleClock
from locomotion_aorta.state_arrays import BodyStateArrays, decode_body_state_arrays
from locomotion_aorta._body_codec import decode_body_state
from lowlevel.BodyState import BodyState

spec = importlib.util.spec_from_file_location('body31_fixture', ROOT / 'tests/sdkmain_body31_offline.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class NoDTO(BodyStateArrays):
    @property
    def joints(self):
        raise AssertionError('lazy joint DTO materialization is forbidden')

    @property
    def imu_states(self):
        raise AssertionError('lazy IMU DTO materialization is forbidden')

    def body_state(self):
        raise AssertionError('body DTO materialization is forbidden')


def arrays_from_payload(payload):
    arrays = decode_body_state_arrays(payload)
    return NoDTO(**{f.name: getattr(arrays, f.name) for f in dataclasses.fields(arrays)})


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.fixtures = json.loads((Path(os.environ['T074_FIXTURES']) / 'fixtures.json').read_text())
        cls.payload = fixture.body_buffer(cls.fixtures[0])

    def arrays(self, **values):
        result = arrays_from_payload(self.payload)
        return dataclasses.replace(result, received_monotonic_ns=1_000_000_000,
            received_realtime_ns=100_000_000_000, pull_context_receive_stamp_ns=99_000_000_000,
            **values)


class NativeMappingTests(FixtureCase):
    def test_256_native_frames_equal_python_oracle_and_policy_float32(self):
        from deployment.vita_s100_aorta_30dof.live_shadow import policy_state_from_lowstate
        for f in self.fixtures:
            payload = fixture.body_buffer(f)
            dto = decode_body_state(BodyState.GetRootAs(payload))
            expected = boundary.raw_state_from_body(dto, received_monotonic_ns=123)
            actual = boundary.raw_state_from_arrays(arrays_from_payload(payload), received_monotonic_ns=123)
            for name in ('q_by_slot', 'dq_by_slot', 'tau_est_by_slot', 'temperature_by_slot',
                         'lost_by_slot', 'slot_role_by_slot'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            self.assertEqual(actual.joint_timestamps_us, expected.joint_timestamps_us)
            self.assertEqual(actual.state_id, expected.state_id)
            for a, e in zip(actual.imu_samples, expected.imu_samples, strict=True):
                for name in ('quaternion_wxyz', 'gyroscope_rad_s', 'accelerometer_m_s2', 'rpy_deg'):
                    np.testing.assert_array_equal(getattr(a, name), getattr(e, name))
            a = policy_state_from_lowstate(actual, base_from_hip_quaternion_wxyz=f['alignment'])
            e = policy_state_from_lowstate(expected, base_from_hip_quaternion_wxyz=f['alignment'])
            for name in ('robot_dof_pos', 'robot_dof_vel', 'robot_root_quat_wxyz', 'robot_root_angvel_local'):
                np.testing.assert_array_equal(getattr(a, name), getattr(e, name))

    def test_uint64_identity_applied_and_timestamp_high_bits_survive(self):
        f = dict(self.fixtures[0], state_id=(1 << 64) - 1,
                 applied_session_id=(1 << 63) + 1, applied_sequence=(1 << 62) + 3,
                 ts_first_output_us=(1 << 63) + 7)
        arrays = arrays_from_payload(fixture.body_buffer(f))
        stamps = np.full((31, 3), (1 << 63) + 9, dtype=np.uint64)
        arrays = dataclasses.replace(arrays, joint_stamps_us=stamps)
        raw = boundary.raw_state_from_arrays(arrays, received_monotonic_ns=123)
        self.assertEqual((raw.state_id, raw.applied_session_id, raw.applied_sequence, raw.ts_first_output_us),
                         (f['state_id'], f['applied_session_id'], f['applied_sequence'], f['ts_first_output_us']))
        self.assertEqual(raw.joint_timestamps_us, (((1 << 63) + 9,) * 3,) * 31)

    def test_uint32_faults_reject_without_signed_conversion(self):
        for value in (8, 17, 30, 255, 0x80000000, 0xffffffff):
            arrays = arrays_from_payload(fixture.body_buffer(dict(self.fixtures[0], error_code=value)))
            self.assertEqual(arrays.error_code.dtype, np.uint32)
            self.assertEqual(int(arrays.error_code[0]), value)
            with self.subTest(code=value), self.assertRaisesRegex(ValueError, 'Q/Dq unavailable'):
                boundary.raw_state_from_arrays(arrays, received_monotonic_ns=1)

    def test_shape_protocol_attitude_and_finite_policy_guards(self):
        arrays = self.arrays()
        for change in ({'q': arrays.q[:-1]}, {'dq': arrays.dq.astype(np.float64)},
                       {'error_code': arrays.error_code.astype(np.int32)},
                       {'joint_stamps_us': arrays.joint_stamps_us.astype(np.float64)},
                       {'protocol_version': 1}, {'attitude_valid': False}, {'state_id': 1 << 64}):
            with self.subTest(change=next(iter(change))), self.assertRaises(ValueError):
                boundary.raw_state_from_arrays(dataclasses.replace(arrays, **change), received_monotonic_ns=1)
        for name in ('q', 'dq'):
            values = getattr(arrays, name).copy(); values[0] = np.nan
            with self.assertRaises(ValueError):
                boundary.raw_state_from_arrays(dataclasses.replace(arrays, **{name: values}), received_monotonic_ns=1)

    def test_hole14_is_unmapped_and_input_arrays_are_not_mutated(self):
        arrays = self.arrays(); before = arrays.q.copy()
        raw = boundary.raw_state_from_arrays(arrays, received_monotonic_ns=1)
        self.assertIsNone(boundary.BODY_TO_LEGACY[14])
        self.assertTrue(np.isnan(arrays.q[14]))
        self.assertTrue(np.isfinite(raw.q_by_slot).all())
        np.testing.assert_array_equal(arrays.q, before)
        self.assertFalse(arrays.q.flags.writeable)

    def test_hip_presence_status_duplicate_and_absent_imu_guards(self):
        arrays = self.arrays()
        for change in ({'imu_location': np.array([1], np.uint8)},
                       {'imu_status': np.array([0], np.uint8)},
                       {'imu_state_present': np.array([0], np.uint8)},
                       {'imu_location': np.array([], np.uint8)}):
            with self.subTest(change=next(iter(change))), self.assertRaises(ValueError):
                boundary.raw_state_from_arrays(dataclasses.replace(arrays, **change), received_monotonic_ns=1)
        duplicate = dataclasses.replace(arrays, imu_location=np.array([0, 0], np.uint8),
            imu_status=np.array([1, 1], np.uint8), imu_state_present=np.array([1, 1], np.uint8),
            imu_values=np.repeat(arrays.imu_values, 2, axis=0),
            imu_stamps_us=np.repeat(arrays.imu_stamps_us, 2, axis=0))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            boundary.raw_state_from_arrays(duplicate, received_monotonic_ns=1)


class FakePull:
    def __init__(self, latest, group='default'):
        self.latest = latest; self.calls = []; self.closed = 0
        self.transport = types.SimpleNamespace(node=types.SimpleNamespace(group=lambda: group))

    def wait_state(self, timeout):
        self.calls.append(timeout)
        if self.closed:
            raise RuntimeError('fake closed')
        if self.latest is None:
            raise TimeoutError('no first state')
        return self.latest

    def close(self):
        self.closed += 1


class PullSourceTests(FixtureCase):
    def setUp(self):
        self.published_ns = self.arrays().ts_state_pub_us * 1000
        patcher = patch.object(adapter.time, 'time_ns', return_value=self.published_ns + 10_000_000)
        patcher.start()
        self.addCleanup(patcher.stop)

    def source(self, fake, *, raw=False):
        # Explicit inert factories; this does not waive a production guard.
        with patch('locomotion_aorta.pull.PullStateSource', return_value=fake), \
             patch.object(guard, 'verify_vendor_sources', return_value={}), \
             patch.object(guard, 'require_wheel_guard'):
            if raw:
                return adapter.VitaRawStateSource(group='default')
            return live_io.OnboardLiveStateSource(group='default', alignment=[1, 0, 0, 0])

    def test_empty_control_read_waits_original_budget_without_reusing_frame(self):
        fake = FakePull(self.arrays()); source = self.source(fake)
        now = [0.]
        sleeps = []
        def sleep(seconds):
            now[0] += seconds
            sleeps.append(seconds)
        with patch.object(adapter.time, 'monotonic_ns', return_value=1_010_000_000), \
             patch.object(adapter.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(adapter.time, 'sleep', side_effect=sleep):
            a = source.read(.1)
            with self.assertRaisesRegex(TimeoutError, 'No advancing'):
                source.read(.1)
        self.assertEqual(a.received_monotonic_ns, 1_000_000_000)
        self.assertEqual(a.received_wall_ns, 100_000_000_000)
        self.assertEqual(a.native_receive_stamp_ns, 99_000_000_000)
        self.assertEqual(fake.calls[0], .1)
        self.assertTrue(all(value == 0 for value in fake.calls[1:]))
        self.assertAlmostEqual(now[0], .1)
        self.assertTrue(sleeps)
        self.assertEqual(source.received, 1)
        source.close(); source.close(); self.assertEqual(fake.closed, 1)

    def test_new_snapshot_with_same_or_lower_state_id_cannot_be_reused(self):
        fake = FakePull(self.arrays()); source = self.source(fake)
        now = [0.]
        with patch.object(adapter.time, 'monotonic_ns', return_value=1_010_000_000), \
             patch.object(adapter.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(adapter.time, 'sleep', side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            first = source.read(.1)
            for identity in (first.state_id, first.state_id - 1):
                fake.latest = dataclasses.replace(fake.latest, state_id=identity)
                with self.assertRaisesRegex(TimeoutError, 'No advancing'):
                    source.read(.1)
        self.assertEqual(source.received, 1)
        source.close()

    def test_20_to_100_ms_gaps_wait_for_new_frame_without_failing(self):
        for gap_ns in (20_000_000, 50_000_000, 99_000_000, 100_000_000):
            with self.subTest(gap_ns=gap_ns):
                fake = FakePull(self.arrays()); source = self.source(fake)
                now = [0]
                def sleep(seconds):
                    now[0] += max(1, round(seconds * 1e9))
                    if now[0] >= gap_ns:
                        fake.latest = dataclasses.replace(fake.latest,
                            state_id=self.arrays().state_id + 1,
                            received_monotonic_ns=1_000_000_000 + now[0],
                            ts_state_pub_us=(self.published_ns + gap_ns) // 1000)
                with patch.object(adapter.time, 'monotonic', side_effect=lambda: now[0] / 1e9), \
                     patch.object(adapter.time, 'monotonic_ns', side_effect=lambda: 1_000_000_000 + now[0]), \
                     patch.object(adapter.time, 'time_ns', side_effect=lambda: self.published_ns + now[0]), \
                     patch.object(adapter.time, 'sleep', side_effect=sleep):
                    first = source.read(.1)
                    second = source.read(.1)
                self.assertEqual(second.state_id, first.state_id + 1)
                self.assertEqual(now[0], gap_ns)
                self.assertEqual(source.received, 2)
                self.assertEqual(fake.closed, 0)  # No error/teardown path during the gap.
                source.close()

    def test_gap_beyond_100_ms_times_out_without_returning_old_frame(self):
        fake = FakePull(self.arrays()); source = self.source(fake)
        now = [0]
        def sleep(seconds):
            now[0] += max(1, round(seconds * 1e9))
            if now[0] >= 101_000_000:
                fake.latest = dataclasses.replace(fake.latest, state_id=fake.latest.state_id + 1)
        with patch.object(adapter.time, 'monotonic', side_effect=lambda: now[0] / 1e9), \
             patch.object(adapter.time, 'monotonic_ns', side_effect=lambda: 1_000_000_000 + now[0]), \
             patch.object(adapter.time, 'time_ns', side_effect=lambda: self.published_ns + now[0]), \
             patch.object(adapter.time, 'sleep', side_effect=sleep):
            source.read(.1)
            with self.assertRaisesRegex(TimeoutError, 'No advancing'):
                source.read(.1)
        self.assertEqual(now[0], 100_000_000)
        self.assertEqual(source.received, 1)
        source.close()

    def test_fresh_dequeue_cannot_refresh_an_old_publication(self):
        for publication in (self.published_ns - 100_001_000,
                            self.published_ns + 1000, 0):
            source = self.source(FakePull(self.arrays(ts_state_pub_us=publication // 1000)))
            with patch.object(adapter.time, 'monotonic_ns', return_value=1_000_000_000), \
                 patch.object(adapter.time, 'time_ns', return_value=self.published_ns):
                with self.assertRaisesRegex(RuntimeError, 'stale'):
                    source.read(.1)
            source.close()

    def test_publication_age_boundary_is_checked_in_realtime(self):
        for age, stale in ((100_000_000, False), (100_000_001, True)):
            source = self.source(FakePull(self.arrays()))
            with patch.object(adapter.time, 'monotonic_ns', return_value=1_000_000_000), \
                 patch.object(adapter.time, 'time_ns', return_value=self.published_ns + age):
                if stale:
                    with self.assertRaisesRegex(RuntimeError, 'stale'): source.read(.1)
                else:
                    self.assertEqual(source.read(.1).state_id, self.arrays().state_id)
            source.close()

    def test_diagnostic_wait_sleeps_until_an_advancing_frame_and_times_out(self):
        from deployment.vita_s100_aorta_30dof import diagnostic_monitor as monitor
        fake = FakePull(self.arrays()); source = self.source(fake, raw=True)
        records = []
        class Recorder:
            directory = Path('/inert/diagnostic')
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def write(self, value): records.append(value)
        now = [0.]
        sleeps = []
        def sleep(seconds):
            now[0] += seconds
            sleeps.append(seconds)
            if len(sleeps) == 1:
                fake.latest = dataclasses.replace(fake.latest, state_id=fake.latest.state_id + 1)
        with patch.object(adapter.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(adapter.time, 'monotonic_ns', return_value=1_010_000_000), \
             patch.object(adapter.time, 'sleep', side_effect=sleep), \
             patch.object(monitor, 'VitaRawStateSource', return_value=source), \
             patch.object(monitor, 'JsonlSessionRecorder', return_value=Recorder()), \
             patch.object(monitor.signal, 'signal'):
            with self.assertRaisesRegex(TimeoutError, 'No advancing'):
                monitor.run_diagnostic_monitor(types.SimpleNamespace(aorta_group='default', logs=Path('/inert')))
        self.assertEqual([r['state_id'] for r in records],
                         [self.arrays().state_id, self.arrays().state_id + 1])
        self.assertTrue(sleeps)
        self.assertTrue(all(0 < s <= .001 for s in sleeps))
        self.assertGreaterEqual(now[0], .5)
        self.assertEqual(fake.closed, 1)

    def test_first_frame_timeout_and_closed_read(self):
        fake = FakePull(None); source = self.source(fake)
        with self.assertRaises(TimeoutError): source.read(.1)
        source.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'): source.read(.1)

    def test_future_receipt_rejected(self):
        source = self.source(FakePull(self.arrays()))
        with patch.object(adapter.time, 'monotonic_ns', return_value=999_999_999):
            with self.assertRaisesRegex(RuntimeError, 'stale'): source.read(.1)
        source.close()

    def test_binding_shares_controller_without_owning_or_closing_it(self):
        readonly = FakePull(self.arrays()); source = self.source(readonly)
        controller = FakePull(self.arrays(state_id=999))
        source.bind_session(types.SimpleNamespace(controller=controller))
        self.assertEqual(readonly.closed, 1)
        with patch.object(adapter.time, 'monotonic_ns', return_value=1_010_000_000):
            self.assertEqual(source.read(.1).state_id, 999)
        source.close(); self.assertEqual(controller.closed, 0)

    def test_mismatched_group_closes_only_own_new_readonly_source(self):
        readonly = FakePull(self.arrays(), group='other')
        with self.assertRaisesRegex(RuntimeError, 'group'): self.source(readonly)
        self.assertEqual(readonly.closed, 1)
        readonly = FakePull(self.arrays()); source = self.source(readonly)
        with self.assertRaisesRegex(RuntimeError, 'groups differ'):
            source.bind_session(types.SimpleNamespace(controller=FakePull(self.arrays(), group='other')))
        self.assertEqual(readonly.closed, 0); source.close()

    def test_400hz_source_is_selected_on_original_50hz_grid_no_catchup(self):
        now = [1_000_000_000]; fake = FakePull(self.arrays()); source = self.source(fake)
        def sleep(seconds): now[0] += round(seconds * 1e9)
        clock = PolicyCycleClock(.02, monotonic_ns=lambda: now[0], sleep=sleep)
        starts = []
        with patch.object(adapter.time, 'monotonic_ns', side_effect=lambda: now[0]):
            for i in range(50):
                tick = clock.wait(); starts.append(tick.started_ns)
                fake.latest = dataclasses.replace(fake.latest, state_id=1 + (now[0] - 1_000_000_000) // 2_500_000,
                    received_monotonic_ns=now[0], ts_state_pub_us=(self.published_ns + now[0] - 1_000_000_000) // 1000)
                with patch.object(adapter.time, 'time_ns', return_value=self.published_ns + now[0] - 1_000_000_000):
                    source.read(.1)
        self.assertEqual(np.diff(starts).tolist(), [20_000_000] * 49)
        self.assertEqual(source.received, 50)
        self.assertEqual(fake.calls[1:], [0] * 49)
        now[0] += 75_000_000
        tick = clock.wait(); self.assertEqual(tick.skipped_ticks, 2)
        following = clock.wait()
        self.assertGreater(following.started_ns, tick.started_ns)
        source.close()


class GuardAndScopeTests(unittest.TestCase):
    def test_original_15_pins_retained_and_final_wheel_required_before_node(self):
        self.assertEqual(set(guard.VENDOR_SOURCE_SHA256), set(guard.ORIGINAL_VENDOR_FILES))
        self.assertEqual(len(guard.VENDOR_SOURCE_SHA256), 15)
        with patch.object(guard, 'GUARD', {**guard.GUARD, 'status': 'AWAITING_REQ182_FINAL_WHEEL'}), \
             patch('aorta.Node', side_effect=AssertionError('must not open a Node')):
            with self.assertRaisesRegex(RuntimeError, 'wheel guard is pending'):
                OnboardLiveSession('default')
            with self.assertRaisesRegex(RuntimeError, 'wheel guard is pending'):
                live_io.OnboardLiveStateSource(group='default', alignment=[1, 0, 0, 0])

    def test_all_15_and_new_hot_path_files_are_checked_not_just_length(self):
        import hashlib
        payloads = {n: n.encode() for n in guard.ORIGINAL_VENDOR_FILES + guard.RUNTIME_FILES}
        native = {'_body_cmd_native.fixture.so': b'encoder',
                  '_onnx/.libs/libonnxruntime.so.1': b'bundled runtime'}
        payloads.update(native)
        original = {n: hashlib.sha256(payloads[n]).hexdigest() for n in guard.ORIGINAL_VENDOR_FILES}
        runtime = {n: hashlib.sha256(payloads[n]).hexdigest() for n in guard.RUNTIME_FILES}
        native_pins = {n: hashlib.sha256(payloads[n]).hexdigest() for n in native}
        loader = types.SimpleNamespace(get_data=lambda path: payloads[str(Path(path).relative_to('/fake/locomotion_aorta'))])
        spec = types.SimpleNamespace(origin='/fake/locomotion_aorta/__init__.py', loader=loader)
        extension = types.SimpleNamespace(origin='/fake/locomotion_aorta/_body_cmd_native.fixture.so')
        def find_spec(name):
            return spec if name == 'locomotion_aorta' else extension
        with patch.object(guard, 'GUARD', {**guard.GUARD, 'status': 'SEALED_FROM_REVIEWED_WHEEL',
                    'runtime_source_sha256': runtime, 'native_sha256': native_pins}), \
             patch.object(guard, 'VENDOR_SOURCE_SHA256', original), \
             patch.object(guard.importlib.util, 'find_spec', side_effect=find_spec):
            self.assertEqual(guard.verify_vendor_sources(), original)
            for name in payloads:
                before = payloads[name]; payloads[name] = b'tampered'
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'Unreviewed'):
                    guard.verify_vendor_sources()
                payloads[name] = before
            for origin in ('/elsewhere/native.so', '/fake/locomotion_aorta/unpinned.so'):
                extension.origin = origin
                with self.subTest(origin=origin), self.assertRaisesRegex(RuntimeError, 'outside|Unreviewed'):
                    guard.verify_vendor_sources()

    def test_missing_config_explains_src_only_deployment_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'config/sdk021-wheel-guard.json.*src-only'):
            guard.load_guard(Path('/inert/missing-sdkmain-guard.json'))

    def test_missing_native_guard_refuses_handles_before_node(self):
        with patch.object(guard, 'GUARD', {**guard.GUARD, 'native_sha256': {}}), \
             patch('aorta.Node', side_effect=AssertionError('must not open a Node')):
            with self.assertRaisesRegex(RuntimeError, 'native library guard is pending'):
                OnboardLiveSession('default')

    def test_session_uses_sdk_trace_without_monkeypatching_typed_publisher(self):
        session = OnboardLiveSession.__new__(OnboardLiveSession)
        session.controller = types.SimpleNamespace(_last_publish_timing=lambda: {'encode_call_wall_ns': 10})
        self.assertEqual(session.last_publish_profile, {'encode_call_wall_ns': 10})

    def test_inert_session_passes_transport_to_sdk_pull_controller_and_closes_on_failure(self):
        transport = types.SimpleNamespace(close=lambda: None)
        controller = types.SimpleNamespace()
        with patch.object(guard, 'verify_vendor_sources', return_value={'guard': 'fixture'}), \
             patch.object(adapter, 'RealtimeBodyTransport', return_value=transport), \
             patch('locomotion_aorta.pull.PullController', return_value=controller) as factory:
            session = OnboardLiveSession('default')
        self.assertIs(session.controller, controller)
        self.assertIs(factory.call_args.kwargs['_transport'], transport)
        self.assertTrue(controller._capture_publish_timing)
        self.assertEqual(session.vendor_source_hashes, {'guard': 'fixture'})
        closed = []
        transport.close = lambda: closed.append(True)
        with patch.object(guard, 'verify_vendor_sources', return_value={}), \
             patch.object(adapter, 'RealtimeBodyTransport', return_value=transport), \
             patch('locomotion_aorta.pull.PullController', side_effect=RuntimeError('inert failure')):
            with self.assertRaisesRegex(RuntimeError, 'inert failure'):
                OnboardLiveSession('default')
        self.assertEqual(closed, [True])

    def test_final_sdk_closed_and_timeout_contract_with_inert_transport(self):
        from locomotion_aorta.pull import PullController
        from locomotion_aorta import ControllerClosedError, ExternalControlTimeoutError
        from unittest.mock import Mock
        transport = Mock()
        transport.subscribe.return_value = types.SimpleNamespace(close=lambda: None)
        controller = PullController(group='default', _transport=transport)
        transport.wait_state.side_effect = TimeoutError('first frame timeout')
        with self.assertRaises(ExternalControlTimeoutError):
            controller.wait_body_state(.01)
        snapshot = object()
        controller._latest_status = snapshot
        controller.close()
        self.assertIs(controller.status, snapshot)
        with self.assertRaises(ControllerClosedError): controller.wait_state(.01)
        with self.assertRaises(ControllerClosedError): controller.publish(object())


class GuardGeneratorTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location('guard_generator', ROOT / 'scripts/generate_sdk021_guard.py')
        self.generator = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.generator)
        self.compat = {'aorta_version': '2026.10.20', 'humanoid_msgs_version': '0.1.5',
                       'flatbuffers_version': '25.12.19', 'schema_sha256': {'body_state': 'fixture-sha'}}
        self.sources = {name: ('fixture:' + name).encode()
                        for name in self.generator.ORIGINAL + self.generator.RUNTIME}

    def wheel(self, *, changed=None, dependencies=True):
        import hashlib, zipfile
        path = self.directory / 'fixture.whl'
        metadata = 'Name: locomotion-aorta\nVersion: 0.2.1\n'
        if dependencies:
            metadata += ''.join('Requires-Dist: ' + name + '\n' for name in (
                'aorta-sdk==2026.10.20', 'aorta-msgs==2026.10.20',
                'vbot-humanoid-msgs==0.1.5', 'flatbuffers==25.12.19'))
        with zipfile.ZipFile(path, 'w') as wheel:
            wheel.writestr('locomotion_aorta-0.2.1.dist-info/METADATA', metadata)
            for name, data in self.sources.items():
                wheel.writestr('locomotion_aorta/' + name, b'changed' if name == changed else data)
            wheel.writestr('locomotion_aorta/_body_cmd_native.fixture.so', b'native encoder')
            wheel.writestr('locomotion_aorta/_onnx/.libs/libonnxruntime.so.1', b'bundled runtime')
        return types.SimpleNamespace(sdk_commit='a' * 40, sdk_git=Path('/unused'),
            sdk_wheel=path, wheel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=self.directory / 'guard.json')

    def git_blob(self, argv):
        path = argv[-1].split(':', 1)[1]
        if path == 'compatibility.json': return json.dumps(self.compat).encode()
        return self.sources[Path(path).name]

    def test_reviewed_wheel_seals_exact_15_plus_hotpath_and_records_sha(self):
        args = self.wheel()
        with patch.object(self.generator.subprocess, 'check_output', side_effect=self.git_blob):
            result = self.generator.generate(args)
        self.assertEqual(result['status'], 'SEALED_FROM_REVIEWED_WHEEL')
        self.assertEqual(set(result['source_sha256']), set(guard.ORIGINAL_VENDOR_FILES))
        self.assertEqual(len(result['source_sha256']), 15)
        self.assertEqual(set(result['runtime_source_sha256']), set(guard.RUNTIME_FILES))
        self.assertEqual(result['wheel']['sha256'], args.wheel_sha256)
        self.assertEqual(set(result['native_sha256']),
            {'_body_cmd_native.fixture.so', '_onnx/.libs/libonnxruntime.so.1'})

    def test_wrong_reviewed_wheel_sha_does_not_overwrite_existing_guard(self):
        args = self.wheel(); args.wheel_sha256 = '0' * 64; args.output.write_text('keep')
        with patch.object(self.generator.subprocess, 'check_output', side_effect=self.git_blob):
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): self.generator.generate(args)
        self.assertEqual(args.output.read_text(), 'keep')

    def test_same_version_old_wheel_source_mismatch_is_rejected(self):
        args = self.wheel(changed='pull.py')
        with patch.object(self.generator.subprocess, 'check_output', side_effect=self.git_blob):
            with self.assertRaisesRegex(ValueError, 'reviewed Git source'): self.generator.generate(args)
        self.assertFalse(args.output.exists())

    def test_dependency_pin_mismatch_is_rejected(self):
        args = self.wheel(dependencies=False)
        with patch.object(self.generator.subprocess, 'check_output', side_effect=self.git_blob):
            with self.assertRaisesRegex(ValueError, 'dependency pins'): self.generator.generate(args)
        self.assertFalse(args.output.exists())

    def test_short_commit_and_source_only_generation_are_rejected(self):
        args = self.wheel(); args.sdk_commit = '49d4152b'
        with self.assertRaisesRegex(ValueError, 'complete fixed'): self.generator.generate(args)
        args.sdk_commit = 'a' * 40; args.sdk_wheel = None
        with self.assertRaisesRegex(ValueError, 'final reviewed wheel'): self.generator.generate(args)


if __name__ == '__main__':
    import aorta
    with patch.object(aorta, 'Node', side_effect=AssertionError('host tests forbid Aorta Node')):
        unittest.main(verbosity=2)
