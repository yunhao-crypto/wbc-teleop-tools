"""Pico stream conditioning, control-clock playback and the frozen robot42 session."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import actor_pico_stream as stream
from deployment.vita_s100_aorta_30dof.actor_pico_live import pico_reference_command
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import (SessionFixture, SessionReplay,
                                                                        advance_statistics)
from deployment.vita_s100_aorta_30dof.actor_pico_worker import ReferenceStream

SESSION = ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz'


def tilt_of(q):
    w, x, y, z = q
    return math.acos(min(1., 1-2*(x*x+y*y)))


class StreamSettingsTests(unittest.TestCase):
    def test_absent_block_keeps_original_stream_and_invalid_values_fail(self):
        self.assertIsNone(stream.stream_settings({}))
        good = dict(ik_max_iter=3, playback_margin_s=.06)
        self.assertEqual(stream.stream_settings(dict(pico_stream=good)).ik_max_iter, 3)
        for change in (dict(ik_max_iter=11), dict(ik_max_iter=True), dict(playback_margin_s=-.01),
                       dict(playback_margin_s=.5), dict(root_max_tilt_deg=0), dict(worker_cpus=[1, 1]),
                       dict(joint_filter=dict(min_cutoff_hz=0, beta=0, d_cutoff_hz=1)), dict(unknown=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                stream.stream_settings(dict(pico_stream=dict(good, **change)))

    def test_one_euro_holds_constants_and_follows_steps_with_bounded_lag(self):
        f = stream.OneEuro(stream.OneEuroSettings(3., .5, 1.))
        for _ in range(10):
            np.testing.assert_allclose(f(np.ones(3), .02), 1.)
        out = [f(np.full(3, 2.), .02)[0] for _ in range(25)]
        self.assertTrue(all(1 < a <= b <= 2 for a, b in zip(out, out[1:])))
        self.assertGreater(out[-1], 1.99)

    def test_quaternion_filter_converges_along_the_geodesic(self):
        f = stream.QuaternionOneEuro(stream.OneEuroSettings(1.5, .3, 1.))
        f([1., 0, 0, 0], .02)
        target = np.array([math.cos(.3), 0., 0., math.sin(.3)])
        for _ in range(100):
            q = f(target, .02)
        self.assertAlmostEqual(abs(float(q @ target)), 1., places=6)
        self.assertLess(tilt_of(q), 1e-9)

    def test_tilt_limit_clamps_lean_and_keeps_heading(self):
        yaw, lean = .8, math.radians(40)
        q = stream._multiply([math.cos(yaw/2), 0, 0, math.sin(yaw/2)], [math.cos(lean/2), 0, math.sin(lean/2), 0])
        limited = stream.limit_tilt(q, math.radians(25))
        self.assertAlmostEqual(tilt_of(limited), math.radians(25), places=9)
        heading = lambda q: math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
        self.assertAlmostEqual(heading(limited), heading(q), places=9)
        np.testing.assert_allclose(stream.limit_tilt(limited, math.radians(30)), limited)

    def test_source_clock_tracks_lower_envelope_and_reanchors_on_jumps(self):
        c = stream.SourceClock()
        self.assertEqual(c.update(0., 10.03), 10.03)
        self.assertAlmostEqual(c.update(.04, 10.05), 10.03-c.FALL*.04)
        before = c.offset
        self.assertAlmostEqual(c.update(.08, 10.2), before+c.RISE*.04)
        self.assertEqual(c.update(700., 20.), 20.-700.)


class PlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort', policy='supermimic')

    def test_same_worker_snapshot_advances_the_reference_with_the_control_clock(self):
        r = self.runtime
        now = [0.]
        worker = ReferenceStream(r.root, r.config, r.model.contract, r.math, clock=lambda: now[0])
        body = np.zeros((24, 7)); body[:, 6] = 1
        for i in range(60):
            now[0] = 1.+i*.033
            snapshot = worker.push(body, dict(timestamp_ns=5_000_000_000+i*33_000_000,
                                              timestamp_monotonic=now[0]), (1., 0., 0., 0.))
        self.assertEqual(snapshot['state'], 'ready')
        self.assertNotIn('q', snapshot)
        times = []
        for k in range(3):
            r.reset_inputs()
            _, _, row = pico_reference_command(r, snapshot, now[0]+.01+k*.02, q=r.model.contract.default,
                dq=np.zeros(28), root=[1, 0, 0, 0], gyro=np.zeros(3), yaw_offset=0.)
            times.append(row['playback_time_s'])
            self.assertIn('retarget_cpu_ms', row)
        np.testing.assert_allclose(np.diff(times), .02, atol=1e-12)

    def test_settings_cap_gmr_iterations(self):
        self.assertEqual(self.runtime.retarget.engine.max_iter, self.runtime.stream.ik_max_iter)


class FrozenSessionTests(unittest.TestCase):
    """robot42 2026-10-29 SuperMimic session: every Pico frame the robot's GMR processed."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='ort', policy='supermimic')
        cls.fixture = SessionFixture(SESSION)
        cls.legacy = copy.deepcopy(cls.runtime.config)
        cls.legacy.pop('pico_stream')

    def test_fixture_matches_the_deployed_model_and_replays_robot_gmr_exactly(self):
        f = self.fixture
        self.assertEqual(f.meta['model']['sha256'], self.runtime.config['artifacts']['model']['sha256'])
        self.assertEqual(f.human_height_m, 1.7)
        self.assertEqual(len(f.following_segments()), 4)
        replay = SessionReplay(self.runtime, f, config=self.legacy)
        for i, expected in enumerate(f.frame_qpos_head):
            replay.stream.push(f.frame_body[i].astype(np.float64), dict(timestamp_ns=int(f.frame_stamp_ns[i]),
                timestamp_monotonic=float(f.frame_received_monotonic[i])), (1., 0., 0., 0.))
            np.testing.assert_allclose(replay.stream.retarget.last_qpos, expected, rtol=0, atol=1e-9)

    def advance(self, config, seconds=8.):
        replay = SessionReplay(self.runtime, self.fixture, config=config)
        start, _ = self.fixture.following_segments()[2]
        times = [replay.current_source_time(s, now) for now, s in replay.ticks(start, start+seconds)
                 if s.get('state') == 'ready']
        return advance_statistics(times)

    def test_recorded_timing_stutters_with_the_newest_sample_anchor(self):
        stats = self.advance(self.legacy)
        self.assertGreater(stats['repeated'], .3)
        self.assertGreater(stats['step_p95_ms'], 40)

    def test_control_clock_playback_advances_every_tick(self):
        stats = self.advance(self.runtime.config)
        self.assertEqual((stats['repeated'], stats['backwards']), (0., 0))
        self.assertGreater(stats['nominal'], .9)
        self.assertLess(stats['step_max_ms'], 30)

    def test_gmr_benchmark_reports_wall_and_cpu_time(self):
        import json, subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'bench.json'
            subprocess.run([sys.executable, str(ROOT/'scripts/bench_pico_gmr.py'), '--frames', '3',
                            '--output', str(out)], check=True, capture_output=True)
            report = json.loads(out.read_text())
        self.assertEqual((report['frames'], report['ik_max_iter']), (3, self.runtime.stream.ik_max_iter))
        self.assertGreater(report['wall_ms']['mean'], 0)
        self.assertEqual(set(report['cpu_ms']), {'mean', 'p50', 'p95', 'p99', 'max'})


if __name__ == '__main__':
    unittest.main()
