"""GMR calibration profile, planted-foot targets and SONIC control-clock playback."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof import actor_pico_gmr_profile as profile_module
from deployment.vita_s100_aorta_30dof.actor_pico_gmr import create_retargeter
from deployment.vita_s100_aorta_30dof.actor_pico_gmr_profile import (
    FootSupportSettings, StanceDetector, foot_support_settings, planted_targets)
from deployment.vita_s100_aorta_30dof.actor_pico_live import pico_reference_command
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime, load_actor_config
from deployment.vita_s100_aorta_30dof.actor_pico_reference import CausalActorReference
from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import (SessionFixture, SessionReplay,
                                                                        advance_statistics)
from deployment.vita_s100_aorta_30dof.actor_pico_stream import stream_settings
from deployment.vita_s100_aorta_30dof.actor_pico_worker import ReferenceStream

PROFILE_PATH = 'config/gmr_calibration/vita02a_neutral_20261110_h180.json'
SESSION = ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz'
FOOT = dict(enter_height_m=.03, exit_height_m=.05, enter_speed_m_s=.15, exit_speed_m_s=.35,
            enter_hold_s=.05, blend_s=.15, speed_cutoff_hz=4.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_spec(path=PROFILE_PATH):
    return dict(path=str(path), sha256=sha256(ROOT/path))


def plain_head_offset():
    gmr = json.loads((ROOT/'third_party/motion_stream_bridge/motion_stream_bridge/gmr_lite/configs/'
                      'xrobot_to_vita02a.json').read_text())
    return Rotation.from_quat(np.roll(gmr['ik_match_table1']['waist_pitch_link'][4], -1)).as_matrix()


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='auto', human_height=1.8)
        cls.profile = json.loads((ROOT/PROFILE_PATH).read_text())

    def retarget(self, **retargeting):
        config = copy.deepcopy(self.runtime.config)
        config['retargeting'].update(retargeting)
        return create_retargeter(ROOT, config, self.runtime.model.contract, self.runtime.math)

    def scratch_profile(self, tmp, **changes):
        data = copy.deepcopy(self.profile)
        data.update(changes)
        path = Path(tmp)/'profile.json'
        path.write_text(json.dumps(data))
        return profile_spec(path.relative_to(ROOT))

    def test_profile_offsets_reach_both_ik_tables(self):
        engine = self.retarget(calibration_profile=profile_spec()).engine
        tasks = self.profile['tasks']
        for frame, human in (('base_link', 'Pelvis'), ('left_toe_link', 'Left_Foot'), ('waist_pitch_link', 'Spine3')):
            for offsets, rotations, table in ((engine.pos_offsets1, engine.rot_offsets1, engine.ik_match_table1),
                                              (engine.pos_offsets2, engine.rot_offsets2, engine.ik_match_table2)):
                i = [k for k, v in table.items() if v[0] == human].index(frame)
                np.testing.assert_allclose(offsets[human][i], tasks[frame]['position_offset'], rtol=0, atol=1e-15)
                wxyz = tasks[frame]['rotation_offset_wxyz']
                self.assertLess((rotations[human][i].inv()*Rotation.from_quat(np.roll(wxyz, -1))).magnitude(), 1e-12)

    def test_profile_mismatches_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.retarget(calibration_profile=dict(path=PROFILE_PATH, sha256='0'*64))
        # Strict loader keeps refusing another height; launch paths switch off instead.
        with self.assertRaisesRegex(ValueError, 'human_height_m'):
            profile_module.load_calibration_profile(ROOT, dict(self.runtime.config['retargeting'], human_height_m=1.75),
                                                    ROOT/self.runtime.config['artifacts']['gmr_source_4']['path'])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.retarget(calibration_profile=dict(path=PROFILE_PATH, sha256='0'*64), human_height_m=1.75)
        with self.assertRaisesRegex(ValueError, 'inside the deployment root'):
            self.retarget(calibration_profile=dict(path='/etc/hostname', sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'inside the deployment root'):
            self.retarget(calibration_profile=dict(path='../x.json', sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'path and sha256'):
            self.retarget(calibration_profile=dict(path=PROFILE_PATH))
        (ROOT/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            bad_task = dict(self.profile['tasks'], no_such_link=self.profile['tasks']['base_link'])
            bad_quat = copy.deepcopy(self.profile['tasks'])
            bad_quat['base_link']['rotation_offset_wxyz'] = [1, 1, 0, 0]
            far = copy.deepcopy(self.profile['tasks'])
            far['base_link']['position_offset'] = [0, 0, .8]
            for changes, message in ((dict(base_config_sha256='0'*64), 'another GMR configuration'),
                                     (dict(schema='other'), 'schema'), (dict(robot='g1'), 'robot'),
                                     (dict(offset_to_ground=False), 'offset_to_ground'),
                                     (dict(tasks=bad_task), 'unknown GMR task'),
                                     (dict(tasks=bad_quat), 'Invalid calibration offsets'),
                                     (dict(tasks=far), 'Invalid calibration offsets'),
                                     (dict(extra=1), 'schema')):
                with self.subTest(changes=list(changes)), self.assertRaisesRegex(ValueError, message):
                    self.retarget(calibration_profile=self.scratch_profile(tmp, **changes))
                # A different launch height does not hide an invalid profile.
                with self.subTest(changes=list(changes), height=1.75), self.assertRaisesRegex(ValueError, message):
                    self.retarget(calibration_profile=self.scratch_profile(tmp, **changes), human_height_m=1.75)

    def test_shipped_configs_enable_calibration_and_planted_feet_at_180(self):
        for policy in ('sonic', 'supermimic', 'morphmimic'):
            with self.subTest(policy=policy):
                config = load_actor_config(ROOT, policy)
                settings = config['retargeting']
                self.assertEqual(settings['human_height_m'], 1.8)
                self.assertEqual(settings['calibration_profile'], profile_spec())
                self.assertEqual(foot_support_settings(settings['foot_support']), FootSupportSettings(**FOOT))
                self.assertIsNotNone(stream_settings(config))
        self.assertEqual(self.profile['human_height_m'], 1.8)
        self.assertNotIn('root_max_tilt_deg', load_actor_config(ROOT, 'sonic')['pico_stream'])
        rt = self.retarget()
        self.assertEqual(rt.calibration['path'], PROFILE_PATH)
        self.assertEqual(rt.foot_support, FootSupportSettings(**FOOT))
        self.assertIsNone(rt.calibration_notice)
        self.assertEqual(self.runtime.gmr_calibration, dict(calibration_profile=PROFILE_PATH, foot_support=True,
                                                            human_height_m=1.8, notice=None))

    def test_other_operator_height_switches_calibration_and_planted_feet_off(self):
        rt = self.retarget(human_height_m=1.7)
        self.assertIsNone(rt.calibration)
        self.assertIsNone(rt.foot_support)
        self.assertEqual(rt.calibration_notice, dict(reason='human_height_mismatch', profile=PROFILE_PATH,
            fitted_human_height_m=1.8, human_height_m=1.7, calibration_profile_disabled=True,
            foot_support_disabled=True))
        # Same engine as a launch without either setting.
        config = copy.deepcopy(self.runtime.config)
        for key in ('calibration_profile', 'foot_support'):
            config['retargeting'].pop(key)
        config['retargeting']['human_height_m'] = 1.7
        plain = create_retargeter(ROOT, config, self.runtime.model.contract, self.runtime.math)
        self.assertEqual(rt.engine.ik_match_table1, plain.engine.ik_match_table1)
        body = np.load(ROOT/'tests/fixtures/actor_pico/bridge_gmr_golden.npz')['body_poses']
        for frame in body[:5]:
            np.testing.assert_array_equal(rt.solve(frame)[0], plain.solve(frame)[0])
        runtime = ActorPicoRuntime(ROOT, backend='auto', human_height=1.7)
        self.assertEqual(runtime.gmr_calibration['notice'], rt.calibration_notice)
        self.assertEqual((runtime.gmr_calibration['calibration_profile'], runtime.gmr_calibration['foot_support']),
                         (None, False))
        np.testing.assert_allclose(runtime.head.offset, plain_head_offset(), atol=0)

    def test_head_control_uses_the_calibrated_chest_frame(self):
        config = load_actor_config(ROOT, 'sonic')
        config['retargeting'].update(calibration_profile=profile_spec())
        with patch('deployment.vita_s100_aorta_30dof.actor_pico_runtime.load_actor_config', return_value=config):
            runtime = ActorPicoRuntime(ROOT, backend='auto', human_height=1.8)
        wxyz = self.profile['tasks']['waist_pitch_link']['rotation_offset_wxyz']
        np.testing.assert_allclose(runtime.head.offset, Rotation.from_quat(np.roll(wxyz, -1)).as_matrix(), atol=1e-12)
        self.assertGreater(np.abs(runtime.head.offset-plain_head_offset()).max(), 1e-3)
        np.testing.assert_allclose(self.runtime.head.offset, runtime.head.offset, atol=0)

    def test_foot_support_settings_are_explicit_and_need_a_profile(self):
        self.assertIsNone(foot_support_settings(None))
        self.assertEqual(foot_support_settings(FOOT), FootSupportSettings(**FOOT))
        for change in (dict(enter_height_m=.06), dict(enter_speed_m_s=.5), dict(blend_s=0), dict(blend_s=True),
                       dict(speed_cutoff_hz=float('nan')), dict(unknown=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                foot_support_settings(dict(FOOT, **change))
        with self.assertRaises(ValueError):
            foot_support_settings({k: v for k, v in FOOT.items() if k != 'blend_s'})
        with self.assertRaisesRegex(ValueError, 'requires retargeting.calibration_profile'):
            self.retarget(calibration_profile=None, foot_support=FOOT)

    def test_foot_support_needs_frame_timestamps(self):
        rt = self.retarget(calibration_profile=profile_spec(), foot_support=FOOT)
        body = np.load(ROOT/'tests/fixtures/actor_pico/bridge_gmr_golden.npz')['body_poses']
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            rt.solve(body[0])
        for i, frame in enumerate(body):
            q, _, _ = rt.solve(frame, stamp_ns=1_000_000_000+i*12_000_000)
            self.assertTrue(np.isfinite(q).all())
        self.assertEqual(set(rt.last_foot_support), {'left', 'right'})


class StanceDetectorTests(unittest.TestCase):
    def setUp(self):
        self.settings = FootSupportSettings(**FOOT)
        self.detector = StanceDetector(self.settings)
        self.t = 1_000_000_000

    def step(self, left, right=(0., -.1, .1), dt=.012):
        self.t += int(dt*1e9)
        return self.detector.update(self.t, dict(left=np.array(left), right=np.array(right)))

    def test_planted_start_lift_and_return(self):
        state = self.step((0., .1, .1))
        self.assertTrue(state['left']['stance'] and state['right']['stance'])
        self.assertEqual(state['left']['weight'], 1.)
        lifted = [self.step((0., .1, .18))['left'] for _ in range(20)]
        self.assertFalse(any(s['stance'] for s in lifted))
        weights = [s['weight'] for s in lifted]
        np.testing.assert_allclose(np.diff(weights[:12]), -.012/self.settings.blend_s)
        self.assertEqual(weights[-1], 0.)
        down = [self.step((0., .1, .1))['left'] for _ in range(30)]
        entered = next(i for i, s in enumerate(down) if s['stance'])
        # Enter needs enter_hold_s below both thresholds.
        self.assertAlmostEqual((entered+1)*.012, math.ceil(self.settings.enter_hold_s/.012)*.012)
        self.assertEqual(down[entered-1]['weight'], 0.)
        self.assertGreater(down[-1]['weight'], .9)

    def test_sliding_foot_leaves_stance_and_gaps_reset(self):
        self.step((0., .1, .1))
        slid = [self.step((.012*.6*k, .1, .1))['left'] for k in range(1, 30)]
        self.assertFalse(slid[-1]['stance'])
        self.assertGreater(slid[-1]['speed_m_s'], self.settings.exit_speed_m_s)
        state = self.step((0., .1, .1), dt=.6)
        self.assertTrue(state['left']['stance'])
        self.assertEqual((state['left']['weight'], state['left']['speed_m_s']), (1., 0.))


class PlantedTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = ActorPicoRuntime(ROOT, backend='auto', human_height=1.8)
        config = copy.deepcopy(runtime.config)
        config['retargeting'].update(calibration_profile=profile_spec(), foot_support=FOOT)
        cls.engine = create_retargeter(ROOT, config, runtime.model.contract, runtime.math).engine

    def test_targets_are_level_on_the_ground_and_rigidly_consistent(self):
        import mujoco as mj
        e = self.engine
        data = mj.MjData(e.model)
        data.qpos[:] = e.default_qpos
        data.qpos[3:7] = Rotation.from_euler('z', .7).as_quat()[[3, 0, 1, 2]]
        mj.mj_kinematics(e.model, data)
        geom = e._foot_geometry['left']
        toe = mj.mj_name2id(e.model, mj.mjtObj.mjOBJ_BODY, 'left_toe_link')
        ankle = mj.mj_name2id(e.model, mj.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        home_toe = Rotation.from_quat(data.xquat[toe][[1, 2, 3, 0]])
        # An untilted (yawed home) toe target maps onto itself, so the ankle lands on the home ankle.
        (toe_rot, toe_pos), (ankle_rot, ankle_pos) = planted_targets(home_toe, data.xpos[toe]+[.01, .02, .05], geom)
        self.assertLess((toe_rot.inv()*home_toe).magnitude(), 1e-9)
        np.testing.assert_allclose(toe_pos[:2], data.xpos[toe][:2]+[.01, .02], atol=1e-12)
        self.assertAlmostEqual(toe_pos[2], geom.ground_toe_z, places=12)
        lift = geom.ground_toe_z-data.xpos[toe][2]
        np.testing.assert_allclose(ankle_pos-[.01, .02, 0.], data.xpos[ankle]+[0, 0, lift], atol=1e-9)
        # A tilted target only keeps its heading: the result differs from the home sole by a pure yaw.
        tilted = home_toe*Rotation.from_euler('xy', [.12, -.2])
        (toe_rot, toe_pos), (ankle_rot, ankle_pos) = planted_targets(tilted, data.xpos[toe], geom)
        rel = (toe_rot*home_toe.inv()).as_rotvec()
        np.testing.assert_allclose(rel[:2], 0, atol=1e-12)
        forward = tilted.apply([1., 0., 0.]), toe_rot.apply([1., 0., 0.])
        self.assertAlmostEqual(math.atan2(forward[0][1], forward[0][0]), math.atan2(forward[1][1], forward[1][0]),
                               places=12)
        np.testing.assert_allclose(ankle_pos+ankle_rot.apply(geom.toe_in_ankle_pos), toe_pos, atol=1e-12)
        self.assertLess((ankle_rot*geom.toe_in_ankle_rot*toe_rot.inv()).magnitude(), 1e-12)

    def test_planted_session_frames_get_level_grounded_targets(self):
        # The achieved sole tilt also depends on how well the profile fits the
        # operator and on the IK task weights, so this checks what foot_support
        # owns: planted frames are detected and their IK targets are level,
        # grounded and rigidly consistent, and the solved toe sits on the ground.
        import mujoco as mj
        fixture = SessionFixture(SESSION)
        runtime = ActorPicoRuntime(ROOT, backend='auto', human_height=fixture.human_height_m)
        (ROOT/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            data = json.loads((ROOT/PROFILE_PATH).read_text())
            data['human_height_m'] = fixture.human_height_m
            path = Path(tmp)/'profile.json'
            path.write_text(json.dumps(data))
            config = copy.deepcopy(runtime.config)
            config['retargeting'].update(calibration_profile=profile_spec(path.relative_to(ROOT)), foot_support=FOOT)
            rt = create_retargeter(ROOT, config, runtime.model.contract, runtime.math)
        e = rt.engine
        tasks = {task.frame_name: task for task in e.tasks1}
        toes = {s: mj.mj_name2id(e.model, mj.mjtObj.mjOBJ_BODY, s+'_toe_link') for s in ('left', 'right')}
        planted, ground_error = [], []
        for i in range(150):
            q, _, _ = rt.solve(fixture.frame_body[i].astype(np.float64), stamp_ns=int(fixture.frame_stamp_ns[i]))
            self.assertTrue(np.isfinite(q).all())
            if i < 30:
                continue
            for side in ('left', 'right'):
                if rt.last_foot_support[side]['weight'] < 1:
                    continue
                planted.append(side)
                geom = e._foot_geometry[side]
                toe_rot, toe_pos = e._target(tasks[side+'_toe_link'])
                ankle_rot, ankle_pos = e._target(tasks[side+'_ankle_roll_link'])
                np.testing.assert_allclose((toe_rot*geom.level.inv()).as_rotvec()[:2], 0, atol=1e-9)
                self.assertAlmostEqual(toe_pos[2], geom.ground_toe_z, places=9)
                np.testing.assert_allclose(ankle_pos+ankle_rot.apply(geom.toe_in_ankle_pos), toe_pos, atol=1e-9)
                ground_error.append(e.configuration.data.xpos[toes[side]][2]-geom.ground_toe_z)
        self.assertGreater(len(planted), 120)
        self.assertLess(np.median(np.abs(ground_error)), .005)


class SonicPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ActorPicoRuntime(ROOT, backend='auto', policy='sonic')
        cls.stream_config = copy.deepcopy(cls.runtime.config)
        cls.stream_config['pico_stream'] = copy.deepcopy(
            json.loads((ROOT/'config/actor_pico_supermimic.json').read_text())['pico_stream'])

    def with_stream(self):
        r = self.runtime
        r.config = copy.deepcopy(self.stream_config)
        r.stream = stream_settings(r.config)
        r.reset_inputs()
        return r

    def tearDown(self):
        r = self.runtime
        r.config = load_actor_config(ROOT, 'sonic')
        r.stream = stream_settings(r.config)
        r.reset_inputs()

    def test_trimmed_window_matches_interpolation_over_all_samples(self):
        c, m = self.runtime.model.contract, self.runtime.math
        ref = CausalActorReference(c, m, history_s=.41)
        rng = np.random.default_rng(3)
        stamps = 1_000_000_000+np.cumsum(rng.integers(8_000_000, 40_000_000, 300))
        qs = rng.normal(size=(300, 28))
        for s, q in zip(stamps, qs):
            ref.append(int(s), q, [1, 0, 0, 0], received_monotonic=1.)
        self.assertLess(len(ref.samples), 60)
        t = (stamps-stamps[0])*1e-9
        query = t[-1]-ref.delay+np.arange(ref.steps)*c.period
        expected = np.stack([np.interp(query, t, qs[:, j]) for j in range(28)], axis=1)
        np.testing.assert_array_equal(np.asarray(ref.motion().q), expected)
        exported = dict(ref.export(), clock_offset_s=0.)
        played, info = ref.playback(exported, t[-1], 0.)
        np.testing.assert_array_equal(np.asarray(played.q), expected)
        self.assertEqual(info['playback_lead_s'], 0.)

    def test_sonic_accepts_pico_stream_and_advances_with_the_control_clock(self):
        r = self.with_stream()
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
        np.testing.assert_allclose(np.diff(times), .02, atol=1e-12)

    def test_frozen_session_playback_never_repeats_or_reverses(self):
        r = self.with_stream()
        fixture = SessionFixture(SESSION)
        replay = SessionReplay(r, fixture, config=r.config)
        start, _ = fixture.following_segments()[2]
        times = [replay.current_source_time(s, now) for now, s in replay.ticks(start, start+8.)
                 if s.get('state') == 'ready']
        stats = advance_statistics(times)
        self.assertEqual((stats['repeated'], stats['backwards']), (0., 0))
        self.assertGreater(stats['nominal'], .9)


if __name__ == '__main__':
    unittest.main()
