"""Operator calibration and planted-foot targets for the pinned GMR.

Both are selected through the actor ``retargeting`` block and leave the
vendored solver, IK configuration and MJCF byte-for-byte unchanged:

* ``calibration_profile`` replaces per-task position/rotation offsets with a
  hash-pinned profile fitted on a confirmed neutral stance. The offsets live in
  GMR's height-scaled, ground-normalised human space, so a profile is only
  valid for the base configuration, operator height and ``offset_to_ground``
  it was fitted with. A different operator height switches the profile and
  planted-foot targets off for that launch (see ``operator_calibration``);
  every other mismatch raises instead of silently mis-mapping.
* ``foot_support`` rewrites the toe/ankle task targets of a foot classified as
  planted: level sole with the tracked heading, on the calibrated ground, and
  the ankle target derived from the toe through the model's rigid foot offset.
  Independent ankle and toe position targets otherwise disagree with the robot
  foot geometry and the IK resolves the conflict by tilting the foot. Swing
  feet keep the tracked targets; a weight blends between the two.
"""
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CALIBRATION_SCHEMA = 'vita_gmr_task_calibration_v1'
_PROFILE_REQUIRED = {'schema', 'robot', 'base_config_sha256', 'human_height_m', 'offset_to_ground', 'tasks'}
_PROFILE_OPTIONAL = {'provenance'}
# Fitted offsets attach a tracked human joint to a robot link; tens of
# centimetres occur (waist ~25 cm). Beyond this the profile is not plausible.
MAX_POSITION_OFFSET_M = .5


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_profile(root, settings, base_config_path, *, check_height):
    spec = settings['calibration_profile']
    if not isinstance(spec, dict) or set(spec) != {'path', 'sha256'}:
        raise ValueError('retargeting.calibration_profile needs exactly path and sha256')
    if root is None:
        raise ValueError('A calibration profile needs the deployment root')
    root = Path(root).resolve()
    relative = Path(str(spec['path']))
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError('Calibration profile must be a path inside the deployment root')
    if not path.is_file():
        raise ValueError('Calibration profile missing: '+str(relative))
    if sha256_file(path) != spec['sha256']:
        raise ValueError('Calibration profile hash mismatch: '+str(relative))
    profile = json.loads(path.read_text())
    if (not isinstance(profile, dict) or profile.get('schema') != CALIBRATION_SCHEMA
            or _PROFILE_REQUIRED-set(profile) or set(profile)-_PROFILE_REQUIRED-_PROFILE_OPTIONAL):
        raise ValueError('Unsupported GMR calibration profile; expected schema '+CALIBRATION_SCHEMA)
    if profile['robot'] != settings.get('robot'):
        raise ValueError('Calibration profile robot differs from retargeting.robot')
    if profile['base_config_sha256'] != sha256_file(base_config_path):
        raise ValueError('Calibration profile was fitted against another GMR configuration')
    height, fitted = float(settings['human_height_m']), profile['human_height_m']
    if isinstance(fitted, bool) or not isinstance(fitted, (int, float)) or not math.isfinite(fitted):
        raise ValueError('Calibration profile has an invalid human_height_m')
    if check_height and abs(float(fitted)-height) > 1e-9:
        raise ValueError(f'Calibration profile was fitted at human_height_m={fitted}, this launch uses '
                         f'{height}; offsets live in the height-scaled space, refit for this operator')
    if profile['offset_to_ground'] is not True or settings.get('offset_to_ground') is not True:
        raise ValueError('Calibration profiles require offset_to_ground in both the fit and the launch')
    tasks = profile['tasks']
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError('Calibration profile has no tasks')
    base = json.loads(Path(base_config_path).read_text())
    parsed = {}
    for frame, entry in tasks.items():
        if any(frame not in base[key] for key in ('ik_match_table1', 'ik_match_table2')):
            raise ValueError('Calibration profile names an unknown GMR task: '+str(frame))
        if not isinstance(entry, dict) or set(entry) != {'position_offset', 'rotation_offset_wxyz'}:
            raise ValueError('Calibration task needs position_offset and rotation_offset_wxyz: '+str(frame))
        position = np.asarray(entry['position_offset'], dtype=float)
        rotation = np.asarray(entry['rotation_offset_wxyz'], dtype=float)
        if (position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(rotation).all() or abs(np.linalg.norm(rotation)-1) > 1e-6
                or np.linalg.norm(position) > MAX_POSITION_OFFSET_M):
            raise ValueError('Invalid calibration offsets for task '+str(frame))
        parsed[str(frame)] = (position.tolist(), rotation.tolist())
    return dict(path=str(relative), sha256=spec['sha256'], tasks=parsed,
                human_height_m=float(fitted), provenance=profile.get('provenance'))


def load_calibration_profile(root, settings, base_config_path):
    """Validated ``retargeting.calibration_profile``, or None when absent.

    Strict: a launch height other than the fitted one raises. Launch paths use
    ``operator_calibration``, which switches the profile off instead.
    """
    if settings.get('calibration_profile') is None:
        return None
    return _parse_profile(root, settings, base_config_path, check_height=True)


def operator_calibration(root, settings, base_config_path):
    """Effective ``(calibration, foot_support, notice)`` for this launch.

    A profile is fitted for one operator height. At another height both the
    profile and the planted-foot targets (which need its calibrated ground) are
    off for the launch and ``notice`` records why; otherwise ``notice`` is None.
    Every other profile or setting problem raises, at any height.
    """
    support = foot_support_settings(settings.get('foot_support'))
    calibration = None
    if settings.get('calibration_profile') is not None:
        calibration = _parse_profile(root, settings, base_config_path, check_height=False)
    # The planted sole height comes from the calibrated home pose; the
    # shipped offsets do not place the human ground on the robot ground.
    if support is not None and calibration is None:
        raise ValueError('retargeting.foot_support requires retargeting.calibration_profile')
    height = float(settings['human_height_m'])
    if calibration is not None and abs(calibration['human_height_m']-height) > 1e-9:
        return None, None, dict(reason='human_height_mismatch', profile=calibration['path'],
                                fitted_human_height_m=calibration['human_height_m'], human_height_m=height,
                                calibration_profile_disabled=True, foot_support_disabled=support is not None)
    return calibration, support, None

def apply_calibration(config, calibration):
    """Write profile offsets into both IK tables of a parsed GMR configuration."""
    for frame, (position, rotation) in calibration['tasks'].items():
        for key in ('ik_match_table1', 'ik_match_table2'):
            table = config[key]
            if frame not in table:
                raise ValueError('Calibration profile names an unknown GMR task: '+frame)
            table[frame][3] = list(position)
            table[frame][4] = list(rotation)
    return config


@dataclass(frozen=True)
class FootSupportSettings:
    enter_height_m: float
    exit_height_m: float
    enter_speed_m_s: float
    exit_speed_m_s: float
    enter_hold_s: float
    blend_s: float
    speed_cutoff_hz: float


FOOT_SUPPORT_KEYS = tuple(FootSupportSettings.__dataclass_fields__)


def foot_support_settings(raw):
    """Validated ``retargeting.foot_support``, or None when absent."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(FOOT_SUPPORT_KEYS)-set(raw) or set(raw)-set(FOOT_SUPPORT_KEYS)-{'note'}:
        raise ValueError('retargeting.foot_support needs exactly '+', '.join(FOOT_SUPPORT_KEYS))
    values = {}
    for key in FOOT_SUPPORT_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('retargeting.foot_support.'+key+' must be a finite number')
        values[key] = float(value)
    s = FootSupportSettings(**values)
    if (not 0 < s.enter_height_m < s.exit_height_m <= .3 or not 0 < s.enter_speed_m_s < s.exit_speed_m_s <= 3
            or not 0 <= s.enter_hold_s <= .5 or not 0 < s.blend_s <= 1 or not 0 < s.speed_cutoff_hz <= 50):
        raise ValueError('retargeting.foot_support needs 0<enter_height<exit_height<=0.3 m, '
                         '0<enter_speed<exit_speed<=3 m/s, enter_hold in [0,0.5] s, '
                         'blend in (0,1] s and speed cutoff in (0,50] Hz')
    return s


SIDES = (('left', 'Left_Foot'), ('right', 'Right_Foot'))


class StanceDetector:
    """Per-foot planted/swing classification of the processed human feet.

    Height is measured against the lower foot: ``offset_human_data_to_ground``
    re-anchors the lowest foot every frame, so absolute height carries no
    information. Speed is the low-passed horizontal foot speed. Entering
    stance needs both below their enter thresholds for ``enter_hold_s``;
    either above its exit threshold leaves at once. The blend weight ramps
    over ``blend_s`` so a transition never steps the reference. A missing or
    out-of-order timestamp, or a gap above ``RESET_GAP_S``, restarts from the
    height-only classification, like the stream filters do.
    """
    RESET_GAP_S = .5

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.last_ns = None
        self.state = {side: None for side, _ in SIDES}

    def update(self, stamp_ns, positions):
        """positions: {side: xyz}. Returns {side: dict(stance, weight, height_m, speed_m_s)}."""
        s = self.settings
        stamp_ns = int(stamp_ns)
        dt = None if self.last_ns is None else (stamp_ns-self.last_ns)*1e-9
        if dt is not None and not 0 < dt <= self.RESET_GAP_S:
            dt = None
        self.last_ns = stamp_ns
        low = min(float(positions[side][2]) for side, _ in SIDES)
        result = {}
        for side, _ in SIDES:
            position = np.asarray(positions[side], dtype=float)
            height, xy = float(position[2])-low, position[:2].copy()
            previous = self.state[side]
            if dt is None or previous is None:
                stance = height < s.enter_height_m
                state = dict(stance=stance, weight=1. if stance else 0., hold=0., speed=0., xy=xy)
            else:
                raw = float(np.linalg.norm(xy-previous['xy']))/dt
                alpha = 1./(1.+1./(2*math.pi*s.speed_cutoff_hz*dt))
                speed = previous['speed']+alpha*(raw-previous['speed'])
                stance, hold = previous['stance'], previous['hold']
                if stance:
                    if height > s.exit_height_m or speed > s.exit_speed_m_s:
                        stance, hold = False, 0.
                elif height < s.enter_height_m and speed < s.enter_speed_m_s:
                    hold += dt
                    stance = hold >= s.enter_hold_s
                else:
                    hold = 0.
                if stance:
                    hold = 0.
                step = dt/s.blend_s
                weight = min(1., previous['weight']+step) if stance else max(0., previous['weight']-step)
                state = dict(stance=stance, weight=weight, hold=hold, speed=speed, xy=xy)
            self.state[side] = state
            result[side] = dict(stance=bool(state['stance']), weight=float(state['weight']),
                                height_m=height, speed_m_s=float(state['speed']))
        return result


@dataclass(frozen=True)
class FootGeometry:
    toe_in_ankle_pos: np.ndarray
    toe_in_ankle_rot: Rotation
    level: Rotation
    ground_toe_z: float


def planted_targets(toe_rot, toe_pos, geometry):
    """Rigidly consistent planted toe and ankle targets for a tracked toe target.

    Keeps the tracked heading (local +X projected on the ground; immune to the
    euler gimbal) and horizontal toe position, levels the sole and puts it on
    the calibrated ground. Returns ((toe_rot, toe_pos), (ankle_rot, ankle_pos)).
    """
    forward = toe_rot.apply([1., 0., 0.])
    yaw = math.atan2(forward[1], forward[0])
    toe_rot = Rotation.from_euler('z', yaw)*geometry.level
    toe_pos = np.array([toe_pos[0], toe_pos[1], geometry.ground_toe_z])
    ankle_rot = toe_rot*geometry.toe_in_ankle_rot.inv()
    ankle_pos = toe_pos-ankle_rot.apply(geometry.toe_in_ankle_pos)
    return (toe_rot, toe_pos), (ankle_rot, ankle_pos)


def _blend(rot_a, pos_a, rot_b, pos_b, weight):
    rot = rot_a*Rotation.from_rotvec(weight*(rot_a.inv()*rot_b).as_rotvec())
    return rot, np.asarray(pos_a)+weight*(np.asarray(pos_b)-np.asarray(pos_a))


def foot_geometry(model, home_qpos, ground_geom_ids, clearance, side):
    """Planted-foot geometry from the model at its home pose, ground-snapped like GMR."""
    import mujoco as mj
    geom_utils = importlib.import_module('motion_stream_bridge.gmr_lite.geom_utils')
    data = mj.MjData(model)
    data.qpos[:] = home_qpos
    mj.mj_kinematics(model, data)
    if not ground_geom_ids:
        raise ValueError('foot_support needs the GMR ground_snap foot geoms')
    lift = clearance-min(geom_utils.geom_min_z(model, data, g) for g in ground_geom_ids)
    toe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, side+'_toe_link')
    ankle = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, side+'_ankle_roll_link')
    if toe < 0 or ankle < 0 or model.body_parentid[toe] != ankle or model.body_jntnum[toe] != 0:
        raise ValueError('foot_support expects '+side+'_toe_link rigidly attached to its ankle roll link')
    home = Rotation.from_quat(data.xquat[toe][[1, 2, 3, 0]])
    forward = home.apply([1., 0., 0.])
    level = Rotation.from_euler('z', -math.atan2(forward[1], forward[0]))*home
    return FootGeometry(model.body_pos[toe].copy(), Rotation.from_quat(model.body_quat[toe][[1, 2, 3, 0]]),
                        level, float(data.xpos[toe][2]+lift))


def foot_support_engine(base):
    """Subclass of a GMR engine class that rewrites planted-foot targets."""
    import mink

    class FootSupportRetargeting(base):
        def configure_foot_support(self, settings):
            self.foot_support = settings
            self.stance = StanceDetector(settings)
            self._foot_geometry = {side: foot_geometry(self.model, self._default_qpos, self.ground_snap_geom_ids,
                                                       self.ground_snap_clearance, side) for side, _ in SIDES}
            self._foot_tasks = {}
            for side, _ in SIDES:
                pairs = []
                for used, tasks in ((self.use_ik_match_table1, self.tasks1), (self.use_ik_match_table2, self.tasks2)):
                    if not used:
                        continue
                    named = {task.frame_name: task for task in tasks}
                    if side+'_toe_link' not in named or side+'_ankle_roll_link' not in named:
                        raise ValueError('foot_support needs '+side+' toe and ankle IK tasks')
                    pairs.append((named[side+'_toe_link'], named[side+'_ankle_roll_link']))
                self._foot_tasks[side] = pairs
            self._frame_ns = None
            self.foot_support_state = None

        def set_frame_time(self, stamp_ns):
            self._frame_ns = int(stamp_ns)

        def update_targets(self, human_data, offset_to_ground=False):
            super().update_targets(human_data, offset_to_ground)
            if not offset_to_ground:
                raise ValueError('foot_support needs offset_to_ground human data')
            if self._frame_ns is None:
                raise ValueError('foot_support needs the Pico frame timestamp of every solve')
            stamp, self._frame_ns = self._frame_ns, None
            human = self.scaled_human_data
            state = self.stance.update(stamp, {side: human[name][0] for side, name in SIDES})
            self.foot_support_state = state
            for side, _ in SIDES:
                weight = state[side]['weight']
                if weight <= 0.:
                    continue
                for toe_task, ankle_task in self._foot_tasks[side]:
                    self._plant(toe_task, ankle_task, self._foot_geometry[side], weight)

        @staticmethod
        def _target(task):
            pose = np.asarray(task.transform_target_to_world.wxyz_xyz, dtype=float)
            return Rotation.from_quat(pose[[1, 2, 3, 0]]), pose[4:]

        def _plant(self, toe_task, ankle_task, geometry, weight):
            toe_rot, toe_pos = self._target(toe_task)
            (toe_rot_s, toe_pos_s), (ankle_rot_s, ankle_pos_s) = planted_targets(toe_rot, toe_pos, geometry)
            if weight >= 1.:
                toe, ankle = (toe_rot_s, toe_pos_s), (ankle_rot_s, ankle_pos_s)
            else:
                ankle_rot, ankle_pos = self._target(ankle_task)
                toe = _blend(toe_rot, toe_pos, toe_rot_s, toe_pos_s, weight)
                ankle = _blend(ankle_rot, ankle_pos, ankle_rot_s, ankle_pos_s, weight)
            for task, (rot, pos) in ((toe_task, toe), (ankle_task, ankle)):
                task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3(rot.as_quat()[[3, 0, 1, 2]]), np.asarray(pos, dtype=float)))

    return FootSupportRetargeting
