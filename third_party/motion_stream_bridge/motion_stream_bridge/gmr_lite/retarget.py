"""General Motion Retargeting (GMR) — slim, robot-agnostic vendored core.

This is a trimmed copy of GMR's ``general_motion_retargeting/motion_retarget.py``
carrying ONLY the live-retarget path the motion-stream bridge uses. Two changes
versus upstream:

  * No ``rich`` / ``params`` / ``_geom_utils`` package imports — the geom helpers
    are vendored alongside (``geom_utils``) and print is the builtin.

  * **Robot-agnostic tuning.** Upstream hardcodes four module-level dicts
    (waist regularizer, prev-posture smoothing, ground snap, qpos smoothing)
    keyed by ``"<src_human>:<tgt_robot>"`` / ``"<tgt_robot>"`` strings, so adding
    or changing a robot means editing this file. Here those live in an optional
    ``"tuning"`` block inside each IK-config JSON, and the config also names its
    own ``"robot_xml"``. A new or changed humanoid (e.g. an updated vita02
    description) drops in as a config + XML with zero code edits. The
    ``qpos_smoothing`` spec additionally accepts extra ``"groups"`` entries so
    different joint sets can smooth with different alphas (bridge extension;
    a legacy single-group spec behaves exactly as upstream). A further bridge
    extension, ``tuning.input_pose_correction``, right-multiplies a fixed
    rotation onto one tracked human body's orientation before targets are
    built, blended by arm pose (see ``_build_input_pose_corrections``) — used
    to compensate a pose-dependent left/right asymmetry in the PICO's wrist
    tracking that no constant per-task ``rot_offset`` can express.

The retarget math is unchanged, so output qpos is bit-for-bit identical to
upstream GMR for the same input and equivalent config.
"""

from __future__ import annotations

import json
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from .geom_utils import collect_geom_ids_by_token, geom_min_z

_CONFIG_ROOT = Path(__file__).parent / "configs"


def resolve_config_path(name_or_path: str) -> Path:
    """A bare name (``"xrobot_to_g1"``) resolves to ``configs/<name>.json``;
    a path (absolute or containing a separator / ``.json`` suffix) is used as-is.
    This lets a future changed robot description ship as an external JSON without
    touching the vendored package."""
    p = Path(name_or_path)
    if p.suffix == ".json" or p.is_absolute() or len(p.parts) > 1:
        return p
    return _CONFIG_ROOT / f"{name_or_path}.json"


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR), config-driven.

    ``config`` may be a bare config name resolved under ``gmr_lite/configs`` or a
    path to an external IK-config JSON. ``src_human`` / ``tgt_robot`` are kept for
    log messages and back-compat; when ``config`` is omitted they select the
    vendored ``<src_human>_to_<tgt_robot>``-style config via the caller.
    """

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        config,
        actual_human_height: float = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = True,
        use_velocity_limit: bool = False,
    ) -> None:
        self.tgt_robot = tgt_robot
        self.src_human = src_human

        config_path = resolve_config_path(config)
        with open(config_path) as f:
            ik_config = json.load(f)
        if verbose:
            print("[GMR] Use IK config:", config_path)

        # Resolve the robot XML relative to the config file so a config can live
        # anywhere and still point at its model.
        robot_xml = ik_config["robot_xml"]
        self.xml_file = str((config_path.parent / robot_xml).resolve())
        if verbose:
            print("[GMR] Use robot model:", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # Per-robot tuning lives in the config now (was module-level dicts).
        self._tuning = ik_config.get("tuning", {}) or {}

        # Narrow the model's joint ranges to what the consuming policy will
        # actually accept. The MJCF carries the mechanical travel; a tracking
        # policy is trained on a soft margin inside it and its contract states
        # that margin explicitly. Applied HERE, before ConfigurationLimit is
        # built from the model, so the IK simply cannot solve outside it.
        self._apply_joint_limit_overrides(
            ik_config.get("joint_limits") or {}, verbose)
        # Read here, applied in setup_retarget_configuration() once the
        # configuration (and therefore _default_qpos) exists.
        self._home_qpos_spec = ik_config.get("home_qpos") or {}
        self._verbose = verbose

        # Optional wire layout: which model joints go on the stream, in what
        # order. When the retarget model carries joints the policy wire
        # doesn't (vita02a's neck) or chains them in a different order than
        # the wire convention (vita02a's wrist roll->yaw->pitch vs the wire's
        # roll->pitch->yaw), sources use this to subset/permute qpos[7:].
        # Absent = stream the full model joint order unchanged.
        self.stream_joint_names = ik_config.get("stream_joint_names")

        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i]
            )
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"[GMR] DoF {i}: {dof_name}")

        self.robot_body_names = {}
        for i in range(self.model.nbody):
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i

        self.robot_motor_names = {}
        for i in range(self.model.nu):
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i

        # compute the scale ratio based on given human height and the IK config's
        # assumption
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = (
                ik_config["human_scale_table"][key] * ratio
            )

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping
        self.first_frame_damping = max(float(damping), 2.0)
        self.first_frame_max_iter = max(int(self.max_iter), 10)
        self._is_first_frame = True

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self._arm_task_original_orientation_costs = {}
        self._first_frame_arm_orientation_cost = 1.0

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        # Velocity limit. NOTE the units this lands in: solve_ik is called with
        # dt = model.opt.timestep (2 ms here), and the solve loop runs up to
        # max_iter+1 times per retargeted frame, so a limit of L rad/s caps a
        # joint at roughly L * timestep * iterations per frame — with the old
        # hardcoded 3*pi that is ~12 deg/frame at ~100 Hz, i.e. no constraint
        # at all. It only bites at values well below that, which is why the
        # ceiling is config-driven instead of a constant.
        vel_spec = self._tuning.get("velocity_limit") or {}
        if use_velocity_limit or vel_spec.get("enabled"):
            default = float(vel_spec.get("default_rad_s", 3 * np.pi))
            per_joint = vel_spec.get("joints") or {}
            limits = {
                name: float(per_joint.get(name, default))
                for name in self.robot_motor_names
            }
            self.ik_limits.append(mink.VelocityLimit(self.model, limits))
            if verbose:
                print(f"[GMR] IK velocity limit on ({default} rad/s default, "
                      f"{len(per_joint)} per-joint overrides)")

        self.setup_retarget_configuration()

        self.ground_offset = 0.0

        # Runtime recalibration (see request_recalibration): frames still to
        # capture, per-body accumulators, and the caller's completion hook.
        self._recal_frames_left = 0
        self._recal_acc: dict = {}
        self._recal_on_done = None

    @property
    def default_qpos(self) -> np.ndarray:
        """The model's home qpos captured at setup ([root pos 3, root quat
        wxyz 4, joints]) — the pose the robot model holds before any
        retargeting has run."""
        return self._default_qpos

    def _apply_joint_limit_overrides(self, spec, verbose: bool) -> None:
        """Replace `model.jnt_range` for the named joints, `{name: [lo, hi]}`
        in radians. Must run before `mink.ConfigurationLimit`, which snapshots
        the ranges. An override wider than the mechanical travel is refused —
        the point is to narrow the IK's freedom, never to grant it more."""
        applied = 0
        for name, bounds in spec.items():
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                print(f"[GMR] joint_limits: no joint {name!r} in the model")
                continue
            lo, hi = (float(bounds[0]), float(bounds[1]))
            mech_lo, mech_hi = self.model.jnt_range[joint_id]
            if lo < mech_lo - 1e-9 or hi > mech_hi + 1e-9:
                print(f"[GMR] joint_limits: {name} [{lo:.4f}, {hi:.4f}] is "
                      f"wider than the model's [{mech_lo:.4f}, {mech_hi:.4f}] "
                      f"— ignored")
                continue
            self.model.jnt_range[joint_id] = (lo, hi)
            applied += 1
        if applied and verbose:
            print(f"[GMR] narrowed {applied} joint range(s) from the config")

    def _apply_home_qpos_override(self, spec, verbose: bool) -> None:
        """Move the neutral pose off the MJCF's `qpos0`, `{name: rad}`.

        `qpos0` is all zeros for the vita02a models (they define no keyframe),
        which puts the knees exactly on their lower stop — a straight leg is a
        kinematic singularity, so the knee contributes almost no gradient and
        the IK cannot shorten the leg to lift a foot. The same pose is what the
        posture regularizers pull toward, what the safe return walks to and
        where an engage ramp starts from, so it is worth stating explicitly
        rather than inheriting. Use the consuming policy's own default pose."""
        if not spec:
            return
        clipped = []
        for name, value in spec.items():
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                print(f"[GMR] home_qpos: no joint {name!r} in the model")
                continue
            lo, hi = self.model.jnt_range[joint_id]
            val = float(value)
            if self.model.jnt_limited[joint_id] and not (lo <= val <= hi):
                clipped.append(name)
                val = float(np.clip(val, lo, hi))
            self._default_qpos[self.model.jnt_qposadr[joint_id]] = val
        if clipped:
            print(f"[GMR] home_qpos: clipped to the joint range: {clipped}")
        # The neutral pose is also the starting configuration, so the first
        # solve begins there instead of slewing out of the singular one.
        self.configuration.data.qpos[:] = self._default_qpos
        mj.mj_forward(self.model, self.configuration.data)
        if verbose:
            print(f"[GMR] home pose overridden for {len(spec)} joint(s)")

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        self._default_qpos = self.configuration.data.qpos.copy()
        self._apply_home_qpos_override(
            self._home_qpos_spec, getattr(self, "_verbose", False))
        self.posture_task = mink.PostureTask(self.model, cost=1e-2)
        self.posture_task.set_target(self._default_qpos)
        self.prev_posture_task = mink.PostureTask(
            self.model,
            cost=self._build_prev_posture_cost(),
        )
        self.prev_posture_task.set_target(self._default_qpos)
        self.waist_posture_task = self._build_waist_regularizer_task()
        self.ground_snap_geom_ids, self.ground_snap_clearance = self._build_ground_snap()
        self.qpos_smoothing_groups = self._build_qpos_smoothing_spec()
        # Public on purpose: a source feeding synthetic (perfectly mirrored)
        # human data can clear this to () to opt out of the tracker
        # compensation (see PicoGmrSource --mock).
        self.input_pose_corrections = self._build_input_pose_corrections()
        # Public alongside input_pose_corrections and cleared the same way: a
        # source feeding synthetic data whose pelvis is already correct (the
        # mock) can set this to None to opt out.
        self.pelvis_yaw_pin = self._build_pelvis_yaw_pin()

        self.tasks1 = []
        self.tasks2 = []

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if self._is_nonzero_cost(pos_weight) or self._is_nonzero_cost(rot_weight):
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1.setdefault(body_name, []).append(task)
                self.pos_offsets1.setdefault(body_name, []).append(
                    np.array(pos_offset) - self.ground
                )
                self.rot_offsets1.setdefault(body_name, []).append(
                    R.from_quat(rot_offset, scalar_first=True)
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
                if self._is_arm_body(body_name):
                    self._arm_task_original_orientation_costs[task] = np.atleast_1d(
                        np.asarray(rot_weight, dtype=float)
                    )

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if self._is_nonzero_cost(pos_weight) or self._is_nonzero_cost(rot_weight):
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2.setdefault(body_name, []).append(task)
                self.pos_offsets2.setdefault(body_name, []).append(
                    np.array(pos_offset) - self.ground
                )
                self.rot_offsets2.setdefault(body_name, []).append(
                    R.from_quat(rot_offset, scalar_first=True)
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []
                if self._is_arm_body(body_name):
                    self._arm_task_original_orientation_costs[task] = np.atleast_1d(
                        np.asarray(rot_weight, dtype=float)
                    )

        self._flat_targets1 = self._flatten_target_table(
            self.human_body_to_task1, self.pos_offsets1, self.rot_offsets1
        )
        self._flat_targets2 = self._flatten_target_table(
            self.human_body_to_task2, self.pos_offsets2, self.rot_offsets2
        )

    @staticmethod
    def _flatten_target_table(body_to_tasks, pos_offsets, rot_offsets):
        """Stack a table's tasks/offsets so the per-frame target math can run as
        a handful of batched scipy calls instead of 5 single-quat calls per task
        (each single-quat Rotation op costs ~2-5 us of pure call overhead, which
        dominated retarget() at ~40% of frame time)."""
        tasks, bodies, flat_pos, flat_rot = [], [], [], []
        for body_name, body_tasks in body_to_tasks.items():
            for task, pos_offset, rot_offset in zip(
                body_tasks, pos_offsets[body_name], rot_offsets[body_name]
            ):
                tasks.append(task)
                bodies.append(body_name)
                flat_pos.append(pos_offset)
                flat_rot.append(rot_offset.as_quat())
        if not tasks:
            return None
        return tasks, bodies, np.asarray(flat_pos), R.from_quat(np.asarray(flat_rot))

    @staticmethod
    def _is_nonzero_cost(cost):
        return bool(np.any(np.asarray(cost, dtype=float) != 0.0))

    def _build_waist_regularizer_task(self):
        """Per-DOF posture task that pins regularizer joints to default qpos."""
        spec = self._tuning.get("waist_regularizer")
        if not spec:
            return None

        cost_vec = np.zeros(self.model.nv)
        missing = []
        for joint_name in spec.get("joints", ()):
            dof_idx = self.robot_dof_names.get(joint_name)
            if dof_idx is None:
                missing.append(joint_name)
            else:
                cost_vec[dof_idx] = spec["cost"]
        for joint_name, joint_cost in spec.get("joint_costs", {}).items():
            dof_idx = self.robot_dof_names.get(joint_name)
            if dof_idx is None:
                missing.append(joint_name)
            else:
                cost_vec[dof_idx] = float(joint_cost)

        if missing:
            print(
                f"[GMR] warning: waist regularizer skipped — joints not found "
                f"in {self.tgt_robot}: {missing}"
            )
            return None

        task = mink.PostureTask(self.model, cost=cost_vec)
        task.set_target(self._default_qpos)
        return task

    def _build_prev_posture_cost(self):
        spec = self._tuning.get("prev_posture")
        if not spec:
            return 1e-3

        cost_vec = np.full(self.model.nv, float(spec["default_cost"]))
        missing = []
        for joint_name, joint_cost in spec.get("joint_costs", {}).items():
            dof_idx = self.robot_dof_names.get(joint_name)
            if dof_idx is None:
                missing.append(joint_name)
            else:
                cost_vec[dof_idx] = float(joint_cost)
        if missing:
            print(
                f"[GMR] warning: prev_posture per-joint cost ignored — joints "
                f"not found in {self.tgt_robot}: {missing}"
            )
        return cost_vec

    def _build_ground_snap(self):
        """Returns (geom_ids, clearance) for foot-snapping; ([], 0.0) if disabled."""
        spec = self._tuning.get("ground_snap")
        if not spec:
            return [], 0.0

        token = spec["geom_name_token"]
        geom_ids = collect_geom_ids_by_token(self.model, token)
        if not geom_ids:
            print(
                f"[GMR] warning: ground snap disabled — no geom name contains "
                f"'{token}' in {self.tgt_robot}"
            )
        return geom_ids, float(spec["clearance"])

    def _build_qpos_smoothing_spec(self):
        """Returns a tuple of (alpha, qpos_indices) EMA groups.

        The spec may be a single ``{"alpha": ..., "joints": [...]}`` group
        and/or carry extra groups under ``"groups"`` so different joint sets
        smooth with different alphas (e.g. heavy waist EMA, light wrist EMA).
        """
        spec = self._tuning.get("qpos_smoothing")
        if not spec:
            return ()

        group_specs = []
        if spec.get("joints"):
            group_specs.append(spec)
        group_specs.extend(spec.get("groups", ()))

        groups = []
        for group in group_specs:
            qpos_indices = []
            missing = []
            for joint_name in group.get("joints", ()):
                joint_id = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id == -1:
                    missing.append(joint_name)
                    continue
                qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))

            if missing:
                print(
                    f"[GMR] warning: qpos smoothing group skipped — joints not "
                    f"found in {self.tgt_robot}: {missing}"
                )
                continue
            if qpos_indices:
                groups.append((float(group["alpha"]), tuple(qpos_indices)))
        return tuple(groups)

    def _build_input_pose_corrections(self):
        """Returns ((body, rotvec, blend), ...) from ``tuning.input_pose_correction``
        (a single spec or a list of them).

        Each spec right-multiplies ``rotvec`` (axis*angle, radians, in the
        tracked body's LOCAL frame) onto that body's orientation before task
        targets are built — i.e. it corrects the *human input*, so the same
        values are valid for every robot config. The optional ``blend`` block
        scales the rotation 0..1 by the elevation of the ``from_body`` →
        ``to_body`` segment (degrees above horizontal): identity at/below
        ``zero_below_deg``, full at/above ``full_above_deg``, linear between.
        This expresses corrections that hold in one arm pose but must vanish
        in another — exactly what a constant per-task ``rot_offset`` cannot.
        """
        spec = self._tuning.get("input_pose_correction")
        if not spec:
            return ()
        corrections = []
        for entry in spec if isinstance(spec, list) else [spec]:
            blend = entry.get("blend")
            corrections.append((
                entry["body"],
                np.asarray(entry["rotvec"], dtype=float),
                None if not blend else (
                    blend["from_body"],
                    blend["to_body"],
                    float(blend["zero_below_deg"]),
                    float(blend["full_above_deg"]),
                ),
            ))
        return tuple(corrections)

    def _build_pelvis_yaw_pin(self):
        """Returns (body, reference_bodies, max_correction_rad) from
        ``tuning.pelvis_yaw_pin``, or None.

        Holds the tracked pelvis's YAW at a fixed offset from the reference
        bodies (normally the feet), leaving its position, pitch and roll alone.
        The offset is captured on the first frame, so the reference frames'
        arbitrary rest orientations cancel — only their CHANGE in heading
        moves the pelvis, which is what a planted stance provides. (Their
        absolute yaws are unusable directly: the tracked feet rest ~160 deg
        apart in the raw stream.)

        Needed when the operator's pelvis reference is a waist-mounted tracker:
        the waist joint tracks chest-relative-to-pelvis, so a tracker that turns
        WITH the operator moves both references together and the relative angle
        collapses. The IK then satisfies the rotated pelvis by yawing the free
        root — the whole robot pivots instead of its waist bending. Pinning the
        pelvis yaw to the planted feet restores the relative signal without
        touching the arm/leg calibration.

        ``max_correction_deg`` bounds how much yaw may be overridden, so a real
        whole-body turn (feet turning with the operator) still rotates the root
        instead of being absorbed as an ever-growing waist angle.
        """
        spec = self._tuning.get("pelvis_yaw_pin")
        if not spec:
            return None
        # Rest offset between the pelvis and the reference heading, captured on
        # the first frame seen (None until then).
        self._pelvis_yaw_rest_offset = None
        return (
            spec.get("body", self.human_root_name),
            tuple(spec["reference_bodies"]),
            np.radians(float(spec.get("max_correction_deg", 180.0))),
        )

    # Heading is read by projecting the frame's local +X into the ground plane,
    # NOT from an euler yaw: these tracked human frames sit at ~-87 deg 'zyx'
    # pitch (they are not Z-up-aligned), which is gimbal lock — there the yaw
    # term is numerically meaningless and swings ~155 deg between adjacent
    # poses. The projection is well conditioned for any frame whose local +X
    # is not near-vertical.
    _HEADING_AXIS = np.array([1.0, 0.0, 0.0])
    # Below this horizontal length the axis is too close to vertical to give a
    # trustworthy heading.
    _HEADING_MIN_HORIZ = 0.2

    @classmethod
    def _heading_of(cls, quat_wxyz):
        """Ground-plane heading (rad) of a frame's local +X, or None if that
        axis is too near-vertical to define one (or the quat is degenerate —
        an all-zero pose from a source that hasn't produced real data yet)."""
        quat = np.asarray(quat_wxyz, dtype=float)
        if float(quat @ quat) < 1e-6:
            return None
        vec = R.from_quat(quat[[1, 2, 3, 0]]).apply(cls._HEADING_AXIS)
        if np.hypot(vec[0], vec[1]) < cls._HEADING_MIN_HORIZ:
            return None
        return float(np.arctan2(vec[1], vec[0]))

    @classmethod
    def _mean_heading(cls, human_data, bodies):
        """Circular mean of the bodies' headings, or None if none are usable."""
        sin_sum = cos_sum = 0.0
        seen = False
        for name in bodies:
            if name not in human_data:
                continue
            heading = cls._heading_of(human_data[name][1])
            if heading is None:
                continue
            sin_sum += np.sin(heading)
            cos_sum += np.cos(heading)
            seen = True
        if not seen or (sin_sum == 0.0 and cos_sum == 0.0):
            return None
        return float(np.arctan2(sin_sum, cos_sum))

    def _apply_pelvis_yaw_pin(self, human_data):
        body_name, reference_bodies, max_corr = self.pelvis_yaw_pin
        if body_name not in human_data:
            return human_data
        ref_heading = self._mean_heading(human_data, reference_bodies)
        pos, quat = human_data[body_name]
        heading = self._heading_of(quat)
        if ref_heading is None or heading is None:
            return human_data

        if self._pelvis_yaw_rest_offset is None:
            # First frame defines the operator's neutral stance: whatever the
            # pelvis-to-feet heading relation is now is what we hold it at.
            self._pelvis_yaw_rest_offset = (
                (heading - ref_heading + np.pi) % (2 * np.pi) - np.pi
            )

        target = ref_heading + self._pelvis_yaw_rest_offset
        # Shortest signed way round to the target heading, then clamped so a
        # genuine whole-body turn still reaches the root.
        delta = (target - heading + np.pi) % (2 * np.pi) - np.pi
        delta = float(np.clip(delta, -max_corr, max_corr))
        # Rotate about world Z: composing on the left preserves the frame's
        # pitch/roll exactly, unlike rebuilding it from euler angles.
        pinned = R.from_euler("z", delta) * R.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        human_data[body_name] = [pos, pinned.as_quat()[[3, 0, 1, 2]]]
        return human_data

    def _apply_input_pose_corrections(self, human_data):
        for body_name, rotvec, blend in self.input_pose_corrections:
            if body_name not in human_data:
                continue
            weight = 1.0
            if blend is not None:
                from_body, to_body, zero_deg, full_deg = blend
                if from_body not in human_data or to_body not in human_data:
                    continue
                segment = human_data[to_body][0] - human_data[from_body][0]
                length = np.linalg.norm(segment)
                if length < 1e-9:
                    continue
                elevation = np.degrees(
                    np.arcsin(np.clip(segment[2] / length, -1.0, 1.0))
                )
                weight = np.clip(
                    (elevation - zero_deg) / (full_deg - zero_deg), 0.0, 1.0
                )
                if weight <= 0.0:
                    continue
            pos, quat = human_data[body_name]
            # Human quats are wxyz; scipy is xyzw.
            rot = R.from_quat(quat[[1, 2, 3, 0]]) * R.from_rotvec(weight * rotvec)
            human_data[body_name] = [pos, rot.as_quat()[[3, 0, 1, 2]]]
        return human_data

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(
            human_data, self.human_root_name, self.human_scale_table
        )
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        if self.pelvis_yaw_pin is not None:
            human_data = self._apply_pelvis_yaw_pin(human_data)
        if self.input_pose_corrections:
            human_data = self._apply_input_pose_corrections(human_data)
        self.scaled_human_data = human_data

        # Recalibration captures the FULLY processed pose (post scaling and
        # corrections) — the same data the task targets are built from, so the
        # recomputed offsets live in exactly the space they are applied in.
        if self._recal_frames_left > 0:
            self._accumulate_recalibration(human_data)

        if self.use_ik_match_table1 and self._flat_targets1 is not None:
            self._update_table_targets(self._flat_targets1, human_data)
        if self.use_ik_match_table2 and self._flat_targets2 is not None:
            self._update_table_targets(self._flat_targets2, human_data)

    @staticmethod
    def _update_table_targets(flat_targets, human_data):
        tasks, bodies, pos_offsets, rot_offsets = flat_targets
        quats = np.empty((len(bodies), 4))
        positions = np.empty((len(bodies), 3))
        for i, body_name in enumerate(bodies):
            positions[i], quats[i] = human_data[body_name]
        # Human quats are wxyz; reindex to scipy's xyzw instead of paying
        # scalar_first's np.roll per conversion.
        rot = R.from_quat(quats[:, [1, 2, 3, 0]]) * rot_offsets
        target_quats = rot.as_quat()[:, [3, 0, 1, 2]]
        target_pos = positions + rot.apply(pos_offsets)
        for i, task in enumerate(tasks):
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(target_quats[i]), target_pos[i]
                )
            )

    # One runtime recalibration averages this many consecutive processed
    # frames (~0.5–1 s at the 50–100 Hz live retarget rate) — long enough to
    # soak up tracker jitter, short enough that "hold still" is no burden.
    _RECAL_FRAMES = 50

    def request_recalibration(self, on_done=None) -> None:
        """Re-anchor the retargeting to the operator's CURRENT tracked pose.

        PICO trackers/controllers drift over a session (straps slip, the
        tracking origin creeps), which the config's hand-calibrated offsets
        cannot know about. The fix ritual: the operator stands in the robot's
        home pose — ``default_qpos``, the pose the sim robot holds before
        tracking starts — and triggers this. The next ``_RECAL_FRAMES``
        processed frames are averaged, then every IK task's pos/rot offset is
        recomputed so that averaged pose maps EXACTLY onto the home pose's
        forward kinematics: whatever the trackers now report for the known
        stance, the drift is absorbed into the offsets.

        The math replaces rather than adjusts (``off_rot = R_h⁻¹·R_home``,
        ``off_pos = R_home⁻¹·(p_home − p_h)`` — the closed inverse of the
        target build in ``_update_table_targets``), so repeated
        recalibrations never accumulate error. Runtime-only: the config file
        is untouched, a restart returns to the shipped offsets.

        ``on_done(summary)`` fires from the retarget thread once applied,
        with a short human-readable drift summary.
        """
        self._recal_acc = {}
        self._recal_on_done = on_done
        self._recal_frames_left = self._RECAL_FRAMES
        # The pelvis-yaw pin's rest offset (when that tuning exists) is part
        # of the same anchoring — it too was captured on a long-gone first
        # frame. Clearing it makes the pin re-capture on the calibration
        # stance, consistent with the recomputed offsets.
        if getattr(self, "pelvis_yaw_pin", None) is not None:
            self._pelvis_yaw_rest_offset = None

    def _accumulate_recalibration(self, human_data) -> None:
        for body_name, (pos, quat) in human_data.items():
            acc = self._recal_acc.get(body_name)
            if acc is None:
                self._recal_acc[body_name] = [
                    np.array(pos, dtype=float), np.array(quat, dtype=float), 1,
                ]
                continue
            acc[0] += pos
            # Quaternion double cover: accumulate on the running hemisphere.
            acc[1] += quat if np.dot(acc[1], quat) >= 0.0 else -quat
            acc[2] += 1
        self._recal_frames_left -= 1
        if self._recal_frames_left <= 0:
            self._finish_recalibration()

    def _finish_recalibration(self) -> None:
        # Mean processed pose per tracked body. Chordal quat mean — the
        # capture is a still hold, spreads are tiny, so it is exact enough.
        mean_pose = {}
        for body_name, (pos_sum, quat_sum, n) in self._recal_acc.items():
            norm = np.linalg.norm(quat_sum)
            if norm < 1e-9:
                continue
            quat = quat_sum / norm  # wxyz
            mean_pose[body_name] = (
                pos_sum / n, R.from_quat(quat[[1, 2, 3, 0]])
            )
        self._recal_acc = {}

        # Robot body world poses at the model home — what the operator is
        # standing in. A scratch MjData so the live IK state is untouched.
        data = mj.MjData(self.model)
        data.qpos[:] = self._default_qpos
        mj.mj_forward(self.model, data)

        deltas = []   # (frame_name, deg, cm) from table1 — table2 mirrors it
        skipped = set()
        for table_idx, (body_to_tasks, pos_offsets, rot_offsets) in enumerate((
            (self.human_body_to_task1, self.pos_offsets1, self.rot_offsets1),
            (self.human_body_to_task2, self.pos_offsets2, self.rot_offsets2),
        )):
            for body_name, tasks in body_to_tasks.items():
                pose = mean_pose.get(body_name)
                if pose is None:
                    skipped.add(body_name)
                    continue
                p_h, r_h = pose
                for i, task in enumerate(tasks):
                    bid = mj.mj_name2id(
                        self.model, mj.mjtObj.mjOBJ_BODY, task.frame_name
                    )
                    p_home = data.xpos[bid]
                    r_home = R.from_quat(data.xquat[bid][[1, 2, 3, 0]])
                    new_rot = r_h.inv() * r_home
                    new_pos = r_home.inv().apply(p_home - p_h)
                    if table_idx == 0:
                        d_rot = np.degrees(
                            (rot_offsets[body_name][i].inv() * new_rot)
                            .magnitude()
                        )
                        d_pos = np.linalg.norm(
                            new_pos - pos_offsets[body_name][i]
                        )
                        deltas.append((task.frame_name, d_rot, d_pos * 100.0))
                    rot_offsets[body_name][i] = new_rot
                    pos_offsets[body_name][i] = new_pos

        self._flat_targets1 = self._flatten_target_table(
            self.human_body_to_task1, self.pos_offsets1, self.rot_offsets1
        )
        self._flat_targets2 = self._flatten_target_table(
            self.human_body_to_task2, self.pos_offsets2, self.rot_offsets2
        )

        for name, deg, cm in deltas:
            print(f"[GMR] recalibrated {name}: offset moved "
                  f"{deg:.1f} deg / {cm:.1f} cm")
        if skipped:
            print(f"[GMR] recalibration: no tracked data for "
                  f"{sorted(skipped)} — their tasks keep the previous offsets")
        summary = "no tasks updated"
        if deltas:
            worst = max(deltas, key=lambda d: d[1])
            summary = (f"{len(deltas)} tasks re-anchored, largest drift "
                       f"{worst[1]:.1f} deg / {worst[2]:.1f} cm ({worst[0]})")
        print(f"[GMR] runtime recalibration applied: {summary}")
        on_done, self._recal_on_done = self._recal_on_done, None
        if on_done is not None:
            on_done(summary)

    @staticmethod
    def _is_arm_body(body_name):
        return any(
            token in body_name
            for token in (
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
            )
        )

    def _set_first_frame_arm_task_costs(self, enabled):
        for task, original_orientation_cost in (
            self._arm_task_original_orientation_costs.items()
        ):
            if enabled and np.any(original_orientation_cost > 0.0):
                orientation_cost = np.where(
                    original_orientation_cost > 0.0,
                    np.minimum(
                        original_orientation_cost,
                        self._first_frame_arm_orientation_cost,
                    ),
                    0.0,
                )
            else:
                orientation_cost = original_orientation_cost
            task.set_orientation_cost(orientation_cost)

    def _apply_ground_snap(self):
        if not self.ground_snap_geom_ids:
            return
        mj.mj_forward(self.model, self.configuration.data)
        min_z = min(
            geom_min_z(self.model, self.configuration.data, gid)
            for gid in self.ground_snap_geom_ids
        )
        # This is a kinematic retarget, not a dynamics simulation: mj_forward
        # will never let gravity settle a hovering pose onto the floor.  Move
        # the free root in BOTH directions so the lowest foot lands exactly at
        # the configured clearance.  The human input was already normalised by
        # offset_human_data_to_ground(), so this does not discard an absolute
        # jump height that would otherwise need preserving.
        correction = self.ground_snap_clearance - min_z
        if abs(correction) > 1e-6:
            self.configuration.data.qpos[2] += correction
            mj.mj_forward(self.model, self.configuration.data)

    def _apply_qpos_smoothing(self, prev_q):
        if not self.qpos_smoothing_groups:
            return
        for alpha, qpos_indices in self.qpos_smoothing_groups:
            for qpos_idx in qpos_indices:
                current = self.configuration.data.qpos[qpos_idx]
                self.configuration.data.qpos[qpos_idx] = prev_q[qpos_idx] + alpha * (
                    current - prev_q[qpos_idx]
                )
        mj.mj_forward(self.model, self.configuration.data)

    def _solve_task_group(
        self,
        tasks,
        error_fn,
        *,
        damping,
        max_iter,
        include_posture,
        include_prev_posture,
    ):
        solve_tasks = list(tasks)
        if include_posture:
            solve_tasks.append(self.posture_task)
        if include_prev_posture:
            solve_tasks.append(self.prev_posture_task)
        if self.waist_posture_task is not None:
            solve_tasks.append(self.waist_posture_task)

        curr_error = error_fn()
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(
            self.configuration,
            solve_tasks,
            dt,
            self.solver,
            damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(vel, dt)
        next_error = error_fn()
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < max_iter:
            curr_error = next_error
            dt = self.configuration.model.opt.timestep
            vel = mink.solve_ik(
                self.configuration,
                solve_tasks,
                dt,
                self.solver,
                damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel, dt)
            next_error = error_fn()
            num_iter += 1

    def retarget(
        self,
        human_data,
        offset_to_ground=False,
        apply_ground_snap=True,
        apply_qpos_smoothing=True,
    ):
        is_first_frame = self._is_first_frame
        prev_q = self.configuration.data.qpos.copy()
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)
        include_posture = is_first_frame
        include_prev_posture = True
        solve_damping = self.first_frame_damping if is_first_frame else self.damping
        solve_max_iter = (
            self.first_frame_max_iter if is_first_frame else self.max_iter
        )
        self.prev_posture_task.set_target(prev_q)
        if is_first_frame:
            self._set_first_frame_arm_task_costs(True)

        if self.use_ik_match_table1:
            self._solve_task_group(
                self.tasks1,
                self.error1,
                damping=solve_damping,
                max_iter=solve_max_iter,
                include_posture=include_posture,
                include_prev_posture=include_prev_posture,
            )

        if self.use_ik_match_table2:
            self._solve_task_group(
                self.tasks2,
                self.error2,
                damping=solve_damping,
                max_iter=solve_max_iter,
                include_posture=include_posture,
                include_prev_posture=include_prev_posture,
            )

        if is_first_frame:
            self._set_first_frame_arm_task_costs(False)
        self._is_first_frame = False
        if apply_ground_snap:
            self._apply_ground_snap()
        # Skip EMA on the very first frame: prev_q is the home pose, so EMA would
        # pull the initial pose toward T-pose for one rendered frame.
        if apply_qpos_smoothing and not is_first_frame:
            self._apply_qpos_smoothing(prev_q)
        return self.configuration.data.qpos.copy()

    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )

    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )

    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [
                np.asarray(human_data[body_name][0]),
                np.asarray(human_data[body_name][1]),
            ]
        return human_data

    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (
                    human_data[body_name][0] - root_pos
                ) * human_scale_table[body_name]

        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (
                human_data_local[body_name] + scaled_root_pos,
                human_data[body_name][1],
            )

        return human_data_global

    def offset_human_data_to_ground(self, human_data):
        """Find the lowest point of the human data and offset it to the ground."""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name  # noqa: F841
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = (
                pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
            )
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name] = [pos - np.array([0, 0, self.ground_offset]), quat]
        return human_data
