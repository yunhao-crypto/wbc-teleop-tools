"""Adapter to the pinned motion-stream-bridge Vita02A GMR implementation.

The vendored solver, Unity conversion, XML and IK tuning are unchanged. This
adapter only validates raw frames, selects joints by name and anchors world
yaw at following entry. It does not calibrate an arbitrary first pose to home.

For speed it runs the pinned engine through ``_accelerated_engine``, which
keeps the solver arithmetic and only drops work whose results GMR never
reads. Output is bit-identical to the pinned engine (tested on the Pico
replay and the bridge golden fixture). Its dependencies are required: a
mismatch raises instead of silently running the slower pinned engine;
``accelerated=False`` selects the pinned engine explicitly.
"""
from ctypes import c_int
import importlib
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# The accelerated engine reuses mink/daqp internals of exactly these releases.
_ACCELERATED_VERSIONS = {'mink': '1.3.0', 'daqp': '0.10.3', 'qpsolvers': '4.13.0'}


def _installed_versions():
    found = {}
    for name in _ACCELERATED_VERSIONS:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = 'missing'
    return found


def _versions(table):
    return ', '.join(f'{k} {v}' for k, v in table.items())


def _accelerated_engine(module):
    """Subclass of the pinned engine, or None when its dependencies differ.

    Changes, each leaving every qpos bit unchanged:
    * ground snap / qpos smoothing call mj_kinematics + mj_comPos instead of
      mj_forward. GMR only reads poses and Jacobians afterwards; mj_forward's
      collision and dynamics stages were ~1.6 ms of every frame.
    * daqp is called directly with the arrays qpsolvers would pass it.
    * body FrameTask residuals are assembled in one loop over mink's native
      Lie ops, with the same operations mink performs per task.
    """
    try:
        if any(importlib.metadata.version(k) != v for k, v in _ACCELERATED_VERSIONS.items()):
            return None
        import daqp
        import mujoco as mj
        from mink import constants
        from mink.exceptions import NoSolutionFound, TargetNotSet
        from mink.lie import _lie_ops_c as lie
        from mink.tasks import frame_task
        from mink import configuration as mink_configuration
        ik = importlib.import_module('mink.solve_ik')
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return None
    if (frame_task._native is None or mink_configuration._native is None
            or constants.FRAME_TO_JAC_FUNC['body'] is not mj.mj_jacBody):
        return None
    base = module.GeneralMotionRetargeting

    def kinematics(model, data):
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)

    class AcceleratedGeneralMotionRetargeting(base):
        def setup_retarget_configuration(self):
            super().setup_retarget_configuration()
            self._body_ids = {
                task: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, task.frame_name)
                for task in self.tasks1+self.tasks2
                if type(task) is frame_task.FrameTask and task.frame_type == 'body'}

        def _apply_ground_snap(self):
            if not self.ground_snap_geom_ids:
                return
            data = self.configuration.data
            kinematics(self.model, data)
            min_z = min(module.geom_min_z(self.model, data, gid) for gid in self.ground_snap_geom_ids)
            correction = self.ground_snap_clearance - min_z
            if abs(correction) > 1e-6:
                data.qpos[2] += correction
                kinematics(self.model, data)

        def _apply_qpos_smoothing(self, prev_q):
            if not self.qpos_smoothing_groups:
                return
            qpos = self.configuration.data.qpos
            for alpha, qpos_indices in self.qpos_smoothing_groups:
                for qpos_idx in qpos_indices:
                    qpos[qpos_idx] = prev_q[qpos_idx]+alpha*(qpos[qpos_idx]-prev_q[qpos_idx])
            kinematics(self.model, self.configuration.data)

        def _objective(self, tasks, damping):
            # mink.solve_ik._compute_qp_objective with FrameTask inlined.
            model, data = self.model, self.configuration.data
            nv = model.nv
            jacobians, errors, mu_total = [], [], 0.
            H_dense = c_dense = None
            for task in tasks:
                bid = self._body_ids.get(task)
                if bid is None:
                    residual = task.compute_qp_residual(self.configuration)
                    if residual is None:
                        H_task, c_task = task.compute_qp_objective(self.configuration)
                        H_dense = H_task if H_dense is None else H_dense+H_task
                        c_dense = c_task if c_dense is None else c_dense+c_task
                        continue
                else:
                    if task.transform_target_to_world is None:
                        raise TargetNotSet(type(task).__name__)
                    xmat = data.xmat[bid]
                    frame = lie.xmat_xpos_to_wxyz_xyz(xmat, data.xpos[bid])
                    jac = np.empty((6, nv))
                    mj.mj_jacBody(model, data, jac[:3], jac[3:], bid)
                    jac = lie.se3_rotation_adjoint_from_xmat(xmat) @ jac
                    target = task.transform_target_to_world.wxyz_xyz
                    error = lie.se3_rminus(target, frame)
                    jacobian = -lie.se3_jlog(lie.se3_inverse_multiply(target, frame)) @ jac
                    residual = task._weighted_residual(error, jacobian)
                jacobian, error, mu = residual
                jacobians.append(jacobian)
                errors.append(error)
                mu_total += mu
            if jacobians:
                W = np.vstack(jacobians)
                H = W.T @ W
                c = -(np.concatenate(errors) @ W)
            else:
                H, c = np.zeros((nv, nv)), np.zeros(nv)
            H.flat[::nv+1] += damping+mu_total
            if H_dense is not None:
                H += H_dense
                c += c_dense
            return H, c

        def _solve_ik(self, tasks, damping):
            configuration = self.configuration
            dt = configuration.model.opt.timestep
            if self.solver != 'daqp':
                return module.mink.solve_ik(configuration, tasks, dt, self.solver, damping,
                                            limits=self.ik_limits)
            configuration.check_limits(safety_break=False)
            H, c = self._objective(tasks, damping)
            G, h = ik._compute_qp_inequalities(configuration, self.ik_limits, dt)
            if G is None:
                G, h = np.zeros((0, c.size)), np.zeros(0)
            x, _, exitflag, _ = daqp.solve(H, c, G, h, np.full(h.size, -1e30),
                                           np.zeros(h.size, dtype=c_int), primal_start=None)
            if exitflag <= 0:
                raise NoSolutionFound(self.solver)
            return x/dt

        def _solve_task_group(self, tasks, error_fn, *, damping, max_iter,
                              include_posture, include_prev_posture):
            solve_tasks = list(tasks)
            if include_posture:
                solve_tasks.append(self.posture_task)
            if include_prev_posture:
                solve_tasks.append(self.prev_posture_task)
            if self.waist_posture_task is not None:
                solve_tasks.append(self.waist_posture_task)
            curr_error = error_fn()
            dt = self.configuration.model.opt.timestep
            self.configuration.integrate_inplace(self._solve_ik(solve_tasks, damping), dt)
            next_error = error_fn()
            num_iter = 0
            while curr_error-next_error > 0.001 and num_iter < max_iter:
                curr_error = next_error
                self.configuration.integrate_inplace(self._solve_ik(solve_tasks, damping), dt)
                next_error = error_fn()
                num_iter += 1

    return AcceleratedGeneralMotionRetargeting


class PicoGmrRetargeter:
    def __init__(self, root, contract, math_api, settings, *, accelerated=True):
        self.math = math_api
        self.contract = contract
        self.settings = settings
        vendor = Path(root)/'third_party/motion_stream_bridge'
        sys.path.insert(0, str(vendor))
        module = importlib.import_module('motion_stream_bridge.gmr_lite')
        if Path(module.__file__).resolve() != vendor.resolve()/'motion_stream_bridge/gmr_lite/__init__.py':
            raise RuntimeError('Expected the pinned motion-stream-bridge GMR package')
        self.streamer = module.XRobotStreamer
        self.accelerated = bool(accelerated)
        if self.accelerated:
            engine = _accelerated_engine(importlib.import_module('motion_stream_bridge.gmr_lite.retarget'))
            if engine is None:
                raise RuntimeError('Accelerated GMR requires '+_versions(_ACCELERATED_VERSIONS)+
                    '; installed '+_versions(_installed_versions())+
                    '. Reinstall the pinned wheelhouse; the pinned engine is not substituted.')
        else:
            engine = module.GeneralMotionRetargeting
        self.engine = engine(
            src_human='xrobot', tgt_robot='vita02a',
            config=str(vendor/'motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'),
            actual_human_height=float(settings['human_height_m']), verbose=False)
        import mujoco
        self.indices = []
        for name in contract.names:
            joint = mujoco.mj_name2id(self.engine.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or self.engine.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError('GMR does not provide actor hinge joint: '+name)
            self.indices.append(int(self.engine.model.jnt_qposadr[joint]))
        # The bridge wire format permutes wrist yaw/pitch. Never use it as the
        # model action order: the qpos addresses above are resolved by name.
        e = self.engine
        # Identical tables yield identical residuals; evaluate one of them.
        self.same_tables = (e.use_ik_match_table1 and e.use_ik_match_table2
                            and e.ik_match_table1 == e.ik_match_table2)
        self.unity_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.unity_quat = Rotation.from_matrix(self.unity_matrix).as_quat(scalar_first=True)
        self.last_qpos = None
        self.last_error = None

    def processed_body(self, body):
        body = np.asarray(body, dtype=np.float64)
        if (body.shape != (24, 7) or not np.isfinite(body).all()
                or np.any(np.linalg.norm(body[:, 3:7], axis=1) < 1e-3)):
            raise ValueError('Expected finite Pico [24,7] with valid xyzw quaternions')
        if not self.accelerated:
            poses = {name: [row[:3].tolist(), row[[6, 3, 4, 5]].tolist()]
                     for name, row in zip(self.streamer.body_joint_names, body)}
            return self.streamer.coordinate_transform_unity_data(poses)
        # XRobotStreamer.coordinate_transform_unity_data for all 24 joints at
        # once, with its exact per-element arithmetic.
        x0, x1, x2, x3 = self.unity_quat
        y0, y1, y2, y3 = body[:, 6], body[:, 3], body[:, 4], body[:, 5]
        quat = np.stack((x0*y0-x1*y1-x2*y2-x3*y3, x0*y1+x1*y0+x2*y3-x3*y2,
                         x0*y2-x1*y3+x2*y0+x3*y1, x0*y3+x1*y2-x2*y1+x3*y0), axis=1)
        # Row-wise like the original: a batched product may sum signed zeros differently.
        rotation = self.unity_matrix.T
        return {name: [(p @ rotation).tolist(), q] for name, p, q in
                zip(self.streamer.body_joint_names, body[:, :3], quat.tolist())}

    def solve(self, body, *, robot_root=None):
        # robot_root is deliberately unused here. The GMR root is in its own
        # world frame; following entry aligns yaw once against current feedback.
        qpos = self.engine.retarget(self.processed_body(body), offset_to_ground=True)
        if qpos.shape != (self.engine.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError('Nonfinite/invalid GMR qpos')
        self.last_qpos = qpos.copy()
        error = float(self.engine.error1())
        self.last_error = error if self.same_tables else max(error, float(self.engine.error2()))
        # This is a mixed position/orientation weighted task norm, NOT radians.
        return qpos[self.indices].copy(), self.math.quaternion(qpos[3:7]), self.last_error


def create_retargeter(root, config, contract, math_api, *, accelerated=True):
    from .actor_pico_stream import worker_settings
    settings = config['retargeting']
    if settings['implementation'] != 'motion_stream_bridge_gmr_vita02a':
        raise ValueError('Only the pinned Vita02A bridge GMR retargeter is supported')
    height = float(settings['human_height_m'])
    if not np.isfinite(height) or not .8 <= height <= 2.5:
        raise ValueError('Invalid GMR operator height in metres')
    retargeter = PicoGmrRetargeter(root, contract, math_api, settings, accelerated=accelerated)
    iterations = worker_settings(config).ik_max_iter
    if iterations is not None:
        # Extra refinement passes per frame (pinned default 10) are where the
        # IK occasionally walks to another arm branch; the warm-started solve
        # converges across frames instead. The first frame keeps its own cap.
        retargeter.engine.max_iter = iterations
    return retargeter
