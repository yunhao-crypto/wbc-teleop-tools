"""Continuous Pico inputs and reference state machine; no robot command transport.

Both routes infer at 50Hz. STAND means a standing-reference inference, never a
firmware FIXED_STAND request. Live actuation remains outside this module.
"""
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .actor_pico_buttons import DualTriggerToggle
from . import pico_threepoint_frontend as front

ROUTES = ('pico_smpl', 'pico_teleop')
PERIOD_NS = 20_000_000
MARGIN_NS = 100_000_000
HORIZON_NS = 180_000_000
BLEND_NS = 1_000_000_000


def next_deadline_ns(previous_ns, now_ns):
    """Next grid tick strictly AFTER now, including exact-deadline overruns."""
    skipped = max(1, (int(now_ns)-int(previous_ns))//PERIOD_NS+1)
    return int(previous_ns)+skipped*PERIOD_NS


@dataclass
class Feedback:
    q: np.ndarray
    dq: np.ndarray
    root: np.ndarray
    gyro: np.ndarray

    def validate(self):
        for name, shape in (('q', (28,)), ('dq', (28,)), ('root', (4,)), ('gyro', (3,))):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Invalid feedback '+name)
        front.qmatrix(self.root)


class SmplWindow:
    """Measured Pico positions, canonical root, explicit standing robot wrists.

    This uses the supplied XYZ joints, not pose-only FK or a replacement body
    model. The 180ms future horizon is entirely received before it is queried.
    """
    def __init__(self, math_api, wrists):
        self.math = math_api
        self.wrists = np.asarray(wrists, np.float32)
        if self.wrists.shape != (6,) or not np.isfinite(self.wrists).all():
            raise ValueError('Six explicit standing wrist references are required')
        self.samples = deque(maxlen=512)
        self.offset_ns = None
        self.yaw_rotation = np.eye(3)
        self.yaw_quat = np.array([1., 0., 0., 0.])

    def frame(self, body):
        from .onboard_preprocessing import CPUPicoToSmplConverter
        from .actor_smpl import root_from_pose_aa
        front.anatomical_sites(body)
        pose, _ = CPUPicoToSmplConverter().to_device(body)
        root = root_from_pose_aa(self.math, pose, convention='z_up_smpl_base')[0]
        joints = (np.asarray(body[:, :3], dtype=float)-body[0, :3])@front.AXES.T
        return joints@self.yaw_rotation.T, self.math.multiply(self.yaw_quat, root)

    def align(self, body, robot_root):
        _, root = self.frame(body)
        yaw = self.math.yaw(robot_root)-self.math.yaw(root)
        self.yaw_quat = self.math.yaw_quaternion(yaw)
        self.yaw_rotation = self.math.rotation(self.yaw_quat)

    def append(self, body, stamp_ns, received_ns):
        joints, root = self.frame(body)
        if self.samples and stamp_ns <= self.samples[-1][0]:
            raise ValueError('SMPL stream timestamp did not advance')
        if self.offset_ns is None: self.offset_ns = int(received_ns)-int(stamp_ns)
        self.samples.append((int(stamp_ns), joints, root))

    def clear(self):
        self.samples.clear(); self.offset_ns = None

    def reference(self, now_ns):
        from .actor_smpl import SmplReference
        if len(self.samples) < 2: return None
        end = int(now_ns)-self.offset_ns-MARGIN_NS
        query = end-HORIZON_NS+np.arange(10, dtype=np.int64)*PERIOD_NS
        stamps = np.array([x[0] for x in self.samples], np.int64)
        if query[0] < stamps[0] or query[-1] > stamps[-1]: return None
        joints = []; roots = []
        for stamp in query:
            i = min(max(int(np.searchsorted(stamps, stamp, side='right'))-1, 0), len(stamps)-2)
            alpha = float((int(stamp)-int(stamps[i]))/(int(stamps[i+1])-int(stamps[i])))
            joints.append((1-alpha)*self.samples[i][1]+alpha*self.samples[i+1][1])
            roots.append(front.slerp(self.samples[i][2], self.samples[i+1][2], alpha))
        return SmplReference(self.math, joints, roots, np.repeat(self.wrists[None], 10, axis=0))


class PicoFlow:
    """Identical continuous state machine in replay and device-only shadow.

    Actions returned by infer are inert candidates. There is deliberately no
    publisher/controller argument or live-mode escape hatch.
    """
    def __init__(self, root, route, *, backend='auto', threads=1, wrist_mode=None, runtime=None):
        if route not in ROUTES: raise ValueError('Unknown Pico route')
        if route == 'pico_smpl' and wrist_mode != 'stand':
            raise ValueError('pico_smpl requires explicit wrist_mode=stand; no implicit wrist fallback')
        self.root = Path(root).resolve(); self.route = route
        settings = json.loads((self.root/'config/pico_inputs.json').read_text())
        if (settings.get('schema') != 'vita_pico_inputs_shadow_v1' or settings.get('routes') != list(ROUTES)
                or settings.get('motion_enabled') is not False or settings.get('neutral_seconds') != 2.
                or settings.get('reference_freshness_ms') != 100 or settings.get('maximum_frame_gap_ms') != 100
                or settings.get('smpl_wrist_policy') != 'explicit_contract_stand'
                or set(settings.get('assets', {})) != {'smpl_config', 'teleop_config', 'standing_reference', 'site_interface'}):
            raise ValueError('Unsupported Pico input settings; motion/timing protection cannot be overridden')
        for item in settings['assets'].values():
            path = (self.root/item['path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or front.sha(path) != item['sha256']:
                raise ValueError('Pico input asset hash mismatch: '+item['path'])
        if runtime is None:
            if route == 'pico_smpl':
                from .actor_smpl import SmplActorRuntime
                runtime = SmplActorRuntime(self.root, backend=backend, threads=threads)
            else:
                from .actor_threepoint import ThreePointActorRuntime
                runtime = ThreePointActorRuntime(self.root, backend=backend, threads=threads)
        self.runtime = runtime; self.math = runtime.math
        self.contract = runtime.model.contract if route == 'pico_smpl' else runtime.contract
        self.gate = front.TimingGate(); self.toggle = DualTriggerToggle(.6)
        self.raw = deque(maxlen=512); self.window = None; self.mapper = None
        self.standing = None; self.latest = None; self.last_reference = None
        self.calibration = None; self.mode = 'calibrating'
        self.events = []; self.frames = 0; self.inferences = 0; self.losses = 0
        self.track_ticks = 0; self.stand_ticks = 0; self.transitions = 0
        self.source_fault = None; self.last_frame_ns = None; self.button_not_before_ns = 0
        self.transition_start = None; self.transition_from = None
        self.last_step_ns = None; self.started_following = 0
        self.standing_path = self.root/'artifacts/actor_threepoint/standing_reference.npz'
        self.sites_path = self.root/'artifacts/actor_threepoint/teleop_site_interface.json'
        if route == 'pico_teleop':
            with np.load(self.standing_path, allow_pickle=False) as z:
                self.standing = (z['positions_base_m'].copy(), z['quaternions_base_wxyz'].copy())

    def event(self, kind, now_ns, **details):
        self.events.append(dict(record_type=kind, time_ns=int(now_ns), route=self.route,
            mode=self.mode, actions_sent=0, control_handles_created=0, **details))

    def lose(self, now_ns, reason):
        if self.source_fault is not None: return
        self.source_fault = str(reason); self.losses += 1
        self.toggle = DualTriggerToggle(.6); self.button_not_before_ns = int(now_ns)
        self.raw.clear(); self.gate = front.TimingGate()
        if self.window is not None: self.window.clear()
        self.latest = None; self.last_frame_ns = None
        if self.mode == 'tracking': self.return_to_stand(now_ns)
        self.event('pico_loss_stand', now_ns, reason=str(reason), automatic_reengagement=False)

    def push(self, body, stamp_ns, received_ns, feedback):
        feedback.validate()
        try:
            advanced = self.gate.accept(stamp_ns, body, received_ns)
        except ValueError as exc:
            self.lose(received_ns, str(exc)); return False
        if not advanced: return False
        self.frames += 1; self.last_frame_ns = int(received_ns)
        if self.source_fault is not None:
            self.source_fault = None; self.button_not_before_ns = int(received_ns)
            self.toggle = DualTriggerToggle(.6)
            self.event('pico_recovered_wait_trigger', received_ns, automatic_reengagement=False)
        self.raw.append((np.array(body, copy=True), int(stamp_ns), int(received_ns)))
        if self.calibration is None:
            duration = (self.raw[-1][1]-self.raw[0][1])*1e-9
            if duration < 2.: return True
            bodies = np.asarray([x[0] for x in self.raw])
            times = (np.array([x[1] for x in self.raw], np.int64)-self.raw[0][1])*1e-9
            try:
                calibration = front.fit_calibration(bodies, times, self.standing_path, self.sites_path)
            except ValueError as exc:
                self.lose(received_ns, 'Neutral calibration rejected: '+str(exc)); return False
            self.calibration = calibration
            if self.route == 'pico_teleop':
                self.mapper = front.CalibratedPicoThreePoint(calibration)
            else:
                from .actor_smpl import WRIST_NAMES, SmplReference
                wrists = np.array([self.contract.default[self.contract.names.index(n)] for n in WRIST_NAMES])
                self.window = SmplWindow(self.math, wrists)
                self.window.align(bodies[0], feedback.root)
                frames = [self.window.frame(x) for x in bodies[times <= 2.]]
                neutral_joints = np.median(np.array([x[0] for x in frames]), axis=0)
                average = sum(front.qmatrix(x[1]) for x in frames)
                u, _, v = np.linalg.svd(average)
                if np.linalg.det(u@v) < 0: u[:, -1] *= -1
                neutral_q = front.matrixq(u@v)
                self.standing = SmplReference(self.math, np.repeat(neutral_joints[None], 10, axis=0),
                    np.repeat(np.asarray(neutral_q)[None], 10, axis=0), np.repeat(wrists[None], 10, axis=0))
                for b, t, r in self.raw: self.window.append(b, t, r)
            self.mode = 'stand'
            self.event('neutral_calibrated', received_ns, standing_reference='Pico neutral human pose' if self.route == 'pico_smpl' else 'pinned robot site poses',
                wrist_mode='explicit_contract_stand' if self.route == 'pico_smpl' else 'site_orientation')
        elif self.route == 'pico_smpl':
            self.window.append(body, int(stamp_ns), int(received_ns))
        if self.route == 'pico_teleop': self.latest = self.mapper.map(body)
        return True

    def reference_ready(self, now_ns):
        if self.source_fault is not None or self.last_frame_ns is None or not 0 <= int(now_ns)-self.last_frame_ns <= 100_000_000:
            return False
        if self.calibration is None: return False
        return self.latest is not None if self.route == 'pico_teleop' else self.window.reference(now_ns) is not None

    def button(self, sample, now_ns):
        available, stamp, age, left, right = sample
        if (not np.isfinite(age) or int(now_ns)-int(round(age*1e9)) < self.button_not_before_ns
                or not self.reference_ready(now_ns) or self.mode == 'returning_stand'):
            self.toggle.armed = False; return False
        if not self.toggle.update(sample, int(now_ns)*1e-9): return False
        if self.mode == 'stand':
            self.mode = 'tracking'; self.started_following += 1
            self.transition_from = self.last_reference or self.standing
            self.transition_start = int(now_ns); self.transitions += 1
            self.event('tracking_started', now_ns)
        elif self.mode == 'tracking': self.return_to_stand(now_ns)
        return True

    def return_to_stand(self, now_ns):
        self.mode = 'returning_stand'; self.transition_from = self.last_reference
        self.transition_start = int(now_ns); self.transitions += 1
        self.event('tracking_paused', now_ns, hardware_stand_requested=False)

    def blend(self, start, target, weight):
        if self.route == 'pico_teleop':
            return (1-weight)*start[0]+weight*target[0], front.slerp(start[1], target[1], weight)
        from .actor_smpl import SmplReference
        return SmplReference(self.math, (1-weight)*start.joints+weight*target.joints,
            front.slerp(start.quat, target.quat, weight), target.wrists)

    def step(self, now_ns, feedback):
        feedback.validate(); now_ns = int(now_ns)
        if self.mode in ('failed', 'stopped'): raise RuntimeError('Pico flow is '+self.mode)
        if self.last_step_ns is not None and now_ns <= self.last_step_ns:
            raise ValueError(f'Policy control clock must strictly advance: now={now_ns}, previous={self.last_step_ns}')
        self.last_step_ns = now_ns
        if self.last_frame_ns is not None and now_ns-self.last_frame_ns > 100_000_000:
            self.lose(now_ns, 'No fresh Pico reference within100ms')
        if self.mode == 'tracking' and not self.reference_ready(now_ns):
            self.lose(now_ns, 'Complete causal reference unavailable')
        if self.standing is None:
            return dict(mode=self.mode, inferred=False, actions_sent=0, control_handles_created=0)
        desired = self.standing
        if self.mode == 'tracking':
            desired = self.latest if self.route == 'pico_teleop' else self.window.reference(now_ns)
        if self.transition_start is not None:
            fraction = min(max((now_ns-self.transition_start)/BLEND_NS, 0.), 1.)
            a = fraction*fraction*(3-2*fraction)
            desired = self.blend(self.transition_from or self.standing, desired, a)
            if fraction >= 1.:
                self.transition_start = self.transition_from = None
                if self.mode == 'returning_stand':
                    self.mode = 'stand'; self.event('stand_reference_reached', now_ns)
        self.last_reference = desired
        try:
            if self.route == 'pico_smpl':
                command, obs, report = self.runtime.infer(desired, 0., q=feedback.q, dq=feedback.dq, root=feedback.root, gyro=feedback.gyro)
            else:
                from .actor_threepoint import ThreePointTarget
                target = ThreePointTarget(*desired, now_ns*1e-9)
                command, obs, report = self.runtime.infer(target, q=feedback.q, dq=feedback.dq, root=feedback.root, gyro=feedback.gyro)
        except Exception as exc:
            self.mode = 'failed'; self.event('inference_failed', now_ns, error=str(exc)); raise
        self.inferences += 1
        if self.mode == 'tracking': self.track_ticks += 1
        else: self.stand_ticks += 1
        return dict(mode=self.mode, inferred=True, command=command, observation=obs, report=report,
            actions_sent=0, control_handles_created=0)

    def stop(self, now_ns):
        self.mode = 'stopped'; self.event('stopped', now_ns, hardware_exit_requested=False)

    def summary(self):
        return dict(route=self.route, mode=self.mode, pico_frames=self.frames, inference_steps=self.inferences,
            tracking_steps=self.track_ticks, stand_steps=self.stand_ticks, tracking_entries=self.started_following,
            pico_losses=self.losses, transitions=self.transitions,
            causal_horizon_ms=180 if self.route == 'pico_smpl' else 0,
            playback_margin_ms=100 if self.route == 'pico_smpl' else 0,
            standing_mode='neutral_human_reference' if self.route == 'pico_smpl' else 'pinned_robot_threepoint_reference',
            wrist_mode='explicit_contract_stand' if self.route == 'pico_smpl' else 'site_orientation',
            actions_sent=0, control_handles_created=0, robot_configuration_modified=False,
            protection_relaxed=False, hardware_startup_exit_validated=False)
