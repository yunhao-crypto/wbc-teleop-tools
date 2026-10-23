"""Full onboard software flow with an inert output sink only.

No actuator transport, lease, mode-switch service or command publisher exists
in this module. START/STAND here are simulated FSM states, not robot commands.
"""
from __future__ import annotations

import time
import numpy as np

from deployment.common.controller_input import VelocityLimits
from deployment.common.mode_fsm import TeleopMode
from .teleop_engine import VitaTeleopEngine


def yaw_alignment_wxyz(robot, reference):
    def yaw(q):
        w, x, y, z = np.asarray(q, np.float64).reshape(4)
        return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    delta = yaw(robot) - yaw(reference)
    return np.asarray([np.cos(delta/2), 0, 0, np.sin(delta/2)], np.float32)


class OnboardShadowFlow:
    def __init__(self, runtime, *, maximum_reference_age=.6, limits=None):
        self.runtime = runtime
        self.reference = runtime.create_online_reference()
        self.engine = VitaTeleopEngine(limits or VelocityLimits())
        self.maximum_reference_age = float(maximum_reference_age)
        self.yaw_alignment = None

    def tick(self, *, state, controller, pico=None, wall_time=None, robot_ready=True):
        """Return an inert PolicyStep plus timings; never send it anywhere.

        pico: optional (body[24,7], sample_time_seconds, frame_index).
        wall_time is injectable only to replay recorded freshness deterministically.
        """
        started = time.monotonic_ns()
        now = time.time() if wall_time is None else float(wall_time)
        stored = False
        if pico is not None:
            body, timestamp, frame = pico[:3]
            received_wall = float(pico[3]) if len(pico)>3 else now
            stored = self.reference.store_pico(body, sample_time=float(timestamp),
                                               current_time=received_wall, frame_index=int(frame))
            if not stored:
                raise ValueError("Invalid or non-monotonic Pico frame")
        preprocessed = time.monotonic_ns()
        ready = bool(self.reference.has_future_sequence(10, include_derivative_tail=True)
                     and self.reference.data_age(now) <= self.maximum_reference_age)
        output = self.engine.update(controller, robot_ready=bool(robot_ready),
                                    motion_reference_ready=ready)
        transition = output.transition
        if transition is not None and transition.changed:
            if transition.current in {TeleopMode.VELOCITY, TeleopMode.MOTION}:
                if transition.previous is not TeleopMode.STAND:
                    self.runtime.reset()
                if transition.current is TeleopMode.MOTION:
                    self.reference.reset_root_anchor()
                    root = self.reference.current_root_rot()
                    if root is None:
                        raise RuntimeError("Motion entry requires reference root")
                    self.yaw_alignment = yaw_alignment_wxyz(state.root_quat_wxyz, root)
                else:
                    self.yaw_alignment = None
            elif transition.current in {TeleopMode.IDLE, TeleopMode.ESTOP}:
                self.runtime.reset()
                self.yaw_alignment = None
        switched = time.monotonic_ns()
        step = None
        if output.mode in {TeleopMode.STAND, TeleopMode.VELOCITY}:
            step = self.runtime.step_velocity(state, output.velocity_command)
        elif output.mode is TeleopMode.MOTION:
            step = self.runtime.step_motion(state, self.reference,
                    reference_yaw_alignment_wxyz=self.yaw_alignment)
        completed = time.monotonic_ns()
        timing = {
            "pico_preprocess_ms": (preprocessed-started)/1e6,
            "fsm_switch_ms": (switched-preprocessed)/1e6,
            "policy_ms": (completed-switched)/1e6,
            "tick_ms": (completed-started)/1e6,
        }
        if step is not None:
            timing.update(self.runtime.last_timing_ms)
        return step, {"mode": output.mode.value, "reference_ready": ready,
                      "pico_stored": bool(stored), "transition": transition,
                      "timing_ms": timing, "action_sent": False}


class ReadOnlyStateSource:
    """Keep the callback minimal; decode the latest payload at the policy tick."""
    def __init__(self):
        import aorta
        import humanoid_low_state_schema_meta as schema
        if schema.SCHEMA_HASH_HEX != 'de514c40e02f4cd730a3154970e57f48f8f6ecbeaf0af9ed3f19b2df79bf3765':
            raise RuntimeError("Unexpected LowState schema")
        def forbidden(*args, **kwargs):
            raise RuntimeError("Shadow process forbids publishing/control handles")
        for method in dir(aorta.Node):
            if method.startswith('create_') and any(x in method for x in ('publisher','client','action','service')):
                setattr(aorta.Node, method, forbidden)
        self.latest = None
        self.received = 0
        self.node = aorta.Node('vita-onboard-shadow-'+str(time.monotonic_ns()), 'default')
        try:
            self.subscriber = self.node.create_subscriber('/sdk/foot_humanoid/lowstate',
                    self._callback, qos=aorta.QoS.sensor_data(),
                    options=aorta.SubscriberOptions(include_receive_timestamp=True,
                                                     enable_header_context=True))
        except BaseException:
            try:
                self.node.close()
            except Exception:
                pass  # Preserve the subscription failure.
            raise

    def _callback(self, payload, context):
        self.latest = (bytes(payload), time.monotonic_ns())
        self.received += 1

    def close(self):
        try:
            self.subscriber.close()
        finally:
            self.node.close()
