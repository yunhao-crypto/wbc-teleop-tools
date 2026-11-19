"""Send bounded Pico head and grip intent through the owned native action."""
import threading
import time

from .actor_pico_gripper import SIDES, SideGripGate, load_config

EVENT = 'humanoid_tracking.pico_accessories'


def load_native_gripper_config(path):
    config = load_config(path)
    if config['control_mode'] != 'POSITION_TORQUE':
        raise ValueError('Native OP3 requires POSITION_TORQUE')
    return config


def require_accessories(capability, *, head=False, gripper=None):
    if capability.get('native_pico_accessories_v1') is not True:
        raise RuntimeError('Update the native locomotion and tracking client for Pico accessories')
    if head and capability.get('native_neck_follow_reference') is not True:
        raise RuntimeError('Update the MimicLite native profile for head following')
    if gripper is not None:
        configured = sum(1 << SIDES.index(side) for side in gripper['sides'])
        available = capability.get('native_pico_op3_sides')
        if type(available) is not int or configured & ~available:
            raise RuntimeError('Configured OP3 sides are absent from native RobotInfo')


def require_grip_binding():
    import xrobotoolkit_sdk
    if not callable(getattr(xrobotoolkit_sdk, 'get_controller_grip_snapshot', None)):
        raise RuntimeError('Update the Pico binding for fresh, atomic side-grip input')


class NativePicoAccessories:
    """Latest-only input and bounded RPC off the body reference producer."""

    def __init__(self, send, controls, *, head=None, gripper=None, clock=time.monotonic):
        self.send, self.controls, self.head = send, controls, head
        self.gripper, self.clock = gripper, clock
        self.snapshot = (0, False, None)
        self.epoch = -1
        self.gates = {}
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.period = max(.05, 1/gripper['rate_hz']) if gripper and head is None else .05

    def update(self, following, head_snapshot=None):
        epoch, previous, _ = self.snapshot
        self.snapshot = (epoch + (following != previous), following, head_snapshot)

    def tick(self):
        epoch, following, head_snapshot = self.snapshot
        if self.epoch != epoch:
            self.epoch = epoch
            if self.head is not None:
                self.head.reset()
            if self.gripper is not None:
                self.gates = {side: SideGripGate(self.gripper) for side in self.gripper['sides']}
        if not following:
            return
        if self.gripper is not None:
            with self.controls.get_lock():
                received, stamp, left, right = self.controls[3:7]
        now = self.clock()
        payload = dict(capture_ns=int(now * 1e9))
        head_pending = None
        if self.head is not None and head_snapshot is not None:
            try:
                if 0 <= now - head_snapshot['received_monotonic'] <= self.head.control.hold_age:
                    angles, head_pending = self.head.preview_targets(head_snapshot, now)
                    payload.update(neck_yaw_rad=angles['neck_yaw_joint'],
                                   neck_pitch_rad=angles['neck_pitch_joint'])
            except (KeyError, TypeError, ValueError):
                pass
        for side, gate in self.gates.items():
            closed = gate.update(received, stamp, left if side == 'left' else right, now)
            if gate.valid and gate.armed:
                spec = self.gripper['sides'][side]
                payload.update({f'{side}_closed': closed, f'{side}_close_m': spec['close_m'],
                                f'{side}_rate_m_s': spec['vel_limit_m_s'],
                                f'{side}_current_a': spec['current_a']})
        self.send(EVENT, payload)
        if head_pending is not None:
            self.head.commit(head_pending)

    def _run(self):
        try:
            while not self.stop.is_set():
                self.tick()
                self.stop.wait(self.period)
        except Exception as error:
            self.error = error

    def start(self):
        self.thread = threading.Thread(target=self._run, name='native-pico-accessories', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Native Pico accessory publication failed') from self.error

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
