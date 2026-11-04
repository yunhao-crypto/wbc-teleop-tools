"""Opt-in dual-grip BrainCo control, isolated from the body policy clock."""
import json
import math
import threading
import time


def load_config(path):
    with open(path) as stream:
        config = json.load(stream)
    for side in ('left', 'right'):
        hand = config[side]
        names = hand['joint_names']
        if len(names) != 6 or len(set(names)) != 6 or not all(isinstance(n, str) and n for n in names):
            raise ValueError('Each BrainCo hand requires six explicit joint names')
        for key in ('open_rad', 'grasp_rad', 'speed_rad_s'):
            values = hand[key]
            if len(values) != 6 or not all(type(x) in (int, float) and math.isfinite(x) for x in values):
                raise ValueError('Hand poses/speeds require six finite numbers')
        if not all(x > 0 for x in hand['speed_rad_s']):
            raise ValueError('Explicit positive hand speeds required')
    return config


class GripGate:
    """Fresh release before arm; 50 ms close debounce; immediate release."""
    def __init__(self):
        self.armed = False
        self.since = None
        self.last_stamp = 0
        self.last_change = -math.inf

    def update(self, sample, now):
        received, stamp, left, right = sample
        valid = all(math.isfinite(x) for x in (*sample, now))
        valid = valid and stamp > 0 and received > 0 and 0 <= now-received <= .2
        valid = valid and 0 <= left <= 1 and 0 <= right <= 1
        if valid and stamp > self.last_stamp:
            self.last_stamp, self.last_change = stamp, now
        elif stamp < self.last_stamp:
            valid = False
        valid = valid and now-self.last_change <= .2
        if not valid:
            self.armed, self.since = False, None
            return False
        if left <= .7 or right <= .7:
            self.since = None
            if left <= .7 and right <= .7:
                self.armed = True
            return False
        if not self.armed:
            return False
        if self.since is None:
            self.since = now
        return now-self.since >= .05


class HandGripRunner:
    def __init__(self, config, shared, *, group, factory=None):
        if factory is None:
            from locomotion_aorta.hand import HandClient
            factory = HandClient
        self.clients = {}
        self.config, self.shared = config, shared
        self.stop = threading.Event()
        self.error = None
        self.thread = None
        self.feedback = {}
        try:
            for side in ('left', 'right'):
                client = factory(side, group=group)
                self.clients[side] = client
                info = client.wait_info(timeout=3.)
                spec = config[side]
                if int(info.hand_type) != 1 or tuple(info.joint_names) != tuple(spec['joint_names']):
                    raise ValueError(f'{side}: BrainCo type/joint order differs from configuration')
                for key in ('open_rad', 'grasp_rad'):
                    if not all(lo <= q <= hi for lo, q, hi in zip(info.q_min, spec[key], info.q_max)):
                        raise ValueError(f'{side}: {key} outside advertised limits')
                if not all(0 < v <= maximum for v, maximum in zip(spec['speed_rad_s'], info.dq_max)):
                    raise ValueError(f'{side}: speed outside advertised limits')
        except BaseException:
            for client in self.clients.values():
                client.close()
            raise

    def start(self):
        self.thread = threading.Thread(target=self.run, name='pico-hand-grip', daemon=True)
        self.thread.start()

    def publish(self, grasp, now):
        states = {}
        for side, client in self.clients.items():
            state = client.state
            if state is None:
                raise RuntimeError(f'{side}: missing hand feedback')
            previous, changed = self.feedback.get(side, (None, now))
            if previous != state.state_id:
                changed = now
            self.feedback[side] = state.state_id, changed
            if now-changed > .2 or not state.output_allowed:
                raise RuntimeError(f'{side}: stale or inhibited hand feedback')
            states[side] = state
        for side, client in self.clients.items():
            spec = self.config[side]
            client.publish(spec['grasp_rad' if grasp else 'open_rad'],
                           spec['speed_rad_s'], input_state=states[side])

    def run(self):
        gate = GripGate()
        previous = None
        try:
            while not self.stop.is_set():
                with self.shared.get_lock():
                    sample = tuple(self.shared[3:7])
                grasp = gate.update(sample, time.monotonic())
                self.publish(grasp, time.monotonic())
                if grasp != previous:
                    print('PICO_HANDS: '+('grasp' if grasp else 'open'), flush=True)
                    previous = grasp
                self.stop.wait(.02)
        except BaseException as error:
            self.error = error
        finally:
            # Best effort only: emergency/inhibit cannot promise physical opening.
            try:
                self.publish(False, time.monotonic())
            except Exception:
                pass

    def check(self):
        if self.error is not None:
            raise RuntimeError('Pico hand controller failed') from self.error

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        for client in self.clients.values():
            try:
                client.close()
            except Exception as error:
                print(f'PICO_HANDS close: {error}', flush=True)
