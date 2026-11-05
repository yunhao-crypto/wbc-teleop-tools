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
    def __init__(self, config, shared, *, connection, factory=None):
        if factory is None:
            from locomotion_aorta.end_effector import EndEffectorClient
            factory = EndEffectorClient
        from locomotion_aorta import _end_effector_codec as codec
        from locomotion_aorta.end_effector_types import (
            ControlMode, DeviceModel, EndEffectorTarget, Field, PositionDomain, SampleStream)
        self.codec = codec
        self.clients = {}
        self.targets = {}
        self.infos = {}
        self.config, self.shared = config, shared
        self.stop = threading.Event()
        self.error = None
        self.thread = None
        self.feedback = {}
        try:
            for side in ('left', 'right'):
                client = factory(side, connection=connection,
                                 observation_sources=((SampleStream.EE_STATE, 1 if side == 'left' else 2),))
                self.clients[side] = client
                info = client.wait_info(timeout=3.)
                spec = config[side]
                codec.validate_info(info)
                if (info.device_model != DeviceModel.BRAINCO_REVO2
                        or tuple(info.axis_names) != tuple(spec['joint_names'])
                        or any(domain != PositionDomain.JOINT_ANGLE for domain in info.position_domain)):
                    raise ValueError(f'{side}: BrainCo type/joint order/SI domain differs from configuration')
                self.infos[side] = info
                self.targets[side] = {
                    grasp: codec.prepare_target(info, EndEffectorTarget(
                        (ControlMode.POSITION,) * 6, (Field.POS | Field.VEL_LIMIT,) * 6,
                        pos=tuple(spec['grasp_rad' if grasp else 'open_rad']),
                        vel_limit=tuple(spec['speed_rad_s'])))
                    for grasp in (False, True)}
        except BaseException:
            for client in self.clients.values():
                client.close()
            raise

    def start(self):
        self.thread = threading.Thread(target=self.run, name='pico-hand-grip', daemon=True)
        self.thread.start()

    def publish(self, grasp, now):
        from locomotion_aorta.end_effector_types import (
            Field, SampleStatus, InhibitReason, _STATE_READY_VALID,
            _STATE_OUTPUT_ALLOWED_VALID, _STATE_INHIBIT_REASON_VALID)
        states = {}
        for side, client in self.clients.items():
            # The SDK preserves the original position sample across telemetry-only
            # publications. Latest status is authoritative for permission revocation.
            latest, position, fault, received = client.motion_feedback()
            info = client.info  # also detects a retired descriptor/runtime
            for state in (latest, position, fault):
                if state is not None:
                    self.codec.validate_state(info, state)
                    if any(mask & Field.ERROR_CODE and code
                           for mask, code in zip(state.valid_fields, state.error_code)):
                        raise RuntimeError(f'{side}: valid end-effector fault')
            required = _STATE_READY_VALID | _STATE_OUTPUT_ALLOWED_VALID | _STATE_INHIBIT_REASON_VALID
            if (latest is None or latest.status_valid_mask & required != required
                    or not latest.ready or not latest.output_allowed
                    or latest.inhibit_reason != InhibitReason.NONE):
                raise RuntimeError(f'{side}: unavailable or revoked output permission')
            if (position is None or received is None or not 0 <= now-received < .2
                    or position.sample_status != SampleStatus.OK
                    or not all(mask & Field.POS and math.isfinite(q) and lo <= q <= hi
                        for mask, q, lo, hi in zip(position.valid_fields, position.pos,
                                                  info.pos_min, info.pos_max))):
                raise RuntimeError(f'{side}: stale or invalid original position feedback')
            states[side] = position
        for side, client in self.clients.items():
            client.publish(self.targets[side][grasp],
                           input_observations=(states[side].sample_ref,))

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
