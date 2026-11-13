"""Reference-only action14 lifecycle; no transport or model effects on import.

Action activation and Pico engagement are separate: an activated native policy
keeps balancing in stand while references buffer, and only a fresh trigger can
request stream_start. Returning to stand never means cancelling to PASSIVE.
"""
import math
import time
import uuid

from .sonic_action14 import ReferenceEncoder
from .sonic_action14_resample import SourceGrid
from .sonic_action14_source import reference_contract


class Action14Session:
    def __init__(self, control, publisher, *, clock=time.monotonic, margin_min_rad=1.5, head=None):
        self.control, self.publisher, self.clock = control, publisher, clock
        self.margin_min_rad = margin_min_rad
        self.head = head
        self.request_id = None
        self.encoder = None
        self.active = False
        self.following = False
        self.last_trigger = 0
        self.missed_grid = 0
        self.last_status_at = None
        self._admission_ok = False
        # 500 ms ownership limit. The 20 ms publisher must not RPC on every tick.
        self.admission_interval_s = .1
        self.last_runtime_check = None
        self.runtime_id = None
        self.frames_published = 0
        self.last_engage_refusal = None
        self.last_posture = None
        self._reset_source()

    def _reset_source(self):
        self.grid = SourceGrid()
        self.last_source_sequence = 0
        self.last_receipt = None
        self.pending = []
        self.next_send = None
        self.frames_published = 0
        # Source-grid jitter buffer only, not the policy's future window.
        # Due time follows each sample's own receipt, not a source-clock anchor.
        self.source_playback_margin_s = .15
        if self.runtime_id is not None:
            # A new source session cannot replay the previous segment's tail.
            self.encoder = ReferenceEncoder(self.runtime_id)

    def _owned(self):
        if not self.active:
            return False
        status = self.control.status()
        age = status.get('status_age_ms')
        return (status.get('owned_req_id') == self.request_id and
                status.get('active') is True and status.get('superseded') is False and
                status.get('current_action') == 'HUMANOID_TRACKING' and
                type(age) in (int, float) and math.isfinite(age) and 0 <= age < 500)

    def _capabilities(self):
        result = self.control.event(self.request_id, 'humanoid_tracking.query', {})
        data = result.get('data', {})
        margin = data.get('native_target_margin_rad')
        if (data.get('native_raw_affine_targets') is not True or
                data.get('native_stream_auto_resume') is not False or
                data.get('native_stream_auto_start') is not False or
                type(margin) not in (int, float) or not math.isfinite(margin) or
                not self.margin_min_rad <= margin <= math.pi or
                data.get('native_runtime_config') !=
                'actions/policy_profiles/vt_human_v2_tracking_asample_wtt_native.yaml'):
            raise RuntimeError('active native profile/capability mismatch')
        if self.head is not None and data.get('native_neck_follow_reference') is not True:
            raise RuntimeError('native profile is not following the neck reference')
        return data

    def initialize(self):
        if self.active:
            raise RuntimeError('action already activated')
        info = self.publisher.robot_info()
        self.runtime_id = info.runtime_id
        self.request_id = str(uuid.uuid4())
        self.control.activate(self.request_id, 'profile_wtt_native')
        self.active = True
        try:
            # action.started can precede the status rename. One sample of
            # FIXED_STAND here is not a failed activation.
            deadline = self.clock()+2.
            while not self._owned():
                if self.clock() >= deadline:
                    raise RuntimeError('no correlated action14 activation: '+
                                       repr(self.control.status()))
                time.sleep(.02)
            self._admission_ok = True
            self.last_status_at = self.clock()
            self._capabilities()
            self._reset_source()
            self.last_runtime_check = self.clock()
        except BaseException:
            # No generic CANCEL on ambiguous/foreign ownership. A caller must
            # treat an incomplete activation as requiring operator attention.
            self.active = False
            raise

    def engage(self):
        if not self.active:
            self.last_engage_refusal = 'inactive'
            return False
        if not self._owned():
            raise RuntimeError('action14 is not our active request')
        if self.following:
            return True
        now = self.clock()
        if (self.frames_published < 1 or self.last_receipt is None or
                not 0 <= now-self.last_receipt <= .6):
            self.last_engage_refusal = 'no_local_reference'
            return False
        try:
            # Local receipt is not C++ acceptance. Refuse this trigger and
            # stay in stand; do not queue a later automatic engage.
            queried = self.control.event(self.request_id, 'humanoid_tracking.query', {})
            data = queried.get('data') or {}
            received = data.get('stream_frames_received')
            age = data.get('stream_frame_age_ms')
            if (type(received) not in (int, float) or received < 1 or
                    type(age) not in (int, float) or not math.isfinite(age) or
                    not 0 <= age <= 600):
                self.last_engage_refusal = 'no_stream'
                return False
            self.control.event(self.request_id, 'humanoid_tracking.stream_start', {})
        except Exception:
            self.following = False
            self.last_engage_refusal = 'refused'
            return False
        self.following = True
        self.last_engage_refusal = None
        return True

    def disengage(self):
        outcome = 'inactive'
        try:
            if not self.active:
                outcome = 'inactive'
            elif not self._owned():
                outcome = 'ambiguous'
            else:
                try:
                    result = self.control.event(
                        self.request_id, 'humanoid_tracking.return_to_stand', {})
                    data = result.get('data') or {}
                    if (data.get('lead_in_started') is True and
                            data.get('stay_state') == 'lead_in' and
                            data.get('following') is False):
                        outcome = 'lead_in_accepted'
                    else:
                        outcome = 'ambiguous'
                except Exception:
                    outcome = 'ambiguous'
        finally:
            self.following = False
            self._reset_source()
            self.last_posture = outcome
        return outcome

    def update_trigger(self, count, receipt, fresh_until=.6):
        now = self.clock()
        if type(count) is not int or count < self.last_trigger:
            self.last_trigger = max(0, count) if type(count) is int else 0
            self.disengage()
            return 'disengaged'
        if count == self.last_trigger:
            return 'ignored'
        delta = count-self.last_trigger
        self.last_trigger = count
        if delta != 1 or not math.isfinite(receipt) or not 0 <= now-receipt <= fresh_until:
            self.disengage()
            return 'disengaged'
        if self.following:
            self.disengage()
            return 'disengaged'
        if self.engage():
            return 'following'
        return 'refused'

    def accept(self, snapshot):
        if snapshot.get('state') == 'failed':
            self.disengage()
            raise RuntimeError('Pico/GMR worker failed: '+str(snapshot.get('error')))
        if snapshot.get('state') != 'ready' or not self.active:
            return
        now = self.clock()
        seq, receipt = snapshot['sequence'], snapshot['received_monotonic']
        if seq <= self.last_source_sequence:
            return  # An unchanged latest-value snapshot cannot renew freshness.
        if not math.isfinite(receipt) or not 0 <= now-receipt <= .6:
            if self.last_receipt is not None:
                self.disengage()
            return
        if self.head is not None and not isinstance(snapshot.get('head_snapshot'), dict):
            self.disengage()
            return
        self.last_source_sequence = seq
        self.last_receipt = receipt
        try:
            samples = self.grid.push(snapshot['source_ns'], receipt,
                                     snapshot['q'], snapshot['root_wxyz'])
        except ValueError:
            self.disengage()
            return
        for sample in samples:
            sample['due'] = sample['received_monotonic'] + self.source_playback_margin_s
            if self.head is not None:
                sample['head_snapshot'] = snapshot['head_snapshot']
            self.pending.append(sample)
        # A bounded source-only queue cannot build an unbounded playback tail.
        if len(self.pending) > 32:
            self.disengage()
            return
        if self.pending:
            self.next_send = self.pending[0]['due']

    def _admission_current(self, now):
        if (self._admission_ok and self.last_status_at is not None and
                now-self.last_status_at < self.admission_interval_s):
            return True
        self._admission_ok = self._owned()
        self.last_status_at = now
        return self._admission_ok

    def tick(self):
        if not self.active:
            return
        now = self.clock()
        if not self._admission_current(now):
            self.active = False
            self.following = False
            self.pending = []
            raise RuntimeError('action14 superseded or status stale; no cancellation sent')
        if self.last_receipt is not None and now-self.last_receipt > .6:
            self.disengage()
            return
        if self.pending and now >= self.next_send:
            # Drop expired slots instead of bursting stale references. The
            # published sequence stays contiguous: a retained grid index is a
            # C++ sequence_gap and freezes the hold. The pose is the newest due
            # sample. A real Pico pause still ends follow on the receipt age above.
            while len(self.pending) > 1 and self.pending[1]['due'] <= now:
                self.pending.pop(0)
                self.missed_grid += 1
            sample = self.pending.pop(0)
            self.next_send = self.pending[0]['due'] if self.pending else None
            neck_q = None
            head_pending = None
            if self.head is not None:
                try:
                    neck_q, head_pending = self.head.preview_targets(sample.get('head_snapshot'), now)
                except ValueError:
                    self.disengage()
                    return
            names = reference_contract().names
            try:
                command = self.encoder.encode(dict(zip(names, sample['q'])), sample['root_wxyz'],
                    source_ns=sample['source_ns'], received_monotonic=sample['received_monotonic'],
                    now_monotonic=now, neck_q=neck_q)
            except ValueError:
                self.disengage()
                return
            self.publisher.publish(command, authorized=lambda: self._admission_ok)
            self.frames_published += 1
            if head_pending is not None:
                self.head.commit(head_pending)
        # Identity is not the 20 ms path. After the due frame, at most once a
        # second, and the query itself cannot sit on the publisher for the
        # transport's 2 s deadline.
        if now-self.last_runtime_check >= 1.0:
            self.last_runtime_check = now
            try:
                info = self.publisher.robot_info(timeout=.05)
            except Exception:
                info = None
            if info is not None and info.runtime_id != self.runtime_id:
                self.active = False
                self.pending = []
                raise RuntimeError('robot runtime identity changed')

    def close(self, *, return_to_stand=True):
        outcome = 'ambiguous'
        try:
            if return_to_stand:
                outcome = self.disengage()
            else:
                self.following = False
                self._reset_source()
                outcome = 'passive_requested'
                self.last_posture = outcome
        except Exception:
            outcome = 'ambiguous'
            self.last_posture = outcome
        finally:
            self.active = False
            try:
                self.publisher.close()
            finally:
                self.control.close()
        return outcome
