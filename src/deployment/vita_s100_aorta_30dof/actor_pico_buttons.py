"""Fresh same-packet double-trigger edges; no robot or Pico initialization."""
import math
import threading
import time


class DualTriggerToggle:
    """Bridge semantics: >0.7, both-release latch, 300 ms debounce.

    A connection/stale interval additionally requires a fresh released sample.
    Held/cached buttons therefore cannot engage on connection or recovery.
    """
    def __init__(self):
        self.armed = False
        self.last_stamp = 0
        self.last_toggle = -math.inf

    def update(self, sample, now):
        available, stamp, age, left, right = sample
        if (not available or type(stamp) is not int or stamp <= 0
                or not all(math.isfinite(x) for x in (age,left,right,now))
                or not 0 <= age <= .1 or not 0 <= left <= 1 or not 0 <= right <= 1):
            self.armed = False
            return False
        if stamp == self.last_stamp:
            return False
        if stamp < self.last_stamp:
            self.armed = False
            return False
        self.last_stamp = stamp
        if left <= .7 and right <= .7:
            self.armed = True
        elif left > .7 and right > .7 and self.armed:
            self.armed = False
            if now-self.last_toggle >= .3:
                self.last_toggle = now
                return True
        return False


class TriggerReader:
    """Poll independently of IK; shared monotonic event counter survives coalescing."""
    def __init__(self, sdk, shared):
        self.get = getattr(sdk,'get_controller_trigger_snapshot',None)
        if not callable(self.get):
            raise RuntimeError('Pico binding lacks atomic controller trigger snapshots; update the binding')
        self.shared = shared
        self.toggle = DualTriggerToggle()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run,name='pico-trigger-reader',daemon=True)

    def start(self): self.thread.start()

    def run(self):
        while not self.stop.is_set():
            now = time.monotonic()
            try:
                sample = self.get()
                fired = self.toggle.update(sample,now)
                available,stamp,age,left,right = sample
                with self.shared.get_lock():
                    self.shared[2] = now-age if available and 0 <= age <= .1 else 0.
                    if fired:
                        self.shared[0] += 1
                        self.shared[1] = now
            except Exception:
                self.toggle.armed = False
                with self.shared.get_lock(): self.shared[2] = 0.
            self.stop.wait(.005)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)
