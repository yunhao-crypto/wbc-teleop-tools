"""One atomic reader snapshot for head control, independent of the GMR queue."""
import numpy as np


class LatestPicoPose:
    def __init__(self, context):
        self.lock = context.Lock()
        self.body = context.RawArray('f', 24*7)
        # Keep nanosecond stamps as int64, never rounded through float64.
        self.clocks = context.RawArray('q', 3)
        self.previous = None

    def publish(self, sample):
        # Neither the reader nor the control loop waits for the other process.
        # A contended slot retains its previous frame and original receipt age.
        if not self.lock.acquire(False):
            return False
        try:
            np.copyto(np.frombuffer(self.body, dtype=np.float32).reshape(24, 7),
                      sample['body_poses'])
            self.clocks[:] = (sample['timestamp_ns'], sample['reader_monotonic_ns'],
                              sample['reader_frame_sequence'])
        finally:
            self.lock.release()
        return True

    def snapshot(self):
        if self.lock.acquire(False):
            try:
                stamp, receipt, sequence = self.clocks[:]
                body = np.frombuffer(self.body, dtype=np.float32).reshape(24, 7).copy() if stamp else None
            finally:
                self.lock.release()
            if body is not None:
                self.previous = dict(raw_body_poses=body, pico_timestamp_ns=stamp,
                    received_monotonic=receipt*1e-9, reader_monotonic_ns=receipt,
                    reader_frame_sequence=sequence, source='pico_reader')
        return self.previous
