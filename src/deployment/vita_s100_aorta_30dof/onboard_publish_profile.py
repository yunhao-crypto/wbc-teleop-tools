"""Time the original synchronous publisher, retaining its header, QoS and errors."""

import time


class ProfiledCommandPublisher:
    def __init__(self, publisher):
        self.publisher = publisher
        self.last = {}
        self._native_publish = publisher.publish_bytes
        # The original publish_typed calls this instance method after Finish.
        publisher.publish_bytes = self._publish_bytes

    def _publish_bytes(self, payload):
        start, cpu = time.time_ns(), time.thread_time_ns()
        self.last["bytes_started_wall_ns"] = start
        self.last["payload_bytes"] = len(payload)
        try:
            return self._native_publish(payload)
        finally:
            self.last["bytes_returned_wall_ns"] = time.time_ns()
            self.last["bytes_wall_ms"] = (self.last["bytes_returned_wall_ns"] - start) / 1e6
            self.last["bytes_cpu_ms"] = (time.thread_time_ns() - cpu) / 1e6

    def publish_typed(self, fill):
        self.last = {
            "clock": "CLOCK_REALTIME timestamps and elapsed time; thread CPU separately",
            "scope": "Synchronous original publisher; no bottom delivery timestamp",
        }
        start, cpu = time.time_ns(), time.thread_time_ns()
        self.last["typed_started_wall_ns"] = start

        def measured_fill(builder, header):
            begin, cpu_begin = time.time_ns(), time.thread_time_ns()
            try:
                return fill(builder, header)
            finally:
                self.last["fill_started_wall_ns"] = begin
                self.last["fill_returned_wall_ns"] = time.time_ns()
                self.last["fill_wall_ms"] = (
                    self.last["fill_returned_wall_ns"] - begin
                ) / 1e6
                self.last["fill_cpu_ms"] = (
                    time.thread_time_ns() - cpu_begin
                ) / 1e6

        try:
            # Avoid repeated growth/copy for the ~1 KiB 42-slot command. Builder
            # capacity changes allocation only; emitted bytes are identical.
            return self.publisher.publish_typed(
                measured_fill, initial_size=2048
            )
        finally:
            self.last["typed_returned_wall_ns"] = time.time_ns()
            self.last["typed_wall_ms"] = (self.last["typed_returned_wall_ns"] - start) / 1e6
            self.last["typed_cpu_ms"] = (time.thread_time_ns() - cpu) / 1e6

    def close(self):
        self.publisher.close()
