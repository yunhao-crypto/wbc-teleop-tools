"""Bounded, best-effort GMR evidence sink; never a source of control authority.

Only the daemon writer opens/serializes/writes/closes files. enqueue owns a deep
copy, so mutable vendor/numpy buffers can be reused immediately. At most capacity
queued records plus one active record exist. A stalled filesystem can outlive
close's timeout; the caller must treat that capture as incomplete, not drained.
"""
import copy
import json
import os
import queue
import threading
import time


def _json_default(value):
    # Conversion is deliberately on the writer, not on the GMR producer.
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported source-log value: {type(value).__name__}')


class SourceWriter:
    def __init__(self, path=None, *, header=None, capacity=64, sink=None, opener=None):
        if capacity < 1:
            raise ValueError('source writer capacity must be positive')
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._closed = False
        self._failed = False
        self._path, self._sink, self._opener = path, sink, opener
        self._header = copy.deepcopy(header)
        self._counts = dict(attempted=0, enqueued=0, dropped_full=0,
                            dropped_closed=0, dropped_error=0, written=0, errors=0, abandoned=0,
                            enqueue_ns_total=0, enqueue_ns_max=0,
                            serialize_ns_total=0, write_ns_total=0, write_ns_max=0)
        self._error = None
        self._thread = threading.Thread(target=self._run, name='gmr-source-writer', daemon=True)
        self._thread.start()

    def enqueue(self, record):
        begin = time.monotonic_ns()
        # Producer-side bounded copy only. No JSON, tolist, or sink operations.
        # Holding this short lock also linearizes concurrent enqueue/close.
        with self._lock:
            self._counts['attempted'] += 1
            reason = 'dropped_closed' if self._closed else ('dropped_error' if self._failed else None)
            if reason is None and self._queue.full():
                reason = 'dropped_full'
            if reason is not None:
                self._counts[reason] += 1
                return False
            try:
                owned = copy.deepcopy(record)
            except Exception as error:
                self._counts['dropped_error'] += 1
                self._counts['errors'] += 1
                self._error = repr(error)
                return False
            end = time.monotonic_ns()
            owned['source_log_enqueue_start_monotonic_ns'] = begin
            owned['source_log_enqueue_ready_monotonic_ns'] = end
            # This is snapshot ownership/preparation, NOT queue insertion latency.
            self._queue.put_nowait(owned)
            elapsed = time.monotonic_ns() - begin
            self._counts['enqueued'] += 1
            self._counts['enqueue_ns_total'] += elapsed
            self._counts['enqueue_ns_max'] = max(self._counts['enqueue_ns_max'], elapsed)
            return True

    def stats(self):
        with self._lock:
            return dict(self._counts, queue_capacity=self._queue.maxsize,
                        queued=self._queue.qsize(), error=self._error,
                        complete=self._done.is_set() and not self._failed,
                        complete_scope='accepted_records_drained',
                        full_source_coverage=None,
                        closed=self._closed)

    def close(self, timeout=0.5):
        with self._lock:
            self._closed = True
            self._stop.set()
        self._thread.join(timeout=max(0., timeout))
        return self.stats()

    def _write(self, sink, record):
        begin = time.monotonic_ns()
        text = json.dumps(record, default=_json_default, allow_nan=False,
                          separators=(',', ':')) + '\n'
        serialized = time.monotonic_ns()
        count = sink.write(text)
        if count is not None and count != len(text):
            raise OSError('short GMR source-log write')
        sink.flush()
        end = time.monotonic_ns()
        with self._lock:
            self._counts['serialize_ns_total'] += serialized-begin
            self._counts['write_ns_total'] += end-serialized
            self._counts['write_ns_max'] = max(self._counts['write_ns_max'], end-serialized)
        return dict(serialize_ns=serialized-begin, write_flush_ns=end-serialized)

    def _run(self):
        sink = None
        try:
            # Explicitly disallow inherited real-time priority for file work.
            if hasattr(os, 'sched_setscheduler'):
                os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
            sink = self._sink
            if sink is None:
                sink = (self._opener or (lambda path: open(path, 'x')))(self._path)
            if self._header is not None:
                self._write(sink, dict(self._header, record_type='header',
                    source_log_encoding='bounded_async_v1', queue_capacity=self._queue.maxsize))
            previous_write = None
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=.02)
                except queue.Empty:
                    continue
                record['source_log_writer_start_monotonic_ns'] = time.monotonic_ns()
                # Same-row write duration cannot be known before writing that row.
                record['source_log_previous_write'] = previous_write
                previous_write = self._write(sink, record)
                with self._lock:
                    self._counts['written'] += 1
            with self._lock:
                counters = dict(self._counts)
            self._write(sink, dict(record_type='footer', complete=True,
                                  complete_scope='accepted_records_drained',
                                  full_source_coverage=None,
                                  durability_confirmed=False, close_confirmed=False,
                                  counters=counters, last_write=previous_write))
        except BaseException as error:
            with self._lock:
                self._failed = True
                self._counts['errors'] += 1
                self._error = repr(error)
                self._counts['abandoned'] = self._counts['enqueued']-self._counts['written']
            # Release queued snapshots; never retry an unbounded failing sink.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        finally:
            if sink is not None and self._sink is None:
                try:
                    sink.close()
                except BaseException as error:
                    with self._lock:
                        self._failed = True
                        self._counts['errors'] += 1
                        self._error = repr(error)
            self._done.set()
