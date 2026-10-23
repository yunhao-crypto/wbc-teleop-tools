"""Verify immutable records, drain ordering and fail-closed writer failures."""

import json
import queue

import numpy as np
import pytest

from deployment.vita_s100_aorta_30dof.async_recording import (
    ProcessJsonlSessionRecorder,
)


def test_snapshot_order_and_drain(tmp_path):
    with ProcessJsonlSessionRecorder(tmp_path, prefix="test") as recorder:
        state = np.arange(3126, dtype=np.float32)
        payload = {"observation": state, "nested": [{"value": 1}]}
        recorder.write(payload)
        state[:] = -1
        payload["nested"][0]["value"] = 9
        for n in range(80):
            recorder.write({"n": n, "action": np.arange(30)})
    records = [
        json.loads(line) for line in recorder.path.read_text().splitlines()
    ]
    assert len(records) == 81
    assert records[0]["observation"] == list(range(3126))
    assert records[0]["nested"] == [{"value": 1}]
    assert [r["n"] for r in records[1:]] == list(range(80))
    assert recorder._process.exitcode == 0
    recorder.close()


def test_full_queue_is_error_and_accepted_records_drain(tmp_path, monkeypatch):
    recorder = ProcessJsonlSessionRecorder(tmp_path, prefix="test")
    recorder.write({"accepted": 1})

    def full(*_):
        raise queue.Full

    monkeypatch.setattr(recorder._messages, "put_nowait", full)
    with pytest.raises(RuntimeError, match="queue full"):
        recorder.write({"rejected": 2})
    with pytest.raises(RuntimeError, match="queue full"):
        recorder.check_health()
    recorder.close()
    assert json.loads(recorder.path.read_text()) == {"accepted": 1}


def test_dead_worker_does_not_hang_or_claim_success(tmp_path):
    recorder = ProcessJsonlSessionRecorder(tmp_path, prefix="test", timeout=1)
    recorder._process.terminate()
    recorder._process.join(2)
    with pytest.raises(RuntimeError, match="exited"):
        recorder.write({"cannot_accept": 1})
    with pytest.raises(RuntimeError, match="before log drain"):
        recorder.close()
    assert recorder._closed


def test_parent_validates_payload_and_worker_write_failure(tmp_path):
    recorder = ProcessJsonlSessionRecorder(tmp_path, prefix="test")
    with pytest.raises(TypeError, match="Unsupported"):
        recorder.write({"bad": object()})
    # Serialization failure is returned by the worker and fails drain.
    recorder.write({"bad_json": np.array([1 + 2j])})
    with pytest.raises(
        RuntimeError, match="Incomplete log drain|before log drain"
    ):
        recorder.close()


def test_startup_failure(tmp_path):
    file = tmp_path / "not_a_directory"
    file.write_text("x")
    with pytest.raises(RuntimeError, match="startup failed"):
        ProcessJsonlSessionRecorder(file, prefix="test")


def test_ctrl_c_is_owned_by_parent_and_logs_drain(tmp_path):
    import os
    import signal

    with ProcessJsonlSessionRecorder(tmp_path, prefix="test") as recorder:
        recorder.write({"before": True})
        os.kill(recorder._process.pid, signal.SIGINT)
        recorder.write({"after": True})
    assert [
        json.loads(line) for line in recorder.path.read_text().splitlines()
    ] == [{"before": True}, {"after": True}]
