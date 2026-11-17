"""Offline fixtures, including real FlatBuffers bindings and multi-topic MCAP."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from deployment.vita_s100_aorta_30dof.actor_body_trace import (
    DEFAULT_TOPIC, _joint_names, extract_records, extract_mcap, write_npz, load_binding,
)

NAMES = [f"joint_{i}" for i in range(31)]
SDK = Path(__file__).parents[1] / "venv/lib/python3.10/site-packages"

class J:
    Q = lambda s: float("nan")
    Dq = lambda s: 0.
    TauEst = lambda s: float("inf")
    TsAcquiredHwUs = lambda s: 2**54 + 1
    TsIngestedUs = lambda s: 0
    TsAssembledUs = lambda s: 8
    ErrorCode = lambda s: 2**32 - 1

class S:
    def __init__(self, x=1.): self.x = x
    Quaternion = lambda s: [s.x, 0, 0, 0]
    Accelerometer = lambda s: [0, 0, 9.8]
    Gyroscope = lambda s: [0, 0, 0]
    Rpy = lambda s: [0, 0, 0]

class I:
    def __init__(self, time=2**54+3, x=1., status=1): self.time, self.x, self.status = time, x, status
    Location = lambda s: 0
    Status = lambda s: s.status
    TsSampledUs = lambda s: 0
    TsAcquiredHwUs = lambda s: s.time
    TsIngestedUs = lambda s: 0
    TsAssembledUs = lambda s: 5
    State = lambda s: S(s.x)

class B:
    def __init__(self, imus=None): self.imus = [I()] if imus is None else imus
    ProtocolVersion = lambda s: 3
    TsStateRealUs = lambda s: 2**54 + 1
    TsStatePubUs = lambda s: 9
    AppliedSessionId = lambda s: 0
    AppliedSequence = lambda s: 0
    AppliedKeyValid = lambda s: True
    AssociationSourceType = lambda s: 2
    StateId = lambda s: 2**63 + 1
    RuntimeId = lambda s: 2**63 + 2
    TsFirstOutputUs = lambda s: 0
    JointsLength = lambda s: 31
    Joints = lambda s, i: J()
    ImuStatesLength = lambda s: len(s.imus)
    ImuStates = lambda s, i: s.imus[i]


def test_uint64_masks_zero_protocol3_and_npz(tmp_path):
    b, i = extract_records([(B(), None), (B(), None)], NAMES)
    assert len(i["location"]) == 1
    assert b["state_id"][0] == 2**63 + 1
    assert b["joint_ts_acquired_hw_us"][0, 0] == 2**54 + 1
    assert b["joint_ts_ingested_us_field_present"][0, 0]
    assert not b["joint_ts_ingested_us_present"][0, 0]
    assert not b["joint_q_finite"][0, 0]
    assert b["applied_association_valid"].all()
    assert b["applied_sequence"][0] == 0
    assert not b["applied_firstout_present"][0]
    write_npz(tmp_path / "out", b, i, NAMES)
    with np.load(tmp_path / "out/body.npz", allow_pickle=False) as z:
        assert z["state_id"].dtype == np.uint64
        assert int(z["state_id"][0]) == 2**63 + 1
    with pytest.raises(FileExistsError): write_npz(tmp_path / "out", b, i, NAMES)


def test_missing_identity_and_invalid_status_preserved():
    b, i = extract_records([(B([I(0, status=0), I(0, status=0)]), None)] * 2, NAMES)
    assert len(i["location"]) == 4
    assert not i["identity_valid"].any()
    assert not i["sample_status_ok"].any()
    assert (i["status"] == 0).all()


def test_contradictory_identity_tags_all_payloads():
    _, i = extract_records([(B([I(x=1), I(x=2), I(x=1)]), None)], NAMES)
    assert len(i["location"]) == 2
    assert i["identity_contradiction"].all()


def test_roster_exact_and_consistent_stream(tmp_path):
    p = tmp_path / "actor.jsonl"
    p.write_text('ACTORLOG ' + json.dumps({"body_order": NAMES}) + '\n' + json.dumps({"body_order": NAMES}) + '\n')
    assert _joint_names(actor_log=p) == NAMES
    with p.open("a") as f: f.write(json.dumps({"body_order": NAMES[::-1]}) + '\n')
    with pytest.raises(ValueError, match="inconsistent"): _joint_names(actor_log=p)
    with pytest.raises(ValueError): extract_records([], NAMES[:-1])
    with pytest.raises(ValueError): extract_records([], ["x"] * 31)
    class Bad(B):
        JointsLength = lambda s: 30
    with pytest.raises(ValueError, match="count"): extract_records([(Bad(), None)], NAMES)


def binary_body(protocol=3):
    pytest.importorskip("flatbuffers")
    if not SDK.exists(): pytest.skip("SDK generated bindings not installed")
    load_binding(str(SDK))
    import flatbuffers
    from lowlevel.BodyState import BodyStateT
    from lowlevel.BodyJointState import BodyJointStateT
    from lowlevel.BodyImuSample import BodyImuSampleT
    from lowlevel.BodyImuState import BodyImuStateT
    b = BodyStateT()
    b.protocolVersion = protocol
    b.stateId = 2**63 + 1
    b.runtimeId = 2**63 + 2 if protocol == 3 else 0
    b.appliedKeyValid = protocol == 3
    b.associationSourceType = 2 if protocol == 3 else 0
    b.joints = [BodyJointStateT() for _ in NAMES]
    u = BodyImuSampleT(); u.location = 0; u.status = 1; u.tsAcquiredHwUs = 2**54 + 3
    u.state = BodyImuStateT(); u.state.quaternion = [1, 0, 0, 0]
    u.state.accelerometer = [0, 0, 9.8]; u.state.gyroscope = [0, 0, 0]; u.state.rpy = [0, 0, 0]
    b.imuStates = [u]
    builder = flatbuffers.Builder(4096); offset = b.Pack(builder); builder.Finish(offset)
    return bytes(builder.Output())


@pytest.mark.parametrize("protocol", [2, 3])
def test_real_binding_absent_scalar_defaults(protocol):
    raw = binary_body(protocol)
    b, i = extract_records([(raw, raw)], NAMES, str(SDK))
    assert not b["applied_sequence_field_present"][0]
    assert bool(b["applied_association_valid"][0]) == (protocol == 3)
    assert not b["body_realtime_present"][0]
    assert not b["body_realtime_us_field_present"][0]
    assert b["joint_q_field_present"].all()  # fixed struct, zero is serialized
    assert not b["joint_ts_ingested_us_present"].any()
    assert i["location"][0] == 0  # HIP default omitted: still a real location
    assert not i["location_field_present"][0]


def make_mcap(path):
    pytest.importorskip("mcap")
    from mcap.writer import Writer, CompressionType
    raw = binary_body()
    with path.open("wb") as f:
        w = Writer(f, compression=CompressionType.NONE); w.start()
        other = w.register_schema("Other", "jsonschema", b"unrelated schema")
        oc = w.register_channel("other", "json", other)
        w.add_message(oc, 1, b"{}", 1)
        sid = w.register_schema("BodyState", "flatbuffer", b"selected schema")
        cid = w.register_channel(DEFAULT_TOPIC, "flatbuffer", sid)
        w.add_message(cid, 2, raw, 2); w.finish()
    return hashlib.sha256(b"selected schema").hexdigest()


def test_mcap_selected_schema_only_receipt_and_exclusive(tmp_path):
    path = tmp_path / "input.mcap"; pin = make_mcap(path)
    out = tmp_path / "out"
    meta = extract_mcap([path], out, pin, joint_names=NAMES, sdk_python_path=str(SDK))
    assert meta["body_rows"] == 1
    assert len(meta["schemas"]) == 1
    assert meta["inputs"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["binding_source_sha256"]
    assert meta["crc"]["enabled"]
    assert meta["extraction_valid"]
    with pytest.raises(FileExistsError): extract_mcap([path], out, pin, joint_names=NAMES)
    with pytest.raises(ValueError, match="hash mismatch"):
        extract_mcap([path], tmp_path / "bad", "0"*64, joint_names=NAMES)
    assert not (tmp_path / "bad/metadata.json").exists()
    with pytest.raises(ValueError, match="no records"):
        extract_mcap([path], tmp_path / "absent", pin, topic="/"+DEFAULT_TOPIC, joint_names=NAMES)


def test_corrupt_chunk_crc_is_rejected(tmp_path):
    from mcap.stream_reader import CRCValidationError
    path = tmp_path / "corrupt.mcap"; pin = make_mcap(path)
    data = path.read_bytes()
    assert b"unrelated schema" in data
    path.write_bytes(data.replace(b"unrelated schema", b"Xnrelated schema", 1))
    with pytest.raises(CRCValidationError):
        extract_mcap([path], tmp_path / "out", pin, joint_names=NAMES, sdk_python_path=str(SDK))
    assert not (tmp_path / "out/metadata.json").exists()


def test_unsupported_crc_fails_closed(tmp_path, monkeypatch):
    import mcap.reader
    monkeypatch.setattr(mcap.reader, "make_reader", lambda stream: None)
    with pytest.raises(RuntimeError, match="CRC"):
        extract_mcap([], tmp_path / "crc", "0"*64, joint_names=NAMES)
