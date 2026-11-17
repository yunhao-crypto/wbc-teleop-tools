"""Offline, schema-pinned BodyState extraction; no control/network imports.

*_field_present describes serialized table presence (default-valued FlatBuffers
scalars may be omitted). *_present on timestamps means nonzero/observed, not
clock comparability. Struct floats retain NaN/Inf; *_finite is NOT physical
validity. Body realtime on IMU rows is a publication snapshot PROXY only.
"""
from __future__ import annotations
import hashlib
import importlib
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

DEFAULT_TOPIC = "aorta/default/pub/sdk/foot_humanoid/body_state"
BODY_FIELDS = {
    "protocol_version": ("ProtocolVersion", 6), "roster_version": ("RosterVersion", 8),
    "state_id": ("StateId", 10), "body_realtime_us": ("TsStateRealUs", 12),
    "body_pub_us": ("TsStatePubUs", 14), "body_consumed_us": ("TsConsumedUs", 16),
    "attitude_valid": ("AttitudeValid", 18), "applied_session": ("AppliedSessionId", 24),
    "applied_sequence": ("AppliedSequence", 26), "applied_firstout_us": ("TsFirstOutputUs", 28),
    "runtime_id": ("RuntimeId", 30), "body_control_us": ("TsControlUs", 32),
    "applied_key_valid": ("AppliedKeyValid", 34), "association_source_type": ("AssociationSourceType", 36),
}
IMU_FIELDS = {"ts_sampled_us": ("TsSampledUs", 4), "ts_acquired_hw_us": ("TsAcquiredHwUs", 6),
              "ts_ingested_us": ("TsIngestedUs", 8), "ts_assembled_us": ("TsAssembledUs", 10),
              "location": ("Location", 12), "status": ("Status", 16)}
JOINT_FIELDS = {"joint_q": "Q", "joint_dq": "Dq", "joint_tau": "TauEst",
                "joint_ts_acquired_hw_us": "TsAcquiredHwUs", "joint_ts_ingested_us": "TsIngestedUs",
                "joint_ts_assembled_us": "TsAssembledUs", "joint_error_code": "ErrorCode"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _get(obj, name, default=0):
    fn = getattr(obj, name, None)
    return fn() if callable(fn) else default


def _field(obj, name, offset):
    value = _get(obj, name)
    tab = getattr(obj, "_tab", None)
    return value, bool(tab.Offset(offset)) if tab is not None else hasattr(obj, name)


def _roster(names):
    if not isinstance(names, list) or len(names) != 31 or not all(isinstance(n, str) and n for n in names) or len(set(names)) != 31:
        raise ValueError("exactly 31 unique nonempty body joint names required")
    return names


def _joint_names(args_names=None, actor_log=None):
    names = None
    if args_names:
        names = json.loads(Path(args_names).read_text())
        if isinstance(names, dict): names = names.get("joint_names", names.get("body_order"))
        _roster(names)
    if actor_log:
        found = False
        with open(actor_log) as f:
            for line in f:
                # ACTORLOG console prefixes are accepted, without loading the log.
                start = line.find("{")
                if start < 0: continue
                try: row = json.loads(line[start:])
                except json.JSONDecodeError: continue
                if not isinstance(row, dict) or "body_order" not in row: continue
                order = _roster(row["body_order"]); found = True
                if names is not None and order != names: raise ValueError("inconsistent body_order in actor log / supplied roster")
                names = order
        if not found: raise ValueError("actor log contains no body_order")
    return _roster(names)


def load_binding(sdk_python_path=None):
    if sdk_python_path:
        path = str(Path(sdk_python_path).resolve())
        if path not in sys.path: sys.path.insert(0, path)
    return importlib.import_module("lowlevel.BodyState").BodyState


def decode_body_state(data, sdk_python_path=None):
    return load_binding(sdk_python_path).GetRootAs(data, 0)


def _scalar_row(obj, fields):
    row = {}
    for key, (method, offset) in fields.items():
        value, present = _field(obj, method, offset)
        row[key] = int(value); row[key + "_field_present"] = present
        # Historical key names retained; timestamp availability differs from serialization.
        alias = key[:-3] if key.endswith("_us") and key.startswith(("body_", "applied_")) else key
        row[alias + "_present"] = bool(value) if key.endswith("_us") else present
    return row


def _append(columns, row):
    for k, v in row.items(): columns[k].append(v)


def _arrays(columns):
    result = {}
    for k, values in columns.items():
        first = values[0]
        dtype = bool if isinstance(first, (bool, np.bool_)) else None if isinstance(first, np.ndarray) else np.uint64
        result[k] = np.asarray(values, dtype=dtype)
    return result


def extract_records(records, joint_names, sdk_python_path=None):
    """Stream records; retain only numeric output columns and dedup fingerprints."""
    _roster(joint_names)
    bodies, imus, seen = defaultdict(list), defaultdict(list), {}
    decoder = None
    for msg, raw in records:
        if isinstance(msg, (bytes, bytearray)):
            if decoder is None: decoder = load_binding(sdk_python_path)
            b = decoder.GetRootAs(msg, 0)
        else: b = msg
        row = _scalar_row(b, BODY_FIELDS)
        if int(_get(b, "JointsLength")) != 31: raise ValueError("BodyState joint count does not match exact 31-slot roster")
        joints = [b.Joints(i) for i in range(31)]
        for key, method in JOINT_FIELDS.items():
            floating = key in ("joint_q", "joint_dq", "joint_tau")
            row[key] = np.array([_get(j, method, np.nan if floating else 0) for j in joints], dtype=np.float32 if floating else np.uint64)
            present = np.array([hasattr(j, method) for j in joints], bool)
            row[key + "_field_present"] = present
            row[key + "_present"] = present & (row[key] != 0) if "_ts_" in key else present
            if floating: row[key + "_finite"] = present & np.isfinite(row[key])
        row["joint_present"] = np.ones(31, bool)
        protocol = row["protocol_version"]
        row["applied_association_valid"] = bool(
            (protocol == 3 and row["applied_key_valid"] and 1 <= row["association_source_type"] <= 6)
            or (protocol == 2 and row["applied_session"] != 0 and row["applied_sequence"] != 0))
        body_index = len(bodies["state_id"])
        _append(bodies, row)
        for idx in range(int(_get(b, "ImuStatesLength"))):
            sample = b.ImuStates(idx); ir = _scalar_row(sample, IMU_FIELDS)
            state = _get(sample, "State", None)
            for key, method, size in (("quaternion_wxyz", "Quaternion", 4), ("accel_body", "Accelerometer", 3), ("gyro", "Gyroscope", 3), ("rpy", "Rpy", 3)):
                value = _get(state, method, None)
                ir[key] = np.full(size, np.nan, np.float32) if value is None else np.asarray(value, np.float32)
                if ir[key].shape != (size,): raise ValueError("incorrect IMU vector shape")
                ir[key + "_present"] = value is not None
                ir[key + "_finite"] = np.isfinite(ir[key])
            key = tuple(ir[k] for k in ("location", "ts_acquired_hw_us", "ts_sampled_us", "ts_ingested_us"))
            identity_valid = key[0] in (0, 1, 2) and any(key[1:])
            # Assembly and BodyState proxy may change on repeated publication;
            # compare actual sample payload/quality, not publication metadata.
            fingerprint = (ir["status"],) + tuple(ir[k].tobytes() for k in ("quaternion_wxyz", "accel_body", "gyro", "rpy"))
            previous = seen.get(key, {}) if identity_valid else {}
            if fingerprint in previous: continue
            contradiction = bool(previous)
            if contradiction:
                for prior_index in previous.values():
                    imus["identity_contradiction"][prior_index] = True
            if identity_valid:
                seen.setdefault(key, {})[fingerprint] = len(imus["location"])
            ir.update(sample_key=np.asarray(key, np.uint64), identity_valid=bool(identity_valid),
                      identity_contradiction=contradiction, sample_status_ok=ir["status"] == 1,
                      source_body_index=body_index, body_realtime_us=row["body_realtime_us"], body_realtime_present=row["body_realtime_present"])
            _append(imus, ir)
    return _arrays(bodies), _arrays(imus)


def write_npz(out_dir, bodies, imus, joint_names, unknown_default="mask", *, create=True):
    if unknown_default not in ("mask", "nan"): raise ValueError("invalid unknown_default")
    if create: Path(out_dir).mkdir()
    np.savez(Path(out_dir) / "body.npz", **bodies, joint_names=np.asarray(joint_names), unknown_default=np.asarray(unknown_default))
    np.savez(Path(out_dir) / "imu.npz", **imus, body_realtime_role=np.asarray("PROXY"), unknown_default=np.asarray(unknown_default))


def extract_mcap(paths, out_dir, schema_sha256, topic=DEFAULT_TOPIC, joint_names=None, sdk_python_path=None,
                 clock_manifest=None, unknown_default="mask", validate_crcs=True, roster_sources=None):
    from mcap.reader import make_reader
    import mcap
    expected = schema_sha256.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected): raise ValueError("invalid SHA256 pin")
    _roster(joint_names)
    if clock_manifest is not None and not isinstance(clock_manifest, dict): raise ValueError("clock manifest must be an object")
    Path(out_dir).mkdir()  # Exclusive reservation BEFORE expensive processing; failures leave no success receipt.
    supported = "validate_crcs" in inspect.signature(make_reader).parameters
    if validate_crcs and not supported: raise RuntimeError("installed MCAP reader does not support requested CRC validation")
    decoder = load_binding(sdk_python_path)
    binding_root = Path(sys.modules[decoder.__module__].__file__).parent
    binding_sources = {str(p.resolve()): sha256_file(p) for p in sorted(binding_root.glob("*.py"))}
    inputs = [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in paths]
    schemas = {}; selected_count = 0
    def records():
        nonlocal selected_count
        for path in paths:
            with open(path, "rb") as f:
                reader = make_reader(f, **({"validate_crcs": validate_crcs} if supported else {}))
                for schema, channel, msg in reader.iter_messages(topics=[topic], log_time_order=False):
                    if channel.topic != topic: continue
                    if schema is None: raise ValueError("selected topic has no schema")
                    schema_key = (str(path), schema.id)
                    if schema_key not in schemas:
                        digest = hashlib.sha256(schema.data).hexdigest()
                        if digest != expected: raise ValueError(f"selected schema hash mismatch: {digest}")
                        schemas[schema_key] = {"path": str(path), "id": schema.id, "name": schema.name, "encoding": schema.encoding, "sha256": digest}
                    selected_count += 1
                    yield decoder.GetRootAs(msg.data, 0), None
        if not selected_count: raise ValueError("no records for selected topic")
    b, i = extract_records(records(), joint_names)
    write_npz(out_dir, b, i, joint_names, unknown_default, create=False)
    counts = Counter(zip(i.get("location", []), i.get("status", [])))
    meta = {"format": "wtt.body-trace.v2", "schema_sha256": expected, "schemas": list(schemas.values()),
            "topic": topic, "inputs": inputs, "roster_sources": [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in (roster_sources or [])],
            "binding_path": str(binding_root.resolve()), "binding_source_sha256": binding_sources,
            "extractor_sha256": sha256_file(__file__), "mcap_version": getattr(mcap, "__version__", "unknown"),
            "crc": {"requested": validate_crcs, "reader_support": supported, "enabled": validate_crcs and supported,
                    "scope": "reader-supported CRCs encountered; absent CRCs cannot be verified; not a claim every file record was checked"},
            "body_rows": selected_count, "imu_rows": len(i.get("location", [])),
            "imu_location_status_counts": {f"{int(loc)}:{int(status)}": count for (loc, status), count in sorted(counts.items())},
            "protocol_versions": [int(x) for x in np.unique(b["protocol_version"])],
            "clock_manifest": clock_manifest, "timestamp_domains": "unknown unless explicitly declared; no cross-clock subtraction performed",
            "imu_body_realtime": "PROXY only; first retained publication, not sample time",
            "mask_semantics": "field_present=serialized table slot or struct accessor; timestamp present=nonzero observed; omitted scalar defaults do not prove invalidity; finite is not physical validity",
            "imu_identity": "location,hardware,sampled,ingested; any nonzero time required; missing identity rows preserved; distinct contradictory payloads preserved and ALL rows for that identity tagged; assembly/proxy excluded",
            "extraction_valid": not bool(np.any(i.get("identity_contradiction", []))),
            "errors": (["contradictory_imu_identity"] if np.any(i.get("identity_contradiction", [])) else []),
            "imu_contradiction_rows": int(np.count_nonzero(i.get("identity_contradiction", []))),
            "joint_validity": "31-slot structure includes uninstalled slots; q/dq/tau zeros not evidence of measurement; NaN/Inf retained with finite masks; error code raw",
            "applied_validity": "protocol2 nonzero session+sequence; protocol3 applied_key_valid and known source_type, zero keys permitted; no physical-send guarantee",
            "world_rotation_convention": "unverified; declaration required", "no_accelerometer_displacement": True}
    Path(out_dir, "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta
