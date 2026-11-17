"""Actual WTT actor_inference schema fixtures, including offline ORT replay."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deployment.vita_s100_aorta_30dof.actor_control_audit import (
    ALIASES, CONTRACT_KEY, audit_log, sha256_file, uint64,
)
from deployment.vita_s100_aorta_30dof.actor_control_timeline import header_metadata

SID = 2**63 + 123
NAMES = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
         'left_knee_joint', 'left_A_achilles_joint', 'left_B_achilles_joint',
         'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
         'right_knee_joint', 'right_A_achilles_joint', 'right_B_achilles_joint',
         'waist_yaw_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
         'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
         'left_wrist_roll_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint',
         'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
         'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint']
ORDER = [ALIASES.get(n, n) for n in NAMES] + ['waist_roll_joint', 'neck_pitch_joint', 'neck_yaw_joint']


def header():
    return dict(record_type='header', robot_id='robot67', policy='mimiclite',
                model=dict(sha256='a'*64), motion=None)


def row(seq, echo=None):
    source = 1794896347000000 + seq * 20000
    applied = seq-1 if echo is None else echo
    feedback = dict(q_body=[0.]*31, dq_body=[0.]*31, root_wxyz=[1., 0., 0., 0.], gyro=[0.]*3,
                    applied_session_id=SID if applied else 0, applied_sequence=applied,
                    ts_first_output_us=1794896347000000 + applied*20000 + 19000 if applied else 0)
    targets = [dict(q=0., dq=0., tau=0., kp=20., kd=1.) for _ in ORDER]
    return dict(record_type='actor_inference', session_id=SID, sequence=seq,
                source_state_us=source, feedback=feedback, body_order=ORDER[:],
                stage='pico_tracking', raw_action=[0.]*28, decoded=dict(target=[0.]*28),
                observation=[0.]*1224, body_targets=targets, sent_body_targets=copy.deepcopy(targets),
                feedback_timing=dict(at_read=dict(source_age_ms=5.), before_publish=dict(source_age_ms=15.)),
                observation_ms=4., inference_ms=1., compute_ms=10., publish_ms=.5)


def log(tmp_path, rows, head=None, summary=True):
    records = [header() if head is None else head] + rows
    if summary:
        records.append(dict(record_type='summary', actions_sent=len(rows)))
    path = tmp_path/'actor.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return path


def fields(report, code):
    return {f['field'] for f in report['findings'] if f['code'] == code}


def test_origin_matching_not_reading_row_and_duplicate_echo(tmp_path):
    report = audit_log(log(tmp_path, [row(1), row(2), row(3, echo=1), row(4, echo=2)]))
    assert report['coverage']['command_rows'] == 4
    assert report['coverage']['matched_commands'] == 2
    assert report['coverage']['duplicate_echoes'] == 1
    assert report['timing']['source_to_first_output']['p50_ms'] == 19.
    assert report['coverage']['commands_without_observed_echo'] == 2
    assert report['coverage']['echo_absence_is_not_loss']
    assert report['status'] == 'PARTIAL'
    assert not report['safety_approval']


def test_conflicting_echo_and_session_isolation(tmp_path):
    rows = [row(1), row(2), row(3, echo=1), row(4)]
    rows[2]['feedback']['ts_first_output_us'] += 1
    rows[3]['feedback']['applied_session_id'] = SID+1
    report = audit_log(log(tmp_path, rows))
    assert 'conflicting_echo_timestamp' in fields(report, 'INVALID')
    assert report['coverage']['unmatched_echo_rows'] == 1
    assert report['status'] == 'INVALID'


@pytest.mark.parametrize('bad', [True, False, 1.0, 2**64, -1, '3'])
def test_uint64_exact(bad):
    assert not uint64(bad)
    assert uint64(SID)
    assert uint64(2**64-1)


def test_bad_echo_and_missing_not_zero(tmp_path):
    rows = [row(1), row(2)]
    rows[1]['feedback']['applied_session_id'] = float(SID)
    del rows[0]['compute_ms']
    del rows[1]['compute_ms']
    report = audit_log(log(tmp_path, rows))
    assert 'feedback_echo_uint64' in fields(report, 'INVALID')
    assert report['timing']['compute'] == dict(status='MISSING', count=0, p50_ms=None,
                                               p95_ms=None, p99_ms=None, max_ms=None)


@pytest.mark.parametrize('field,value', [('observation', [0.]*1223), ('raw_action', [float('nan')]*28),
                                         ('body_order', ['duplicate']*31)])
def test_invalid_vectors(tmp_path, field, value):
    r = row(1); r[field] = value
    report = audit_log(log(tmp_path, [r]))
    assert ('json_line' if field == 'raw_action' else field) in fields(report, 'INVALID')
    json.dumps(report, allow_nan=False)


def test_truncation_duplicate_headers_and_summary_count(tmp_path):
    report = audit_log(log(tmp_path, [row(1)], summary=False))
    assert report['status'] == 'PARTIAL'
    assert 'exactly_one_summary' in fields(report, 'MISSING')
    path = log(tmp_path, [row(1)])
    with path.open('a') as f:
        f.write(json.dumps(header())+'\n')
        f.write('{truncated')
    report = audit_log(path)
    assert {'json_line', 'exactly_one_header'} <= fields(report, 'INVALID')


def test_declared_timeline_only_and_wrong_robot_clock(tmp_path):
    h = header(); h['trace_clock_schema'] = header_metadata(True)
    r = row(1)
    names = ('feedback_read', 'observation_begin', 'observation_end', 'onnx_begin',
             'onnx_end', 'beforepublish', 'publishreturn')
    r['control_timeline'] = dict(schema='wtt.control_timeline.v1', units='ns',
                                clock_domain='host_monotonic', events=[
                                    dict(event=n, host_monotonic_ns=1_000_000+i*1_000_000)
                                    for i, n in enumerate(names)])
    report = audit_log(log(tmp_path, [r, row(2)], head=h))
    assert report['timing']['timeline_publish']['p50_ms'] == 1.
    assert report['timing']['timeline_read_to_publish']['p50_ms'] == 5.
    h['trace_clock_schema']['clock_domains']['ts_first_output_us'] = 'another_host_wall'
    report = audit_log(log(tmp_path, [r, row(2)], head=h))
    assert report['timing']['source_to_first_output']['count'] == 0
    assert 'verified_robot_clock_domain' in fields(report, 'MISSING')
    del h['trace_clock_schema']
    report = audit_log(log(tmp_path, [r], head=h))
    assert 'timeline_clock_declaration' in fields(report, 'INVALID')


def model(tmp_path, raw=False):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    helper, tensor = onnx.helper, onnx.TensorProto
    inputs = [helper.make_tensor_value_info('obs', tensor.FLOAT, [1, 1224])]
    if raw:
        inputs.append(helper.make_tensor_value_info('other', tensor.FLOAT, [1, 1]))
    node = helper.make_node('Constant', [], ['action'], value=helper.make_tensor('zero', tensor.FLOAT, [1, 28], [0.]*28))
    graph = helper.make_graph([node], 'wrapped', inputs, [helper.make_tensor_value_info('action', tensor.FLOAT, [1, 28])])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    terms = [dict(kind='root_ang_vel_history', width=30, history_steps=list(range(10))),
             dict(kind='projected_gravity_history', width=30, history_steps=list(range(10))),
             dict(kind='joint_pos_history', width=280, history_steps=list(range(10)), joint_names=NAMES, relative_to_default=True),
             dict(kind='joint_vel_history', width=280, history_steps=list(range(10)), joint_names=NAMES),
             dict(kind='prev_actions', width=84, steps=3)]
    contract = dict(input_dim=1224, action_dim=28, input_groups=['command', 'policy'],
                    action_semantics='default_plus_scale_times_raw_action_no_clipping',
                    motor_deployment=dict(joint_names=NAMES, default_position=[.1]*28,
                        action_scale=[.2]*28, joints={n: dict(kp=20., kd=1.) for n in NAMES},
                        observations=dict(command=[dict(kind='reference', width=520)], policy=terms)))
    helper.set_model_props(m, {CONTRACT_KEY: json.dumps(contract)})
    path = tmp_path/'policy.onnx'; onnx.save(m, path)
    return path


def replay_rows():
    rows = [row(i+1) for i in range(13)]
    for r in rows:
        r['decoded']['target'] = [.1]*28
        for field in ('body_targets', 'sent_body_targets'):
            for target in r[field][:28]:
                target['q'] = .1
        r['observation'][550:580] = [0., 0., -1.]*10
        r['observation'][580:860] = [-.1]*280
        # Intentionally unverified ankle histories must not fail serial checks.
        for step in range(10):
            for j in (4, 5, 10, 11):
                r['observation'][580+step*28+j] = 99.
    return rows


def test_numeric_replay_hash_history_defaults_and_target(tmp_path):
    path = model(tmp_path); h = header(); h['model']['sha256'] = sha256_file(path)
    report = audit_log(log(tmp_path, replay_rows(), head=h), path)
    assert report['replay']['status'] == 'PASS'
    assert report['replay']['count'] == 13
    assert report['contract']['joint_pos_history_serial']['count'] == 3
    assert report['contract']['joint_pos_history_serial']['max_abs_error'] < 1e-7
    assert report['coverage']['history_warmup_rows_excluded'] == 10
    assert not fields(report, 'INVALID')
    rows = replay_rows(); rows[-1]['observation'][1140] = .5
    report = audit_log(log(tmp_path, rows, head=h), path)
    assert 'previous_raw_actions_history' in fields(report, 'INVALID')
    rows = replay_rows(); rows[-1]['observation'][580] = 0.
    report = audit_log(log(tmp_path, rows, head=h), path)
    assert 'joint_pos_history_serial' in fields(report, 'INVALID')
    rows = replay_rows(); rows[-1]['raw_action'][0] = 1.
    report = audit_log(log(tmp_path, rows, head=h), path)
    assert {'raw_action_replay', 'affine_target'} <= fields(report, 'INVALID')


def test_wrong_model_hash_and_raw_graph_rejected(tmp_path):
    path = model(tmp_path)
    report = audit_log(log(tmp_path, replay_rows()), path)
    assert 'model_sha256_mismatch' in fields(report, 'INVALID')
    path = model(tmp_path, raw=True)
    report = audit_log(log(tmp_path, replay_rows()), path)
    assert report['replay']['status'] == 'UNSUPPORTED'
    assert report['replay']['count'] == 0


def test_invalid_source_stamp_and_missing_target_components(tmp_path):
    rows = [row(1), row(2)]
    rows[0]['source_state_us'] = True
    del rows[0]['body_targets'][0]['kp']
    rows[1]['position'] = [0.] * 27
    report = audit_log(log(tmp_path, rows))
    assert {'source_state_us', 'position_or_decoded.target'} <= fields(report, 'INVALID')
    assert 'body_targets' in fields(report, 'MISSING')
    assert report['timing']['source_to_first_output']['count'] == 0


def test_incomplete_echo_does_not_count_as_drop(tmp_path):
    r = row(1)
    del r['feedback']['ts_first_output_us']
    report = audit_log(log(tmp_path, [r]))
    assert 'feedback_echo' in fields(report, 'MISSING')
    assert report['coverage']['matched_commands'] == 0
    assert report['coverage']['commands_without_observed_echo'] == 1
    assert report['events']['firmware_ingress_drop_increments'] == 0


def test_producer_protocol3_zero_keys_sensors_and_order(tmp_path):
    from types import SimpleNamespace
    from deployment.vita_s100_aorta_30dof.actor_control_timeline import (
        ControlTimeline, association_metadata, sensor_timing, EVENT_NAMES)
    state = SimpleNamespace(protocol_version=3, runtime_id=9, applied_key_valid=True,
                            association_source_type=1, joint_stamps_us=[[1, 2, 0]],
                            imu_stamps_us=[[None, 2, 3, 4]])
    h = header(); h['audit_timing'] = header_metadata(True)
    r = row(1); r.update(session_id=0, sequence=0, runtime_id=9, command_source_type=1)
    r['feedback'].update(association_metadata(state))
    r['feedback'].update(applied_session_id=0, applied_sequence=0,
                         ts_first_output_us=r['source_state_us']+19000)
    r['sensor_timing'] = sensor_timing(state)
    ticks = iter(range(100, 107))
    timeline = ControlTimeline(SimpleNamespace(monotonic_ns=lambda: next(ticks)))
    for event in EVENT_NAMES: timeline.mark(event)
    r['control_timeline'] = timeline.as_dict()
    report = audit_log(log(tmp_path, [r], head=h))
    assert not fields(report, 'INVALID')
    assert report['coverage']['matched_commands'] == 1
    assert report['sensor_timing']['coverage']['joint_stamps_us_unobserved'] == 1
    r['feedback']['association_source_type'] = 3
    assert audit_log(log(tmp_path, [r], head=h))['coverage']['matched_commands'] == 0
    r['control_timeline']['events'][3]['host_monotonic_ns'] = 100
    assert 'timeline_event_order' in fields(audit_log(log(tmp_path, [r], head=h)), 'INVALID')
    h['trace_clock_schema'] = header_metadata(False)
    assert 'conflicting_clock_declarations' in fields(audit_log(log(tmp_path, [r], head=h)), 'INVALID')


def test_runtime_reset_sensor_regressions_and_exit(tmp_path):
    h = header(); h['audit_timing'] = header_metadata(True)
    h['audit_timing']['scope'] = 'supported_external_trial'
    rows = [row(1), row(2), row(3)]
    for i, r in enumerate(rows):
        r.update(runtime_id=1 if i < 2 else 2, command_source_type=1)
        r['feedback'].update(protocol_version=3, runtime_id=r['runtime_id'],
                             applied_key_valid=True, association_source_type=1)
        r['sensor_timing'] = dict(joint_stamps_us=[[10-i, 0, None]], imu_stamps_us=[[1.5, 2, 3, 4]])
    report = audit_log(log(tmp_path, rows, head=h))
    assert report['coverage']['matched_commands'] == 1
    assert report['coverage']['history_warmup_rows_excluded'] == 3
    assert 'joint_stamps_us_regression' in fields(report, 'WARNING')
    assert 'imu_stamps_us_integer' in fields(report, 'WARNING')
    assert 'summary_exit_confirmed' in fields(report, 'MISSING')
    assert not fields(report, 'INVALID')


@pytest.mark.parametrize('literal', ['NaN', 'Infinity', '-Infinity', '1e999'])
def test_nonfinite_anywhere(tmp_path, literal):
    path = log(tmp_path, [row(1)])
    with path.open('a') as stream:
        stream.write('{"record_type":"extra","nested":{"value":'+literal+'}}\n')
    assert 'json_line' in fields(audit_log(path), 'INVALID')


def test_shadow_and_events_and_missing_histories(tmp_path):
    h = header(); h['audit_timing'] = header_metadata(True)
    h['audit_timing']['scope'] = 'inference_only'
    records = [row(1), dict(record_type='policy_period_overrun', phase='infer', elapsed_ms=23),
               dict(record_type='policy_ticks_skipped', missed_ticks=3),
               dict(record_type='firmware_ingress_drop', previous=1, count=4),
               dict(record_type='firmware_ingress_drop', previous=4, count=0),
               dict(record_type='summary', actions_sent=0)]
    report = audit_log(log(tmp_path, records, head=h, summary=False))
    assert not fields(report, 'INVALID')
    assert report['coverage']['command_rows'] == 0
    assert report['coverage']['inference_rows'] == 1
    assert report['events']['policy_ticks_skipped'] == 3
    assert report['events']['firmware_ingress_drop_increments'] == 3
    assert report['events']['firmware_ingress_counter_resets'] == 1
    path = model(tmp_path); h = header(); h['model']['sha256'] = sha256_file(path)
    rows = replay_rows(); rows[-2]['raw_action'] = None
    rows[-3]['feedback']['q_body'] = None
    report = audit_log(log(tmp_path, rows, head=h), path)
    assert 'previous_raw_actions_history' in fields(report, 'MISSING')


def test_provenance_and_cli(tmp_path):
    path = log(tmp_path, [row(1), row(2)])
    report = audit_log(path)
    assert report['identity']['log_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    digest = report.pop('report_sha256')
    assert digest == hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    output = tmp_path/'audit.json'
    cli = Path(__file__).resolve().parents[1]/'scripts/audit_actor_control.py'
    run = subprocess.run([sys.executable, str(cli), str(path), '--output', str(output)], capture_output=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text())['schema'] == 'wtt.control_audit.v1'
    run = subprocess.run([sys.executable, str(cli), str(path), '--output', str(path)], capture_output=True)
    assert run.returncode == 2
