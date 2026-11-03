#!/usr/bin/env python3
"""Freeze one on-robot Pico session into a compact replay fixture (.npz).

Inputs are the two logs every live SuperMimic/SONIC session writes:
``outputs/actor_pico_gmr_input_*.jsonl`` (every Pico frame the GMR worker
processed: source stamp, local receipt time, raw body poses, GMR qpos) and
``outputs/actor_pico_live_*.jsonl`` (control ticks with robot feedback and,
while following, the snapshot each tick consumed).

The fixture keeps every processed Pico frame (bit-exact float32 body poses)
and the control ticks around each following segment, so an offline replay can
reproduce the original arrival/processing timing (frames, GMR durations,
control-tick times) without the robot. See actor_pico_session_replay.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

STAGES = {'prepare_stand': 0, 'pico_tracking': 1}
QPOS_HEAD = 300


def load_frames(path):
    header, rows = None, []
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row['record_type'] == 'header':
                header = row
            elif row['record_type'] == 'gmr_frame':
                rows.append(row)
    if header is None or not rows:
        raise ValueError('GMR input log needs a header and frames')
    body = np.array([r['body_poses'] for r in rows], dtype=np.float64)
    if not np.array_equal(body.astype(np.float32).astype(np.float64), body):
        raise ValueError('Body poses are not float32 values; refusing a lossy fixture')
    frames = dict(
        frame_stamp_ns=np.array([r['timestamp_ns'] for r in rows], dtype=np.int64),
        frame_received_monotonic=np.array([r['received_monotonic'] for r in rows]),
        frame_body=body.astype(np.float32),
        # Robot GMR output for the first frames only: enough to prove the
        # offline GMR replay is bit-exact without doubling the fixture size.
        frame_qpos_head=np.array([r['qpos'] for r in rows[:QPOS_HEAD]], dtype=np.float64),
        frame_task_error=np.array([r['gmr_task_error'] for r in rows], dtype=np.float32))
    if 'available_monotonic' in rows[0]:
        frames['frame_available_monotonic'] = np.array([r['available_monotonic'] for r in rows])
    return header, frames


def load_ticks(path, margin_ticks):
    ticks, events, body_order, model = [], [], None, None
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get('record_type')
            if kind == 'header':
                model = row.get('model')
            elif kind in ('tracking_started', 'tracking_paused', 'pico_trigger_toggle'):
                events.append(dict(record_type=kind, actions_sent=row.get('actions_sent')))
            elif kind == 'actor_inference' and row.get('stage') in STAGES:
                body_order = body_order or row['body_order']
                feedback, reference = row['feedback'], row.get('pico_reference') or {}
                ticks.append(dict(
                    policy_tick=row['policy_tick'], stage=STAGES[row['stage']],
                    state_us=row['source_state_us'], pico_age_s=row.get('pico_age_s', np.nan),
                    retarget_ms=row.get('retarget_ms', np.nan),
                    pico_stamp_ns=reference.get('timestamp_ns') or 0,
                    q=feedback['q_body'], dq=feedback['dq_body'], root=feedback['root_wxyz'],
                    gyro=feedback['gyro'], target=[j['q'] for j in row['sent_body_targets']]))
    stage = np.array([t['stage'] for t in ticks])
    keep = np.zeros(len(ticks), bool)
    for i in np.flatnonzero(stage == 1):
        keep[max(0, i-margin_ticks):i+margin_ticks+1] = True
    ticks = [t for t, k in zip(ticks, keep) if k]
    arrays = dict(
        tick_policy_tick=np.array([t['policy_tick'] for t in ticks], dtype=np.int32),
        tick_stage=np.array([t['stage'] for t in ticks], dtype=np.int8),
        tick_state_us=np.array([t['state_us'] for t in ticks], dtype=np.int64),
        tick_pico_age_s=np.array([t['pico_age_s'] for t in ticks]),
        tick_retarget_ms=np.array([t['retarget_ms'] for t in ticks], dtype=np.float32),
        tick_pico_stamp_ns=np.array([t['pico_stamp_ns'] for t in ticks], dtype=np.int64),
        tick_q_body=np.array([t['q'] for t in ticks], dtype=np.float32),
        tick_dq_body=np.array([t['dq'] for t in ticks], dtype=np.float32),
        tick_root_wxyz=np.array([t['root'] for t in ticks], dtype=np.float32),
        tick_gyro=np.array([t['gyro'] for t in ticks], dtype=np.float32),
        tick_target_q_body=np.array([t['target'] for t in ticks], dtype=np.float32))
    return arrays, dict(body_order=body_order, actor_model=model, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gmr-input', type=Path, required=True)
    parser.add_argument('--actor-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session', required=True, help='free-form session identifier stored in the metadata')
    parser.add_argument('--margin-ticks', type=int, default=50, help='stand ticks kept around each following segment')
    args = parser.parse_args()
    header, frames = load_frames(args.gmr_input)
    ticks, info = load_ticks(args.actor_log, args.margin_ticks)
    # Robot GMR duration per processed frame, where a control tick logged it.
    index = {int(s): i for i, s in enumerate(frames['frame_stamp_ns'])}
    robot_ms = np.full(len(index), np.nan, dtype=np.float32)
    for stamp, ms in zip(ticks['tick_pico_stamp_ns'], ticks['tick_retarget_ms']):
        if stamp in index and np.isfinite(ms):
            robot_ms[index[int(stamp)]] = ms
    # Control ticks that consumed a snapshot sit at receipt + snapshot age.
    received = {s: frames['frame_received_monotonic'][i] for s, i in index.items()}
    ticks['tick_monotonic'] = np.array([received.get(int(s), np.nan)+a if s else np.nan
                                        for s, a in zip(ticks['tick_pico_stamp_ns'], ticks['tick_pico_age_s'])])
    meta = dict(schema='vita_pico_session_fixture_v1', session=args.session,
                retargeting=header['retargeting'], model=header['model'],
                gmr_input=args.gmr_input.name, actor_log=args.actor_log.name, **info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta)), frame_robot_retarget_ms=robot_ms,
                        **frames, **ticks)
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
                          frames=len(index), ticks=len(ticks['tick_stage']),
                          following_ticks=int((ticks['tick_stage'] == 1).sum()))))


if __name__ == '__main__':
    main()
