"""Offline same-model/same-NPZ SDK/direct-exec comparison. No robot I/O.

Passing the recorded-contract gate is NOT proof of equal contact conditions,
physical causal identification, model safety, or safe initial-state tolerances.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def _json(line):
    def reject(value):
        raise ValueError('Non-finite JSON constant: '+value)
    value = json.loads(line, parse_constant=reject)
    if not isinstance(value, dict): raise ValueError('Expected a JSON object')
    return value


def _finite(value, name, shape=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid '+name) from exc
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError('Invalid shape/nonfinite '+name)
    return a


def _integer(value, name):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError('Expected uint64 integer '+name)
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Missing/invalid SHA256 '+name)
    return value


def _rotation(q):
    q = _finite(q, 'quaternion')
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < .5): raise ValueError('Invalid quaternion norm')
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))


def control_path(header):
    value = header.get('control_path')
    for key in ('audit_timing', 'control_timeline', 'timing_capture'):
        block = header.get(key)
        if value is None and isinstance(block, dict):
            value = block.get('control_path')
    return value


def model_layout(path):
    import onnx
    data = Path(path).read_bytes()
    graph = onnx.load_model_from_string(data)
    meta = {v.key: v.value for v in graph.metadata_props}
    if 'mimiclite.tracking_contract.v1' not in meta:
        raise ValueError('This comparator requires a MimicLite wrapped-ONNX contract')
    c = _json(meta['mimiclite.tracking_contract.v1'])
    d = c['motor_deployment']
    offset, terms = 0, {}
    for group in c['input_groups']:
        for term in d['observations'][group]:
            terms[term['kind']] = (offset, term)
            offset += term['size']
    if offset != c['input_dim']:
        raise ValueError('Unsupported observation contract width')
    if len(graph.graph.input)!=1 or len(graph.graph.output)!=1:
        raise ValueError('Need the deployed wrapped single-input/single-output ONNX, not the raw training graph')
    for tensor,expected in ((graph.graph.input[0],[1,offset]),(graph.graph.output[0],[1,28])):
        kind=tensor.type.tensor_type
        if kind.elem_type!=onnx.TensorProto.FLOAT or [d.dim_value for d in kind.shape.dim]!=expected:
            raise ValueError('Unexpected deployed ONNX dtype/shape')
    return dict(sha256=hashlib.sha256(data).hexdigest(), width=offset, terms=terms, deployment=d)


def read_trial(path, layout):
    headers, starts, summaries, robot_hashes, frames = [], [], [], set(), {}
    body_order = None
    inference_count = 0
    log_digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            log_digest.update(line)
            if not line.strip():
                continue
            row = _json(line)
            kind = row.get('record_type')
            if kind == 'header': headers.append(row)
            if kind == 'motion_started': starts.append(row)
            if kind == 'summary': summaries.append(row)
            if isinstance(row.get('robot_config'), dict):
                robot_hashes.add(row['robot_config'].get('ankle_path_comparison_sha256'))
            if kind == 'actor_inference':
                inference_count += 1
                if _integer(row.get('actions_sent'),'actions_sent') != inference_count or row.get('action_sent') is False:
                    raise ValueError('Missing/discontinuous actual publication evidence')
                _integer(row.get('session_id'),'session_id');_integer(row.get('sequence'),'sequence')
            if kind != 'actor_inference' or row.get('stage') != 'motion_tracking' or row.get('motion_phase') != 'clip':
                continue
            tick = _integer(row.get('motion_tick'), 'motion_tick')
            if tick in frames:
                raise ValueError('Duplicate clip tick at line '+str(line_number))
            names = row.get('body_order')
            if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
                raise ValueError('Invalid named feedback order')
            if body_order is not None and names != body_order:
                raise ValueError('Feedback joint ordering changed within trial')
            body_order = names
            feedback = row.get('feedback', {})
            q = _finite(feedback.get('q_body'), 'q_body', (len(names),))
            dq = _finite(feedback.get('dq_body'), 'dq_body', (len(names),))
            root = _finite(feedback.get('root_wxyz'), 'root_wxyz', (4,))
            if np.linalg.norm(root) < .5:
                raise ValueError('Invalid feedback quaternion')
            root = root / np.linalg.norm(root)
            targets = row.get('sent_body_targets')
            if not isinstance(targets, list) or len(targets) != len(names):
                raise ValueError('Missing actual sent targets/gains')
            gains = _finite([[j['kp'], j['kd']] for j in targets], 'sent gains', (len(names), 2))
            target = _finite([j['q'] for j in targets], 'sent q', (len(names),))
            feedforward = _finite([[j['dq'],j['tau']] for j in targets], 'sent dq/tau', (len(names),2))
            obs = _finite(row.get('observation'), 'observation (use --record-observations)', (layout['width'],))
            reference = row.get('motion_reference', {})
            refq = _finite(reference.get('q'), 'motion_reference.q')
            refbase = _finite(reference.get('base'), 'motion_reference.base')
            refquat = _finite(reference.get('root_wxyz'), 'motion_reference.root_wxyz')
            if refq.ndim != 2 or not len(refq) or refbase.shape != (len(refq), 3) or refquat.shape != (len(refq), 4):
                raise ValueError('Malformed full reference window')
            ref_time = reference.get('ref_time')
            if isinstance(ref_time, bool) or not isinstance(ref_time, (int, float)) or not np.isfinite(ref_time) or ref_time < 0:
                raise ValueError('Invalid reference time')
            # Keep only invariant reference terms, NOT orientation relative to measured root.
            commands = {}
            for term_kind in ('ref_body_pos_future_local', 'ref_joint_pos_future', 'ref_joint_vel_future'):
                if term_kind in layout['terms']:
                    start, term = layout['terms'][term_kind]
                    commands[term_kind] = obs[start:start+term['size']].copy()
            term_pair = layout['terms'].get('ref_root_ori_future_b')
            if term_pair is not None:
                yaw = reference.get('yaw_offset')
                if isinstance(yaw,bool) or not isinstance(yaw,(float,int)) or not np.isfinite(yaw):
                    raise ValueError('Missing reference yaw_offset for orientation audit')
                period = layout['deployment']['control_period_s']
                ids = np.clip(round(ref_time/period)+np.asarray(layout['deployment']['future_steps']),0,len(refq)-1)
                off, term = term_pair
                ids = ids[term['future_indices']]
                cy, sy = np.cos(yaw), np.sin(yaw)
                yaw_matrix = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]])
                relative = _rotation(root).T @ (yaw_matrix @ _rotation(refquat[ids]))
                if not np.allclose(obs[off:off+term['size']],relative[:,:2,:].reshape(-1),rtol=0,atol=1e-5):
                    raise ValueError('Consumed reference orientation does not match logged reference/feedback')
            age = row.get('feedback_timing', {}).get('before_publish', {}).get('source_age_ms')
            compute = row.get('compute_ms')
            for label, value in [('source_age_ms', age), ('compute_ms', compute)]:
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0):
                    raise ValueError('Invalid '+label)
            frames[tick] = dict(q=q, dq=dq, root=root, gains=gains, target=target,feedforward=feedforward,
                refq=refq, refbase=refbase, refquat=refquat, ref_time=float(ref_time),
                commands=commands, age=age, compute=compute)
    if len(headers) != 1 or len(starts) != 1 or len(summaries) != 1:
        raise ValueError('Need exactly one header, motion_started, and summary')
    h, started, summary = headers[0], starts[0], summaries[0]
    if h.get('reference_source') != 'npz':
        raise ValueError('Pico/unpaired references cannot establish fixed-motion A/B comparability')
    expected_scope = 'supported_exec_trial' if control_path(h) == 'locomotion_exec' else 'supported_external_trial'
    if h.get('scope') != expected_scope or summary.get('mode') != 'live':
        raise ValueError('Need completed live trials, not shadow/offline predictions')
    if not inference_count or _integer(summary.get('actions_sent'),'summary.actions_sent') != inference_count:
        raise ValueError('Summary does not confirm recorded command count')
    if summary.get('exit_confirmed') is not True or summary.get('stop_reason') != 'motion_complete':
        raise ValueError('Incomplete motion or unconfirmed safe exit; do not label comparable')
    if _hash(h.get('model', {}).get('sha256'), 'model') != layout['sha256']:
        raise ValueError('Supplied ONNX does not match recorded model')
    if len(robot_hashes) != 1 or None in robot_hashes:
        raise ValueError('Missing/inconsistent robot configuration receipt')
    _hash(next(iter(robot_hashes)), 'robot configuration')
    lead = _integer(started.get('lead_frames'), 'lead_frames')
    motion = h.get('motion')
    if not isinstance(motion, dict): raise ValueError('Missing motion identity')
    _hash(motion.get('sha256'), 'NPZ')
    count = _integer(motion.get('policy_frames'), 'policy_frames')
    if count == 0 or len(frames) != count:
        raise ValueError('Incomplete clip frame coverage')
    frames = {k-lead: v for k, v in frames.items()}
    if min(frames) != 0 or max(frames) != count-1:
        raise ValueError('Clip frames do not cover 0..policy_frames-1')
    if not isinstance(h.get('robot_id'), str) or not h['robot_id']:
        raise ValueError('Missing robot identity')
    _hash(h.get('comparison_config_sha256'), 'actor semantic configuration')
    return dict(path=str(path), log_sha256=log_digest.hexdigest(), header=h, started=started, summary=summary,
                frames=frames, body_order=body_order, robot_hash=next(iter(robot_hashes)))


def _stats(values):
    x = [v for v in values if v is not None]
    return None if not x else dict(count=len(x), p50=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))


def compare_trials(a, b, layout, warmup_s=2., max_initial_joint_deg=3., max_initial_root_deg=3., max_initial_velocity_rad_s=.5):
    for value in (warmup_s, max_initial_joint_deg, max_initial_root_deg, max_initial_velocity_rad_s):
        if not np.isfinite(value) or value < 0: raise ValueError('Invalid comparison tolerance')
    ha, hb = a['header'], b['header']
    paths = [control_path(ha), control_path(hb)]
    if set(paths) != {'sdk', 'locomotion_exec'}:
        raise ValueError('Need explicitly declared sdk versus locomotion_exec; unknown/same paths are not this A/B')
    for key in ('policy', 'robot_id', 'comparison_config_sha256', 'contract', 'head_control'):
        if ha.get(key) != hb.get(key): raise ValueError('A/B confound: different '+key)
    if not isinstance(ha.get('contract'), dict) or any(k not in ha['contract'] for k in ('input_dim','action_dim','policy_hz')):
        raise ValueError('Missing policy/control contract')
    if ha['contract']['input_dim'] != layout['width'] or ha['contract']['action_dim'] != 28:
        raise ValueError('Header/model dimensions disagree')
    if a['robot_hash'] != b['robot_hash'] or a['body_order'] != b['body_order']:
        raise ValueError('A/B confound: robot settings or feedback coordinates differ')
    # Require explicit equal declarations; absent-on-both is not agreement.
    required_motion = ('sha256','policy_frames','period_s','joint_names','limit_corrections','end_hold_s','root_body_index','joint_space')
    for key in required_motion:
        if key not in ha['motion'] or key not in hb['motion'] or ha['motion'][key] != hb['motion'][key]:
            raise ValueError('Missing/different motion '+key)
    if ha['motion']['joint_space'] != 'virtual_pitch_roll':
        raise ValueError('This body-feedback comparison requires explicitly virtual_pitch_roll reference coordinates')
    for key in ('velocity_source','velocity_units','source_control_space'):
        if key in ha['motion'] or key in hb['motion']:
            if ha['motion'].get(key) != hb['motion'].get(key): raise ValueError('Different motion '+key)
    period = ha['motion']['period_s']
    if isinstance(period, bool) or not isinstance(period, (float,int)) or not np.isfinite(period) or period <= 0:
        raise ValueError('Invalid motion period')
    if not np.isclose(1/period, ha['contract']['policy_hz'], rtol=0, atol=1e-8):
        raise ValueError('Reference and policy frequency differ')
    common = sorted(set(a['frames']) & set(b['frames']))
    ia, ib = a['frames'][0], b['frames'][0]
    qdiff = float(np.rad2deg(abs(ia['q']-ib['q']).max()))
    dqdiff = float(abs(ia['dq']-ib['dq']).max())
    rdiff = float(np.rad2deg(2*np.arccos(np.clip(abs(np.dot(ia['root'], ib['root'])), 0, 1))))
    if qdiff > max_initial_joint_deg or dqdiff > max_initial_velocity_rad_s or rdiff > max_initial_root_deg:
        raise ValueError('A/B confound: initial measured state differs beyond explicitly requested matching tolerances')
    # Gains must agree on ALL shared clip rows, including warmup.
    for frame in common:
        if not np.allclose(a['frames'][frame]['gains'], b['frames'][frame]['gains'], rtol=0, atol=1e-5):
            raise ValueError('A/B confound: actual sent gains differ')
        if not np.allclose(a['frames'][frame]['feedforward'],b['frames'][frame]['feedforward'],rtol=0,atol=1e-7):
            raise ValueError('A/B confound: actual sent dq/tau feedforward differs')
    common = [k for k in common if k*period >= warmup_s]
    if len(common) < 2: raise ValueError('Not enough frames after warmup')
    joint_names = ha['motion']['joint_names']
    if not isinstance(joint_names, list) or len(set(joint_names)) != len(joint_names): raise ValueError('Invalid reference joint names')
    if any(n not in a['body_order'] for n in joint_names): raise ValueError('Reference/feedback coordinate names disagree')
    current_indices, references = [], []
    for frame in common:
        aa, bb = a['frames'][frame], b['frames'][frame]
        for kind in aa['commands']:
            if not np.allclose(aa['commands'][kind], bb['commands'][kind], rtol=0, atol=1e-5):
                raise ValueError('A/B confound: actually consumed '+kind+' differs on matched clip frames')
        ii = []
        for row in (aa, bb):
            index = int(round(row['ref_time']/period))
            if not np.isclose(index*period, row['ref_time'], atol=1e-8, rtol=0) or not 0 <= index < len(row['refq']):
                raise ValueError('Invalid current reference index')
            if row['refq'].shape[1] != len(joint_names): raise ValueError('Reference joint dimensions disagree')
            ii.append(index)
        if not np.allclose(aa['refq'][ii[0]], bb['refq'][ii[1]], atol=1e-8, rtol=0):
            raise ValueError('A/B confound: reference pose differs')
        # Check all consumed reference orientations, not only the current pose.
        offsets = np.asarray(layout['deployment']['future_steps'])
        qa = aa['refquat'][np.clip(ii[0]+offsets,0,len(aa['refquat'])-1)]
        qb = bb['refquat'][np.clip(ii[1]+offsets,0,len(bb['refquat'])-1)]
        if not np.allclose(_rotation(qa),_rotation(qb),rtol=0,atol=1e-7):
            raise ValueError('A/B confound: reference root orientation differs')
        references.append(aa['refq'][ii[0]])
    ref = np.asarray(references)
    reports = []
    for trial, path in zip((a,b), paths):
        selected = [trial['frames'][k] for k in common]
        ids = [trial['body_order'].index(n) for n in joint_names]
        q = np.array([r['q'][ids] for r in selected])
        target = np.array([r['target'][ids] for r in selected])
        reports.append(dict(path=trial['path'], log_sha256=trial['log_sha256'], control_path=path,
            reported_feedback_age_ms=_stats([r['age'] for r in selected]),
            compute_ms=_stats([r['compute'] for r in selected]),
            tracking_rmse_deg=dict(zip(joint_names,np.rad2deg(np.sqrt(np.mean((q-ref)**2,axis=0))).tolist())),
            serial_target_reference_rmse_deg={n:float(np.rad2deg(np.sqrt(np.mean((target[:,i]-ref[:,i])**2))))
                                              for i,n in enumerate(joint_names) if '_ankle_' not in n}))
    return dict(schema='wtt.control_compare.v1',status='COMPARABLE_RECORDED_CONTRACT',
        model_sha256=layout['sha256'], motion_sha256=ha['motion']['sha256'], common_frames=len(common),
        warmup_s=warmup_s, initial_state=dict(max_joint_difference_deg=qdiff,max_velocity_difference_rad_s=dqdiff,root_rotation_difference_deg=rdiff),
        trials=reports, limits=['Recorded-contract comparability is not physical causal identification or safety acceptance.',
          'Initial contact/support/load and actuator internal state were not measured; pose tolerances are comparison gates, not safety limits.',
          'Target-reference error is not measured tracking error.',
          'Age and compute statistics are reported telemetry; use audit_actor_control for clock/echo validation.',
          'No IMU displacement or vibration reduction is inferred from joint error; use separately timestamped IMU metrics.'])
