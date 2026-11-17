"""Frozen sensor/selected-reference shadow fixtures; no robot I/O or model import."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .actor_control_compare import _json, _finite, _integer, _hash, control_path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare_fixture(log_path, output_path, layout):
    if layout['width']!=1224:raise ValueError('Shadow fixture v1 currently supports MimicLite hist10/refvel V2 only')
    before=sha256(log_path);header=None;body_order=None;rows=[];headers=0
    period=layout['deployment']['control_period_s'];future=np.asarray(layout['deployment']['future_steps'])
    with Path(log_path).open() as stream:
        for line in stream:
            if not line.strip():continue
            row=_json(line)
            if row.get('record_type')=='header':header=row;headers+=1
            if row.get('record_type')!='actor_inference' or row.get('stage')!='motion_tracking':continue
            if header is None:raise ValueError('Missing leading header')
            names=row.get('body_order')
            if not isinstance(names,list) or not names or len(set(names))!=len(names):raise ValueError('Invalid body ordering')
            if body_order is not None and names!=body_order:raise ValueError('Body ordering changed')
            body_order=names
            f=row.get('feedback',{});reference=row.get('motion_reference',{})
            if reference.get('velocity_space')!='virtual_pitch_roll_rad_s':
                raise ValueError('Need actual dense reference rates: record fixed NPZ with --audit-timing --record-observations')
            rq=_finite(reference.get('q'),'reference q')
            if rq.ndim!=2 or rq.shape[1]!=28:raise ValueError('Invalid reference q')
            rv=_finite(reference.get('joint_velocity'),'reference velocity',rq.shape)
            rp=_finite(reference.get('base'),'reference position',(len(rq),3))
            rr=_finite(reference.get('root_wxyz'),'reference quaternion',(len(rq),4))
            ref_time=reference.get('ref_time');yaw=reference.get('yaw_offset')
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in (ref_time,yaw)):raise ValueError('Invalid reference time/yaw')
            center=int(round(ref_time/period))
            if not 0<=center<len(rq) or not np.isclose(center*period,ref_time,atol=1e-8,rtol=0):raise ValueError('Invalid reference center')
            ids=np.clip(center+future,0,len(rq)-1)
            rows.append(dict(frame_id=_integer(row.get('policy_tick'),'policy_tick'),
                source_state_us=_integer(row.get('source_state_us'),'source_state_us'),
                q_body=_finite(f.get('q_body'),'q_body',(len(names),)),dq_body=_finite(f.get('dq_body'),'dq_body',(len(names),)),
                root_wxyz=_finite(f.get('root_wxyz'),'root_wxyz',(4,)),gyro=_finite(f.get('gyro'),'gyro',(3,)),
                reference_q_virtual=rq[ids],reference_dq_virtual=rv[ids],reference_root_position=rp[ids],reference_root_wxyz=rr[ids],
                yaw_offset=float(yaw),teacher_raw_action=_finite(row.get('raw_action'),'raw_action',(28,)),
                expected_observation=_finite(row.get('observation'),'observation',(1224,))))
    if headers!=1 or header.get('reference_source')!='npz' or control_path(header)!='sdk':raise ValueError('Need one SDK fixed-NPZ header')
    if _hash(header.get('model',{}).get('sha256'),'model')!=layout['sha256']:raise ValueError('ONNX/header model mismatch')
    motion=header.get('motion',{});_hash(motion.get('sha256'),'NPZ')
    if motion.get('joint_space')!='virtual_pitch_roll' or len(motion.get('joint_names',[]))!=28:raise ValueError('Missing named virtual reference coordinates')
    if len(rows)<=10:raise ValueError('Need >10 consecutive motion rows to exclude history initialization')
    ids=[r['frame_id'] for r in rows]
    if any(b!=a+1 for a,b in zip(ids,ids[1:])):raise ValueError('Shadow source has non-consecutive policy ticks')
    after=sha256(log_path)
    if before!=after:raise ValueError('Input log changed during extraction')
    metadata=dict(schema='wtt.shadow_fixture.v1',model_sha256=layout['sha256'],source_log_sha256=before,
        motion_sha256=motion['sha256'],body_order=body_order,reference_joint_names=motion['joint_names'],
        future_steps=future.tolist(),control_period_s=period,
        history_semantics='prepend each sensor sample; after inference set previous raw action from teacher_raw_action, NOT candidate output',
        warmup_rows=10,scope='sensor and selected-reference observation construction plus inference; excludes NPZ loading/reference selection and physical dynamics')
    arrays={key:np.array([r[key] for r in rows],dtype=np.uint64 if key in ('frame_id','source_state_us') else float) for key in rows[0]}
    arrays['evaluation_mask']=np.arange(len(rows))>=10
    arrays['metadata_json']=np.asarray(json.dumps(metadata,sort_keys=True))
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return dict(schema='wtt.shadow_fixture_receipt.v1',fixture_sha256=sha256(out),path=str(out),rows=len(rows),evaluated_rows=len(rows)-10,metadata=metadata)


def compare_shadow(fixture_path, trace_path, observation_atol=1e-6, action_atol=5e-5):
    for tolerance in (observation_atol,action_atol):
        if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid tolerance')
    fixture_hash=sha256(fixture_path);trace_hash=sha256(trace_path)
    with np.load(fixture_path,allow_pickle=False) as f, np.load(trace_path,allow_pickle=False) as t:
        metadata=_json(str(f['metadata_json'].item()))
        if metadata.get('schema')!='wtt.shadow_fixture.v1':raise ValueError('Unsupported fixture schema')
        if str(t['fixture_sha256'].item())!=fixture_hash:raise ValueError('Trace used a different frozen fixture')
        if str(t['model_sha256'].item())!=metadata['model_sha256']:raise ValueError('Trace used a different model')
        if str(t['control_path'].item())!='locomotion_exec':raise ValueError('Expected locomotion_exec shadow trace')
        for key in ('actions_sent','control_handles_created'):
            value=t[key]
            if value.shape!=() or value.dtype.kind not in 'iu' or int(value)!=0:raise ValueError('Shadow must declare no actual commands or control handles')
        if t['frame_id'].dtype.kind not in 'iu' or not np.array_equal(f['frame_id'],t['frame_id']):raise ValueError('Frame identities/order differ')
        expected=f['expected_observation'];expected_action=f['teacher_raw_action']
        actual=_finite(t['observation'],'observation',expected.shape);action=_finite(t['raw_action'],'raw_action',expected_action.shape)
        _finite(expected,'fixture observation');_finite(expected_action,'fixture action')
        mask=f['evaluation_mask']
        if mask.dtype!=np.bool_ or mask.shape!=(len(expected),) or not mask.any():raise ValueError('Invalid fixture evaluation mask')
        if len(expected)<=10 or not np.array_equal(mask,np.arange(len(expected))>=10):raise ValueError('Fixture warmup must exclude exactly the first 10 rows')
        obs_error=np.abs(actual[mask]-expected[mask]);action_error=np.abs(action[mask]-expected_action[mask])
        obs_max=float(obs_error.max());act_max=float(action_error.max())
        passed=obs_max<=observation_atol and act_max<=action_atol
    if sha256(fixture_path)!=fixture_hash or sha256(trace_path)!=trace_hash:raise ValueError('Input changed during shadow comparison')
    return dict(schema='wtt.shadow_compare.v1',status='NUMERICAL_PARITY' if passed else 'NUMERICAL_MISMATCH',
        fixture_sha256=fixture_hash,trace_sha256=trace_hash,model_sha256=metadata['model_sha256'],
        evaluated_rows=int(mask.sum()),warmup_rows=10,max_observation_error=obs_max,max_raw_action_error=act_max,
        observation_atol=observation_atol,action_atol=action_atol,robot_safety_acceptance=False,
        scope=metadata['scope'],limits=['Teacher-forced histories are not a closed-loop test.',
        'Fixture starts at motion tracking; ten warmup rows are excluded to refill all sensor/action histories.',
        'No inference runtime/deadline or actual robot motion is validated by numerical parity.'])
