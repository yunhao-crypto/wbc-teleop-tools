import copy
import json
from pathlib import Path
import numpy as np
import pytest
from deployment.vita_s100_aorta_30dof.actor_control_compare import read_trial, compare_trials
from test_actor_mimiclite_v2 import v2_deployment
from deployment.vita_s100_aorta_30dof.actor_mimiclite import ALIASES


@pytest.fixture
def sample():
    d=v2_deployment(); names=[ALIASES.get(n,n) for n in d['joint_names']]
    terms={};offset=0
    for group in ['command','policy']:
        for term in d['observations'][group]:
            terms[term['kind']]=(offset,term);offset+=term['size']
    layout=dict(sha256='a'*64,width=1224,terms=terms,deployment=d)
    h=dict(record_type='header',policy='mimiclite',scope='supported_external_trial',robot_id='test67',control_path='sdk',reference_source='npz',model={'sha256':'a'*64},comparison_config_sha256='b'*64,head_control=None,
           contract={'input_dim':1224,'action_dim':28,'policy_hz':50,'required_ankle_command_space':'motor_theta','required_ankle_virtual_pd_intent':False},
           motion={'sha256':'c'*64,'policy_frames':12,'period_s':.02,'joint_names':names,'limit_corrections':[],'end_hold_s':1.,'root_body_index':0,'joint_space':'virtual_pitch_roll','velocity_source':'explicit_npz_virtual_rad_s','velocity_units':'rad/s'})
    rows=[h,dict(record_type='robot_configuration',robot_config={'ankle_path_comparison_sha256':'d'*64}),dict(record_type='motion_started',lead_frames=2)]
    obs=np.zeros(1224);start,term=terms['ref_root_ori_future_b'];obs[start:start+term['size']]=np.tile([1,0,0,0,1,0],8)
    start,term=terms['ref_body_pos_future_local'];obs[start:start+term['size']]=np.tile([0,0,.9],8)
    for i in range(12):
        rows.append(dict(record_type='actor_inference',stage='motion_tracking',motion_phase='clip',motion_tick=i+2,body_order=names,actions_sent=i+1,session_id=2**63+9,sequence=i+1,
                         feedback={'q_body':[0.]*28,'dq_body':[0.]*28,'root_wxyz':[1,0,0,0]},
                         sent_body_targets=[{'q':0.,'dq':0.,'tau':0.,'kp':100.,'kd':2.} for _ in names],observation=obs.tolist(),
                         motion_reference={'q':[[0.]*28]*13,'base':[[0,0,.9]]*13,'root_wxyz':[[1,0,0,0]]*13,'ref_time':.16,'yaw_offset':0.},
                         feedback_timing={'before_publish':{'source_age_ms':20.}},compute_ms=10.))
    rows.append(dict(record_type='summary',mode='live',exit_confirmed=True,stop_reason='motion_complete',actions_sent=12))
    return rows,layout


def pair(tmp_path,sample,mutate=None):
    rows,layout=sample;b=copy.deepcopy(rows);b[0]['control_path']='locomotion_exec';b[0]['scope']='supported_exec_trial'
    if mutate:mutate(b)
    paths=[]
    for name,data in [('a',rows),('b',b)]:
        p=tmp_path/(name+'.jsonl');p.write_text(''.join(json.dumps(r)+'\n' for r in data));paths.append(p)
    return read_trial(paths[0],layout),read_trial(paths[1],layout),layout


def test_comparable_descriptive_not_safety(tmp_path,sample):
    a,b,l=pair(tmp_path,sample)
    result=compare_trials(a,b,l,warmup_s=0)
    assert result['status']=='COMPARABLE_RECORDED_CONTRACT'
    assert result['common_frames']==12
    assert result['trials'][0]['reported_feedback_age_ms']['p95']==20
    assert not any('_ankle_' in n for n in result['trials'][0]['serial_target_reference_rmse_deg'])
    assert any('not physical causal' in x for x in result['limits'])


@pytest.mark.parametrize('mutation,match',[
    (lambda x:x[0].update(control_path='sdk',scope='supported_external_trial'),'explicitly declared'),
    (lambda x:x[0].update(comparison_config_sha256='e'*64),'comparison_config_sha256'),
    (lambda x:x[0].update(robot_id='different'),'robot_id'),
    (lambda x:x[0]['motion'].update(sha256='e'*64),'motion sha256'),
    (lambda x:x[3]['feedback']['q_body'].__setitem__(0,.2),'initial measured'),
    (lambda x:x[3]['feedback']['dq_body'].__setitem__(0,1.),'initial measured'),
    (lambda x:x[3]['sent_body_targets'][0].update(kp=101.),'sent gains'),
    (lambda x:x[3]['sent_body_targets'][0].update(tau=1.),'feedforward'),
])
def test_rejects_confound(tmp_path,sample,mutation,match):
    with pytest.raises(ValueError,match=match):
        a,b,l=pair(tmp_path,sample,mutation);compare_trials(a,b,l,warmup_s=0)


def test_reference_velocity_mismatch(tmp_path,sample):
    _,layout=sample;offset,_=layout['terms']['ref_joint_vel_future']
    def mutate(rows):rows[4]['observation'][offset]=.123
    a,b,l=pair(tmp_path,sample,mutate)
    with pytest.raises(ValueError,match='ref_joint_vel_future'):compare_trials(a,b,l,warmup_s=0)


def test_orientation_must_match_recorded_feedback(tmp_path,sample):
    _,l=sample;offset,_=l['terms']['ref_root_ori_future_b']
    with pytest.raises(ValueError,match='orientation'):
        pair(tmp_path,sample,lambda x:x[3]['observation'].__setitem__(offset,.2))


@pytest.mark.parametrize('mutation,match',[
    (lambda x:x[0].update(reference_source='pico'),'Pico'),
    (lambda x:x[-1].update(exit_confirmed=False),'unconfirmed'),
    (lambda x:x[-1].update(stop_reason='operator_stop'),'Incomplete'),
    (lambda x:x[0]['model'].update(sha256='e'*64),'ONNX'),
    (lambda x:x.pop(4),'publication evidence'),
    (lambda x:x[4].update(motion_tick=x[3]['motion_tick']),'Duplicate'),
    (lambda x:x[3].pop('observation'),'observation'),
    (lambda x:x[3]['feedback'].update(root_wxyz=[0,0,0,0]),'quaternion'),
    (lambda x:x[3]['observation'].__setitem__(0,float('nan')),'Non-finite'),
])
def test_invalid_or_incomplete_trace(tmp_path,sample,mutation,match):
    with pytest.raises(ValueError,match=match):pair(tmp_path,sample,mutation)


def test_missing_fields_do_not_equal_agreement(tmp_path,sample):
    sample[0][0]['motion'].pop('root_body_index')
    a,b,l=pair(tmp_path,sample)
    with pytest.raises(ValueError,match='Missing/different motion'):compare_trials(a,b,l,warmup_s=0)


def test_missing_timing_is_not_zero(tmp_path,sample):
    for row in sample[0]:
        if row.get('record_type')=='actor_inference':row.pop('feedback_timing');row.pop('compute_ms')
    a,b,l=pair(tmp_path,sample);r=compare_trials(a,b,l,warmup_s=0)
    assert r['trials'][0]['reported_feedback_age_ms'] is None
    assert r['trials'][0]['compute_ms'] is None


def test_no_frames_after_warmup(tmp_path,sample):
    a,b,l=pair(tmp_path,sample)
    with pytest.raises(ValueError,match='after warmup'):compare_trials(a,b,l)
