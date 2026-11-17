import json
import numpy as np
import pytest
from test_actor_control_compare import sample
from deployment.vita_s100_aorta_30dof.actor_control_shadow import prepare_fixture,compare_shadow,sha256


def make_fixture(tmp_path,sample,mutation=None):
    rows,layout=sample
    for i,row in enumerate(r for r in rows if r.get('record_type')=='actor_inference'):
        row.update(policy_tick=i,source_state_us=2**63+123+i*20000,raw_action=[0.]*28)
        row['feedback']['gyro']=[0,0,0]
        row['motion_reference'].update(joint_velocity=[[0.]*28]*13,velocity_space='virtual_pitch_roll_rad_s')
    if mutation:mutation(rows)
    p=tmp_path/'sdk.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=tmp_path/'fixture.npz';receipt=prepare_fixture(p,out,layout)
    return out,receipt


def candidate(tmp_path,fixture,mutation=None):
    with np.load(fixture,allow_pickle=False) as f:
        data=dict(frame_id=f['frame_id'],observation=f['expected_observation'],raw_action=f['teacher_raw_action'],
                  fixture_sha256=np.asarray(sha256(fixture)),model_sha256=np.asarray(json.loads(str(f['metadata_json'].item()))['model_sha256']),
                  actions_sent=np.asarray(0),control_handles_created=np.asarray(0),control_path=np.asarray('locomotion_exec'))
    if mutation:mutation(data)
    p=tmp_path/'exec-shadow.npz';np.savez_compressed(p,**data);return p


def test_exact_parity_and_frozen_uint64_source(tmp_path,sample):
    f,receipt=make_fixture(tmp_path,sample);t=candidate(tmp_path,f);r=compare_shadow(f,t)
    assert r['status']=='NUMERICAL_PARITY' and r['evaluated_rows']==2
    assert r['robot_safety_acceptance'] is False
    with np.load(f,allow_pickle=False) as data:
        assert data['source_state_us'][0]==np.uint64(2**63+123)
        assert data['reference_dq_virtual'].shape==(12,8,28)
    assert receipt['fixture_sha256']==sha256(f)


def test_warmup_exclusion_is_explicit(tmp_path,sample):
    f,_=make_fixture(tmp_path,sample)
    def mutate(d):d['observation'][:10]=1e3
    t=candidate(tmp_path,f,mutate);assert compare_shadow(f,t)['status']=='NUMERICAL_PARITY'


def test_numeric_mismatch(tmp_path,sample):
    f,_=make_fixture(tmp_path,sample)
    t=candidate(tmp_path,f,lambda d:d['observation'].__setitem__((11,520),.01))
    assert compare_shadow(f,t)['status']=='NUMERICAL_MISMATCH'


@pytest.mark.parametrize('mutation,match',[
    (lambda d:d.update(fixture_sha256=np.asarray('0'*64)),'different frozen fixture'),
    (lambda d:d.update(model_sha256=np.asarray('0'*64)),'different model'),
    (lambda d:d.update(actions_sent=np.asarray(1)),'no actual commands'),
    (lambda d:d.update(control_handles_created=np.asarray(1)),'no actual commands'),
    (lambda d:d.update(frame_id=d['frame_id'][::-1]),'Frame identities'),
    (lambda d:d.update(observation=d['observation'][:-1]),'shape/nonfinite'),
])
def test_shadow_rejects_identity_and_actuation_claims(tmp_path,sample,mutation,match):
    f,_=make_fixture(tmp_path,sample);t=candidate(tmp_path,f,mutation)
    with pytest.raises(ValueError,match=match):compare_shadow(f,t)


def test_missing_dense_reference_velocity_not_differenced(tmp_path,sample):
    with pytest.raises(ValueError,match='dense reference rates'):
        make_fixture(tmp_path,sample,lambda r:r[3]['motion_reference'].pop('velocity_space'))


def test_output_is_exclusive(tmp_path,sample):
    f,_=make_fixture(tmp_path,sample)
    with pytest.raises(FileExistsError):make_fixture(tmp_path,sample)
