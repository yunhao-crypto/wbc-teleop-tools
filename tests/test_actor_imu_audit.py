import json
import numpy as np
import pytest
from deployment.vita_s100_aorta_30dof.actor_imu_audit import audit_imu


def fixture(tmp_path,mutation=None,metadata_mutation=None):
    t=np.arange(2000)*.005
    data=dict(time=np.arange(2000,dtype=np.uint64)*5000+np.uint64(2**63+123),
              a=np.column_stack((t*0,t*0,9.81+np.sin(2*np.pi*6*t))),q=np.tile([1.,0,0,0],(len(t),1)),g=np.zeros((len(t),3)),status=np.ones(len(t),dtype=int))
    metadata=dict(schema='wtt.imu_analysis.v1',evidence='synthetic exact body-to-world identity and uint64 microsecond clock',
                  time={'key':'time','unit':'us','domain':'synthetic_clock','semantics':'sample','zero_is_missing':True},
                  acceleration={'key':'a','unit':'m/s^2','frame':'body'},quaternion={'key':'q','order':'wxyz','direction':'body_to_world'},
                  gyro={'key':'g','unit':'rad/s','frame':'body'},world_up_axis='z',sensor={'unique_samples':True,'status_key':'status','valid_status':1})
    if mutation:mutation(data)
    if metadata_mutation:metadata_mutation(metadata)
    p=tmp_path/'imu.npz';np.savez(p,**data);m=tmp_path/'metadata.json';m.write_text(json.dumps(metadata));return p,m


def test_band_rms_uint64_and_constant_gyro(tmp_path):
    p,m=fixture(tmp_path);r=audit_imu(p,m)
    assert r['timestamp_origin_raw']==2**63+123
    assert r['world_z_2_8hz_rms_m_s2']==pytest.approx(1/np.sqrt(2),abs=.002)
    assert r['segments'][0]['quaternion_gyro_correlation']==[None,None,None]
    assert r['segments'][0]['quaternion_gyro_rmse_rad_s']==[0,0,0]


def test_no_interpolation_over_large_gap(tmp_path):
    def change(d):d['time'][1000:]+=np.uint64(500000)
    p,m=fixture(tmp_path,change);r=audit_imu(p,m)
    assert len(r['segments'])==2
    assert r['requested_duration_coverage_fraction']<.96


@pytest.mark.parametrize('mutate,match',[
    (lambda m:m['time'].update(domain='unknown'),'unit/domain'),
    (lambda m:m['quaternion'].update(direction='world_to_body'),'quaternion'),
    (lambda m:m['acceleration'].update(unit='g'),'acceleration'),
    (lambda m:m['sensor'].update(unique_samples=False),'deduplication'),
])
def test_rejects_undeclared_conventions(tmp_path,mutate,match):
    p,m=fixture(tmp_path,metadata_mutation=mutate)
    with pytest.raises(ValueError,match=match):audit_imu(p,m)


def test_clock_reset_not_silently_sorted(tmp_path):
    def change(d):d['time'][1000]=d['time'][999]
    p,m=fixture(tmp_path,change)
    with pytest.raises(ValueError,match='nonmonotonic'):audit_imu(p,m)


def test_invalid_status_samples_excluded_before_finiteness_check(tmp_path):
    def change(d):d['status'][0]=0;d['a'][0]=np.nan
    p,m=fixture(tmp_path,change);r=audit_imu(p,m)
    assert r['selected_samples']==1999


def test_explicit_contradiction_exclusion_is_counted(tmp_path):
    def change(d):
        d['identity_contradiction']=np.zeros(2000,dtype=bool)
        d['identity_contradiction'][0]=True
        d['a'][0]=np.nan
    p,m=fixture(tmp_path,change,lambda meta:meta['sensor'].update(require_boolean_fields={'identity_contradiction':False}))
    assert audit_imu(p,m)['selected_samples']==1999


def test_insufficient_segment_has_null_metric_not_zero(tmp_path):
    p,m=fixture(tmp_path);r=audit_imu(p,m,end_s=2)
    assert r['status']=='INSUFFICIENT_COVERAGE' and r['world_z_2_8hz_rms_m_s2'] is None


def test_proxy_label_is_not_claimed_sample_time(tmp_path):
    p,m=fixture(tmp_path,metadata_mutation=lambda m:m['time'].update(semantics='body_state_proxy'))
    r=audit_imu(p,m)
    assert any('proxies' in x for x in r['limits'])
