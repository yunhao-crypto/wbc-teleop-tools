import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from deployment.vita_s100_aorta_30dof.actor_control_timeline import (  # noqa: E402
    ControlTimeline, SCHEMA, header_metadata,
)


class FakeClock:
    def __init__(self):
        self.value = 1000

    def monotonic_ns(self):
        self.value += 10
        return self.value


def test_audit_event_order_and_fake_clock_domain():
    clock = FakeClock()
    timeline = ControlTimeline(clock)
    for event in ('feedback_read', 'observation_begin', 'observation_end',
                  'onnx_begin', 'onnx_end', 'beforepublish', 'publishreturn'):
        timeline.mark(event)
    payload = timeline.as_dict()
    assert payload['schema'] == SCHEMA
    assert [item['event'] for item in payload['events']] == [
        'feedback_read', 'observation_begin', 'observation_end',
        'onnx_begin', 'onnx_end', 'beforepublish', 'publishreturn']
    assert all(item['host_monotonic_ns'] >= 1000 for item in payload['events'])
    assert [item['host_monotonic_ns'] for item in payload['events']] == sorted(
        item['host_monotonic_ns'] for item in payload['events'])


def test_header_is_explicit_about_unavailable_sensor_timestamps():
    metadata = header_metadata(True)
    assert metadata['schema'] == 'wtt.control_timeline.v1'
    assert metadata['control_path'] == 'sdk'
    assert metadata['clock_domains']['source_state_us'] == 'robot_realtime_us'
    assert metadata['optional_sensor_fields']['imu_stamps_us']['domains']=='unknown'
    assert metadata['optional_sensor_fields']['joint_stamps_us']['columns']==['acquired_hw','ingested','assembled']
    assert 'physical_response_onset' in metadata['unavailable']


def test_optional_sensor_arrays_preserve_ints_and_unknowns():
    from types import SimpleNamespace
    import numpy as np
    from deployment.vita_s100_aorta_30dof.actor_control_timeline import sensor_timing, association_metadata
    state=SimpleNamespace(joint_stamps_us=np.array([[2**63+1,2,3]],dtype=np.uint64),
                          imu_values=np.array([[np.nan,np.inf,1.]]),protocol_version=3,
                          applied_key_valid=True,association_source_type=0,runtime_id=2**63+3)
    r=sensor_timing(state)
    assert r['joint_stamps_us'][0][0]==2**63+1
    assert r['imu_values']==[[None,None,1.]]
    assert r['imu_stamps_us'] is None
    a=association_metadata(state)
    assert a['applied_key_valid'] is True and a['association_source_type']==0
    assert a['ts_control_us'] is None


def test_off_is_metadata_only_and_does_not_capture():
    assert header_metadata(False)['enabled'] is False


def test_runtime_infer_decode_numeric_outputs_unchanged_with_fake_io():
    """Actual runtime method, inert policy/model/DTOs; never constructs an SDK client."""
    from types import SimpleNamespace
    import numpy as np
    from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
    class Policy:
        order = ['test_joint']
        def __init__(self): self.history=[]
        def observe(self,motion,ref_time,q,dq,root,gyro,yaw):
            self.history.append(float(q[0]))
            return np.array([sum(self.history[-3:]),dq[0]],dtype=np.float32)
        def decode(self,action):
            joint=SimpleNamespace(q=float(action[0]),dq=0.,tau=0.,kp=20.,kd=1.)
            return SimpleNamespace(joints=[joint]),{'position':action.tolist()}
    def run(enabled):
        runtime=ActorPicoRuntime.__new__(ActorPicoRuntime)
        runtime.policy=Policy();runtime.policy_name='test'
        runtime.model=SimpleNamespace(backend='fake',infer=lambda obs:obs[:1]*.3)
        published=[];observations=[]
        for step in range(12):
            timeline=ControlTimeline(FakeClock()) if enabled else None
            runtime._control_timeline=timeline
            if timeline:timeline.mark('feedback_read')
            command,obs,row=runtime.infer_motion_command(None,q=[step*.01],dq=[.5],root=[1,0,0,0],gyro=[0,0,0])
            if timeline:timeline.mark('beforepublish')
            published.append(row['body_targets'])  # fake publish of inert DTO
            if timeline:
                timeline.mark('publishreturn')
                assert [x['event'] for x in timeline.as_dict()['events']] == [
                    'feedback_read','observation_begin','observation_end','onnx_begin','onnx_end','beforepublish','publishreturn']
            observations.append(obs)
            assert row['actions_sent']==0 and row['control_handles_created']==0
        return published,np.asarray(observations)
    off,on=run(False),run(True)
    assert off[0]==on[0]
    np.testing.assert_array_equal(off[1],on[1])
