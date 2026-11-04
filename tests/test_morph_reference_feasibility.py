import unittest
import numpy as np
from test_native_ankle_commands import contracts

class ReferenceFeasibilityTests(unittest.TestCase):
    def test_recorded_bad_reference_becomes_feasible_and_feedback_is_not_mutated(self):
        c=contracts()[2];q=np.tile(c.default,(12,1));before=q.copy()
        for p,r,sign in c.ankle.legs:
            q[:,p]=-.88139116764;q[:,r]=sign*.3
        before=q.copy()
        out,changed=c.ankle.project_reference(q)
        self.assertTrue(changed.all());self.assertTrue(np.array_equal(before,q))
        motor=c.ankle.to_policy(out)[:,c.ankle.slots]
        self.assertGreaterEqual(motor.min(),c.ankle.motor_limits[0])
        self.assertLessEqual(motor.max(),c.ankle.motor_limits[1])
        untouched=[i for i in range(28) if i not in c.ankle.slots]
        self.assertTrue(np.array_equal(out[:,untouched],q[:,untouched]))

    def test_legal_references_unchanged_and_extremes_close(self):
        c=contracts()[2];q=np.tile(c.default,(51,1))
        q[:,c.ankle.slots]=0
        out,changed=c.ankle.project_reference(q)
        self.assertFalse(changed.any());self.assertTrue(np.array_equal(out,q))
        rng=np.random.default_rng(9);q[:,c.ankle.slots]=rng.uniform(-1.5,1.5,(51,4))
        out,_=c.ankle.project_reference(q);motors=c.ankle.to_policy(out)
        self.assertTrue(np.all(motors[:,c.ankle.slots]>=c.ankle.motor_limits[0]))
        self.assertTrue(np.all(motors[:,c.ankle.slots]<=c.ankle.motor_limits[1]))
        for pose,target in zip(out,motors):
            recovered,error=c.ankle.to_virtual(target)
            self.assertLess(error,1e-7)
            np.testing.assert_allclose(recovered[c.ankle.slots],pose[c.ankle.slots],atol=1e-7)

    def test_projected_velocity_uses_correct_trajectory(self):
        c=contracts()[2];q=np.tile(c.default,(60,1))
        for p,r,sign in c.ankle.legs:
            q[:,p]=np.linspace(-.74,-.9,60);q[:,r]=sign*.3
        out,_=c.ankle.project_reference(q)
        v=np.gradient(out,.02,axis=0)
        actual=c.ankle.velocity_to_policy(out,v)
        # On the saturated tail, virtual and motor pitch/roll rates must be zero.
        np.testing.assert_allclose(actual[-10:,c.ankle.slots],0,atol=1e-10)

if __name__=='__main__':unittest.main()


def test_gmr_limits_are_applied_before_engine_construction_without_source_mutation():
    import json
    from pathlib import Path
    from deployment.vita_s100_aorta_30dof.actor_pico_gmr import _construct_engine
    path=Path(__file__).parents[1]/'third_party/motion_stream_bridge/motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'
    before=path.read_bytes();seen=[]
    def engine(**kwargs):
        config=json.loads(Path(kwargs['config']).read_text());seen.append(config);return config
    settings=dict(human_height_m=1.75,ankle_reference_limits=dict(pitch=[-.751,.414],roll=[-.3926987648,.3926987648]))
    config=_construct_engine(engine,path,settings)
    assert path.read_bytes()==before
    assert Path(config['robot_xml']).is_absolute() and Path(config['robot_xml']).is_file()
    for side in ('left','right'):
        assert config['joint_limits'][side+'_ankle_pitch_joint']==[-.751,.414]
    original=_construct_engine(engine,path,dict(human_height_m=1.75))
    assert original==json.loads(before)


def test_gmr_override_rejects_unsafe_or_wider_ranges():
    import pytest
    from pathlib import Path
    from deployment.vita_s100_aorta_30dof.actor_pico_gmr import _construct_engine
    path=Path(__file__).parents[1]/'third_party/motion_stream_bridge/motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'
    for pitch in [[-.88,.53],[.4,-.7],[float('nan'),.4]]:
        with pytest.raises(ValueError):
            _construct_engine(lambda **_:None,path,dict(human_height_m=1.75,ankle_reference_limits=dict(pitch=pitch,roll=[-.39,.39])))


def test_conservative_reference_box_stays_inside_coupled_motor_travel():
    c=contracts()[2]
    pitch,roll=np.meshgrid(np.linspace(-.751,.414,41),np.linspace(-.43625,.43625,41))
    q=np.tile(c.default,(pitch.size,1))
    for p,r,sign in c.ankle.legs:q[:,p]=pitch.ravel();q[:,r]=sign*roll.ravel()
    out,changed=c.ankle.project_reference(q)
    assert not changed.any()
    motors=c.ankle.to_policy(out)[:,c.ankle.slots]
    assert motors.min()>c.ankle.motor_limits[0] and motors.max()<c.ankle.motor_limits[1]


def test_guard_is_versioned_in_model_contract():
    import json
    from pathlib import Path
    from deployment.vita_s100_aorta_30dof.actor_morphmimic import MorphMimicContract,CONTRACT_KEY
    import locomotion_aorta.wbc_tracking_policy as sdk
    # Real exported contract, supplied by the checked-in source contract fixture
    # plus the same layout fields the exporter emits.
    from test_native_ankle_commands import ANKLE
    from deployment.vita_s100_aorta_30dof.actor_parallel_ankle import MotorThetaAnkle
    root=Path(__file__).parents[1]
    control=json.loads((root/'tests/fixtures/actor_pico/actor_g1_contract.json').read_text())
    ankle=MotorThetaAnkle(control['joint_names'],ANKLE)
    control['default_joint_pos']=ankle.to_policy(control['default_joint_pos']).tolist();control['ankle']=ANKLE
    for i in ankle.slots:
        control['joint_pos_limit_lower'][i],control['joint_pos_limit_upper'][i]=ANKLE['motor_limits']
    raw=dict(schema_version=1,policy_kind='morphmimic_vita02a_tracking',goal_anchor='reference',history_length=10,input_dim=1600,future_frames=10,
             velocity_rule='virtual_gradient_then_motor_jacobian',canonical_indices=[i for i in range(29) if i!=13],nominal_root_height=.891088,
             control=control,anchor_chain=[dict(joint='waist_yaw_joint'),dict(joint='waist_pitch_joint')])
    legacy=MorphMimicContract.from_metadata(sdk,{CONTRACT_KEY:json.dumps(raw)})
    assert not legacy.reference_feasibility_guard
    raw['reference_feasibility_guard']='virtual_box_and_coupled_motor_v1'
    candidate=MorphMimicContract.from_metadata(sdk,{CONTRACT_KEY:json.dumps(raw)})
    assert candidate.reference_feasibility_guard
