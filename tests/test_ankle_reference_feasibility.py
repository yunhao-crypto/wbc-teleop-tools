"""Shared legacy reference projection and GMR bounds; no hardware."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_native_ankle_commands import contracts
from deployment.vita_s100_aorta_30dof.actor_pico_gmr import _construct_engine

ROOT = Path(__file__).parents[1]


def test_projected_reference_closes_without_mutating_feedback():
    c=contracts()[0];q=np.tile(c.default,(12,1))
    for p,r,sign in c.ankle.legs:q[:,p]=-.88139116764;q[:,r]=sign*.3
    before=q.copy();out,changed=c.ankle.project_reference(q)
    assert changed.all();np.testing.assert_array_equal(before,q)
    untouched=[i for i in range(28) if i not in c.ankle.slots]
    np.testing.assert_array_equal(out[:,untouched],q[:,untouched])
    for pose,target in zip(out,c.ankle.to_policy(out)):
        recovered,error=c.ankle.to_virtual(target)
        assert error<1e-7
        np.testing.assert_allclose(recovered[c.ankle.slots],pose[c.ankle.slots],atol=1e-7)


def test_legal_reference_unchanged_and_projected_extremes_within_travel():
    c=contracts()[0];q=np.tile(c.default,(51,1));q[:,c.ankle.slots]=0
    out,changed=c.ankle.project_reference(q)
    assert not changed.any();np.testing.assert_array_equal(out,q)
    q[:,c.ankle.slots]=np.random.default_rng(9).uniform(-1.5,1.5,(51,4))
    out,_=c.ankle.project_reference(q);motor=c.ankle.to_policy(out)[:,c.ankle.slots]
    assert motor.min()>=c.ankle.motor_limits[0] and motor.max()<=c.ankle.motor_limits[1]


def test_projected_velocity_uses_correct_trajectory():
    c=contracts()[0];q=np.tile(c.default,(60,1))
    for p,r,sign in c.ankle.legs:q[:,p]=np.linspace(-.74,-.9,60);q[:,r]=sign*.3
    out,_=c.ankle.project_reference(q)
    v=c.ankle.velocity_to_policy(out,np.gradient(out,.02,axis=0))
    np.testing.assert_allclose(v[-10:,c.ankle.slots],0,atol=1e-10)


def test_gmr_limits_precede_construction_without_mutating_source():
    path=ROOT/'third_party/motion_stream_bridge/motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'
    before=path.read_bytes()
    class Engine(dict):
        pass
    def engine(**kwargs):return Engine(json.loads(Path(kwargs['config']).read_text()))
    config=_construct_engine(engine,path,dict(human_height_m=1.75,
        ankle_reference_limits=dict(pitch=[-.751,.414],roll=[-.3926987648,.3926987648])))
    assert path.read_bytes()==before
    assert Path(config['robot_xml']).is_absolute() and Path(config['robot_xml']).is_file()
    assert config['joint_limits']['left_ankle_pitch_joint']==[-.751,.414]
    assert _construct_engine(engine,path,dict(human_height_m=1.75))==json.loads(before)


def test_gmr_rejects_unsafe_or_wider_ranges():
    path=ROOT/'third_party/motion_stream_bridge/motion_stream_bridge/gmr_lite/configs/xrobot_to_vita02a.json'
    for pitch in [[-.88,.53],[.4,-.7],[float('nan'),.4]]:
        with pytest.raises(ValueError):
            _construct_engine(lambda **_:None,path,dict(human_height_m=1.75,
                ankle_reference_limits=dict(pitch=pitch,roll=[-.39,.39])))


def test_conservative_reference_box_is_inside_coupled_travel():
    c=contracts()[0]
    pitch,roll=np.meshgrid(np.linspace(-.751,.414,41),np.linspace(-.43625,.43625,41))
    q=np.tile(c.default,(pitch.size,1))
    for p,r,sign in c.ankle.legs:q[:,p]=pitch.ravel();q[:,r]=sign*roll.ravel()
    out,changed=c.ankle.project_reference(q)
    assert not changed.any()
    motor=c.ankle.to_policy(out)[:,c.ankle.slots]
    assert motor.min()>c.ankle.motor_limits[0] and motor.max()<c.ankle.motor_limits[1]
