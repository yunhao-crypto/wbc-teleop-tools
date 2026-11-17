"""Host-only v2 import/runtime hold smoke with synthetic weights and fake I/O."""
import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import yaml
from mimiclite_fixture import ROOT, source_model
from test_actor_mimiclite_v2 import v2_deployment, policy_sdk
from import_mimiclite_policy import import_policy
from deployment.vita_s100_aorta_30dof import actor_pico_runtime as runtime_module
from deployment.vita_s100_aorta_30dof.actor_motion import lead_in_with_velocity


@pytest.fixture
def v2_runtime(tmp_path, policy_sdk):
    config=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())
    config['model_ready']=False
    config['artifacts']['model'].update(sha256=None,bytes=None)
    config.pop('provenance',None)
    for key,item in config['artifacts'].items():
        if key=='model': continue
        target=tmp_path/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/item['path'],target)
    (tmp_path/'config').mkdir(exist_ok=True)
    (tmp_path/'config/actor_pico_mimiclite.json').write_text(json.dumps(config))
    d=v2_deployment()
    source=tmp_path/'hist10.yaml'
    source.write_text(yaml.safe_dump({'motor_deployment':d}))
    source.with_suffix('.onnx').write_bytes(source_model(d))
    receipt=import_policy(tmp_path,source)
    assert receipt['input_widths']==[520,704]
    with patch.object(runtime_module,'sdk_math',return_value=policy_sdk), patch.object(runtime_module,'create_retargeter',return_value=SimpleNamespace()):
        runtime=runtime_module.ActorPicoRuntime(tmp_path,policy='mimiclite',backend='ort',reference_source='npz')
    return runtime


def test_v2_import_runtime_stand_and_stale_reference(v2_runtime, policy_sdk):
    runtime=v2_runtime
    c=runtime.model.contract
    motion=runtime.hold_motion()
    np.testing.assert_array_equal(motion.joint_vel,0.)
    for _ in range(12):
        command,obs,row=runtime.infer_motion_command(motion,q=c.default,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))
        assert obs.shape==(1224,) and len(command.joints)==31 and row['actions_sent']==0
        np.testing.assert_array_equal(obs[296:520],0.)
    # Stale reference uses the same explicit zero-rate contract.
    hold=policy_sdk.Motion(np.repeat(motion.q,13,axis=0),np.repeat(motion.base,13,axis=0),np.repeat(motion.quat,13,axis=0))
    hold.joint_vel=np.ones_like(hold.q)
    stale,flag=runtime.hold_stale_reference(hold,.3)
    assert flag
    np.testing.assert_array_equal(stale.joint_vel,0.)


def test_v2_lead_in_retains_source_rates_and_encodes_transition(v2_runtime):
    runtime=v2_runtime
    c=runtime.model.contract
    motion=runtime.hold_motion()
    q=c.default.copy()
    joint=c.names.index('waist_yaw_joint')
    q[joint]+=.2
    # The underlying SDK returns a pose-only Motion, reproducing the live bug.
    cfg=runtime.config['body_profile']['lead_in']
    assert not hasattr(motion.lead_in(q,[1,0,0,0],cfg),'joint_vel')
    lead=lead_in_with_velocity(motion,q,[1,0,0,0],cfg,c.period)
    n=len(lead.q)-len(motion.q)
    expected=np.gradient(lead.q[:n+1],c.period,axis=0)[:n]
    np.testing.assert_array_equal(lead.joint_vel[:n],expected)
    np.testing.assert_array_equal(lead.joint_vel[n:],motion.joint_vel)
    assert np.max(np.abs(lead.joint_vel[:n,joint]))>0
    for i in [0,n//2,n]:
        _,obs,_=runtime.infer_motion_command(lead,q=q,dq=np.zeros(28),
            root=[1,0,0,0],gyro=np.zeros(3),ref_time=i*c.period)
        assert obs.shape==(1224,) and np.isfinite(obs).all()
        if i==n//2:
            assert np.max(np.abs(obs[296:520]))>0
    # Existing source velocities are not replaced by a derivative or zeros.
    motion.joint_vel[:]=.123
    prepared=lead_in_with_velocity(motion,q,[1,0,0,0],cfg,c.period)
    np.testing.assert_array_equal(prepared.joint_vel[-1],motion.joint_vel[0])
    del motion.joint_vel
    missing=lead_in_with_velocity(motion,q,[1,0,0,0],cfg,c.period)
    with pytest.raises(ValueError,match='velocity is missing'):
        runtime.infer_motion_command(missing,q=q,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))


def test_v2_pose_snapshot_preserves_explicit_velocity(v2_runtime):
    from deployment.vita_s100_aorta_30dof.actor_pico_live import manual_reference
    runtime=v2_runtime
    c=runtime.model.contract
    n=runtime.reference.steps
    rates=np.full((n,28),.123)
    snapshot=dict(state='ready',received_monotonic=100.,retarget_ms=.1,gmr_task_error=0.,
        q=np.tile(c.default,(n,1)),base=np.tile([0,0,c.nominal_root_height],(n,1)),
        quat=np.tile([1,0,0,0],(n,1)),joint_vel=rates)
    motion,_=manual_reference(runtime,snapshot,100.)
    np.testing.assert_array_equal(motion.joint_vel,rates)
    _,obs,_=runtime.infer_motion_command(motion,q=c.default,dq=np.zeros(28),
        root=[1,0,0,0],gyro=np.zeros(3),ref_time=runtime.reference.ref_time)
    assert np.max(np.abs(obs[296:520]))>0
    for invalid in [np.zeros((1,28)),np.full((n,28),np.nan)]:
        with pytest.raises(ValueError,match='joint_vel must match'):
            manual_reference(runtime,dict(snapshot,joint_vel=invalid),100.)
    del snapshot['joint_vel']
    missing,_=manual_reference(runtime,snapshot,100.)
    with pytest.raises(ValueError,match='velocity is missing'):
        runtime.infer_motion_command(missing,q=c.default,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))


@pytest.mark.parametrize('scenario',['auto_start','direct_stand','pause','pico_loss'])
@pytest.mark.parametrize('weights',['synthetic','delivery'])
def test_v2_live_lead_in_lifecycle(v2_runtime, scenario, weights):
    # Reuse the SDK loopback transport; no robot connection or control output.
    from test_actor_pico_live import LiveTests
    case=LiveTests(methodName='runTest')
    runtime=v2_runtime
    if weights=='delivery':
        if not (ROOT/'artifacts/mimiclite/policy.onnx').is_file():
            pytest.skip('Optional delivered hist10 model is not installed')
        runtime=runtime_module.ActorPicoRuntime(ROOT,policy='mimiclite',backend='ort',reference_source='npz')
        if runtime.model.contract.input_dim!=1224:
            pytest.skip('Optional installed MimicLite model is not hist10 v2')
    case.runtime=runtime
    case.setUp()
    try:
        if scenario=='auto_start':
            case.test_online_entry_runs_onnx_during_lead_in_from_straight_knees()
        elif scenario=='direct_stand':
            case.test_stand_only_never_switches_to_pico_tracking()
        else:
            worker,request=case.manual_trial()
            case.args.seconds=4.2
            original_snapshot=worker.snapshot
            def snapshot():
                value=original_snapshot()
                if value['state']=='ready':
                    # Production stream path computes rates on retained source
                    # samples before selecting the policy observation window.
                    times=np.arange(40)*runtime.model.contract.period
                    positions=np.repeat(runtime.model.contract.default[None],len(times),axis=0)
                    positions[:,runtime.model.contract.names.index('waist_yaw_joint')]+=.01*times
                    value['stream']=dict(t=times,q=positions,
                        base=np.tile([0,0,runtime.model.contract.nominal_root_height],(len(times),1)),
                        quat=np.tile([1,0,0,0],(len(times),1)),
                        clock_offset_s=case.clock.monotonic()-times[-1]-runtime.stream.playback_margin_s)
                return value
            worker.snapshot=snapshot
            if scenario=='pause':
                request=lambda:{50:'connect',155:'track',170:'toggle'}.get(len(case.transport.commands))
            else:
                def after_publish(command):
                    if len(case.transport.commands)>160:
                        worker.fault=lambda:dict(state='failed',error='injected disconnect')
                case.transport.after_publish=after_publish
            result=case.run_trial(manual_pico=worker,poll_command=request)
            event='tracking_paused' if scenario=='pause' else 'pico_loss_stand'
            assert any(row.get('record_type')==event for row in case.rows)
            assert any(row.get('stage')=='pico_tracking' for row in case.rows)
            assert result['exit_confirmed']
            case.assert_cancelled()
    finally:
        case.doCleanups()
