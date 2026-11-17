"""MimicLite export/SDK/Body31 tests: synthetic weights and fake IO only."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from mimiclite_fixture import ROOT, deployment, contract, source_model
from deployment.vita_s100_aorta_30dof import actor_mimiclite as ml
from deployment.vita_s100_aorta_30dof import actor_pico_runtime as runtime_module
from import_mimiclite_policy import wrap, validate_parity, import_policy


@pytest.fixture(scope='module')
def sdk():
    config=json.loads((ROOT/'config/actor_pico.json').read_text())
    return runtime_module.sdk_math(ROOT,config)


@pytest.fixture
def policy(sdk):
    profile=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())['body_profile']
    return ml.policy_class(sdk)(contract(),profile)


@pytest.fixture
def imported(tmp_path):
    config=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())
    # Synthetic fixtures must not inherit an operator's pinned real policy.
    config['model_ready']=False
    config['artifacts']['model'].update(sha256=None,bytes=None)
    config.pop('provenance',None)
    # Include real pinned non-model assets so the runtime hash guards execute.
    for key,item in config['artifacts'].items():
        if key=='model':continue
        target=tmp_path/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/item['path'],target)
    (tmp_path/'config').mkdir(exist_ok=True)
    (tmp_path/'config/actor_pico_mimiclite.json').write_text(json.dumps(config))
    path=tmp_path/'original.yaml'
    path.write_text(yaml.safe_dump(dict(motor_deployment=deployment())))
    path.with_suffix('.onnx').write_bytes(source_model())
    receipt=import_policy(tmp_path,path)
    return NS(root=tmp_path,yaml=path,receipt=receipt)


def make_runtime(imported,sdk,*,source='npz'):
    # Reuse actual pinned SDK math from this host, not an ARM extension.
    with patch.object(runtime_module,'sdk_math',return_value=sdk), \
            patch.object(runtime_module,'create_retargeter',return_value=NS()):
        return runtime_module.ActorPicoRuntime(imported.root,policy='mimiclite',backend='ort',reference_source=source)


def test_default_is_unconfigured_and_legacy_entry_is_removed(tmp_path):
    assert set(runtime_module.POLICY_CONFIGS)=={'sonic','supermimic','supermimic_op3','mimiclite'}
    config=json.loads((ROOT/'config/actor_pico_mimiclite.json').read_text())
    config['model_ready']=False
    (tmp_path/'config').mkdir()
    (tmp_path/'config/actor_pico_mimiclite.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='not configured'):
        runtime_module.load_actor_config(tmp_path,'mimiclite')
    with pytest.raises(ValueError,match='Unknown actor policy'):
        runtime_module.load_actor_config(ROOT,'morphmimic')


@pytest.mark.parametrize('batched',[False,True])
def test_wrapper_preserves_two_input_graph_normalizers_and_aux_output(batched):
    import onnxruntime as ort
    d=deployment();source=source_model(d,batched=batched,bias=30.)
    converted,metadata,shapes=wrap(source,d,'b'*64)
    result=validate_parity(source,converted,metadata,shapes)
    assert result['samples']==9 and result['max_abs_error']<1e-5
    session=ort.InferenceSession(converted,providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,814]
    assert len(session.get_inputs())==len(session.get_outputs())==1
    assert session.get_outputs()[0].shape==[1,28]
    c=ml.MimicLiteContract.from_metadata(None,session.get_modelmeta().custom_metadata_map)
    assert c.input_groups==('command','policy') and c.input_widths==(296,518)
    assert c.action_clip is None and c.history_length==17
    assert c.past_frames==8 and c.future_frames==4


@pytest.mark.parametrize('change',[
    lambda d:d.update(model_sha256='0'*64), lambda d:d.update(actuator_schema='old'),
    lambda d:d.update(reference_root_body='waist_pitch_link'),
    lambda d:d.update(control_space='virtual_pitch_roll'),
    lambda d:d.update(quaternion_order='xyzw'), lambda d:d.update(control_period_s=.01),
    lambda d:d.update(deployment_alpha=.9),
    lambda d:d['joint_names'].__setitem__(0,'neck_yaw_joint'),
    lambda d:d['observations']['policy'][0].update(history_steps=[1,2,3,4,5,8,16]),
    lambda d:d['observations']['command'][1].update(size=47),
    lambda d:d['observations'].update(priv=[]),
    lambda d:d['joints']['left_A_achilles_joint'].update(kp=float('nan')),
])
def test_import_rejects_changed_physics_observation_or_teacher_contract(change):
    d=deployment();change(d)
    with pytest.raises(ValueError):ml.metadata_contract(d,'a'*64,'b'*64)


def test_wrapper_rejects_semantic_relabelling_and_unpinned_provenance():
    raw=ml.metadata_contract(deployment(),'a'*64,'b'*64)
    for key,value in [('action_semantics','clip'),('input_widths',[518,296]),
                      ('input_groups',['policy','command']),('schema_version',True),
                      ('source_onnx_sha256','unreviewed')]:
        bad=copy.deepcopy(raw);bad[key]=value
        with pytest.raises(ValueError):
            ml.MimicLiteContract.from_metadata(None,{ml.CONTRACT_KEY:json.dumps(bad)})


def test_import_preserves_old_model_until_explicit_replacement(imported):
    initial=imported.receipt
    again=import_policy(imported.root,imported.yaml)
    assert initial['sha256']==again['sha256']
    imported.yaml.with_suffix('.onnx').write_bytes(source_model(bias=1.))
    with pytest.raises(RuntimeError,match='Preserving existing'):
        import_policy(imported.root,imported.yaml)
    assert json.loads((imported.root/'config/actor_pico_mimiclite.json').read_text())['artifacts']['model']['sha256']==initial['sha256']
    replaced=import_policy(imported.root,imported.yaml,replace=True)
    assert replaced['sha256']!=initial['sha256']


def test_import_and_runtime_reject_corrupt_model_and_operator_lock(imported):
    import fcntl
    with (imported.root/'build/operator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='operator is active'):
            import_policy(imported.root,imported.yaml)
    model=Path(imported.receipt['model']);model.write_bytes(model.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        runtime_module.load_actor_config(imported.root,'mimiclite')


def test_full_runtime_infers_and_emits_named_native_body31(imported,sdk):
    r=make_runtime(imported,sdk);c=r.model.contract
    assert r.policy.clip_target_positions is False and r.policy.clip_actions is False
    assert r.reference.steps==13 and r.reference.current_index==8 and r.reference.delay==.08
    assert r.reference_source=='npz' and r.retarget is None
    motion=r.hold_motion();assert motion.base[0,2]==c.nominal_root_height
    for _ in range(20):
        command,obs,row=r.infer_motion_command(motion,q=c.default,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))
        assert obs.shape==(814,) and len(command.joints)==31
        assert row['policy']=='mimiclite' and row['actions_sent']==0
        np.testing.assert_array_equal(row['raw_action'],r.policy.last_action)
    # Deliberately different order from SDK breadth-first and wire order.
    for name in ('left_wrist_roll_joint','left_wrist_pitch_joint','right_wrist_yaw_joint'):
        i=c.names.index(name);assert r.policy.order[r.policy.mapping[i]]==name
        assert command.joints[r.policy.mapping[i]].kp==float(np.float32(c.kp[i]))


def test_raw_action_history_and_unclipped_targets_survive_decode(policy):
    c=policy.contract;action=np.linspace(-35.,35.,28).astype(np.float32)
    command,row=policy.decode(action)
    np.testing.assert_array_equal(policy.last_action,action)
    np.testing.assert_allclose(policy.previous_actions[0],action)
    expected=c.policy_default+c.scale*action.astype(float)
    np.testing.assert_allclose(row['target'],expected)
    for i,slot in enumerate(policy.mapping):
        assert command.joints[slot].q==float(np.float32(expected[i]))
        assert command.joints[slot].tau==0.
    assert not any(row['clamped'])
    policy.reset();assert not policy.previous_actions.any() and not policy.signal_history
    for bad in (np.full(28,np.nan),np.full(28,1e100),np.ones(29)):
        with pytest.raises(ValueError):policy.decode(bad)
    policy.profile['ankle_command_space']='virtual_pitch_roll'
    with pytest.raises(ValueError,match='native motor_theta'):policy.decode(np.zeros(28))


def test_reference_offsets_and_sparse_history_are_not_other_policy_layouts(policy,sdk):
    c=policy.contract
    q=np.tile(c.default,(40,1));j=c.names.index('waist_yaw_joint');q[:,j]=np.arange(40)*.001
    base=np.tile([0.,0.,c.nominal_root_height],(40,1));base[:,0]=np.arange(40)*.01
    motion=sdk.Motion(q,base,np.tile([1.,0.,0.,0.],(40,1)))
    for tick in range(20):
        measured=c.default.copy();measured[j]+=.002*tick
        obs=policy.observe(motion,tick*.02,measured,np.zeros(28),[1,0,0,0],[tick*.01,0,0])
        gyro=obs[296:317].reshape(7,3)
        expected=np.maximum(tick-np.array([0,1,2,3,4,8,16]),0)*.01
        np.testing.assert_allclose(gyro[:,0],expected,atol=1e-8)
    # Current ref frame 19; root X local offsets include negative and positive time.
    np.testing.assert_allclose(obs[:24].reshape(8,3)[:,0],np.array([-8,-4,-2,0,1,2,3,4])*.01,atol=1e-7)
    assert obs[2]==pytest.approx(c.nominal_root_height)


def test_causal_pico_window_and_stale_input_gate(imported,sdk):
    r=make_runtime(imported,sdk,source='pico');c=r.model.contract
    for i in range(20):
        r.reference.append(1_000_000_000+i*20_000_000,c.default,[1,0,0,0],
            base=[i*.01,0,c.nominal_root_height],received_monotonic=100.+i*.02)
    motion=r.reference.motion()
    assert len(motion.q)==13
    assert motion.base[r.reference.current_index,0]==pytest.approx(.15)
    assert motion.base[-1,0]==pytest.approx(.19)
    body=np.zeros((24,7));body[:,6]=1.
    r.latest_pico=dict(received_monotonic=100.38,pico_timestamp_ns=1_380_000_000,raw_body_poses=body)
    command,obs,row=r.infer_command(q=c.default,dq=np.zeros(28),root=[1,0,0,0],
        gyro=np.zeros(3),now_monotonic=100.38)
    assert obs.shape==(814,) and len(command.joints)==31 and row['actions_sent']==0
    with pytest.raises(TimeoutError,match='stale'):
        r.infer_command(q=c.default,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3),now_monotonic=101.)
    held,flag=r.hold_stale_reference(motion,.3)
    assert flag
    np.testing.assert_allclose(held.base,np.tile(motion.base[8],(13,1)))


def test_infeasible_reference_is_rejected_without_mutation(policy):
    q=np.tile(policy.contract.default,(3,1))
    for p,r,sign in policy.contract.ankle.legs:q[:,p]=-.95;q[:,r]=sign*.43
    before=q.copy()
    with pytest.raises(ValueError,match='reference'):
        ml.checked_reference(policy.contract.ankle,q)
    np.testing.assert_array_equal(before,q)


def test_fixed_motion_bounded_windows_match_full_reference(imported,sdk):
    from deployment.vita_s100_aorta_30dof.actor_motion import load_motion
    r=make_runtime(imported,sdk);c=r.model.contract
    q=np.tile(c.default,(50,1));q[:,c.names.index('waist_yaw_joint')]+=.02*np.sin(np.arange(50)*.1)
    path=imported.root/'clip.npz'
    np.savez(path,joint_names=np.array(c.names),joint_pos=q,base_pos_w=np.tile([0.,0.,c.nominal_root_height],(50,1)),
             base_quat_w=np.tile([1.,0.,0.,0.],(50,1)),fps=np.array(50.))
    config=imported.root/'motion.json';config.write_text(json.dumps(dict(schema='vita_actor_motion_v1',path=str(path),end_hold_s=.04)))
    source=load_motion(r,config);source.begin(c.default,sdk.yaw_quaternion(.4))
    for tick in (0,1,source.lead_frames,40,source.total_ticks-1):
        source.tick=tick;window,t=source.sample();assert len(window.q)<=r.reference.steps
        args=(c.default,np.zeros(28),sdk.yaw_quaternion(.4),np.zeros(3),source.yaw_offset)
        r.policy.reset();full=r.policy.observe(source.prepared,tick*.02,*args)
        r.policy.reset();bounded=r.policy.observe(window,t,*args)
        np.testing.assert_allclose(full,bounded,atol=1e-6)


def test_training_side_numpy_interface_matches_wtt_term_order_and_history(policy,sdk):
    # Optional local checkout oracle, never a runtime dependency. The fallback
    # tests above still exercise a self-contained deployment package.
    training=ROOT.parent/'MimicLite/mimic-lite/mimic_lite/vita02a_deploy.py'
    if not training.is_file():pytest.skip('training-side checkout unavailable for independent parity oracle')
    sys.path.insert(0,str(training.parent))
    try:
        spec=importlib.util.spec_from_file_location('_mimiclite_training_oracle',training)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path.pop(0)
    c=policy.contract;oracle=module.MotorAngleInterface(c.deployment)
    rng=np.random.default_rng(17)
    from scipy.spatial.transform import Rotation
    refq=np.tile(c.default,(40,1));refq[:,c.names.index('waist_yaw_joint')]+=np.sin(np.arange(40)*.2)*.04
    pos=rng.normal(size=(40,3))*.02;pos[:,2]+=.891088
    quat=Rotation.from_euler('xyz',rng.normal(size=(40,3))*.15).as_quat()[:,[3,0,1,2]]
    motion=sdk.Motion(refq,pos,quat)
    for tick in range(24):
        q=c.default.copy();q[c.names.index('left_elbow_joint')]+=.03*np.sin(tick)
        dq=rng.normal(size=28)*.1;gyro=rng.normal(size=3)*.1
        root=Rotation.from_euler('xyz',[.1,-.2,tick*.03]).as_quat()[[3,0,1,2]]
        actual=policy.observe(motion,tick*.02,q,dq,root,gyro)
        ids=np.clip(tick+np.array(c.deployment['future_steps']),0,len(motion.q)-1)
        ref=dict(root_position=motion.base[ids],root_quaternion=motion.quat[ids],
                 joint_position=c.ankle.to_policy(motion.q[ids]))
        expected=oracle.observe(c.ankle.to_policy(q),c.ankle.velocity_to_policy(q,dq),root,gyro,ref)
        np.testing.assert_allclose(actual,np.concatenate([expected[k][0] for k in c.input_groups]),atol=2e-6,rtol=1e-6)
        action=rng.normal(size=28).astype(np.float32)*30
        policy.decode(action);oracle.decode(action)


def test_fake_live_session_keeps_native_config_and_cancels_on_exit(imported,sdk):
    from test_actor_pico_live import Clock,Loopback,robot_config
    from deployment.vita_s100_aorta_30dof.actor_pico_live import run_live
    from locomotion_aorta.pull import PullController
    r=make_runtime(imported,sdk);clock=Clock();transport=Loopback(r,clock)
    controller=PullController(group='default',client_id=0x5eed,_transport=transport)
    path=imported.root/'robot.yaml';path.write_text(yaml.safe_dump(robot_config(r)));before=path.read_bytes()
    args=NS(robot_config=path,robot_id='TEST-ONLY',startup_timeout=5.,seconds=.12,
            stand_only=True,auto_start=False,manual_pico=False)
    rows=[]
    try:
        with redirect_stdout(io.StringIO()):
            result=run_live(r,args,lambda row,obs=None:rows.append(row),controller,None,
                threading.Event(),lambda:(_ for _ in ()).throw(AssertionError('stand-only must not start Pico')),clock=clock)
        assert result['exit_confirmed']
        assert transport.commands
        assert [q.operation.value for q in transport.requests]==['PREPARE','COMMIT','CANCEL']
        assert path.read_bytes()==before
        assert all(len(c.joints)==31 for c in transport.commands)
    finally:controller.close()


@pytest.mark.parametrize('change',[
    lambda c:c['actions']['EXTERNAL']['params']['interpolation'].update(enabled=True),
    lambda c:c['actions']['EXTERNAL']['params'].update(allow_position_targets_outside_mechanical_range=False),
    lambda c:c['actions']['EXTERNAL']['params'].update(position_target_margin_rad=1.),
    lambda c:c['actions']['EXTERNAL']['params'].update(ankle_virtual_pd_intent=True),
    lambda c:c['parallel_ankle']['caps'].update(tau_motor_max=70.),
    lambda c:c['parallel_ankle']['caps'].update(qd_motor_max=20.),
])
def test_native_safety_gates_are_not_bypassed(imported,sdk,change):
    from test_actor_pico_live import robot_config
    from deployment.vita_s100_aorta_30dof.actor_pico_live import load_robot_config
    r=make_runtime(imported,sdk);config=robot_config(r);change(config)
    path=imported.root/'bad-robot.yaml';path.write_text(yaml.safe_dump(config));before=path.read_bytes()
    with pytest.raises(ValueError):load_robot_config(path,r)
    assert path.read_bytes()==before


def test_import_rejects_symlink_destination(imported):
    target=Path(imported.receipt['model']);original=target.read_bytes()
    target.unlink();other=imported.root/'preserve.onnx';other.write_bytes(original)
    target.symlink_to(other)
    with pytest.raises(ValueError,match='symlinked'):
        import_policy(imported.root,imported.yaml,replace=True)
    assert other.read_bytes()==original


def test_reference_window_and_model_size_are_pinned(imported,sdk):
    path=imported.root/'config/actor_pico_mimiclite.json';original=path.read_text()
    config=json.loads(original);config['artifacts']['model']['bytes']+=1;path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='hash mismatch'):runtime_module.load_actor_config(imported.root,'mimiclite')
    config=json.loads(original);config['reference_window']['past_frames']=1;path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='reference window'):make_runtime(imported,sdk)


def test_wrapper_rejects_unsupported_graph_abi_and_versions():
    import onnx
    for change in (
        lambda m:setattr(m,'ir_version',11),
        lambda m:setattr(m.opset_import[0],'version',25),
        lambda m:setattr(m.graph.input[0].type.tensor_type.shape.dim[-1],'dim_value',295),
        lambda m:setattr(m.graph.output[0],'name','not_action'),
    ):
        model=onnx.load_model_from_string(source_model());change(model)
        with pytest.raises((ValueError,onnx.checker.ValidationError)):
            wrap(model.SerializeToString(),deployment(),'b'*64)


def test_retired_policy_fails_before_any_launcher_services():
    import subprocess
    for args in (
        ['bash',str(ROOT/'scripts/start_actor_pico.sh'),'morphmimic','check'],
        ['bash',str(ROOT/'scripts/start_actor_motion.sh'),'morphmimic','check'],
        [sys.executable,str(ROOT/'scripts/actor_pico.py'),'check','--policy','morphmimic'],
    ):
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=10)
        assert result.returncode==2
        assert 'mimiclite' in result.stderr


def test_wrapper_helper_names_do_not_collide_with_unused_source_initializers():
    import onnx
    model=onnx.load_model_from_string(source_model())
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.zeros(1,np.float32),'__wtt_mimiclite_axis'))
    source=model.SerializeToString()
    converted,metadata,shapes=wrap(source,deployment(),'b'*64)
    assert validate_parity(source,converted,metadata,shapes)['max_abs_error']<1e-5


def test_nonfinite_source_does_not_replace_a_previously_pinned_model(imported):
    model=Path(imported.receipt['model']);before=model.read_bytes()
    config=imported.root/'config/actor_pico_mimiclite.json';before_config=config.read_bytes()
    imported.yaml.with_suffix('.onnx').write_bytes(source_model(bias=float('nan')))
    with pytest.raises(ValueError,match='Nonfinite'):
        import_policy(imported.root,imported.yaml,replace=True)
    assert model.read_bytes()==before and config.read_bytes()==before_config
