"""Host-only v2 import/runtime hold smoke with synthetic weights and fake I/O."""
import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import yaml
from mimiclite_fixture import ROOT, source_model
from test_actor_mimiclite_v2 import v2_deployment, policy_sdk
from import_mimiclite_policy import import_policy
from deployment.vita_s100_aorta_30dof import actor_pico_runtime as runtime_module


def test_v2_import_runtime_stand_and_stale_reference(tmp_path, policy_sdk):
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
