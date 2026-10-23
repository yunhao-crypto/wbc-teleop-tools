"""No-SDK integration footprint for the resident native runtime factory."""
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from deployment.vita_s100_aorta_30dof.onboard_runtime import OnboardDualPolicyRuntime, native_bundle_metadata
from deployment.vita_s100_aorta_30dof.onboard_output import load_output_envelopes
from deployment.vita_s100_aorta_30dof.onboard_runtime import OUTPUT_ENVELOPES_SHA256
from deployment.common.model_contract import sha256_file
from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState,VITA_CONTROLLER_JOINT_NAMES

class Runner:
    def __init__(self,width):self.width=width;self.calls=[];self.resets=0;self.step_index=np.array([0]);self.fail=False
    def run(self,observation):
        self.calls.append(np.asarray(observation).copy())
        if self.fail:raise RuntimeError('injected inference failure')
        self.step_index+=1
        return np.zeros(self.width,np.float32)
    def reset(self):self.resets+=1;self.step_index[:]=0
    def get_providers(self):return ['InertFakeRunner']

class RuntimeTests(unittest.TestCase):
    def make(self):
        motion,velocity=Runner(30),Runner(28)
        runtime=OnboardDualPolicyRuntime(contracts_path=ROOT/'artifacts/policy_contracts.json',
            skeleton_path=ROOT/'artifacts/preprocessing_skeleton.npz',motion_runner=motion,velocity_runner=velocity)
        return runtime,motion,velocity

    def test_original_constructor_and_warmup_reset(self):
        runtime,motion,velocity=self.make()
        runtime.last_motion_action[:]=1;runtime.last_velocity_action[:]=1
        report=runtime.warmup()
        self.assertEqual(len(motion.calls),20);self.assertEqual(len(velocity.calls),20)
        self.assertEqual(motion.calls[0].shape,(1,3126));self.assertEqual(velocity.calls[0].shape,(1,990))
        self.assertEqual(motion.resets,2);self.assertEqual(motion.step_index[0],0)
        self.assertTrue(np.all(runtime.last_motion_action==0));self.assertTrue(np.all(runtime.last_velocity_action==0))
        self.assertIsNone(runtime.last_raw_action);self.assertTrue(report['no_robot_action'])
        json.dumps(report)
        runtime.close();self.assertEqual(motion.resets,2) # injected resources remain caller-owned

    def test_failed_warmup_also_resets(self):
        runtime,motion,velocity=self.make();motion.fail=True
        with self.assertRaisesRegex(RuntimeError,'injected inference'):runtime.warmup(2)
        self.assertEqual(motion.resets,2);self.assertIsNone(runtime.last_raw_action)
        with self.assertRaises(ValueError):runtime.warmup(0)

    def test_pinned_envelopes_match_and_are_copied(self):
        runtime,_,_=self.make()
        frozen=load_output_envelopes(ROOT/'artifacts/output_envelopes.json',expected_sha256=OUTPUT_ENVELOPES_SHA256)
        for mode in ('motion','velocity'):
            actual=runtime.command_envelope(frozen['base'],mode)
            for item in fields(actual):
                x,y=getattr(actual,item.name),getattr(frozen[mode],item.name)
                if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
                else:self.assertEqual(x,y)
            actual.lower_rad[:]=0
            self.assertFalse(np.all(runtime.command_envelope(frozen['base'],mode).lower_rad==0))
        with self.assertRaisesRegex(ValueError,'Live interface differs'):
            runtime.command_envelope(replace(frozen['base'],max_kp=501),'motion')
        with self.assertRaises(ValueError):runtime.command_envelope(frozen['base'],'unknown')

    def test_composite_recorder_metadata_is_truthful(self):
        manifest=ROOT/'outputs/model_runtime_manifest.json'
        for backend,artifact_name in [('ort','velocity'),('hbm','native_velocity_hbm')]:
            bundle=native_bundle_metadata(manifest,velocity_backend=backend,deployment_scope='suspended_validation')
            self.assertEqual(bundle.manifest_path,manifest.resolve())
            self.assertEqual(bundle.artifacts['motion'].path,manifest.resolve())
            self.assertEqual(bundle.artifacts['motion'].sha256,sha256_file(manifest))
            self.assertEqual(bundle.artifacts['velocity'].sha256,sha256_file(bundle.artifacts['velocity'].path))
            self.assertEqual(bundle.deployment_scope.value,'suspended_validation')
            json.dumps({'bundle':str(bundle.manifest_path),'policies':{
                name:{'path':str(bundle.artifacts[name].path),'sha256':bundle.artifacts[name].sha256}
                for name in ('velocity','motion')}})

    def test_cpu_reference_and_original_policy_step_interface(self):
        runtime,motion,velocity=self.make()
        stand=runtime.stand_step();self.assertEqual(stand.target_joint_pos.shape,(30,))
        reference=runtime.create_online_reference()
        with np.load(ROOT/'reference/flow_inputs_480.npz',allow_pickle=False) as data:
            body=data['body_poses'][:80];sample_times=data['sample_times'][:80];wall=data['wall_times'][:80]
            state=VitaRobotState(VITA_CONTROLLER_JOINT_NAMES,data['joint_pos'][79],data['joint_vel'][79],
                data['root_quat_wxyz'][79],data['root_ang_vel_b'][79])
        for i in range(80):self.assertTrue(reference.store_pico(body[i],sample_time=float(sample_times[i]),current_time=float(wall[i]),frame_index=i))
        self.assertTrue(reference.has_future_sequence(10,include_derivative_tail=True))
        reference.reset_root_anchor();self.assertEqual(reference.current_root_rot().shape,(4,))
        self.assertEqual(runtime.step_velocity(state,(0,0,0)).target_joint_pos.shape,(30,))
        self.assertEqual(runtime.step_motion(state,reference,reference_yaw_alignment_wxyz=(1,0,0,0)).target_joint_pos.shape,(30,))
        self.assertEqual(motion.calls[-1].shape,(1,3126))

    def test_factory_owns_backend_and_closes_on_failure(self):
        from deployment.vita_s100_aorta_30dof import onboard_backend
        created=[]
        class Backend:
            def __init__(self,*a,**kw):self.closed=0;created.append(self)
            def close(self):self.closed+=1
        with patch.object(onboard_backend,'NativeBackend',Backend), \
             patch.object(onboard_backend,'NativeMotionRunner',lambda b:Runner(30)), \
             patch.object(onboard_backend,'NativeVelocityRunner',lambda b:Runner(28)):
            with OnboardDualPolicyRuntime.from_onboard_root(ROOT,deployment_scope='suspended_validation') as runtime:
                self.assertEqual(runtime.bundle.deployment_scope.value,'suspended_validation')
            runtime.close();self.assertEqual(created[-1].closed,1)
            with patch.object(OnboardDualPolicyRuntime,'_load_output_envelopes',side_effect=RuntimeError('bad envelope')):
                with self.assertRaisesRegex(RuntimeError,'bad envelope'):
                    OnboardDualPolicyRuntime.from_onboard_root(ROOT)
            self.assertEqual(created[-1].closed,1)

if __name__=='__main__':unittest.main()
