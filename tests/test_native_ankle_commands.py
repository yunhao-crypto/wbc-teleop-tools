"""Offline SDK math/Body31 tests; no controller or transport is constructed."""
import copy
from dataclasses import replace
import json
import tempfile
import yaml
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import locomotion_aorta.wbc_tracking_policy as sdk
from deployment.vita_s100_aorta_30dof import actor_groot_ankle as groot
from deployment.vita_s100_aorta_30dof import actor_supermimic as supermimic
from deployment.vita_s100_aorta_30dof import actor_morphmimic as morph
from deployment.vita_s100_aorta_30dof.actor_parallel_ankle import MotorThetaAnkle, command_space
from deployment.vita_s100_aorta_30dof.actor_pico_live import load_robot_config, CommandEnvelope
from deployment.vita_s100_aorta_30dof.actor_pico_startup import StartupPlan
from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime

ROOT=Path(__file__).resolve().parents[1]
ANKLE=dict(schema_version=3,control_space='motor_theta',robot_type='vita02a_closed_loop',
    reference_conversion='virtual_pitch_roll_to_motor_theta',
    geometry=dict(lx=.05,ly=.02175,z_top=.31,z_bot=.235),motor_limits=[-.95975,.61075],
    roll_sign=dict(left=1,right=-1),slots={f'{s}_ankle_{a}_joint':dict(joint=f'{s}_{b}_achilles_joint',motor=m,side=s)
    for s in ('left','right') for a,b,m in (('pitch','A','theta_top'),('roll','B','theta_bottom'))})


def contracts():
    raw=json.loads((ROOT/'tests/fixtures/actor_pico/actor_g1_contract.json').read_text())
    ankle=MotorThetaAnkle(raw['joint_names'],ANKLE);slots=ankle.slots
    raw['default_joint_pos']=ankle.to_policy(raw['default_joint_pos']).tolist()
    raw['ankle']=ANKLE
    for key,value in (('joint_stiffness',20.),('joint_damping',1.),
                       ('joint_pos_limit_lower',-.95975),('joint_pos_limit_upper',.61075)):
        array=np.asarray(raw[key]);array[slots]=value;raw[key]=array.tolist()
    gc=groot.contract_from_metadata(sdk,{groot.CONTRACT_KEY:json.dumps(raw)})
    # Reuse the same valid physical contract, with each actor's own observation layout.
    sm=supermimic.SuperMimicContract(gc.names,gc.default,gc.scale,gc.kp,gc.kd,gc.lower,gc.upper,
        gc.action_clip,gc.period,10,.02,10,1670,96,71,.1,
        policy_default=gc.policy_default,policy_lower=gc.policy_lower,policy_upper=gc.policy_upper,ankle=gc.ankle)
    # Output decoder is shared; actual MorphMimic input/ONNX parity has a separate integration check.
    mc=morph.MorphMimicContract(**{f: getattr(gc,f) for f in gc.__dataclass_fields__ if f!='policy_kind'})
    return gc,sm,mc


class NativeAnkleTests(unittest.TestCase):
    def policies(self):
        profile=json.loads((ROOT/'config/actor_pico.json').read_text())['body_profile']
        profile['ankle_command_space']='policy'
        gc,sm,mc=contracts()
        return [groot.create_policy(gc,copy.deepcopy(profile),sdk),
                supermimic.policy_class(sdk)(sm,copy.deepcopy(profile),past_frames=1,tail_frames=1),
                morph.policy_class(sdk)(mc,copy.deepcopy(profile))]

    def test_all_three_send_ab_and_independent_gains_without_fk_or_compensation(self):
        for policy in self.policies():
            with self.subTest(kind=policy.contract.policy_kind):
                c=policy.contract; slots=c.ankle.slots
                kp=c.kp.copy();kd=c.kd.copy();kp[slots]=[21,22,23,24];kd[slots]=[1,2,3,4]
                policy.contract=replace(c,kp=kp,kd=kd)
                action=(np.full(28,.55)-c.policy_default)/c.scale
                # The legacy virtual roll guard is narrower than this valid motor target.
                with patch.object(c.ankle,'to_virtual',side_effect=AssertionError('native path ran FK')):
                    cmd,row=policy.decode(action)
                for i in slots:
                    j=cmd.joints[policy.mapping[i]]
                    self.assertAlmostEqual(j.q,.55,places=6)
                    self.assertEqual(j.kp,kp[i]);self.assertEqual(j.kd,kd[i]);self.assertEqual(j.tau,0.)
                self.assertEqual(row['ankle_command_space'],'motor_theta')
                self.assertEqual(len(cmd.joints),31)
                for name in ('left_wrist_pitch_joint','left_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_yaw_joint'):
                    i=c.names.index(name);self.assertEqual(policy.order[policy.mapping[i]],name)

    def test_all_three_clip_actions_without_clipping_position_targets(self):
        for policy in self.policies():
            self.assertFalse(policy.clip_target_positions)
            for sign in (-1., 1.):
                with self.subTest(kind=policy.contract.policy_kind, sign=sign):
                    c = policy.contract
                    cmd, row = policy.decode(np.full(28, sign*100.))
                    expected = c.policy_default+c.scale*(sign*c.action_clip)
                    np.testing.assert_array_equal(policy.last_action, sign*c.action_clip)
                    np.testing.assert_allclose(row['target'], expected)
                    self.assertFalse(any(row['clamped']))
                    self.assertFalse(any(row['tau_ff']))
                    for i, slot in enumerate(policy.mapping):
                        self.assertEqual(cmd.joints[slot].q, float(np.float32(expected[i])))
                    lo, hi = c.ankle.motor_limits
                    self.assertTrue(np.any((expected[c.ankle.slots] < lo) | (expected[c.ankle.slots] > hi)))

    def test_all_three_reject_unrepresentable_virtual_wire_targets(self):
        for policy in self.policies():
            policy.profile['ankle_command_space'] = 'virtual_pitch_roll'
            with self.assertRaisesRegex(ValueError, 'motor_theta command space'):
                policy.decode(np.full(28, 100.))

    def test_legacy_wire_path_retains_fk(self):
        for policy in self.policies():
            policy.profile['ankle_command_space']='virtual_pitch_roll'
            cmd,row=policy.decode(np.zeros(28))
            for i in policy.contract.ankle.slots:
                self.assertAlmostEqual(cmd.joints[policy.mapping[i]].q,policy.contract.default[i],places=6)
            self.assertEqual(row['ankle_command_space'],'virtual_pitch_roll')

    def test_nonfinite_and_wrong_width_rejected(self):
        for policy in self.policies():
            for action in (np.zeros(29),np.full(28,np.nan),np.full(28,np.inf)):
                with self.assertRaises(ValueError):policy.decode(action)

    def config(self, policy):
        root = dict(parallel_ankle=dict(geometry=ANKLE['geometry'],
            caps=dict(theta_min=-.95975,theta_max=.61075,qd_motor_max=12.25,tau_motor_max=50),
            pd=dict(strategy='virtual_pd',kp_motor=20,kd_motor=1,kp_motor_max=600,kd_motor_max=60)),
            actions=dict(EXTERNAL=dict(generator='builtin::external',params=dict(
                ankle_command_space='motor_theta',ankle_virtual_pd_intent=False,
                command_timeout_ms=1000,recover_valid_frames=1,
                interpolation=dict(min_period_ms=2,max_period_ms=100),
                safe_hold=dict(kp=[20.]*31,kd=[1.]*31)))))
        root['actions']['FIXED_STAND']=dict(generator='builtin::keyframe',params=dict(
            keyframes=[dict(timestamp=0,positions=[0.]*31,kp=[20.]*31,kd=[1.]*31)]))
        for name in ('PASSIVE','QUIET_PASSIVE'):
            root['actions'][name]=dict(params=dict(keyframes=[dict(timestamp=0,kd=5.),dict(timestamp=4,kd=0.)]))
        runtime=SimpleNamespace(head=None,policy=policy,model=SimpleNamespace(contract=policy.contract),
                                config=dict(body_profile=policy.profile),math=sdk)
        return root,runtime

    def test_config_mismatch_rejected_before_session(self):
        for policy in self.policies():
            root,runtime=self.config(policy)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'robot.yaml';path.write_text(yaml.safe_dump(root))
                load_robot_config(path,runtime)
                for space,intent in [('virtual_pitch_roll',False),('motor_theta',True),('auto',False)]:
                    root['actions']['EXTERNAL']['params'].update(ankle_command_space=space,ankle_virtual_pd_intent=intent)
                    path.write_text(yaml.safe_dump(root))
                    with self.assertRaises(ValueError):load_robot_config(path,runtime)

    def test_native_command_limits_keep_feedback_and_startup_virtual(self):
        from locomotion_aorta.body_types import BodyJointCapability,BodyInfo,RobotInfo,GetRobotInfoResponse
        for policy in self.policies():
            root,runtime=self.config(policy)
            caps=[]
            for i,name in enumerate(policy.order):
                ankle='_ankle_' in name
                caps.append(BodyJointCapability(i,1,False,i!=14,31,1 if ankle else 31,
                    -.43625 if ankle else -4.,.43625 if ankle else 4.,
                    -25.,25.,-100.,100.,0.,600.,0.,60.))
            body=BodyInfo(1,31,tuple(policy.order),tuple(caps),())
            reply=GetRobotInfoResponse(0,RobotInfo(1,1,body,()),'')
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'robot.yaml';path.write_text(yaml.safe_dump(root))
                params,_=load_robot_config(path,runtime)
                envelope=CommandEnvelope(runtime,reply,params)
                startup=StartupPlan(runtime,envelope,path)
                self.assertEqual(startup.damping,dict(PASSIVE=4.,QUIET_PASSIVE=4.))
                action=np.zeros(28);slots=policy.contract.ankle.slots
                action[slots]=(.55-policy.contract.policy_default[slots])/policy.contract.scale[slots]
                command,_=policy.decode(action)
                envelope.check(command)
                if getattr(policy, 'clip_target_positions', True) is False:
                    # Exercise the complete policy -> Body31 -> envelope path:
                    # all controlled PD targets survive advertised q ranges.
                    extreme, row = policy.decode(np.full(28, 100.))
                    envelope.check(extreme)
                    expected = policy.contract.policy_default+policy.contract.scale*policy.contract.action_clip
                    for i, slot in enumerate(policy.mapping):
                        self.assertEqual(extreme.joints[slot].q, float(np.float32(expected[i])))
                    first = int(policy.mapping[0])
                    for field, value in [('dq', 100.), ('tau', 1000.), ('kp', 1000.), ('kd', 1000.), ('q', float('nan'))]:
                        joints = list(extreme.joints)
                        joints[first] = replace(joints[first], **{field:value})
                        with self.assertRaises(ValueError):
                            envelope.check(replace(extreme, joints=tuple(joints)))
                    fixed = policy.order.index(next(iter(policy.profile['fixed_joints'])))
                    joints = list(extreme.joints)
                    joints[fixed] = replace(joints[fixed], q=99.)
                    with self.assertRaises(ValueError):
                        envelope.check(replace(extreme, joints=tuple(joints)))
                # Keep float32 boundary coverage for caps still enforced.
                for index in policy.contract.ankle.slots:
                    slot=int(policy.mapping[index])
                    for field,bounds in zip(('q','dq','tau','kp','kd'),envelope.limits[slot]):
                        if field == 'q' and getattr(policy, 'clip_target_positions', True) is False:
                            continue
                        for bound,direction in zip(bounds,(-np.inf,np.inf)):
                            joints=list(command.joints)
                            joints[slot]=replace(joints[slot],**{field:bound})
                            envelope.check(replace(command,joints=tuple(joints)))
                            outside=float(np.nextafter(np.float32(bound),np.float32(direction)))
                            joints[slot]=replace(joints[slot],**{field:outside})
                            with self.assertRaises(ValueError):
                                envelope.check(replace(command,joints=tuple(joints)))
                with self.assertRaises(ValueError):envelope.check(command,virtual=True)
                measured=np.zeros(31);measured[policy.mapping[slots[0]]]=.55
                with self.assertRaises(ValueError):envelope.check_measured_pose(SimpleNamespace(q=measured))

    def test_morph_stand_reference_has_training_height(self):
        for policy in self.policies():
            _,runtime=self.config(policy)
            motion=ActorPicoRuntime.hold_motion(runtime)
            expected=policy.contract.nominal_root_height if isinstance(policy.contract,morph.MorphMimicContract) else 0.
            self.assertAlmostEqual(motion.base[0,2],expected)

    def test_native_caps_must_fit_float32_wire_fields(self):
        for policy in self.policies():
            for section,key in (('caps','qd_motor_max'),('pd','kp_motor_max')):
                root,runtime=self.config(policy)
                root['parallel_ankle'][section][key]=1e100
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'robot.yaml';path.write_text(yaml.safe_dump(root))
                    with self.assertRaisesRegex(ValueError,'represented as float32'):
                        load_robot_config(path,runtime)

    def test_virtual_policy_cannot_claim_native_motor_wire(self):
        raw=json.loads((ROOT/'tests/fixtures/actor_pico/actor_g1_contract.json').read_text())
        c=sdk.Contract.from_metadata({groot.CONTRACT_KEY:json.dumps(raw)})
        profile=json.loads((ROOT/'config/actor_pico.json').read_text())['body_profile']
        profile['ankle_command_space']='motor_theta'
        with self.assertRaises(ValueError):command_space(sdk.WbcTrackingPolicy(c,profile))


if __name__=='__main__':unittest.main()
