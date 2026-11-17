"""WTT v2 contract, velocity lifecycle, and training-interface parity tests."""
import copy
import importlib.util
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mimiclite_fixture import ROOT, deployment, contract as v1_contract
from deployment.vita_s100_aorta_30dof import actor_mimiclite as ml
from deployment.vita_s100_aorta_30dof import actor_pico_runtime as runtime_module


def v2_deployment():
    d = deployment()
    d['observation_schema'] = ml.OBSERVATION_SCHEMA_V2
    d['future_steps'] = [-8, -4, -2, 0, 1, 2, 3, 4]
    for term in d['observations']['command']:
        term['size'] = term['size']
    for term in d['observations']['policy']:
        if term['kind'] in ml.HISTORY_KINDS:
            term['history_steps'] = list(range(10))
            term['size'] = 10 * ml.HISTORY_KINDS[term['kind']]
            if term['kind'] == 'joint_pos_history':
                term['relative_to_default'] = True
    d['observations']['command'].append(dict(kind='ref_joint_vel_future', size=224,
                                             future_indices=list(range(8))))
    d['observations']['command'][0]['size'] = 24
    d['observations']['command'][1]['size'] = 48
    d['observations']['command'][2]['size'] = 224
    d['observations']['policy'][-1]['size'] = 84
    d['reference_joint_names'] = list(reversed(d['joint_names']))
    return d


def v2_contract():
    d = v2_deployment()
    raw = ml.metadata_contract(d, 'a' * 64, 'b' * 64)
    return ml.MimicLiteContract.from_metadata(None, {ml.CONTRACT_KEY: json.dumps(raw)})


def motion(sdk, c, n=40, with_velocity=True):
    q = np.tile(c.default, (n, 1))
    q[:, c.names.index('waist_yaw_joint')] += .03 * np.sin(np.arange(n) * .2)
    base = np.tile([0., 0., c.nominal_root_height], (n, 1))
    quat = np.tile([1., 0., 0., 0.], (n, 1))
    result = sdk.Motion(q, base, quat)
    if with_velocity:
        result.joint_vel = np.gradient(q, c.period, axis=0, edge_order=1)
    return result


def test_v2_schema_is_full_1224_wrap1224_and_relative_history():
    d = v2_deployment()
    checked, widths = ml.validate_deployment(d)
    assert checked['observation_schema'] == 'mimic_lite.motor_observation.v2'
    assert widths == {'command': 520, 'policy': 704}
    assert sum(widths.values()) == 1224
    c = v2_contract()
    assert c.input_dim == 1224
    assert c.deployment['future_steps'] == [-8, -4, -2, 0, 1, 2, 3, 4]


def test_v2_rejects_v1_relative_semantics_and_missing_velocity():
    d = deployment()
    d['observations']['policy'][2]['relative_to_default'] = True
    with pytest.raises(ValueError, match='only valid.*v2'):
        ml.validate_deployment(d)
    assert v2_contract().input_dim == 1224


def test_v2_explicit_velocity_path_and_reordered_names(policy_sdk):
    sdk = policy_sdk
    c = v2_contract()
    profile = json.loads((ROOT / 'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p = ml.policy_class(sdk)(c, profile)
    ref = motion(sdk, c)
    obs = p.observe(ref, .20, c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3))
    assert obs.shape == (1224,)
    assert np.isfinite(obs).all()
    # Explicitly supplied rates take precedence over motion.joint_vel.
    zero = np.zeros_like(ref.joint_vel)
    center = int(round(.20 / c.period))
    ids = np.clip(center + np.asarray(c.deployment['future_steps']), 0, len(ref.q) - 1)
    obs_zero = p.observe(ref, .20, c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3),
                         reference_joint_velocity=zero[ids])
    assert not np.array_equal(obs, obs_zero)


@pytest.fixture
def policy_sdk():
    config = json.loads((ROOT / 'config/actor_pico.json').read_text())
    return runtime_module.sdk_math(ROOT, config)


def test_v2_missing_velocity_fails_closed(policy_sdk):
    c = v2_contract()
    profile = json.loads((ROOT / 'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p = ml.policy_class(policy_sdk)(c, profile)
    ref = motion(policy_sdk, c, with_velocity=False)
    with pytest.raises(ValueError, match='velocity is missing'):
        p.observe(ref, .20, c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3))


def test_npz_velocity_reorder_dense_opt_in_and_full_window_equality(tmp_path, policy_sdk):
    from deployment.vita_s100_aorta_30dof.actor_motion import load_motion
    c = v2_contract()
    n = 30
    names = list(reversed(c.names))
    q = np.tile(c.default, (n, 1))
    q[:, c.names.index('waist_yaw_joint')] += np.arange(n) * .001
    qd = np.zeros_like(q); qd[:, c.names.index('waist_yaw_joint')] = .05
    order = [c.names.index(name) for name in names]
    path = tmp_path / 'v2.npz'
    np.savez(path, joint_names=np.array(names), joint_pos=q[:, order], joint_vel=qd[:, order],
             base_pos_w=np.tile([0., 0., c.nominal_root_height], (n, 1)),
             base_quat_w=np.tile([1., 0., 0., 0.], (n, 1)), fps=np.array(50.))
    runtime = NS(root=tmp_path, model=NS(contract=c), math=policy_sdk,
                 config={'body_profile': {'joint_limits': {name: [-4., 4.] for name in c.names},
                                          'lead_in': {'speed_q_per_s': 1., 'min_duration_s': .02, 'max_duration_s': .02}}},
                 reference=NS(steps=13, current_index=8))
    cfg = tmp_path / 'motion.json'; cfg.write_text(json.dumps({'schema': 'vita_actor_motion_v1',
        'path': str(path), 'end_hold_s': .04}))
    fixed = load_motion(runtime, cfg)
    fixed.begin(c.default, [1, 0, 0, 0])
    window, ref_time = fixed.sample()
    assert hasattr(window, 'joint_vel')
    # Lead-in starts at rest, then preserves explicit source rates.
    assert np.max(np.abs(window.joint_vel[:, c.names.index('waist_yaw_joint')])) == pytest.approx(.05)
    # A bounded window carries the same explicit rates as the corresponding
    # full prepared timeline, not a derivative of the cropped window.
    np.testing.assert_allclose(window.joint_vel,
                               fixed.prepared.joint_vel[:len(window.q)])
    no_vel = tmp_path / 'no_vel.npz'
    np.savez(no_vel, joint_names=np.array(c.names), joint_pos=q,
             base_pos_w=np.tile([0., 0., c.nominal_root_height], (n, 1)),
             base_quat_w=np.tile([1., 0., 0., 0.], (n, 1)), fps=np.array(50.))
    cfg.write_text(json.dumps({'schema': 'vita_actor_motion_v1', 'path': str(no_vel), 'end_hold_s': .04}))
    with pytest.raises(ValueError, match='needs joint_vel'):
        load_motion(runtime, cfg)
    cfg.write_text(json.dumps({'schema': 'vita_actor_motion_v1', 'path': str(no_vel),
                               'end_hold_s': .04, 'velocity_fallback': 'dense_derivative'}))
    assert hasattr(load_motion(runtime, cfg).motion, 'joint_vel')
    # Compare bounded samples against the full prepared timeline at both ends;
    # rates must come from the source timeline, including lead-in/end hold.
    cfg.write_text(json.dumps({'schema': 'vita_actor_motion_v1', 'path': str(path), 'end_hold_s': .04}))
    fixed = load_motion(runtime, cfg)
    fixed.begin(c.default, [1, 0, 0, 0])
    fixed.tick = fixed.total_ticks - 1
    bounded, _ = fixed.sample()
    np.testing.assert_array_equal(bounded.joint_vel[-1], np.zeros(28))


def test_v2_wrap1224_ort_parity_history_actions_and_reset(policy_sdk):
    from import_mimiclite_policy import validate_parity, wrap
    import onnxruntime as ort
    c = v2_contract(); d = copy.deepcopy(c.deployment)
    from mimiclite_fixture import source_model
    source = source_model(d)
    converted, raw, shapes = wrap(source, d, 'b' * 64)
    result = validate_parity(source, converted, raw, shapes)
    assert result['samples'] == 9 and result['max_abs_error'] < 1e-5
    session = ort.InferenceSession(converted, providers=['CPUExecutionProvider'])
    profile = json.loads((ROOT / 'config/actor_pico_mimiclite.json').read_text())['body_profile']
    policy = ml.policy_class(policy_sdk)(c, profile)
    ref = motion(policy_sdk, c, n=40)
    rng = np.random.default_rng(23)
    for step in range(20):
        q = c.default + rng.normal(0., .01, 28)
        obs = policy.observe(ref, step * c.period, q, np.zeros(28), [1, 0, 0, 0], np.zeros(3))
        action = rng.normal(0., 2., 28).astype(np.float32)
        policy.decode(action)
        actual, = session.run(None, {session.get_inputs()[0].name: obs[None]})
        assert actual.shape == (1, 28) and np.isfinite(actual).all()
        # Histories are newest-first: after each decode, the first action row
        # is the latest action and reset clears all three rows.
        assert np.array_equal(policy.previous_actions[0], action)
    policy.reset()
    assert not policy.previous_actions.any() and not policy.signal_history


def test_v2_rejects_invalid_history_steps_indices_dims_and_bool():
    mutations = []
    def bad_history(d): d['observations']['policy'][0]['history_steps'] = list(range(9))
    def bad_indices(d): d['observations']['command'][2]['future_indices'] = list(range(7))
    def bad_dims(d): d['observations']['command'][0]['size'] += 1
    def bad_bool(d): d['observations']['policy'][-1]['steps'] = True
    for mutate in (bad_history, bad_indices, bad_dims, bad_bool):
        d = v2_deployment(); mutate(d); mutations.append(d)
    for d in mutations:
        with pytest.raises(ValueError):
            ml.validate_deployment(d)


def test_v2_rejects_nonfinite_explicit_and_missing_velocities(policy_sdk):
    c = v2_contract(); profile = json.loads((ROOT / 'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p = ml.policy_class(policy_sdk)(c, profile); ref = motion(policy_sdk, c)
    ids = np.clip(10 + np.asarray(c.deployment['future_steps']), 0, len(ref.q) - 1)
    bad = np.zeros((len(ids), 28)); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        p.observe(ref, .20, c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3),
                  reference_joint_velocity=bad)
    # Float16 export quantization may introduce tiny norm drift; the oracle
    # accepts near-unit quaternions while still rejecting malformed ones.
    near = np.tile(np.array([.9995, .0316, 0., 0.], dtype=np.float16), (len(ids), 1)).astype(float)
    assert np.max(np.abs(np.linalg.norm(near, axis=1) - 1.)) < 1e-3
    ref_near = motion(policy_sdk, c)
    assert np.isfinite(p.observe(ref_near, .20, c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3),
        reference={'root_quaternion': near, 'root_position': ref_near.base[ids]})).all()
    ref = motion(policy_sdk, c, with_velocity=False)
    with pytest.raises(ValueError, match='velocity is missing'):
        p.observe(ref, .20, c.default, np.zeros(28), [1, 0, 0, 0], np.zeros(3))


def test_supermimic_velocity_is_global_and_hold_zero(policy_sdk):
    from deployment.vita_s100_aorta_30dof.actor_supermimic import SuperMimicReference
    c = v2_contract()
    r = SuperMimicReference(c, policy_sdk, past_frames=8, tail_frames=0)
    for i in range(20):
        q = c.default.copy(); q[c.names.index('waist_yaw_joint')] = .01 * i * i
        r.append(1_000_000_000 + i * 20_000_000, q, [1, 0, 0, 0],
                 base=[0., 0., c.nominal_root_height], received_monotonic=float(i),)
    m = r.motion()
    assert hasattr(m, 'joint_vel')
    # Window velocity at the center agrees with the source-timeline slope.
    assert m.joint_vel[r.current_index, c.names.index('waist_yaw_joint')] == pytest.approx(15., abs=1e-6)
    stream = r.export(); stream['clock_offset_s'] = 0.
    held, diag = r.playback(stream, 1.0, 0.)
    assert diag['playback_lead_s'] < 0 or np.allclose(held.joint_vel[-1], 0.)
    # A held/stale reference must never report a residual rate.
    np.testing.assert_array_equal(held.joint_vel[-1], np.zeros(28))


def test_training_numpy_interface_matches_v2_term_order_and_velocity(policy_sdk):
    training = ROOT.parent / 'MimicLite/mimic-lite/mimic_lite/vita02a_deploy.py'
    if not training.is_file():
        pytest.skip('training-side checkout unavailable')
    spec = importlib.util.spec_from_file_location('_mimiclite_training_v2', training)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(training.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    c = v2_contract(); d = copy.deepcopy(c.deployment)
    # The synthetic reference arrays are in natural motor order; the deployment
    # deliberately reverses reference_joint_names to exercise relabelling.
    profile = json.loads((ROOT / 'config/actor_pico_mimiclite.json').read_text())['body_profile']
    p = ml.policy_class(policy_sdk)(c, profile)
    oracle = module.MotorAngleInterface(d)
    ref = motion(policy_sdk, c)
    q, dq, gyro = c.default.copy(), np.ones(28) * .03, np.ones(3) * .02
    actual = p.observe(ref, .20, q, dq, [1, 0, 0, 0], gyro)
    ids = np.clip(10 + np.asarray(d['future_steps']), 0, len(ref.q) - 1)
    expected_ref = dict(root_position=ref.base[ids], root_quaternion=ref.quat[ids],
                        joint_names=list(c.motor_names),
                        joint_position=c.ankle.to_policy(ref.q[ids]),
                        joint_velocity=c.ankle.velocity_to_policy(ref.q[ids], ref.joint_vel[ids]))
    expected = oracle.observe(c.ankle.to_policy(q), c.ankle.velocity_to_policy(q, dq),
                              [1, 0, 0, 0], gyro, expected_ref)
    np.testing.assert_allclose(actual, np.concatenate([expected[k][0] for k in c.input_groups]),
                               atol=2e-6, rtol=1e-6)
