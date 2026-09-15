"""Continuous primitive checks with an explicitly fake, hardware-free controller."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import ggx_motion as motion, run_ggx_real as real


@pytest.fixture
def scene(monkeypatch):
    # Six synthetic coordinates: fake FK/IK are identity mappings. Geometry
    # checks are injected below so we can place an obstacle between endpoints.
    model = real.sim.mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/><joint axis="1 0 0"/>
      <joint axis="0 1 0"/><joint axis="0 0 1"/><geom size=".01"/>
    </body></worldbody></mujoco>''')
    monkeypatch.setattr(motion.sim, 'initial_qpos', lambda m,q: np.array(q))
    monkeypatch.setattr(motion.sim, 'set_hand_qpos', lambda *a: None)
    monkeypatch.setattr(motion.support, 'fk_tcp', lambda m,q: list(q))
    monkeypatch.setattr(motion.sim, 'desk_clearance', lambda m,d: (.1, 'fake'))
    monkeypatch.setattr(motion.sim, 'object_clearance', lambda m,d,*args: (.02, 'fake'))
    start = [-.2,-.4,.35,0.,0.,0.]
    pre = [-.3,-.5,.4,0.,0.,0.]
    approach = [-.3,-.5,.32,0.,0.,0.]
    rows = []
    def add(phase, point, alpha=0.):
        rows.append(dict(index=len(rows), phase=phase, tcp=list(point), q_nominal=list(point),
                         closure_alpha=alpha, hand_raw=[round(1000*(1-alpha))]*6))
    add('start', start)
    for t in np.linspace(0,1,41)[1:]: add('start_to_pregrasp', np.array(start)+t*(np.array(pre)-start))
    for t in np.linspace(0,1,17)[1:]: add('approach', np.array(pre)+t*(np.array(approach)-pre))
    for t in np.linspace(0,.5,15)[1:]: add('close', approach, t)
    for t in np.linspace(0,1,17)[1:]: add('lift', np.array(approach)+t*(np.array(pre)-approach), .5)
    snapshot = dict(q=list(start), tcp=list(start))
    prepared = dict(snapshot=snapshot, rows=rows, hand_open_raw=[1000]*6)
    control = SimpleNamespace(getTCPOffset=lambda: [0.]*6,
        getInverseKinematics=lambda target,*a: list(target),
        getForwardKinematics=lambda q,*a: list(q),
        isPoseWithinSafetyLimits=lambda p: True, isJointsWithinSafetyLimits=lambda q: True)
    backend = SimpleNamespace(control=control, snapshot=lambda: snapshot)
    config = real.load_config(real.ROOT/'configs/ggx_real_lab.yaml')
    return SimpleNamespace(prepared=prepared, model=model, backend=backend, config=config)


def check(scene, mode='grasp-lift'):
    return motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, {}, mode)


@pytest.mark.parametrize('margin,gap,blocked',[(None,.0018,True),(1.5,.0018,False),
    (1.5,.0014,True),(1.5,0.,True),(1.5,-.001,True),
    (0.,.0005,False),(0.,0.,True),(0.,-.001,True)])
def test_reflex_profile_controller_desk_margin_keeps_real_overlap_blocked(scene,monkeypatch,margin,gap,blocked):
    config=dict(scene.config)
    if margin is not None: config['reflex_desk_clearance_mm']=margin
    monkeypatch.setattr(motion.sim,'desk_clearance',lambda *a:(gap,'test_thumb'))
    report={}
    if blocked:
        with pytest.raises(motion.PathCollision,match='desk'):
            motion.preflight(scene.backend,scene.prepared,scene.model,config,report)
    else:
        assert motion.preflight(scene.backend,scene.prepared,scene.model,config,report)
    assert report['required_desk_clearance_mm']==(2. if margin is None else margin)


@pytest.mark.parametrize('palm_x,blocked',[(.3,False),(.01,True)])
def test_reflex_contact_does_not_hide_simultaneous_palm_collision(palm_x,blocked):
    model=motion.sim.mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="object"><geom name="preflight_object_visual" type="box" size=".02 .02 .02"/></body>
      <body name="thumb_distal"><geom type="sphere" size=".015"/></body>
      <body name="root" pos="{palm_x} 0 0"><geom type="sphere" size=".01"/></body>
    </worldbody></mujoco>''')
    data=motion.sim.mujoco.MjData(model);motion.sim.mujoco.mj_forward(model,data)
    contact=motion.reflex_target_check(model,data,'approach')
    assert contact['target_overlap']
    assert contact['forbidden_target_overlap']==blocked
    assert 'root' in contact['nearest_forbidden_target_geom']
    assert motion.reflex_target_check(model,data,'position_to_trial_start')['forbidden_target_overlap']
    assert motion.object_clearance(model,data)[0]<=0  # Default GGX remains strict.


def test_reflex_controller_policy_is_explicit_and_phase_scoped(scene,monkeypatch):
    goals=motion.phase_goals(scene.prepared)
    calls=[]
    def contact(m,d,phase):
        calls.append(phase)
        return dict(minimum_forbidden_target_gap_mm=20.,nearest_forbidden_target_geom='root',
                    minimum_target_gap_mm=-5.,target_overlap=True,forbidden_target_overlap=False)
    monkeypatch.setattr(motion,'reflex_target_check',contact)
    for goal in goals[1:]:
        goal.update(hand_ctrl=dict(motion.sim.GGX_OPEN_CTRL),target_contact_policy=motion.REFLEX_CONTACT_POLICY)
    result=motion.preflight_goals(scene.backend,scene.prepared['snapshot'],goals,scene.model,scene.config,{})
    assert set(calls)=={'approach','lift'}
    assert result[1]['samples'][0]['target_contact']['target_overlap']
    goals[0].update(hand_ctrl=dict(motion.sim.GGX_OPEN_CTRL),target_contact_policy=motion.REFLEX_CONTACT_POLICY)
    with pytest.raises(ValueError,match='contact policy'):
        motion.preflight_goals(scene.backend,scene.prepared['snapshot'],goals,scene.model,scene.config,{})


def add_trial_start(prepared):
    row = dict(prepared['rows'][1], phase='position_to_trial_start')
    prepared['rows'].insert(1, row)
    for index, item in enumerate(prepared['rows']):
        item['index'] = index
    prepared['trial_start'] = dict(id='P2')


def test_ten_start_adds_checked_positioning_without_changing_grasp_or_hand(scene):
    original = deepcopy(scene.prepared['rows'])
    add_trial_start(scene.prepared)
    checked = check(scene)
    assert [c['phase'] for c in checked] == ['position_to_trial_start', 'start_to_pregrasp', 'approach', 'lift']
    assert [r for r in scene.prepared['rows'] if r['phase'] == 'close'] == [
        dict(r, index=r['index']+1) for r in original if r['phase'] == 'close']
    events = []
    backend = SimpleNamespace(open_hand=lambda v: events.append('open'),
        move_phase=lambda c: events.append(c['phase']), hand_angles=lambda v: events.append('hand'),
        dwell=lambda s: None, set_stage=lambda s: None)
    real.execute_checked(backend, scene.prepared, checked, scene.config, 'grasp-lift')
    assert events[:4] == ['open', 'position_to_trial_start', 'start_to_pregrasp', 'approach']
    assert events[-1] == 'lift'


def test_new_positioning_phase_checks_object_before_any_motion(scene, monkeypatch):
    add_trial_start(scene.prepared)
    monkeypatch.setattr(motion, 'object_clearance', lambda *a: (-.002, 'object'))
    report = {}
    with pytest.raises(RuntimeError, match='hits object'):
        motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
    assert report['preflight_current_sample']['phase'] == 'position_to_trial_start'


def test_extra_positioning_phase_requires_explicit_trial_metadata(scene):
    add_trial_start(scene.prepared)
    scene.prepared.pop('trial_start')
    with pytest.raises(ValueError, match='ordered GGX phases'):
        motion.phase_goals(scene.prepared)


def test_explicit_reflex_phase_hand_shape_overrides_only_that_goal(scene,monkeypatch):
    observed=[]
    monkeypatch.setattr(motion.sim,'set_hand_qpos',lambda model,q,ctrl:observed.append(dict(ctrl)))
    goals=motion.phase_goals(scene.prepared)
    thumb=dict(motion.sim.GGX_OPEN_CTRL,thumb_proximal=.4)
    goals[0]['hand_ctrl']=thumb
    commands=motion.preflight_goals(scene.backend,scene.prepared['snapshot'],goals,scene.model,scene.config,{})
    assert commands[0]['hand_ctrl']==thumb
    n=len(commands[0]['samples'])
    assert observed[:n]==[thumb]*n
    assert observed[n]==motion.sim._lerp_ctrl(0.)


def test_partial_explicit_hand_shape_is_rejected(scene):
    goals=motion.phase_goals(scene.prepared)
    goals[0]['hand_ctrl']={'thumb_proximal':.4}
    with pytest.raises(ValueError,match='all finite'):
        motion.preflight_goals(scene.backend,scene.prepared['snapshot'],goals,scene.model,scene.config,{})


@pytest.fixture
def skew_boxes():
    # Recorded rank53 failure, 20260911T205153: literal world geometry, no local
    # artifacts/hardware dependency. Face axes overlap but edge axes separate.
    model = real.sim.mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="gripper_attachment" pos="-.14340495522478963 -.6101360433240486 .22661476970253358"
            quat=".9773323418603806 .12248495080191812 -.14937099621236516 -.08664430662935205">
        <freejoint/><geom name="mount" type="box" size=".0161 .02 .0407"/>
      </body>
      <body name="preflight_object" pos="-.21906597900390626 -.6459269409179688 .13793038326475576"
            quat=".9997731796206614 0 0 -.02129763628182179">
        <geom name="preflight_object_visual" type="box" size=".046930649961476535 .025131112345664314 .09851265904926755"/>
      </body>
    </worldbody></mujoco>''')
    data = real.sim.mujoco.MjData(model)
    real.sim.mujoco.mj_forward(model, data)
    return model, data, model.geom('preflight_object_visual').id, model.geom('mount').id


def test_skew_box_false_zero_requires_edge_axis_certificate(skew_boxes, monkeypatch):
    model, data, obj, mount = skew_boxes
    assert not len(data.contact)
    assert motion.sim.certified_separation(model, data, obj, mount) == 0.
    assert motion._box_edge_separation(model, data, obj, mount) == pytest.approx(.00344935519665)
    monkeypatch.setattr(real.sim.mujoco, 'mj_geomDistance', lambda *a: 0.)
    assert motion.object_clearance(model, data)[0] == pytest.approx(.00344935519665)


@pytest.mark.parametrize('fault', ['negative', 'contact'])
def test_edge_certificate_does_not_override_signed_overlap_or_contact(skew_boxes, monkeypatch, fault):
    model, data, obj, mount = skew_boxes
    if fault == 'contact':
        data.qpos[:3] = data.geom_xpos[obj]
        real.sim.mujoco.mj_forward(model, data)
        assert len(data.contact)
    distance = -.001 if fault == 'negative' else 0.
    monkeypatch.setattr(real.sim.mujoco, 'mj_geomDistance', lambda *a: distance)
    monkeypatch.setattr(motion, '_box_edge_separation', lambda *a: pytest.fail('Actual contact/signed overlap must win'))
    assert motion.object_clearance(model, data)[0] == distance


@pytest.mark.parametrize('distance', [0., -.001])
def test_execution_review_uses_same_zero_correction_but_preserves_penetration(scene, monkeypatch, tmp_path, distance):
    commands = check(scene, 'approach-only')
    original = motion.sim.check_trajectory
    def annotate(model, samples, **kwargs):
        result = original(model, samples, **kwargs)
        samples[-1].object_clearance_m = distance
        samples[-1].object_collision = True
        return result
    monkeypatch.setattr(motion.sim, 'check_trajectory', annotate)
    monkeypatch.setattr(motion, 'object_clearance', lambda *a: (.003449, 'mount'))
    if distance < 0:
        with pytest.raises(RuntimeError, match='collision recheck'):
            motion.write_review(tmp_path, scene.prepared, scene.model, commands, video=False)
    else:
        assert motion.write_review(tmp_path, scene.prepared, scene.model, commands, video=False) is None


def test_dense_samples_become_three_primitives_not_stop_waypoints(scene):
    commands = check(scene)
    assert [c['kind'] for c in commands] == ['moveJ','moveL','moveL']
    assert [c['replaced_sample_count'] for c in commands] == [40,16,16]
    assert all(len(c['samples']) >= 10 for c in commands)
    for c in commands:
        joints = np.array([s['q'] for s in c['samples']])
        assert np.max(np.abs(np.diff(joints,axis=0))) <= .01+1e-10
        for s in c['samples']:
            if c['kind']=='moveJ':
                expected = np.array(c['start_q'])+s['fraction']*(np.array(c['q'])-c['start_q'])
                assert np.allclose(s['q'],expected)
            else:
                assert np.allclose(s['tcp'],motion.pose_at(c['start_tcp'],c['goal_tcp'],s['fraction']))


def test_approach_mode_checks_and_executes_only_two_primitives(scene):
    commands = check(scene,'approach-only')
    assert [c['phase'] for c in commands] == ['start_to_pregrasp','approach']


def test_preflight_reuses_only_exact_fk_and_times_every_controller_query(scene, monkeypatch):
    scene.config['controller_preflight_mode'] = 'dense'
    clock = [0.]
    calls = {}
    monkeypatch.setattr(motion.time, 'monotonic', lambda: clock[0])
    for name in ('getInverseKinematics', 'getForwardKinematics',
                 'isPoseWithinSafetyLimits', 'isJointsWithinSafetyLimits'):
        original = getattr(scene.backend.control, name)
        def counted(*args, name=name, original=original):
            calls.setdefault(name, []).append(tuple(args[0]))
            clock[0] += .01  # Synthetic query latency, not a hardware speed claim.
            return original(*args)
        monkeypatch.setattr(scene.backend.control, name, counted)
    report = {}
    commands = motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
    timing = report['controller_preflight_timing']
    samples = sum(len(c['samples']) for c in commands)
    ik_count = len(calls['getInverseKinematics'])
    fk_calls = calls['getForwardKinematics']
    assert len(fk_calls) == len(set(fk_calls))
    assert len(fk_calls) + timing['fk_cache_hits'] == samples + ik_count
    assert timing['fk_cache_hits'] >= ik_count - 1
    # All sampled joint checks, target/sample pose checks, and geometry remain.
    assert len(calls['isJointsWithinSafetyLimits']) == samples
    assert len(calls['isPoseWithinSafetyLimits']) == samples + ik_count
    assert timing['validated_samples'] == samples
    assert timing['status'] == 'completed'
    assert timing['controller_query_count'] == sum(map(len, calls.values()))
    for name, row in timing['queries'].items():
        assert row['count'] == len(calls[name]) and row['errors'] == 0
        assert row['elapsed_s'] == pytest.approx(.01 * row['count'])
    assert timing['controller_query_s'] == pytest.approx(timing['controller_query_count'] * .01)
    assert timing['elapsed_s'] == pytest.approx(timing['controller_query_s'])
    assert timing['other_s'] == pytest.approx(0.)
    # No result is reused from the previous stationary preflight.
    second = {}
    assert motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, second) == commands
    for name, row in timing['queries'].items():
        next_row = second['controller_preflight_timing']['queries'][name]
        assert next_row['count'] == row['count']
        assert next_row['elapsed_s'] == pytest.approx(row['elapsed_s'])
    assert len(calls['getForwardKinematics']) == 2 * len(set(fk_calls))


def test_keypoints_preserves_ik_calls_primitives_and_all_geometry_samples(scene, monkeypatch):
    # Simulate a stable nominal-vs-calibrated FK offset: interior estimates
    # must be labelled, while actual endpoints and IK seeds remain unchanged.
    monkeypatch.setattr(motion.support, 'fk_tcp', lambda m,q: (np.array(q)+[.002,0,0,0,0,0]).tolist())
    runs = []
    for mode in ['dense', 'keypoints']:
        scene.config['controller_preflight_mode'] = mode
        ik_calls = []
        def ik(target, seed, *a):
            ik_calls.append((list(target), list(seed)))
            return list(target)
        monkeypatch.setattr(scene.backend.control, 'getInverseKinematics', ik)
        report = {}
        commands = motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
        runs.append((commands, ik_calls, report['controller_preflight_timing']))
    old, new = runs
    assert old[1] == new[1]  # Every IK target and seed, not just the last pose.
    for a, b in zip(old[0], new[0]):
        fields = set(a)-{'samples', 'controller_preflight_mode'}
        assert {k:a[k] for k in fields} == {k:b[k] for k in fields}
        assert len(a['samples']) == len(b['samples'])
        for x, y in zip(a['samples'], b['samples']):
            for key in ['q', 'fraction', 'desk_clearance_mm', 'object_clearance_mm']:
                assert x[key] == y[key]
            assert y['tcp_source'] == ('controller_fk' if y['controller_checked'] else 'nominal_fk_estimate')
        assert b['samples'][0]['controller_checked'] and b['samples'][-1]['controller_checked']
        assert sum(s['controller_checked'] for s in b['samples']) == 3
    assert new[2]['queries']['getInverseKinematics']['count'] == old[2]['queries']['getInverseKinematics']['count']
    assert new[2]['controller_query_count'] < old[2]['controller_query_count'] / 3
    assert new[2]['validated_samples'] == old[2]['validated_samples']
    assert new[2]['controller_checked_samples'] == 9
    assert new[2]['nominal_fk_samples'] == new[2]['validated_samples'] - 9


@pytest.mark.parametrize('mode', ['dense', 'keypoints'])
@pytest.mark.parametrize('remote_index', [0, 1, 2])
def test_remote_joint_rejections_block_at_all_keypoints(scene, monkeypatch, mode, remote_index):
    scene.config['controller_preflight_mode'] = mode
    baseline = check(scene)
    samples = baseline[0]['samples']
    flagged = [samples[i]['q'] for i in [0, len(samples)//2, len(samples)-1]]
    q_bad = np.asarray(flagged[remote_index])
    monkeypatch.setattr(scene.backend.control, 'isJointsWithinSafetyLimits',
                        lambda q: not np.allclose(q, q_bad, atol=1e-12, rtol=0))
    with pytest.raises(RuntimeError, match='joint safety'):
        check(scene)


@pytest.mark.parametrize('mode', ['dense', 'keypoints'])
def test_interior_bad_ik_still_rejected(scene, monkeypatch, mode):
    scene.config['controller_preflight_mode'] = mode
    def ik(target, seed, *a):
        q = list(target)
        if -.301 < q[0] < -.299 and .375 < q[2] < .385:
            q[0] += .008  # Not a keypoint; beyond model+IK residual tolerance.
        return q
    monkeypatch.setattr(scene.backend.control, 'getInverseKinematics', ik)
    with pytest.raises(RuntimeError, match='reproduce target|combined tolerances'):
        check(scene)


def test_legacy_configuration_defaults_to_dense_checks(scene):
    scene.config.pop('controller_preflight_mode', None)
    report = {}
    commands = motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
    assert all(s['controller_checked'] for c in commands for s in c['samples'])
    assert report['controller_preflight_timing']['mode'] == 'dense'


def test_keypoint_selection_does_not_change_adaptive_ik_subdivision(scene, monkeypatch):
    # Synthetic kinematics amplify Z joint motion, forcing subdivision even
    # though the scheduled Cartesian spacing is unchanged (2 mm).
    def encode(tcp):
        q = np.array(tcp)
        q[2] *= 10
        return q.tolist()
    def decode(q):
        tcp = np.array(q)
        tcp[2] /= 10
        return tcp.tolist()
    scene.prepared['snapshot']['q'] = encode(scene.prepared['snapshot']['tcp'])
    for row in scene.prepared['rows']:
        row['q_nominal'] = encode(row['tcp'])
    monkeypatch.setattr(scene.backend.control, 'getForwardKinematics', lambda q,*a: decode(q))
    monkeypatch.setattr(motion.support, 'fk_tcp', lambda m,q: decode(q))
    results = []
    for mode in ['dense', 'keypoints']:
        scene.config['controller_preflight_mode'] = mode
        calls = []
        def ik(target, seed, *a):
            calls.append((list(target), list(seed)))
            return encode(target)
        monkeypatch.setattr(scene.backend.control, 'getInverseKinematics', ik)
        commands = check(scene)
        assert len(commands[1]['samples']) > 50
        results.append((calls, [[(s['fraction'], s['q']) for s in c['samples']] for c in commands]))
    assert results[0] == results[1]


def test_changed_preflight_mode_blocks_before_any_real_motion(scene, tmp_path):
    scene.config['controller_preflight_mode'] = 'keypoints'
    commands = check(scene)
    config = dict(scene.config, controller_preflight_mode='dense')
    backend = real.LiveBackend(config, tmp_path, {}, lambda: None)
    with pytest.raises(RuntimeError, match='preflight mode changed'):
        backend.move_phase(commands[0])
    assert not backend.report.get('motion_commands_attempted')


def test_fk_cache_does_not_round_close_but_different_joint_inputs(scene, monkeypatch):
    start = scene.prepared['snapshot']
    end = np.array(start['q'])
    end[0] += 1e-12
    calls = []
    monkeypatch.setattr(scene.backend.control, 'getForwardKinematics',
                        lambda q, *_: calls.append(tuple(q)) or list(q))
    goal = dict(phase='start_to_pregrasp', kind='moveJ', goal_tcp=end.tolist(),
                q_nominal=end.tolist(), closure_alpha=0.)
    commands = motion.preflight_goals(scene.backend, start, [goal], scene.model, scene.config, {})
    assert len(commands[0]['samples']) == 2
    assert set(calls) == {tuple(start['q']), tuple(end)}


@pytest.mark.parametrize('method', ['isJointsWithinSafetyLimits', 'isPoseWithinSafetyLimits'])
def test_safety_decisions_are_not_cached_for_repeated_inputs(scene, monkeypatch, method):
    seen = set()
    def reject_repeat(value):
        key = tuple(value)
        if key in seen:
            return False
        seen.add(key)
        return True
    monkeypatch.setattr(scene.backend.control, method, reject_repeat)
    report = {}
    with pytest.raises(RuntimeError, match='safety'):
        motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
    assert report['controller_preflight_timing']['status'] == 'failed'


@pytest.mark.parametrize('result', [[float('nan')]*6, [0.]*5, [-.2,-.4,.35,0.,0.,0.]])
def test_invalid_or_wrong_controller_fk_still_blocks(scene, monkeypatch, result):
    monkeypatch.setattr(scene.backend.control, 'getForwardKinematics', lambda *a: result)
    report = {}
    with pytest.raises((ValueError, RuntimeError)):
        motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
    assert report['controller_preflight_timing']['status'] == 'failed'
    assert report['controller_preflight'] == []


def test_failed_query_is_timed_and_reported(scene, monkeypatch):
    def unavailable(*args):
        raise RuntimeError('Controller unavailable')
    monkeypatch.setattr(scene.backend.control, 'getInverseKinematics', unavailable)
    report = {}
    with pytest.raises(RuntimeError, match='unavailable'):
        motion.preflight(scene.backend, scene.prepared, scene.model, scene.config, report)
    timing = report['controller_preflight_timing']
    row = timing['queries']['getInverseKinematics']
    assert timing['status'] == 'failed'
    assert row['count'] == row['errors'] == 1
    assert row['elapsed_s'] >= 0.


def test_reset_reuses_dense_primitive_checks_without_old_object_pose(scene, monkeypatch):
    from tools.ggx_reset import reset_goals
    start = scene.prepared['snapshot']
    scan_tcp = [-.2, -.3, .34, 0., 0., 0.]
    scan = dict(tcp=scan_tcp, q=scan_tcp)
    goals, already, retreat = reset_goals(start, scan, scene.config)
    assert not already and retreat == 80.
    monkeypatch.setattr(motion.sim, 'object_clearance', lambda *a: pytest.fail('Unknown moved object'))
    commands = motion.preflight_goals(scene.backend, start, goals, scene.model, scene.config, {})
    assert [c['phase'] for c in commands] == ['reset_retreat', 'reset_return']
    assert [c['kind'] for c in commands] == ['moveL', 'moveJ']
    assert commands[0]['goal_tcp'][:2] == start['tcp'][:2]
    assert commands[0]['goal_tcp'][3:] == start['tcp'][3:]
    for command in commands:
        assert command['closure_alpha'] == 0.
        assert len(command['samples']) > 2
        assert max(np.max(np.abs(np.diff([s['q'] for s in command['samples']], axis=0))), 0.) <= .01 + 1e-10
        assert all(s['desk_clearance_mm'] > 2 for s in command['samples'])


@pytest.mark.parametrize('mode', ['dense', 'keypoints'])
def test_joint_interior_collision_rejected_even_when_both_endpoints_clear(scene, monkeypatch, mode):
    scene.config['controller_preflight_mode'] = mode
    monkeypatch.setattr(motion.sim,'desk_clearance',
        lambda m,d: (-.001 if -.26 < d.qpos[0] < -.24 else .1, 'interior obstacle'))
    with pytest.raises(RuntimeError,match='hits desk'):
        check(scene)


@pytest.mark.parametrize('mode', ['dense', 'keypoints'])
def test_straight_approach_is_checked_not_only_old_detour_waypoints(scene,monkeypatch,mode):
    scene.config['controller_preflight_mode'] = mode
    for row in scene.prepared['rows']:
        if row['phase']=='approach' and .34 < row['tcp'][2] < .36:
            row['tcp'][0]=row['q_nominal'][0]=-.27  # A hypothetical old detour.
    def clearance(m,d):
        hit = abs(d.qpos[0]+.3)<1e-6 and .34<d.qpos[2]<.36
        return (-.001 if hit else .02,'interior object')
    monkeypatch.setattr(motion.sim,'object_clearance',clearance)
    with pytest.raises(RuntimeError,match='approach hits object'):
        check(scene)


def test_full_execution_has_three_arm_calls_and_unchanged_hand_sequence(scene):
    commands = check(scene)
    events = []
    backend = SimpleNamespace(open_hand=lambda x: events.append(('open',x)),
        hand_angles=lambda x: events.append(('close',x)), dwell=lambda t: events.append(('wait',t)),
        set_stage=lambda stage: events.append(('stage',stage)),
        move_phase=lambda c: events.append((c['kind'],c['phase'])))
    real.execute_checked(backend,scene.prepared,commands,scene.config,'grasp-lift')
    assert [e for e in events if e[0] in ('moveJ','moveL')] == [
        ('moveJ','start_to_pregrasp'),('moveL','approach'),('moveL','lift')]
    assert [e[1] for e in events if e[0]=='close'] == [r['hand_raw'] for r in scene.prepared['rows'] if r['phase']=='close']
    assert events[-1]==('moveL','lift')


@pytest.mark.parametrize('drift_mm', [.103883, 1.1])
def test_real_backend_close_settle_then_lift_without_hardware(scene, tmp_path, monkeypatch, drift_mm):
    commands = check(scene)
    clock = [0.]
    monkeypatch.setattr(real.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(real.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    state = deepcopy(scene.prepared['snapshot'])
    events, progress = [], [-1]
    b = real.LiveBackend(scene.config, tmp_path, {}, lambda: None)
    b.held_origin = deepcopy(state)
    b.receiver = SimpleNamespace(getRobotMode=lambda: 7, getSafetyMode=lambda: 1,
        isProtectiveStopped=lambda: False, getActualTCPPose=lambda: state['tcp'],
        getActualQ=lambda: state['q'], getActualQd=lambda: [0.]*6,
        getActualTCPSpeed=lambda: [0.]*6)
    def move(kind, target, speed, accel, asynchronous):
        c = commands[len(events)]
        assert kind == c['kind'] and asynchronous
        events.append(c['phase'])
        state.update(q=list(c['q']), tcp=list(c['goal_tcp']))
        progress[0] = -2 if progress[0] == -1 else -1
        return True
    b.control = SimpleNamespace(getAsyncOperationProgress=lambda: progress[0],
        moveJ=lambda *a: move('moveJ', *a), moveL=lambda *a: move('moveL', *a))
    b.hand = SimpleNamespace(angle_set=lambda values: None)
    b.open_hand = lambda values: b.hand_angles(values)  # Fake transport only; never construct serial.
    dwell = b.dwell
    def drift_during_close_settle(seconds):
        if b.report['stage'] == 'close_settle':
            state['tcp'][0] += drift_mm / 1000
            state['q'][0] += .000168
        dwell(seconds)
    b.dwell = drift_during_close_settle
    if drift_mm < 1:
        real.execute_checked(b, scene.prepared, commands, scene.config, 'grasp-lift')
        assert events == ['start_to_pregrasp', 'approach', 'lift']
        assert b.held_goal == dict(tcp=commands[-1]['goal_tcp'], q=commands[-1]['q'])
    else:
        with pytest.raises(RuntimeError, match='left execution target'):
            real.execute_checked(b, scene.prepared, commands, scene.config, 'grasp-lift')
        assert events == ['start_to_pregrasp', 'approach']
        assert b.report['stage'] == 'close_settle'
        assert b.report['stationary_waits'][-1]['stage'] == 'close_settle'
    closing = [row['hand_raw'] for row in scene.prepared['rows'] if row['phase'] == 'close']
    assert [c['values'] for c in b.report['hand_commands'][1:]] == closing
    assert not b.motion_active


def test_old_waypoint_records_cannot_execute(scene):
    backend = SimpleNamespace(open_hand=lambda x: pytest.fail('Must reject before any write'))
    with pytest.raises(RuntimeError,match='obsolete'):
        real.execute_checked(backend,scene.prepared,[{'phase':'approach','q':[0.]*6}],scene.config,'grasp-lift')


def test_wrist_moving_closure_cannot_silently_be_collapsed(scene):
    altered = deepcopy(scene.prepared)
    next(r for r in altered['rows'] if r['phase']=='close')['q_nominal'][0]+=.01
    with pytest.raises(ValueError,match='wrist-moving'):
        motion.phase_goals(altered)


def test_cartesian_monitor_detects_sideways_drift(scene):
    command=check(scene)[1]
    midway=motion.pose_at(command['start_tcp'],command['goal_tcp'],.5)
    assert max(motion.cartesian_path_error(midway,command)) < 1e-9
    midway[0]+=.005
    assert motion.cartesian_path_error(midway,command)[0] == pytest.approx(5)


@pytest.mark.parametrize('video', [False, True])
@pytest.mark.parametrize('collision', ['', 'close_desk', 'approach_object'])
def test_execution_geometry_gate_kept_without_video(scene, tmp_path, monkeypatch, video, collision):
    import json
    commands = check(scene)
    prepared = dict(scene.prepared, rank=0, source=dict(preflight=dict(
        candidate_yaml='fake.yml', arguments={}, object_center_m=[0., 0., 0.])))
    checked, renders = [], []
    def geometry(model, samples, required_clearance_m):
        assert required_clearance_m == .002
        checked.extend(samples)
        if collision == 'approach_object':
            next(s for s in samples if s.phase == 'approach').object_collision = True
        return (.001 if collision == 'close_desk' else .01, 'fake_geom',
                'close' if collision == 'close_desk' else '')
    monkeypatch.setattr(motion.sim, 'check_trajectory', geometry)
    monkeypatch.setattr(motion.sim, 'load_isaac_grasp_yaml',
                        lambda _: [SimpleNamespace(name='fake_candidate', confidence=.9)])
    monkeypatch.setattr(motion.sim, 'write_video', lambda *a: renders.append(a))
    if collision:
        with pytest.raises(RuntimeError, match='collision recheck'):
            motion.write_review(tmp_path, prepared, scene.model, commands, video=video)
    else:
        result = motion.write_review(tmp_path, prepared, scene.model, commands, video=video)
        assert result == (tmp_path/'execution_review.mp4' if video else None)
    report = json.loads((tmp_path/'execution_precheck.json').read_text())
    assert report['passed'] == (not collision)
    assert report['video_requested'] == video
    closes = [r for r in prepared['rows'] if r['phase'] == 'close']
    assert report['phase_samples']['close'] == len(closes)
    assert report['sample_count'] == sum(len(c['samples']) for c in commands) + len(closes)
    for sample in checked:
        if sample.phase == 'close':
            np.testing.assert_allclose(sample.qpos, commands[-2]['q'])
    assert len(renders) == int(video and not collision)


@pytest.mark.parametrize('deviation',[False,True])
@pytest.mark.parametrize('phase',['approach', 'reset_retreat'])
def test_linear_phase_is_one_moveL_and_monitors_without_intermediate_stops(scene,tmp_path,monkeypatch,deviation,phase):
    from tools import run_vision_pick_cycle
    monkeypatch.setattr(run_vision_pick_cycle,'normal',lambda r:None)
    c=check(scene)[1]
    c['phase'] = phase
    backend=real.LiveBackend(scene.config,tmp_path,{},lambda:None)
    backend.snapshot=lambda:dict(q=c['start_q'],tcp=c['start_tcp'])
    sample_rows=[c['samples'][0],c['samples'][len(c['samples'])//2],c['samples'][-1]]
    calls,waits=[],[]
    progress=iter([-1,0,0,-2])
    def moveL(*args):
        calls.append(args)
        return True
    backend.control=SimpleNamespace(moveL=moveL,moveJ=lambda *a:pytest.fail('Wrong primitive'),
                                    getAsyncOperationProgress=lambda:next(progress))
    backend.receiver=SimpleNamespace(getActualTCPSpeed=lambda:[0.]*6)
    index=[0]
    def telemetry():
        row=sample_rows[index[0]]
        tcp=np.array(row['tcp'])
        if deviation and index[0]==1:tcp[0]+=.005
        qd=np.ones(6)*.01 if index[0]<2 else np.zeros(6)
        index[0]+=1
        return tcp,np.array(row['q']),qd
    backend.telemetry=telemetry
    def wait(seconds,**kw):
        waits.append(seconds)
        return dict(q=c['q'],tcp=c['tcp'])
    backend.wait_stationary=wait
    if deviation:
        with pytest.raises(RuntimeError,match='straight path'):
            backend.move_phase(c)
        assert backend.motion_active and backend.active_kind=='moveL'
        assert waits==[]
    else:
        backend.move_phase(c)
        assert waits==[.3] and not backend.motion_active
    assert len(calls)==1
    assert calls[0]==(c['goal_tcp'],scene.config['cartesian_speed_m_s'],scene.config['cartesian_accel_m_s2'],True)
