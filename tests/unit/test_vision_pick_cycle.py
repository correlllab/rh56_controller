"""Cycle sequencing and failure gates without robot, hand, ROS or camera access."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tools import run_vision_pick_cycle as cycle


@pytest.fixture
def config():
    return cycle.load_config(cycle.ROOT / 'configs/ur5_vision_pick_cycle.example.yaml')


def test_default_cli_is_offline():
    args = cycle.parse_args(['--config', 'config.yaml', '--out', 'out', '--capture', 'saved'])
    assert not args.execute and not args.record_start
    assert not args.reset_from_low


def test_low_entry_reset_needs_separate_physical_scene_confirmation():
    argv = ['--config','c','--out','o','--execute','--confirm-cycle',cycle.CONFIRMATION,
            '--acknowledge-open-loop','--acknowledge-shape-extrapolation','--confirm-high-force',
            '--reset-from-low']
    with pytest.raises(SystemExit):
        cycle.parse_args(argv)
    args = cycle.parse_args([*argv,'--confirm-reset',cycle.RESET_CONFIRMATION])
    assert args.execute and args.reset_from_low


@pytest.mark.parametrize('extra', [
    ['--capture','saved','--reset-from-low','--confirm-reset',cycle.RESET_CONFIRMATION],
    ['--record-start','--reset-from-low','--confirm-reset',cycle.RESET_CONFIRMATION],
    ['--capture','saved','--confirm-reset',cycle.RESET_CONFIRMATION],
])
def test_reset_flag_cannot_silently_change_read_only_modes(extra):
    with pytest.raises(SystemExit):
        cycle.parse_args(['--config','c','--out','o',*extra])


def test_low_reset_is_vertical_and_does_not_change_normal_entry(config):
    current = [-.2,-.46,.213,0,2.6,-1.6]
    scan = [-.227,-.263,.343,0,2.38,-1.98]
    with pytest.raises(RuntimeError, match='explicit'):
        cycle.reset_pose_for_scan(config, current, scan)
    reset = cycle.reset_pose_for_scan(config, current, scan, True)
    np.testing.assert_allclose(np.array(reset)-current, [0,0,.130,0,0,0])
    assert current[2] == .213  # Input not modified.
    goals, limits = cycle.scan_segments(config, scan, reset)
    assert goals == [reset, scan] and limits == [150,300]
    for enabled in (False, True):
        assert cycle.reset_pose_for_scan(config, scan, scan, enabled) is None
        assert cycle.scan_segments(config, scan) == ([scan], [300])


@pytest.mark.parametrize('problem', ['too_far','low_scan','out_of_bounds','nan'])
def test_low_reset_keeps_vertical_and_workspace_caps(config, problem):
    current = [-.2,-.46,.213,0,0,0]
    scan = [-.227,-.263,.343,0,0,0]
    if problem == 'too_far':
        current[2] = .18  # Still inside workspace, but 163 mm retreat is too far.
    elif problem == 'low_scan':
        scan[2] = .28
    elif problem == 'out_of_bounds':
        current[0] = .06
    elif problem == 'nan':
        current[1] = float('nan')
    with pytest.raises((RuntimeError, ValueError)):
        cycle.reset_pose_for_scan(config, current, scan, True)


@pytest.mark.parametrize('fail_retreat', [False, True])
def test_explicit_reset_retreat_precedes_scan_and_failure_blocks_scan(config, tmp_path, fail_retreat):
    live = cycle.LiveCycle(config, SimpleNamespace(out=tmp_path), {'stage':'move_scan','low_entry_reset':{}})
    live.reset_tcp, live.scan = [1]*6, [2]*6
    calls = []
    def move(target):
        calls.append(target)
        if fail_retreat:
            raise RuntimeError('retreat failed')
    live.move_tcp = move
    if fail_retreat:
        with pytest.raises(RuntimeError):
            live.move_scan(None)
        assert calls == [live.reset_tcp]
    else:
        live.move_scan(None)
        assert calls == [live.reset_tcp,live.scan]
        assert live.report['low_entry_reset']['completed']


@pytest.mark.parametrize('extra', [
    ['--execute'],
    ['--execute', '--confirm-cycle', cycle.CONFIRMATION],
    ['--execute', '--confirm-cycle', 'wrong', '--acknowledge-open-loop', '--acknowledge-shape-extrapolation'],
    ['--execute', '--confirm-cycle', cycle.CONFIRMATION, '--acknowledge-open-loop',
     '--acknowledge-shape-extrapolation', '--capture', 'stale'],
    ['--record-start', '--capture', 'stale'],
    ['--preflight-only', '--capture', 'saved'],
    ['--preflight-only', '--capture', 'saved', '--confirm-controller-script'],
])
def test_bad_execution_flags_rejected_before_hardware(extra):
    with pytest.raises(SystemExit):
        cycle.parse_args(['--config', 'unused', '--out', 'unused', *extra])


@pytest.mark.parametrize('key,value', [
    ('closure_margin_mm', -1), ('closure_margin_mm', 3), ('closure_margin_mm', float('nan')),
    ('lift_mm', 0), ('lift_mm', 101), ('hand_force_raw', 1001), ('frames', 1),
    ('open_settle_s', -1), ('grasp_depth_mm', 40), ('max_rotation_deg', float('inf')),
    ('tcp_bounds_mm', [[0, 0, 0], [0, 1, 1]]), ('hsv_upper', [125, 200, 255]),
    ('roi', [0, 0, 1400, 720]), ('erosion_px', 2),
    ('max_scan_to_approach_mm', 301), ('max_reposition_mm', 301),
    ('stationary_wait_grace_s', 0),
    ('stationary_wait_grace_s', 6), ('stationary_wait_grace_s', float('nan')),
    ('grasp_xy_offset_base_mm', [0, -11]), ('grasp_xy_offset_base_mm', [8, 8]),
    ('grasp_xy_offset_base_mm', [0, 0, 1]), ('grasp_xy_offset_base_mm', [float('nan'), 0]),
    ('grasp_xy_offset_base_mm', [0, float('inf')]),
])
def test_invalid_config_rejected(config, tmp_path, key, value):
    config[key] = value
    p = tmp_path / 'config.yaml'
    p.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, TypeError)):
        cycle.load_config(p)


def test_explicit_force_does_not_follow_speed(config, tmp_path, monkeypatch):
    config['hand_force_raw'] = 1000
    p = tmp_path / 'config.yaml'
    p.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(cycle, 'LiveCycle', lambda *a: pytest.fail('hardware constructed'))
    with pytest.raises(ValueError, match='confirm-high-force'):
        cycle.main(['--config', str(p), '--out', str(tmp_path / 'out'), '--execute',
                    '--confirm-cycle', cycle.CONFIRMATION, '--acknowledge-open-loop',
                    '--acknowledge-shape-extrapolation'])


def test_placement_default_is_zero_and_independent_of_width(config, tmp_path):
    assert config['grasp_xy_offset_base_mm'] == [0, -5]
    config.pop('grasp_xy_offset_base_mm')
    p = tmp_path/'legacy.yaml'
    p.write_text(yaml.safe_dump(config))
    loaded = cycle.load_config(p)
    assert loaded['grasp_xy_offset_base_mm'] == [0, 0]
    assert loaded['closure_margin_mm'] == 2 and loaded['width_offset_mm'] == 10
    np.testing.assert_array_equal(cycle.grasp_placement_offset(config), [0,0,0])


def test_lift_is_only_positive_base_z():
    original = np.array([-.2, -.5, .25, .1, -2.7, 1.5])
    target = cycle.lift_pose(original, 50)
    np.testing.assert_allclose(target-original, [0, 0, .05, 0, 0, 0])
    np.testing.assert_allclose(original, [-.2, -.5, .25, .1, -2.7, 1.5])


@pytest.fixture
def lift_preflight(config, tmp_path):
    anchor = [-.2, -.46, .22, 0, 0, 0]
    current = np.array(anchor)
    state = dict(time_s=cycle.time.time(), tcp=anchor, q=[0]*6)
    recv = SimpleNamespace(getRobotMode=lambda:7, getSafetyMode=lambda:1,
        isProtectiveStopped=lambda:False, getActualTCPPose=lambda:current.tolist(),
        getActualQ=lambda:[0]*6, getActualTCPSpeed=lambda:[0]*6)
    ctrl = SimpleNamespace(getTCPOffset=lambda:[0]*6,
        isPoseWithinSafetyLimits=lambda p:True, getInverseKinematics=lambda *a:[0]*6)
    live = cycle.LiveCycle(config, SimpleNamespace(out=tmp_path), {'stage':'lift'})
    live.bridge = SimpleNamespace(_iface=SimpleNamespace(recv=recv,ctrl=ctrl))
    live.speed = .1
    return live, state, current, recv, ctrl


@pytest.mark.parametrize('noise_mm', [-.035, .035])
def test_lift_length_uses_same_snapshot_despite_static_read_noise(lift_preflight, noise_mm):
    live, start, current, _, _ = lift_preflight
    current[2] += noise_mm/1000
    goal = cycle.lift_pose(start['tcp'], 50).tolist()
    live.preflight([goal], [50], start_snapshot=start)
    report = live.report
    assert report['preflight_segments'][-1]['distance_mm'] == pytest.approx(50)
    assert report['controller_preflights'][-1]['start_snapshot'] == start
    assert len(report['controller_preflights'][-1]['stationary_origin_checks']) == 2
    np.testing.assert_array_equal(live.allowed_goals[0], goal)


def test_legacy_two_origins_reproduces_recorded_lift_false_positive(lift_preflight):
    live, start, current, _, _ = lift_preflight
    current[2] -= .000035
    with pytest.raises(RuntimeError, match='50.035000'):
        live.preflight([cycle.lift_pose(start['tcp'], 50).tolist()], [50])
    assert live.report['preflight_segments'][-1]['distance_mm'] == pytest.approx(50.035)


@pytest.mark.parametrize('problem', ['position','rotation','joints','moving','stale','future','nan','ik','safety','too_long','after_ik'])
def test_lift_snapshot_fix_preserves_fail_closed_guards(lift_preflight, problem):
    live, start, current, recv, ctrl = lift_preflight
    goal = cycle.lift_pose(start['tcp'], 50).tolist()
    if problem == 'position':
        current[2] += .0002
    elif problem == 'rotation':
        current[3] = np.deg2rad(.2)
    elif problem == 'joints':
        recv.getActualQ = lambda:[.02]*6
    elif problem == 'moving':
        recv.getActualTCPSpeed = lambda:[.002]*6
    elif problem == 'stale':
        start['time_s'] -= 2
    elif problem == 'future':
        start['time_s'] += 2
    elif problem == 'nan':
        start['tcp'][0] = float('nan')
    elif problem == 'ik':
        ctrl.getInverseKinematics = lambda *a:[.2]*6
    elif problem == 'safety':
        ctrl.isPoseWithinSafetyLimits = lambda p:False
    elif problem == 'too_long':
        goal[2] += .001
    elif problem == 'after_ik':
        def ik(*args):
            current[2] += .0002
            return [0]*6
        ctrl.getInverseKinematics = ik
    with pytest.raises((RuntimeError, ValueError)):
        live.preflight([goal], [50], start_snapshot=start)
    assert 'controller_preflights' not in live.report
    assert not hasattr(live, 'allowed_goals')


def test_lift_passes_its_exact_snapshot_and_never_changes_hand(lift_preflight):
    live, _, current, _, _ = lift_preflight
    plan = SimpleNamespace(tcp=[current.tolist()])
    calls = []
    def preflight(goals, limits, *, start_snapshot):
        calls.append('preflight')
        np.testing.assert_array_equal(goals[0], cycle.lift_pose(start_snapshot['tcp'], 50))
        assert limits == [50]
    def move(goal):
        calls.append('move')
        current[:] = goal
    live.preflight, live.move_tcp = preflight, move
    live.lift(plan)
    assert calls == ['preflight','move']
    np.testing.assert_allclose(np.array(live.report['lift_goal_tcp'])-plan.tcp[0], [0,0,.05,0,0,0])
    assert live.hand is None


def test_direct_approach_without_extra_hover_preserves_other_limits(config, tmp_path):
    assert config['max_reposition_mm'] == 300
    assert config['max_scan_to_approach_mm'] == 300
    plan = SimpleNamespace(tcp=[[1]*6, [2]*6], lift=[3]*6)
    goals, limits = cycle.pick_segments(config, plan)
    assert goals == [*plan.tcp, plan.lift]
    assert len(goals) == len(limits) == 3
    assert limits == [300, 50, 50]
    # The approach fallback still follows each config, not a hard-coded 300 mm.
    config['max_reposition_mm'] = 250
    config.pop('max_scan_to_approach_mm')
    config.pop('stationary_wait_grace_s')
    p = tmp_path/'old.yaml'
    p.write_text(yaml.safe_dump(config))
    loaded = cycle.load_config(p)
    assert loaded['max_scan_to_approach_mm'] == 250
    assert loaded['stationary_wait_grace_s'] == 3


def test_lab_disables_only_scan_to_approach_length():
    c = cycle.load_config(cycle.ROOT / 'configs/ur5_iterative_cube_lab.yaml')
    assert c['max_scan_to_approach_mm'] is None
    assert c['max_reposition_mm'] == 300
    plan = SimpleNamespace(tcp=[[1]*6, [2]*6], lift=[3]*6)
    goals, limits = cycle.pick_segments(c, plan)
    assert goals == [*plan.tcp, plan.lift]
    assert limits == [None, 50, 50]
    assert cycle.scan_segments(c, [0]*6)[1] == [300]


@pytest.mark.parametrize('key', ['max_reposition_mm', 'max_plan_segment_mm', 'lift_mm',
                                'max_rotation_deg'])
def test_other_motion_limits_cannot_be_disabled(config, tmp_path, key):
    config[key] = None
    p = tmp_path / 'invalid.yaml'
    p.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, TypeError)):
        cycle.load_config(p)


@pytest.mark.parametrize('problem', [None, 'legacy_cap', 'safety', 'ik', 'rotation', 'bounds'])
def test_recorded_300_873mm_approach_still_checks_full_path(lift_preflight, problem):
    live, _, current, _, ctrl = lift_preflight
    live.report['stage'] = 'preflight_pick'
    # Recorded failed run 20260909T202015_539047Z; no hardware or local logs needed.
    current[:] = [-.2266804524749249, -.2630880624484019, .34334381066192887,
                  .020462670539907118, 2.380656935347525, -1.9752007638925995]
    goal = [-.21105522426274526, -.5565012797434246, .2786207599984718,
            -.019704671313734396, -2.966403996038477, .9429919321256709]
    checked = []
    def safety(p):
        checked.append(p)
        return problem != 'safety'
    ctrl.isPoseWithinSafetyLimits = safety
    if problem == 'ik':
        ctrl.getInverseKinematics = lambda *a:[.2]*6
    if problem == 'rotation':
        live.c['max_rotation_deg'] = 1
    if problem == 'bounds':
        goal[0] = .06
    limit = 300 if problem == 'legacy_cap' else None
    if problem:
        with pytest.raises(RuntimeError):
            live.preflight([goal], [limit])
        assert 'controller_preflights' not in live.report
    else:
        live.preflight([goal], [None])
        assert len(checked) >= 62  # Sample the entire segment at <=5 mm spacing.
        np.testing.assert_allclose(checked[0], current)
        np.testing.assert_allclose(checked[-1], goal)
        segment = live.report['preflight_segments'][0]
        assert segment['distance_mm'] == pytest.approx(300.8729584513758)
        assert segment['limit_mm'] is None
        json.dumps(live.report, allow_nan=False)  # No Infinity sentinel in reports.


@pytest.mark.parametrize('problem', [None, 'old_limit', 'too_far', 'safety', 'ik', 'rotation', 'bounds'])
def test_automatic_return_cap_keeps_controller_and_route_guards(config, tmp_path, problem):
    live = cycle.LiveCycle(config, SimpleNamespace(out=tmp_path), {})
    current = np.array([-.2, -.55, .33, 0, 0, 0])
    goal = current.copy()
    goal[1] += .301 if problem == 'too_far' else .281
    if problem == 'rotation':
        goal[3] = np.deg2rad(61)
    if problem == 'bounds':
        goal[0] = .06
    calls = []
    def ik(*args):
        calls.append('ik')
        return [.16]*6 if problem == 'ik' else [0]*6
    # Only read/query APIs exist: preflight cannot move or open the hand.
    recv = SimpleNamespace(getRobotMode=lambda:7, getSafetyMode=lambda:1,
        isProtectiveStopped=lambda:False, getActualTCPPose=lambda:current.tolist(),
        getActualQ=lambda:[0]*6)
    ctrl = SimpleNamespace(getTCPOffset=lambda:[0]*6,
        isPoseWithinSafetyLimits=lambda p:problem != 'safety', getInverseKinematics=ik)
    live.bridge = SimpleNamespace(_iface=SimpleNamespace(recv=recv, ctrl=ctrl))
    live.speed = .1
    limit = 250 if problem == 'old_limit' else config['max_reposition_mm']
    if problem:
        with pytest.raises(RuntimeError):
            live.preflight([goal.tolist()], [limit])
        assert 'controller_preflights' not in live.report
    else:
        live.preflight([goal.tolist()], [limit])
        assert len(calls) >= 58
        np.testing.assert_array_equal(live.allowed_goals, [goal])
        assert live.next_goal == 0


@pytest.mark.parametrize('key', ['hover_mm', 'max_scan_to_hover_mm'])
def test_old_hover_config_requires_explicit_route_migration(config, tmp_path, key):
    config[key] = 50
    p = tmp_path/'old.yaml'
    p.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='Extra hover removed'):
        cycle.load_config(p)


class FakeBackend:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def __getattr__(self, name):
        assert name in cycle.STAGES
        def operation(value):
            self.calls.append(name)
            if name == self.fail:
                raise RuntimeError(name)
            return value
        return operation


def test_single_cycle_order_no_retry_or_release():
    backend = FakeBackend()
    report = {'events': []}
    checkpoints = []
    cycle.run_cycle(backend, report, lambda: checkpoints.append(report['stage']))
    assert backend.calls == ['prepare', 'open_hand', 'move_scan', 'detect', 'plan',
                             'preflight_pick', 'grasp', 'settle', 'lift']
    assert len(checkpoints) == 2*len(cycle.STAGES)
    assert report['status'] == 'cycle_commands_completed_grasp_unverified'


@pytest.mark.parametrize('stage', cycle.STAGES)
def test_failure_prevents_every_subsequent_stage(stage):
    backend = FakeBackend(fail=stage)
    report = {'events': []}
    with pytest.raises(RuntimeError, match=stage):
        cycle.run_cycle(backend, report)
    assert backend.calls == list(cycle.STAGES[:cycle.STAGES.index(stage)+1])
    assert report['events'][-1]['event'] == 'start'


def test_target_stability_requires_multiple_consistent_frames():
    reports = [{'estimated_target_top_center_base_mm': p} for p in
               ([0, 0, 80], [1, 0, 81], [0, 1, 80])]
    assert cycle.stable_target(reports, 3) < 3
    reports[-1]['estimated_target_top_center_base_mm'] = [5, 0, 80]
    with pytest.raises(RuntimeError, match='unstable'):
        cycle.stable_target(reports, 3)
    with pytest.raises(ValueError, match='At least two'):
        cycle.stable_target(reports[:1], 3)


@pytest.fixture
def saved_capture(config, tmp_path):
    cv2 = pytest.importorskip('cv2')
    capture = tmp_path / 'capture'
    capture.mkdir()
    camera = dict(serial=config['camera_serial'], stream=dict(width=1280, height=720),
                  depth=dict(scale_m_per_unit=.001))
    (capture / 'camera_check.yaml').write_text(yaml.safe_dump(dict(
        schema='rh56_realsense_camera_check/v1', camera=camera)))
    T = np.diag([1., -1., -1., 1.])
    T[2, 3] = .6
    cal = tmp_path / 'calibration.yaml'
    cal.write_text(yaml.safe_dump(dict(
        schema='rh56_ur5_external_camera_calibration/v2', camera=camera,
        selected_intrinsics=dict(camera_matrix=[[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]],
                                 distortion_coefficients=[0.]*5),
        transforms=dict(base_from_camera=dict(matrix=T.tolist())))))
    desk = tmp_path / 'desk.yaml'
    desk.write_text(yaml.safe_dump({'plane': {'normal_base': [0, 0, 1]}}))
    config.update(calibration=str(cal), desk_plane=str(desk), desk_z_mm=0,
                  object_height_mm=100, target_bounds_mm=[[-100, -100, 90], [100, 100, 110]])
    image = np.zeros((720, 1280, 3), np.uint8)
    image[300:360, 600:660] = [255, 100, 0]
    cv2.imwrite(str(capture/'camera_check_rgb.png'), image)
    cv2.imwrite(str(capture/'camera_check_depth_u16.png'), np.full((720, 1280), 500, np.uint16))
    return config, capture, image


def test_saved_detector_reuses_calibrated_target_without_hardware(saved_capture, tmp_path):
    c, capture, _ = saved_capture
    path, report = cycle.detect_saved(c, capture, tmp_path/'detected')
    assert path.exists() and report['quality_pass']
    assert report['estimated_target_top_center_base_mm'][2] == pytest.approx(100)


def test_main_preview_cannot_construct_hardware(saved_capture, tmp_path, monkeypatch):
    c, capture, _ = saved_capture
    for key in ('local_alignment', 'alignment_validation'):
        p = tmp_path/f'{key}.json'
        p.write_text('{}')
        c[key] = str(p)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(cycle, 'LiveCycle', lambda *a: pytest.fail('hardware constructed'))
    monkeypatch.setattr(cycle, 'Plan', lambda *a: SimpleNamespace(report={'offline':True}, close=lambda: None))
    out = tmp_path/'preview'
    assert cycle.main(['--config', str(config), '--capture', str(capture), '--out', str(out)]) == 0
    report = json.loads((out/'summary.json').read_text())
    assert report['status'] == 'offline_preview_only'
    assert report['hardware_commands_attempted'] is False
    assert (out/'summary.csv').exists()


@pytest.mark.parametrize('ik_fails', [False, True])
@pytest.mark.parametrize('reset', [False, True])
def test_preflight_mode_never_calls_motion_hand_or_full_cycle(saved_capture, tmp_path, monkeypatch, ik_fails, reset):
    c, capture, _ = saved_capture
    for key in ('local_alignment', 'alignment_validation'):
        p = tmp_path/f'{key}.json'
        p.write_text('{}')
        c[key] = str(p)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(c))
    calls = []
    plan = SimpleNamespace(report={'hardware_ik_validated':False},
                           tcp=[[0]*6, [0]*6], lift=[0]*6)
    class PreflightBackend:
        def __init__(self, c, args, report):
            self.scan = [1]*6
            self.reset_tcp = [2]*6 if reset else None
            self.report = report
        def plan(self, target):
            calls.append('plan')
            self.report['plan'] = plan.report
            return plan
        def prepare(self, unused):
            calls.append('prepare')
        def preflight(self, goals, limits):
            calls.append('preflight')
            expected = [self.scan, *plan.tcp, plan.lift]
            expected_limits = [300,300,50,50]
            if reset:
                expected.insert(0, self.reset_tcp)
                expected_limits.insert(0, 150)
            assert goals == expected
            assert limits == expected_limits
            if ik_fails:
                raise RuntimeError('IK failed')
        def close(self, failed):
            # Preflight does not request emergency stopL: it never moved.
            assert failed is False
            calls.append('close')
    monkeypatch.setattr(cycle, 'LiveCycle', PreflightBackend)
    monkeypatch.setattr(cycle, 'run_cycle', lambda *a: pytest.fail('Motion cycle entered'))
    out = tmp_path/'preflight'
    argv = ['--config', str(config), '--capture', str(capture), '--out', str(out),
            '--preflight-only', '--confirm-controller-script', '--acknowledge-shape-extrapolation']
    if reset:
        argv += ['--reset-from-low','--confirm-reset',cycle.RESET_CONFIRMATION]
    if ik_fails:
        with pytest.raises(RuntimeError, match='IK failed'):
            cycle.main(argv)
    else:
        cycle.main(argv)
    report = json.loads((out/'summary.json').read_text())
    assert calls == ['plan', 'prepare', 'preflight', 'close']
    assert report['hardware_commands_attempted'] is False
    assert report['status'] == ('failed' if ik_fails else 'controller_preflight_passed_no_motion')


def test_record_start_only_uses_receive_interface(config, tmp_path, monkeypatch):
    from tools import calibrate_ur5_external_camera_real as camera
    calls = []
    receiver = SimpleNamespace(getRobotMode=lambda:7, getSafetyMode=lambda:1,
        isProtectiveStopped=lambda:False, getActualTCPSpeed=lambda:[0]*6,
        getActualTCPPose=lambda:[-.2, -.4, .4, 0, 0, 0], getActualQ=lambda:[0]*6,
        disconnect=lambda:calls.append('disconnect'))
    def receive(ip):
        calls.append(('receive', ip))
        return receiver
    monkeypatch.setattr(camera, '_open_rtde_receiver', receive)
    monkeypatch.setattr(cycle, 'LiveCycle', lambda *a: pytest.fail('control constructed'))
    monkeypatch.setattr(cycle, 'Plan', lambda *a: pytest.fail('planner constructed'))
    p = tmp_path/'config.yaml'
    p.write_text(yaml.safe_dump(config))
    out = tmp_path/'start'
    cycle.main(['--config', str(p), '--out', str(out), '--record-start'])
    result = json.loads((out/'start_pose.json').read_text())
    assert result['schema'] == 'rh56_vision_pick_start/v1'
    assert result['clearance_verified'] is False
    assert calls == [('receive', config['ur_ip']), 'disconnect']


@pytest.mark.parametrize('problem', ['missing', 'ambiguous', 'camera_mismatch', 'height', 'bounds', 'truncated'])
def test_bad_detection_blocks_planning(saved_capture, tmp_path, problem):
    cv2 = pytest.importorskip('cv2')
    c, capture, image = saved_capture
    if problem == 'missing':
        image[:] = 0
    elif problem == 'ambiguous':
        image[400:460, 700:760] = [255, 100, 0]
    elif problem == 'camera_mismatch':
        c['camera_serial'] = 'different'
    elif problem == 'height':
        c['object_height_mm'] = 40
    elif problem == 'bounds':
        c['target_bounds_mm'] = [[100, 100, 90], [200, 200, 110]]
    elif problem == 'truncated':
        c['roi'] = [610, 50, 1150, 680]
    cv2.imwrite(str(capture/'camera_check_rgb.png'), image)
    with pytest.raises(RuntimeError):
        cycle.detect_saved(c, capture, tmp_path/'detected')


@pytest.mark.parametrize('margin', [1, 2])
@pytest.mark.parametrize('xy_offset', [[0, 0], [0, -5], [3, 4]])
def test_original_plan_and_separate_margin_preserved(config, tmp_path, monkeypatch, margin, xy_offset):
    pytest.importorskip('mujoco')
    config['closure_margin_mm'] = margin
    config['grasp_xy_offset_base_mm'] = xy_offset
    from rh56_controller.grasp_executor import GraspExecutor
    from tools import run_vision_plan_trial as existing
    path = tmp_path/'target.yaml'
    path.write_text(yaml.safe_dump(dict(schema='rh56_ur5_camera_colored_target_validation/v1',
        quality_pass=True, estimated_target_top_center_base_mm=[-190, -637, 82],
        physical_reference={'target_height_mm': 40})))
    validation = tmp_path/'validation.json'
    validation.write_text(json.dumps({'hand_command_assumed': [1000, 762, 727, 778, 727, 0]}))
    config['alignment_validation'] = str(validation)
    monkeypatch.setattr(existing, 'load_local_alignment', lambda *a: (None, {'scope':'test'}))
    plan = cycle.Plan(config, path)
    try:
        report = plan.report
        assert report['target']['object_width_mm'] == 40
        assert report['target']['width_offset_mm'] == 10-margin
        assert report['target']['planner_width_m'] == pytest.approx((50-margin)/1000)
        assert report['closure_margin_mm'] == margin
        assert report['final_shape_extrapolated'] is True
        assert not report['hardware_ik_validated']
        final = plan.core.closure.solve(plan.core._mode, (50-margin)/1000)
        original = plan.core._compute_plan_closures(10, final)
        mapper = GraspExecutor(None, None, plan.core.fk)
        assert plan.commands == [mapper._ctrl_to_real(r.ctrl_values) for r in original]
        original_tcp = [existing.pose(existing.UR5Bridge.hand_T_to_tcp_T(
            existing.command_hand_pose(plan.core._build_world_T_hand(r), None))) for r in original]
        expected = np.array(original_tcp)
        expected[:,:2] += np.array(xy_offset)/1000
        np.testing.assert_allclose(plan.tcp, expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(np.array(plan.tcp)[:,2:], np.array(original_tcp)[:,2:])
        np.testing.assert_allclose(np.diff(np.array(plan.tcp)[:,:3], axis=0),
                                   np.diff(np.array(original_tcp)[:,:3], axis=0), atol=1e-12, rtol=0)
        assert report['grasp_xy_offset_base_mm'] == xy_offset
        assert report['target']['center_base_m'] == (-.190, -.637, .072)
        np.testing.assert_allclose(report['commanded_grasp_center_base_m'],
                                   np.array(report['target']['center_base_m']) + np.r_[xy_offset,0]/1000)
        assert report['entry_policy'] == 'direct_to_original_plan_approach'
        assert 'hover_tcp' not in report and not hasattr(plan, 'hover')
        np.testing.assert_allclose(np.array(plan.lift)-plan.tcp[-1], [0, 0, .05, 0, 0, 0])
        assert plan.core._arm is None and plan.core._hand is None
    finally:
        plan.close()


def test_old_executor_caught_serial_error_cannot_report_success_or_lift(config, tmp_path, monkeypatch):
    from rh56_controller import grasp_executor
    live = cycle.LiveCycle(config, SimpleNamespace(out=tmp_path), {'stage':'grasp'})
    live.hand = SimpleNamespace(angle_set=lambda c: (_ for _ in ()).throw(OSError('serial broken')))
    live.capture_time = cycle.time.monotonic()
    live.monitor = lambda fn: fn()
    class Executor:
        def __init__(self, arm, hand, fk, status_cb, **kwargs):
            self.hand, self.status, self._abort = hand, status_cb, cycle.threading.Event()
        def execute_plan_waypoints(self, waypoints, force_N):
            try:
                self.hand.angle_set([1]*6)
            except Exception:
                pass  # Emulate the legacy executor's swallowed finger exception.
            self.status('Plan: complete.')
        def is_running(self):
            return False
    monkeypatch.setattr(grasp_executor, 'GraspExecutor', Executor)
    plan = SimpleNamespace(core=SimpleNamespace(fk=None), waypoints=[], commands=[[1]*6])
    with pytest.raises(RuntimeError, match='do not lift'):
        live.grasp(plan)
    assert live.cancelled.is_set() and live.serial_errors == ['serial broken']


def test_cleanup_still_closes_hand_after_stop_failure(config, tmp_path):
    live = cycle.LiveCycle(config, SimpleNamespace(out=tmp_path), {})
    closed = []
    def bad_stop(_):
        raise OSError('stop failed')
    live.bridge = SimpleNamespace(_iface=SimpleNamespace(ctrl=SimpleNamespace(stopL=bad_stop)),
                                  disconnect=lambda: closed.append('arm'))
    live.hand = SimpleNamespace(ser=SimpleNamespace(close=lambda: closed.append('hand')))
    live.close(failed=True)
    assert closed == ['arm', 'hand']
    assert live.report['stopL_requested_on_failure']
    assert 'stop failed' in live.report['cleanup_errors'][0]


@pytest.fixture
def stationary_wait(config, tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.)
    def sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(cycle.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(cycle.time, 'sleep', sleep)
    recv = SimpleNamespace(getRobotMode=lambda:7, getSafetyMode=lambda:1,
        isProtectiveStopped=lambda:False, getActualTCPSpeed=lambda:[0]*6,
        getActualTCPPose=lambda:[-.2, -.5, .33, 0, 0, 0], getActualQ=lambda:[0]*6)
    live = cycle.LiveCycle(config, SimpleNamespace(out=tmp_path), {'stage':'settle'})
    # No controller/hand interface: a wait must never issue a command.
    live.bridge = SimpleNamespace(_iface=SimpleNamespace(recv=recv))
    return live, recv, clock


def test_stationary_wait_accepts_transient_then_full_stable_dwell(stationary_wait):
    live, recv, clock = stationary_wait
    recv.getActualTCPSpeed = lambda: [.002 if clock.now < .1 else 0, 0, 0, 0, 0, 0]
    live.wait_stationary(.3)
    record = live.report['stationary_waits'][0]
    assert record['status'] == 'stable'
    assert .4 <= record['elapsed_s'] < .43
    assert record['samples'][0]['tcp_speed'][0] == .002
    assert record['samples'][-1]['stationary']
    json.dumps(record, allow_nan=False)


def test_stationary_wait_spike_resets_continuous_dwell(stationary_wait):
    live, recv, clock = stationary_wait
    # Rotation speed counts too; do not silently relax the old combined gate.
    recv.getActualTCPSpeed = lambda: [0, 0, 0, 0, 0, .002 if .1 <= clock.now < .2 else 0]
    live.wait_stationary(.3)
    assert .5 <= clock.now < .55


def test_stationary_wait_keeps_legacy_combined_speed_gate(stationary_wait):
    live, recv, clock = stationary_wait
    recv.getActualTCPSpeed = lambda: [.0008, 0, 0, .0008, 0, 0]
    with pytest.raises(RuntimeError, match='timed out'):
        live.wait_stationary(.3)
    assert clock.now == pytest.approx(3.3)


@pytest.mark.parametrize('intermittent', [False, True])
def test_stationary_wait_timeout_cannot_be_extended_by_motion(stationary_wait, intermittent):
    live, recv, clock = stationary_wait
    recv.getActualTCPSpeed = lambda: [
        .002 if not intermittent or int(clock.now*10) % 2 else 0, 0, 0, 0, 0, 0]
    with pytest.raises(RuntimeError, match='timed out'):
        live.wait_stationary(.3)
    assert clock.now == pytest.approx(3.3)
    record = live.report['stationary_waits'][0]
    assert record['status'] == 'failed' and 'timed out' in record['error']


@pytest.mark.parametrize('method,value,error', [
    ('getRobotMode', 5, 'normal running/safety'),
    ('getSafetyMode', 3, 'normal running/safety'),
    ('isProtectiveStopped', True, 'normal running/safety'),
    ('getActualTCPSpeed', [float('nan')]*6, 'finite'),
    ('getActualTCPSpeed', [0]*3, 'finite'),
    ('getActualTCPPose', [float('inf')]*6, 'finite'),
    ('getActualQ', [], 'finite'),
    ('getActualTCPPose', [-.2, -.5, .05, 0, 0, 0], 'outside reviewed bounds'),
])
def test_stationary_wait_safety_and_bad_telemetry_fail_immediately(stationary_wait, method, value, error):
    live, recv, clock = stationary_wait
    setattr(recv, method, lambda: value)
    with pytest.raises((RuntimeError, ValueError), match=error):
        live.wait_stationary(.3)
    assert clock.now == 0
    assert live.report['stationary_waits'][0]['status'] == 'failed'


def test_stationary_wait_cancellation_prevents_next_stage(stationary_wait):
    live, _, _ = stationary_wait
    live.cancelled.set()
    with pytest.raises(RuntimeError, match='cancelled'):
        live.settle(None)
    assert live.report['stationary_waits'][0]['status'] == 'failed'


def test_stationary_snapshot_still_rejects_moving_entry(stationary_wait):
    _, recv, _ = stationary_wait
    recv.getActualTCPSpeed = lambda: [.002, 0, 0, 0, 0, 0]
    with pytest.raises(RuntimeError, match='must be stationary'):
        cycle.stationary_snapshot(recv)
