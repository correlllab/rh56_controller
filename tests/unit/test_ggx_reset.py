"""Reset-before-perception regressions: fake transports only, no hardware."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import ggx_reset as reset, run_ggx_review as review


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = reset.real.load_config(reset.real.ROOT / 'configs/ggx_real_lab.yaml')
    # Low-setting fixture for tests without the explicit high-setting flag.
    config.update(hand_speed_raw=100, hand_force_raw=100)
    config_path = tmp_path / 'real.yaml'
    config_path.write_text(yaml.safe_dump(config))
    plane = tmp_path / 'desk.yaml'
    plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    calibration = tmp_path / 'camera.yaml'
    calibration.write_text('fixture: no camera\n')
    perception = dict(schema='rh56_tabletop_vision_config/v1', camera_serial='fake',
        calibration=str(calibration), desk_plane=str(plane), desk_z_mm=34, roi=[320,60,1100,620])
    vision_path = tmp_path / 'vision.yaml'
    vision_path.write_text(yaml.safe_dump(perception))
    start = dict(schema='rh56_vision_pick_start/v1', ur_ip=config['ur_ip'],
                 tcp=[-.2,-.3,.35,0.,0.,0.], q=[0.]*6)
    start_path = tmp_path / 'start.json'
    start_path.write_text(json.dumps(start))
    args = SimpleNamespace(reset_first=True, capture_live=True, execute=False, allow_high_force=False,
        real_config=config_path, config=vision_path, start_snapshot=start_path)
    events, faults = [], set()
    state = dict(tcp=[-.25,-.5,.32,0.,0.,0.], q=[.1]*6)
    class Backend:
        def __init__(self, config, out, report, checkpoint):
            self.report = report
            events.append('construct')
            self.control = SimpleNamespace(getTCPOffset=lambda: [0.]*6)
        def open_receiver(self): events.append('receive')
        def snapshot(self): return deepcopy(state)
        def held_snapshot(self): return self.snapshot()
        def open_control(self): events.append('control')
        def open_hand(self, command):
            events.append('open')
            self.report['hand_commands'] = [dict(kind='angles', values=command)]
            if 'open' in faults: raise RuntimeError('mock opening failure')
        def move_phase(self, command):
            events.append(command['phase'])
            self.report['motion_commands_attempted'] = True
            if 'move' in faults: raise RuntimeError('mock motion failure')
            state.update(q=list(command['q_nominal']), tcp=list(command['goal_tcp']))
        def close(self):
            events.append('disconnect')
            self.report['cleanup_errors'] = ['mock disconnect failure'] if 'cleanup' in faults else []
    def preflight(backend, snapshot, goals, model, config, report):
        events.append('preflight')
        if 'preflight' in faults: raise RuntimeError('mock IK failure')
        assert all(goal['closure_alpha'] == 0. for goal in goals)
        report['controller_preflight'] = deepcopy(goals)
        return goals
    monkeypatch.setattr(reset.motion, 'preflight_goals', preflight)
    def model(source, *, include_object):
        assert include_object is False
        return object()
    monkeypatch.setattr(reset.support, 'build_model', model)
    monkeypatch.setattr(reset.support, 'raw_hand_command', lambda model, alpha: [1000]*5+[0])
    monkeypatch.setattr(reset.real, 'confirm', lambda prompt: events.append('confirm_reset') or True)
    out = tmp_path / 'reset'
    return SimpleNamespace(args=args, out=out, config=config, state=state, start=start,
        perception=perception, events=events, faults=faults, backend=Backend,
        run=lambda: reset.run_reset(args, out, backend_factory=Backend))


def test_reset_orders_preflight_open_retreat_return_disconnect(setup):
    t = setup
    rc, snapshot_path = t.run()
    assert rc == 0
    assert t.events == ['construct', 'receive', 'confirm_reset', 'control', 'preflight',
                        'open', 'preflight', 'reset_retreat', 'reset_return', 'disconnect']
    snapshot = review.read_json(snapshot_path)
    assert snapshot['tcp'] == t.start['tcp'] and snapshot['q'] == t.start['q']
    report = review.read_json(t.out / 'summary.json')
    assert report['status'] == 'reset_completed' and report['retreat_mm'] == 80
    assert report['goals'][0]['goal_tcp'][2] == pytest.approx(.4)
    assert (t.out / 'summary.csv').is_file()
    assert report['reset_motion']['cartesian_speed_m_s'] == .01
    assert report['reset_motion']['joint_speed_rad_s'] == .10
    assert t.config['cartesian_speed_m_s'] == .01  # Grasp config is unchanged.
    assert [r['stage'] for r in report['timings']] == [
        'connect_controller', 'preflight', 'hand_open', 'post_open_preflight', 'reset_retreat', 'reset_return']
    assert all(r['status'] == 'completed' and r['elapsed_s'] >= 0 for r in report['timings'])


def test_finished_lift_omits_duplicate_retreat_but_checks_return(setup):
    path=setup.out.parent/'lift.json'
    path.write_text(json.dumps(dict(status='commands_completed_grasp_unverified',requested_phase='grasp-lift',
        lift_mm=80.,lift_hold_completed=True,grasp_close_command_attempted=True,cleanup_errors=[],
        final_snapshot=setup.state)))
    setup.args.completed_lift_report=path
    rc,_=setup.run()
    assert rc==0
    assert setup.events.index('preflight') < setup.events.index('open') < setup.events.index('reset_return')
    assert 'reset_retreat' not in setup.events
    assert review.read_json(setup.out/'summary.json')['retreat_mm']==0


@pytest.mark.parametrize('shift, succeeds', [(.000104211, True), (.002, False)])
def test_opening_shift_rebuilds_path_only_within_fixed_arrival_bound(setup, monkeypatch, shift, succeeds):
    old = deepcopy(setup.state)
    original = setup.backend.open_hand
    def opening(self, command):
        assert self.held_goal == old
        original(self, command)
        setup.state['tcp'][0] += shift
    monkeypatch.setattr(setup.backend, 'open_hand', opening)
    rc, _ = setup.run()
    assert (rc == 0) == succeeds
    report = review.read_json(setup.out/'summary.json')
    if succeeds:
        assert setup.events.count('preflight') == 2
        assert report['post_open_origin_refresh']['previous'] == old
        assert report['goals'][0]['goal_tcp'][0] == pytest.approx(old['tcp'][0]+shift)
    else:
        assert not report['motion_commands_attempted']
        assert 'reset_return' not in setup.events


def test_drift_after_post_open_preflight_still_blocks_arm(setup, monkeypatch):
    original = reset.motion.preflight_goals
    def preflight(*args):
        result = original(*args)
        if setup.events.count('preflight') == 2:
            setup.state['tcp'][0] += .000130422
        return result
    monkeypatch.setattr(reset.motion, 'preflight_goals', preflight)
    assert setup.run()[0] == 1
    report = review.read_json(setup.out/'summary.json')
    assert 'open' in setup.events and not report['motion_commands_attempted']
    assert 'Preflight start drifted' in report['error']


def test_manual_recovery_uses_current_pose_without_opening_or_old_lift(setup):
    setup.args.recover_from_current = True
    setup.args.completed_lift_report = setup.out.parent/'nonexistent_old_lift.json'
    assert setup.run()[0] == 0
    assert setup.events == ['construct', 'receive', 'confirm_reset', 'control',
                            'preflight', 'reset_return', 'disconnect']
    report = review.read_json(setup.out/'summary.json')
    assert report['retreat_mm'] == 0 and not report.get('hand_commands')
    assert report['manual_current_recovery']['opening_command_skipped']
    assert not report['manual_current_recovery']['old_lift_target_used']


@pytest.mark.parametrize('failure', ['low', 'cancel', 'preflight'])
def test_manual_recovery_still_requires_height_confirmation_and_preflight(setup, monkeypatch, failure):
    setup.args.recover_from_current = True
    if failure == 'low': setup.state['tcp'][2] = .29
    if failure == 'cancel': monkeypatch.setattr(reset.real, 'confirm', lambda _: False)
    if failure == 'preflight': setup.faults.add('preflight')
    assert setup.run()[0] != 0
    report = review.read_json(setup.out/'summary.json')
    assert not report['motion_commands_attempted'] and not report.get('hand_commands')
    assert 'entry_snapshot' in report


def test_completed_lift_pose_change_stops_before_control(setup):
    path=setup.out.parent/'lift.json'
    target=deepcopy(setup.state);target['tcp'][2]+=.02
    path.write_text(json.dumps(dict(status='commands_completed_grasp_unverified',requested_phase='grasp-lift',
        lift_mm=80.,lift_hold_completed=True,grasp_close_command_attempted=True,cleanup_errors=[],final_snapshot=target)))
    setup.args.completed_lift_report=path
    assert setup.run()[0]==1
    assert 'control' not in setup.events and 'open' not in setup.events


def test_finished_lift_does_not_bypass_failed_return_preflight(setup):
    path=setup.out.parent/'lift.json'
    path.write_text(json.dumps(dict(status='commands_completed_grasp_unverified',requested_phase='grasp-lift',
        lift_mm=80.,lift_hold_completed=True,grasp_close_command_attempted=True,cleanup_errors=[],final_snapshot=setup.state)))
    setup.args.completed_lift_report=path;setup.faults.add('preflight')
    assert setup.run()[0]==1
    assert 'open' not in setup.events and 'reset_return' not in setup.events


def test_cancel_before_control_or_hand(setup, monkeypatch):
    monkeypatch.setattr(reset.real, 'confirm', lambda _: False)
    rc, snapshot = setup.run()
    assert rc == 2 and snapshot is None
    assert setup.events == ['construct', 'receive', 'disconnect']


@pytest.mark.parametrize('allowed', [False, True])
def test_reset_1000_settings_need_flag_and_are_not_silently_capped(setup, allowed):
    setup.config.update(hand_speed_raw=1000, hand_force_raw=1000)
    setup.args.real_config.write_text(yaml.safe_dump(setup.config))
    setup.args.allow_high_force = allowed
    rc, _ = setup.run()
    report = review.read_json(setup.out/'summary.json')
    if allowed:
        assert rc == 0
        assert report['opening_speed_raw'] == report['opening_force_raw'] == 1000
        assert setup.events.index('confirm_reset') < setup.events.index('open')
    else:
        assert rc == 1 and not setup.events
        assert '--allow-high-force' in report['error']


@pytest.mark.parametrize('fault', ['preflight', 'open', 'move', 'cleanup'])
def test_reset_fault_never_reports_ready_for_perception(setup, fault):
    setup.faults.add(fault)
    rc, snapshot = setup.run()
    assert rc == 1 and snapshot is None
    assert setup.events[-1] == 'disconnect'
    if fault == 'preflight': assert 'open' not in setup.events
    if fault in ('preflight', 'open', 'move'): assert 'reset_return' not in setup.events


def test_already_at_start_still_checks_and_opens_but_never_moves(setup):
    setup.state.update(q=list(setup.start['q']), tcp=list(setup.start['tcp']))
    rc, _ = setup.run()
    assert rc == 0
    assert 'preflight' in setup.events and 'open' in setup.events
    assert 'reset_retreat' not in setup.events and 'reset_return' not in setup.events
    assert review.read_json(setup.out / 'reset_execution.json')['commands'] == []


@pytest.mark.parametrize('change', ['file', 'pose'])
def test_changed_start_at_reset_prompt_blocks_control(setup, monkeypatch, change):
    def confirm(_):
        if change == 'file': setup.args.start_snapshot.write_text('{}')
        else: setup.state['tcp'][0] += .001
        return True
    monkeypatch.setattr(reset.real, 'confirm', confirm)
    rc, _ = setup.run()
    assert rc == 1 and 'control' not in setup.events and 'open' not in setup.events


def test_small_prompt_drift_rebuilds_reset_before_preflight(setup,monkeypatch):
    old=deepcopy(setup.state)
    def confirm(_):
        setup.state['tcp'][0]+=.000130422
        setup.state['q'][0]+=.000180
        return True
    monkeypatch.setattr(reset.real,'confirm',confirm)
    rc,_=setup.run()
    assert rc==0
    report=review.read_json(setup.out/'summary.json')
    fresh=report['preflight_snapshot']
    assert fresh['tcp'][0]==pytest.approx(old['tcp'][0]+.000130422)
    assert report['goals'][0]['goal_tcp'][0]==fresh['tcp'][0]
    assert report['preflight_origin_refresh']['previous']==old
    assert setup.events.index('preflight')<setup.events.index('open')


def test_same_drift_after_preflight_still_blocks_motion(setup,monkeypatch):
    original=reset.motion.preflight_goals
    def preflight(*args):
        result=original(*args)
        setup.state['tcp'][0]+=.000130422
        return result
    monkeypatch.setattr(reset.motion,'preflight_goals',preflight)
    rc,_=setup.run()
    assert rc==1 and 'open' not in setup.events
    assert 'Preflight start drifted' in review.read_json(setup.out/'summary.json')['error']


def test_nominal_scan_preset_arrival_uses_checked_controller_endpoint(setup,monkeypatch):
    setup.start['pose_source']='nominal_wrist_flip_preset'
    setup.args.start_snapshot.write_text(json.dumps(setup.start))
    original=reset.motion.preflight_goals
    def preflight(*args):
        commands=original(*args)
        last=commands[-1]
        last['q_nominal']=list(last['q_nominal']);last['q_nominal'][0]+=.004
        last['q']=list(last['q_nominal']);last['tcp']=list(last['goal_tcp'])
        return commands
    monkeypatch.setattr(reset.motion,'preflight_goals',preflight)
    rc,_=setup.run()
    assert rc==0
    report=review.read_json(setup.out/'summary.json')
    assert report['controller_resolved_start_target']['q'][0]==pytest.approx(.004)
    assert report['final_snapshot']['q'][0]==pytest.approx(.004)


def test_reset_requires_explicit_live_flag_before_backend(setup):
    setup.args.reset_first = False
    with pytest.raises(ValueError, match='explicit'):
        setup.run()
    assert not setup.events


def test_retreat_does_not_silently_clip_or_turn_into_descent(setup):
    # Remain 50 mm below the configured ceiling; the fixed 80 mm retreat must
    # fail, whether the config uses a cube box or the URDF-derived envelope.
    setup.state['tcp'][2] = setup.config['tcp_bounds_mm'][1][2]/1000 - .05
    with pytest.raises(RuntimeError, match='bounds'):
        reset.reset_goals(setup.state, setup.start, setup.config)


def test_unconfirmed_desk_or_wrong_robot_blocks_before_connection(setup):
    setup.start['ur_ip'] = 'wrong'
    setup.args.start_snapshot.write_text(json.dumps(setup.start))
    rc, _ = setup.run()
    assert rc == 1 and not setup.events


def test_desk_model_height_encloses_tilted_plane_over_fixed_envelope(setup):
    normal = np.array([.005, .007, 1.])
    from pathlib import Path
    Path(setup.perception['desk_plane']).write_text(yaml.safe_dump(dict(plane=dict(normal_base=normal.tolist()))))
    source = reset.desk_source(setup.config, setup.perception)
    center, radius = reset.support.desk_envelope_geometry(setup.config['desk_envelope'])
    angles = np.linspace(0, 2*np.pi, 1000)
    xy = center + radius*np.c_[np.cos(angles), np.sin(angles)]
    heights = .034 - xy @ normal[:2] / normal[2]
    assert source['preflight']['table_height_m'] >= max(heights)
    assert source['preflight']['table_height_m'] - max(heights) < 1e-7


def test_explicit_calibrated_plane_uses_existing_anchor_not_arbitrary_lowering(setup):
    from pathlib import Path
    normal=np.array([.005,.007,1.])
    Path(setup.perception['desk_plane']).write_text(yaml.safe_dump(dict(plane=dict(normal_base=normal.tolist()))))
    old=reset.desk_source(setup.config,setup.perception)
    config=dict(setup.config,reflex_desk_surface='calibrated-plane')
    new=reset.desk_source(config,setup.perception)
    center,radius=reset.support.desk_envelope_geometry(config['desk_envelope'])
    assert new['desk_envelope']==old['desk_envelope']
    assert new['desk_surface_plane']['point_base_m']==[0.,0.,.034]
    assert np.allclose(new['desk_surface_plane']['normal_base'],normal/np.linalg.norm(normal))
    assert new['preflight']['table_height_m']==pytest.approx(.034-center@normal[:2]/normal[2])
    assert old['preflight']['table_height_m']-new['preflight']['table_height_m']==pytest.approx(
        radius*np.linalg.norm(normal[:2])/normal[2])
    assert 'reflex_desk_surface' not in setup.config


def test_table_only_model_has_no_old_object_proxy():
    config = reset.real.load_config(reset.real.ROOT / 'configs/ggx_real_lab.yaml')
    source = dict(preflight=dict(table_height_m=.04), desk_envelope=config['desk_envelope'])
    model = reset.support.build_model(source, include_object=False)
    assert model.geom('floor').id >= 0
    assert reset.real.sim.mujoco.mj_name2id(model, reset.real.sim.mujoco.mjtObj.mjOBJ_BODY,
                                          'preflight_object') == -1


@pytest.mark.parametrize('answer', ['', 'q'])
def test_reset_disconnects_then_waits_for_enter_before_camera(setup, tmp_path, monkeypatch, answer):
    # Keep the real generate_candidates placement gate. Stop the fake camera
    # stage immediately; neither camera, inference nor hardware is constructed.
    events = setup.events
    def fake_reset(args, out):
        out.mkdir()
        events.extend(['open', 'return', 'disconnect'])
        (out / 'summary.json').write_text(json.dumps(dict(status='reset_completed',
            motion_commands_attempted=True)))
        return 0, setup.args.start_snapshot
    monkeypatch.setattr(reset, 'run_reset', fake_reset)
    def confirm(prompt):
        assert events == ['open', 'return', 'disconnect']
        events.append('placement_enter' if not answer else 'placement_cancel')
        return answer
    monkeypatch.setattr('builtins.input', confirm)
    def camera(*args, **kwargs):
        assert events[-1] == 'placement_enter'
        events.append('fresh_capture')
        raise RuntimeError('mock camera stopped here')
    monkeypatch.setattr(review, 'run_stage', camera)
    out = tmp_path / 'review'
    rc = review.main(['--capture-live', '--reset-first', '--dimensions-mm', '40', '90', '176',
        '--config', str(setup.args.config), '--start-snapshot', str(setup.args.start_snapshot),
        '--real-config', str(setup.args.real_config),
        '--out', str(out), '--no-open'])
    assert rc == (1 if not answer else 2)
    assert ('fresh_capture' in events) == (not answer)
    report = review.read_json(out / 'summary.json')
    assert report['robot_commands_sent'] and report['selected_rank'] is None


def test_reset_failure_blocks_generator_and_real_handoff(setup, tmp_path, monkeypatch):
    def failed(args, out):
        out.mkdir()
        (out / 'summary.json').write_text(json.dumps(dict(status='failed')))
        return 1, None
    monkeypatch.setattr(reset, 'run_reset', failed)
    monkeypatch.setattr(review, 'generate_candidates', lambda *a: pytest.fail('No camera/generation'))
    monkeypatch.setattr(reset.real, 'main', lambda *a: pytest.fail('No grasp handoff'))
    out = tmp_path / 'review'
    assert review.main(['--capture-live', '--reset-first', '--execute', '--dimensions-mm', '40', '90', '176',
                        '--real-config', str(setup.args.real_config),
                        '--out', str(out), '--no-open']) == 1
    assert review.read_json(out / 'summary.json')['status'] == 'reset_failed'


def test_reset_flag_cannot_mix_with_saved_scene():
    for mode in ('--capture', '--review-run'):
        with pytest.raises(SystemExit):
            review.parse_args([mode, 'old', '--reset-first', '--dimensions-mm', '40', '90', '176'])
    assert not review.parse_args(['--review-run', 'old']).reset_first
