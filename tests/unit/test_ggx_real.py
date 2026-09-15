"""CB3 preparation and authorization regression tests: no hardware access."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import ggx_real_support as support
from tools import run_ggx_real as real


@pytest.fixture
def config():
    return real.load_config(real.ROOT / 'configs/ggx_real_lab.yaml')


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),True,'1.5'])
def test_invalid_reflex_desk_margin_rejected(config,tmp_path,value):
    import yaml
    config['reflex_desk_clearance_mm']=value
    path=tmp_path/'invalid_margin.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='reflex_desk_clearance_mm'):
        real.load_config(path)


def test_raspberry_profile_changes_only_explicit_margin():
    from tools import ggx_motion
    old=real.load_config(real.ROOT/'configs/reflex_real_lab.yaml')
    new=real.load_config(real.ROOT/'configs/reflex_raspberry_lab.yaml')
    assert new.pop('reflex_desk_clearance_mm')==1.5
    assert old==new
    assert ggx_motion.reflex_desk_clearance_m(old)==.002


def test_calibrated_plane_profile_is_separate_and_zero_margin_is_not_disabled():
    from tools import ggx_motion
    old=real.load_config(real.ROOT/'configs/reflex_raspberry_lab.yaml')
    new=real.load_config(real.ROOT/'configs/reflex_raspberry_calibrated_plane_lab.yaml')
    assert new.pop('reflex_desk_surface')=='calibrated-plane'
    assert new.pop('reflex_desk_clearance_mm')==0
    old.pop('reflex_desk_clearance_mm')
    assert new==old
    for gap in (-.001,0.,float('nan'),float('inf')):
        assert ggx_motion.desk_clearance_blocked(gap,0.)
    assert not ggx_motion.desk_clearance_blocked(.0005,0.)


def test_calibrated_finite_desk_preserves_xy_footprint_and_plane(monkeypatch):
    from tools import preflight_ggx_ur5_table as sim
    monkeypatch.setattr(support,'cb3_spec',lambda:sim.mujoco.MjSpec())
    envelope=dict(center_base_xy_m=[-.2,-.6],sides_upper_mm=[550,500],
                  center_offset_upper_mm=100,reference_margin_mm=10,confirmed=True)
    center,radius=support.desk_envelope_geometry(envelope)
    normal=np.array([.005,.007,1.]);normal/=np.linalg.norm(normal)
    point=np.array([0.,0.,.034])
    center_z=point[2]-normal[:2]@center/normal[2]
    source=dict(preflight=dict(table_height_m=center_z),desk_envelope=envelope,
                desk_surface_plane=dict(normal_base=normal.tolist(),point_base_m=point.tolist()))
    model=support.build_model(source,include_object=False)
    data=sim.mujoco.MjData(model);sim.mujoco.mj_forward(model,data)
    vertices=sim._world_convex_vertices(model,data,model.geom('floor').id)
    signed=(vertices-point)@normal
    top=vertices[signed>-1.]
    assert len(top)==64
    assert np.max(np.abs((top-point)@normal))<1e-6
    assert np.allclose(signed[signed<-1.],-2*normal[2],atol=1e-6,rtol=0)
    assert np.allclose(np.linalg.norm(top[:,:2]-center,axis=1),
                       (radius+1e-6)/np.cos(np.pi/64),atol=1e-6,rtol=0)
    bad=dict(source,preflight=dict(table_height_m=center_z-.008))
    with pytest.raises(ValueError,match='reference height disagrees'):
        support.build_model(bad,include_object=False)


@pytest.mark.parametrize('field,value', [
    ('scene_check_mode', 'skip'), ('reset_motion', {}), ('reset_motion', 'fast'),
    ('preview_videos', 'false'), ('preview_videos', 0),
    ('controller_preflight_mode', 'skip'), ('controller_preflight_mode', []),
    ('joint_speed_rad_s', 2.), ('cartesian_speed_m_s', float('nan')),
])
def test_invalid_reset_or_scene_check_config_rejected(config, tmp_path, field, value):
    import yaml
    if field in ('scene_check_mode', 'reset_motion', 'preview_videos', 'controller_preflight_mode'):
        config[field] = value
    else:
        config['reset_motion'] = {key: config[key] for key in (
            'joint_speed_rad_s', 'joint_accel_rad_s2', 'cartesian_speed_m_s', 'cartesian_accel_m_s2')}
        config['reset_motion'][field] = value
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        real.load_config(path)


@pytest.fixture
def orchestration(tmp_path, monkeypatch, config):
    snapshot = dict(q=[0.] * 6, tcp=[-.2, -.4, .35, 0., 0., 0.])
    start = tmp_path / 'start.json'
    start.write_text(json.dumps(snapshot))
    args = SimpleNamespace(execute=False, preflight_only=False, video=True, open=False,
                           start_snapshot=start, phase='grasp-lift')
    source = dict(pins={}, rank=49)
    report = dict(code_config_sha256={})
    events = []
    rows = [dict(index=i, phase=phase, hand_raw=[1000-i]*6, tcp=snapshot['tcp'])
            for i, phase in enumerate(['start', 'start_to_pregrasp', 'approach', 'close', 'close', 'lift'])]
    prepared = dict(rows=rows, snapshot=snapshot, hand_open_raw=[1000]*6)

    class Backend:
        control = SimpleNamespace(getTCPOffset=lambda: [0.]*6)
        def __init__(self, *a): events.append('construct')
        def open_receiver(self): events.append('receive')
        def snapshot(self): return snapshot
        def open_control(self): events.append('control')
        def open_hand(self, command): events.append('open')
        def hand_angles(self, command): events.append(('hand', command))
        def move_phase(self, command): events.append(command['phase'])
        def dwell(self, seconds): events.append(('dwell', seconds))
        def set_stage(self, stage): report['stage'] = stage
        def close(self): events.append('disconnect')

    def prepare(source, snapshot, out, video):
        events.append('prepare')
        (out/'prepared.json').write_text(json.dumps(prepared))
        if video:
            (out/'cb3_review.mp4').write_bytes(b'fake')
        return prepared, None

    def scene(source, config, out, name):
        events.append(name + '_scene')
        return dict(acquired_monotonic=time.monotonic())

    def preflight(*a):
        events.append('preflight')
        return [dict(revision=real.motion.REVISION, kind=real.motion.KINDS[r['phase']],
                     phase=r['phase'], index=r['index'], q=[0.]*6, tcp=r['tcp']) for r in rows
                if r['phase'] not in ('start', 'close')
                and not (args.phase == 'approach-only' and r['phase'] == 'lift')]

    def preview(out, *a, video=True):
        (out/'execution_precheck.json').write_text(json.dumps(dict(
            schema='rh56_ggx_execution_precheck/v1', passed=True)))
        if not video:
            return None
        path = out / 'execution_review.mp4'
        path.write_bytes(b'checked path preview')
        return path

    monkeypatch.setattr(real, 'preflight_controller', preflight)
    monkeypatch.setattr(real, 'show', lambda *a: None)
    def run(**overrides):
        options = dict(backend_factory=Backend, prepare_fn=prepare, scene_check=scene, review_fn=preview)
        options.update(overrides)
        return real.run(args, config, source, tmp_path, report, lambda: None, **options)
    return SimpleNamespace(run=run, args=args, events=events, report=report, scene=scene,
                           tmp=tmp_path, snapshot=snapshot)


def answers(monkeypatch, values):
    values = iter(values)
    monkeypatch.setattr(real, 'confirm', lambda _: next(values))


def test_offline_never_constructs_hardware(orchestration):
    t = orchestration
    assert t.run() == 0
    assert t.events == ['prepare']
    assert t.report['status'] == 'offline_prepared'


def test_unconfirmed_desk_bounds_block_before_devices(orchestration, config):
    t = orchestration
    t.args.execute = True
    config['desk_envelope'] = dict(center_base_xy_m=[-.2,-.6], sides_upper_mm=[550,500],
                                   center_offset_upper_mm=100, reference_margin_mm=10, confirmed=False)
    with pytest.raises(ValueError, match='operator confirmation'):
        t.run()
    assert not t.events


@pytest.mark.parametrize('confirmations,status,control', [
    ([False], 'cancelled_before_controller', False),
    ([True, False], 'cancelled_before_motion', True),
])
def test_cancel_never_moves(orchestration, monkeypatch, confirmations, status, control):
    t = orchestration
    t.args.execute = True
    answers(monkeypatch, confirmations)
    assert t.run() == 2
    assert t.report['status'] == status
    assert ('control' in t.events) == control
    assert 'open' not in t.events and 'lift' not in t.events
    assert t.events[-1] == 'disconnect'


def test_preflight_only_never_writes_hand_or_moves(orchestration, monkeypatch):
    t = orchestration
    t.args.preflight_only = True
    answers(monkeypatch, [True])
    assert t.run() == 0
    assert 'preflight' in t.events
    assert 'open' not in t.events


def test_confirmed_sequence_holds_closure_on_lift(orchestration, monkeypatch):
    t = orchestration
    t.args.execute = True
    answers(monkeypatch, [True, True])
    assert t.run() == 0
    assert t.events.index('final_scene') < t.events.index('open')
    assert t.events.index('open') < t.events.index('start_to_pregrasp') < t.events.index('approach')
    assert [x for x in t.events if isinstance(x, tuple) and x[0] == 'hand'] == [
        ('hand', [997]*6), ('hand', [996]*6)]
    assert t.events[-2:] == ['lift', 'disconnect']
    assert t.report['status'] == 'commands_completed_grasp_unverified'


def test_approach_only_skips_close_and_lift(orchestration, monkeypatch):
    t = orchestration
    t.args.execute, t.args.phase = True, 'approach-only'
    answers(monkeypatch, [True, True])
    assert t.run() == 0
    assert 'approach' in t.events and 'lift' not in t.events
    assert not any(isinstance(x, tuple) and x[0] == 'hand' for x in t.events)
    assert t.report['status'] == 'approach_completed'


def test_changed_video_cancels_before_control(orchestration, monkeypatch):
    t = orchestration
    t.args.execute = True
    def change(_):
        (t.tmp/'cb3_review.mp4').write_bytes(b'changed')
        return True
    monkeypatch.setattr(real, 'confirm', change)
    with pytest.raises(ValueError, match='changed'):
        t.run()
    assert 'control' not in t.events and 'open' not in t.events


def test_changed_execution_preview_cancels_before_hand_write(orchestration, monkeypatch):
    t = orchestration
    t.args.execute = True
    count = [0]
    def approve(_):
        count[0] += 1
        if count[0] == 2:
            (t.tmp/'execution_review.mp4').write_bytes(b'changed after review')
        return True
    monkeypatch.setattr(real,'confirm',approve)
    with pytest.raises(ValueError,match='changed'):
        t.run()
    assert 'open' not in t.events


def test_final_scene_failure_never_opens_hand(orchestration, monkeypatch):
    t = orchestration
    t.args.execute = True
    answers(monkeypatch, [True, True])
    # The fixture passes the same function object, so fail from its timestamp.
    monkeypatch.setattr(real.time, 'monotonic', lambda: float('nan'))
    with pytest.raises(RuntimeError, match='expired'):
        t.run()
    assert 'open' not in t.events


@pytest.mark.parametrize('stage',['initial','final'])
def test_cancelled_depth_warning_review_stops_without_hand_or_motion(orchestration,monkeypatch,stage):
    t = orchestration
    t.args.execute = True
    answers(monkeypatch,[True,True])
    def scene(source,config,out,name):
        value = t.scene(source,config,out,name)
        if name == stage:
            raise RuntimeError('Reference depth warning photo review cancelled')
        return value
    with pytest.raises(RuntimeError,match='photo review cancelled'):
        t.run(scene_check=scene)
    assert 'open' not in t.events and 'approach' not in t.events
    assert t.events[-1] == 'disconnect'
    if stage == 'initial': assert 'control' not in t.events


def test_warning_review_wait_does_not_refresh_final_capture_age(orchestration,monkeypatch,config):
    t = orchestration
    t.args.execute = True
    answers(monkeypatch,[True,True])
    def scene(source,c,out,name):
        value = t.scene(source,c,out,name)
        if name == 'final':
            value.update(visual_confirmation_received=True,
                         acquired_monotonic=time.monotonic()-config['max_scene_age_s']-1)
        return value
    with pytest.raises(RuntimeError,match='expired'):
        t.run(scene_check=scene)
    assert 'open' not in t.events and 'approach' not in t.events


@pytest.mark.parametrize('answer',['','q','eof','interrupt'])
def test_capture_warning_shows_photo_and_requires_enter_without_refreshing_time(monkeypatch,tmp_path,answer):
    from tools import ggx_scene_check as check
    from tools import preview_tabletop_target as preview
    clock = [100.]
    shown = []
    monkeypatch.setattr(real.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(preview,'load_perception_config',lambda *a:(dict(camera_serial='fake',roi=[0,0,1280,720]),None))
    monkeypatch.setattr(real,'run_stage',lambda *a:None)
    monkeypatch.setattr(real,'show',lambda path,auto_open:shown.append((path,auto_open)))
    def ask(prompt):
        assert shown and '不是独立运动授权' in prompt
        clock[0] += 30
        if answer == 'eof': raise EOFError
        if answer == 'interrupt': raise KeyboardInterrupt
        return answer
    monkeypatch.setattr('builtins.input',ask)
    def verify_saved(source,capture,c,prefix,*,review_warning):
        accepted = review_warning(tmp_path/'photo.svg',[dict(row=0,column=0)])
        if not accepted:
            raise RuntimeError('photo review cancelled')
        return dict(passed=True,visual_confirmation_received=True)
    monkeypatch.setattr(check,'verify_saved_target',verify_saved)
    options = dict(scene_check_mode='reference-rgbd',max_scene_age_s=20.)
    if answer == '':
        result = real.capture_and_check(dict(perception=dict(config='fake')),options,tmp_path,'final',auto_open=False)
        assert result['acquired_monotonic'] == 100. and clock[0] == 130.
    else:
        with pytest.raises(RuntimeError,match='cancelled'):
            real.capture_and_check(dict(perception=dict(config='fake')),options,tmp_path,'final',auto_open=False)
    assert shown == [(tmp_path/'photo.svg',False)]


def test_controller_failure_never_reaches_motion_confirmation(orchestration, monkeypatch):
    t = orchestration
    t.args.execute = True
    answers(monkeypatch, [True])
    def fail(*a):
        raise RuntimeError('IK missing')
    monkeypatch.setattr(real, 'preflight_controller', fail)
    with pytest.raises(RuntimeError, match='IK missing'):
        t.run()
    assert 'open' not in t.events and t.events[-1] == 'disconnect'


@pytest.mark.parametrize('choice,status', [('', 'approach_completed'), ('n', 'cb3_review_vetoed'),
                                         ('q', 'cancelled_before_motion'), ('r', 'scene_invalidated')])
def test_checked_video_is_the_only_motion_approval_after_all_preflight(orchestration,monkeypatch,choice,status):
    t = orchestration
    t.args.execute, t.args.checked_review, t.args.phase = True, True, 'approach-only'
    confirmations = []
    def controller(prompt):
        confirmations.append(prompt)
        assert 'control' not in t.events and 'preflight' not in t.events
        assert not (t.tmp/'cb3_review.mp4').exists()  # No redundant nominal video.
        return True
    monkeypatch.setattr(real,'confirm',controller)
    def choose(prompt):
        assert '批准并执行此 CB3' in prompt
        assert 'preflight' in t.events and (t.tmp/'execution_precheck.json').is_file()
        assert (t.tmp/'execution_review.mp4').is_file()
        assert 'open' not in t.events
        t.events.append('checked_video_decision')
        return choice
    monkeypatch.setattr('builtins.input',choose)
    assert t.run() == (0 if choice == '' else 2)
    assert len(confirmations) == 1  # RTDE-only; no earlier visual or motion approval.
    assert t.report['status'] == status
    assert ('open' in t.events) == (choice == '')
    if choice == '':
        assert t.events.index('preflight') < t.events.index('checked_video_decision') < t.events.index('final_scene')
        assert t.events.count('preflight') == t.events.count('prepare') == 1
        assert t.events.index('final_scene') < t.events.index('open')
        assert str(t.tmp/'execution_review.mp4') in t.report['checked_review']['reviewed_sha256']


def test_checked_video_replay_does_not_repeat_preflight(orchestration,monkeypatch):
    t = orchestration
    t.args.execute = t.args.checked_review = True
    answers(monkeypatch,[True])
    choices = iter(['p','q'])
    monkeypatch.setattr('builtins.input',lambda _:next(choices))
    assert t.run() == 2
    assert t.events.count('prepare') == t.events.count('preflight') == 1
    assert 'open' not in t.events


def test_rtde_permission_is_once_per_invocation_not_motion_permission(orchestration,monkeypatch):
    t = orchestration
    t.args.execute = t.args.checked_review = True
    grants = []; authorization = {}
    monkeypatch.setattr(real,'confirm',lambda p:grants.append(p) or True)
    choices = iter(['n','q'])
    monkeypatch.setattr('builtins.input',lambda _:next(choices))
    assert t.run(controller_authorization=authorization) == 2
    assert t.run(controller_authorization=authorization) == 2
    assert len(grants) == 1 and 'open' not in t.events
    # A new invocation must obtain permission again.
    monkeypatch.setattr('builtins.input',lambda _:'q')
    assert t.run(controller_authorization={}) == 2
    assert len(grants) == 2


@pytest.mark.parametrize('reason',['ik_failed_pregrasp','desk_clearance','object_collision_before_closure','unknown_error'])
def test_only_explicit_nominal_rejections_can_advance_to_next_candidate(orchestration,monkeypatch,reason):
    t = orchestration
    t.args.execute = t.args.checked_review = True
    monkeypatch.setattr(real,'confirm',lambda _:pytest.fail('No controller grant for failed nominal path'))
    def fail(source,snapshot,out,video):
        (out/'cb3_rejection.json').write_text(json.dumps(dict(rank=source['rank'],passed=False,reason=reason)))
        raise ValueError('nominal rejection')
    if reason == 'unknown_error':
        with pytest.raises(ValueError,match='nominal rejection'):
            t.run(prepare_fn=fail)
    else:
        assert t.run(prepare_fn=fail) == 2
        assert t.report['status'] == 'cb3_candidate_rejected'
    assert 'control' not in t.events and 'open' not in t.events


@pytest.mark.parametrize('fault',['video','commands','start','scene'])
def test_checked_approval_still_blocks_post_review_changes(orchestration,monkeypatch,fault):
    t = orchestration
    t.args.execute = t.args.checked_review = True
    answers(monkeypatch,[True])
    def choose(_):
        if fault == 'video': (t.tmp/'execution_review.mp4').write_bytes(b'changed')
        if fault == 'commands': (t.tmp/'execution.json').write_text('{}')
        if fault == 'start':
            # The origin snapshot object is shared by this small fixture;
            # exercise the real guard hook without pretending that alias is a live sensor.
            monkeypatch.setattr(real,'require_same_start',lambda *a: (_ for _ in ()).throw(RuntimeError('start changed')))
        if fault == 'scene': monkeypatch.setattr(real.time,'monotonic',lambda:float('nan'))
        return ''
    monkeypatch.setattr('builtins.input',choose)
    with pytest.raises((RuntimeError,ValueError),match='changed|expired'):
        t.run()
    assert 'open' not in t.events


def test_candidate_proposal_is_not_a_saved_visual_approval(tmp_path,monkeypatch):
    path = tmp_path/'candidate.json'
    path.write_text(json.dumps(dict(schema='rh56_ggx_execution_candidate/v1',human_approved=False,
        executable=False,rank=2,preflight=str(tmp_path/'sim/summary.json'),source_sha256={})))
    with pytest.raises(ValueError,match='visual-only'):
        support.read_approval(path)
    calls = []
    monkeypatch.setattr(support,'read_candidate_source',lambda *a: calls.append(a) or {'rank':2})
    assert support.read_candidate(path)['rank'] == 2
    assert calls[0][-1] == 'unreviewed_candidate'
    proposal = json.loads(path.read_text()); proposal['human_approved'] = True
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError,match='unapproved'):
        support.read_candidate(path)


def test_unreviewed_candidate_cannot_use_legacy_no_video_motion_gate(orchestration,config):
    t = orchestration
    t.args.execute = True
    with pytest.raises(ValueError,match='Unreviewed candidate requires'):
        real.run(t.args,config,dict(source_kind='unreviewed_candidate'),t.tmp,{},lambda:None,
                 backend_factory=lambda *a:pytest.fail('No device construction'))


def test_tcp_offset_fails_closed():
    for offset in ([0.]*5, [0., 0., .001, 0., 0., 0.], [float('nan')]*6):
        with pytest.raises(RuntimeError, match='zero active TCP'):
            real.check_tcp_offset(SimpleNamespace(getTCPOffset=lambda: offset))


def test_scene_requires_unique_unchanged_target(config):
    patch = dict(center_base_mm=[-200., -600., 210.], short_mm=40., long_mm=90.,
                 height_median_mm=176., short_axis_yaw_deg=89., yaw_ambiguous=False)
    ref = dict(crop=dict(top_patch=patch, dimensions_prior_mm=[40.,90.,176.]))
    row = dict(top_patch=dict(patch), geometry_candidate=True, rejection_reasons=[], component_id=1)
    assert real.scene_match(ref, dict(candidates=[row]), config)['component_id'] == 1
    with pytest.raises(RuntimeError, match='ambiguous'):
        real.scene_match(ref, dict(candidates=[row, row]), config)
    row['top_patch']['center_base_mm'] = [-190., -600., 210.]
    with pytest.raises(RuntimeError, match='Scene changed'):
        real.scene_match(ref, dict(candidates=[row]), config)


def test_foreground_incompatible_scene_mode_blocks_before_devices(orchestration,config):
    t = orchestration
    t.args.execute = True
    source = dict(crop=dict(crop_mode='foreground'))
    with pytest.raises(ValueError,match='reference-rgbd'):
        real.run(t.args,dict(config,scene_check_mode='components'),source,t.tmp,{},lambda:None,
                 backend_factory=lambda *a:pytest.fail('Must reject before connecting'))
    with pytest.raises(ValueError,match='reference-rgbd'):
        real.scene_match(source,{},config)


@pytest.mark.parametrize('status',['manual_review_accepted_not_executable','review_approved'])
def test_manual_collision_approval_cannot_be_relabelled_as_real(tmp_path,status):
    approval=tmp_path/'summary.json'
    approval.write_text(json.dumps(dict(schema='rh56_ggx_interactive_review/v1',status=status,
        executable=False,selected_rank=6,reviews=[dict(rank=6,decision='approve',scope='manual_object_collision')])))
    with pytest.raises(ValueError,match='not a real-motion approval'):
        support.read_approval(approval)


@pytest.mark.parametrize('fault',[None,'mask','size','yaw','center','height'])
def test_foreground_approval_binds_segmentation_and_same_simulation_proxy(tmp_path,monkeypatch,fault):
    from pathlib import Path
    from tools import preview_tabletop_target as preview
    from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA
    plane = tmp_path/'plane.yaml'; plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    video = tmp_path/'sim'/'video.mp4'
    preflight_path = video.parent/'summary.json'
    cloud_path = tmp_path/'crop'/'points.npy'
    perception_path = tmp_path/'perception'/'summary.json'
    candidate_path = tmp_path/'inference'/'candidates.yaml'
    crop = dict(schema=FOREGROUND_SCHEMA,crop_mode='foreground',
        status='measured_crop_ready_for_offline_inference',hardware_contacted=False,executable_target=False,
        source_preview=str(perception_path),point_cloud=str(cloud_path),source_preview_sha256='hash',
        calibration_sha256='hash',source_depth_sha256='hash',source_foreground_sha256='hash',
        dimensions_prior_mm=[40,90,180],
        object_frame=dict(center_base_mm=[-200,-600,34],origin='desk_footprint_center',
                          yaw_ambiguous=False,short_axis_yaw_deg=85),
        collision_proxy=dict(shape='box',dimensions_mm=[54,95,196],observed_bounds_mm=[53.9,94.7,195.1]))
    simulation = dict(object_collision_gate=dict(revision=real.sim.OBJECT_GATE_REVISION),
        automatic_pass_ranks=[0],candidate_frame='ur-base',candidate_yaml=str(candidate_path),
        candidate_yaml_sha256='hash',object_size_mm=[95,54,196],object_center_m=[-.2,-.6,.132],
        object_yaw_deg=-5,table_height_m=.034)
    if fault=='size': simulation['object_size_mm']=[90,40,180]
    if fault=='yaw': simulation['object_yaw_deg']=0
    if fault=='center': simulation['object_center_m'][0]=-.18
    if fault=='height': simulation['table_height_m']=.05
    approval_path = tmp_path/'approval.json'
    records = {approval_path:dict(schema='rh56_ggx_interactive_review/v1',status='review_approved',
        executable=False,selected_rank=0,reviews=[dict(decision='approve',rank=0,video=str(video),
        reviewed_sha256={str(preflight_path):'hash'})]),preflight_path:simulation,
        candidate_path.parent/'summary.json':dict(schema='rh56_ggx_point_cloud_inference/v1',
        input_frame='ur_base',candidate_sha256='hash',source_point_cloud=str(cloud_path),source_point_cloud_sha256='hash'),
        cloud_path.parent/'summary.json':crop,
        perception_path:dict(calibration='calibration',capture=str(tmp_path/'capture'),
                             config='config',desk_plane=str(plane))}
    monkeypatch.setattr(support,'read_json',lambda path:records[Path(path)])
    monkeypatch.setattr(support,'verify',lambda *a:None)
    monkeypatch.setattr(support,'digest',lambda path:'changed' if fault=='mask' and Path(path).name=='foreground.png' else 'hash')
    monkeypatch.setattr(support,'pin',lambda paths:{str(path):'hash' for path in paths})
    monkeypatch.setattr(preview,'load_perception_config',lambda *a:(dict(desk_plane=str(plane),desk_z_mm=34),None))
    if fault is not None:
        with pytest.raises(ValueError,match='mask changed|collision proxy'):
            support.read_approval(approval_path)
    else:
        source = support.read_approval(approval_path)
        assert str(perception_path.parent/'foreground.png') in source['pins']
        assert source['preflight']['object_size_mm'] == [95,54,196]


def test_cb3_fk_and_unchanged_hand_mapping():
    model = support.cb3_spec().compile()
    q = [.2840830684,-1.8313415686,2.4213135242,3.1709938049,-.3150118033,-.5482876937]
    actual = [-.2266686637,-.2630852635,.3433573266,.0203508655,2.3806605480,-1.9752828313]
    pe, re = support.pose_error(support.fk_tcp(model, q), actual)
    assert pe < 2 and re < 1  # Recorded zero-TCP CB3 snapshot; no live access.
    assert support.raw_hand_command(model, 0) == [1000]*5 + [0]
    assert all(0 <= x <= 1000 for x in support.raw_hand_command(model, 1))
    for alpha in np.linspace(0, 1, 31):
        assert all(0 <= x <= 1000 for x in support.raw_hand_command(model, alpha))
    for alpha in [-.1, 1.1, float('nan')]:
        with pytest.raises(ValueError):
            support.raw_hand_command(model, alpha)


def test_finite_desk_requires_explicit_edge_and_supports_object():
    source = dict(preflight=dict(table_height_m=.034, object_center_m=[-.2, -.6, .12],
                                 object_yaw_deg=0, object_size_mm=[90,40,176]))
    infinite = support.build_model(source)
    assert infinite.geom('floor').type == support.mujoco.mjtGeom.mjGEOM_PLANE
    finite = support.build_model(dict(source, desk_edge_y_m=-.25))
    assert finite.geom('floor').type == support.mujoco.mjtGeom.mjGEOM_BOX
    with pytest.raises(ValueError, match='support object'):
        support.build_model(dict(source, desk_edge_y_m=-.59))


def test_desk_envelope_contains_all_disk_directions_after_compilation():
    from scipy.spatial import ConvexHull
    envelope = dict(center_base_xy_m=[-.23,-.65], sides_upper_mm=[550,500],
                    center_offset_upper_mm=100, reference_margin_mm=10, confirmed=True)
    source = dict(preflight=dict(table_height_m=.034, object_center_m=[-.23,-.65,.12],
                                 object_yaw_deg=0, object_size_mm=[90,40,176]),
                  desk_envelope=envelope)
    center, radius = support.desk_envelope_geometry(envelope)
    assert radius == pytest.approx(np.hypot(.550,.500)/2 + .110)
    m = support.build_model(source)
    d = support.mujoco.MjData(m)
    support.mujoco.mj_forward(m, d)
    vertices = real.sim._world_convex_vertices(m, d, m.geom('floor').id)
    hull = ConvexHull(vertices)
    angles = np.linspace(0,2*np.pi,1440)
    points = np.c_[center[0]+radius*np.cos(angles), center[1]+radius*np.sin(angles),
                   np.full_like(angles, -.5)]
    assert np.max(points @ hull.equations[:,:3].T + hull.equations[:,3]) <= 1e-7
    # Bounds remain fixed even when a later object moves within the table.
    shifted = dict(source, preflight=dict(source['preflight'], object_center_m=[-.28,-.65,.12]))
    assert np.allclose(m.geom('floor').pos, support.build_model(shifted).geom('floor').pos)
    with pytest.raises(ValueError, match='OR envelope'):
        support.build_model(dict(source, desk_edge_y_m=-.25))


@pytest.mark.parametrize('field,value', [
    ('sides_upper_mm', [550,float('nan')]), ('center_offset_upper_mm', -1),
    ('reference_margin_mm', 0), ('confirmed', 'true'), ('center_base_xy_m',[0]),
])
def test_desk_envelope_rejects_invalid_bounds(field, value):
    envelope = dict(center_base_xy_m=[-.23,-.65], sides_upper_mm=[550,500],
                    center_offset_upper_mm=100, reference_margin_mm=10, confirmed=True)
    envelope[field] = value
    with pytest.raises(ValueError):
        support.desk_envelope_geometry(envelope)


def test_recorded_cb3_desk_zero_has_independent_face_separation():
    # Offline regression from 2026-09-10 start-to-pregrasp sample 22. Three
    # geom axes fail to prove separation; the desk hull face normals do prove it.
    source = dict(preflight=dict(table_height_m=.03951569483323472,
        object_center_m=[-.22982383728027345,-.6525315551757812,.12751569483323472],
        object_size_mm=[90,40,176], object_yaw_deg=-1.1608084440231323),
        desk_envelope=dict(center_base_xy_m=[-.22982383728027345,-.6525315551757812],
            sides_upper_mm=[550,500],center_offset_upper_mm=100,reference_margin_mm=10,confirmed=True))
    m = support.build_model(source)
    d = support.mujoco.MjData(m)
    q = [.5668837540932601,-1.6376852746469615,2.0460736539448114,
         3.450939592148924,-.6344712282885235,-.5443292797292127]
    d.qpos[:] = real.sim.initial_qpos(m, np.array(q))
    support.mujoco.mj_forward(m, d)
    arm_geom = int(m.body('upper_arm_link').geomadr[0])
    assert real.sim.certified_separation(m, d, m.geom('floor').id, arm_geom, True) > .05


@pytest.mark.parametrize('distance,proof,contact,expected', [
    (0., .01, False, .01), (0., 0., False, 0.),
    (-.01, .1, False, -.01), (0., .1, True, 0.),
])
def test_finite_desk_zero_fallback_is_conservative(monkeypatch, distance, proof, contact, expected):
    import mujoco
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="box" size="1 1 .1"/>
      <body name="moving" pos="0 0 .19"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if not contact:
        d.ncon = 0
    else:
        assert d.ncon > 0
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *a: distance)
    monkeypatch.setattr(real.sim, 'certified_separation', lambda *a: proof)
    assert real.sim.desk_clearance(m, d)[0] == pytest.approx(expected)


@pytest.mark.parametrize('fault', [None, 'ik', 'model', 'offset', 'joints'])
def test_controller_preflight_uses_fk_and_fails_closed(config, fault):
    model = support.build_model(dict(preflight=dict(table_height_m=-2.,
        object_center_m=[1.,1.,1.], object_size_mm=[40,40,40], object_yaw_deg=0)))
    q = np.array([.2840830684,-1.8313415686,2.4213135242,3.1709938049,-.3150118033,-.5482876937])
    goal = q.copy()
    goal[0] += .005
    tcp = support.fk_tcp(model, q)
    snapshot = dict(q=q.tolist(), tcp=tcp)
    target = support.fk_tcp(model, goal)
    def fk(q, offset):
        p = support.fk_tcp(model, q)
        if fault == 'model':
            p[0] += .006
        return p
    control = SimpleNamespace(
        getTCPOffset=lambda: [0.,0.,.01 if fault == 'offset' else 0.,0.,0.,0.],
        getInverseKinematics=lambda *a: [] if fault == 'ik' else goal.tolist(),
        getForwardKinematics=fk, isPoseWithinSafetyLimits=lambda _: True,
        isJointsWithinSafetyLimits=lambda _: fault != 'joints')
    backend = SimpleNamespace(control=control, snapshot=lambda: snapshot)
    prepared = dict(snapshot=snapshot, rows=[
        dict(index=i, phase=phase, tcp=tcp if i==0 else target,
             q_nominal=q.tolist() if i==0 else goal.tolist(), closure_alpha=0.)
        for i, phase in enumerate(['start','start_to_pregrasp','approach','close','lift'])])
    if fault:
        with pytest.raises(RuntimeError):
            real.preflight_controller(backend, prepared, model, config, {})
    else:
        checked = real.preflight_controller(backend, prepared, model, config, {})
        assert np.allclose(checked[0]['q'], goal)


def test_offline_cli_defaults_and_no_video_never_implies_skip_checks():
    args = real.parse_args(['--approval', 'approval.json'])
    assert not args.execute and not args.preflight_only
    assert args.phase == 'approach-only'
    assert args.video is None
    live = real.parse_args(['--approval', 'approval.json', '--execute', '--no-video'])
    assert live.execute and live.video is False


def test_static_scene_requires_explicit_execute_and_checked_review():
    base = ['--approval','saved.json']
    assert not real.parse_args(base).static_scene
    for extra in [[],['--execute'],['--checked-review'],['--preflight-only','--checked-review']]:
        with pytest.raises(SystemExit):
            real.parse_args([*base,*extra,'--static-scene'])
    assert real.parse_args([*base,'--execute','--checked-review','--static-scene']).static_scene


@pytest.mark.parametrize('static',[False,True])
def test_static_flag_skips_both_captures_not_motion_pipeline(orchestration,monkeypatch,static):
    t = orchestration
    t.args.execute = t.args.checked_review = True
    t.args.static_scene = static
    calls = []
    first = dict(passed=True,depth_checked=not static,acquired_monotonic=time.monotonic()-100)
    def capture(s,c,o,n,*,auto_open):
        if static: pytest.fail('Static mode must skip camera, RGB and depth checks entirely')
        calls.append(n)
        assert auto_open is False
        return first if n == 'initial' else dict(passed=True,depth_checked=not static,acquired_monotonic=time.monotonic())
    monkeypatch.setattr(real,'capture_and_check',capture)
    answers(monkeypatch,[True])
    monkeypatch.setattr('builtins.input',lambda _: '')
    assert t.run(scene_check=None) == 0
    assert calls == ([] if static else ['initial','final'])
    assert t.report['static_scene_confirmed'] is static
    assert t.events.count('prepare') == t.events.count('preflight') == 1
    assert 'open' in t.events and 'approach' in t.events
    if static:
        assert t.report['initial_scene']['depth_checked'] is False
        assert t.report['final_scene']['depth_checked'] is False
        for name in ('initial_scene','final_scene'):
            scene = t.report[name]
            assert scene['capture_skipped'] is True and scene['checked'] is False
            assert scene['rgb_checked'] is False
            assert 'passed' not in scene and 'acquired_monotonic' not in scene
        assert t.report['initial_scene']['operator_confirmed'] is False
        assert t.report['final_scene']['operator_confirmed'] is True
        assert t.report['checked_review']['scene_policy'] == 'operator_fixed_reference'


@pytest.mark.parametrize('fault',['start','tcp','cancel'])
def test_static_mode_still_blocks_failures_before_hand_or_motion(orchestration,monkeypatch,fault):
    t = orchestration
    t.args.execute = t.args.checked_review = t.args.static_scene = True
    monkeypatch.setattr(real,'capture_and_check',lambda *a,**kw:pytest.fail('No scene rechecks in static mode'))
    answers(monkeypatch,[True])
    def video_answer(_):
        if fault == 'start':
            monkeypatch.setattr(real,'require_same_start',lambda *a:(_ for _ in ()).throw(RuntimeError('start changed')))
        if fault == 'tcp':
            monkeypatch.setattr(real,'check_tcp_offset',lambda *a:(_ for _ in ()).throw(RuntimeError('TCP changed')))
        return 'q' if fault == 'cancel' else ''
    monkeypatch.setattr('builtins.input',video_answer)
    if fault == 'cancel':
        assert t.run(scene_check=None) == 2
        assert t.report['static_scene_confirmed'] is False
    else:
        with pytest.raises(RuntimeError): t.run(scene_check=None)
    assert 'open' not in t.events and 'approach' not in t.events


def test_static_flag_requires_execution_even_in_direct_call(orchestration):
    t = orchestration
    t.args.static_scene = True
    with pytest.raises(ValueError,match='requires'):
        t.run()
    assert not t.events


@pytest.mark.parametrize('execute', [False, True])
def test_no_video_keeps_controller_checks_and_final_motion_gate(orchestration, monkeypatch, execute):
    t = orchestration
    t.args.video, t.args.execute, t.args.preflight_only = False, execute, not execute
    answers(monkeypatch, [True, True])
    assert t.run() == 0
    assert 'preflight' in t.events
    assert ('open' in t.events) == execute
    if execute:
        assert t.events.index('preflight') < t.events.index('final_scene') < t.events.index('open')
    assert t.report['numerical_preflight_required'] is True
    assert t.report['execution_video'] is None
    assert not list(t.tmp.glob('*.mp4'))
    assert str(t.tmp/'execution.json') in t.report['execution_sha256']
    assert str(t.tmp/'execution_precheck.json') in t.report['execution_sha256']
    assert [r['stage'] for r in t.report['timings']] == ['nominal_cb3_preflight',
        'connect_controller', 'controller_preflight', 'execution_geometry_and_optional_video']


def test_no_video_controller_failure_still_stops(orchestration, monkeypatch):
    t = orchestration
    t.args.video, t.args.execute = False, True
    answers(monkeypatch, [True])
    def fail(*a):
        raise RuntimeError('IK missing')
    monkeypatch.setattr(real, 'preflight_controller', fail)
    with pytest.raises(RuntimeError, match='IK missing'):
        t.run()
    assert 'open' not in t.events
    assert t.report['timings'][-1]['status'] == 'failed'


@pytest.mark.parametrize('filename', ['execution.json', 'execution_precheck.json'])
def test_no_video_changed_execution_records_stop_before_hand_write(orchestration, monkeypatch, filename):
    t = orchestration
    t.args.video, t.args.execute = False, True
    calls = [0]
    def confirm(prompt):
        calls[0] += 1
        if calls[0] == 2:
            (t.tmp/filename).write_text('changed')
        return True
    monkeypatch.setattr(real, 'confirm', confirm)
    with pytest.raises(ValueError, match='changed'):
        t.run()
    assert 'open' not in t.events


@pytest.mark.parametrize('field', ['hand_speed_raw', 'hand_force_raw'])
def test_high_hand_values_need_explicit_permission_before_live_setup(config, tmp_path, monkeypatch, field):
    import yaml
    config.update(hand_speed_raw=100, hand_force_raw=100)
    config[field] = 1000
    path = tmp_path/'high.yaml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(real.support, 'read_approval', lambda *a: pytest.fail('Must reject first'))
    with pytest.raises(ValueError, match='--allow-high-force'):
        real.main(['--approval', 'unused.json', '--config', str(path), '--execute', '--no-video'])
    real.require_hand_setting_authorization(config, True)


def test_lab_requests_1000_for_both_hand_settings_without_speeding_arm(config):
    assert config['hand_speed_raw'] == config['hand_force_raw'] == 1000
    assert config['preview_videos'] is False
    assert config['joint_speed_rad_s'] == .1 and config['cartesian_speed_m_s'] == .01


def test_open_hand_transmits_configured_1000_settings_to_fake_transport(config, tmp_path, monkeypatch):
    from rh56_controller import rh56_hand
    from tools import run_vision_plan_trial
    events = []
    class FakeHand:
        def __init__(self, *a, **kw):
            self.ser = SimpleNamespace(exclusive=False, close=lambda: None)
        def speed_set(self, values): events.append(('speed', values))
        def force_set(self, values): events.append(('force', values))
        def angle_set(self, values): events.append(('angles', values))
    monkeypatch.setattr(rh56_hand, 'RH56Hand', FakeHand)
    monkeypatch.setattr(run_vision_plan_trial, 'SerialLog', lambda ser, log: ser)
    backend = real.LiveBackend(config, tmp_path, {}, lambda: None)
    backend.held_snapshot = lambda: dict(q=[0.]*6, tcp=[0.]*6)
    backend.dwell = lambda seconds: None
    command = [1000, 1000, 1000, 1000, 900, 0]
    try:
        backend.open_hand(command)
        assert events == [('speed', [1000]*6), ('force', [1000]*6), ('angles', command)]
    finally:
        backend.close()


@pytest.mark.parametrize('active,expected', [(False, ['script', 'disconnect']), (True, ['stop', 'script', 'disconnect'])])
def test_cleanup_stops_only_active_motion_never_releases_hand(config, tmp_path, active, expected):
    events = []
    b = real.LiveBackend(config, tmp_path, {}, lambda: None)
    b.control = SimpleNamespace(stopJ=lambda _: events.append('stop'),
                                stopScript=lambda: events.append('script'),
                                disconnect=lambda: events.append('disconnect'))
    b.motion_active = active
    b.close()
    assert events == expected


def test_short_motion_waits_for_new_completion_token(config, tmp_path, monkeypatch):
    from tools import run_vision_pick_cycle
    monkeypatch.setattr(run_vision_pick_cycle, 'normal', lambda _: None)
    b = real.LiveBackend(config, tmp_path, {}, lambda: None)
    tcp = [-.2, -.4, .35, 0., 0., 0.]
    snapshot = dict(q=[0.]*6, tcp=tcp)
    b.snapshot = lambda: snapshot
    b.telemetry = lambda: (np.array(tcp), np.zeros(6), np.zeros(6))
    b.receiver = SimpleNamespace(getActualTCPSpeed=lambda: [0.]*6)
    settles = []
    def settle(seconds, **kwargs):
        settles.append((seconds, kwargs))
        return snapshot
    b.wait_stationary = settle
    values = iter([-1, -1, 0, -2])
    consumed = []
    def progress():
        value = next(values)
        consumed.append(value)
        return value
    b.control = SimpleNamespace(getAsyncOperationProgress=progress, moveJ=lambda *a: True)
    command = dict(revision=real.motion.REVISION, kind='moveJ', phase='start_to_pregrasp',
        controller_preflight_mode=config['controller_preflight_mode'],
        q=[0.]*6, goal_tcp=tcp, start_q=[0.]*6, start_tcp=tcp,
        speed=.1, accel=.2, samples=[dict(q=[0.]*6),dict(q=[0.]*6)])
    b.move_phase(command)
    assert consumed == [-1, -1, 0, -2]
    assert len(settles) == 1 and settles[0][0] == .3
    assert not b.motion_active
    assert b.held_goal == dict(q=command['q'], tcp=command['goal_tcp'])
    assert b.held_goal['tcp'] is not command['goal_tcp']
    assert b.held_goal['q'] is not command['q']


@pytest.fixture
def settling(config, tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(real.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(real.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    state = dict(tcp=[-.2,-.4,.35,0.,0.,0.], q=[0.]*6, safety=1,
                 speed=lambda: [0.]*6)
    b = real.LiveBackend(config, tmp_path, {}, lambda: None)
    b.receiver = SimpleNamespace(
        getRobotMode=lambda: 7, getSafetyMode=lambda: state['safety'],
        isProtectiveStopped=lambda: False, getActualTCPPose=lambda: state['tcp'],
        getActualQ=lambda: state['q'], getActualTCPSpeed=lambda: state['speed']())
    return b, state, clock


def test_settling_waits_through_transient_speed_without_relaxing_threshold(settling):
    b, state, clock = settling
    # Reproduce completed motion followed by transient TCP speed. A second
    # excursion resets the continuous dwell; no new motion command is sent.
    state['speed'] = lambda: [.002 if clock[0] < .1 or .26 <= clock[0] < .30 else 0.] + [0.]*5
    result = b.wait_stationary(.3, goal_q=np.zeros(6), goal_tcp=state['tcp'])
    assert clock[0] >= .6
    assert result['tcp'] == state['tcp'] and result['q'] == state['q']
    record = b.report['stationary_waits'][-1]
    assert record['status'] == 'stable' and record['speed_norm_limit'] == .001
    assert any(not row['stationary'] for row in record['samples'])
    assert b.control is None and b.hand is None


def test_settling_timeout_is_bounded_and_recorded(settling):
    b, state, clock = settling
    state['speed'] = lambda: [.002,0.,0.,0.,0.,0.]
    with pytest.raises(RuntimeError, match='timed out'):
        b.wait_stationary(.3, goal_q=np.zeros(6), goal_tcp=state['tcp'])
    assert 3.3 <= clock[0] <= 3.32
    assert b.report['stationary_waits'][-1]['status'] == 'failed'


@pytest.mark.parametrize('fault', ['safety', 'goal_position', 'goal_joints', 'nan_speed'])
def test_settling_fails_immediately_on_safety_telemetry_or_goal_drift(settling, fault):
    b, state, clock = settling
    tcp = list(state['tcp'])
    if fault == 'safety': state['safety'] = 3
    if fault == 'goal_position': state['tcp'][0] += .002
    if fault == 'goal_joints': state['q'][0] = .004
    if fault == 'nan_speed': state['speed'] = lambda: [float('nan')]*6
    with pytest.raises((RuntimeError, ValueError)):
        b.wait_stationary(.3, goal_q=np.zeros(6), goal_tcp=tcp)
    assert clock[0] == 0


def test_hand_dwell_preserves_original_held_origin_guard(settling):
    b, state, clock = settling
    b.held_origin = dict(tcp=list(state['tcp']), q=list(state['q']))
    state['tcp'][0] += .0002
    with pytest.raises(RuntimeError, match='drifted'):
        b.dwell(.125)
    assert clock[0] == 0


def test_next_phase_entry_transient_speed_uses_bounded_wait(settling):
    b, state, clock = settling
    b.held_origin = dict(tcp=list(state['tcp']), q=list(state['q']))
    state['speed'] = lambda: [.002 if clock[0] < .06 else 0.]+[0.]*5
    snapshot = b.held_snapshot()
    assert snapshot['q']==state['q'] and snapshot['tcp']==state['tcp']
    assert clock[0]>=.36
    assert b.report['stationary_waits'][-1]['status']=='stable'


def test_next_phase_entry_drift_still_fails(settling):
    b, state, clock = settling
    b.held_origin = dict(tcp=list(state['tcp']), q=list(state['q']))
    state['speed'] = lambda: [.002]+[0.]*5
    state['tcp'][0]+=.0002
    with pytest.raises(RuntimeError,match='drifted'):
        b.held_snapshot()
    assert clock[0]==0


@pytest.mark.parametrize('transient_speed', [False, True])
def test_post_motion_hold_uses_existing_arrival_band_not_preflight_band(settling, transient_speed):
    b, state, clock = settling
    b.held_origin = dict(tcp=list(state['tcp']), q=list(state['q']))
    b.held_goal = dict(tcp=list(state['tcp']), q=list(state['q']))
    # Regression magnitude from the 20260910T223418 run's close-settle error.
    state['tcp'][0] += .000103883
    state['q'][0] += .000168
    state['tcp'][5] += np.deg2rad(.019663)
    if transient_speed:
        state['speed'] = lambda: [.002 if clock[0] < .06 else 0.] + [0.]*5
    assert b.held_snapshot()['tcp'] == state['tcp']
    b.dwell(1.)
    assert b.report['stationary_waits'][-1]['status'] == 'stable'
    assert b.held_goal == b.held_origin  # No recentering on the changed telemetry.
    assert clock[0] >= 1.


@pytest.mark.parametrize('action', ['dwell', 'held_snapshot'])
@pytest.mark.parametrize('fault', ['position', 'rotation', 'joints', 'nan'])
def test_post_motion_hold_still_rejects_target_departure(settling, action, fault):
    b, state, clock = settling
    b.held_goal = dict(tcp=list(state['tcp']), q=list(state['q']))
    if fault == 'position': state['tcp'][0] += .0011
    if fault == 'rotation': state['tcp'][5] += np.deg2rad(.51)
    if fault == 'joints': state['q'][0] += .0031
    if fault == 'nan': state['q'][0] = float('nan')
    with pytest.raises((RuntimeError, ValueError)):
        b.dwell(1.) if action == 'dwell' else b.held_snapshot()
    assert clock[0] == 0


def test_post_motion_hold_never_walks_the_reference(settling):
    b, state, _ = settling
    b.held_goal = dict(tcp=list(state['tcp']), q=list(state['q']))
    origin_x = state['tcp'][0]
    for step in range(1, 10):
        state['tcp'][0] = origin_x + step * .0001
        b.held_snapshot()
        b.dwell(.02)
        assert b.held_goal['tcp'][0] == origin_x
    state['tcp'][0] = origin_x + .0011
    with pytest.raises(RuntimeError, match='left execution target'):
        b.dwell(.02)


def test_post_motion_hold_is_anchored_to_command_not_arrival_sample(settling):
    b, state, _ = settling
    b.held_goal = dict(tcp=list(state['tcp']), q=list(state['q']))
    b.held_origin = dict(tcp=list(state['tcp']), q=list(state['q']))
    b.held_origin['tcp'][0] += .00095
    state['tcp'][0] += .00104  # Only .09 mm from arrival, but outside goal band.
    with pytest.raises(RuntimeError, match='left execution target'):
        b.held_snapshot()


def test_linear_failure_cleanup_uses_stopL(config,tmp_path):
    events=[]
    b=real.LiveBackend(config,tmp_path,{},lambda:None)
    b.control=SimpleNamespace(stopL=lambda a:events.append('stopL'),
        stopJ=lambda a:pytest.fail('Wrong stop primitive'),stopScript=lambda:events.append('script'),
        disconnect=lambda:events.append('disconnect'))
    b.motion_active,b.active_kind=True,'moveL'
    b.close()
    assert events==['stopL','script','disconnect']
