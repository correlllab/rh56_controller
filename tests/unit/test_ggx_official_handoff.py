"""Official endpoint selection is not a local path or physical-hand certificate."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mink')

from tools import ggx_real_support as support
from tools import ggx_planning as planning
from tools import ggx_official_candidates as official_candidates

STRICT_OFFICIAL_READER = official_candidates.reviewed_candidates


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def selection(tmp_path, monkeypatch):
    """Real hashed provenance files, with only external configuration APIs mocked."""
    from tools import ggx_cb3_candidates, ggx_reset, preview_tabletop_target, run_ggx_real
    candidate = write(tmp_path/'inference/candidates.yml', {'fixture': True})
    cloud = write(tmp_path/'crop/cloud.json', {'points': [[0, 0, 0]]})
    calibration = write(tmp_path/'calibration.json', {})
    desk_plane = write(tmp_path/'desk.json', {})
    config = write(tmp_path/'perception_config.json', {})
    real_config = write(tmp_path/'real_config.json', {})
    capture = tmp_path/'capture'
    for name in ('camera_check_rgb.png', 'camera_check_depth_u16.png', 'camera_check.yaml'):
        write(capture/name, {})
    perception = write(tmp_path/'perception/summary.json', dict(
        calibration=str(calibration), config=str(config), desk_plane=str(desk_plane), capture=str(capture)))
    write(cloud.parent/'summary.json', dict(schema='rh56_ggx_measured_crop/v1',
        point_cloud=str(cloud), source_preview=str(perception),
        source_preview_sha256=support.digest(perception),
        calibration_sha256=support.digest(calibration),
        source_depth_sha256=support.digest(capture/'camera_check_depth_u16.png')))
    write(candidate.parent/'summary.json', dict(schema='rh56_ggx_point_cloud_inference/v1',
        input_frame='ur_base', candidate_sha256=support.digest(candidate),
        source_point_cloud=str(cloud), source_point_cloud_sha256=support.digest(cloud)))
    official = write(tmp_path/'official.json', {})
    envelope = {'fixture': 'same fixed desk'}
    dependencies = support.pin([candidate, official, config, real_config])
    preflight = dict(schema=support.OFFICIAL_SELECTION_SCHEMA,
        selection_scope=support.OFFICIAL_SELECTION_SCOPE, official_review_ranks=[0],
        passed=False, executable=False, automatic_pass_ranks=[],
        local_execution_checked=False, actual_travel_checked=False,
        candidate_frame='ur-base', candidate_yaml=str(candidate),
        candidate_yaml_sha256=support.digest(candidate),
        object_center_m=[-.2, -.6, .15], object_size_mm=[90, 40, 180],
        object_yaw_deg=0., table_height_m=.06, arguments={},
        official_filter_report=str(official), source_sha256=dependencies,
        real_config=str(real_config), perception_config=str(config), desk_envelope=envelope)
    report_path = write(tmp_path/'rank_000/summary.json', preflight)
    video = write(report_path.parent/'review.svg', {})
    approval_path = tmp_path/'review.json'
    approval = dict(schema='rh56_ggx_interactive_review/v1',
        status='official_candidate_selected', executable=False, selected_rank=0,
        reviews=[dict(rank=0, decision='approve', scope=support.OFFICIAL_SELECTION_SCOPE,
                      video=str(video), reviewed_sha256={})])
    def save():
        write(report_path, preflight)
        approval['reviews'][0]['reviewed_sha256'] = {
            **dependencies, **support.pin([report_path, video])}
        write(approval_path, approval)
        return approval_path
    save()
    official_result = {'candidates': [{'official_scene_clear': True}]}
    shared_reader_calls = []
    def shared_reader(directory, *, require_video):
        shared_reader_calls.append((directory, require_video))
        yield 0, video, approval['reviews'][0]['reviewed_sha256']
    monkeypatch.setattr(official_candidates, 'reviewed_candidates', shared_reader)
    monkeypatch.setattr(ggx_cb3_candidates, 'load_filter', lambda *a: official_result)
    monkeypatch.setattr(run_ggx_real, 'load_config', lambda *a: {'desk_envelope': envelope})
    monkeypatch.setattr(preview_tabletop_target, 'load_perception_config', lambda *a: ({}, None))
    monkeypatch.setattr(ggx_reset, 'desk_source', lambda *a: {'preflight': {'table_height_m': .05}})
    return SimpleNamespace(path=approval_path, approval=approval, report=preflight,
        report_path=report_path, save=save, official=official, official_result=official_result,
        shared_reader_calls=shared_reader_calls)


def test_official_selection_preserves_provenance_without_local_certificate(selection):
    source = support.read_approval(selection.path)
    assert source['selection_scope'] == support.OFFICIAL_SELECTION_SCOPE
    assert source['official_scene_clear'] is True
    assert source['local_execution_checked'] is False
    assert source['physical_hand_verified'] is False
    assert source['desk_top_m'] == .05  # Independent fixed desk, not proxy bottom .06.
    assert source['preflight']['automatic_pass_ranks'] == []
    assert str(selection.official) in source['pins']
    assert selection.shared_reader_calls == [(selection.report_path.parent, False)]


def test_handoff_invokes_actual_strict_reader_and_rejects_unchecked_metadata(selection, monkeypatch):
    monkeypatch.setattr(official_candidates, 'reviewed_candidates', STRICT_OFFICIAL_READER)
    # All handoff provenance is valid, but these fields must never be upgraded
    # to local passes; this is rejected by the actual shared reader, not a mock.
    selection.report.update(status='official_candidate_reviewable', rank=0,
        hardware_contacted=False, local_checks={'ik': 'PASS'})
    selection.save()
    with pytest.raises(ValueError, match='non-executable official-only candidate selection'):
        support.read_approval(selection.path)


@pytest.mark.parametrize('fault', ['rank', 'changed_pin', 'unapproved_dependency'])
def test_shared_reader_cannot_change_selected_rank_or_refresh_approved_pins(selection, monkeypatch, fault):
    pins = dict(selection.approval['reviews'][0]['reviewed_sha256'])
    rank = 0
    if fault == 'rank':
        rank = 1
    elif fault == 'changed_pin':
        pins[str(selection.official)] = 'newly derived but not approved digest'
    else:
        pins[str(selection.path.parent/'unapproved-new-asset')] = 'unapproved digest'
    def reopen(*a, **kw):
        yield rank, selection.report_path, pins
    monkeypatch.setattr(official_candidates, 'reviewed_candidates', reopen)
    with pytest.raises(ValueError, match='handoff rank|handoff provenance'):
        support.read_approval(selection.path)


@pytest.mark.parametrize('field,value', [
    ('status', 'review_approved'), ('scope', 'grasp_geometry')])
def test_official_approval_cannot_impersonate_local_review(selection, field, value):
    if field == 'scope':
        selection.approval['reviews'][0][field] = value
    else:
        selection.approval[field] = value
    selection.save()
    with pytest.raises(ValueError, match='scope/status'):
        support.read_approval(selection.path)


@pytest.mark.parametrize('field,value', [
    ('automatic_pass_ranks', [0]), ('local_execution_checked', True),
    ('actual_travel_checked', True), ('passed', True), ('executable', True)])
def test_official_report_cannot_inherit_local_or_motion_pass(selection, field, value):
    selection.report[field] = value
    selection.save()
    with pytest.raises(ValueError, match='no local/path approval'):
        support.read_approval(selection.path)


def test_changed_official_filter_is_not_rebound_at_handoff(selection):
    selection.official.write_text('changed filter after selection')
    with pytest.raises(ValueError, match='changed'):
        support.read_approval(selection.path)


def test_official_filter_reject_is_never_a_selected_candidate(selection):
    selection.official_result['candidates'][0]['official_scene_clear'] = False
    with pytest.raises(ValueError, match='Official reject'):
        support.read_approval(selection.path)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    from tools import run_ggx_real
    source = dict(selection_scope=support.OFFICIAL_SELECTION_SCOPE, pins={}, rank=0,
        checker_config='fixture', official_scene_clear=True, physical_hand_verified=False,
        preflight=dict(arguments={}, candidate_yaml='fixture.yml', object_center_m=[-.2, -.6, .15]))
    monkeypatch.setattr(support, 'build_model', lambda *a: 'same-local-model')
    monkeypatch.setattr(run_ggx_real, 'load_config', lambda *a: {'fixture': 'same config'})
    monkeypatch.setattr(support.sim, 'load_isaac_grasp_yaml', lambda *a: ['same-ranked-candidate'])
    monkeypatch.setattr(support.sim, 'candidate_for_object_origin', lambda c, *a: c)
    detail = dict(state='geometry_eligible', standoff_mm=80,
                  local_execution_check={'passed': True}, actual_travel_checked=False)
    events = []
    samples = [SimpleNamespace(min_desk_clearance_m=.01)]
    def select(model, candidate, rank, args, center, config):
        assert model == 'same-local-model' and candidate == 'same-ranked-candidate' and rank == 0
        assert args.desk_clearance_mm == 2. and config == {'fixture': 'same config'}
        events.append('select_geometry')
        return detail, samples
    monkeypatch.setattr(planning, 'select_geometry', select)
    rows = [dict(phase=phase, tcp=[0., 0., .4, 0., 0., 0.], q_nominal=[0.]*6,
                 closure_alpha=.5 if phase in ('close', 'lift') else 0., hand_raw=[1000]*5+[0])
            for phase in ('start', 'start_to_pregrasp', 'approach', 'close', 'lift')]
    def exact(*a):
        events.append('exact_reference_rows')
        return rows
    monkeypatch.setattr(planning, 'exact_reference_rows', exact)
    monkeypatch.setattr(support, 'raw_hand_command', lambda *a: [1000]*5+[0])
    def execution(model, selected, reference, canonical, snapshot, config, out, video):
        events.append('execution')
        assert selected is source and reference['source'] is source
        assert canonical == rows[1:] and selected['physical_hand_verified'] is False
        return dict(reference, geometry_reference_only=False), model
    monkeypatch.setattr(planning, 'execution', execution)
    return SimpleNamespace(source=source, original=copy.deepcopy(source), detail=detail,
        events=events, out=tmp_path, run=lambda: support.prepare(source, {'q': [0.]*6}, tmp_path))


def test_deferred_local_preparation_reuses_policy_without_modifying_selection(preparation):
    p = preparation
    prepared, _ = p.run()
    assert p.events == ['select_geometry', 'exact_reference_rows', 'execution']
    assert p.source == p.original
    assert prepared['physical_hand_verified'] is False and prepared['executable'] is False
    report = support.read_json(p.out/'local_assessment.json')
    assert report['status'] == 'local_geometry_ready'
    assert report['actual_travel_status'] == 'nominal_path_ready'
    assert report['official_scene_clear'] and report['official_selection_unchanged']
    assert not report['hardware_contacted'] and not report['physical_hand_verified']
    assert report['hand_contract']['local_open_raw_command'] == [1000]*5+[0]
    assert report['hand_contract']['status'] == 'physical_mapping_unverified'


def test_local_proxy_failure_retains_official_pass_and_blocks_actual_route(preparation):
    p = preparation
    p.detail.update(state='local_approach_unresolved', local_execution_check={'passed': False},
        local_failures=[dict(kind='target_proxy_uncertain', phase='approach', clearance_mm=-1.)])
    with pytest.raises(planning.PlanningUnavailable, match='local_geometry_unavailable'):
        p.run()
    assert p.events == ['select_geometry'] and p.source == p.original
    report = support.read_json(p.out/'local_assessment.json')
    assert report['status'] == 'local_geometry_unavailable' and report['official_scene_clear']
    assert report['actual_travel_checked'] is False
    assert report['geometry']['local_failures'][0]['kind'] == 'target_proxy_uncertain'


def test_actual_route_failure_keeps_local_geometry_and_official_attribution(preparation, monkeypatch):
    p = preparation
    def fail(*a, **kw):
        raise planning.PlanningUnavailable('actual_route_search_exhausted', [{'kind': 'desk'}])
    monkeypatch.setattr(planning, 'execution', fail)
    with pytest.raises(planning.PlanningUnavailable, match='actual_route_search_exhausted'):
        p.run()
    report = support.read_json(p.out/'local_assessment.json')
    assert report['status'] == 'local_geometry_ready' and report['official_scene_clear']
    assert report['actual_travel_status'] == 'unavailable'
    assert report['actual_travel_attempts'] == [{'kind': 'desk'}]
    assert p.source == p.original


def test_official_handoff_does_not_change_existing_raw_open_mapping():
    model = support.build_model(dict(preflight=dict(object_center_m=[-.2, -.6, .14],
        object_size_mm=[90, 40, 180], object_yaw_deg=0., table_height_m=.05)))
    assert support.raw_hand_command(model, 0.) == [1000, 1000, 1000, 1000, 1000, 0]
    assert np.isfinite(model.qpos0).all()
