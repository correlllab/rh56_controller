"""Interactive orchestration tests; no camera, GPU, robot, or video player."""
import json
from pathlib import Path

import pytest

from tools import run_ggx_review as review


@pytest.fixture
def preflight(tmp_path):
    folder = tmp_path / 'preflight'
    folder.mkdir()
    candidate = folder / 'candidates.yml'
    candidate.write_text('poses')
    (folder / 'summary.json').write_text(json.dumps(dict(
        schema='rh56_ggx_ur5_table_preflight/v1', simulation_only=True,
        candidate_yaml=str(candidate), candidate_yaml_sha256=review.digest(candidate),
        automatic_pass_ranks=[2, 5],
        object_collision_gate={'revision': 'robot_object_before_closure/v2'},
    )))
    (folder / 'candidates.csv').write_text('rank,passed\n0,False\n2,True\n5,True\n')
    for rank in [2, 5]:
        (folder / f'review_rank_{rank:03d}.mp4').write_bytes(b'fake video')
        (folder / f'trajectory_rank_{rank:03d}.csv').write_text('fake trajectory')
    return folder


def inputs(monkeypatch, answers):
    answers = iter(answers)
    def read(_):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr('builtins.input', read)


def test_reject_then_approve_records_exact_rank_no_execution(preflight, tmp_path, monkeypatch):
    inputs(monkeypatch, ['n fingers hit object', ''])
    monkeypatch.setattr(review.subprocess, 'Popen', lambda *a, **k: pytest.fail('No viewer expected'))
    out = tmp_path / 'run'
    assert review.main(['--review-run', str(preflight), '--out', str(out), '--no-open']) == 0
    report = review.read_json(out / 'summary.json')
    assert report['selected_rank'] == 5
    assert not report['executable'] and not report['robot_commands_sent']
    assert report['status'] == 'review_approved'
    assert report['reviews'][0]['reason'] == 'fingers hit object'
    assert str(preflight / 'review_rank_005.mp4') in report['reviews'][1]['reviewed_sha256']


@pytest.mark.parametrize('answers,status', [
    ([], 'cancelled'), (['q'], 'cancel'), (['r'], 'scene_invalidated'),
    (['n', 'n'], 'no_candidate_approved'),
])
def test_cancellation_invalidation_and_exhaustion(preflight, tmp_path, monkeypatch, answers, status):
    inputs(monkeypatch, answers)
    out = tmp_path / 'run'
    assert review.main(['--review-run', str(preflight), '--out', str(out), '--no-open']) == 2
    report = review.read_json(out / 'summary.json')
    assert report['status'] == status and report['selected_rank'] is None
    assert not report['executable']


def test_unknown_input_and_replay_never_approve(preflight, monkeypatch):
    inputs(monkeypatch, ['yes', 'p', 'q'])
    opened = []
    monkeypatch.setattr(review, 'show', lambda path, auto: opened.append(path))
    _, video, pins = next(review.reviewed_candidates(preflight))
    assert review.decision(video, pins, True) == ('cancel', '')
    assert opened == [video, video]


def test_artifact_modified_during_prompt_invalidates_approval(preflight, monkeypatch):
    _, video, pins = next(review.reviewed_candidates(preflight))
    def change(_):
        video.write_bytes(b'changed')
        return ''
    monkeypatch.setattr('builtins.input', change)
    with pytest.raises(ValueError, match='changed'):
        review.decision(video, pins, False)


def test_changed_candidates_and_missing_video_fail_closed(preflight):
    (preflight / 'candidates.yml').write_text('changed')
    with pytest.raises(ValueError, match='Candidates changed'):
        list(review.reviewed_candidates(preflight))


def test_missing_video_fails_closed(preflight):
    (preflight / 'review_rank_002.mp4').unlink()
    with pytest.raises(FileNotFoundError):
        list(review.reviewed_candidates(preflight))


@pytest.mark.parametrize('revision', [None, 'robot_object_before_closure/v1'])
def test_old_desk_only_pass_cannot_enter_new_review(preflight, revision):
    path = preflight / 'summary.json'
    report = review.read_json(path)
    del report['object_collision_gate']
    if revision:
        report['object_collision_gate'] = {'revision': revision}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Outdated collision'):
        list(review.reviewed_candidates(preflight))


def test_batch_dependencies_cannot_change_between_candidates(preflight):
    candidates = review.reviewed_candidates(preflight)
    next(candidates)
    (preflight / 'candidates.yml').write_text('changed after first review')
    with pytest.raises(ValueError, match='changed'):
        next(candidates)


def test_saved_cb3_review_retains_official_filter_bindings(preflight, monkeypatch):
    from tools import ggx_cb3_candidates as cb3
    path = preflight / 'summary.json'
    report = review.read_json(path)
    official = preflight / 'official.json'
    official.write_text('saved official result')
    report.update(schema=cb3.SCHEMA, feasibility={'local_state': 'passed_current_checks'},
                  source_sha256=review.pin([official]), official_filter_report=str(official))
    path.write_text(json.dumps(report))
    monkeypatch.setattr(cb3, 'load_filter', lambda *a: {
        'candidates': [{'official_scene_clear': True}] * 6})
    candidates = review.reviewed_candidates(preflight)
    rank, _, pins = next(candidates)
    assert rank == 2 and pins[str(official)] == review.digest(official)
    official.write_text('changed after review')
    with pytest.raises(ValueError, match='changed'):
        next(candidates)


def test_saved_cb3_review_rejects_uncertain_state(preflight, monkeypatch):
    from tools import ggx_cb3_candidates as cb3
    path = preflight / 'summary.json'
    report = review.read_json(path)
    official = preflight / 'official.json'
    official.write_text('saved official result')
    report.update(schema=cb3.SCHEMA, feasibility={'local_state': 'uncertain_target_proxy'},
                  source_sha256=review.pin([official]), official_filter_report=str(official))
    path.write_text(json.dumps(report))
    monkeypatch.setattr(cb3, 'load_filter', lambda *a: {
        'candidates': [{'official_scene_clear': True}] * 6})
    with pytest.raises(ValueError, match='not an automatic-pass'):
        list(review.reviewed_candidates(preflight))


def test_execute_rejected_before_any_camera_or_stage_call(monkeypatch):
    monkeypatch.setattr(review, 'run_stage', lambda *a, **k: pytest.fail('Must not run stage'))
    with pytest.raises(SystemExit) as exc:
        review.main(['--capture-live', '--execute'])
    assert exc.value.code == 2


@pytest.mark.parametrize('static',[False,True])
@pytest.mark.parametrize('statuses', [['approach_completed'], ['cb3_review_vetoed','approach_completed'],
                                    ['cb3_candidate_rejected','approach_completed'],
                                    ['cancelled_before_motion'], ['scene_invalidated'], ['failed']])
def test_execute_checks_cb3_before_approval_and_only_veto_advances(preflight, tmp_path, monkeypatch, statuses, static):
    from tools import run_ggx_real
    calls = []
    authorization_objects = []
    def real_run(argv, *, controller_authorization):
        parsed = run_ggx_real.parse_args(argv)
        proposal = review.read_json(parsed.candidate)
        assert proposal['human_approved'] is False and proposal['executable'] is False
        assert '--approval' not in argv and parsed.checked_review
        assert parsed.allow_high_force
        assert parsed.static_scene is static
        assert proposal['rank'] == [2, 5][len(calls)]
        calls.append(argv)
        authorization_objects.append(controller_authorization)
        status = statuses[len(calls)-1]
        parsed.out.mkdir()
        (parsed.out/'summary.json').write_text(json.dumps(dict(status=status,
            motion_confirmation_received=status=='approach_completed',
            motion_commands_attempted=status=='approach_completed',
            checked_review=dict(decision='veto' if status=='cb3_review_vetoed' else 'approve'))))
        return 0 if status=='approach_completed' else 1 if status=='failed' else 2
    monkeypatch.setattr(run_ggx_real, 'main', real_run)
    monkeypatch.setattr(review,'decision',lambda *a,**kw:pytest.fail('No earlier UR5e video approval'))
    out = tmp_path / 'handoff'
    assert review.main(['--review-run', str(preflight), '--out', str(out), '--no-open',
                        '--execute', '--allow-high-force', *(['--static-scene'] if static else [])]) == (0 if statuses[-1]=='approach_completed' else
                                                                 1 if statuses[-1]=='failed' else 2)
    assert len(calls) == len(statuses)
    assert all(a is authorization_objects[0] for a in authorization_objects)
    report = review.read_json(out/'summary.json')
    assert report['status'] == statuses[-1]
    assert report['selected_rank'] == ([2,5][len(calls)-1] if statuses[-1]=='approach_completed' else None)
    assert report['robot_commands_sent'] == (statuses[-1]=='approach_completed')


def test_checked_execution_video_cannot_be_disabled():
    from tools import run_ggx_real as real
    with pytest.raises(SystemExit):
        review.parse_args(['--review-run','saved','--execute','--no-real-video'])
    with pytest.raises(SystemExit):
        real.parse_args(['--candidate','candidate.json','--execute','--no-video'])
    assert real.parse_args(['--candidate','candidate.json']).checked_review
    assert real.parse_args(['--approval','approval.json','--checked-review']).checked_review


def test_static_scene_review_flag_requires_execute():
    with pytest.raises(SystemExit):
        review.parse_args(['--review-run','saved','--static-scene'])


def test_high_settings_without_flag_stop_before_reset_or_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'run_stage', lambda *a, **kw: pytest.fail('No stages before authorization'))
    monkeypatch.setattr(review, 'generate_candidates', lambda *a: pytest.fail('No inference'))
    out = tmp_path/'not_created'
    with pytest.raises(ValueError, match='--allow-high-force'):
        review.main(['--capture-live', '--reset-first', '--execute', '--dimensions-mm', '40', '90', '176',
                     '--out', str(out)])
    assert not out.exists()


def test_input_mode_must_be_explicit_and_cannot_mix_live_with_old_capture():
    for argv in [[], ['--capture-live', '--capture', 'old'], ['--capture-live'],
                 ['--capture-live', '--dimensions-mm', '40', '90', 'nan']]:
        with pytest.raises(SystemExit):
            review.parse_args(argv)


def test_sim_configuration_reuses_perception_desk_origin_and_box_axes(tmp_path):
    plane = tmp_path / 'plane.yaml'
    plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    config = review.sim_config(
        dict(center_base_mm=[-230, -630, 210], short_axis_yaw_deg=88),
        [40, 90, 176], dict(desk_plane=str(plane), desk_z_mm=34),
        dict(q=[0, 0, 0, 0, 0, 0]), Path('candidates.yml'),
    )
    assert config['object_yaw_deg'] == -2
    assert config['candidate_frame'] == 'ur-base'
    assert config['object_size_mm'] == [90, 40, 176]
    assert config['table_height_m'] == pytest.approx(.034)


def test_foreground_mode_is_explicit_and_flat_top_remains_default():
    base = ['--capture','saved','--dimensions-mm','40','90','180']
    assert review.parse_args(base).crop_mode == 'flat-top'
    assert review.parse_args([*base,'--crop-mode','foreground']).crop_mode == 'foreground'
    with pytest.raises(SystemExit):
        review.parse_args([*base,'--crop-mode','skip'])
    with pytest.raises(SystemExit):
        review.parse_args(['--review-run','old','--crop-mode','foreground'])


def test_manual_fallback_defaults_enabled_and_can_be_disabled():
    assert review.parse_args(['--review-run','saved']).fallback_review
    assert not review.parse_args(['--review-run','saved','--no-fallback-review']).fallback_review


def test_execution_suppresses_diagnostic_rendering_and_enter(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import preflight_ggx_ur5_table as preflight
    monkeypatch.setattr(preflight, 'write_video', lambda *a, **kw: pytest.fail('No diagnostic rendering in execution mode'))
    monkeypatch.setattr(preflight, 'manual_object_review_eligible', lambda *a: pytest.fail('Already evaluated; do not recalculate'))
    report = dict(status='running', executable=False, robot_commands_sent=False)
    candidates = review.manual_review_candidates(tmp_path, None, None,
        [(SimpleNamespace(rank=20), [], {})], {}, report, execution=True)
    assert list(candidates) == []
    assert report['status'] == 'no_executable_candidates'
    assert report['manual_fallback']['suppressed_for_execution'] is True
    assert report['manual_fallback']['shown_ranks'] == []
    assert not report['robot_commands_sent']


@pytest.mark.parametrize('execute', [False, True])
def test_manual_candidate_enter_is_offline_only(preflight, tmp_path, monkeypatch, execute):
    candidate = next(review.reviewed_candidates(preflight))
    monkeypatch.setattr(review, 'reviewed_candidates', lambda *a: (c for c in [candidate]))
    monkeypatch.setattr(review, 'review_is_manual_only', lambda *a: True)
    monkeypatch.setattr(review, 'review_checked_candidate', lambda *a: pytest.fail('Rejected candidate must not reach hardware'))
    prompts = []
    def decide(*a, **kw):
        assert kw['manual_only'] is True
        prompts.append(kw)
        return 'approve', ''
    monkeypatch.setattr(review, 'decision', decide)
    out = tmp_path/'manual_handoff'
    args = ['--review-run', str(preflight), '--out', str(out), '--no-open']
    if execute: args += ['--execute', '--allow-high-force']
    assert review.main(args) == (2 if execute else 0)
    result = review.read_json(out/'summary.json')
    assert bool(prompts) is not execute
    assert result['status'] == ('no_candidate_approved' if execute else 'manual_review_accepted_not_executable')
    assert result['robot_commands_sent'] is False


def test_manual_review_type_requires_a_pinned_manifest(tmp_path):
    video=tmp_path/'manual.mp4';video.write_bytes(b'video')
    manifest=tmp_path/'summary.json'
    manifest.write_text(json.dumps(dict(schema=review.MANUAL_REVIEW_SCHEMA)))
    with pytest.raises(ValueError,match='does not bind'):
        review.review_is_manual_only(video,review.pin([video]))
    pins=review.pin([manifest,video])
    assert review.review_is_manual_only(video,pins)
    manifest.write_text(json.dumps(dict(schema='rh56_ggx_ur5_table_preflight/v1')))
    with pytest.raises(ValueError,match='changed'):
        review.review_is_manual_only(video,pins)


def test_foreground_sim_uses_enclosing_proxy_not_top_median_or_smaller_prior(tmp_path):
    from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA
    plane = tmp_path/'plane.yaml'; plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    crop = dict(schema=FOREGROUND_SCHEMA,crop_mode='foreground',
        status='measured_crop_ready_for_offline_inference',hardware_contacted=False,executable_target=False,
        dimensions_prior_mm=[40,90,180],top_patch=None,
        object_frame=dict(center_base_mm=[-200,-600,34],origin='desk_footprint_center',
                          yaw_ambiguous=False,short_axis_yaw_deg=85),
        collision_proxy=dict(shape='box',dimensions_mm=[54,95,196],observed_bounds_mm=[53.9,94.7,195.1]))
    config = review.crop_sim_config(crop,[40,90,180],dict(desk_plane=str(plane),desk_z_mm=34),
                                   dict(q=[0]*6),tmp_path/'candidates.yml')
    assert config['object_size_mm'] == [95,54,196]
    assert config['object_yaw_deg'] == -5
    assert config['object_x_m'] == -.2 and config['object_y_m'] == -.6
    assert crop['dimensions_prior_mm'] == [40,90,180]
    with pytest.raises(ValueError,match='dimensions changed'):
        review.crop_sim_config(crop,[40,90,176],{},dict(q=[0]*6),tmp_path/'candidates.yml')


def test_subprocess_failure_surfaces_specific_crop_error(tmp_path,monkeypatch):
    from types import SimpleNamespace
    out = tmp_path/'crop'; out.mkdir()
    (out/'summary.json').write_text(json.dumps(dict(error='Flat top mismatch; use --crop-mode foreground')))
    monkeypatch.setattr(review.subprocess,'run',lambda *a,**kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError,match='Flat top mismatch; use --crop-mode foreground'):
        review.run_stage(tmp_path,'crop','prepare_ggx_tabletop_cloud.py',[])


def test_stage_failure_stops_and_records_error(preflight, tmp_path, monkeypatch):
    def failed(*args):
        raise RuntimeError('mock inference failure')
    monkeypatch.setattr(review, 'generate_candidates', failed)
    out = tmp_path / 'run'
    assert review.main(['--capture', str(preflight), '--dimensions-mm', '40', '90', '176',
                        '--out', str(out), '--no-open']) == 1
    report = review.read_json(out / 'summary.json')
    assert report['status'] == 'failed'
    assert report['reviews'] == [] and not report['executable']


def test_existing_output_is_never_overwritten(preflight, tmp_path):
    with pytest.raises(FileExistsError):
        review.main(['--review-run', str(preflight), '--out', str(tmp_path), '--no-open'])


def test_ggx_interpreter_preserves_virtual_environment_symlink(tmp_path):
    binary = tmp_path / 'base_python'
    binary.touch()
    venv_python = tmp_path / 'venv_python'
    venv_python.symlink_to(binary)
    args = review.parse_args(['--review-run', str(tmp_path), '--ggx-python', str(venv_python)])
    assert args.ggx_python == venv_python
    assert args.ggx_python != binary
