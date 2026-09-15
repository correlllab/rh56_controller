"""Official-only selection keeps local execution unresolved; no lab dependencies."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tools import ggx_official_candidates as selection
from tools import run_ggx_official_filter as official
from tools import run_ggx_review as ui


def fixture(tmp_path, *, count=3, rejected=(1,), scores=None):
    scores = scores or [1-i/(count+1) for i in range(count)]
    candidate = tmp_path/'candidates.yml'
    candidate.write_text(yaml.safe_dump(dict(format='isaac_grasp', grasps={
        f'grasp_{i}': dict(confidence=float(score), position=[.1, -.5, .2],
                         orientation=dict(xyz=[0., 0., 0.], w=1.))
        for i, score in enumerate(scores)}), sort_keys=False))
    cloud = tmp_path/'object.npy'; np.save(cloud, np.ones((5, 3)))
    inference = tmp_path/'inference.json'
    inference.write_text(json.dumps(dict(schema='rh56_ggx_point_cloud_inference/v1',
        status='candidates_generated_not_motion_validated', topk=100, candidates=count,
        input_frame='ur_base', candidate_file=str(candidate), candidate_sha256=ui.digest(candidate),
        source_point_cloud=str(cloud), source_point_cloud_sha256=ui.digest(cloud))))
    mesh = tmp_path/'coll_mesh.obj'
    mesh.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n')
    (tmp_path/'config.json').write_text(json.dumps(dict(open=dict(thumb_proximal_pitch_joint=0.))))
    function = tmp_path/'collision_filter.py'; function.write_text('# fixed official source fixture\n')
    scene = tmp_path/'scene_without_target_ur_base_m.npy'; np.save(scene, np.ones((8, 3)))
    surface = tmp_path/'official_hand_surface_m.npy'; np.save(surface, np.ones((9, 3)))
    report = dict(schema=official.SCHEMA, status='completed', executable=False,
        hardware_contacted=False, candidate_file=str(candidate), candidate_sha256=ui.digest(candidate),
        inference=str(inference), official_function=str(function), threshold_mm=20.,
        source_sha256=ui.pin([candidate, cloud, inference, mesh, scene, surface, function]),
        candidates=[dict(rank=i, name=f'grasp_{i}', confidence=float(score),
                         official_scene_clear=i not in rejected) for i, score in enumerate(scores)])
    filter_path = tmp_path/'filter.json'; filter_path.write_text(json.dumps(report))
    config = tmp_path/'simulation.yaml'
    config.write_text(yaml.safe_dump(dict(candidate_frame='ur-base', graspgenx_yaml=str(candidate))))
    out = tmp_path/'output'; out.mkdir()
    args = SimpleNamespace(max_candidates=100, candidate_rank=None, render_candidates=False)
    return args, out, config, filter_path, candidate


def test_original_order_filter_only_and_every_local_gate_unchecked(tmp_path, monkeypatch):
    args, out, config, path, candidates = fixture(tmp_path)
    original_import = builtins.__import__
    forbidden = ('mujoco', 'mink', 'rtde', 'serial', 'pyrealsense2', 'tools.ggx_planning',
                 'tools.ggx_real_support', 'tools.preflight_ggx_ur5_table', 'tools.ggx_template')
    def no_local_import(name, *a, **kw):
        assert not name.startswith(forbidden), f'Candidate selection imported execution dependency: {name}'
        return original_import(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', no_local_import)
    report = {'status': 'running'}
    values = list(selection.official_candidates(args, out, report, config, ui.pin([config]), filter_path=path))
    assert [rank for rank, _, _ in values] == [0, 2]
    assert report['feasibility_counts'] == dict(candidates=3, official_rejected=1,
        official_scene_eligible=2, local_evaluated=0, actual_travel_checked=0)
    assert [r['rank'] for r in report['feasibility_stages']] == [0, 1, 2]
    for rank, view, pins in values:
        result = ui.read_json(view)
        assert result['schema'] == selection.SCHEMA
        assert result['automatic_pass_ranks'] == [] and result['official_review_ranks'] == [rank]
        assert result['executable'] is result['passed'] is result['hardware_contacted'] is False
        assert set(result['local_checks'].values()) == {'NOT_EVALUATED'}
        assert result['candidate_yaml_sha256'] == ui.digest(candidates)
        assert result['actual_travel_checked'] is False
        assert result['local_execution_checked'] is False
        ui.verify(pins)
    assert (out/'feasibility.csv').is_file() and (out/'summary.csv').is_file()


def test_no_lower_rank_refill_after_official_rejections(tmp_path):
    args, out, config, path, _ = fixture(tmp_path, rejected=(0, 1))
    args.max_candidates = 2
    report = {'status': 'running'}
    assert not list(selection.official_candidates(args, out, report, config, {}, filter_path=path))
    assert report['feasibility_counts']['candidates'] == 2
    assert report['full_batch_counts']['candidates'] == 3
    assert report['full_batch_counts']['official_scene_eligible'] == 1


@pytest.mark.parametrize('count', [101, 1000])
def test_all_candidate_archive_cannot_replace_top100(tmp_path, count):
    _, _, _, path, candidates = fixture(tmp_path, count=count)
    with pytest.raises(ValueError, match='top-100'):
        selection.load_inputs(path, candidates)


def test_file_order_not_silently_resorted(tmp_path):
    _, _, _, path, candidates = fixture(tmp_path, scores=[.5, .9, .7])
    with pytest.raises(ValueError, match='Exporter order'):
        selection.load_inputs(path, candidates)


@pytest.mark.parametrize('field,value', [('confidence', .01), ('name', 'other'),
                                      ('rank', 2), ('official_scene_clear', 1)])
def test_filter_metadata_cannot_relabel_or_rerank(tmp_path, field, value):
    _, _, _, path, candidates = fixture(tmp_path)
    report = ui.read_json(path); report['candidates'][0][field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Official filter changed'):
        selection.load_inputs(path, candidates)


def test_exact_filter_mesh_and_surface_are_bound(tmp_path):
    _, _, _, path, candidates = fixture(tmp_path)
    _, _, _, assets, _ = selection.load_inputs(path, candidates)
    assert assets['mesh'].name == 'coll_mesh.obj'
    assets['surface'].write_bytes(b'changed samples')
    with pytest.raises(ValueError, match='artifact changed'):
        selection.load_inputs(path, candidates)


def test_uses_exact_official_assets_when_rendering(tmp_path, monkeypatch):
    args, out, config, path, _ = fixture(tmp_path)
    args.render_candidates = True
    seen = []
    def render(view, candidate, rank, assets):
        seen.append((rank, assets)); view.write_bytes(b'preview fixture')
    monkeypatch.setattr(selection, 'render_candidate', render)
    results = list(selection.official_candidates(args, out, {'status':'running'}, config, {}, filter_path=path))
    assert [rank for rank, _ in seen] == [0, 2]
    assert all(assets['mesh'] == tmp_path/'coll_mesh.obj' for _, assets in seen)
    assert all(view.name == 'review_official.png' for _, view, _ in results)


def test_pinned_source_change_during_review_stops_next_candidate(tmp_path):
    args, out, config, path, _ = fixture(tmp_path)
    generator = selection.official_candidates(args, out, {'status':'running'}, config, {}, filter_path=path)
    assert next(generator)[0] == 0
    (tmp_path/'coll_mesh.obj').write_text('modified')
    with pytest.raises(ValueError, match='artifact changed'):
        next(generator)


@pytest.mark.parametrize('maximum,rank', [(101, None), (0, None), (100, 100), (2, 2)])
def test_explicit_review_budget_cannot_expand_top100(tmp_path, maximum, rank):
    args, out, config, path, _ = fixture(tmp_path)
    args.max_candidates, args.candidate_rank = maximum, rank
    with pytest.raises(ValueError, match='top-100'):
        list(selection.official_candidates(args, out, {}, config, {}, filter_path=path))


def test_cli_help_without_hardware_and_no_execute_option():
    with pytest.raises(SystemExit) as raised:
        selection.main(['--help'])
    assert raised.value.code == 0
    with pytest.raises(SystemExit) as raised:
        selection.main(['--execute'])
    assert raised.value.code == 2


def test_official_selection_reopens_without_becoming_execution_approval(tmp_path):
    args, out, config, path, _ = fixture(tmp_path)
    created = list(selection.official_candidates(args, out, {'status':'running'}, config, {}, filter_path=path))
    rank, view, _ = created[0]
    reopened = list(selection.reviewed_candidates(view.parent, require_video=False))
    assert reopened[0][0] == rank
    summary = ui.read_json(view); summary['local_checks']['robot_ik'] = 'PASS'
    view.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='non-executable'):
        list(selection.reviewed_candidates(view.parent, require_video=False))


def test_deferred_proxy_and_config_metadata_do_not_claim_checks(tmp_path):
    real = tmp_path/'real.yaml'
    real.write_text(yaml.safe_dump(dict(desk_envelope={'confirmed':True})))
    perception = tmp_path/'perception.yaml'; perception.write_text('schema: fixture\n')
    metadata, pins = selection.deferred_metadata(SimpleNamespace(real_config=real, config=perception),
        dict(object_x_m=.1, object_y_m=-.5, table_height_m=.03, object_size_mm=[40, 90, 180],
             object_yaw_deg=10))
    assert np.allclose(metadata['object_center_m'], [.1, -.5, .12])
    assert metadata['desk_envelope'] == {'confirmed':True}
    assert 'desk_top_m' not in metadata and metadata['local_execution_checked'] is False
    ui.verify(pins)


def test_saved_perception_not_overridden_by_current_cli_default(tmp_path):
    saved = tmp_path/'saved.yaml'; saved.write_text('schema: saved\n')
    current = tmp_path/'current.yaml'; current.write_text('schema: current\n')
    preview = tmp_path/'preview.json'; preview.write_text(json.dumps(dict(config=str(saved))))
    crop = tmp_path/'summary.json'; crop.write_text(json.dumps(dict(source_preview=str(preview))))
    metadata, pins = selection.deferred_metadata(SimpleNamespace(config=current), {},
        inference=dict(source_point_cloud=str(tmp_path/'object.npy')),
        source_pins=ui.pin([saved, preview, crop]))
    assert metadata['perception_config'] == str(saved)
    assert str(current) not in pins


def test_saved_filter_threshold_cannot_silently_replace_protocol(tmp_path):
    args, out, config, path, _ = fixture(tmp_path)
    args.scene_collision_mm = 2.
    with pytest.raises(ValueError, match='threshold differs'):
        list(selection.official_candidates(args, out, {}, config, {}, filter_path=path))
