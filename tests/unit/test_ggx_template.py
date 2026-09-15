"""Ten-start geometry and authorization regression; no local logs or hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('mink')
from tools import ggx_template as template
from tools import run_ggx_ten_starts as cli
from tools import run_ggx_real as real


def test_exact_historical_order_labels_and_offsets():
    points = template.paper_points()
    assert [p['id'] for p in points] == [f'P{i}' for i in range(1, 11)]
    assert [p['point_id'] for p in points] == ['P1', 'P2', 'L1_d-150', 'L1_d-50',
        'L1_d+50', 'L1_d+150', 'L2_d-150', 'L2_d-50', 'L2_d+50', 'L2_d+150']
    assert np.array_equal([p['offset_mm'] for p in points],
        [[0,-250,0], [0,0,250], [-150,-250,100], [-50,-250,100],
         [50,-250,100], [150,-250,100], [-150,-250,250], [-50,-250,250],
         [50,-250,250], [150,-250,250]])


def test_current_default_only_translates_whole_set_to_audited_tcp_region():
    data = {'open_grasp_center_in_tcp_m': [.01,-.08,.13]}
    rows = [dict(phase='approach', tcp=[-.2,-.6,.3, 1.1,-1.7,1.2])]
    points = template.layout(data, {}, rows, template.default_origin(data, rows))
    assert np.allclose(points[0]['tcp'][:3], [-.14,-.455,.345])
    assert np.allclose(points[1]['tcp'][:3], [-.14,-.205,.595])


@pytest.mark.parametrize('origin', [[0,0,0], [-210,-400,225], [110,-650,340]])
def test_layout_translation_preserves_vectors_and_center_to_tcp(origin):
    data = {'open_grasp_center_in_tcp_m': [.01,-.08,.13]}
    rows = [dict(phase='approach', tcp=[-.2,-.6,.3, 1.1,-1.7,1.2])]
    points = template.layout(data, {}, rows, origin)
    centers = np.array([p['center_base_mm'] for p in points])
    offsets = np.array([p['offset_mm'] for p in template.paper_points()])
    assert np.allclose(centers, np.array(origin) + offsets)
    assert np.allclose(centers[:,None]-centers[None,:], offsets[:,None]-offsets[None,:])
    for point in points:
        T = real.support.pose_matrix(point['tcp'])
        center = (T @ np.r_[data['open_grasp_center_in_tcp_m'],1])[:3]*1000
        assert np.allclose(center, point['center_base_mm'])
        assert point['tcp'][3:] == rows[0]['tcp'][3:]


@pytest.mark.parametrize('bad', [[1,2], [1,2,3,4], [float('nan'),0,0], [0,float('inf'),0]])
def test_invalid_origin_rejected(bad):
    with pytest.raises(ValueError):
        template.layout({'open_grasp_center_in_tcp_m': [0,0,0]}, {},
                        [dict(phase='approach',tcp=[0.]*6)], bad)


def test_object_relocation_keeps_dimensions_and_rigid_relative_pose(tmp_path):
    plane = tmp_path/'plane.yaml'
    plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    old_T = template.object_matrix([-.2,-.6,.14], 15.)
    source = dict(preflight=dict(object_center_m=[-.2,-.6,.14], object_yaw_deg=15.,
        object_size_mm=[90,40,200], arguments={}), perception=dict(desk_plane=str(plane)))
    data = dict(reference_object_T=old_T.tolist())
    new, delta = template.relocate(data, source, {}, xy_mm=[-170,-520], yaw_deg=75.)
    new_T = template.object_matrix(new['preflight']['object_center_m'], 75.)
    grasp = real.support.pose_matrix([-.1,-.4,.2,1.,2.,-1.])
    assert np.allclose(np.linalg.inv(new_T) @ delta @ grasp, np.linalg.inv(old_T) @ grasp)
    assert source['preflight']['object_center_m'] == [-.2,-.6,.14]
    assert new['preflight']['object_size_mm'] == [90,40,200]
    assert new['preflight']['table_height_m'] == pytest.approx(.04)
    assert new['source_kind'] == 'grasp_template'
    assert not new['template_relocation']['old_approval_is_motion_authorization']


def test_same_object_pose_is_identity(tmp_path):
    source = dict(preflight=dict(object_center_m=[-.2,-.6,.14], object_yaw_deg=-12.,
        object_size_mm=[90,40,200], arguments={}))
    T = template.object_matrix(source['preflight']['object_center_m'], -12.)
    _, delta = template.relocate(dict(reference_object_T=T.tolist()), source, {})
    assert np.allclose(delta, np.eye(4), atol=1e-12)


def test_template_cannot_substitute_unbound_reference_prepared(tmp_path, monkeypatch):
    path = tmp_path/'template.json'
    template.write_json(path, dict(schema=template.SCHEMA, executable=False, motion_authorized=False,
        source_sha256={}, approval='approval.json', reference_prepared='substituted.json'))
    monkeypatch.setattr(template.support, 'read_approval', lambda *a: pytest.fail('Validate binding first'))
    with pytest.raises(ValueError, match='does not bind'):
        template.load(path)


def test_cli_default_is_offline_all_points():
    args = cli.parse_args(['--approval', 'approval.json'])
    assert args.point == 'all'
    assert not args.execute and not args.reset_first and not args.static_scene


@pytest.mark.parametrize('flags', [
    ['--execute'], ['--execute','--static-scene'], ['--reset-first'], ['--static-scene'],
])
def test_cli_blocks_ambiguous_execution(flags):
    with pytest.raises(SystemExit):
        cli.parse_args(['--template', 'template.json', *flags])


def test_selected_trial_has_explicit_execute_and_scene_flags():
    args = cli.parse_args(['--template','template.json','--execute','--static-scene','--point','P2'])
    assert args.execute and args.point == 'P2'


@pytest.mark.parametrize('execute,checked,static', [(True,False,True), (True,True,False), (False,True,True)])
def test_template_cannot_bypass_current_video_and_scene_policy(tmp_path, execute, checked, static):
    args = SimpleNamespace(execute=execute, preflight_only=not execute, checked_review=checked, static_scene=static)
    with pytest.raises(ValueError, match='Template execution requires'):
        real.run(args, {}, dict(source_kind='grasp_template'), tmp_path, {}, lambda: None,
                 backend_factory=lambda *a: pytest.fail('Must fail before connecting any hardware'))
