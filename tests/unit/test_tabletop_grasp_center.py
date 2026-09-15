"""Offline grasp-center visualization; never connects to hardware."""
import json

import numpy as np
import pytest

from tools.preview_tabletop_grasp_center import main, plan_geometry, project_base_mm


def patch():
    return dict(center_base_mm=[-200., -600., 80.], short_mm=89.,
                height_median_mm=40., short_axis_yaw_deg=89., yaw_ambiguous=False)


@pytest.mark.parametrize('bad', ['depth', 'width', 'center', 'yaw'])
def test_invalid_geometry_cannot_create_plan(bad):
    pytest.importorskip('mujoco')
    p = patch()
    if bad == 'width': p['short_mm'] = -1
    if bad == 'center': p['center_base_mm'][0] = float('nan')
    if bad == 'yaw': p['yaw_ambiguous'] = True
    with pytest.raises(ValueError):
        plan_geometry(p, 40 if bad == 'depth' else 10)


def test_actual_plan_center_and_axis_match_markers(tmp_path, monkeypatch):
    pytest.importorskip('mujoco')
    monkeypatch.setattr('rh56_controller.grasp_geometry._CACHE_PATH',str(tmp_path/'fk.npz'))
    from rh56_controller.grasp_viz_core import GraspVizCore
    monkeypatch.setattr(GraspVizCore, '_init_hand',lambda *a,**k: pytest.fail('hardware init'))
    result = plan_geometry(patch(),10.)
    center = np.array(result['grasp_center_base_mm'])
    thumb = np.array(result['thumb_tip_base_mm'])
    opposite = np.array(result['opposing_finger_centroid_base_mm'])
    np.testing.assert_allclose(center,[-200,-600,70])
    np.testing.assert_allclose((thumb+opposite)/2,center,atol=1e-8)
    d = (thumb-opposite)[:2]
    assert np.rad2deg(np.arctan2(d[1],d[0])) == pytest.approx(89)
    assert result['all_waypoint_centers_verified']
    assert result['waypoint_count'] > 1
    assert result['input']['width_offset_mm'] == 0
    assert result['solver_width_residual_mm'] == pytest.approx(result['achieved_solver_width_mm']-89)


def test_projection_uses_base_to_camera_inverse_and_mm_units():
    pytest.importorskip('cv2')
    T = np.eye(4); T[0,3] = 1.
    calibration = dict(transforms=dict(base_from_camera=dict(matrix=T.tolist())),
                       selected_intrinsics=dict(camera_matrix=[[100,0,320],[0,100,240],[0,0,1]],
                                                distortion_coefficients=[0]*5))
    uv = project_base_mm([[1000,0,1000],[1100,200,1000]],calibration)
    np.testing.assert_allclose(uv,[[320,240],[330,260]])
    with pytest.raises(ValueError,match='behind'):
        project_base_mm([[0,0,-100]],calibration)


def test_missing_input_writes_failure_without_executable_output(tmp_path):
    pytest.importorskip('cv2')
    out = tmp_path/'out'
    assert main(['--preview',str(tmp_path/'missing.json'),'--component-id','7','--out',str(out)]) == 1
    report = json.loads((out/'summary.json').read_text())
    assert report['status'] == 'failed'
    assert report['executable_target'] is False and report['hardware_contacted'] is False


def test_weak_surface_is_not_promoted_to_plan(tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    from tools import preview_tabletop_grasp_center as tool
    monkeypatch.setattr(tool,'plan_geometry',lambda *a: pytest.fail('weak evidence used'))
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(schema='rh56_tabletop_foreground_preview/v2',
        executable_target=False,candidates=[dict(component_id=7,top_patch=patch(),
        rejection_reasons=['insufficient_background_change_support'])])))
    assert main(['--preview',str(source),'--component-id','7','--out',str(tmp_path/'out')]) == 1


def test_no_execute_option():
    with pytest.raises(SystemExit):
        main(['--preview','missing','--component-id','7','--out','unused','--execute'])
