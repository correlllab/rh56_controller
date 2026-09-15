"""Pure measured-point crop checks: no GPU, camera, ROS or robot required."""
import json
import numpy as np
import pytest
from tools import prepare_ggx_tabletop_cloud as cloud
from tools.prepare_ggx_tabletop_cloud import crop_mask


def test_crop_rejects_desk_and_depth_outliers_without_filling():
    points=np.array([[0,0,176],[0,0,100],[0,0,0],[0,0,322],[30,0,176],[0,60,176]])
    result=crop_mask(points,[0,0,176],0,[40,90,176],5)
    assert result.tolist()==[True,True,False,False,False,False]
    np.testing.assert_array_equal(points[result],points[:2])


def test_short_axis_yaw_changes_crop_axes():
    points=np.array([[35,0,176],[0,35,176]])
    assert crop_mask(points,[0,0,176],0,[40,90,176],0).tolist()==[False,True]
    assert crop_mask(points,[0,0,176],90,[40,90,176],0).tolist()==[True,False]


def test_crop_is_translation_equivariant():
    points=np.array([[0,0,170],[35,0,176]]);offset=np.array([-232,-630,40])
    np.testing.assert_array_equal(crop_mask(points,[0,0,176],88,[40,90,176]),
        crop_mask(points+offset,np.array([0,0,176])+offset,88,[40,90,176]))


@pytest.mark.parametrize('fault',['nan_points','nan_yaw','negative_dims','nan_dims','large_margin'])
def test_invalid_geometry_rejected(fault):
    points=np.zeros((3,3));yaw=0;dims=[40,90,176];margin=5
    if fault=='nan_points':points[0,0]=np.nan
    if fault=='nan_yaw':yaw=np.nan
    if fault=='negative_dims':dims[0]=-1
    if fault=='nan_dims':dims[0]=np.nan
    if fault=='large_margin':margin=100
    with pytest.raises(ValueError):crop_mask(points,[0,0,176],yaw,dims,margin)


def curved_points():
    y, x = np.mgrid[-20:21, -45:46]
    z = 214. - .018 * x**2 - .01 * y**2
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_foreground_preserves_curved_heights_and_removes_only_isolated_spike():
    pytest.importorskip('scipy')
    points = np.vstack((curved_points(), [0, 0, 280]))
    original = points.copy()
    keep, quality = cloud.supported_surface(points)
    np.testing.assert_array_equal(points, original)
    assert keep[:-1].all() and not keep[-1]
    assert quality['removed_points'] == 1
    assert np.ptp(points[keep, 2]) > 30  # Not replaced by one median/top plane.


def test_foreground_proxy_encloses_prior_and_all_supported_points():
    pytest.importorskip('cv2')
    points = curved_points() + [100, -600, 15]  # Depth exceeds known total height.
    frame, proxy, warnings = cloud.foreground_geometry(points, [35, 85, 180], [0, 0, 1], 34)
    assert warnings
    assert frame['origin'] == 'desk_footprint_center'
    assert frame['center_base_mm'][2] == 34
    assert proxy['operator_dimensions_mm'] == [35, 85, 180]
    size = np.array(proxy['dimensions_mm'])
    assert np.all(size >= [40, 90, 195])
    theta = np.deg2rad(frame['short_axis_yaw_deg'])
    d = points - frame['center_base_mm']
    assert np.max(np.abs(d @ [np.cos(theta), np.sin(theta), 0])) <= size[0]/2
    assert np.max(np.abs(d @ [-np.sin(theta), np.cos(theta), 0])) <= size[1]/2
    assert d[:, 2].max() <= size[2]


def test_foreground_frame_reuses_sloped_desk_origin_not_top_median():
    pytest.importorskip('cv2')
    points = curved_points() + [-200, -600, 0]
    frame, proxy, _ = cloud.foreground_geometry(points, [40, 90, 180], [.01, .02, 1], 34)
    assert frame['center_base_mm'] == pytest.approx([-200, -600, 48])
    assert proxy['dimensions_mm'][2] == 180


def test_foreground_rejects_ambiguous_yaw_and_sparse_noise():
    pytest.importorskip('cv2')
    pytest.importorskip('scipy')
    y,x = np.mgrid[-20:21,-20:21]
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 100)))
    with pytest.raises(ValueError, match='yaw ambiguous'):
        cloud.foreground_geometry(points, [40,40,66], [0,0,1], 34)
    with pytest.raises(ValueError, match='local depth support'):
        cloud.supported_surface(np.arange(300).reshape(100,3)*10.)


def test_foreground_selection_excludes_other_component_inside_same_bbox():
    cv2 = pytest.importorskip('cv2')
    mask = np.zeros((60,100), np.uint8)
    mask[10:50,10:90] = 255
    mask[20:40,30:70] = 0
    mask[25:35,40:60] = 255  # Separate object/fixture enclosed by the bbox.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    y,x = np.nonzero(mask)
    uv = np.column_stack((x,y))
    row = dict(component_id=1,bbox_xywh=stats[1,:4].tolist(),pixels=int(stats[1,4]))
    keep = cloud.selected_component_mask(mask,row,uv)
    assert keep.sum() == row['pixels']
    assert not keep[(labels[y,x] == 2)].any()
    row['pixels'] += 1
    with pytest.raises(ValueError, match='no longer matches'):
        cloud.selected_component_mask(mask,row,uv)


@pytest.fixture
def crop_scene(tmp_path, monkeypatch):
    cv2 = pytest.importorskip('cv2')
    pytest.importorskip('scipy')
    import yaml
    from tools import preview_tabletop_target as preview
    from tools import run_vision_pick_cycle as cycle
    from tools import analyze_ur5_camera_desk_plane as desk
    capture = tmp_path/'capture'; capture.mkdir()
    perception = tmp_path/'perception'; perception.mkdir()
    rgb = np.full((720,1280,3), 120, np.uint8)
    depth = np.full((720,1280), 500, np.uint16)
    mask = np.zeros((720,1280), np.uint8); mask[200:241,620:711] = 255
    cv2.imwrite(str(capture/'camera_check_rgb.png'),rgb)
    cv2.imwrite(str(capture/'camera_check_depth_u16.png'),depth)
    cv2.imwrite(str(perception/'foreground.png'),mask)
    cal = tmp_path/'cal.yaml'
    cal.write_text(yaml.safe_dump(dict(selected_intrinsics=dict(camera_matrix=[],distortion_coefficients=[]),
                                      transforms=dict(base_from_camera=dict(matrix=[])))))
    plane = tmp_path/'plane.yaml'; plane.write_text('plane:\n  normal_base: [0, 0, 1]\n')
    config = tmp_path/'config.yaml'; config.write_text('offline synthetic configuration')
    points = curved_points()
    y,x = np.nonzero(mask); pixels = np.column_stack((x,y))
    patch = dict(center_base_mm=[0,0,214],short_mm=40,long_mm=90,height_median_mm=158.7,
                 short_axis_yaw_deg=90,yaw_ambiguous=False)
    row = dict(component_id=1,pixels=len(points),bbox_xywh=[620,200,91,41],geometry_candidate=True,
               rejection_reasons=[],top_patch=patch)
    source = dict(schema='rh56_tabletop_foreground_preview/v2',executable_target=False,candidates=[row],
                  capture=str(capture),config=str(config),config_sha256=cloud.hashlib.sha256(config.read_bytes()).hexdigest())
    path = perception/'summary.json'; path.write_text(json.dumps(source))
    monkeypatch.setattr(preview,'load_perception_config', lambda _: (dict(calibration=str(cal),
        desk_plane=str(plane),desk_z_mm=34,roi=[320,60,1100,620]),None))
    monkeypatch.setattr(cycle,'validate_capture',lambda *a: dict(camera=dict(depth=dict(scale_m_per_unit=.001))))
    monkeypatch.setattr(desk,'deproject_aligned_depth',lambda *a: (points/1000,pixels))
    monkeypatch.setattr(desk,'transform_points',lambda matrix,points: points)
    return path, source, points, pixels


def test_foreground_cli_saved_cloud_is_exact_measured_subset(crop_scene, tmp_path):
    preview,source,points,pixels = crop_scene
    source['candidates'][0]['top_patch'] = None  # No flat top is required.
    preview.write_text(json.dumps(source))
    out = tmp_path/'crop'
    assert cloud.main(['--preview',str(preview),'--component-id','1','--dimensions-mm','40','90','180',
                       '--crop-mode','foreground','--out',str(out)]) == 0
    report = json.loads((out/'summary.json').read_text())
    assert report['schema'] == cloud.FOREGROUND_SCHEMA
    assert report['top_patch'] is None
    assert not report['hardware_contacted'] and not report['executable_target']
    assert report['dimensions_prior_mm'] == [40,90,180]
    np.testing.assert_array_equal(np.load(out/'object_points_ur_base_m.npy'),points/1000)
    np.testing.assert_array_equal(np.load(out/'object_pixels.npy'),pixels)
    assert (out/'summary.csv').is_file() and (out/'input_overlay.png').is_file()
    cloud.validate_foreground_report(report)
    report['collision_proxy']['dimensions_mm'][2] = 170
    with pytest.raises(ValueError, match='does not enclose'):
        cloud.validate_foreground_report(report)


def test_legacy_flat_top_still_rejects_nonflat_median_with_actionable_error(crop_scene,tmp_path):
    preview,*_ = crop_scene
    out = tmp_path/'flat'
    assert cloud.main(['--preview',str(preview),'--component-id','1','--dimensions-mm','40','90','180',
                       '--out',str(out)]) == 1
    report = json.loads((out/'summary.json').read_text())
    assert report['schema'] == 'rh56_ggx_measured_crop/v1'
    assert '--crop-mode foreground' in report['error']
    assert '158.7' in report['error'] and '180.0' in report['error']
    assert not (out/'object_points_ur_base_m.npy').exists()


def test_legacy_flat_top_success_preserves_previous_crop_formula(crop_scene,tmp_path):
    preview,source,points,pixels = crop_scene
    source['candidates'][0]['top_patch']['height_median_mm'] = 180
    preview.write_text(json.dumps(source))
    out = tmp_path/'legacy'
    assert cloud.main(['--preview',str(preview),'--component-id','1','--dimensions-mm','40','90','180',
                       '--out',str(out)]) == 0
    report = json.loads((out/'summary.json').read_text())
    expected = crop_mask(points,[0,0,214],90,[40,90,180],5)
    np.testing.assert_array_equal(np.load(out/'object_points_ur_base_m.npy'),points[expected]/1000)
    assert report['schema'] == 'rh56_ggx_measured_crop/v1'
    assert 'collision_proxy' not in report and 'object_frame' not in report


@pytest.mark.parametrize('fault',['rejected','missing_depth','missing_mask','changed_config','invalid_dimensions'])
def test_foreground_cli_invalid_scene_never_emits_cloud(crop_scene,tmp_path,fault):
    preview,source,points,pixels = crop_scene
    dims = ['40','90','180']
    if fault == 'rejected': source['candidates'][0]['rejection_reasons'] = ['touches_roi_or_exclusion_boundary']
    if fault == 'missing_depth':
        from tools import analyze_ur5_camera_desk_plane as desk
        # Test fixture replaces deprojection with a camera-free deterministic function.
        from unittest.mock import patch
        with patch.object(desk,'deproject_aligned_depth',return_value=(points[:-1]/1000,pixels[:-1])):
            out = tmp_path/'invalid'
            assert cloud.main(['--preview',str(preview),'--component-id','1','--dimensions-mm',*dims,
                               '--crop-mode','foreground','--out',str(out)]) == 1
        assert not (out/'object_points_ur_base_m.npy').exists()
        return
    if fault == 'missing_mask': (preview.parent/'foreground.png').unlink()
    if fault == 'changed_config':
        from pathlib import Path
        Path(source['config']).write_text('changed')
    if fault == 'invalid_dimensions': dims = ['90','40','180']
    preview.write_text(json.dumps(source))
    out = tmp_path/'invalid'
    assert cloud.main(['--preview',str(preview),'--component-id','1','--dimensions-mm',*dims,
                       '--crop-mode','foreground','--out',str(out)]) == 1
    assert not (out/'object_points_ur_base_m.npy').exists()
