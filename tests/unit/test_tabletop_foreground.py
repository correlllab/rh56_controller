"""Synthetic geometry tests; no robot/camera or lab artifacts."""
import json

import numpy as np
import pytest

pytest.importorskip('cv2')
from tools.preview_tabletop_target import (components, main, background_foreground,
                                          compatible_background, top_patch_estimate,
                                          load_perception_config)


def rectangle(x0=30, y0=30, height=40):
    y,x = np.mgrid[y0:y0+40, x0:x0+60]
    pixels = np.column_stack((x.ravel(), y.ravel()))
    xyz = np.column_stack((pixels, np.full(len(pixels),34+height), np.full(len(pixels),height)))
    return xyz, pixels


def analyze(xyz, pixels, **kwargs):
    return components(xyz,pixels,(160,240),(10,10,230,150),[[0,0],[240,160]],**kwargs)[0]


def test_visible_box_measurement_without_color_or_known_size():
    rows = analyze(*rectangle())
    assert len(rows)==1 and rows[0]['geometry_candidate']
    assert rows[0]['visible_short_mm']==pytest.approx(39)
    assert rows[0]['visible_long_mm']==pytest.approx(59)
    assert rows[0]['height_p95_mm']==40


def test_two_objects_not_silently_selected():
    a,p = rectangle(); b,q = rectangle(140,80)
    rows = analyze(np.vstack((a,b)),np.vstack((p,q)))
    assert len(rows)==2 and all(r['geometry_candidate'] for r in rows)


def test_explicit_fixture_exclusion():
    a,p = rectangle(); b,q = rectangle(140,80)
    rows = analyze(np.vstack((a,b)),np.vstack((p,q)),exclusions=[(130,70,210,130)])
    assert len(rows)==1


def test_partial_exclusion_is_not_a_complete_object():
    rows = analyze(*rectangle(),exclusions=[(20,20,50,100)])
    assert not rows[0]['geometry_candidate']
    assert 'touches_roi_or_exclusion_boundary' in rows[0]['rejection_reasons']


def test_tall_occluder_rejected_and_thin_object_missed():
    rows = analyze(*rectangle(height=220))
    assert 'foreground_too_tall_or_occluder' in rows[0]['rejection_reasons']
    assert not analyze(*rectangle(height=4))


def test_missing_capture_never_outputs_executable_target(tmp_path):
    out = tmp_path/'preview'
    assert main(['--capture',str(tmp_path/'missing'),'--config',str(tmp_path/'missing.yaml'),
                 '--out',str(out)]) == 1
    report=json.loads((out/'summary.json').read_text())
    assert report['status']=='failed' and report['executable_target'] is False
    assert report['hardware_commands_sent'] is False


def test_background_difference_sign_holes_and_measured_noise():
    bg=np.full((2,3),600,dtype=np.uint16)
    depth=np.array([[560,600,620],[0,560,590]],dtype=np.uint16)
    bg[1,1]=0
    mad=np.ones((2,3)); mad[1,2]=4
    changed,known=background_foreground(depth,bg,.001,mad)
    np.testing.assert_array_equal(changed,[[True,False,False],[False,False,False]])
    assert not known[1,1]


def test_identical_empty_background_has_no_target():
    bg=np.full((160,240),600,dtype=np.uint16)
    gate,_=background_foreground(bg,bg,.001)
    rows=analyze(*rectangle(),foreground_gate=gate)
    assert not rows


def test_background_gate_removes_unchanged_fixture_without_color():
    a,p=rectangle(); b,q=rectangle(140,80)
    gate=np.zeros((160,240),dtype=bool); gate[30:70,30:90]=True
    rows=analyze(np.vstack((a,b)),np.vstack((p,q)),foreground_gate=gate)
    assert len(rows)==1 and rows[0]['bbox_xywh']==[30,30,60,40]


def test_top_patch_rejects_low_depth_edge_tail_without_known_width():
    core,_=rectangle(height=25)
    tail,_=rectangle(70,30,height=10)
    patch=top_patch_estimate(np.vstack((core,tail)),core)
    assert patch['short_mm']==pytest.approx(39)
    assert patch['long_mm']==pytest.approx(59)
    assert patch['height_median_mm']==25
    assert patch['points']==len(core)
    assert top_patch_estimate(core,core[:10]) is None


@pytest.mark.parametrize('key',['serial','stream','factory_color_intrinsics','depth'])
def test_changed_background_camera_settings_rejected(key):
    import copy
    current={'camera':dict(serial='serial',stream={'width':1280},factory_color_intrinsics={},depth={'scale_m_per_unit':.001})}
    background=copy.deepcopy(current)
    compatible_background(current,background)
    background['camera'][key]='changed'
    with pytest.raises(ValueError,match='mismatch'):
        compatible_background(current,background)


@pytest.mark.parametrize('bad', ['shape','dtype','scale','mad'])
def test_invalid_background_depth_rejected(bad):
    bg=np.full((2,2),600,dtype=np.uint16); depth=bg.copy(); mad=np.zeros((2,2)); scale=.001
    if bad=='shape': bg=bg[:1]
    if bad=='dtype': depth=depth.astype(float)
    if bad=='scale': scale=float('nan')
    if bad=='mad': mad[0,0]=-1
    with pytest.raises(ValueError):
        background_foreground(depth,bg,scale,mad)


def recovery_masks():
    known = np.ones((160,240), dtype=bool)
    changed = np.zeros_like(known)
    changed[30:70,30:90] = True
    # Missing background strip bisects the object, but current depth is valid.
    known[30:70,50:65] = False
    changed &= known
    return dict(foreground_gate=changed, known_background=known,
                background_mode='supported-surface')


def test_supported_surface_recovers_unknown_background_without_known_size():
    kwargs = recovery_masks()
    legacy = analyze(*rectangle(), foreground_gate=kwargs['foreground_gate'])
    assert len(legacy) == 2
    rows = analyze(*rectangle(), **kwargs)
    assert len(rows) == 1 and rows[0]['geometry_candidate']
    assert rows[0]['top_patch']['long_mm'] == pytest.approx(59)
    assert rows[0]['background_support']['recovered_unknown_background_pixels'] == 600
    assert rows[0]['background_support']['changed_fraction'] == pytest.approx(.75)


def test_recovery_does_not_fill_missing_current_depth():
    xyz, pixels = rectangle()
    keep = (pixels[:,0] < 50) | (pixels[:,0] >= 65)
    rows = analyze(xyz[keep], pixels[keep], **recovery_masks())
    assert len(rows) == 2
    assert sum(r['pixels'] for r in rows) == keep.sum()
    assert all(r['background_support']['recovered_unknown_background_pixels'] == 0 for r in rows)


@pytest.mark.parametrize('seeds', [0, 100, 200])
def test_unknown_region_needs_substantial_background_change(seeds):
    kwargs = recovery_masks()
    kwargs['known_background'][:] = False
    kwargs['foreground_gate'][:] = False
    xyz, pixels = rectangle()
    p = pixels[:seeds]
    kwargs['known_background'][p[:,1],p[:,0]] = True
    kwargs['foreground_gate'][p[:,1],p[:,0]] = True
    rows = analyze(xyz, pixels, **kwargs)
    assert len(rows) == 1
    assert not rows[0]['geometry_candidate']
    assert 'insufficient_background_change_support' in rows[0]['rejection_reasons']


def test_recovery_keeps_known_unchanged_fixture_excluded():
    a,p = rectangle(); b,q = rectangle(140,80)
    rows = analyze(np.vstack((a,b)), np.vstack((p,q)), **recovery_masks())
    assert len(rows) == 1 and rows[0]['bbox_xywh'] == [30,30,60,40]


def test_recovery_never_crosses_known_unchanged_strip():
    kwargs = recovery_masks()
    kwargs['known_background'][30:70,55:60] = True
    rows = analyze(*rectangle(), **kwargs)
    assert len(rows) == 2 and all(r['geometry_candidate'] for r in rows)


def test_recovery_keeps_two_objects_separate():
    a,p = rectangle(); b,q = rectangle(140,80)
    kwargs = recovery_masks()
    kwargs['foreground_gate'][80:120,140:200] = True
    rows = analyze(np.vstack((a,b)), np.vstack((p,q)), **kwargs)
    assert len(rows) == 2 and all(r['geometry_candidate'] for r in rows)


def test_recovery_preserves_exclusion_and_workspace_checks():
    rows = analyze(*rectangle(), exclusions=[(20,20,40,100)], **recovery_masks())
    assert 'touches_roi_or_exclusion_boundary' in rows[0]['rejection_reasons']
    xyz, pixels = rectangle()
    rows, _ = components(xyz, pixels, (160,240), (10,10,230,150), [[40,0],[240,160]],
                         **recovery_masks())
    assert 'outside_target_xy_workspace' in rows[0]['rejection_reasons']


@pytest.mark.parametrize('bad', ['missing', 'shape', 'seed_outside_known', 'fraction', 'mode'])
def test_recovery_rejects_invalid_inputs(bad):
    kwargs = recovery_masks()
    if bad == 'missing': kwargs['known_background'] = None
    if bad == 'shape': kwargs['known_background'] = np.ones((2,2), bool)
    if bad == 'seed_outside_known': kwargs['foreground_gate'][35,55] = True
    if bad == 'fraction': kwargs['min_background_support_fraction'] = float('nan')
    if bad == 'mode': kwargs['background_mode'] = 'fill-everything'
    with pytest.raises(ValueError):
        analyze(*rectangle(), **kwargs)


def vision_profile():
    return dict(schema='rh56_tabletop_vision_config/v1', camera_serial='test',
                calibration='calibration.yaml', desk_plane='plane.yaml',
                desk_z_mm=34, roi=[320,60,1100,620])


def test_perception_profile_does_not_import_robot_workspace(tmp_path):
    import yaml
    path = tmp_path/'vision.yaml'
    path.write_text(yaml.safe_dump(vision_profile()))
    c, bounds = load_perception_config(path)
    assert bounds is None
    assert c['roi'] == [320,60,1100,620]
    assert 'tcp_bounds_mm' not in c and 'target_bounds_mm' not in c
    # The hardware cycle must reject this profile, not inherit permissive bounds.
    from tools.run_vision_pick_cycle import load_config
    with pytest.raises(ValueError,match='rh56_vision_pick_cycle_config'):
        load_config(path)


@pytest.mark.parametrize('key,value', [
    ('roi', [0,0,1281,720]), ('roi', [10,10,5,20]), ('roi', [0,0,100.5,100]),
    ('roi', [1,2,3]), ('roi', [0,0,float('nan'),100]),
    ('desk_z_mm', float('nan')), ('camera_serial', ''),
    ('tcp_bounds_mm', [[-999]*3,[999]*3]), ('target_bounds_mm', None),
])
def test_invalid_or_mixed_perception_profile_rejected(tmp_path,key,value):
    import yaml
    c = vision_profile(); c[key] = value
    path = tmp_path/'vision.yaml'; path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        load_perception_config(path)


def test_legacy_config_explicitly_retains_target_gate(tmp_path, monkeypatch):
    import yaml
    path = tmp_path/'legacy.yaml'
    path.write_text(yaml.safe_dump(dict(schema='rh56_vision_pick_cycle_config/v1')))
    legacy = dict(target_bounds_mm=[[-300,-750,60],[-100,-500,120]])
    monkeypatch.setattr('tools.run_vision_pick_cycle.load_config',lambda p: legacy)
    c,bounds = load_perception_config(path)
    assert c is legacy
    np.testing.assert_array_equal(bounds,[[-300,-750],[-100,-500]])


def test_full_object_visible_outside_legacy_bounds_is_perception_candidate():
    xyz,pixels = rectangle()
    xyz[:,0] -= 345  # Visible object's base X crosses the old -300 mm edge.
    args = (xyz,pixels,(160,240),(10,10,230,150))
    old,_ = components(*args,[[-300,-750],[-100,100]])
    new,_ = components(*args)
    assert 'outside_target_xy_workspace' in old[0]['rejection_reasons']
    assert new[0]['geometry_candidate']
    assert new[0]['top_patch'] == old[0]['top_patch']  # No refit/recentering.
    truncated,_ = components(*args, exclusions=[(20,20,40,100)])
    assert not truncated[0]['geometry_candidate']
    assert 'touches_roi_or_exclusion_boundary' in truncated[0]['rejection_reasons']
