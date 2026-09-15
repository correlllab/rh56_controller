"""Reference verification must tolerate depth fragmentation, not scene changes."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from tools import ggx_scene_check as check


@pytest.fixture
def target():
    rgb = np.full((180, 260, 3), 60, np.uint8)
    bbox = [45, 50, 170, 85]
    rgb[50:135, 45:215] = [210, 215, 220]
    cv2.putText(rgb, 'SUGAR', (65, 105), cv2.FONT_HERSHEY_SIMPLEX, .7, (10, 20, 50), 2)
    depth = np.full((180, 260), 600, np.uint16)
    depth[50:135, 45:215] = 420
    depth[55:130, 124:130] = 0
    config = dict(scene_center_tolerance_mm=3., scene_dimension_tolerance_mm=5.)
    return rgb, depth, bbox, config


def compare(target, rgb=None, depth=None):
    ref, d, box, config = target
    return check.compare_target_rgbd(ref, ref if rgb is None else rgb,
        d, d if depth is None else depth, box, .001, config)


def test_unchanged_target_with_disconnected_depth_is_accepted(target):
    rgb, depth, _, _ = target
    current = depth.copy()
    current[55:130, 130:133] = 0  # Widen a missing-depth seam; never fill it.
    assert compare(target, depth=current)['passed']
    assert np.all(current[55:130, 124:133] == 0)


def test_small_uniform_exposure_offset_is_not_geometric_alignment(target):
    rgb = target[0].copy()
    rgb += 5
    assert compare(target, rgb=rgb)['passed']


def test_pixel_noise_does_not_become_object_motion(target):
    rgb,depth,box,config = target
    noise = np.random.default_rng(4).normal(0,4,rgb.shape)
    current = np.clip(rgb.astype(float)+noise,0,255).astype(np.uint8)
    assert compare(target,rgb=current)['passed']


@pytest.mark.parametrize('shift',[3,4,8])
def test_denoising_and_weak_edges_do_not_hide_translation(target,shift):
    rgb = cv2.warpAffine(target[0],np.array([[1.,0.,shift],[0.,1.,0.]]),(260,180),
                         borderValue=(60,60,60))
    assert not compare(target,rgb=rgb)['passed']


def test_depth_scale_float_roundoff_is_not_a_physical_threshold_change(target):
    rgb,depth,box,config = target
    current = depth.copy();current[current>0] += 3
    scale = float(np.float32(.001))
    assert check.compare_target_rgbd(rgb,rgb,depth,current,box,scale,config)['passed']
    current[current>0] += 1
    assert not check.compare_target_rgbd(rgb,rgb,depth,current,box,scale,config)['passed']


def test_curved_minority_surface_not_discarded_by_percentile(target):
    rgb,depth,box,config = target
    ref = depth.copy()
    ref[61,61:66] = 400  # <1% of the cell, but already measured.
    ref[65:70,65:70] = 0
    current = ref.copy();current[65:70,65:70] = 400
    assert check.compare_target_rgbd(rgb,rgb,ref,current,box,.001,config)['passed']
    current[65:70,65:70] = 390  # A genuinely new nearer surface is still rejected.
    assert not check.compare_target_rgbd(rgb,rgb,ref,current,box,.001,config)['passed']


@pytest.mark.parametrize('fault', ['translate', 'rotate', 'occlude', 'disappear', 'lighting'])
def test_rgb_change_cannot_be_hidden_by_unchanged_depth(target, fault):
    rgb = target[0].copy()
    if fault == 'translate':
        rgb = cv2.warpAffine(rgb, np.array([[1., 0., 8.], [0., 1., 0.]]), (260, 180), borderValue=(60, 60, 60))
    if fault == 'rotate':
        rgb = cv2.warpAffine(rgb, cv2.getRotationMatrix2D((130, 92), 8, 1), (260, 180), borderValue=(60, 60, 60))
    if fault == 'occlude': rgb[65:120, 70:120] = 0
    if fault == 'disappear': rgb[:] = 60
    if fault == 'lighting': rgb = (rgb.astype(float)*.5).astype(np.uint8)
    result = compare(target, rgb=rgb)
    assert not result['passed']
    assert any('appearance' in r or 'texture' in r or 'visual_structure' in r for r in result['rejection_reasons'])


@pytest.mark.parametrize('fault', ['raise', 'remove', 'local_occluder', 'loss', 'new_surface_in_holes'])
def test_depth_change_cannot_be_hidden_by_identical_rgb(target, fault):
    depth = target[1].copy()
    if fault == 'raise':
        sub = depth[50:135, 45:215]
        sub[sub > 0] -= 4
    if fault == 'remove': depth[50:135, 45:215] = 600
    if fault == 'local_occluder': depth[80:100, 70:100] = 390
    if fault == 'loss': depth[55:90, 50:110] = 0
    if fault == 'new_surface_in_holes': depth[55:130, 124:130] = 390
    assert not compare(target, depth=depth)['passed']


def test_weak_visual_or_depth_evidence_fails_closed(target):
    rgb, depth, box, config = target
    plain = np.full_like(rgb, 100)
    assert not check.compare_target_rgbd(plain, plain, depth, depth, box, .001, config)['passed']
    holes = np.zeros_like(depth)
    assert not compare(target, depth=holes)['passed']


def test_only_one_depth_patch_is_not_enough_spatial_support(target):
    rgb, depth, box, config = target
    sparse = np.zeros_like(depth)
    sparse[60:80, 60:80] = 420
    result = check.compare_target_rgbd(rgb, rgb, sparse, sparse, box, .001, config)
    assert not result['passed']
    assert 'insufficient_spatial_depth_support' in result['rejection_reasons']


@pytest.mark.parametrize('fault', ['rgb_shape', 'depth_dtype', 'bbox', 'scale'])
def test_invalid_inputs_fail_closed(target, fault):
    rgb, depth, box, config = target
    current = rgb
    scale = .001
    if fault == 'rgb_shape': current = rgb[:2]
    if fault == 'depth_dtype': depth = depth.astype(float)
    if fault == 'bbox': box = [0, 0, 20, 20]
    if fault == 'scale': scale = float('nan')
    with pytest.raises(ValueError):
        check.compare_target_rgbd(rgb, current, depth, depth, box, scale, config)


def test_saved_failure_writes_diagnostic_report_without_hardware(tmp_path):
    source = dict(pins={}, perception=dict(candidates=[]), crop=dict(reviewed_component_id=1))
    with pytest.raises(ValueError, match='reference component'):
        check.verify_saved_target(source, tmp_path, {}, tmp_path/'check')
    report = json.loads((tmp_path/'check.json').read_text())
    assert not report['passed'] and not report['hardware_contacted']
    assert (tmp_path/'check.csv').is_file()


def test_reference_mode_does_not_run_component_segmentation(monkeypatch, tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('mink')
    from tools import run_ggx_real as real
    from tools import preview_tabletop_target as preview
    stages = []
    monkeypatch.setattr(preview, 'load_perception_config', lambda p: (dict(camera_serial='fake', roi=[0,0,1280,720]), None))
    monkeypatch.setattr(real, 'run_stage', lambda out, name, script, command: stages.append(name))
    monkeypatch.setattr(check, 'verify_saved_target', lambda *a, **kw: dict(passed=True))
    result = real.capture_and_check(dict(perception=dict(config='fake')), dict(scene_check_mode='reference-rgbd'), tmp_path, 'final')
    assert stages == ['final_capture']
    assert result['passed'] and 'acquired_monotonic' in result


def test_legacy_component_mode_remains_available(monkeypatch, tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('mink')
    from tools import run_ggx_real as real
    from tools import preview_tabletop_target as preview
    stages = []
    monkeypatch.setattr(preview, 'load_perception_config', lambda p: (dict(camera_serial='fake', roi=[0,0,1280,720]), None))
    monkeypatch.setattr(real, 'run_stage', lambda out, name, script, command: stages.append(name))
    monkeypatch.setattr(real, 'read_json', lambda path: {})
    monkeypatch.setattr(real, 'scene_match', lambda *a: dict(component_id=1))
    monkeypatch.setattr(check, 'verify_saved_target', lambda *a: pytest.fail('Legacy mode must not change'))
    real.capture_and_check(dict(perception=dict(config='fake')), {}, tmp_path, 'final')
    assert stages == ['final_capture', 'final_perception']


@pytest.mark.parametrize('with_burst', [False, True])
def test_nonflat_saved_reference_needs_no_top_patch_but_still_rejects_motion(target,tmp_path,monkeypatch,with_burst):
    from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA
    from tools import preview_tabletop_target as preview
    from tools import run_vision_pick_cycle as cycle
    rgb,depth,bbox,config = target
    before = tmp_path/'before'; before.mkdir()
    after = tmp_path/'after'; after.mkdir()
    # Multiple surface heights are legitimate; no equality to total height.
    curved = depth.copy()
    for i in range(bbox[3]):
        line = curved[bbox[1]+i,bbox[0]:bbox[0]+bbox[2]]
        line[line > 0] += i // 3
    for directory in (before,after):
        cv2.imwrite(str(directory/'camera_check_rgb.png'),rgb)
        cv2.imwrite(str(directory/'camera_check_depth_u16.png'),curved)
    crop = dict(schema=FOREGROUND_SCHEMA,crop_mode='foreground',
        status='measured_crop_ready_for_offline_inference',hardware_contacted=False,executable_target=False,
        reviewed_component_id=9,dimensions_prior_mm=[40,90,180],top_patch=None,
        object_frame=dict(center_base_mm=[-200,-600,34],origin='desk_footprint_center',
                          yaw_ambiguous=False,short_axis_yaw_deg=85),
        collision_proxy=dict(shape='box',dimensions_mm=[54,95,196],observed_bounds_mm=[53.9,94.7,195.1]))
    source = dict(pins={},crop=crop,perception=dict(config='fake',capture=str(before),
        candidates=[dict(component_id=9,geometry_candidate=True,rejection_reasons=[],bbox_xywh=bbox,top_patch=None)]))
    camera = dict(serial='fake',stream={},factory_color_intrinsics={},depth=dict(scale_m_per_unit=.001))
    ref_meta = write_reference_burst(before,curved,camera=camera) if with_burst else dict(camera=camera)
    monkeypatch.setattr(preview,'load_perception_config',lambda *a: ({},None))
    monkeypatch.setattr(cycle,'validate_capture',lambda c,p: ref_meta if p == before else dict(camera=camera))
    result = check.verify_saved_target(source,after,config,tmp_path/'unchanged')
    assert result['passed']
    saved = json.loads((tmp_path/'unchanged.json').read_text())
    if with_burst:
        assert len(saved['reference_burst_sha256']) == 6
        assert saved['reference_burst_sha256'].items() <= source['pins'].items()
        assert saved['reference_quality']['method'] == 'reference_burst_half_error_budget/v1'
    current = curved.copy(); current[current > 0] -= 10
    cv2.imwrite(str(after/'camera_check_depth_u16.png'),current)
    with pytest.raises(RuntimeError,match='target_depth_changed'):
        check.verify_saved_target(source,after,config,tmp_path/'moved')


def noisy_reference_burst(depth):
    burst = np.repeat(depth[None], 9, axis=0)
    # First cell is already unreliable during REFERENCE capture. It is never
    # selected using residuals/validity in the current image.
    for frame, offset in zip(burst, [-8, -6, -4, -2, 0, 2, 4, 6, 8]):
        frame[55:80, 50:103] = 420 + offset
    return burst


def test_reference_unstable_patch_is_unknown_but_stable_loss_still_rejects(target):
    rgb, depth, box, config = target
    burst = noisy_reference_burst(depth)
    current = depth.copy(); current[55:80, 50:103] = 0
    assert not compare(target, depth=current)['passed']  # Strict legacy behavior.
    result = check.compare_target_rgbd(rgb, rgb, depth, current, box, .001, config,
                                       reference_depth_burst=burst)
    assert result['passed']
    assert result['depth_cells'][0]['status'] == 'unstable_in_reference'
    assert result['depth_cells'][0]['stable_reference_points'] == 0
    assert np.all(current[55:80, 50:103] == 0)  # No hole filling.
    current[80:105, 50:103] = 0  # Previously reliable region must not be ignored.
    changed = check.compare_target_rgbd(rgb, rgb, depth, current, box, .001, config,
                                        reference_depth_burst=burst)
    assert not changed['passed'] and 'localized_depth_loss' in changed['rejection_reasons']
    assert changed['reference_quality'] == result['reference_quality']


@pytest.mark.parametrize('fault', ['shift', 'rotate', 'occlude', 'raise', 'remove',
    'local_occluder', 'total_loss', 'new_surface_in_holes', 'new_surface_in_unstable_region'])
def test_reference_quality_does_not_hide_scene_changes(target, fault):
    rgb, depth, box, config = target
    current_rgb, current = rgb.copy(), depth.copy()
    if fault == 'shift':
        current_rgb = cv2.warpAffine(rgb, np.array([[1.,0.,3.],[0.,1.,0.]]), (260,180))
    if fault == 'rotate':
        current_rgb = cv2.warpAffine(rgb, cv2.getRotationMatrix2D((130,92),5,1), (260,180))
    if fault == 'occlude': current_rgb[65:120,70:120] = 0
    if fault == 'raise': current[current > 0] -= 4
    if fault == 'remove': current[50:135,45:215] = 600
    if fault == 'local_occluder': current[80:100,70:100] = 390
    if fault == 'total_loss': current[:] = 0
    if fault == 'new_surface_in_holes': current[55:130,124:130] = 390
    if fault == 'new_surface_in_unstable_region': current[55:80,50:103] = 390
    result = check.compare_target_rgbd(rgb,current_rgb,depth,current,box,.001,config,
                                       reference_depth_burst=noisy_reference_burst(depth))
    assert not result['passed']
    if fault == 'new_surface_in_unstable_region':
        assert 'unstable_region_new_depth_surface' in result['rejection_reasons']


@pytest.mark.parametrize('distributed', [False, True])
def test_inadequate_reference_quality_fails_even_for_identical_current(target, distributed):
    rgb, depth, box, config = target
    burst = np.repeat(depth[None], 9, axis=0)
    for frame, offset in zip(burst, [-8,-6,-4,-2,0,2,4,6,8]):
        frame[frame > 0] = (frame[frame > 0].astype(int) + offset).astype(np.uint16)
    if distributed:
        # Numerous reliable pixels, but all in the bottom two rows.
        burst[:,80:130] = depth[80:130]
    result = check.compare_target_rgbd(rgb,rgb,depth,depth,box,.001,config,
                                       reference_depth_burst=burst)
    assert not result['passed']
    assert 'insufficient_spatial_depth_support' in result['rejection_reasons']
    if not distributed:
        assert 'insufficient_stable_reference_depth' in result['rejection_reasons']


@pytest.mark.parametrize('fault', ['mismatched_median', 'dtype', 'shape', 'short', 'tolerance'])
def test_bad_reference_burst_is_not_a_legacy_fallback(target, fault):
    rgb, depth, box, config = target
    burst = np.repeat(depth[None], 9, axis=0)
    if fault == 'mismatched_median': burst[:,60,60] += 1
    if fault == 'dtype': burst = burst.astype(float)
    if fault == 'shape': burst = burst[:,:-1,:]
    if fault == 'short': burst = burst[:2]
    if fault == 'tolerance': config = dict(config, scene_center_tolerance_mm=float('nan'))
    with pytest.raises(ValueError):
        check.compare_target_rgbd(rgb,rgb,depth,depth,box,.001,config,reference_depth_burst=burst)


def write_reference_burst(directory, depth, *, camera=None):
    import yaml
    meta = dict(schema='rh56_realsense_camera_check/v1', camera=camera or dict(serial='test'),
        depth_aggregation='temporal median; RGB is the middle frame', source_frame_numbers=[10,20,30])
    for i, number in enumerate(meta['source_frame_numbers']):
        folder = directory/f'capture_{i:02d}'; folder.mkdir()
        (folder/'camera_check.yaml').write_text(yaml.safe_dump(dict(
            schema=meta['schema'], camera=meta['camera'], frame_number=number)))
        cv2.imwrite(str(folder/'camera_check_depth_u16.png'), depth)
    (directory/'camera_check.yaml').write_text(yaml.safe_dump(meta))
    return meta


def test_reference_burst_is_pinned_and_cannot_be_replaced_between_checks(target, tmp_path):
    from tools.run_ggx_review import pin
    depth = target[1]
    meta = write_reference_burst(tmp_path, depth)
    pins = pin([tmp_path/'camera_check.yaml'])
    burst, raw_pins = check.load_reference_burst(tmp_path,meta,pins,depth.shape)
    assert len(raw_pins) == 6 and raw_pins.items() <= pins.items()
    assert np.array_equal(burst,np.repeat(depth[None],3,axis=0))
    path = tmp_path/'capture_00/camera_check_depth_u16.png'
    changed = depth.copy(); changed[60,60] += 1
    cv2.imwrite(str(path),changed)
    with pytest.raises(ValueError,match='artifact changed'):
        check.load_reference_burst(tmp_path,meta,pins,depth.shape)
    assert pins[str(path)] == raw_pins[str(path)]


@pytest.mark.parametrize('fault', ['missing', 'frame_number', 'camera', 'duplicate', 'aggregation', 'no_numbers'])
def test_declared_reference_burst_missing_or_incompatible_fails(target,tmp_path,fault):
    import yaml
    depth = target[1]
    meta = write_reference_burst(tmp_path, depth)
    if fault == 'missing': (tmp_path/'capture_00/camera_check_depth_u16.png').unlink()
    if fault in ('frame_number','camera'):
        path = tmp_path/'capture_00/camera_check.yaml'
        frame = yaml.safe_load(path.read_text()); frame[fault] = 'changed'
        path.write_text(yaml.safe_dump(frame))
    if fault == 'duplicate': meta['source_frame_numbers'] = [10,10,30]
    if fault == 'aggregation': meta['depth_aggregation'] = 'unknown'
    if fault == 'no_numbers': del meta['source_frame_numbers']
    with pytest.raises((ValueError,FileNotFoundError)):
        check.load_reference_burst(tmp_path,meta,{},depth.shape)


def test_legacy_single_reference_never_creates_a_quality_mask(tmp_path):
    assert check.load_reference_burst(tmp_path,dict(camera={}),{},(180,260)) == (None,{})


@pytest.fixture
def warning_target(target):
    rgb, depth, box, config = target
    depth = depth.copy()
    burst = noisy_reference_burst(depth)
    depth[60:65,55:65] = 0
    burst[:,60:65,55:65] = 0
    current = depth.copy(); current[60:65,55:65] = 400
    return rgb, depth, current, box, config, burst


def test_unstable_reference_holes_warn_but_do_not_automatically_pass(warning_target):
    rgb, depth, current, box, config, burst = warning_target
    result = check.compare_target_rgbd(rgb,rgb,depth,current,box,.001,config,reference_depth_burst=burst)
    assert result['numerical_checks_passed'] and not result['rejection_reasons']
    assert result['visual_confirmation_required']
    assert not result['passed'] and not result['automatic_passed']
    assert result['warnings'] == [dict(code='new_depth_in_unstable_reference_holes',row=0,column=0,
        bbox_xywh=[50,55,53,25],new_depth_points=50,outlier_points=50,outlier_fraction=1.)]
    # Neither a single image nor an entirely stable reference gets this exception.
    for other_burst in (None, np.repeat(depth[None],9,axis=0)):
        strict = check.compare_target_rgbd(rgb,rgb,depth,current,box,.001,config,reference_depth_burst=other_burst)
        assert not strict['passed'] and not strict['warnings']
        assert 'new_depth_surface_in_reference_holes' in strict['rejection_reasons']


@pytest.fixture
def saved_warning(warning_target,tmp_path,monkeypatch):
    import yaml
    from tools import preview_tabletop_target as preview
    from tools import run_vision_pick_cycle as cycle
    from tools.run_ggx_review import pin
    rgb,depth,current,box,config,burst = warning_target
    before = tmp_path/'before'; before.mkdir()
    after = tmp_path/'after'; after.mkdir()
    camera = dict(serial='fake',stream={},factory_color_intrinsics={},depth=dict(scale_m_per_unit=.001))
    meta = dict(schema='rh56_realsense_camera_check/v1',camera=camera,
        depth_aggregation='temporal median; RGB is the middle frame',source_frame_numbers=list(range(9)))
    for directory,d in ((before,depth),(after,current)):
        cv2.imwrite(str(directory/'camera_check_rgb.png'),rgb)
        cv2.imwrite(str(directory/'camera_check_depth_u16.png'),d)
        (directory/'camera_check.yaml').write_text(yaml.safe_dump(meta))
    for i,d in enumerate(burst):
        directory = before/f'capture_{i:02d}'; directory.mkdir()
        cv2.imwrite(str(directory/'camera_check_depth_u16.png'),d)
        (directory/'camera_check.yaml').write_text(yaml.safe_dump(dict(
            schema=meta['schema'],camera=camera,frame_number=i)))
    row = dict(component_id=1,geometry_candidate=True,rejection_reasons=[],bbox_xywh=box,
               top_patch=dict(yaw_ambiguous=False))
    source = dict(pins=pin([before/'camera_check_rgb.png',before/'camera_check_depth_u16.png',
                           before/'camera_check.yaml']),
        perception=dict(config='fake',capture=str(before),candidates=[row]),crop=dict(reviewed_component_id=1))
    monkeypatch.setattr(preview,'load_perception_config',lambda *a:({},None))
    monkeypatch.setattr(cycle,'validate_capture',lambda *a:meta)
    return source, after, config, tmp_path


@pytest.mark.parametrize('answer', [None, False, True, 'yes', 1])
def test_warning_needs_explicit_photo_confirmation_and_saves_evidence(saved_warning,answer):
    from xml.etree import ElementTree
    source, after, config, out = saved_warning
    def confirm(path,warnings):
        assert path.is_file() and len(warnings) == 1
        tree = ElementTree.parse(path)
        images = tree.findall('.//{http://www.w3.org/2000/svg}image')
        assert len(images) == 4 and all(i.attrib['href'].startswith('data:image/png;base64,') for i in images)
        return answer
    options = {} if answer is None else dict(review_warning=confirm)
    if answer is True:
        result = check.verify_saved_target(source,after,config,out/'check',**options)
        assert result['passed'] and not result['automatic_passed']
        assert result['visual_confirmation_received']
        assert str(after/'camera_check_rgb.png') in source['pins']
        assert str(out/'check_review.svg') in source['pins']
        assert str(out/'check.json') in source['pins']
    else:
        with pytest.raises(RuntimeError,match='confirmation|cancelled'):
            check.verify_saved_target(source,after,config,out/'check',**options)
    report = json.loads((out/'check.json').read_text())
    assert report['passed'] is (answer is True)
    assert report['visual_confirmation_received'] is (answer is True)
    assert report['visual_confirmation_required'] and not report['automatic_passed']
    assert report['reviewed_sha256'] and not report['hardware_contacted']
    assert (out/'check.csv').is_file()


@pytest.mark.parametrize('fault', ['rgb_shift','depth_shift','stable_loss','unstable_existing_surface'])
def test_warning_cannot_override_a_hard_failure(saved_warning,fault):
    source,after,config,out = saved_warning
    rgb_path, depth_path = after/'camera_check_rgb.png',after/'camera_check_depth_u16.png'
    if fault == 'rgb_shift':
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.warpAffine(rgb,np.array([[1.,0.,8.],[0.,1.,0.]]),(260,180))
        cv2.imwrite(str(rgb_path),rgb)
    else:
        depth = cv2.imread(str(depth_path),-1)
        if fault == 'depth_shift': depth[depth>0] -= 4
        if fault == 'stable_loss': depth[80:105,50:103] = 0
        if fault == 'unstable_existing_surface': depth[55:60,50:103] = 390
        cv2.imwrite(str(depth_path),depth)
    with pytest.raises(RuntimeError,match='changed/uncertain'):
        check.verify_saved_target(source,after,config,out/'check',
            review_warning=lambda *a:pytest.fail('Hard failure cannot be approved'))
    report = json.loads((out/'check.json').read_text())
    assert not report['passed'] and not report['visual_confirmation_received']
    assert report['rejection_reasons']


@pytest.mark.parametrize('what', ['rgb','depth','review','reference'])
def test_warning_review_artifact_change_cannot_be_approved(saved_warning,what):
    source,after,config,out = saved_warning
    def confirm(path,warnings):
        changed = dict(rgb=after/'camera_check_rgb.png',depth=after/'camera_check_depth_u16.png',
            review=path,reference=next(p for p in source['pins'] if 'capture_00' in p))
        from pathlib import Path
        Path(changed[what]).write_bytes(b'changed while reviewing')
        return True
    with pytest.raises(ValueError,match='artifact changed'):
        check.verify_saved_target(source,after,config,out/'check',review_warning=confirm)
    assert not json.loads((out/'check.json').read_text())['passed']


def test_initial_photo_acknowledgement_is_not_reused_for_final_photo(saved_warning):
    source,after,config,out = saved_warning
    assert check.verify_saved_target(source,after,config,out/'initial',review_warning=lambda *a:True)['passed']
    with pytest.raises(RuntimeError,match='requires explicit'):
        check.verify_saved_target(source,after,config,out/'final')


@pytest.mark.parametrize('stage',['initial','final'])
@pytest.mark.parametrize('depth_fault',['unchanged','shift','missing','unstable_existing_surface'])
def test_static_rgb_does_not_require_any_new_depth_pass(saved_warning,stage,depth_fault,monkeypatch):
    source,after,config,out = saved_warning
    depth_path = after/'camera_check_depth_u16.png'
    if depth_fault in ('shift','unstable_existing_surface'):
        depth = cv2.imread(str(depth_path),-1)
        if depth_fault == 'shift': depth[depth>0] -= 100
        else: depth[55:60,50:103] = 390
        cv2.imwrite(str(depth_path),depth)
        # The exact reported failure class still blocks default mode.
        with pytest.raises(RuntimeError,match='changed/uncertain'):
            check.verify_saved_target(source,after,config,out/'full_check')
    if depth_fault == 'missing': depth_path.unlink()
    monkeypatch.setattr(check,'compare_target_rgbd',lambda *a,**kw:pytest.fail('Static mode must not compare depth'))
    monkeypatch.setattr(check,'load_reference_burst',lambda *a,**kw:pytest.fail('Static mode must not load depth burst'))
    prefix = out/f'{stage}_scene_check'
    result = check.verify_static_scene_rgb(source,after,config,prefix)
    assert result['passed'] and result['depth_checked'] is False
    assert result['method'] == 'static_scene_rgb/v2'
    report = json.loads(prefix.with_suffix('.json').read_text())
    assert report['schema'] == 'rh56_ggx_static_scene_check/v2'
    assert 'initial_depth_report' not in report
    assert not report['depth_checked'] and not report['hardware_contacted']
    assert not any('depth_u16' in p for p in report['input_sha256'])
    assert str(after/'camera_check_rgb.png') in source['pins']


@pytest.mark.parametrize('fault',['missing_component','ambiguous','rejected','tampered_reference','missing_current_rgb'])
def test_static_mode_still_requires_valid_reference_and_current_rgb(saved_warning,fault):
    from pathlib import Path
    source,after,config,out = saved_warning
    prefix = out/'initial_scene_check'
    if fault == 'missing_component': source['perception']['candidates'] = []
    if fault == 'ambiguous': source['perception']['candidates'][0]['top_patch']['yaw_ambiguous'] = True
    if fault == 'rejected': source['perception']['candidates'][0]['rejection_reasons'] = ['bad']
    if fault == 'tampered_reference': (Path(source['perception']['capture'])/'camera_check_rgb.png').write_bytes(b'changed')
    if fault == 'missing_current_rgb': (after/'camera_check_rgb.png').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        check.verify_static_scene_rgb(source,after,config,prefix)
    assert not json.loads(prefix.with_suffix('.json').read_text())['passed']


@pytest.mark.parametrize('stage',['initial','final'])
@pytest.mark.parametrize('fault',['shift','occlusion','blank','camera'])
def test_static_still_blocks_rgb_or_camera_changes(saved_warning,monkeypatch,fault,stage):
    from pathlib import Path
    import yaml
    from tools import run_vision_pick_cycle as cycle
    source,after,config,out = saved_warning
    if stage == 'final':
        assert check.verify_static_scene_rgb(source,Path(source['perception']['capture']),config,
                                             out/'initial_scene_check')['passed']
    if fault == 'camera':
        path = after/'camera_check.yaml'
        meta = yaml.safe_load(path.read_text());meta['camera']['serial'] = 'changed'
        path.write_text(yaml.safe_dump(meta))
        monkeypatch.setattr(cycle,'validate_capture',lambda c,p:yaml.safe_load((p/'camera_check.yaml').read_text()))
    else:
        path = after/'camera_check_rgb.png'; rgb = cv2.imread(str(path))
        if fault == 'shift': rgb = cv2.warpAffine(rgb,np.array([[1.,0.,3.],[0.,1.,0.]]),(260,180))
        if fault == 'occlusion': rgb[60:110,65:120] = 0
        if fault == 'blank': rgb[:] = 100
        cv2.imwrite(str(path),rgb)
    with pytest.raises((ValueError,RuntimeError)):
        check.verify_static_scene_rgb(source,after,config,out/f'{stage}_scene_check')
    assert not json.loads((out/f'{stage}_scene_check.json').read_text())['passed']
