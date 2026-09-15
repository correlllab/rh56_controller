"""Fixed-reference RGB-D scene verification, without re-segmenting a known target.

Not a detector, pose estimator or hole filler. Default mode requires measured
RGB-D consistency. Explicit static-scene mode checks RGB only and relies on the
operator to confirm unchanged 3D geometry; it does not authorize robot motion.
"""
from __future__ import annotations

import csv
import base64
import hashlib
import json
from pathlib import Path
import warnings

import cv2
import numpy as np

REVISION = 'fixed_target_rgbd/v4'
# Conservative operational assumptions, not calibrated classifier confidence.
# Pixel checks are intentionally stricter than allowing a new translated grasp.
LIMITS = dict(rgb_delta=25., rgb_changed_fraction=.02, rgb_p95=20., exposure_offset=20.,
              edge_p95_px=2., min_edges=100, min_depth_fraction=.25,
              min_common_reference_fraction=.5, min_cell_points=20, min_supported_cells=4,
              new_depth_outlier_fraction=.1, min_stable_reference_fraction=.5)
DEPTH_ROUNDOFF_MM = 1e-6  # Float32 depth-scale representation, not sensor tolerance.


def reference_burst_paths(capture, metadata=None):
    """Declared reference frames only; missing/corrupt bursts must not fall back."""
    import yaml
    if metadata is None:
        metadata = yaml.safe_load((capture / 'camera_check.yaml').read_text())
    if not isinstance(metadata, dict):
        raise ValueError('Invalid reference capture metadata')
    if 'depth_aggregation' not in metadata and 'source_frame_numbers' not in metadata:
        return []  # Legacy single image: retain the strict all-measured-pixel check.
    numbers = metadata.get('source_frame_numbers')
    if (metadata.get('depth_aggregation') != 'temporal median; RGB is the middle frame'
            or not isinstance(numbers, list) or not 3 <= len(numbers) <= 60
            or any(type(n) is not int or n < 0 for n in numbers)
            or any(a >= b for a, b in zip(numbers, numbers[1:]))):
        raise ValueError('Invalid reference burst declaration')
    return [capture / f'capture_{i:02d}' / name for i in range(len(numbers))
            for name in ('camera_check.yaml', 'camera_check_depth_u16.png')]


def load_reference_burst(capture, metadata, pins, shape):
    """Bind raw reference evidence before CB3 review; never use current noise to mask errors."""
    import yaml
    from tools.run_ggx_review import pin, verify
    paths = reference_burst_paths(capture, metadata)
    if not paths:
        return None, {}
    verify(pins)
    raw_pins = pin(paths)
    # Existing pins are never overwritten with new hashes to bless an edit.
    verify(pins)
    if any(p in pins and pins[p] != h for p, h in raw_pins.items()):
        raise ValueError('Reference burst changed')
    pins.update(raw_pins)
    frames = []
    for i in range(0, len(paths), 2):
        frame = yaml.safe_load(paths[i].read_text())
        if (not isinstance(frame, dict) or frame.get('schema') != metadata.get('schema')
                or frame.get('camera') != metadata.get('camera')
                or frame.get('frame_number') != metadata['source_frame_numbers'][i // 2]):
            raise ValueError('Reference burst frame metadata mismatch')
        depth = cv2.imread(str(paths[i+1]), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape != shape or depth.dtype != np.uint16:
            raise ValueError('Invalid reference burst depth frame')
        frames.append(depth)
    verify(raw_pins)
    return np.stack(frames), raw_pins


def reference_stability(depth, burst, scale_m, config):
    """Reference-only repeatability budget, not an estimate of absolute accuracy.

    Reserve half the existing paired-image error budget for reference noise.
    This is a provisional quality criterion, not a probabilistic noise bound.
    Unreliable pixels remain unknown; no depth is filled or changed.
    """
    from tools.capture_tabletop_background import summarize_depth
    reconstructed, _, _ = summarize_depth(burst, scale_m)
    if not np.array_equal(reconstructed, depth):
        raise ValueError('Reference burst does not reproduce approved depth in target ROI')
    error = np.where(burst > 0, np.abs(burst.astype(float)-depth)*scale_m*1000, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # All-invalid stays unknown.
        median, p90 = np.nanpercentile(error, [50, 90], axis=0)
    median_limit = config['scene_center_tolerance_mm'] / 2
    tail_limit = config['scene_dimension_tolerance_mm'] / 2
    stable = ((depth > 0) & (median <= median_limit+DEPTH_ROUNDOFF_MM)
              & (p90 <= tail_limit+DEPTH_ROUNDOFF_MM))
    return stable, dict(method='reference_burst_half_error_budget/v1', frames=len(burst),
        temporal_median_abs_limit_mm=median_limit, temporal_p90_abs_limit_mm=tail_limit,
        measured_points=int((depth > 0).sum()), stable_points=int(stable.sum()),
        stable_reference_fraction=float(stable.sum()/max(1, (depth > 0).sum())),
        mask_sha256=hashlib.sha256(stable.tobytes()).hexdigest(),
        assumptions=['Quality mask uses reference frames only, never current residuals.',
            'Recomputed 80%-valid temporal median matches approved target ROI exactly.',
            'Unstable reference pixels are unknown for millimetre checks; coarse envelope/RGB remain.',
            'Short-term repeatability is not absolute accuracy or a complete obstacle certificate.'])


def compare_target_rgb(reference_rgb, current_rgb, bbox):
    """Fixed-pixel appearance only. A pass does NOT establish unchanged 3D pose."""
    a, b = map(np.asarray, (reference_rgb, current_rgb))
    if (a.ndim != 3 or a.shape[2] != 3 or a.shape != b.shape
            or a.dtype != np.uint8 or b.dtype != np.uint8):
        raise ValueError('Expected matching uint8 RGB images')
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)) or not np.all(box == np.floor(box)):
        raise ValueError('Invalid reference target bbox')
    x, y, w, h = box.astype(int)
    if min(w, h) < 30 or x < 4 or y < 4 or x+w+4 > a.shape[1] or y+h+4 > a.shape[0]:
        raise ValueError('Reference target too small or too near image boundary')
    result = dict(revision='fixed_target_rgb/v1', bbox_xywh=box.astype(int).tolist(),
        depth_checked=False, rejection_reasons=[], limits={k:LIMITS[k] for k in
            ('rgb_delta','rgb_changed_fraction','rgb_p95','exposure_offset','edge_p95_px','min_edges')})
    reasons = result['rejection_reasons']
    # Include the object outline. A single exposure offset is allowed, but
    # no geometric registration or per-pixel correction can conceal movement.
    # Identical local denoising, no registration or target pose adjustment.
    aa, bb = [cv2.GaussianBlur(im[y-4:y+h+4, x-4:x+w+4], (3, 3), 0) for im in (a, b)]
    result['appearance_method'] = 'fixed_pixels_gaussian_3x3/v1'
    delta = bb.astype(float)-aa.astype(float)
    offset = np.median(delta, axis=(0, 1))
    error = np.max(np.abs(delta-offset), axis=2)
    changed = float(np.mean(error > LIMITS['rgb_delta']))
    rgb_p95 = float(np.percentile(error, 95))
    result['appearance'] = dict(exposure_offset=offset.tolist(), changed_fraction=changed, delta_p95=rgb_p95)
    if (np.max(np.abs(offset)) > LIMITS['exposure_offset'] or changed > LIMITS['rgb_changed_fraction']
            or rgb_p95 > LIMITS['rgb_p95']):
        reasons.append('target_appearance_changed_or_occluded')
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (aa, bb)]
    edges = [cv2.Canny(im, 100, 200) > 0 for im in gray]
    # Strong reference features must still have a nearby edge in the current
    # image, and vice versa. Match against weaker edges so crossing the strong
    # threshold is not misreported as a 100-pixel object displacement.
    matching_edges = [cv2.Canny(im, 50, 100) > 0 for im in gray]
    counts = [int(e.sum()) for e in edges]
    result['edges'] = dict(counts=counts, method='bidirectional_strong_to_weak/v1',
                          strong_thresholds=[100, 200], matching_thresholds=[50, 100])
    if min(counts) < LIMITS['min_edges']:
        reasons.append('insufficient_visual_structure')
    else:
        distances = [cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                     for e in matching_edges]
        errors = [float(np.percentile(distances[1][edges[0]], 95)),
                  float(np.percentile(distances[0][edges[1]], 95))]
        result['edges']['bidirectional_p95_px'] = errors
        if max(errors) > LIMITS['edge_p95_px']:
            reasons.append('target_outline_or_texture_moved')
    result['passed'] = not reasons
    return result


def compare_target_rgbd(reference_rgb, current_rgb, reference_depth, current_depth,
                        bbox, scale_m, config, *, reference_depth_burst=None):
    """Compare fixed pixels; never align/warp a moved target into acceptance."""
    result = compare_target_rgb(reference_rgb, current_rgb, bbox)
    a, da, db = map(np.asarray, (reference_rgb, reference_depth, current_depth))
    if da.shape != a.shape[:2] or db.shape != da.shape or da.dtype != np.uint16 or db.dtype != np.uint16:
        raise ValueError('Expected matching uint8 RGB and uint16 aligned depth images')
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError('Invalid depth scale')
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in
           ('scene_center_tolerance_mm', 'scene_dimension_tolerance_mm')):
        raise ValueError('Invalid depth tolerances')
    if reference_depth_burst is not None:
        reference_depth_burst = np.asarray(reference_depth_burst)
        if (reference_depth_burst.ndim != 3 or reference_depth_burst.shape[1:] != da.shape
                or reference_depth_burst.dtype != np.uint16
                or not 3 <= len(reference_depth_burst) <= 60):
            raise ValueError('Expected 3..60 matching uint16 reference depth frames')
    result.update(revision=REVISION, depth_checked=True, limits=dict(LIMITS), warnings=[], depth_cells=[])
    reasons = result['rejection_reasons']
    x, y, w, h = result['bbox_xywh']
    # Inset reduces bbox-boundary mixing; it does not make a curved object flat.
    aa, bb = [im[y+5:y+h-5, x+5:x+w-5] for im in (da, db)]
    va, vb = aa > 0, bb > 0
    stable = va
    result['reference_quality'] = dict(method='legacy_all_measured_pixels')
    if reference_depth_burst is not None:
        stable, quality = reference_stability(aa, reference_depth_burst[:, y+5:y+h-5, x+5:x+w-5],
                                             scale_m, config)
        result['reference_quality'] = quality
        if quality['stable_reference_fraction'] < LIMITS['min_stable_reference_fraction']:
            reasons.append('insufficient_stable_reference_depth')
    common = va & vb
    result['depth'] = dict(reference_valid_fraction=float(va.mean()), current_valid_fraction=float(vb.mean()),
                          common_reference_fraction=float(common.sum()/max(1, va.sum())))
    if (min(va.mean(), vb.mean()) < LIMITS['min_depth_fraction']
            or common.sum() < LIMITS['min_common_reference_fraction'] * va.sum()):
        reasons.append('insufficient_common_depth')
    depth_tol = config['scene_center_tolerance_mm']
    tail_tol = config['scene_dimension_tolerance_mm']
    result['limits'].update(depth_cell_median_mm=depth_tol, depth_cell_p90_mm=tail_tol,
                            depth_roundoff_mm=DEPTH_ROUNDOFF_MM)
    supported = []
    for row in range(3):
        for col in range(3):
            sy = slice(round(row*aa.shape[0]/3), round((row+1)*aa.shape[0]/3))
            sx = slice(round(col*aa.shape[1]/3), round((col+1)*aa.shape[1]/3))
            ar, br = aa[sy, sx], bb[sy, sx]
            ref_valid, cur_valid = ar > 0, br > 0
            reliable = stable[sy, sx]
            both = reliable & cur_valid
            nref, ncommon = int(ref_valid.sum()), int(both.sum())
            nstable = int(reliable.sum())
            cell = dict(row=row, column=col, reference_points=nref, stable_reference_points=nstable,
                        common_points=ncommon)
            result['depth_cells'].append(cell)
            if nref < LIMITS['min_cell_points']:
                cell['status'] = 'unobserved_in_reference'
                continue  # Missing depth is explicitly unknown, never filled.
            # Newly valid pixels must also agree with the measured local top,
            # not introduce a nearer object into old missing-depth pixels.
            # Curved shoulders can occupy <1% of a cell's measured pixels.
            # Percentile trimming erased these real surfaces. Keep the observed
            # envelope; this still does not fill holes or synthesize geometry.
            measured = ar[ref_valid].astype(float)*scale_m*1000
            lo, hi = float(measured.min()), float(measured.max())
            new = br[cur_valid & ~ref_valid].astype(float)*scale_m*1000
            cell.update(new_depth_reference_envelope_mm=[lo, hi], new_depth_points=len(new))
            outside = (new < lo-tail_tol-DEPTH_ROUNDOFF_MM) | (new > hi+tail_tol+DEPTH_ROUNDOFF_MM)
            cell['new_depth_outlier_fraction'] = float(outside.mean()) if len(new) else 0.
            if (len(new) >= LIMITS['min_cell_points']
                    and np.mean(outside) > LIMITS['new_depth_outlier_fraction']):
                if reference_depth_burst is not None and nstable < LIMITS['min_cell_points']:
                    # Only a reference-proven unreliable cell is eligible for
                    # an explicit photo review. This is NOT an automatic pass.
                    cell['warning'] = 'new_depth_in_unstable_reference_holes'
                    result['warnings'].append(dict(code=cell['warning'], row=row, column=col,
                        bbox_xywh=list(map(int, [x+5+sx.start, y+5+sy.start, sx.stop-sx.start, sy.stop-sy.start])),
                        new_depth_points=len(new), outlier_points=int(outside.sum()),
                        outlier_fraction=cell['new_depth_outlier_fraction']))
                else:
                    reasons.append('new_depth_surface_in_reference_holes')
            # Even excluded noisy pixels may not introduce a grossly different
            # measured surface. Loss there is unknown, not affirmative clearance.
            uncertain = br[ref_valid & ~reliable & cur_valid].astype(float)*scale_m*1000
            outside = (uncertain < lo-tail_tol-DEPTH_ROUNDOFF_MM) | (uncertain > hi+tail_tol+DEPTH_ROUNDOFF_MM)
            cell['unstable_current_points'] = len(uncertain)
            cell['unstable_outside_envelope_fraction'] = float(outside.mean()) if len(uncertain) else 0.
            if len(uncertain) >= LIMITS['min_cell_points'] and outside.mean() > LIMITS['new_depth_outlier_fraction']:
                reasons.append('unstable_region_new_depth_surface')
            if nstable < LIMITS['min_cell_points']:
                cell['status'] = 'unstable_in_reference'
                continue
            if ncommon < max(LIMITS['min_cell_points'], LIMITS['min_common_reference_fraction']*nstable):
                cell['status'] = 'insufficient_common_depth'
                reasons.append('localized_depth_loss')
                continue
            supported.append((row, col))
            mm = (br[both].astype(float)-ar[both].astype(float))*scale_m*1000
            median, p90 = float(np.median(mm)), float(np.percentile(np.abs(mm), 90))
            cell.update(status='checked', median_delta_mm=median, absolute_delta_p90_mm=p90)
            if abs(median) > depth_tol + DEPTH_ROUNDOFF_MM or p90 > tail_tol + DEPTH_ROUNDOFF_MM:
                reasons.append('target_depth_changed')
    if (len(supported) < LIMITS['min_supported_cells'] or len({r for r, c in supported}) < 3
            or len({c for r, c in supported}) < 3):
        reasons.append('insufficient_spatial_depth_support')
    result['rejection_reasons'] = sorted(set(reasons))
    result['numerical_checks_passed'] = not result['rejection_reasons']
    result['visual_confirmation_required'] = bool(result['warnings'])
    result['automatic_passed'] = result['numerical_checks_passed'] and not result['warnings']
    result['passed'] = bool(result['automatic_passed'])
    return result


def write_warning_review(before, capture, comparison, path):
    """Native SVG layout of unchanged PNGs: full views + aligned zooms, no registration."""
    x, y, w, h = comparison['bbox_xywh']
    reference = cv2.imread(str(before / 'camera_check_rgb.png'))
    ih, iw = reference.shape[:2]
    panels = []
    for column, (folder, title) in enumerate(((before, '参考 / Reference'), (capture, '当前 / Current'))):
        encoded = base64.b64encode((folder / 'camera_check_rgb.png').read_bytes()).decode('ascii')
        picture = f'<image width="{iw}" height="{ih}" href="data:image/png;base64,{encoded}"/>'
        boxes = ''.join(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                        'fill="none" stroke="#ff7800" stroke-width="1.5"/>'
                        for bx, by, bw, bh in (r['bbox_xywh'] for r in comparison['warnings']))
        panels.append(f'<g transform="translate({column*640},0)">'
            f'<text x="20" y="30" fill="white" font-size="24">{title}</text>'
            f'<svg y="45" width="640" height="360" viewBox="0 0 {iw} {ih}">{picture}{boxes}</svg>'
            f'<svg y="420" width="640" height="410" viewBox="{x-15} {y-15} {w+30} {h+30}">'
            f'{picture}{boxes}</svg></g>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="900" '
        'viewBox="0 0 1280 900"><rect width="1280" height="900" fill="#202020"/>'
        + ''.join(panels) + '<text x="20" y="870" fill="white" font-size="18">'
        '橙框 = 待人工复核区域；照片不能证明高度不变。 Orange = unknown depth; NOT clearance.</text></svg>\n')
    return path


def verify_saved_target(source, capture, config, prefix, *, review_warning=None):
    """Camera-free file check; save diagnostics on both pass and failure."""
    from tools.preview_tabletop_target import load_perception_config, compatible_background
    from tools.run_vision_pick_cycle import validate_capture
    from tools.run_ggx_review import pin, verify
    report = dict(schema='rh56_ggx_reference_scene_check/v1', passed=False,
        hardware_contacted=False, executable_target=False, visual_confirmation_received=False,
        assumptions=['Same fixed camera, desk and opaque reviewed target; not arbitrary-object detection.',
            'Fixed-pixel appearance/edge and measured-depth checks; no pose update, warp or missing-depth fill.',
            'Local target verification, not a complete obstacle or collision certificate.',
            'Provisional visual quality thresholds; weak appearance/depth evidence fails closed.',
            'Reference-unstable hole warnings require explicit current-photo review; never automatic approval.'])
    path = prefix.with_suffix('.json')
    try:
        verify(source['pins'])
        reference = source['perception']
        component = source['crop']['reviewed_component_id']
        rows = [r for r in reference['candidates'] if r['component_id'] == component]
        if len(rows) != 1 or not rows[0]['geometry_candidate'] or rows[0]['rejection_reasons']:
            raise ValueError('Missing valid reviewed reference component')
        from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA, validate_foreground_report
        if source['crop'].get('schema') == FOREGROUND_SCHEMA:
            validate_foreground_report(source['crop'])
        elif not rows[0].get('top_patch') or rows[0]['top_patch']['yaw_ambiguous']:
            raise ValueError('Missing valid reviewed reference component')
        c, _ = load_perception_config(Path(reference['config']))
        before = Path(reference['capture'])
        ref_check, cur_check = validate_capture(c, before), validate_capture(c, capture)
        compatible_background(cur_check, ref_check)
        image_pins = pin([folder / name for folder in (before, capture)
            for name in ('camera_check_rgb.png', 'camera_check_depth_u16.png', 'camera_check.yaml')
            if (folder / name).is_file()])
        images = [cv2.imread(str(folder / name), flag) for name, flag in
                  [('camera_check_rgb.png', cv2.IMREAD_COLOR), ('camera_check_depth_u16.png', cv2.IMREAD_UNCHANGED)]
                  for folder in (before, capture)]
        if any(im is None for im in images):
            raise ValueError('Missing reference/current RGB-D image')
        burst, raw_pins = load_reference_burst(before, ref_check, source['pins'], images[2].shape)
        report['reference_burst_sha256'] = raw_pins
        comparison = compare_target_rgbd(*images, rows[0]['bbox_xywh'],
            cur_check['camera']['depth']['scale_m_per_unit'], config, reference_depth_burst=burst)
        report.update(comparison, reference_capture=str(before), current_capture=str(capture),
                      reference_component_id=component)
        verify(source['pins'])
        verify(image_pins)
        if comparison['rejection_reasons']:
            raise RuntimeError('Reference target changed/uncertain: ' + ', '.join(comparison['rejection_reasons']))
        if comparison['visual_confirmation_required']:
            review_path = write_warning_review(before, capture, comparison, prefix.with_name(prefix.name+'_review.svg'))
            review_pins = {**source['pins'], **image_pins, **pin([review_path])}
            report.update(review_image=str(review_path), reviewed_sha256=review_pins)
            verify(review_pins)
            if review_warning is None:
                raise RuntimeError('Reference depth warning requires explicit current-photo confirmation')
            accepted = review_warning(review_path, comparison['warnings'])
            verify(review_pins)
            if accepted is not True:
                raise RuntimeError('Reference depth warning photo review cancelled; no motion authorized')
            report.update(passed=True, visual_confirmation_received=True)
            source['pins'].update(review_pins)  # Also verified again before motion.
    except Exception as exc:
        report.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        with prefix.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['passed', 'automatic_passed',
                'visual_confirmation_required', 'visual_confirmation_received', 'error'])
            writer.writeheader()
            writer.writerow({key: report.get(key) for key in writer.fieldnames})
    source['pins'].update(pin([path]))
    return dict(method=REVISION, reference_component_id=component, verification_report=str(path), passed=True,
        automatic_passed=report['automatic_passed'], visual_confirmation_required=report['visual_confirmation_required'],
        visual_confirmation_received=report['visual_confirmation_received'])


def verify_static_scene_rgb(source, capture, config, prefix):
    """Check each fresh RGB against the ORIGINAL reference, never a cached pass.

    Explicit operator-fixed-scene mode: neither initial nor final depth is
    revalidated. Approved geometry stays unchanged; RGB is not a 3D certificate.
    """
    from tools.preview_tabletop_target import load_perception_config, compatible_background
    from tools.run_vision_pick_cycle import validate_capture
    from tools.run_ggx_review import pin, verify
    report = dict(schema='rh56_ggx_static_scene_check/v2', passed=False, depth_checked=False,
        hardware_contacted=False, executable_target=False,
        assumptions=['Explicit static-scene mode, one invocation only; not a permanent scene certificate.',
            'Approved reference geometry retained under operator confirmation that object/camera/desk/mount stay fixed.',
            'No new depth consistency check at either initial or final stage; no prior live-depth pass required.',
            'Fresh RGB checks appearance only; height, hidden obstacles and unchanged 3D pose are NOT established.'])
    path = prefix.with_suffix('.json')
    try:
        verify(source['pins'])
        component = source['crop']['reviewed_component_id']
        reference = source['perception']
        before = Path(reference['capture'])
        rows = [r for r in reference['candidates'] if r['component_id'] == component]
        if len(rows) != 1 or not rows[0]['geometry_candidate'] or rows[0]['rejection_reasons']:
            raise ValueError('Missing valid reviewed reference component')
        from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA, validate_foreground_report
        if source['crop'].get('schema') == FOREGROUND_SCHEMA:
            validate_foreground_report(source['crop'])
        elif not rows[0].get('top_patch') or rows[0]['top_patch']['yaw_ambiguous']:
            raise ValueError('Missing valid reviewed reference component')
        c,_ = load_perception_config(Path(reference['config']))
        compatible_background(validate_capture(c,capture),validate_capture(c,before))
        paths = [folder/name for folder in (before,capture)
                 for name in ('camera_check.yaml','camera_check_rgb.png')]
        inputs = pin(paths)
        a,b = [cv2.imread(str(folder/'camera_check_rgb.png')) for folder in (before,capture)]
        comparison = compare_target_rgb(a,b,rows[0]['bbox_xywh'])
        report.update(comparison,
            reference_component_id=component,reference_capture=str(before),current_capture=str(capture),
            input_sha256=inputs)
        verify(inputs)
        verify(source['pins'])
        if not comparison['passed']:
            raise RuntimeError('Static scene RGB changed/uncertain: '+', '.join(comparison['rejection_reasons']))
    except Exception as exc:
        report.update(passed=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        with prefix.with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['passed','depth_checked','error'])
            writer.writeheader();writer.writerow({k:report.get(k) for k in writer.fieldnames})
    source['pins'].update(inputs)
    source['pins'].update(pin([path]))
    return dict(passed=True,method='static_scene_rgb/v2',depth_checked=False,
                reference_component_id=component,verification_report=str(path))
