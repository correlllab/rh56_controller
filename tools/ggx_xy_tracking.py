"""Small XY-only relocation of one selected, replaced object using reference RGB.

Current depth is NOT used. Reference depth + calibrated camera rays supply the
metric translation at fixed reference point heights. Not a universal detector,
yaw estimator, safety check, or physical accuracy certificate.
"""
from pathlib import Path
import cv2
import numpy as np
import yaml

from tools import ggx_template as t
from tools.run_ggx_review import pin, verify


def match_translation(reference, current, mask, *, search_px=60):
    if (reference.dtype != np.uint8 or current.dtype != np.uint8 or reference.shape != current.shape
            or reference.ndim != 3 or reference.shape[2] != 3 or mask.shape != reference.shape[:2]):
        raise ValueError('Expected matching RGB images and reference target mask')
    if type(search_px) is not int or not 5 <= search_px <= 150:
        raise ValueError('Invalid bounded XY search window')
    yy, xx = np.nonzero(mask)
    if len(xx) < 150:
        raise ValueError('Insufficient reference target pixels for XY tracking')
    # Include a narrow silhouette band so uniform colored cubes still provide
    # edges. Do not register the whole static background or the robot.
    weight = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    y, x = np.nonzero(weight)
    x0, x1, y0, y1 = int(x.min()), int(x.max())+1, int(y.min()), int(y.max())+1
    H, W = mask.shape
    if x0-search_px < 0 or y0-search_px < 0 or x1+search_px > W or y1+search_px > H:
        raise ValueError('XY search window extends outside calibrated image')
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (reference, current)]
    patch = gray[0][y0:y1, x0:x1]
    weights = weight[y0:y1, x0:x1]*255
    if float(np.std(patch[weights != 0])) < 8:
        raise ValueError('Insufficient RGB contrast for XY-only localization')
    scores = cv2.matchTemplate(gray[1][y0-search_px:y1+search_px, x0-search_px:x1+search_px],
                              patch, cv2.TM_CCOEFF_NORMED, mask=weights)
    scores = np.where(np.isfinite(scores), scores, -1.)
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    best = float(scores[iy, ix])
    competing = scores.copy()
    competing[max(0, iy-5):iy+6, max(0, ix-5):ix+6] = -1
    separation = best - float(competing.max())
    if (best < .7 or separation < .02 or ix in (0, scores.shape[1]-1)
            or iy in (0, scores.shape[0]-1)):
        raise ValueError(f'XY localization uncertain: correlation={best:.3f}, peak separation={separation:.3f}')
    def subpixel(a, b, c):
        curvature = a - 2*b + c
        return float(np.clip(.5*(a-c)/curvature, -.5, .5)) if curvature < -1e-9 else 0.
    shift = [ix-search_px + subpixel(scores[iy,ix-1], best, scores[iy,ix+1]),
             iy-search_px + subpixel(scores[iy-1,ix], best, scores[iy+1,ix])]
    return np.array(shift), dict(correlation=best, peak_separation=separation,
                                search_px=search_px, bbox_xywh=[x0,y0,x1-x0,y1-y0])


def metric_translation(pixels, depths_m, shift, K, distortion, base_from_camera):
    """Intersect translated image rays with each reference feature's fixed Z."""
    from tools.analyze_ur5_camera_colored_target import pixel_depth_to_base
    T = np.asarray(base_from_camera, dtype=float)
    pixels, depths_m, shift = (np.asarray(v, dtype=float) for v in (pixels, depths_m, shift))
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or depths_m.shape != (len(pixels),)
            or shift.shape != (2,) or not all(np.all(np.isfinite(v)) for v in (pixels, depths_m, shift))
            or np.any(depths_m <= 0)):
        raise ValueError('Invalid reference pixels/depth or image translation')
    if T.shape != (4,4) or not np.all(np.isfinite(T)):
        raise ValueError('Invalid calibrated camera transform')
    differences = []
    for uv, depth in zip(pixels, depths_m):
        old = pixel_depth_to_base(uv, float(depth), K, distortion, T)
        ray = pixel_depth_to_base(np.asarray(uv)+shift, 1., K, distortion, T) - T[:3,3]
        if abs(ray[2]) < .1:
            raise ValueError('Camera ray too parallel to constant-height plane')
        distance = (old[2]-T[2,3])/ray[2]
        if distance <= 0:
            raise ValueError('Fixed-height ray intersection is behind camera')
        new = T[:3,3] + distance*ray
        differences.append((new[:2]-old[:2])*1000)
    differences = np.asarray(differences)
    if len(differences) < 30 or not np.all(np.isfinite(differences)):
        raise ValueError('Insufficient valid reference depth for metric XY')
    delta = np.median(differences, axis=0)
    spread = float(np.percentile(np.linalg.norm(differences-delta, axis=1), 90))
    if spread > 3.:
        raise ValueError(f'XY translation inconsistent across reference heights: {spread:.3f} mm')
    return delta, spread


def locate(source, capture, out, *, search_px=60, max_shift_mm=40.):
    from tools.preview_tabletop_target import load_perception_config, compatible_background
    from tools.run_vision_pick_cycle import validate_capture
    report = dict(schema='rh56_ggx_xy_relocation/v1', status='starting', hardware_contacted=False,
        current_depth_used=False, estimated_dofs=['x','y'], z_locked=True, yaw_locked=True,
        assumptions=['Same object, camera, installation, resting face and approximately same yaw.',
            'Only RGB translation is measured; current depth is neither checked nor used.',
            'Metric conversion uses original measured depths with each original base-Z held fixed.',
            'Small rotation/occlusion/appearance change can bias XY; no yaw compensation or guaranteed physical accuracy.',
            'Uncertain localization stops the run, never falls back to the old XY.'])
    out.mkdir(parents=True, exist_ok=False)
    try:
        verify(source['pins'])
        ref = source['perception']
        c, _ = load_perception_config(Path(ref['config']))
        before, capture = Path(ref['capture']), Path(capture)
        ref_check, cur_check = validate_capture(c, before), validate_capture(c, capture)
        compatible_background(ref_check, cur_check)
        image_paths = [before/'camera_check_rgb.png', before/'camera_check_depth_u16.png',
            capture/'camera_check_rgb.png', capture/'camera_check.yaml', before/'camera_check.yaml']
        report['input_sha256'] = pin(image_paths)
        reference, current = [cv2.imread(str(p)) for p in (image_paths[0], image_paths[2])]
        depth = cv2.imread(str(image_paths[1]), cv2.IMREAD_UNCHANGED)
        if (reference is None or current is None or depth is None
                or reference.shape != (720,1280,3) or current.shape != reference.shape or depth.shape != (720,1280)):
            raise ValueError('Missing calibrated RGB / reference depth images')
        component = source['crop']['reviewed_component_id']
        rows = [r for r in ref['candidates'] if r['component_id'] == component]
        if len(rows) != 1 or not rows[0]['geometry_candidate'] or rows[0]['rejection_reasons']:
            raise ValueError('Missing valid reference object selection')
        mask_path = Path(source['crop']['source_preview']).parent/'foreground.png'
        foreground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if foreground is None or foreground.shape != depth.shape:
            raise ValueError('Missing reviewed reference foreground')
        report['input_sha256'].update(pin([mask_path]))
        _, labels = cv2.connectedComponents((foreground > 0).astype(np.uint8))
        mask = labels == component
        shift, match = match_translation(reference, current, mask, search_px=search_px)
        valid = (cv2.erode(mask.astype(np.uint8), np.ones((5,5), np.uint8)) != 0) & (depth > 0)
        ys, xs = np.nonzero(valid)
        if len(xs) < 150:
            raise ValueError('Reference object lacks reliable interior depth for metric XY')
        indices = np.linspace(0, len(xs)-1, min(512,len(xs)), dtype=int)
        pixels = np.c_[xs[indices], ys[indices]]
        cal = yaml.safe_load(Path(c['calibration']).read_text())
        intr = cal['selected_intrinsics']
        delta, spread = metric_translation(pixels, depth[ys[indices],xs[indices]] *
            ref_check['camera']['depth']['scale_m_per_unit'], shift, intr['camera_matrix'],
            intr['distortion_coefficients'], cal['transforms']['base_from_camera']['matrix'])
        if not np.isfinite(max_shift_mm) or not 1 <= max_shift_mm <= 100 or np.linalg.norm(delta) > max_shift_mm:
            raise ValueError('Object moved beyond the configured near-reference XY search')
        xy = np.array(source['preflight']['object_center_m'][:2])*1000 + delta
        report.update(status='localized', reference_capture=str(before), capture=str(capture),
            pixel_translation=shift.tolist(), delta_xy_mm=delta.tolist(), object_xy_mm=xy.tolist(),
            translation_spread_p90_mm=spread, match=match)
        overlay = current.copy()
        x,y,w,h = match['bbox_xywh']
        p = np.rint(np.array([x,y])+shift).astype(int)
        cv2.rectangle(overlay, tuple(p), tuple(p+[w,h]), (0,255,0), 2)
        cv2.putText(overlay, f'XY delta {delta[0]:+.1f}, {delta[1]:+.1f} mm | Z/yaw locked',
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
        cv2.imwrite(str(out/'overlay.png'), overlay)
        verify(source['pins'])
        verify(report['input_sha256'])
        return report
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        t.write_json(out/'summary.json', report)
