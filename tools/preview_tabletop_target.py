#!/usr/bin/env python3
"""Offline, color-independent tabletop foreground preview. NEVER a robot target.

Reuses calibrated depth deprojection and desk-plane APIs. Dimensions describe
visible surfaces, not hidden geometry. The operator must identify hand/clamps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_perception_config(path):
    """Independent image region; explicit legacy configs retain their XY gate."""
    import yaml
    path = Path(path)
    c = yaml.safe_load(path.read_text())
    if not isinstance(c, dict):
        raise ValueError('Expected a perception or legacy cycle config mapping')
    if c.get('schema') == 'rh56_vision_pick_cycle_config/v1':
        from tools.run_vision_pick_cycle import load_config
        legacy = load_config(path)
        return legacy, np.asarray(legacy['target_bounds_mm'], dtype=float)[:,:2]
    required = {'schema', 'camera_serial', 'calibration', 'desk_plane', 'desk_z_mm', 'roi'}
    if c.get('schema') != 'rh56_tabletop_vision_config/v1' or set(c) != required:
        raise ValueError('Expected tabletop vision v1 with only camera/calibration/desk/ROI fields')
    if not isinstance(c['camera_serial'], str) or not c['camera_serial']:
        raise ValueError('Camera serial must be a nonempty string')
    for key in ('calibration', 'desk_plane'):
        if not isinstance(c[key], str) or not c[key]:
            raise ValueError(f'{key} must be a nonempty path')
        p = Path(c[key])
        c[key] = str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    roi = np.asarray(c['roi'], dtype=float)
    if (roi.shape != (4,) or not np.all(np.isfinite(roi))
            or not np.all(roi == np.floor(roi))
            or not 0 <= roi[0] < roi[2] <= 1280 or not 0 <= roi[1] < roi[3] <= 720):
        raise ValueError('ROI must lie inside the calibrated 1280x720 image')
    c['roi'] = roi.astype(int).tolist()
    if isinstance(c['desk_z_mm'], bool) or not np.isfinite(c['desk_z_mm']) or not -100 <= c['desk_z_mm'] <= 300:
        raise ValueError('Invalid calibrated desk height')
    return c, None


def background_foreground(depth, background, scale_m, mad_mm=None, min_change_mm=8.):
    """Keep measured depth nearer than a supported background; never fill holes."""
    depth, background = np.asarray(depth), np.asarray(background)
    if depth.shape != background.shape or depth.ndim != 2 or depth.dtype != np.uint16 or background.dtype != np.uint16:
        raise ValueError('Background and current depth must be matching uint16 images')
    if not np.isfinite(scale_m) or scale_m <= 0 or not np.isfinite(min_change_mm) or min_change_mm <= 0:
        raise ValueError('Scale and depth-change threshold must be positive and finite')
    noise = np.zeros(depth.shape) if mad_mm is None else np.asarray(mad_mm, dtype=float)
    if noise.shape != depth.shape or np.any(noise[np.isfinite(noise)] < 0):
        raise ValueError('Invalid background MAD image')
    known = (background > 0) & np.isfinite(noise)
    delta = (background.astype(float)-depth.astype(float))*scale_m*1000
    threshold = np.maximum(min_change_mm, 3*noise)
    return known & (depth > 0) & (delta > threshold), known


def compatible_background(current_check, background_check):
    """Same camera/stream/depth settings; this cannot detect physical camera motion."""
    for key in ('serial', 'stream', 'factory_color_intrinsics', 'depth'):
        if current_check['camera'][key] != background_check['camera'][key]:
            raise ValueError(f'Background camera settings mismatch: {key}')


def top_patch_estimate(xyz, core):
    """A robust near-horizontal surface patch, NOT arbitrary-object dimensions.

    Estimate height from the eroded interior, then reject depth-edge tails using
    a band of max(3 mm, 3 * spatial MAD). No known object dimension is used.
    """
    import cv2
    if len(core) < 30:
        return None
    height = float(np.median(core[:, 3]))
    mad = float(np.median(np.abs(core[:, 3]-height)))
    band = max(3., 3*mad)
    patch = xyz[np.abs(xyz[:, 3]-height) <= band]
    if len(patch) < 30:
        return None
    center, sides, angle = cv2.minAreaRect(patch[:, :2].astype(np.float32))
    width, length = sides
    if width > length:
        width, length = length, width
        angle += 90
    return dict(center_base_mm=[float(center[0]), float(center[1]), float(np.median(core[:, 2]))],
                short_mm=float(width), long_mm=float(length), height_median_mm=height,
                short_axis_yaw_deg=float((angle+90)%180-90),
                yaw_ambiguous=bool(length <= width*1.1),
                core_spatial_height_mad_mm=mad, accepted_height_band_mm=band,
                points=len(patch), total_points=len(xyz),
                assumption='Eroded interior represents a near-horizontal top surface; not validated for arbitrary shapes')


def components(points, pixels, image_shape, roi, xy_bounds=None, *, min_height_mm=8.,
               max_height_mm=200., min_pixels=150, exclusions=(), foreground_gate=None,
               known_background=None, background_mode='pixels',
               min_background_seed_pixels=150, min_background_support_fraction=.25):
    """points columns: base X/Y/Z mm and height above desk in base Z mm.

    supported-surface recovers only measured current foreground where background
    is unknown, connected to substantial measured background-change evidence.
    Known unchanged pixels remain excluded. No missing current depth is filled.
    xy_bounds is an optional legacy perception gate, never a robot safety check.
    """
    import cv2
    points = np.asarray(points, dtype=float)
    pixels = np.asarray(pixels, dtype=int)
    if points.shape != (len(pixels), 4) or pixels.shape[1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError('Expected finite Nx4 points and Nx2 pixels')
    if background_mode not in ('pixels', 'supported-surface'):
        raise ValueError('Invalid background mode')
    if (not np.isfinite(min_background_support_fraction)
            or not 0 < min_background_support_fraction <= 1
            or min_background_seed_pixels < 1):
        raise ValueError('Invalid background support thresholds')
    recover = background_mode == 'supported-surface'
    if recover and (foreground_gate is None or known_background is None):
        raise ValueError('Surface recovery requires background change and support masks')
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = roi
    region = np.zeros_like(mask)
    region[y0:y1, x0:x1] = 1
    for a, b, c, d in exclusions:
        region[b:d, a:c] = 0
    eligible = (points[:, 3] > min_height_mm) & (region[pixels[:, 1], pixels[:, 0]] != 0)
    if foreground_gate is not None:
        if np.shape(foreground_gate) != mask.shape:
            raise ValueError('Foreground gate must match image dimensions')
        gate = np.asarray(foreground_gate, dtype=bool)
        if recover:
            known_background = np.asarray(known_background, dtype=bool)
            if known_background.shape != mask.shape or np.any(gate & ~known_background):
                raise ValueError('Background support must match image and contain all change seeds')
            gate = gate | ~known_background
        eligible &= gate[pixels[:, 1], pixels[:, 0]]
    mask[pixels[eligible, 1], pixels[eligible, 0]] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    border = region - cv2.erode(region, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    point_labels = labels[pixels[:, 1], pixels[:, 0]]
    rows = []
    if xy_bounds is not None:
        bounds = np.asarray(xy_bounds, dtype=float)
        if bounds.shape != (2,2) or not np.all(np.isfinite(bounds)) or not np.all(bounds[0] < bounds[1]):
            raise ValueError('Legacy XY bounds must be finite and ordered')
        lo, hi = bounds
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_pixels:
            continue
        xyz = points[point_labels == label]
        reasons = []
        support = None
        if recover:
            selected = labels == label
            seeds = int(np.count_nonzero(np.asarray(foreground_gate)[selected]))
            fraction = seeds / len(xyz)
            support = dict(changed_pixels=seeds, measured_surface_pixels=len(xyz),
                           changed_fraction=fraction,
                           recovered_unknown_background_pixels=int(np.count_nonzero(~known_background[selected])),
                           min_seed_pixels=min_background_seed_pixels,
                           min_changed_fraction=min_background_support_fraction)
            if seeds < min_background_seed_pixels or fraction < min_background_support_fraction:
                reasons.append('insufficient_background_change_support')
        if np.any(border[labels == label]):
            reasons.append('touches_roi_or_exclusion_boundary')
        if xy_bounds is not None and np.any((xyz[:, :2] < lo) | (xyz[:, :2] > hi)):
            reasons.append('outside_target_xy_workspace')
        height = float(np.percentile(xyz[:, 3], 95))
        if height > max_height_mm:
            reasons.append('foreground_too_tall_or_occluder')
        rectangle = cv2.minAreaRect(xyz[:, :2].astype(np.float32))
        center, (width, length), angle = rectangle
        if width > length:
            width, length = length, width
            angle += 90
        angle = (angle + 90) % 180 - 90
        top = xyz[xyz[:, 3] >= height-3.]
        top_z = float(np.median(top[:, 2]))
        top_flat_fraction = float(np.mean(np.abs(xyz[:, 3]-height) <= 3.))
        interior = cv2.erode((labels == label).astype(np.uint8), np.ones((5,5), np.uint8))
        core = points[interior[pixels[:,1], pixels[:,0]] != 0]
        rows.append(dict(component_id=label, pixels=len(xyz), bbox_xywh=stats[label, :4].tolist(),
                         top_center_base_mm=[float(center[0]), float(center[1]), top_z],
                         visible_short_mm=float(width), visible_long_mm=float(length),
                         height_p95_mm=height, short_axis_yaw_deg=float(angle),
                         top_flat_fraction=top_flat_fraction,
                         yaw_ambiguous=bool(length <= width*1.1),
                         top_patch=top_patch_estimate(xyz, core),
                         background_support=support,
                         rejection_reasons=reasons, geometry_candidate=not reasons))
    return rows, mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='Existing aligned RGB-D capture; no live camera')
    parser.add_argument('--background', type=Path, help='Operator-reviewed empty-table capture with matching camera settings')
    parser.add_argument('--background-mode', choices=('pixels', 'supported-surface'), default='supported-surface',
                        help='With --background: recover measured surfaces across unknown background (default), or legacy strict pixel gate')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/tabletop_vision_lab.yaml', help='Perception-only camera/desk/ROI profile; explicit legacy cycle configs retain their old XY gate')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--exclude-roi', type=int, nargs=4, action='append', default=[], metavar=('X0','Y0','X1','Y1'), help='Manually reviewed image exclusion for fixed clamps/fixtures; repeatable')
    parser.add_argument('--min-height-mm', type=float, default=8., help='Depth foreground threshold; misses thin objects')
    parser.add_argument('--max-height-mm', type=float, default=200.)
    args = parser.parse_args(argv)
    if not 0 < args.min_height_mm < args.max_height_mm or not np.isfinite(args.max_height_mm):
        parser.error('Require finite 0 < min-height-mm < max-height-mm')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_tabletop_foreground_preview/v2', status='starting',
                  hardware_commands_sent=False, executable_target=False, candidates=[],
                  robot_feasibility_checked=False, config=str(args.config.resolve()),
                  capture=str(args.capture.resolve()), min_height_mm=args.min_height_mm,
                  background=None if args.background is None else str(args.background.resolve()),
                  background_mode=args.background_mode if args.background else 'none',
                  background_surface_thresholds=dict(min_seed_pixels=150, min_changed_fraction=.25),
                  max_height_mm=args.max_height_mm, exclusions=args.exclude_roi,
                  assumptions=['Offline preview only; never consumable as a robot target.',
                               'Static calibrated desk/camera; foreground threshold is not semantic recognition.',
                               'Hand and clamps can look like objects. Exclusions must be reviewed; no largest-component auto-selection.',
                               'Visible depth extents do not prove full object dimensions or graspability.',
                               'Thin/transparent/reflective/occluded objects may be missed or measured incorrectly.',
                               'Yaw is a visible rectangle axis, not the iterative planner yaw convention.'])
    report['assumptions'].append('Perception ROI is not the robot workspace; a detected object is not evidence of reachable or collision-free grasping.')
    report['assumptions'] += [
        'Optional background difference requires fixed camera/desk; current missing depth is never filled.',
        'supported-surface admits measured above-desk pixels with unknown background only in components with >=150 changed pixels and >=25% changed support; known unchanged pixels stay excluded.',
        'Surface support thresholds are provisional anti-speckle/weak-evidence assumptions, not a calibrated confidence or universal guarantee. Touching objects/fixtures can still merge.',
        'pixels mode retains the legacy strict background gate, excluding unknown background.',
        'Background threshold max(8 mm, 3 * temporal MAD); thin objects can be missed.',
        'top_patch assumes a near-horizontal top and rejects depth tails using eroded-core median/MAD; raw extents remain recorded.']
    result = 0
    try:
        import cv2
        import yaml
        from tools.run_vision_pick_cycle import validate_capture
        from tools.analyze_ur5_camera_desk_plane import deproject_aligned_depth, transform_points
        from tools.analyze_ur5_camera_colored_target import plane_z_at_xy_mm
        c, xy_bounds = load_perception_config(args.config)
        report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        report['vision_region'] = dict(image_roi=c['roi'], config_schema=c['schema'],
                                      legacy_target_xy_bounds_mm=None if xy_bounds is None else xy_bounds.tolist(),
                                      robot_workspace=False)
        check = validate_capture(c, args.capture)
        rgb = cv2.imread(str(args.capture/'camera_check_rgb.png'))
        depth = cv2.imread(str(args.capture/'camera_check_depth_u16.png'), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape != (720,1280,3) or depth.shape != (720,1280):
            raise ValueError('Expected calibrated 1280x720 RGB-D images')
        foreground_gate = known_background = None
        if args.background is not None:
            background_check = validate_capture(c, args.background)
            compatible_background(check, background_check)
            background_path = args.background/'camera_check_depth_u16.png'
            background_depth = cv2.imread(str(background_path), cv2.IMREAD_UNCHANGED)
            if background_depth is None:
                raise ValueError('Missing background depth')
            stats_path = args.background/'depth_statistics.npz'
            mad = None
            if stats_path.is_file():
                with np.load(stats_path, allow_pickle=False) as stats:
                    mad = stats['mad_mm']
            foreground_gate, known_background = background_foreground(
                depth, background_depth, check['camera']['depth']['scale_m_per_unit'], mad)
            x0,y0,x1,y1 = c['roi']
            report['background_support_fraction_in_roi'] = float(np.mean(known_background[y0:y1,x0:x1]))
            report['background_depth_sha256'] = hashlib.sha256(background_path.read_bytes()).hexdigest()
            cv2.imwrite(str(args.out/'background_change.png'), foreground_gate.astype(np.uint8)*255)
        for a,b,d,e in args.exclude_roi:
            if not 0 <= a < d <= 1280 or not 0 <= b < e <= 720:
                raise ValueError('Exclusion must lie inside the image')
        calibration = yaml.safe_load(Path(c['calibration']).read_text())
        plane = yaml.safe_load(Path(c['desk_plane']).read_text())
        intrinsics = calibration['selected_intrinsics']
        camera, pixels = deproject_aligned_depth(depth, intrinsics['camera_matrix'],
            intrinsics['distortion_coefficients'], check['camera']['depth']['scale_m_per_unit'],
            c['roi'], 1, (.05, 2.))
        xyz = transform_points(calibration['transforms']['base_from_camera']['matrix'], camera)*1000
        n = plane['plane']['normal_base']
        z0 = plane_z_at_xy_mm(n, c['desk_z_mm'], 0, 0)
        dx = plane_z_at_xy_mm(n, c['desk_z_mm'], 1, 0)-z0
        dy = plane_z_at_xy_mm(n, c['desk_z_mm'], 0, 1)-z0
        height = xyz[:,2]-(z0+dx*xyz[:,0]+dy*xyz[:,1])
        rows, mask = components(np.column_stack((xyz,height)), pixels, depth.shape, c['roi'],
            xy_bounds, min_height_mm=args.min_height_mm,
            max_height_mm=args.max_height_mm, exclusions=args.exclude_roi, foreground_gate=foreground_gate,
            known_background=known_background,
            background_mode=args.background_mode if args.background else 'pixels')
        if known_background is not None:
            for row in rows:
                x,y,w,h = row['bbox_xywh']
                row['background_support_fraction_in_bbox'] = float(np.mean(known_background[y:y+h,x:x+w]))
        valid = [r for r in rows if r['geometry_candidate']]
        report.update(candidates=rows, geometry_candidate_count=len(valid), roi=c['roi'],
                      status='single_foreground_needs_visual_review' if len(valid)==1 else 'ambiguous_or_missing_foreground',
                      calibration=c['calibration'], desk_plane=c['desk_plane'])
        overlay = rgb.copy()
        x0,y0,x1,y1 = c['roi']
        cv2.rectangle(overlay,(x0,y0),(x1-1,y1-1),(255,255,0),2)
        cv2.putText(overlay,'VISION ROI ONLY - NOT ROBOT WORKSPACE',(x0,y0-10 if y0>=20 else y0+20),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
        for row in rows:
            x,y,w,h = row['bbox_xywh']
            color = (0,200,0) if row['geometry_candidate'] else (0,0,255)
            cv2.rectangle(overlay, (x,y), (x+w,y+h), color, 2)
            cv2.putText(overlay, f"#{row['component_id']} {row['visible_short_mm']:.1f}x{row['visible_long_mm']:.1f} H{row['height_p95_mm']:.1f}mm", (x,max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            patch = row['top_patch']
            if patch is not None:
                cv2.putText(overlay, f"top patch {patch['short_mm']:.1f}x{patch['long_mm']:.1f} H{patch['height_median_mm']:.1f}",
                            (x,y+h+17), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
        for a,b,d,e in args.exclude_roi:
            cv2.rectangle(overlay, (a,b), (d,e), (255,0,255), 2)
        if not cv2.imwrite(str(args.out/'overlay.png'), overlay) or not cv2.imwrite(str(args.out/'foreground.png'), mask):
            raise OSError('Failed to save preview images')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        result = 1
    finally:
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        with (args.out/'summary.csv').open('w', newline='') as stream:
            keys = ['status','hardware_commands_sent','executable_target']
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerow({k:report[k] for k in keys})
        print(json.dumps(report, indent=2), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
