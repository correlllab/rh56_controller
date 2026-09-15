#!/usr/bin/env python3
"""Offline measured RGB-D crop for GGX; explicit visual component selection.

Default flat-top mode retains the known upright-box crop. Foreground mode
supports non-flat opaque objects without flattening or dimension-fitting points.
No hardware imports, learned segmentation or executable target output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

CROP_MODES = ('flat-top', 'foreground')
FOREGROUND_SCHEMA = 'rh56_ggx_measured_crop/v2'
SUPPORT_RADIUS_MM = 5.
SUPPORT_NEIGHBORS = 4
MIN_SUPPORT_FRACTION = .8


def known_dimensions(value):
    dims = np.asarray(value, dtype=float)
    if (dims.shape != (3,) or not np.all(np.isfinite(dims))
            or np.any(dims <= 0) or dims[0] > dims[1]):
        raise ValueError('Require positive finite SHORT <= LONG and HEIGHT dimensions')
    return dims


def selected_component_mask(mask, row, pixels):
    """Reuse the reviewed segmentation, not a bbox that could contain fixtures."""
    import cv2
    mask = np.asarray(mask)
    uv = np.asarray(pixels)
    if (mask.ndim != 2 or mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 255))
            or uv.ndim != 2 or uv.shape[1] != 2 or not np.all(np.isfinite(uv))
            or not np.all(uv == np.floor(uv))):
        raise ValueError('Invalid saved foreground mask/pixels')
    uv = uv.astype(int)
    if np.any(uv < 0) or np.any(uv[:, 0] >= mask.shape[1]) or np.any(uv[:, 1] >= mask.shape[0]):
        raise ValueError('Foreground pixels outside image')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    label = row['component_id']
    if (not isinstance(label, int) or not 0 < label < count
            or stats[label, :4].tolist() != row['bbox_xywh']
            or int(stats[label, cv2.CC_STAT_AREA]) != row['pixels']):
        raise ValueError('Saved foreground component no longer matches reviewed preview')
    keep = labels[uv[:, 1], uv[:, 0]] == label
    if int(keep.sum()) != row['pixels']:
        raise ValueError('Reviewed component contains missing/unmatched measured depth')
    return keep


def supported_surface(points_mm):
    """Delete isolated depth spikes only; never interpolate or rescale surfaces.

    Four neighbours within 5 mm is a provisional dense tabletop RGB-D quality
    assumption, not calibrated depth accuracy; thin/sparse details may be lost.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.all(np.isfinite(points)):
        raise ValueError('Insufficient/invalid measured foreground points')
    distance = cKDTree(points).query(points, k=SUPPORT_NEIGHBORS + 1)[0][:, -1]
    keep = distance <= SUPPORT_RADIUS_MM
    if keep.sum() < 100 or keep.mean() < MIN_SUPPORT_FRACTION:
        raise ValueError('Insufficient local depth support; do not fit sparse noise to object dimensions')
    return keep, dict(method='radius_support/v1', radius_mm=SUPPORT_RADIUS_MM,
        other_neighbors=SUPPORT_NEIGHBORS, minimum_retained_fraction=MIN_SUPPORT_FRACTION,
        input_points=len(points), removed_points=int((~keep).sum()), retained_points=int(keep.sum()))


def foreground_geometry(points_mm, dimensions_mm, normal_base, desk_z_mm, *, freeze_reference_yaw=False):
    """Separate observed object frame, operator dimensions and collision proxy.

    The proxy encloses supported measured points AND the supplied upright bounds.
    It is deliberately a box, not a reconstruction of bottle shoulders/neck.
    """
    import cv2
    from tools.analyze_ur5_camera_colored_target import plane_z_at_xy_mm
    dims = known_dimensions(dimensions_mm)
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.all(np.isfinite(points)):
        raise ValueError('Invalid supported foreground geometry')
    center, (width, length), angle = cv2.minAreaRect(points[:, :2].astype(np.float32))
    if width > length:
        width, length = length, width
        angle += 90
    ambiguous = length <= width * 1.1
    if width <= 0 or (ambiguous and not freeze_reference_yaw):
        raise ValueError('Foreground footprint yaw ambiguous; this mode requires a distinct long axis')
    yaw = float((angle + 90) % 180 - 90)
    origin = np.array([*center, plane_z_at_xy_mm(normal_base, desk_z_mm, *center)])
    if not np.all(np.isfinite(origin)) or np.any(points[:, 2] <= origin[2]):
        raise ValueError('Foreground is not above its desk-supported proxy base')
    a = np.deg2rad(yaw)
    u, v = np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0])
    delta = points - origin
    # Use exact supported points, not percentile clipping or nominal dimensions,
    # when enlarging the proxy. A tiny rounding pad covers float32 rectangle fit.
    observed = np.array([2 * np.abs(delta @ u).max(), 2 * np.abs(delta @ v).max(), delta[:, 2].max()])
    size = np.maximum(dims, observed + .001)
    warnings = []
    if np.any(np.abs(observed - dims) > 5.):
        warnings.append('Measured visible bounds differ from supplied dimensions by >5 mm; '
                        'points are NOT corrected. Inspect crop and simulation; depth may be biased.')
    frame = dict(center_base_mm=origin.tolist(), origin='desk_footprint_center',
                 short_axis_yaw_deg=yaw, yaw_ambiguous=bool(ambiguous), method='supported_foreground_xy_rectangle/v1')
    if freeze_reference_yaw:
        frame['yaw_policy'] = 'frozen_reference_axis'
        if ambiguous:
            warnings.append('Footprint yaw is ambiguous; freeze this initial enclosing-box axis, not a measured semantic yaw.')
    proxy = dict(shape='box', dimensions_mm=size.tolist(), dimensions_order='short_long_height',
                 method='enclose_supported_points_and_operator_bounds/v1',
                 observed_bounds_mm=observed.tolist(), operator_dimensions_mm=dims.tolist(),
                 assumption='Upright desk-supported object; box over-approximates curved surfaces. '
                            'Not a complete mesh or proof of hidden geometry, depth accuracy or grasp success.')
    return frame, proxy, warnings


def validate_foreground_report(report):
    """Reject incomplete new-mode reports instead of treating them as box tops."""
    if (report.get('schema') != FOREGROUND_SCHEMA or report.get('crop_mode') != 'foreground'
            or report.get('status') != 'measured_crop_ready_for_offline_inference'
            or report.get('hardware_contacted') is not False or report.get('executable_target') is not False):
        raise ValueError('Expected completed non-executable foreground crop v2')
    frame, proxy = report.get('object_frame', {}), report.get('collision_proxy', {})
    center = np.asarray(frame.get('center_base_mm'), dtype=float)
    if (center.shape != (3,) or not np.all(np.isfinite(center))
            or (frame.get('yaw_ambiguous') is not False and not (
                frame.get('yaw_ambiguous') is True and report.get('reference_yaw_frozen') is True
                and frame.get('yaw_policy') == 'frozen_reference_axis'))
            or not np.isfinite(frame.get('short_axis_yaw_deg', float('nan')))
            or frame.get('origin') != 'desk_footprint_center'
            or proxy.get('shape') != 'box'):
        raise ValueError('Invalid foreground frame/proxy')
    # Proxy axes stay attached to the measured footprint even if a supplied
    # dimension expands the first axis past the second.
    prior = known_dimensions(report.get('dimensions_prior_mm'))
    size = np.asarray(proxy.get('dimensions_mm'), dtype=float)
    observed = np.asarray(proxy.get('observed_bounds_mm'), dtype=float)
    if (size.shape != (3,) or observed.shape != (3,)
            or not np.all(np.isfinite(np.r_[size, observed]))
            or np.any(observed <= 0) or np.any(size < np.maximum(prior, observed))):
        raise ValueError('Foreground proxy does not enclose measured/prior bounds')
    return frame, size.tolist()


def crop_mask(points_mm, top_mm, yaw_deg, dimensions_mm, margin_mm=5.):
    points=np.asarray(points_mm,dtype=float);top=np.asarray(top_mm,dtype=float)
    dims=np.asarray(dimensions_mm,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or top.shape!=(3,) or dims.shape!=(3,)
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(top))
            or not np.all(np.isfinite(dims)) or np.any(dims<=0)
            or not np.isfinite(yaw_deg) or not np.isfinite(margin_mm) or not 0<=margin_mm<=10):
        raise ValueError('Invalid crop geometry')
    angle=np.deg2rad(yaw_deg);u=np.array([np.cos(angle),np.sin(angle),0]);v=np.array([-u[1],u[0],0])
    d=points-top
    return ((np.abs(d@u)<=dims[0]/2+margin_mm)&(np.abs(d@v)<=dims[1]/2+margin_mm)
            &(d[:,2]>=-dims[2]+8)&(d[:,2]<=margin_mm))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview',type=Path,required=True)
    p.add_argument('--component-id',type=int,required=True,help='Explicit visually reviewed object component')
    p.add_argument('--dimensions-mm',type=float,nargs=3,required=True,metavar=('SHORT','LONG','HEIGHT'))
    p.add_argument('--crop-mode', choices=CROP_MODES, default='flat-top',
                   help='flat-top: legacy known box; foreground: selected non-flat opaque object surface')
    p.add_argument('--margin-mm',type=float,default=5.)
    p.add_argument('--freeze-reference-yaw', action='store_true',
                   help='Foreground only: allow an ambiguous initial rectangle axis, freeze it for XY-only repeats')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.freeze_reference_yaw and a.crop_mode != 'foreground':
        p.error('--freeze-reference-yaw requires foreground mode')
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=FOREGROUND_SCHEMA if a.crop_mode == 'foreground' else 'rh56_ggx_measured_crop/v1',
        crop_mode=a.crop_mode,status='starting',hardware_contacted=False,
        executable_target=False,frame='ur_base',units='metres',source_preview=str(a.preview.resolve()),
        reviewed_component_id=a.component_id,dimensions_prior_mm=a.dimensions_mm,margin_mm=a.margin_mm,
        assumptions=['Manual top-component selection and known upright box dimensions; not universal segmentation.',
                     'Measured partial surfaces only; no hidden geometry completion or missing-depth fill.',
                     'Oriented XY crop uses measured top center/yaw and supplied dimensions plus margin.',
                     'Exclude lowest8mm above inferred box bottom and points more than margin above top.',
                     'Camera calibration reused; no robot reachability or collision certification.'])
    if a.crop_mode == 'foreground':
        report['assumptions'] = [
            'Manual selection of one isolated opaque upright object with a distinct long axis; not universal segmentation.',
            'Use only measured pixels in the reviewed foreground component; no bbox admission of other objects.',
            'No flat-top assumption, missing-depth fill, dimension-fitting, rescaling or hidden-surface synthesis.',
            'Reject isolated points without four neighbours within 5 mm; >=80% and >=100 points must remain. '
            'Provisional dense-depth assumption; thin/sparse details may be removed.',
            'Known dimensions are physical priors for a separate enclosing box collision proxy, not measured top-patch dimensions.',
            'Camera/desk calibration reused; depth errors and unknown obstacles remain possible. No hardware access.']
    rc=0
    try:
        import cv2
        from tools.preview_tabletop_target import load_perception_config
        from tools.run_vision_pick_cycle import validate_capture
        from tools.analyze_ur5_camera_desk_plane import deproject_aligned_depth,transform_points
        known_dimensions(a.dimensions_mm)
        source=json.loads(a.preview.read_text())
        if source.get('schema')!='rh56_tabletop_foreground_preview/v2' or source.get('executable_target') is not False:
            raise ValueError('Expected non-executable tabletop preview')
        rows=[r for r in source['candidates'] if r['component_id']==a.component_id]
        if len(rows)!=1 or not rows[0]['geometry_candidate'] or rows[0]['rejection_reasons']:
            raise ValueError('Explicit component is missing or rejected')
        patch=rows[0]['top_patch']
        if a.crop_mode == 'flat-top':
            if patch is None or patch['yaw_ambiguous']:raise ValueError('No unambiguous reviewed top')
            measured=np.array([patch['short_mm'],patch['long_mm'],patch['height_median_mm']])
            if not np.all(np.isfinite(measured)) or np.any(np.abs(measured-a.dimensions_mm)>5):
                raise ValueError(f'Reviewed flat top {measured.tolist()} mm inconsistent with '
                                 f'known dimensions {a.dimensions_mm} mm (5 mm tolerance). '
                                 'For non-flat objects use --crop-mode foreground; do not substitute top median for total height.')
        c,_=load_perception_config(Path(source['config']));capture=Path(source['capture'])
        check=validate_capture(c,capture);cal=yaml.safe_load(Path(c['calibration']).read_text());intr=cal['selected_intrinsics']
        depth_path=capture/'camera_check_depth_u16.png';depth=cv2.imread(str(depth_path),-1)
        rgb=cv2.imread(str(capture/'camera_check_rgb.png'))
        if depth is None or rgb is None or depth.shape != (720,1280) or rgb.shape != (720,1280,3):
            raise ValueError('Expected calibrated 1280x720 RGB-D images')
        xyz,pixels=deproject_aligned_depth(depth,intr['camera_matrix'],intr['distortion_coefficients'],
            check['camera']['depth']['scale_m_per_unit'],c['roi'],1,(.05,2.))
        base=transform_points(cal['transforms']['base_from_camera']['matrix'],xyz)
        if a.crop_mode == 'flat-top':
            keep=crop_mask(base*1000,patch['center_base_mm'],patch['short_axis_yaw_deg'],a.dimensions_mm,a.margin_mm)
        else:
            mask_path = a.preview.parent / 'foreground.png'
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != depth.shape:
                raise ValueError('Missing/mismatched reviewed foreground.png; regenerate perception')
            if hashlib.sha256(Path(source['config']).read_bytes()).hexdigest() != source['config_sha256']:
                raise ValueError('Perception configuration changed; regenerate preview')
            keep = selected_component_mask(mask, rows[0], pixels)
            supported, quality = supported_surface(base[keep] * 1000)
            selected_indices = np.flatnonzero(keep)
            keep[selected_indices[~supported]] = False
            plane = yaml.safe_load(Path(c['desk_plane']).read_text())
            frame, proxy, warnings = foreground_geometry(base[keep] * 1000, a.dimensions_mm,
                plane['plane']['normal_base'], c['desk_z_mm'], freeze_reference_yaw=a.freeze_reference_yaw)
            if a.freeze_reference_yaw:
                report['reference_yaw_frozen'] = True
                report['assumptions'][0] = 'One selected opaque object; initial box axes are frozen, not a semantic yaw estimate.'
            report.update(object_frame=frame, collision_proxy=proxy, surface_filter=quality,
                          warnings=warnings, margin_mm=None,
                          source_foreground_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())
        points=base[keep];uv=pixels[keep].astype(int)
        if len(points)<100:raise ValueError('Insufficient measured crop points')
        np.save(a.out/'object_points_ur_base_m.npy',points)
        np.save(a.out/'object_colors_rgb.npy',rgb[uv[:,1],uv[:,0],::-1])
        np.save(a.out/'object_pixels.npy',uv)
        overlay=rgb.copy();overlay[uv[:,1],uv[:,0]]=(.4*overlay[uv[:,1],uv[:,0]]+.6*np.array([0,255,0])).astype(np.uint8)
        cv2.putText(overlay,'MEASURED PARTIAL GGX INPUT - NOT EXECUTABLE',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        cv2.imwrite(str(a.out/'input_overlay.png'),overlay)
        report.update(status='measured_crop_ready_for_offline_inference',points=len(points),
            top_patch=patch,point_cloud=str((a.out/'object_points_ur_base_m.npy').resolve()),
            source_depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            calibration_sha256=hashlib.sha256(Path(c['calibration']).read_bytes()).hexdigest(),
            source_preview_sha256=hashlib.sha256(a.preview.read_bytes()).hexdigest(),
            observed_extent_mm=(np.ptp(points,axis=0)*1000).tolist(),
            observed_base_min_mm=(points.min(axis=0)*1000).tolist(),
            observed_base_max_mm=(points.max(axis=0)*1000).tolist())
        if a.crop_mode == 'foreground':
            validate_foreground_report(report)
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');rc=1
    finally:
        (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (a.out/'summary.csv').open('w',newline='') as f:
            keys=['status','hardware_contacted','executable_target','points'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerow({k:report.get(k) for k in keys})
        print(json.dumps(report,indent=2),flush=True)
    return rc


if __name__=='__main__':raise SystemExit(main())
