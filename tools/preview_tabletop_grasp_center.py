#!/usr/bin/env python3
"""Draw an existing Plan grasp center on saved RGB; offline diagnostic only."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plan_geometry(patch, depth_mm):
    """Use original Plan geometry, without lab cube empirical corrections."""
    from rh56_controller.grasp_viz_core import GraspVizCore
    from rh56_controller.vision_grasp_target import VisionGraspTarget

    top = np.asarray(patch['center_base_mm'], dtype=float)
    width = float(patch['short_mm'])
    height = float(patch['height_median_mm'])
    short_yaw = float(patch['short_axis_yaw_deg'])
    if (top.shape != (3,) or not np.all(np.isfinite(top))
            or not np.all(np.isfinite([width, height, short_yaw, depth_mm]))
            or width <= 0 or not 0 < depth_mm < height):
        raise ValueError('Require finite center, positive width, and depth inside object')
    if patch.get('yaw_ambiguous', True):
        raise ValueError('Short-axis orientation is ambiguous; no automatic yaw preview')
    center = top / 1000 - np.array([0, 0, depth_mm / 1000])
    target = VisionGraspTarget('reviewed_tabletop_surface_offline_only', tuple(center),
                              width, 0., depth_mm, 0.)
    core = GraspVizCore(vision_target=target, mink_viz=False, hand_write_only=True)
    try:
        final = core.closure.solve(core._mode, target.planner_width_m)
        native = final.world_tips(center_policy='antipodal')
        opposing = np.mean([p for name, p in native.items() if name != 'thumb'], axis=0)
        direction = native['thumb'] - opposing
        # Align actual thumb/opposing-centroid direction, not an assumed GUI yaw.
        yaw = np.deg2rad(short_yaw) - np.arctan2(direction[1], direction[0])
        target = replace(target, yaw_deg=float(np.rad2deg(yaw)))
        core._plane_rz = yaw
        closures = core._compute_plan_closures(10., final)
        poses = [core._build_world_T_hand(r) for r in closures]
        for r, T in zip(closures, poses):
            actual = T[:3,:3] @ r.grasp_center('antipodal') + T[:3,3]
            if not np.allclose(actual, center, rtol=0, atol=1e-9):
                raise RuntimeError('Existing Plan did not preserve the grasp center')
        T = poses[-1]
        tips = {name: T[:3,:3] @ p + T[:3,3] for name, p in final.tip_positions.items()}
        opposite = np.mean([p for name, p in tips.items() if name != 'thumb'], axis=0)
        return dict(strategy='original Plan, not GGX', input=target.metadata(),
                    measured_top_center_base_mm=top.tolist(),
                    grasp_center_base_mm=(center*1000).tolist(),
                    thumb_tip_base_mm=(tips['thumb']*1000).tolist(),
                    opposing_finger_centroid_base_mm=(opposite*1000).tolist(),
                    finger_tips_base_mm={k:(v*1000).tolist() for k,v in tips.items()},
                    requested_width_mm=width, achieved_solver_width_mm=final.width*1000,
                    projected_opposition_xy_span_mm=float(np.linalg.norm((tips['thumb']-opposite)[:2])*1000),
                    solver_width_residual_mm=final.width*1000-width,
                    planner_width_range_mm=[v*1000 for v in core._width_range],
                    final_tilt_deg=final.tilt_deg, waypoint_count=len(poses),
                    geometric_hand_poses_base_m=[T.tolist() for T in poses],
                    all_waypoint_centers_verified=True)
    finally:
        core.cleanup()


def project_base_mm(points_mm, calibration):
    from rh56_controller.camera_calibration import invert_transform, project_target_points
    points = np.asarray(points_mm, dtype=float) / 1000
    T = invert_transform(np.asarray(calibration['transforms']['base_from_camera']['matrix']))
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('Expected finite Nx3 base points')
    if np.any((points @ T[:3,:3].T + T[:3,3])[:,2] <= 0):
        raise ValueError('Cannot project points behind the camera')
    intr = calibration['selected_intrinsics']
    return project_target_points(points, T, intr['camera_matrix'], intr['distortion_coefficients'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True, help='Saved tabletop preview summary.json (v2)')
    parser.add_argument('--component-id', type=int, required=True, help='Explicit visually reviewed component; never selects the largest')
    parser.add_argument('--grasp-depth-mm', type=float, default=10., help='Diagnostic center depth below top in base Z; default 10 mm')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_tabletop_grasp_center_diagnostic/v1', status='starting',
                  executable_target=False, hardware_contacted=False,
                  source_preview=str(args.preview.resolve()), component_id=args.component_id,
                  assumptions=['Saved scene; not a live camera or robot state.',
                               'Original Plan antipodal center, not wrist/TCP and not a GGX prediction.',
                               'Near-horizontal rectangular top; grasp depth is a preview assumption.',
                               'Short axis has a 180 degree ambiguity; selected thumb side is not collision-validated.',
                               'No cube empirical width/placement offset, mount correction, force or motion commands.',
                               'Source workspace rejection remains effective; no hardware IK, contact or collision validation.',
                               'Solver achieved width can differ from input; residual is reported, not compensated.'])
    result = 0
    try:
        import cv2
        import yaml
        source = json.loads(args.preview.read_text())
        if source.get('schema') != 'rh56_tabletop_foreground_preview/v2' or source.get('executable_target') is not False:
            raise ValueError('Expected non-executable tabletop preview v2')
        rows = [r for r in source['candidates'] if r['component_id'] == args.component_id]
        if len(rows) != 1 or rows[0]['top_patch'] is None:
            raise ValueError('Selected component must have a measured top patch')
        row = rows[0]
        report['source_rejection_reasons'] = row['rejection_reasons']
        if set(row['rejection_reasons']) - {'outside_target_xy_workspace'}:
            raise ValueError('Cannot draw a planned center from weak, occluded or truncated evidence')
        report['plan'] = plan_geometry(row['top_patch'], args.grasp_depth_mm)
        plan = report['plan']
        calibration = yaml.safe_load(Path(source['calibration']).read_text())
        keys = ['measured_top_center_base_mm', 'grasp_center_base_mm',
                'thumb_tip_base_mm', 'opposing_finger_centroid_base_mm']
        uv = project_base_mm([plan[k] for k in keys], calibration)
        report['projected_pixels'] = dict(zip(keys, uv.tolist()))
        rgb_path = Path(source['capture'])/'camera_check_rgb.png'
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise ValueError('Missing saved RGB image')
        if np.any(uv < 0) or np.any(uv >= [rgb.shape[1],rgb.shape[0]]):
            raise ValueError('Projected markers lie outside saved RGB image')
        top, center, thumb, opposite = [tuple(p) for p in np.rint(uv).astype(int)]
        cv2.line(rgb, thumb, opposite, (255,255,0), 2, cv2.LINE_AA)
        cv2.circle(rgb, top, 9, (0,255,255), 2, cv2.LINE_AA)
        cv2.drawMarker(rgb, center, (0,0,255), cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)
        for p, label in ((thumb,'T'), (opposite,'F')):
            cv2.circle(rgb, p, 5, (255,255,0), -1, cv2.LINE_AA)
            cv2.putText(rgb,label,(p[0]+9,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,0),2,cv2.LINE_AA)
        lines = ['OFFLINE ORIGINAL PLAN - NOT EXECUTABLE',
                 f'Red +: grasp center ({args.grasp_depth_mm:g} mm below top); yellow: top center',
                 'Cyan: modeled thumb (T) to opposing-finger centroid (F)',
                 f'Width input {plan["requested_width_mm"]:.1f} mm; solver achieved {plan["achieved_solver_width_mm"]:.1f} mm',
                 'Source rejection: '+(', '.join(row['rejection_reasons']) or 'none; still no robot preflight')]
        cv2.rectangle(rgb,(0,rgb.shape[0]-145),(rgb.shape[1]-1,rgb.shape[0]-1),(25,25,25),-1)
        for i,line in enumerate(lines):
            cv2.putText(rgb,line,(15,rgb.shape[0]-120+26*i),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1,cv2.LINE_AA)
        if not cv2.imwrite(str(args.out/'grasp_center.png'),rgb):
            raise OSError('Failed to save grasp-center image')
        report.update(status='offline_geometry_only', rgb_source=str(rgb_path))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        result = 1
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (args.out/'summary.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['status','executable_target','hardware_contacted'])
            writer.writeheader()
            writer.writerow({k:report[k] for k in writer.fieldnames})
        print(json.dumps(report,indent=2,allow_nan=False))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
