#!/usr/bin/env python3
"""Offline middle-tip pointing target and route preview; never commands hardware.

Uses measured RGB-D XY, the existing calibrated desk height and unchanged CB3 /
RH56 mount and fingertip site. No fitting against operator-measured errors.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np
import yaml

from tools import ggx_real_support as support, ggx_reset as reset
from tools import ggx_motion as motion, ggx_planning as planning
from tools import preflight_ggx_ur5_table as sim, run_ggx_real as real
from tools.run_ggx_review import pin, verify
from tools.preview_tabletop_target import load_perception_config
from tools.analyze_ur5_camera_desk_plane import deproject_aligned_depth, transform_points
from tools.analyze_ur5_camera_colored_target import plane_z_at_xy_mm
from rh56_controller.ur5_bridge import UR5Bridge


def site_transform(model, data, name):
    result = np.eye(4)
    site = model.site(name).id
    result[:3, :3] = data.site_xmat[site].reshape(3, 3)
    result[:3, 3] = data.site_xpos[site]
    return result


def measured_target(target, perception, height_mm):
    verify(target['source_sha256'])
    capture = Path(target['capture'])
    calibration = yaml.safe_load(Path(perception['calibration']).read_text())
    camera = yaml.safe_load((capture/'camera_check.yaml').read_text())['camera']
    intrinsics = calibration['selected_intrinsics']
    # Compare factory metadata to factory metadata; selected_intrinsics is the
    # existing fitted calibration used by this pipeline, not factory intrinsics.
    if (camera['serial'] != calibration['camera']['serial']
            or camera['stream'] != calibration['camera']['stream']
            or not np.allclose(calibration['camera']['factory_color_intrinsics']['camera_matrix'],
                camera['factory_color_intrinsics']['camera_matrix'], atol=1e-5, rtol=0)):
        raise ValueError('Capture camera/stream differs from frozen calibration')
    depth = cv2.imread(str(capture/'camera_check_depth_u16.png'), cv2.IMREAD_UNCHANGED)
    u, v = target['pixel_uv']
    xyz, pixels = deproject_aligned_depth(depth, intrinsics['camera_matrix'],
        intrinsics['distortion_coefficients'], camera['depth']['scale_m_per_unit'],
        [u-3, v-3, u+4, v+4], 1, (.05, 2.))
    if len(xyz) < 25:
        raise ValueError('Insufficient local depth at the selected grid point')
    # Median depth evaluated on the selected pixel's ray, not a shifted ROI centroid.
    ray = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64),
        np.asarray(intrinsics['camera_matrix']), np.asarray(intrinsics['distortion_coefficients'])).reshape(2)
    camera_xyz = np.r_[ray, 1.] * np.median(xyz[:, 2])
    base = transform_points(calibration['transforms']['base_from_camera']['matrix'], camera_xyz[None])[0]
    normal = yaml.safe_load(Path(perception['desk_plane']).read_text())['plane']['normal_base']
    desk_z = plane_z_at_xy_mm(normal, perception['desk_z_mm'], *(base[:2]*1000))/1000
    tip = np.r_[base[:2], desk_z + height_mm/1000]
    return dict(pixel_uv=[u, v], measured_rgbd_base_m=base.tolist(),
        depth_samples=len(xyz), depth_median_m=float(np.median(xyz[:, 2])),
        depth_mad_mm=float(np.median(np.abs(xyz[:, 2]-np.median(xyz[:, 2])))*1000),
        calibrated_desk_z_m=float(desk_z), raw_depth_minus_desk_mm=float((base[2]-desk_z)*1000),
        desired_tip_base_m=tip.tolist(), height_mm=height_mm,
        coordinate_policy='RGB-D XY on selected pixel ray; existing calibrated desk Z + requested vertical height')


def level_hand_target(model, qpos, tip_base, heading):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    current_hand = UR5Bridge.tcp_T_to_hand_T(site_transform(model, data, 'tcp'))
    tip_hand = (np.linalg.inv(current_hand) @ np.r_[data.site_xpos[model.site('right_middle_tip').id], 1.])[:3]
    # RH56 palm normal is local +X; finger extension is local +Z.
    x = np.array([0., 0., -1.])
    z = np.r_[np.asarray(heading, dtype=float), 0.]
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    hand = np.eye(4)
    hand[:3, :3] = np.column_stack((x, y, z))
    hand[:3, 3] = np.asarray(tip_base) - hand[:3, :3] @ tip_hand
    tcp = UR5Bridge.hand_T_to_tcp_T(hand)
    eeff = hand.copy()
    eeff[:3, 3] += hand[:3, :3] @ sim.EEFF_LOCAL_M
    return hand, tcp, eeff, tip_hand


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/reflex_raspberry_calibrated_plane_lab.yaml')
    parser.add_argument('--perception-config', type=Path, default=ROOT/'configs/tabletop_vision_lab.yaml')
    parser.add_argument('--height-mm', type=float, default=20.)
    parser.add_argument('--heading-xy', nargs=2, type=float, default=[0., -1.], help='Finger direction in base XY, default away from arm toward desk')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.height_mm) or not 0 < args.height_mm <= 300:
        parser.error('height must be finite in (0, 300] mm')
    if not np.isfinite(args.heading_xy).all() or np.linalg.norm(args.heading_xy) < 1e-9:
        parser.error('heading must be finite and nonzero')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_middle_tip_validation_plan/v1', status='preparing',
        hardware_motion_commands_sent=False, executable=False, assumptions=[
            'Offline only. Fresh controller IK/safety/path checks and explicit operator approval required before motion.',
            'Existing right_middle_tip site is the reference, not an independently calibrated physical pointer.',
            'All fingers at original open commands; actual hand state cannot be read back.',
            'Palm +X faces base -Z, fingers +Z extend along specified base XY heading.',
            'Desk only; camera mount, cables and unmodeled obstacles require operator review.',
            'Manual target selection excludes automatic segmentation error; one pose is not all-pose accuracy.'])
    samples = []
    code = 0
    try:
        target = json.loads(args.target.read_text())
        snapshot = json.loads(args.snapshot.read_text())
        config = real.load_config(args.config)
        perception, _ = load_perception_config(args.perception_config)
        measure = measured_target(target, perception, args.height_mm)
        source = reset.desk_source(config, perception)
        model = support.build_model(source, include_object=False)
        qpos = sim.initial_qpos(model, np.asarray(snapshot['q']))
        sim.set_hand_qpos(model, qpos, sim.GGX_OPEN_CTRL)
        hand, tcp, goal, tip_hand = level_hand_target(model, qpos, measure['desired_tip_base_m'], args.heading_xy)
        sim_args = sim.parse_args(['--no-video'])
        sim_args.ik_position_tol_mm = .05
        sim_args.ik_orientation_tol_deg = .05
        attempts = []
        report.update(measurement_target=measure, scene=source, snapshot=snapshot,
            hand_ctrl=dict(sim.GGX_OPEN_CTRL), hand_raw=support.raw_hand_command(model, 0.),
            desired_hand_T=hand.tolist(), desired_tcp=support.matrix_pose(tcp),
            reference_tip_in_hand_m=tip_hand.tolist(), heading_xy=args.heading_xy,
            config=str(args.config.resolve()), target=str(args.target.resolve()),
            source_sha256={**target['source_sha256'], **pin([args.target, args.snapshot, args.config,
                args.perception_config, perception['calibration'], perception['desk_plane'],
                __file__, support.__file__, sim.__file__, planning.__file__, reset.__file__,
                ROOT/'rh56_controller/ur5_bridge.py', *list((ROOT/'h1_mujoco/inspire').glob('*.xml'))])},
            ik_attempts=attempts)
        # First inspect the requested endpoint; no height/pose changes to force clearance.
        solved = next(planning.solutions(model, goal, sim_args, attempts, hints=[qpos]), None)
        if solved is None:
            raise ValueError('Requested level middle-tip target has no nominal IK solution')
        solved = solved[0] if isinstance(solved, tuple) else solved
        sim.set_hand_qpos(model, solved, sim.GGX_OPEN_CTRL)
        data = mujoco.MjData(model)
        data.qpos[:] = solved
        mujoco.mj_forward(model, data)
        gap, nearest = sim.desk_clearance(model, data)
        report.update(goal_q=solved[planning.arm_addresses(model)].tolist(),
            endpoint_minimum_desk_clearance_mm=float(gap*1000), endpoint_nearest_geom=nearest,
            nominal_tip_base_m=data.site_xpos[model.site('right_middle_tip').id].tolist())
        endpoint = sim.TrajectorySample('pointer_endpoint', 1., solved, 0.,
            min_desk_clearance_m=gap, nearest_geom=nearest,
            desk_collision=motion.desk_clearance_blocked(gap, motion.reflex_desk_clearance_m(config)))
        samples = [endpoint]*30
        if motion.desk_clearance_blocked(gap, motion.reflex_desk_clearance_m(config)):
            report['status'] = 'endpoint_desk_rejected'
            code = 2
        else:
            # Diagnostic only: exact joint interpolation, not yet controller-approved.
            arm = planning.arm_addresses(model)
            count = max(2, int(np.ceil(np.max(np.abs(solved[arm]-qpos[arm]))/.01))+1)
            samples = [sim.TrajectorySample('pointer_transit', float(t), (1-t)*qpos+t*solved, 0.) for t in np.linspace(0, 1, count)]
            minimum, nearest, collision = sim.check_trajectory(model, samples, required_clearance_m=max(1e-9, motion.reflex_desk_clearance_m(config)))
            report.update(path_minimum_desk_clearance_mm=float(minimum*1000), path_nearest_geom=nearest)
            report['status'] = 'nominal_path_rejected' if collision else 'nominal_path_passed_needs_controller_review'
            code = 2 if collision else 0
        if args.video:
            result = sim.CandidateResult(0, 'Middle tip A', 0., code==0, report['status'], None, None, float(gap*1000), nearest, '', len(samples))
            sim_args.camera_distance = 1.1
            sim.write_video(args.out/'pointer_preview.mp4', model, samples, result, sim_args, np.asarray(measure['desired_tip_base_m']))
        sim._write_csv(args.out/'trajectory.csv', sim._trajectory_rows(model, samples))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        code = 1
    finally:
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','ik_attempts')}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
