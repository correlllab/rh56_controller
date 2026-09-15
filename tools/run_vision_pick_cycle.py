#!/usr/bin/env python3
"""One supervised open -> scan pose -> detect -> original Plan -> lift cycle.

Default: offline preview using --capture. --record-start reads UR pose only.
--preflight-only checks controller IK without hand connections or movement;
it requires explicit confirmation to start/stop the RTDE control script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'magpie_control/src'))

CONFIRMATION = 'OPEN MOVE DETECT GRASP LIFT'
RESET_CONFIRMATION = 'OBJECT SUPPORTED AND RESET PATH CLEAR'
STAGES = ('prepare', 'open_hand', 'move_scan', 'detect', 'plan',
          'preflight_pick', 'grasp', 'settle', 'lift')
STATIONARY_SPEED_NORM_LIMIT = .001  # Preserve the existing six-component speed gate.


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def finite(value, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or (shape is not None and a.shape != shape):
        raise ValueError(f'Expected finite data with shape {shape}')
    return a


def grasp_placement_offset(c):
    """Explicit horizontal UR-base translation, separate from measured target/width."""
    xy = finite(c.get('grasp_xy_offset_base_mm', [0., 0.]), (2,))
    if np.linalg.norm(xy) > 10:
        raise ValueError('grasp_xy_offset_base_mm must have total magnitude <= 10 mm')
    return np.r_[xy, 0.] / 1000


def load_config(path):
    c = yaml.safe_load(path.read_text())
    if not isinstance(c, dict) or c.get('schema') != 'rh56_vision_pick_cycle_config/v1':
        raise ValueError('Expected rh56_vision_pick_cycle_config/v1')
    if 'hover_mm' in c or 'max_scan_to_hover_mm' in c:
        raise ValueError('Extra hover removed: delete hover_mm/max_scan_to_hover_mm and review '
                         'the direct scan-to-Plan-approach route with max_scan_to_approach_mm')
    c.setdefault('max_scan_to_approach_mm', c['max_reposition_mm'])
    c.setdefault('stationary_wait_grace_s', 3.)
    c['grasp_xy_offset_base_mm'] = (grasp_placement_offset(c)[:2]*1000).tolist()
    for key in ('calibration', 'desk_plane', 'start_pose', 'local_alignment', 'alignment_validation'):
        p = Path(c[key])
        c[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ('ur_ip', 'hand_port', 'camera_serial'):
        if not isinstance(c[key], str) or not c[key]:
            raise ValueError(f'{key} must be a nonempty string')
    ranges = {
        'object_width_mm': (1, 150), 'object_height_mm': (1, 200),
        'width_offset_mm': (-50, 50), 'closure_margin_mm': (0, 2),
        'grasp_depth_mm': (.1, 199), 'yaw_deg': (-180, 180), 'desk_z_mm': (-100, 300),
        'lift_mm': (1, 100),
        'open_settle_s': (.3, 10), 'close_settle_s': (.3, 10),
        'stationary_wait_grace_s': (.1, 5),
        'max_reposition_mm': (1, 300), 'max_plan_segment_mm': (1, 50),
        'max_scan_to_approach_mm': (1, 300),
        'max_rotation_deg': (1, 90), 'min_open_tcp_z_mm': (100, 600),
        'max_height_error_mm': (.1, 5), 'max_center_spread_mm': (.1, 5),
        'max_depth_mad_mm': (.1, 5), 'max_capture_age_s': (1, 30),
    }
    for key, (lo, hi) in ranges.items():
        # Explicit null disables ONLY the scan -> original Plan approach cap.
        # Missing keys retain the legacy fallback; other limits cannot be null.
        if key == 'max_scan_to_approach_mm' and c[key] is None:
            continue
        x = float(finite(c[key], ()))
        if isinstance(c[key], bool) or not lo <= x <= hi:
            raise ValueError(f'{key} must be in [{lo}, {hi}]')
        c[key] = x
    if not 0 < c['grasp_depth_mm'] < c['object_height_mm']:
        raise ValueError('Grasp depth must lie inside the object')
    if c['object_width_mm'] + c['width_offset_mm'] - c['closure_margin_mm'] <= 0:
        raise ValueError('Effective planner width must be positive')
    for key, lo, hi in (('hand_force_raw', 0, 1000), ('frames', 2, 5), ('erosion_px', 1, 15)):
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(f'{key} must be an integer in [{lo}, {hi}]')
    if c['erosion_px'] % 2 != 1 or c['speed_profile'] not in ('slow', 'planner'):
        raise ValueError('Invalid erosion or speed profile')
    for key in ('tcp_bounds_mm', 'target_bounds_mm'):
        b = finite(c[key], (2, 3))
        if not np.all(b[0] < b[1]):
            raise ValueError(f'Invalid {key}')
    low, high = finite(c['hsv_lower'], (3,)), finite(c['hsv_upper'], (3,))
    if (not np.all(low >= 0) or not np.all(low <= high) or low[0] >= high[0]
            or high[0] > 179 or np.max(high[1:]) > 255
            or not np.array_equal(high[1:], [255, 255])
            or not np.all(low == low.astype(int)) or not np.all(high == high.astype(int))):
        raise ValueError('Invalid HSV thresholds')
    c['hsv_lower'], c['hsv_upper'] = low.astype(int).tolist(), high.astype(int).tolist()
    roi = finite(c['roi'], (4,))
    if not (np.all(roi == roi.astype(int)) and 0 <= roi[0] < roi[2] <= 1280
            and 0 <= roi[1] < roi[3] <= 720):
        raise ValueError('ROI must lie inside the calibrated 1280x720 image')
    c['roi'] = roi.astype(int).tolist()
    return c


def within(point, bounds, name):
    p, b = finite(point, (3,)), finite(bounds, (2, 3))
    if np.any(p < b[0]) or np.any(p > b[1]):
        raise RuntimeError(f'{name} outside reviewed bounds: {p.tolist()}')


def lift_pose(tcp, distance_mm):
    result = finite(tcp, (6,)).copy()
    if not np.isfinite(distance_mm) or not 0 < distance_mm <= 100:
        raise ValueError('Lift must be in (0, 100] mm')
    result[2] += distance_mm / 1000
    return result


def pick_segments(c, plan):
    # The original executor owns approach + closure; do not insert a waypoint.
    goals = [*plan.tcp, plan.lift]
    limits = [c['max_scan_to_approach_mm']]
    limits += [c['max_plan_segment_mm']]*(len(plan.tcp)-1) + [c['lift_mm']]
    return goals, limits


def reset_pose_for_scan(c, current_tcp, scan_tcp, enabled=False):
    """Optional open-then-vertical-retreat from a reviewed, supported-object pose.

    Never used implicitly, and never inserted into the Plan's grasp approach.
    The 150 mm reset cap is separate from the final grasp lift's 100 mm cap.
    """
    current, scan = finite(current_tcp, (6,)), finite(scan_tcp, (6,))
    within(current[:3]*1000, c['tcp_bounds_mm'], 'Entry TCP')
    within(scan[:3]*1000, c['tcp_bounds_mm'], 'Scan TCP')
    if scan[2]*1000 < c['min_open_tcp_z_mm']:
        raise RuntimeError('Saved scan pose is below the reviewed elevated entry region')
    if current[2]*1000 >= c['min_open_tcp_z_mm']:
        return None
    if not enabled:
        raise RuntimeError('Low entry requires explicit supported-object/open-path reset confirmation')
    reset = current.copy()
    reset[2] = scan[2]
    if not 0 < (reset[2]-current[2])*1000 <= 150:
        raise RuntimeError('Low-entry vertical reset exceeds the 150 mm cap')
    within(reset[:3]*1000, c['tcp_bounds_mm'], 'Reset TCP')
    return reset.tolist()


def scan_segments(c, scan, reset=None):
    if reset is None:
        return [scan], [c['max_reposition_mm']]
    return [reset, scan], [150., c['max_reposition_mm']]


def normal(recv):
    if recv.getRobotMode() != 7 or recv.getSafetyMode() != 1 or recv.isProtectiveStopped():
        raise RuntimeError('UR5 is not in normal running/safety state')


def stationary_snapshot(recv):
    normal(recv)
    if np.linalg.norm(finite(recv.getActualTCPSpeed(), (6,))) > STATIONARY_SPEED_NORM_LIMIT:
        raise RuntimeError('UR5 must be stationary')
    return dict(time_s=time.time(), tcp=finite(recv.getActualTCPPose(), (6,)).tolist(),
                q=finite(recv.getActualQ(), (6,)).tolist(), robot_mode=7, safety_mode=1)


def check_stationary_origin(expected, current):
    """Bound read-to-read drift separately from the commanded segment length.

    The 0.1 mm / 0.1 degree envelope is an operational stationarity tolerance,
    not absolute calibration accuracy. The recorded static TCP span was ~0.05 mm.
    """
    from tools.run_vision_plan_trial import rot_error
    a, b = finite(expected['tcp'], (6,)), finite(current['tcp'], (6,))
    position_mm = float(np.linalg.norm(a[:3]-b[:3])*1000)
    rotation_deg = float(np.rad2deg(rot_error(a, b)))
    joint_delta = float(np.max(np.abs(finite(expected['q'], (6,))-finite(current['q'], (6,)))))
    if position_mm > .1 or rotation_deg > .1 or joint_delta > .01:
        raise RuntimeError(f'Preflight start drifted: {position_mm:.6f} mm, '
                           f'{rotation_deg:.6f} deg, joint delta {joint_delta:.6f} rad')
    return dict(position_mm=position_mm, rotation_deg=rotation_deg, joint_delta_rad=joint_delta)


def validate_capture(c, capture):
    calibration = yaml.safe_load(Path(c['calibration']).read_text())
    check = yaml.safe_load((capture / 'camera_check.yaml').read_text())
    if calibration.get('schema') != 'rh56_ur5_external_camera_calibration/v2':
        raise ValueError('Expected the existing calibrated eye-to-hand transform')
    if check.get('schema') != 'rh56_realsense_camera_check/v1':
        raise ValueError('Expected an aligned RGB-D capture')
    for data in (calibration, check):
        if (str(data['camera']['serial']) != c['camera_serial']
                or data['camera']['stream']['width'] != 1280
                or data['camera']['stream']['height'] != 720):
            raise RuntimeError('Camera identity/resolution differs from calibration')
    from tools.run_vision_plan_trial import validate_rigid_transform
    validate_rigid_transform(calibration['transforms']['base_from_camera']['matrix'])
    finite(calibration['selected_intrinsics']['camera_matrix'], (3, 3))
    return check


def detect_saved(c, capture, out):
    """Reuse the tested HSV/depth detector; reject ambiguous/partial targets."""
    import cv2
    from tools import analyze_ur5_camera_colored_target as detector
    validate_capture(c, capture)
    image = cv2.imread(str(capture / 'camera_check_rgb.png'))
    if image is None or image.shape != (720, 1280, 3):
        raise RuntimeError('Capture has no calibrated-size color image')
    x0, y0, x1, y1 = c['roi']
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(c['hsv_lower'], np.uint8), np.array(c['hsv_upper'], np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = sorted((int(s[cv2.CC_STAT_AREA]) for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 500), reverse=True)
    if not areas or (len(areas) > 1 and areas[1] >= .5 * areas[0]):
        raise RuntimeError('Target missing or ambiguous; no automatic selection/retry')
    args = detector.parse_args([
        '--calibration', c['calibration'], '--capture', str(capture), '--out', str(out),
        '--desk-plane', c['desk_plane'], '--desk-z-mm', str(c['desk_z_mm']),
        '--target-height-mm', str(c['object_height_mm']),
        '--max-height-error-mm', str(c['max_height_error_mm']),
        '--hue-min', str(c['hsv_lower'][0]), '--hue-max', str(c['hsv_upper'][0]),
        '--min-saturation', str(c['hsv_lower'][1]), '--min-value', str(c['hsv_lower'][2]),
        '--erosion-px', str(c['erosion_px']), '--roi', *map(str, c['roi'])])
    # Existing detector supports S/V lower bounds and full upper bounds only.
    if c['hsv_upper'][1:] != [255, 255]:
        raise ValueError('Existing detector requires HSV upper S/V = 255')
    detector.analyze(args)
    path = out / 'target_validation.yaml'
    report = yaml.safe_load(path.read_text())
    if report['quality_pass'] is not True:
        raise RuntimeError('Target height/depth quality check failed')
    if report['depth_foreground']['mad_mm'] > c['max_depth_mad_mm']:
        raise RuntimeError('Target depth is too noisy')
    x, y, w, h = report['segmentation']['bbox_xywh']
    if x <= x0 or y <= y0 or x+w >= x1 or y+h >= y1:
        raise RuntimeError('Target touches ROI boundary; center may be truncated')
    within(report['estimated_target_top_center_base_mm'], c['target_bounds_mm'], 'Target')
    return path, report


def stable_target(reports, max_spread_mm):
    points = finite([r['estimated_target_top_center_base_mm'] for r in reports])
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError('At least two target measurements are required')
    spread = float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=-1)))
    if spread > max_spread_mm:
        raise RuntimeError(f'Target moved or detection unstable: spread {spread:.2f} mm')
    return spread


class Plan:
    def __init__(self, c, target_path):
        from rh56_controller.vision_grasp_target import load_vision_target
        from rh56_controller.grasp_viz_core import GraspVizCore
        from rh56_controller.grasp_executor import GraspExecutor
        from tools.run_vision_plan_trial import load_local_alignment, command_hand_pose, pose, UR5Bridge
        offset = c['width_offset_mm'] - c['closure_margin_mm']
        target = load_vision_target(target_path, object_width_mm=c['object_width_mm'],
                                   width_offset_mm=offset, yaw_deg=c['yaw_deg'],
                                   depth_below_top_mm=c['grasp_depth_mm'])
        self.core = GraspVizCore(vision_target=target, mink_viz=False, hand_write_only=True)
        try:
            lo, hi = self.core.closure.width_range(self.core._mode)
            if not lo <= target.planner_width_m <= hi:
                raise ValueError('Requested width lies outside the existing planner range')
            # Preserve the strict existing frozen-shape loader. This new workflow
            # explicitly records and gates any narrower-shape extrapolation.
            validation = yaml.safe_load(Path(c['alignment_validation']).read_text())
            anchor_command = validation['hand_command_assumed']
            mount, alignment = load_local_alignment(Path(c['local_alignment']),
                                                    Path(c['alignment_validation']), anchor_command)
            final = self.core.closure.solve(self.core._mode, target.planner_width_m)
            closures = self.core._compute_plan_closures(10, final)
            placement = grasp_placement_offset(c)
            self.waypoints = []
            for r in closures:
                T = command_hand_pose(self.core._build_world_T_hand(r), mount).copy()
                # Translate every approach/closure pose equally in UR base.
                # Do not edit the measured cube, camera calibration or hand model.
                T[:3, 3] += placement
                self.waypoints.append((T, r))
            mapper = GraspExecutor(None, None, self.core.fk, hand_write_only=True)
            self.commands = [mapper._ctrl_to_real(r.ctrl_values) for r in closures]
            final_yaw = self.commands[-1][5]
            for cmd in self.commands:
                cmd[5] = final_yaw
            if self.commands[0] != [1000, 1000, 1000, 1000, 1000, 0]:
                raise RuntimeError('This lab cycle requires the tested opposing-thumb open approach')
            self.tcp = [pose(UR5Bridge.hand_T_to_tcp_T(T)).tolist() for T, _ in self.waypoints]
            self.lift = lift_pose(self.tcp[-1], c['lift_mm']).tolist()
            for p in [*self.tcp, self.lift]:
                within(np.array(p[:3])*1000, c['tcp_bounds_mm'], 'Planned TCP')
            self.report = dict(schema='rh56_vision_cycle_plan/v1', target=target.metadata(),
                               empirical_width_offset_mm=c['width_offset_mm'],
                               closure_margin_mm=c['closure_margin_mm'],
                               grasp_xy_offset_base_mm=(placement[:2]*1000).tolist(),
                               commanded_grasp_center_base_m=(np.array(target.center_base_m)+placement).tolist(),
                               placement_offset_scope='Constant horizontal UR-base translation of all Plan poses; not camera calibration or width margin',
                               alignment=alignment, alignment_anchor_command=anchor_command,
                               final_shape_extrapolated=self.commands[-1] != anchor_command,
                               hand_commands=self.commands, tcp_waypoints=self.tcp,
                               entry_policy='direct_to_original_plan_approach', lift_tcp=self.lift,
                               hardware_ik_validated=False, collision_free_verified=False)
        except BaseException:
            self.core.cleanup()
            raise

    def close(self):
        self.core.cleanup()


def run_cycle(backend, report, checkpoint=None):
    """No retries or release. Any exception prevents all subsequent stages."""
    value = None
    for stage in STAGES:
        report['stage'] = stage
        report['events'].append({'stage': stage, 'event': 'start', 'time_s': time.time()})
        if checkpoint is not None:
            checkpoint()
        print(f'Cycle: {stage}', flush=True)
        value = getattr(backend, stage)(value)
        report['events'].append({'stage': stage, 'event': 'complete', 'time_s': time.time()})
        if checkpoint is not None:
            checkpoint()
    report['status'] = 'cycle_commands_completed_grasp_unverified'


class LiveCycle:
    def __init__(self, c, args, report):
        self.c, self.args, self.report = c, args, report
        self.bridge = self.hand = self.plan_data = self.serial_log = self.executor = None
        self.cancelled = threading.Event()
        self.serial_errors, self.angle_commands = [], []
        self.telemetry = None

    def prepare(self, _):
        from tools.run_vision_plan_trial import CheckedBridge, pose, rot_error
        from tools.calibrate_ur5_external_camera_real import _open_rtde_receiver
        start = json.loads(Path(self.c['start_pose']).read_text())
        if start.get('schema') != 'rh56_vision_pick_start/v1' or start['ur_ip'] != self.c['ur_ip']:
            raise ValueError('Record a start pose for this UR5 first')
        self.scan = finite(start['tcp'], (6,)).tolist()
        within(np.array(self.scan[:3])*1000, self.c['tcp_bounds_mm'], 'Scan TCP')
        receiver = _open_rtde_receiver(self.c['ur_ip'])
        try:
            snapshot = stationary_snapshot(receiver)
        finally:
            receiver.disconnect()
        self.reset_tcp = reset_pose_for_scan(self.c, snapshot['tcp'], self.scan,
                                            self.args.reset_from_low)
        self.report['low_entry_reset'] = dict(enabled=self.args.reset_from_low,
            required=self.reset_tcp is not None, target_tcp=self.reset_tcp,
            vertical_cap_mm=150, supported_object_and_clear_path_confirmed=self.args.reset_from_low)
        # Control is constructed only after receive-only entry checks succeed.
        owner = self
        class GuardedBridge(CheckedBridge):
            def move_to_hand_pose(self, T, **kwargs):
                if owner.cancelled.is_set():
                    raise RuntimeError('Cycle cancelled')
                target = pose(self.hand_T_to_tcp_T(T))
                if owner.next_goal >= len(owner.allowed_goals):
                    raise RuntimeError('Unexpected arm command')
                expected = owner.allowed_goals[owner.next_goal]
                if np.linalg.norm(target[:3]-expected[:3]) > 1e-7 or rot_error(target, expected) > 1e-6:
                    raise RuntimeError('Arm command differs from preflight')
                owner.next_goal += 1
                return super().move_to_hand_pose(T, **kwargs)
        speed, accel, self.hand_speed = ((.1, .2, 1000) if self.c['speed_profile'] == 'planner' else (.01, .02, 100))
        self.speed = speed
        self.bridge = GuardedBridge(ip=self.c['ur_ip'], speed=speed, accel=accel)
        self.report['controller_script_requested'] = True
        if not self.bridge.connect():
            raise RuntimeError('UR5 connection failed')
        actual = stationary_snapshot(self.bridge._iface.recv)
        if (np.linalg.norm(np.array(actual['tcp'][:3])-snapshot['tcp'][:3]) > .001
                or rot_error(actual['tcp'], snapshot['tcp']) > np.deg2rad(.5)
                or np.max(np.abs(np.array(actual['q'])-snapshot['q'])) > .01):
            raise RuntimeError('Entry pose changed while connecting')
        self.report['entry_snapshot'] = actual
        self.telemetry = (self.args.out / 'telemetry.csv').open('w', newline='')
        self.telemetry_writer = csv.writer(self.telemetry)
        self.telemetry_writer.writerow(['time_s', 'stage', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'safety_mode'])
        goals, limits = scan_segments(self.c, self.scan, self.reset_tcp)
        self.preflight(goals, limits)

    def preflight(self, goals, limits, *, start_snapshot=None):
        from tools.run_vision_plan_trial import rot_error
        ctrl, recv = self.bridge._iface.ctrl, self.bridge._iface.recv
        normal(recv)
        if np.max(np.abs(ctrl.getTCPOffset())) > 1e-8:
            raise RuntimeError('Expected zero TCP offset')
        previous = finite(recv.getActualTCPPose(), (6,))
        q = finite(recv.getActualQ(), (6,))
        origin_checks = []
        if start_snapshot is not None:
            age = time.time()-float(finite(start_snapshot['time_s'], ()))
            if not 0 <= age <= 1:
                raise RuntimeError('Preflight start snapshot is stale')
            origin_checks.append(check_stationary_origin(start_snapshot, stationary_snapshot(recv)))
            # Build AND check a relative lift from the SAME stationary sample.
            # A second noisy origin must not turn a 50 mm command into 50.035 mm.
            previous = finite(start_snapshot['tcp'], (6,))
            q = finite(start_snapshot['q'], (6,))
        within(previous[:3]*1000, self.c['tcp_bounds_mm'], 'Current TCP')
        samples, total = [], 0.
        for goal, limit in zip(goals, limits, strict=True):
            goal = finite(goal, (6,))
            within(goal[:3]*1000, self.c['tcp_bounds_mm'], 'Target TCP')
            d, angle = np.linalg.norm(goal[:3]-previous[:3]), rot_error(previous, goal)
            segment = dict(stage=self.report.get('stage'), start_tcp=previous.tolist(),
                           goal_tcp=goal.tolist(), distance_mm=float(d*1000),
                           limit_mm=None if limit is None else float(limit),
                           rotation_deg=float(np.rad2deg(angle)))
            self.report.setdefault('preflight_segments', []).append(segment)
            if ((limit is not None and d*1000 > limit + 1e-6)
                    or np.rad2deg(angle) > self.c['max_rotation_deg']):
                limit_text = 'disabled' if limit is None else f'{limit:.6f}'
                raise RuntimeError(f'Unexpected segment length/rotation: {d*1000:.6f} mm '
                                   f'(limit {limit_text}), {np.rad2deg(angle):.6f} deg '
                                   f'(limit {self.c["max_rotation_deg"]:.6f})')
            total += d
            interpolator = Slerp([0, 1], Rotation.from_rotvec([previous[3:], goal[3:]]))
            count = max(3, int(np.ceil(d/.005))+1, int(np.ceil(angle/np.deg2rad(2)))+1)
            for t in np.linspace(0, 1, count):
                p = np.r_[previous[:3]+t*(goal[:3]-previous[:3]), interpolator(t).as_rotvec()]
                if not ctrl.isPoseWithinSafetyLimits(p.tolist()):
                    raise RuntimeError('Controller pose safety check failed')
                solution = finite(ctrl.getInverseKinematics(p.tolist(), q.tolist(), 1e-5, 1e-5), (6,))
                if np.max(np.abs(solution-q)) > .15:
                    raise RuntimeError('Controller IK changed branch')
                q = solution
                samples.append(q.tolist())
            previous = goal
        if start_snapshot is not None:
            origin_checks.append(check_stationary_origin(start_snapshot, stationary_snapshot(recv)))
        self.allowed_goals, self.next_goal = [np.array(g) for g in goals], 0
        self.timeout_s = total / self.speed + 30
        self.report.setdefault('controller_preflights', []).append(
            {'goals': goals, 'q_samples': samples, 'start_snapshot': start_snapshot,
             'stationary_origin_checks': origin_checks})

    def monitor(self, fn):
        failure = []
        def work():
            try:
                fn()
            except BaseException as exc:
                failure.append(exc)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        begin = time.monotonic()
        try:
            while thread.is_alive():
                normal(self.bridge._iface.recv)
                actual = self.bridge._iface.recv.getActualTCPPose()
                if self.telemetry is not None:
                    self.telemetry_writer.writerow([time.time(), self.report['stage'], *actual,
                                                     self.bridge._iface.recv.getSafetyMode()])
                    self.telemetry.flush()
                within(np.array(actual[:3])*1000,
                       self.c['tcp_bounds_mm'], 'Live TCP')
                if self.cancelled.is_set() or time.monotonic()-begin > self.timeout_s:
                    raise RuntimeError('Cycle cancelled or stage timeout')
                thread.join(.02)
            if failure:
                raise failure[0]
        except BaseException:
            self.cancelled.set()
            self.bridge._iface.ctrl.stopL(.2)
            thread.join(timeout=3)
            raise

    def wait_stationary(self, seconds):
        """Receive-only, bounded continuous dwell; never retry a movement.

        A transient speed excursion resets the dwell, not the timeout. Preserve
        the legacy speed gate; abnormal safety state/telemetry still fails fast.
        These UR samples cannot verify completion of write-only hand commands.
        """
        seconds = float(finite(seconds, ()))
        if seconds <= 0:
            raise ValueError('Stationary dwell must be positive')
        timeout = seconds + self.c['stationary_wait_grace_s']
        begin = time.monotonic()
        stable_since = None
        record = dict(stage=self.report.get('stage'), required_stable_s=seconds,
                      timeout_s=timeout, speed_norm_limit=STATIONARY_SPEED_NORM_LIMIT,
                      speed_units='[vx,vy,vz] m/s; [wx,wy,wz] rad/s; legacy combined norm gate',
                      status='waiting', samples=[])
        self.report.setdefault('stationary_waits', []).append(record)
        try:
            while True:
                if self.cancelled.is_set():
                    raise RuntimeError('Cycle cancelled while waiting for stationary UR5')
                recv = self.bridge._iface.recv
                normal(recv)
                speed = finite(recv.getActualTCPSpeed(), (6,))
                tcp = finite(recv.getActualTCPPose(), (6,))
                finite(recv.getActualQ(), (6,))
                now = time.monotonic()
                stationary = bool(np.linalg.norm(speed) <= STATIONARY_SPEED_NORM_LIMIT)
                record['samples'].append(dict(elapsed_s=now-begin, tcp=tcp.tolist(),
                                              tcp_speed=speed.tolist(), stationary=stationary))
                within(tcp[:3]*1000, self.c['tcp_bounds_mm'], 'Settling TCP')
                if not stationary:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                if now-begin >= timeout:
                    raise RuntimeError(f'UR5 stationary wait timed out after {timeout:.2f}s; '
                                       f'required {seconds:.2f}s continuously stable')
                if stable_since is not None and now-stable_since >= seconds:
                    record['status'] = 'stable'
                    return
                time.sleep(min(.02, timeout-(now-begin)))
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['elapsed_s'] = time.monotonic()-begin

    def open_hand(self, _):
        from rh56_controller.rh56_hand import RH56Hand
        from tools.run_vision_plan_trial import SerialLog
        self.hand = RH56Hand(self.c['hand_port'], hand_id=1)
        self.hand.ser.exclusive = True
        self.serial_log = (self.args.out / 'hand_serial.jsonl').open('w')
        self.hand.ser = SerialLog(self.hand.ser, self.serial_log)
        self.report['hardware_commands_attempted'] = True
        self.hand.speed_set([self.hand_speed]*6)
        time.sleep(.125)
        self.hand.force_set([self.c['hand_force_raw']]*6)
        time.sleep(.125)
        # Same opposing-thumb open shape used by this lab's planar Plan.
        self.hand.angle_set([1000, 1000, 1000, 1000, 1000, 0])
        self.wait_stationary(self.c['open_settle_s'])

    def move_tcp(self, target):
        from rh56_controller.ur5_bridge import UR5Bridge
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(target[3:]).as_matrix()
        T[:3, 3] = target[:3]
        self.monitor(lambda: self.bridge.move_to_hand_pose(UR5Bridge.tcp_T_to_hand_T(T), blocking=True))
        self.wait_stationary(.3)

    def move_scan(self, _):
        if self.reset_tcp is not None:
            self.report['stage'] = 'reset_lift'
            print('Reset: fingers commanded open; vertical retreat to scan height, then move to scan.', flush=True)
            self.move_tcp(self.reset_tcp)
            self.report['low_entry_reset']['completed'] = True
            self.report['stage'] = 'move_scan'
        self.move_tcp(self.scan)

    def detect(self, _):
        from tools import calibrate_ur5_external_camera_real as camera
        args = camera.parse_args(['--check-only', '--camera-serial', self.c['camera_serial']])
        pipeline, align, metadata = camera._start_camera(args)
        reports, paths = [], []
        try:
            for i in range(self.c['frames']):
                stationary_snapshot(self.bridge._iface.recv)
                rgb, depth, stamp, frame = camera._read_aligned_frame(pipeline, align, 5)
                self.capture_time = time.monotonic()
                capture = self.args.out / f'capture_{i:02d}'
                capture.mkdir()
                camera._save_rgb(capture / 'camera_check_rgb.png', rgb)
                camera._save_depth(capture / 'camera_check_depth_u16.png', depth)
                check = dict(schema='rh56_realsense_camera_check/v1', camera=camera._camera_metadata_to_dict(metadata),
                             host_time_s=time.time(), camera_timestamp_ms=stamp, frame_number=frame)
                (capture / 'camera_check.yaml').write_text(yaml.safe_dump(check, sort_keys=False))
                path, report = detect_saved(self.c, capture, self.args.out / f'detection_{i:02d}')
                reports.append(report)
                paths.append(path)
        finally:
            pipeline.stop()
        self.report['target_center_spread_mm'] = stable_target(reports, self.c['max_center_spread_mm'])
        self.report['target_validation'] = str(paths[-1])
        return paths[-1]

    def plan(self, path):
        self.plan_data = Plan(self.c, path)
        self.report['plan'] = self.plan_data.report
        dump(self.args.out / 'plan.json', self.plan_data.report)
        return self.plan_data

    def preflight_pick(self, plan):
        if plan.report['final_shape_extrapolated'] and not self.args.acknowledge_shape_extrapolation:
            raise RuntimeError('Narrower hand shape requires explicit extrapolation acknowledgement')
        goals, limits = pick_segments(self.c, plan)
        self.preflight(goals, limits)
        plan.report['hardware_ik_validated'] = True
        return plan

    def grasp(self, plan):
        from rh56_controller.grasp_executor import GraspExecutor
        if time.monotonic()-self.capture_time > self.c['max_capture_age_s']:
            raise RuntimeError('Detection expired before grasp')
        owner = self
        class CheckedHand:
            def angle_set(self, command):
                if owner.cancelled.is_set():
                    raise RuntimeError('Cycle cancelled; hand write blocked')
                try:
                    result = owner.hand.angle_set(command)
                    owner.angle_commands.append(list(command))
                    return result
                except Exception as exc:
                    owner.serial_errors.append(str(exc))
                    owner.cancelled.set()
                    raise
        statuses = []
        def status(message):
            statuses.append(message)
            print(message, flush=True)
            if 'error' in message.lower():
                self.cancelled.set()
        self.executor = GraspExecutor(self.bridge, CheckedHand(), plan.core.fk,
                                      status_cb=status, hand_write_only=True)
        def execute():
            self.executor.execute_plan_waypoints(plan.waypoints, force_N=0)
            while self.executor.is_running():
                if self.cancelled.is_set():
                    self.executor._abort.set()
                time.sleep(.02)
            if ('Plan: complete.' not in statuses or self.serial_errors
                    or self.angle_commands != plan.commands or self.cancelled.is_set()):
                raise RuntimeError('Plan/serial command failure; do not lift')
        self.monitor(execute)
        self.report['plan_statuses'] = statuses
        return plan

    def settle(self, plan):
        self.wait_stationary(self.c['close_settle_s'])
        return plan

    def lift(self, plan):
        from tools.run_vision_plan_trial import rot_error
        actual = stationary_snapshot(self.bridge._iface.recv)
        if (np.linalg.norm(np.array(actual['tcp'][:3])-plan.tcp[-1][:3]) > .001
                or rot_error(actual['tcp'], plan.tcp[-1]) > np.deg2rad(.5)):
            raise RuntimeError('Arm no longer at final grasp pose; lift blocked')
        # Re-preflight the ACTUAL vertical lift, keeping actual orientation/XY.
        goal = lift_pose(actual['tcp'], self.c['lift_mm']).tolist()
        self.report['lift_start_snapshot'] = actual
        self.report['lift_goal_tcp'] = goal
        self.preflight([goal], [self.c['lift_mm']], start_snapshot=actual)
        self.move_tcp(goal)
        self.report['final_snapshot'] = stationary_snapshot(self.bridge._iface.recv)

    def close(self, failed):
        cleanup_errors = []
        def attempt(action):
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(f'{type(exc).__name__}: {exc}')
        if failed:
            self.cancelled.set()
            if self.executor is not None:
                self.executor._abort.set()
            if self.bridge is not None and self.bridge._iface is not None:
                self.report['stopL_requested_on_failure'] = True
                attempt(lambda: self.bridge._iface.ctrl.stopL(.2))
        if self.executor is not None and self.executor._thread is not None:
            attempt(lambda: self.executor._thread.join(timeout=3))
        if self.bridge is not None:
            attempt(self.bridge.disconnect)
        if self.hand is not None:
            attempt(self.hand.ser.close)
        if self.serial_log is not None:
            attempt(self.serial_log.close)
        if self.telemetry is not None:
            attempt(self.telemetry.close)
        if self.plan_data is not None:
            attempt(self.plan_data.close)
        self.report['cleanup_errors'] = cleanup_errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New artifact directory; never overwritten')
    parser.add_argument('--capture', type=Path, help='Saved RGB-D directory for offline preview only')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight-only', action='store_true', help='Controller IK/safety checks for the full saved-scene path; no opening, movement, grasp or lift')
    mode.add_argument('--record-start', action='store_true', help='Receive-only UR scan-pose recording; no movement or hand connection')
    parser.add_argument('--confirm-controller-script', action='store_true', help='Permit RTDE control-script start/stop for preflight only; does NOT authorize motion')
    parser.add_argument('--confirm-cycle', help=f'Execution requires the exact text: {CONFIRMATION}')
    parser.add_argument('--acknowledge-open-loop', action='store_true', help='Supervised lift after timed closure, NOT confirmed contact/grasp; no automatic release/retry')
    parser.add_argument('--acknowledge-shape-extrapolation', action='store_true', help='Frozen mount and local width response are extrapolated to the narrower closure')
    parser.add_argument('--confirm-high-force', action='store_true', help='Explicitly permit configured force above 100; speed selection does not imply this')
    parser.add_argument('--reset-from-low', action='store_true',
                        help='Explicit supported-object reset: open at low entry, raise vertically <=150 mm to scan height, then scan; no extra Plan hover')
    parser.add_argument('--confirm-reset', help=f'Exact physical-scene confirmation: {RESET_CONFIRMATION}')
    args = parser.parse_args(argv)
    if args.execute and (args.confirm_cycle != CONFIRMATION or not args.acknowledge_open_loop
                         or not args.acknowledge_shape_extrapolation):
        parser.error('Execution requires exact --confirm-cycle, --acknowledge-open-loop and --acknowledge-shape-extrapolation')
    if args.preflight_only and not (args.confirm_controller_script and args.acknowledge_shape_extrapolation):
        parser.error('Preflight requires --confirm-controller-script and --acknowledge-shape-extrapolation')
    if args.reset_from_low and (not (args.execute or args.preflight_only)
                               or args.confirm_reset != RESET_CONFIRMATION):
        parser.error('Low-entry reset requires execute/preflight mode and exact --confirm-reset')
    if args.confirm_reset and not args.reset_from_low:
        parser.error('--confirm-reset requires --reset-from-low')
    if (args.execute or args.record_start) and args.capture:
        parser.error('Hardware modes cannot use a saved capture')
    if not args.execute and not args.record_start and not args.capture:
        parser.error('Offline preview/controller preflight requires --capture')
    return args


def main(argv=None):
    args = parse_args(argv)
    c = load_config(args.config)
    if args.execute and c['hand_force_raw'] > 100 and not args.confirm_high_force:
        raise ValueError('Force above 100 requires --confirm-high-force')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_vision_pick_cycle/v1', status='starting', stage='configuration',
                  config=c, events=[], hardware_commands_attempted=False,
                  grasp_success='unverified', automatic_release=False, automatic_retry=False,
                  assumptions=[
                      'Supervised single cycle for one known-size upright colored target; not general object recognition.',
                      'Operator reviews opening clearance, entry-to-scan, direct scan-to-Plan-approach and lift routes and keeps people clear.',
                      'Camera/desk/hand mounting must be unchanged; identity and height checks do not prove absence of XY drift.',
                      'No hand feedback. Settling times are explicit assumptions, not verified motion completion.',
                      'The measured local width offset and closure margin are experimental, not universal or force calibrated.',
                      'Controller IK/bounds are not obstacle or finger/table collision checks.',
                      'A completed cycle only proves command completion; object pickup must be observed.',
                      'On failure stop further commands; no automatic retry, release or blind lift. Hand targets may persist.',
                      'Optional low-entry reset requires an operator-confirmed supported object and clear opening/vertical-retreat paths; no automatic pose-based inference of support.',
                  ])
    live = plan = None
    failed = True
    try:
        if not args.record_start:
            for key in ('calibration', 'desk_plane', 'local_alignment', 'alignment_validation'):
                report.setdefault('source_sha256', {})[key] = hashlib.sha256(Path(c[key]).read_bytes()).hexdigest()
        print(json.dumps(report, indent=2), flush=True)
        if args.record_start:
            from tools.calibrate_ur5_external_camera_real import _open_rtde_receiver
            receiver = _open_rtde_receiver(c['ur_ip'])
            try:
                state = stationary_snapshot(receiver)
            finally:
                receiver.disconnect()
            state.update(schema='rh56_vision_pick_start/v1', ur_ip=c['ur_ip'],
                         clearance_verified=False, routes_verified=False)
            dump(args.out / 'start_pose.json', state)
            report.update(status='start_pose_recorded_read_only', start_pose=state)
        elif args.execute:
            live = LiveCycle(c, args, report)
            run_cycle(live, report, lambda: dump(args.out / 'summary.json', report))
        elif args.preflight_only:
            # Analyse the saved scene before touching even the RTDE controller.
            target, _ = detect_saved(c, args.capture, args.out / 'detection')
            live = LiveCycle(c, args, report)
            plan = live.plan(target)
            live.prepare(None)
            goals, limits = pick_segments(c, plan)
            # Include entry -> scan in the same IK chain. Do not pretend the
            # current arm has already moved to the saved scan pose.
            entry_goals, entry_limits = scan_segments(c, live.scan, getattr(live, 'reset_tcp', None))
            live.preflight([*entry_goals, *goals], [*entry_limits, *limits])
            plan.report['hardware_ik_validated'] = True
            dump(args.out / 'plan.json', plan.report)
            report.update(status='controller_preflight_passed_no_motion', stage='preflight_only',
                          preflight_limitations=[
                              'Saved scene only; execution must capture a new target and rerun IK.',
                              'Controller IK and pose limits are not scene/finger collision validation.',
                              'No hand was connected and no open/move/grasp/lift command was sent.',
                          ])
        else:
            target, _ = detect_saved(c, args.capture, args.out / 'detection')
            plan = Plan(c, target)
            report.update(status='offline_preview_only', plan=plan.report,
                          preview_limitations=['Saved scene only; no live stability, entry route, controller IK or collision validation.'])
            dump(args.out / 'plan.json', plan.report)
        failed = False
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if live is not None:
                live.close(failed and not args.preflight_only)
            if plan is not None and live is None:
                plan.close()
        finally:
            dump(args.out / 'summary.json', report)
            with (args.out / 'summary.csv').open('w', newline='') as f:
                keys = ('status', 'stage', 'hardware_commands_attempted', 'grasp_success', 'automatic_release', 'automatic_retry')
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerow({key: report[key] for key in keys})
            print(f"Cycle result: {report['status']}; report: {args.out / 'summary.json'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
