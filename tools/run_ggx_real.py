#!/usr/bin/env python3
"""Prepare an approved GGX grasp for CB3; hardware requires --execute + final Enter.

Default is OFFLINE. --preflight-only/--execute read current camera/UR state, then
ask separately before RTDE control-script startup. Only --execute offers the
final motion confirmation. No automatic reset, release, retry, or saved approval
reuse as motion authorization. RH56 feedback is never requested.
In default mode, reference-unstable depth-hole warnings require a current-photo Enter;
RGB/stable-depth failures cannot be overridden by that confirmation.
--static-scene skips BOTH initial and final camera capture/scene comparisons.
The operator must confirm the approved reference geometry is still unchanged;
robot/start checks and the checked-video motion confirmation remain mandatory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import threading
import traceback
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ggx_real_support as support
from tools import ggx_motion as motion
from tools import preflight_ggx_ur5_table as sim
from tools.run_ggx_review import pin, verify, read_json, show, run_stage, decision


# Existing phase-arrival/next-phase tolerances, not preflight-origin tolerances.
TARGET_POSITION_MM = 1.
TARGET_ROTATION_DEG = .5
TARGET_JOINT_RAD = .003


def load_config(path):
    config = yaml.safe_load(path.read_text())
    required = {'schema', 'ur_ip', 'hand_port', 'hand_id', 'joint_speed_rad_s',
                'joint_accel_rad_s2', 'cartesian_speed_m_s', 'cartesian_accel_m_s2',
                'hand_speed_raw', 'hand_force_raw',
                'open_settle_s', 'close_duration_s', 'close_settle_s',
                'tcp_bounds_mm', 'min_open_tcp_z_mm', 'model_position_tolerance_mm',
                'model_rotation_tolerance_deg', 'scene_center_tolerance_mm',
                'scene_yaw_tolerance_deg', 'scene_dimension_tolerance_mm',
                'max_scene_age_s', 'desk_edge_y_m'}
    if (not isinstance(config, dict) or not required <= set(config)
            or set(config) - required - {'desk_envelope', 'reset_motion', 'scene_check_mode',
                                        'preview_videos', 'controller_preflight_mode', 'workspace_policy',
                                        'reflex_desk_clearance_mm', 'reflex_desk_surface'}
            or config['schema'] != 'rh56_ggx_real_config/v1'):
        raise ValueError('Expected exact GGX real config v1 fields')
    motion.reflex_desk_clearance_m(config)
    if config.get('reflex_desk_surface','conservative-horizontal') not in ('conservative-horizontal','calibrated-plane'):
        raise ValueError('Unknown reflex_desk_surface')
    for key in ('ur_ip', 'hand_port'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f'Invalid {key}')
    for key, lower, upper in [('hand_id', 1, 254), ('hand_speed_raw', 1, 1000), ('hand_force_raw', 1, 1000)]:
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ValueError(f'Invalid {key}')
    ranges = {'joint_speed_rad_s': (.01, .3), 'joint_accel_rad_s2': (.01, .6),
              'cartesian_speed_m_s': (.005, .1), 'cartesian_accel_m_s2': (.01, .2),
              'open_settle_s': (1, 10), 'close_duration_s': (1, 10), 'close_settle_s': (.5, 10),
              'min_open_tcp_z_mm': (200, 600), 'model_position_tolerance_mm': (.1, 5),
              'model_rotation_tolerance_deg': (.1, 1), 'scene_center_tolerance_mm': (.1, 5),
              'scene_yaw_tolerance_deg': (.1, 5), 'scene_dimension_tolerance_mm': (.1, 5),
              'max_scene_age_s': (1, 30)}
    for key, (lower, upper) in ranges.items():
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'Invalid {key}')
    reset = config.get('reset_motion')
    if reset is not None:
        keys = {'joint_speed_rad_s', 'joint_accel_rad_s2', 'cartesian_speed_m_s', 'cartesian_accel_m_s2'}
        if not isinstance(reset, dict) or set(reset) != keys:
            raise ValueError('reset_motion requires exactly the four arm speed/acceleration fields')
        for key, value in reset.items():
            lo, hi = ranges[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'Invalid reset_motion {key}')
    if config.get('scene_check_mode', 'components') not in ('components', 'reference-rgbd'):
        raise ValueError('Unknown scene_check_mode')
    if 'preview_videos' in config and type(config['preview_videos']) is not bool:
        raise ValueError('preview_videos must be a boolean; it controls rendering only')
    if (not isinstance(config.get('controller_preflight_mode', 'dense'), str)
            or config.get('controller_preflight_mode', 'dense') not in motion.PREFLIGHT_MODES):
        raise ValueError('controller_preflight_mode must be dense or keypoints')
    workspace_policy = config.get('workspace_policy', 'configured-box')
    if workspace_policy not in ('configured-box', 'urdf-envelope'):
        raise ValueError('Unknown workspace_policy')
    if workspace_policy == 'urdf-envelope':
        from tools.ggx_planning import tcp_bounds_from_urdf
        derived = tcp_bounds_from_urdf()
        if config['tcp_bounds_mm'] is not None and config['tcp_bounds_mm'] != derived:
            raise ValueError('urdf-envelope bounds must be null or the exact derived envelope')
        config['tcp_bounds_mm'] = derived
    bounds = np.asarray(config['tcp_bounds_mm'], dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or not np.all(bounds[0] < bounds[1]):
        raise ValueError('Invalid TCP bounds')
    edge = config['desk_edge_y_m']
    if edge is not None and (isinstance(edge, bool) or not isinstance(edge, (int, float))
                             or not math.isfinite(edge) or not -1 < edge < .5):
        raise ValueError('desk_edge_y_m must be null or measured base-Y edge')
    if config.get('desk_envelope') is not None:
        support.desk_envelope_geometry(config['desk_envelope'])
        if edge is not None:
            raise ValueError('Choose desk edge OR envelope, not both')
    return config


def require_hand_setting_authorization(config, allowed):
    if max(config['hand_force_raw'], config['hand_speed_raw']) > 100 and not allowed:
        raise ValueError('Hand speed/force above 100 requires --allow-high-force and final Enter confirmation')


def inside(tcp, config):
    p = np.asarray(tcp, dtype=float)
    b = np.asarray(config['tcp_bounds_mm']) / 1000
    if p.shape != (6,) or not np.all(np.isfinite(p)) or np.any(p[:3] < b[0]) or np.any(p[:3] > b[1]):
        raise RuntimeError('TCP outside configured bounds or invalid')


def require_same_start(first, second):
    from tools.run_vision_pick_cycle import check_stationary_origin
    return check_stationary_origin(first, second)


def require_at_execution_target(target, current):
    from tools.run_vision_pick_cycle import finite
    pe, re = support.pose_error(current['tcp'], target['tcp'])
    qe = float(np.max(np.abs(finite(current['q'], (6,)) - finite(target['q'], (6,)))))
    if pe >= TARGET_POSITION_MM or re >= TARGET_ROTATION_DEG or qe >= TARGET_JOINT_RAD:
        raise RuntimeError(f'UR5 left execution target: {pe:.6f} mm, {re:.6f} deg, '
                           f'joint delta {qe:.6f} rad')


def scene_match(reference, current, config):
    """Exactly one valid matching top, no nearest-object fallback after movement."""
    if reference['crop'].get('crop_mode') == 'foreground':
        raise ValueError('Foreground crops require scene_check_mode: reference-rgbd, not flat-top component matching')
    expected = reference['crop']['top_patch']
    expected_dims = np.array(reference['crop']['dimensions_prior_mm'])
    matches = []
    for candidate in current.get('candidates', []):
        patch = candidate.get('top_patch')
        if (not candidate.get('geometry_candidate') or candidate.get('rejection_reasons')
                or not patch or patch.get('yaw_ambiguous', True)):
            continue
        xyz = np.asarray(patch['center_base_mm'], dtype=float)
        dims = np.array([patch['short_mm'], patch['long_mm'], patch['height_median_mm']])
        yaw = abs((patch['short_axis_yaw_deg'] - expected['short_axis_yaw_deg'] + 90) % 180 - 90)
        if (xyz.shape != (3,) or not np.all(np.isfinite(np.r_[xyz, dims, yaw]))):
            continue
        distance = float(np.linalg.norm(xyz - expected['center_base_mm']))
        if (distance <= config['scene_center_tolerance_mm']
                and yaw <= config['scene_yaw_tolerance_deg']
                and np.max(np.abs(dims - expected_dims)) <= config['scene_dimension_tolerance_mm']):
            matches.append(dict(component_id=candidate['component_id'], center_shift_mm=distance,
                                yaw_shift_deg=float(yaw), dimensions_mm=dims.tolist()))
    if len(matches) != 1:
        raise RuntimeError(f'Scene changed/ambiguous: {len(matches)} matching objects; regenerate GGX')
    return matches[0]


def capture_and_check(source, config, out, name, *, auto_open=True):
    """Camera only. Retain the original calibrated settings/background workflow."""
    if (source.get('crop', {}).get('crop_mode') == 'foreground'
            and config.get('scene_check_mode', 'components') != 'reference-rgbd'):
        raise ValueError('Foreground crops require scene_check_mode: reference-rgbd')
    from tools.preview_tabletop_target import load_perception_config
    perception_config, _ = load_perception_config(Path(source['perception']['config']))
    capture = out / f'{name}_capture'
    run_stage(out, name + '_capture', 'capture_tabletop_background.py',
              ['--out', capture, '--camera-serial', perception_config['camera_serial'],
               '--scene', 'object', '--roi', *perception_config['roi']])
    acquired = time.monotonic()
    if config.get('scene_check_mode', 'components') == 'reference-rgbd':
        from tools.ggx_scene_check import verify_saved_target
        print(f'GGX: {name}_scene_check (reference RGB-D; no re-segmentation)', flush=True)
        def review_warning(path, warnings):
            cells = ', '.join(f"({w['row']+1},{w['column']+1})" for w in warnings)
            print(f'深度告警：参考时已不稳定的区域 {cells} 出现新增深度点。'
                  '稳定区域及 RGB 检查已通过；这不证明未知区域没有障碍。', flush=True)
            show(path, auto_open)
            if name == 'final':
                remaining = config['max_scene_age_s'] - (time.monotonic()-acquired)
                print(f'本次照片有效期剩余约 {max(0,remaining):.0f} 秒；超时会停止，不延长有效期。', flush=True)
            return confirm('请查看参考/当前照片和橙框，并现场确认物体位置、高度、相机和桌面未变，'
                           '没有新增遮挡/障碍。不确定请取消。此确认仅接受深度告警，不是独立运动授权。')
        result = verify_saved_target(source, capture, config, out / f'{name}_scene_check',
                                     review_warning=review_warning)
        return dict(**result, acquired_monotonic=acquired, capture=str(capture))
    preview = out / f'{name}_perception'
    command = ['--capture', capture, '--out', preview, '--config', source['perception']['config']]
    if source['perception'].get('background'):
        command += ['--background', source['perception']['background']]
    for roi in source['perception'].get('exclusions', []):
        command += ['--exclude-roi', *roi]
    run_stage(out, name + '_perception', 'preview_tabletop_target.py', command)
    result = scene_match(source, read_json(preview / 'summary.json'), config)
    return dict(**result, acquired_monotonic=acquired, capture=str(capture), preview=str(preview))


class LiveBackend:
    """All device construction is explicit; single-threaded async motion polling."""
    def __init__(self, config, out, report, checkpoint):
        self.config, self.out, self.report, self.checkpoint = config, out, report, checkpoint
        self.receiver = self.control = self.hand = self.serial_log = None
        self.motion_active = False
        self.active_kind = None
        self.held_origin = None
        self.held_goal = None
        self.report['execution_hold_policy'] = dict(
            reference='fixed commanded goal after completed motion and settling',
            position_mm_exclusive=TARGET_POSITION_MM,
            rotation_deg_exclusive=TARGET_ROTATION_DEG,
            joint_rad_exclusive=TARGET_JOINT_RAD,
            pre_motion_origin_check='unchanged')

    def set_stage(self, stage):
        self.report['stage'] = stage
        self.checkpoint()

    def open_receiver(self):
        from tools.calibrate_ur5_external_camera_real import _open_rtde_receiver
        self.report['ur_receive_requested'] = True
        self.checkpoint()
        self.receiver = _open_rtde_receiver(self.config['ur_ip'])

    def snapshot(self):
        from tools.run_vision_pick_cycle import normal, finite, STATIONARY_SPEED_NORM_LIMIT
        normal(self.receiver)
        speed = finite(self.receiver.getActualTCPSpeed(), (6,))
        if np.linalg.norm(speed) > STATIONARY_SPEED_NORM_LIMIT:
            # All entry paths use the same bounded settling rule. A fresh
            # packet after a successful dwell must not revive the old error.
            return self.wait_stationary(.3)
        return dict(time_s=time.time(), tcp=finite(self.receiver.getActualTCPPose(), (6,)).tolist(),
                    q=finite(self.receiver.getActualQ(), (6,)).tolist(), robot_mode=7, safety_mode=1)

    def held_snapshot(self):
        current = self.snapshot()
        self.check_held_state(current)
        return current

    def check_held_state(self, current):
        if self.held_goal is not None:
            # Never recenter on each sample: cumulative drift must still fail.
            require_at_execution_target(self.held_goal, current)
        elif self.held_origin is not None:
            # Before the first move the original scene/preflight origin must
            # remain unchanged, including during opening the hand.
            require_same_start(self.held_origin, current)

    def open_control(self):
        import rtde_control
        self.report['controller_script_requested'] = True
        self.checkpoint()
        self.control = rtde_control.RTDEControlInterface(self.config['ur_ip'])

    def telemetry(self):
        from tools.run_vision_pick_cycle import normal, finite
        normal(self.receiver)
        tcp = finite(self.receiver.getActualTCPPose(), (6,))
        q = finite(self.receiver.getActualQ(), (6,))
        qd = finite(self.receiver.getActualQd(), (6,))
        inside(tcp, self.config)
        return tcp, q, qd

    def dwell(self, seconds):
        self.wait_stationary(seconds)

    def wait_stationary(self, seconds, *, goal_q=None, goal_tcp=None):
        """Reuse the frozen cycle's bounded continuous dwell without moving.

        Keep the legacy speed gate and 3 s grace, with extra goal/held-state
        guards. Return the sample that passed the dwell, not a new single-packet
        stationarity assertion that could race the final deceleration packet.
        This does not read RH56 sensors or confirm completion of hand movement.
        """
        from tools.run_vision_pick_cycle import LiveCycle, finite
        if (goal_q is None) != (goal_tcp is None):
            raise ValueError('Settling target requires both joints and TCP')
        goal = None if goal_q is None else dict(q=goal_q, tcp=goal_tcp)
        backend = self

        class GuardedReceiver:
            def __getattr__(self, name):
                return getattr(backend.receiver, name)

            def getActualTCPPose(self):
                self.tcp = finite(backend.receiver.getActualTCPPose(), (6,)).tolist()
                return self.tcp

            def getActualQ(self):
                self.q = finite(backend.receiver.getActualQ(), (6,)).tolist()
                current = dict(q=self.q, tcp=self.tcp)
                if goal is not None:
                    require_at_execution_target(goal, current)
                else:
                    backend.check_held_state(current)
                self.sample_time = time.time()
                return self.q

        receiver = GuardedReceiver()
        adapter = SimpleNamespace(
            c=dict(tcp_bounds_mm=self.config['tcp_bounds_mm'], stationary_wait_grace_s=3.),
            cancelled=threading.Event(),
            bridge=SimpleNamespace(_iface=SimpleNamespace(recv=receiver)), report=self.report)
        try:
            LiveCycle.wait_stationary(adapter, seconds)
            return dict(time_s=receiver.sample_time, tcp=receiver.tcp, q=receiver.q,
                        robot_mode=7, safety_mode=1)
        finally:
            self.checkpoint()

    def open_hand(self, command):
        self.set_stage('hand_open')
        self.held_snapshot()
        from rh56_controller.rh56_hand import RH56Hand
        from tools.run_vision_plan_trial import SerialLog
        self.report['hand_connection_requested'] = True
        self.checkpoint()
        self.hand = RH56Hand(self.config['hand_port'], hand_id=self.config['hand_id'])
        self.hand.ser.exclusive = True
        self.serial_log = (self.out / 'hand_serial.jsonl').open('x')
        self.hand.ser = SerialLog(self.hand.ser, self.serial_log)
        for label, method, value in [('speed', self.hand.speed_set, self.config['hand_speed_raw']),
                                      ('force', self.hand.force_set, self.config['hand_force_raw'])]:
            self.report.setdefault('hand_commands', []).append(dict(kind=label, values=[value] * 6))
            self.checkpoint()
            method([value] * 6)
            self.dwell(.125)
        self.hand_angles(command)
        self.dwell(self.config['open_settle_s'])

    def hand_angles(self, command):
        self.held_snapshot()
        self.report.setdefault('hand_commands', []).append(dict(kind='angles', values=list(command)))
        self.checkpoint()
        # The existing protocol reads possible write acknowledgements, but no
        # sensor-read frames are transmitted and missing feedback is expected.
        self.hand.angle_set(command)

    def move_phase(self, command):
        from tools.run_vision_pick_cycle import normal, finite
        if (command.get('revision') != motion.REVISION
                or command.get('kind') not in ('moveJ', 'moveL')
                or command['kind'] != (motion.KINDS | motion.RESET_KINDS | motion.TRIAL_KINDS).get(command.get('phase'))):
            raise RuntimeError('Expected checked continuous phase command')
        if command.get('controller_preflight_mode', 'dense') != self.config.get('controller_preflight_mode', 'dense'):
            raise RuntimeError('Controller preflight mode changed since path was checked')
        phase, kind = command['phase'], command['kind']
        samples = command.get('samples', [])
        if (len(samples) < 2
                or not np.allclose(finite(samples[0]['q'], (6,)), finite(command['start_q'], (6,)), atol=1e-10, rtol=0)
                or not np.allclose(finite(samples[-1]['q'], (6,)), finite(command['q'], (6,)), atol=1e-10, rtol=0)):
            raise RuntimeError('Checked phase endpoints/samples are inconsistent')
        expected_speed = self.config['joint_speed_rad_s' if kind == 'moveJ' else 'cartesian_speed_m_s']
        expected_accel = self.config['joint_accel_rad_s2' if kind == 'moveJ' else 'cartesian_accel_m_s2']
        if command['speed'] != expected_speed or command['accel'] != expected_accel:
            raise RuntimeError('Checked phase speed/acceleration changed')
        self.set_stage(phase)
        origin = self.held_snapshot()
        q0, q1 = finite(origin['q'], (6,)), finite(command['q'], (6,))
        tcp_goal = command['goal_tcp']
        require_at_execution_target(dict(tcp=command['start_tcp'], q=command['start_q']), origin)
        previous_progress = self.control.getAsyncOperationProgress()
        if previous_progress >= 0:
            raise RuntimeError('Another asynchronous operation is still active')
        self.report['motion_commands_attempted'] = True
        self.report.setdefault('arm_commands', []).append(dict(phase=phase, kind=kind, q=q1.tolist(),
            tcp=tcp_goal, speed=command['speed'], accel=command['accel'],
            checked_samples=len(command['samples']), revision=motion.REVISION))
        self.checkpoint()
        self.motion_active = True
        self.active_kind = kind
        target = q1.tolist() if kind == 'moveJ' else tcp_goal
        if not getattr(self.control, kind)(target, command['speed'], command['accel'], True):
            raise RuntimeError(f'Controller rejected {kind}')
        if kind == 'moveJ':
            distance = float(np.max(np.abs(q1 - q0)))
        else:
            position_mm, angle_deg = support.pose_error(command['start_tcp'], tcp_goal)
            distance = max(position_mm/1000, np.deg2rad(angle_deg))
        timeout = distance / command['speed'] + 2 * math.sqrt(distance / command['accel']) + 5
        begin = time.monotonic()
        trace = []
        try:
            while True:
                normal(self.receiver)
                tcp, q, qd = self.telemetry()
                if motion.joint_path_error(q, command['samples']) > .03:
                    raise RuntimeError('Live joint path deviated from checked continuous phase')
                if kind == 'moveL':
                    path_pe, path_re = motion.cartesian_path_error(tcp, command)
                    if path_pe > 1 or path_re > .5:
                        raise RuntimeError('Live TCP deviated from checked straight path')
                pe, re = support.pose_error(tcp, tcp_goal)
                progress = self.control.getAsyncOperationProgress()
                tcp_speed = finite(self.receiver.getActualTCPSpeed(), (6,))
                trace.append(dict(time_s=time.time(), q=q.tolist(), tcp=tcp.tolist(),
                                  qd=qd.tolist(), tcp_speed=tcp_speed.tolist(),
                                  async_progress=int(progress)))
                # A short segment may already fall within position tolerance
                # before it starts. Require this operation's completion token,
                # not merely an old stationary RTDE packet.
                if (progress < 0 and progress != previous_progress
                        and np.max(np.abs(q - q1)) < TARGET_JOINT_RAD and np.max(np.abs(qd)) < .001
                        and pe < TARGET_POSITION_MM and re < TARGET_ROTATION_DEG):
                    # Reuse the existing real-cycle settling behavior instead
                    # of failing the run on one transient nonzero speed sample.
                    self.held_origin = self.wait_stationary(.3, goal_q=q1, goal_tcp=tcp_goal)
                    # Use the checked command, not a noisy arrival sample, as
                    # the fixed reference throughout closure and the next dwell.
                    self.held_goal = dict(q=q1.tolist(), tcp=list(tcp_goal))
                    self.motion_active = False
                    self.active_kind = None
                    return
                if time.monotonic() - begin > timeout:
                    raise RuntimeError('Motion timeout')
                time.sleep(.02)
        finally:
            with (self.out / 'telemetry.jsonl').open('a') as stream:
                for row in trace:
                    stream.write(json.dumps(dict(phase=phase, **row)) + '\n')

    def close(self):
        errors = []
        def attempt(fn):
            try:
                fn()
            except Exception as exc:
                errors.append(str(exc))
        if self.control is not None:
            if self.motion_active:
                kind = self.active_kind or 'moveJ'
                self.report['stop_requested_on_failure'] = 'stopL' if kind == 'moveL' else 'stopJ'
                if kind == 'moveL':
                    attempt(lambda: self.control.stopL(self.config['cartesian_accel_m_s2']))
                else:
                    attempt(lambda: self.control.stopJ(self.config['joint_accel_rad_s2']))
            attempt(self.control.stopScript)
            attempt(self.control.disconnect)
        if self.hand is not None:
            attempt(self.hand.ser.close)
        if self.serial_log is not None:
            attempt(self.serial_log.close)
        if self.receiver is not None:
            attempt(self.receiver.disconnect)
        self.report['cleanup_errors'] = errors


def check_tcp_offset(control):
    offset = np.asarray(control.getTCPOffset(), dtype=float)
    if offset.shape != (6,) or not np.all(np.isfinite(offset)) or np.max(np.abs(offset)) > 1e-8:
        raise RuntimeError('Expected zero active TCP offset')


def preflight_controller(backend, prepared, model, config, report):
    """Actual CB3 IK/FK on continuous primitives; never call a movement method."""
    check_tcp_offset(backend.control)
    return motion.preflight(backend, prepared, model, config, report,
                            report.get('requested_phase', 'grasp-lift'))


def execute_checked(backend, prepared, checked, config, phase):
    """Only called after final interactive motion confirmation + fresh scene check."""
    expected = ['start_to_pregrasp', 'approach'] + (['lift'] if phase == 'grasp-lift' else [])
    if prepared.get('trial_start') is not None:
        expected.insert(0, 'position_to_trial_start')
    if ([c.get('phase') for c in checked] != expected
            or any(c.get('revision') != motion.REVISION or c.get('kind') != (motion.KINDS | motion.TRIAL_KINDS)[c['phase']]
                   for c in checked)):
        raise RuntimeError('Missing or obsolete continuous-phase preflight')
    print('GGX: hand_open', flush=True)
    backend.open_hand(prepared['hand_open_raw'])
    closing = [row for row in prepared['rows'] if row['phase'] == 'close']
    for command in checked:
        if command['phase'] == 'lift':
            backend.set_stage('close')
            print('GGX: close', flush=True)
            for row in closing:
                backend.hand_angles(row['hand_raw'])
                backend.dwell(config['close_duration_s'] / len(closing))
            backend.set_stage('close_settle')
            print('GGX: close_settle', flush=True)
            backend.dwell(config['close_settle_s'])
        print(f"GGX: {command['phase']} ({command['kind']}, one continuous motion)", flush=True)
        backend.move_phase(command)


def confirm(prompt):
    try:
        return input(prompt + ' 按 Enter 确认；其他输入取消：').strip() == ''
    except (EOFError, KeyboardInterrupt):
        return False


def run(args, config, source, out, report, checkpoint, backend_factory=LiveBackend,
        scene_check=None, prepare_fn=support.prepare, review_fn=motion.write_review,
        controller_authorization=None):
    from tools.ggx_planning import PlanningUnavailable
    live = args.execute or args.preflight_only
    if live and source.get('selection_scope') == 'official_ggx_candidate':
        report.update(status='physical_hand_mapping_unverified',
            failure_layer='integration_hand_contract', official_candidate_retained=True,
            physical_hand_verified=False,
            reason='Official OPEN and local command geometry are not physically verified; '
                   'run without --execute/--preflight-only for offline local assessment. '
                   'See docs/ggx_physical_hand_check_zh.md. No devices opened.')
        checkpoint()
        print(report['reason'], flush=True)
        return 2
    checked_review = getattr(args, 'checked_review', False)
    static_scene = getattr(args, 'static_scene', False)
    if (live and source.get('source_kind') == 'grasp_template'
            and not (args.execute and checked_review and static_scene)):
        raise ValueError('Template execution requires current checked-video approval and explicit static scene')
    if static_scene and (not args.execute or not checked_review or config.get('scene_check_mode') != 'reference-rgbd'):
        raise ValueError('Static scene requires --execute, checked review and reference-rgbd mode')
    if scene_check is None:
        scene_check = lambda s, c, o, n: capture_and_check(s, c, o, n, auto_open=args.open)
    if live and source.get('source_kind') == 'unreviewed_candidate' and not checked_review:
        raise ValueError('Unreviewed candidate requires checked CB3 video approval before motion')
    if checked_review and live and not args.video:
        raise ValueError('Checked CB3 review requires its execution video')
    if controller_authorization is None:
        controller_authorization = {}
    if (live and source.get('crop', {}).get('crop_mode') == 'foreground'
            and config.get('scene_check_mode', 'components') != 'reference-rgbd'):
        raise ValueError('Foreground crops require scene_check_mode: reference-rgbd')
    backend = None
    report['requested_phase'] = args.phase
    report['preview_videos_enabled'] = args.video
    report['numerical_preflight_required'] = True
    report['scene_policy'] = 'operator_fixed_reference' if static_scene else 'fresh_rgbd'
    report['static_scene_confirmed'] = False
    def timed(stage, operation):
        print(f'GGX real: {stage}…', flush=True)
        started = time.monotonic()
        record = dict(stage=stage, status='running')
        report.setdefault('timings', []).append(record)
        checkpoint()
        try:
            value = operation()
            record['status'] = 'completed'
            return value
        except BaseException:
            record['status'] = 'failed'
            raise
        finally:
            record['elapsed_s'] = time.monotonic() - started
            checkpoint()
            print(f"GGX real: {stage} {record['status']} ({record['elapsed_s']:.2f}s)", flush=True)
    try:
        if live and config.get('desk_envelope') is not None and not config['desk_envelope']['confirmed']:
            raise ValueError('Desk envelope dimensions/center bound need operator confirmation')
        if live:
            backend = backend_factory(config, out, report, checkpoint)
            backend.open_receiver()
            snapshot = backend.snapshot()
            inside(snapshot['tcp'], config)
            if snapshot['tcp'][2] * 1000 < config['min_open_tcp_z_mm']:
                raise RuntimeError('Start too low for supervised open; no automatic reset is authorized')
            if static_scene:
                print('GGX: initial_scene_check skipped (--static-scene; no capture/RGB/depth check)', flush=True)
                report['initial_scene'] = dict(method='operator_fixed_reference/v1', checked=False,
                    capture_skipped=True, rgb_checked=False, depth_checked=False, operator_confirmed=False,
                    reference_capture=source.get('perception', {}).get('capture'),
                    assumption='Operator must confirm the original reference scene is unchanged; NOT sensor verified.')
                if source.get('source_kind') == 'grasp_template':
                    report['scene_policy'] = 'operator_template_target'
                    report['initial_scene'].update(target=source['template_relocation'],
                        assumption='Operator must confirm same object at this trial template target; NOT sensor verified.')
            else:
                report['initial_scene'] = scene_check(source, config, out, 'initial')
            require_same_start(snapshot, backend.snapshot())
        else:
            snapshot = read_json(args.start_snapshot)
        report['start_snapshot'] = snapshot
        checkpoint()
        source = dict(source, desk_edge_y_m=config['desk_edge_y_m'],
                      desk_envelope=config.get('desk_envelope'))
        try:
            prepared, model = timed('nominal_cb3_preflight', lambda:
                prepare_fn(source, snapshot, out, video=args.video and not (live and checked_review)))
        except PlanningUnavailable as exc:
            if source.get('source_kind') == 'grasp_template':
                raise  # The experiment records/skips only this typed pre-motion result.
            official_only = source.get('selection_scope') == 'official_ggx_candidate'
            report.update(status='local_execution_unavailable' if official_only else 'cb3_candidate_rejected',
                          planning_reason=exc.reason,planning_attempts=exc.attempts)
            if official_only:
                report.update(failure_layer='local_execution', official_candidate_retained=True,
                              local_assessment=str(out / 'local_assessment.json'))
            checkpoint()
            return 2
        except ValueError:
            rejection_path = out / 'cb3_rejection.json'
            rejection = read_json(rejection_path) if rejection_path.is_file() else {}
            if (checked_review and rejection.get('passed') is False
                    and rejection.get('rank') == source['rank']
                    and rejection.get('reason') in ('ik_failed_pregrasp', 'ik_failed_approach',
                        'ik_failed_lift', 'desk_clearance', 'object_collision_before_closure')):
                report.update(status='cb3_candidate_rejected', nominal_rejection=rejection)
                checkpoint()
                return 2  # No controller or hand opened. Never mask a live-state failure.
            raise
        artifacts = [out / 'prepared.json']
        if args.video and not (live and checked_review):
            artifacts.append(out / 'cb3_review.mp4')
        report['prepared_sha256'] = pin(artifacts)
        report['prepared'] = str(out / 'prepared.json')
        report['status'] = 'offline_prepared' if not live else 'awaiting_controller_confirmation'
        checkpoint()
        if not live:
            return 0
        if checked_review:
            print('先完成控制器检查，再显示唯一的 CB3 执行视频；当前没有运动授权。', flush=True)
        elif args.video:
            show(out / 'cb3_review.mp4', args.open)
        else:
            print('额外CB3视频已关闭；数值预检仍执行，稍后按当前起点和目标确认运动。', flush=True)
        print('这是从当前起点生成的 CB3 新路径；不是原 UR5e 关节回放。', flush=True)
        if not controller_authorization.get('confirmed', False):
            if not confirm('仅允许本轮候选的 RTDE 控制脚本启停和 IK 预检，不允许运动。'):
                report['status'] = 'cancelled_before_controller'
                return 2
            controller_authorization['confirmed'] = True
        report['controller_authorization'] = 'explicit RTDE-only confirmation in this invocation'
        verify(source['pins'])
        verify(report['code_config_sha256'])
        verify(report['prepared_sha256'])
        require_same_start(snapshot, backend.snapshot())
        timed('connect_controller', backend.open_control)
        checked = timed('controller_preflight', lambda:
            preflight_controller(backend, prepared, model, config, report))
        review_path = timed('execution_geometry_and_optional_video', lambda:
            review_fn(out, prepared, model, checked, video=args.video))
        (out / 'execution.json').write_text(json.dumps(dict(revision=motion.REVISION,
            commands=checked), indent=2, allow_nan=False) + '\n')
        execution_artifacts = [out / 'execution.json', out / 'execution_precheck.json']
        if args.video:
            if review_path is None:
                raise ValueError('Requested execution review video is missing')
            execution_artifacts.append(review_path)
        report['execution_sha256'] = pin(execution_artifacts)
        report['execution_video'] = str(review_path) if review_path is not None else None
        report['status'] = 'controller_preflight_passed'
        checkpoint()
        if args.video and not (checked_review and args.execute):
            show(review_path, args.open)
            print('此视频显示实际连续运动的采样路径，不代表真实播放时长。中间采样点不停车。', flush=True)
        elif not args.video:
            print(f'数值检查记录：{out / "execution_precheck.json"}；不播放额外视频。', flush=True)
            print(f"当前起点 TCP（mm）：{np.round(np.asarray(snapshot['tcp'][:3])*1000, 1).tolist()}", flush=True)
            for command in checked:
                target = command.get('goal_tcp', command['tcp'])
                print(f"  {command['phase']} / {command['kind']} → TCP（mm）"
                      f"{np.round(np.asarray(target[:3])*1000, 1).tolist()}", flush=True)
        if not args.execute:
            return 0
        if static_scene:
            if source.get('source_kind') == 'grasp_template':
                print('TEMPLATE：本轮使用模板重定位后的坐标；旧视频不授权新路径。\n'
                      f"物体几何中心（mm）：{np.round(np.asarray(source['preflight']['object_center_m'])*1000, 2).tolist()}；"
                      f"长轴 yaw：{source['preflight']['object_yaw_deg']:.2f}°。\n"
                      'Enter 同时确认：当前物体与模板为同一物体，实际摆放与本轮坐标一致、之后未动，'
                      '相机/桌面/安装未变，无新增障碍；本轮不做视觉复核。', flush=True)
            else:
                print('STATIC SCENE：跳过初始和最终拍摄、RGB／深度检查，直接使用原批准场景的 3D 坐标。\n'
                  '下面的 Enter 同时确认：从原规划参考拍摄到开始动作，物体位置/高度、相机、桌面、安装均未改变，'
                  '无新增障碍；本轮没有视觉核验。若有变化或不确定，请取消。确认仅授权本轮。',flush=True)
        print(f"最终运动确认：rank {source['rank']}；{args.phase}；{len(checked)} 段连续运动；手 force={config['hand_force_raw']}、"
              f"speed={config['hand_speed_raw']}；关节速度={config['joint_speed_rad_s']} rad/s，"
              f"直线速度={config['cartesian_speed_m_s']} m/s。\n"
              '现场确认物体已支撑、张手不会掉物、人员离开运动区、全程及抬升路径畅通，'
              '相机/桌面/手安装未变，你可立即急停。\n'
              '闭合仅按仿真目标定时发送，没有手反馈；完成后保持，不自动松手或重试。', flush=True)
        if checked_review:
            review_pins = {**source['pins'], **report['code_config_sha256'],
                           **report['prepared_sha256'], **report['execution_sha256']}
            action, reason = decision(review_path, review_pins, args.open, execution=True)
            report['checked_review'] = dict(decision=action, reason=reason,
                rank=source['rank'], video=str(review_path), reviewed_sha256=review_pins,
                scene_policy=report['scene_policy'],
                assumptions=['Video uses the checked CB3/controller path, not UR5e joint replay.',
                             'Enter authorizes only this invocation and path; robot/start checks remain.',
                             ('Operator confirms unchanged reference scene; no new camera/scene verification.'
                              if static_scene else 'Fresh final scene verification remains required.')])
            checkpoint()
            if action != 'approve':
                report['status'] = ('cb3_review_vetoed' if action == 'veto' else
                                    'scene_invalidated' if action == 'scene_invalidated' else 'cancelled_before_motion')
                return 2
        elif not confirm('批准上述 CB3 路径并实际执行？'):
            report['status'] = 'cancelled_before_motion'
            return 2
        report['motion_confirmation_received'] = True
        report['static_scene_confirmed'] = bool(static_scene)
        checkpoint()
        verify(source['pins'])
        verify(report['code_config_sha256'])
        verify(report['prepared_sha256'])
        verify(report['execution_sha256'])
        if static_scene:
            print('GGX: final_scene_check skipped (--static-scene; operator confirmed fixed reference)', flush=True)
            report['final_scene'] = dict(report['initial_scene'], operator_confirmed=True)
        else:
            report['final_scene'] = scene_check(source, config, out, 'final')
        require_same_start(snapshot, backend.snapshot())
        if not static_scene:
            age = time.monotonic() - report['final_scene']['acquired_monotonic']
            if not 0 <= age <= config['max_scene_age_s']:
                raise RuntimeError('Final scene check expired')
        verify(source['pins'])
        verify(report['code_config_sha256'])
        verify(report['prepared_sha256'])
        verify(report['execution_sha256'])
        check_tcp_offset(backend.control)
        backend.held_origin = backend.snapshot()
        require_same_start(snapshot, backend.held_origin)
        report['status'] = 'executing'
        checkpoint()
        execute_checked(backend, prepared, checked, config, args.phase)
        report['final_snapshot'] = backend.snapshot()
        report['status'] = ('approach_completed' if args.phase == 'approach-only'
                            else 'commands_completed_grasp_unverified')
        return 0
    finally:
        if backend is not None:
            backend.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--approval', type=Path, help='Saved interactive visual approval summary.json')
    source.add_argument('--candidate', type=Path, help='Unreviewed automatic pass; CB3 video review before execution')
    parser.add_argument('--checked-review', action='store_true',
                        help='Review the checked execution video; Enter executes, n vetoes, p replays, r invalidates')
    parser.add_argument('--static-scene', action='store_true',
                        help='Operator confirms fixed reference: skip BOTH camera captures and RGB/depth comparisons; requires --execute and --checked-review')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ggx_real_lab.yaml')
    parser.add_argument('--start-snapshot', type=Path, default=ROOT / 'artifacts/vision_pick_start/start_pose.json',
                        help='Offline only; hardware modes always read actual joints')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--phase', choices=('approach-only', 'grasp-lift'), default='approach-only')
    parser.add_argument('--allow-high-force', action='store_true',
                        help='Explicitly allow configured raw hand speed/force above 100; final Enter still required')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=None,
                        help='Extra CB3 videos (default: config preview_videos). --no-video keeps all numerical checks')
    parser.add_argument('--open', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    args.checked_review |= args.candidate is not None
    if args.static_scene and (not args.execute or not args.checked_review):
        parser.error('--static-scene requires --execute and --checked-review (or --candidate)')
    if args.checked_review and args.video is False:
        parser.error('--checked-review/--candidate requires a checked CB3 video; cannot use --no-video')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def main(argv=None, *, controller_authorization=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.checked_review:
        args.video = True
    elif args.video is None:
        args.video = config.get('preview_videos', True)
    if args.execute or args.preflight_only:
        require_hand_setting_authorization(config, args.allow_high_force)
    source = support.read_candidate(args.candidate) if args.candidate else support.read_approval(args.approval)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.out or ROOT / 'artifacts/ggx_real_runs' / stamp
    out.mkdir(parents=True, exist_ok=False)
    model_assets = list((support.URDF.parent / 'meshes/ur5/collision').glob('*.stl'))
    model_assets += [p for p in (ROOT / 'h1_mujoco/inspire').rglob('*')
                     if p.is_file() and p.suffix.lower() in ('.xml', '.stl', '.obj')]
    report = dict(schema='rh56_ggx_real_run/v1', status='starting', executable=False,
                  approval=str(args.approval) if args.approval else None,
                  candidate=str(args.candidate) if args.candidate else None, selected_rank=source['rank'],
                  checked_review_required=args.checked_review,
                  execute_requested=args.execute, motion_confirmation_received=False,
                  motion_commands_attempted=False, controller_script_requested=False,
                  code_config_sha256=pin([args.config, __file__, support.__file__, sim.__file__, motion.__file__,
                      ROOT / 'tools/ggx_planning.py',
                      ROOT / 'tools/ggx_scene_check.py',
                      ROOT / 'tools/prepare_ggx_tabletop_cloud.py',
                      support.URDF, ROOT / 'rh56_controller/ur5_bridge.py',
                      ROOT / 'rh56_controller/grasp_executor.py', ROOT / 'rh56_controller/rh56_hand.py',
                      ROOT / 'rh56_controller/graspgenx_baseline.py',
                      ROOT / 'tools/run_ggx_review.py', ROOT / 'tools/preview_tabletop_target.py',
                      ROOT / 'tools/capture_tabletop_background.py',
                      ROOT / 'tools/run_vision_pick_cycle.py',
                      *model_assets]))
    def checkpoint():
        (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        keys = ['status', 'selected_rank', 'execute_requested', 'motion_confirmation_received', 'motion_commands_attempted']
        with (out / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerow({k: report[k] for k in keys})
    checkpoint()
    try:
        return run(args, config, source, out, report, checkpoint,
                   controller_authorization=controller_authorization)
    except (EOFError, KeyboardInterrupt):
        report['status'] = 'interrupted'
        return 130
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      error_traceback=traceback.format_exc())
        print(report['error'], file=sys.stderr)
        return 1
    finally:
        checkpoint()
        print(f"GGX real: {report['status']}；{out / 'summary.json'}", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
