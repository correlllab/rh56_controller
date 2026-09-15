"""Explicit GGX open/retreat/return step, used only by --reset-first.

No camera or candidate input: the object has moved and is intentionally NOT
modeled. Operator confirmation covers object support and clear reset paths.
Reuse the CB3 continuous-motion preflight/backend; never replay sampled stops.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from tools import run_ggx_real as real
from tools import ggx_motion as motion
from tools import ggx_real_support as support
from tools.run_ggx_review import pin, verify, read_json


def load_start(path, config):
    start = read_json(path)
    if start.get('schema') != 'rh56_vision_pick_start/v1' or start.get('ur_ip') != config['ur_ip']:
        raise ValueError('Reset requires a recorded start pose for this UR5')
    motion.finite6(start['q'])
    real.inside(start['tcp'], config)
    if start['tcp'][2] * 1000 < config['min_open_tcp_z_mm']:
        raise ValueError('Saved scan start is below the configured elevated region')
    return start


def desk_source(config, perception):
    """Fixed desk: historical horizontal cap or explicit calibrated-plane profile."""
    from tools.analyze_ur5_camera_colored_target import plane_z_at_xy_mm
    envelope = config.get('desk_envelope')
    if envelope is None:
        raise ValueError('Reset requires the confirmed finite desk envelope')
    center, radius = support.desk_envelope_geometry(envelope)
    normal = np.asarray(yaml.safe_load(Path(perception['desk_plane']).read_text())['plane']['normal_base'])
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or abs(normal[2]) < 1e-6:
        raise ValueError('Invalid calibrated desk normal')
    z = plane_z_at_xy_mm(normal, perception['desk_z_mm'], *(center * 1000)) / 1000
    surface=config.get('reflex_desk_surface','conservative-horizontal')
    if surface=='calibrated-plane':
        # Same measured base-origin height and fitted normal as perception.
        # Do not subtract an operator-observed gap or move object/start coordinates.
        normal=normal/np.linalg.norm(normal)
        if normal[2]<0: normal=-normal
        return dict(preflight=dict(table_height_m=float(z)),desk_envelope=envelope,
                    desk_edge_y_m=None,desk_surface_policy='calibrated-plane',
                    desk_surface_plane=dict(normal_base=normal.tolist(),
                                            point_base_m=[0.,0.,float(perception['desk_z_mm'])/1000]))
    if surface!='conservative-horizontal':
        raise ValueError('Unknown reflex_desk_surface')
    z += radius * np.linalg.norm(normal[:2]) / abs(normal[2])
    return dict(preflight=dict(table_height_m=float(z)), desk_envelope=envelope,
                desk_edge_y_m=None)


def completed_lift_target(path):
    """Evidence for omitting a duplicate retreat, never a movement approval."""
    record = read_json(path)
    if (record.get('status') != 'commands_completed_grasp_unverified'
            or record.get('requested_phase') != 'grasp-lift'
            or record.get('lift_hold_completed') is not True
            or record.get('grasp_close_command_attempted') is not True
            or record.get('lift_mm') != 80. or record.get('cleanup_errors')):
        raise ValueError('Reset needs a completed 80 mm lift/hold report with clean shutdown')
    target = record['final_snapshot']
    motion.finite6(target['tcp']); motion.finite6(target['q'])
    return target


def reset_goals(snapshot, start, config, *, lifted_target=None, recover_from_current=False):
    real.inside(snapshot['tcp'], config)
    try:
        real.require_at_execution_target(start, snapshot)
        already_at_start = True
    except RuntimeError:
        already_at_start = False
    goals = []
    # Existing GGX lift distance, reused as an operator-reviewed retreat, not
    # a claim that 80 mm clears every possible object or finger installation.
    retreat_mm = real.sim.parse_args([]).lift_mm
    if recover_from_current:
        if snapshot['tcp'][2]*1000 < config['min_open_tcp_z_mm']:
            raise ValueError('Manual open-hand recovery requires an elevated current pose')
        retreat_mm = 0.
    elif lifted_target is not None and not already_at_start:
        # Match the recorded finished lift using the existing arrival limits.
        # Height alone is not evidence that retreat already happened.
        real.require_at_execution_target(lifted_target, snapshot)
        if snapshot['tcp'][2]*1000 < config['min_open_tcp_z_mm']:
            raise ValueError('Completed lift is below the elevated reset region')
        retreat_mm = 0.
    if not already_at_start and retreat_mm:
        retreat = motion.finite6(snapshot['tcp']).copy()
        retreat[2] += retreat_mm / 1000
        real.inside(retreat, config)  # Never silently shorten/clip the retreat.
        goals.append(dict(phase='reset_retreat', kind='moveL', goal_tcp=retreat.tolist(),
                          q_nominal=list(snapshot['q']), closure_alpha=0., replaced_sample_count=0))
    goals.append(dict(phase='reset_return', kind='moveJ', goal_tcp=list(start['tcp']),
                      q_nominal=list(start['q']), closure_alpha=0., replaced_sample_count=0))
    return goals, already_at_start, retreat_mm


def run_reset(args, out, *, backend_factory=real.LiveBackend):
    """Return (exit code, new scan snapshot path); disconnect before perception."""
    if not args.reset_first or not args.capture_live:
        raise ValueError('Reset requires explicit --reset-first with --capture-live')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_reset/v1', status='starting',
        motion_confirmation_received=False, motion_commands_attempted=False,
        controller_script_requested=False,
        assumptions=[
            'Base means the saved elevated scan pose, not zero joints or the robot base origin.',
            'Open-loop opening; no RH56 sensor reads or grip/release verification.',
            'Object has moved: no old object pose, candidate or grasp approval is used.',
            'Model checks fixed desk and CB3; no object/clamp, opening-sweep or full self-collision certification.',
            'Operator must confirm supported object and clear opening, vertical retreat and return paths.',
            'Existing GGX lift distance is the requested retreat, not measured obstacle clearance.',
            'No perception until reset succeeds, devices disconnect and operator presses Enter again.',
            'No automatic retry, resume, extra release or grasp.',
        ])
    def checkpoint():
        (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        keys = ['status', 'motion_confirmation_received', 'motion_commands_attempted', 'controller_script_requested']
        with (out / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerow({key: report[key] for key in keys})
    checkpoint()
    backend, result, scan_path = None, 1, None
    try:
        from tools.preview_tabletop_target import load_perception_config
        config = real.load_config(args.real_config)
        real.require_hand_setting_authorization(config, args.allow_high_force)
        config = dict(config, **(config.get('reset_motion') or {}))
        start = load_start(args.start_snapshot, config)
        recover = bool(getattr(args, 'recover_from_current', False))
        lift_path = None if recover else getattr(args, 'completed_lift_report', None)
        lifted_target = completed_lift_target(lift_path) if lift_path is not None else None
        perception, _ = load_perception_config(args.config)
        source = desk_source(config, perception)
        model = support.build_model(source, include_object=False)
        assets = list((support.URDF.parent / 'meshes/ur5/collision').glob('*.stl'))
        assets += [p for p in (real.ROOT / 'h1_mujoco/inspire').rglob('*')
                   if p.is_file() and p.suffix.lower() in ('.xml', '.stl', '.obj')]
        pins = pin([__file__, real.__file__, motion.__file__, support.__file__, real.sim.__file__,
                    real.ROOT / 'tools/ggx_planning.py',
                    real.ROOT / 'tools/run_ggx_review.py', args.real_config, args.config,
                    args.start_snapshot, perception['desk_plane'], perception['calibration'], support.URDF,
                    real.ROOT / 'tools/run_vision_pick_cycle.py', real.ROOT / 'tools/run_vision_plan_trial.py',
                    real.ROOT / 'rh56_controller/rh56_hand.py', real.ROOT / 'rh56_controller/grasp_executor.py',
                    real.ROOT / 'rh56_controller/ur5_bridge.py', *assets])
        if lift_path is not None:
            pins.update(pin([lift_path]))
            report['completed_lift_reset'] = dict(report=str(lift_path), target=lifted_target,
                policy='omit_duplicate_retreat_only_at_recorded_finished_lift/v1',
                motion_authorized=False, current_pose_match_required=True)
        report.update(code_config_sha256=pins, start_target=start, desk_model=source,
                      opening_speed_raw=config['hand_speed_raw'], opening_force_raw=config['hand_force_raw'],
                      reset_motion={key: config[key] for key in ('joint_speed_rad_s', 'joint_accel_rad_s2',
                          'cartesian_speed_m_s', 'cartesian_accel_m_s2')})
        backend = backend_factory(config, out, report, checkpoint)
        backend.open_receiver()  # Read-only; no RTDE control script or hand yet.
        snapshot = backend.snapshot()
        report['entry_snapshot'] = snapshot
        report['manual_current_recovery'] = dict(requested=recover, opening_command_skipped=recover,
            policy='explicit supported-object/already-open operator confirmation; fresh current-pose preflight',
            old_lift_target_used=not recover and lifted_target is not None)
        checkpoint()
        goals, already, retreat_mm = reset_goals(snapshot, start, config, lifted_target=lifted_target,
                                                recover_from_current=recover)
        report.update(entry_snapshot=snapshot, goals=goals, already_at_start=already,
                      retreat_mm=0. if already else retreat_mm, status='awaiting_reset_confirmation')
        checkpoint()
        print(('GGX 人工恢复：保持已张开的手，不再发送张手指令 → ' if recover else 'GGX 复位：张手 → ') + ('保持当前拍摄起始位。' if already else
              '不额外上退；从当前姿态连续 MoveJ 返回保存的拍摄位。' if retreat_mm == 0 else
              f'垂直上退 {retreat_mm:g} mm → 连续 MoveJ 返回保存的拍摄起始位。'), flush=True)
        if start.get('pose_source') == 'nominal_wrist_flip_preset':
            wrist_delta = np.degrees(np.asarray(start['q'])[3:] - np.asarray(snapshot['q'])[3:])
            print(f'新腕分支：当前到拍摄目标的腕关节净变化约 {np.round(wrist_delta, 1).tolist()} deg。'
                  '首次从旧分支切换可能大幅转腕；先确认线缆余量，运动时不要用手整理线缆。'
                  '此数值不是路径最大转角，模型不检查线缆。', flush=True)
        print(f"目标 TCP（mm）：{np.round(np.array(start['tcp'][:3])*1000, 1).tolist()}；"
              f"直线 {config['cartesian_speed_m_s']} m/s；关节 {config['joint_speed_rad_s']} rad/s。\n"
              + ('本次恢复不发送手部命令；手已张开由操作者确认，无手部反馈。\n' if recover else
                 f"张手 raw speed={config['hand_speed_raw']}、force={config['hand_force_raw']}；无手部反馈。\n") +
              '新感知尚未开始，不使用旧盒子位置检查复位路径。请确认物体已有支撑、张手不会掉物，'
              '张手/上退/返回路径畅通，人员离开运动区，相机/桌面/安装未变，可立即急停。\n'
              + ('Enter 允许本次控制脚本启停、预检和返回；到位后会再次等待摆放确认。' if recover else
                 'Enter 允许本次控制脚本启停、预检，以及预检通过后的张手和复位；到位后会再次等待摆放确认。'), flush=True)
        if not real.confirm('确认手已完全张开、物体在桌上有支撑、当前返回路径与线缆畅通；执行此次恢复？' if recover else '执行这次复位？'):
            report['status'], result = 'cancelled_before_reset', 2
        else:
            report['motion_confirmation_received'] = True
            confirmed_at = time.monotonic()
            report['motion_confirmation_time_s'] = time.time()
            checkpoint()
            def timed(stage, operation):
                print(f'GGX reset: {stage}…', flush=True)
                begin = time.monotonic()
                record = dict(stage=stage, started_after_confirmation_s=begin-confirmed_at, status='running')
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
                    record['elapsed_s'] = time.monotonic()-begin
                    checkpoint()
                    print(f"GGX reset: {stage} {record['status']} ({record['elapsed_s']:.2f}s)", flush=True)
            verify(pins)
            # The prompt may have remained open while the arm settled. Bound
            # this change by the EXISTING arrival envelope, then rebuild from
            # a fresh stationary pose; never waive post-preflight drift checks.
            real.require_at_execution_target(snapshot, backend.snapshot())
            timed('connect_controller', backend.open_control)
            real.check_tcp_offset(backend.control)
            fresh = backend.snapshot()
            real.require_at_execution_target(snapshot, fresh)
            report['preflight_origin_refresh'] = dict(previous=snapshot,current=fresh,
                policy='After confirmation, rebuild reset from stationary pose; existing arrival envelope; no post-preflight waiver')
            snapshot = fresh
            goals, already, retreat_mm = reset_goals(snapshot, start, config, lifted_target=lifted_target,
                                                    recover_from_current=recover)
            report.update(preflight_snapshot=snapshot,goals=goals,already_at_start=already,
                          retreat_mm=0. if already else retreat_mm)
            checkpoint()
            checked = timed('preflight', lambda: motion.preflight_goals(backend, snapshot, goals, model, config, report))
            # Even at the saved pose, preflight the open hand's desk clearance;
            # skip the redundant return command only after that check passes.
            if already:
                checked = []
            command_path = out / 'reset_execution.json'
            command_path.write_text(json.dumps(dict(revision=motion.REVISION, commands=checked),
                                               indent=2, allow_nan=False)+'\n')
            report['execution_sha256'] = pin([command_path])
            report['status'] = 'resetting'
            checkpoint()
            verify(pins)
            verify(report['execution_sha256'])
            real.check_tcp_offset(backend.control)
            backend.held_origin = backend.snapshot()
            real.require_same_start(snapshot, backend.held_origin)
            if not recover:
                # Opening may shift load slightly. Use the existing bounded
                # arrival envelope around ONE fixed origin during opening,
                # then rebuild/recheck the arm path; never execute stale IK.
                backend.held_goal = snapshot
                timed('hand_open', lambda: backend.open_hand(support.raw_hand_command(model, 0.)))
                fresh = backend.snapshot()
                real.require_at_execution_target(snapshot, fresh)
                report['post_open_origin_refresh'] = dict(previous=snapshot, current=fresh,
                    policy='fixed existing arrival envelope during opening; full path re-preflight after settling')
                report['preopening_controller_preflight'] = report.get('controller_preflight', [])
                snapshot = fresh
                backend.held_goal = None
                backend.held_origin = snapshot
                goals, already, retreat_mm = reset_goals(snapshot, start, config, lifted_target=lifted_target)
                checked = timed('post_open_preflight', lambda: motion.preflight_goals(backend, snapshot, goals, model, config, report))
                if already: checked = []
                report.update(preflight_snapshot=snapshot, goals=goals, already_at_start=already,
                              retreat_mm=0. if already else retreat_mm)
                command_path.write_text(json.dumps(dict(revision=motion.REVISION, commands=checked),
                                                  indent=2, allow_nan=False)+'\n')
                report['execution_sha256'] = pin([command_path])
                checkpoint()
                verify(pins); verify(report['execution_sha256'])
                real.check_tcp_offset(backend.control)
                backend.held_origin = backend.snapshot()
                real.require_same_start(snapshot, backend.held_origin)
            for command in checked:
                print(f"GGX: {command['phase']} ({command['kind']}, one continuous motion)", flush=True)
                timed(command['phase'], lambda: backend.move_phase(command))
            final = backend.held_snapshot()
            # A nominal scan preset is an IK hint, not calibrated encoder data.
            # Compare arrival to the actually checked controller endpoint. Keep
            # the requested TCP and the <=0.15 rad nominal branch gate upstream.
            arrival_target=start
            if start.get('pose_source')=='nominal_wrist_flip_preset' and checked:
                arrival_target=dict(q=checked[-1]['q'],tcp=checked[-1]['tcp'])
                report['controller_resolved_start_target']=arrival_target
            real.require_at_execution_target(arrival_target, final)
            scan_path = out / 'scan_snapshot.json'
            scan_path.write_text(json.dumps(dict(final, schema='rh56_vision_pick_start/v1',
                ur_ip=config['ur_ip'], clearance_verified=False, routes_verified=False,
                source_reset_report=str(out / 'summary.json')), indent=2, allow_nan=False)+'\n')
            report.update(status='reset_completed', final_snapshot=final, scan_snapshot=str(scan_path))
            result = 0
    except (EOFError, KeyboardInterrupt):
        report['status'], result = 'interrupted', 130
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', error_traceback=traceback.format_exc())
        print(report['error'], flush=True)
    finally:
        if backend is not None:
            backend.close()
            if report.get('cleanup_errors'):
                report['status'], result = 'cleanup_failed', 1
        checkpoint()
        print(f"GGX reset: {report['status']}；{out / 'summary.json'}", flush=True)
    return result, scan_path if result == 0 else None
