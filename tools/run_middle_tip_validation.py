#!/usr/bin/env python3
"""Review/execute one frozen middle-tip pointing target; hardware-free by default.

--controller-preflight starts/stops the RTDE control script but sends no motion.
--execute additionally requires Enter approval of the actual controller path,
then opens the hand at low settings, waits for visual confirmation and moves
once. No closure, lift, retry, calibration fitting or automatic return.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools import prepare_middle_tip_validation as pointer
from tools import ggx_real_support as support, ggx_motion as motion
from tools import run_ggx_real as real, run_ggx_review as review
from tools import preflight_ggx_ur5_table as sim, ggx_template as files


def load_plan(path):
    plan = review.read_json(path)
    if (plan.get('schema') != 'rh56_middle_tip_validation_plan/v1'
            or plan.get('status') != 'nominal_path_passed_needs_controller_review'
            or plan.get('hardware_motion_commands_sent') is not False):
        raise ValueError('Require a passing, offline-only middle-tip plan')
    review.verify(plan['source_sha256'])
    if (plan['hand_ctrl'] != sim.GGX_OPEN_CTRL
            or plan['hand_raw'] != [1000, 1000, 1000, 1000, 1000, 0]):
        raise ValueError('Frozen hand shape differs from standard open')
    motion.finite6(plan['goal_q'])
    motion.finite6(plan['desired_tcp'])
    return plan


def pointer_goal(plan):
    # Existing checked moveJ primitive; reset phase intentionally omits an
    # object proxy because this is the operator-confirmed empty desk target.
    return dict(phase='reset_return', kind='moveJ', goal_tcp=plan['desired_tcp'],
        q_nominal=plan['goal_q'], closure_alpha=0., hand_ctrl=plan['hand_ctrl'],
        replaced_sample_count=0)


def render_checked(model, commands, plan, config, out):
    samples = []
    data = sim.mujoco.MjData(model)
    gaps = []
    for command in commands:
        if command['hand_ctrl'] != plan['hand_ctrl']:
            raise ValueError('Controller review hand shape changed')
        for row in command['samples']:
            qpos = sim.initial_qpos(model, np.asarray(row['q']))
            sim.set_hand_qpos(model, qpos, plan['hand_ctrl'])
            data.qpos[:] = qpos
            sim.mujoco.mj_forward(model, data)
            gap, nearest = sim.desk_clearance(model, data)
            if motion.desk_clearance_blocked(gap, motion.reflex_desk_clearance_m(config)):
                raise RuntimeError('Controller-resolved pointer path fails desk clearance')
            gaps.append(gap)
            samples.append(sim.TrajectorySample('middle_tip_position', row['fraction'], qpos, 0.,
                min_desk_clearance_m=gap, nearest_geom=nearest, desk_collision=False))
    if not samples:
        raise ValueError('No checked samples')
    minimum = min(gaps)
    args = sim.parse_args(['--no-video'])
    args.camera_distance = 1.1
    result = sim.CandidateResult(0, 'Middle tip A', 0., True, 'pointer_path_checked_not_executed',
        None, None, minimum*1000, samples[int(np.argmin(gaps))].nearest_geom, '', len(samples))
    # Show a steady final frame after the travel preview; these repeats are
    # video-only and do not introduce hardware commands.
    path = out/'execution_review.mp4'
    sim.write_video(path, model, samples+[samples[-1]]*30, result, args,
                    np.asarray(plan['measurement_target']['desired_tip_base_m']))
    sim._write_csv(out/'trajectory.csv', sim._trajectory_rows(model, samples))
    return path, minimum*1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--controller-preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-controller', action='store_true', help='Explicit permission for RTDE script start/stop and nonmotion queries')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if (args.controller_preflight or args.execute) and not args.confirm_controller:
        parser.error('Hardware modes require --confirm-controller; motion also requires interactive Enter')
    out = args.out or ROOT/'artifacts/middle_tip_measurements'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_middle_tip_measurement/v1', status='starting',
        execute_requested=args.execute, controller_preflight_requested=args.controller_preflight,
        motion_confirmation_received=False, motion_commands_attempted=False,
        physical_point_error_mm=None, physical_measurement_status='not_measured',
        assumptions=['Same frozen vision/model target; no adjustment to match a measured error.',
            'Known desk only. Operator must confirm camera/table fixed, empty hand, straight fingers, clear path and cables.',
            'Standard open raw command at recovery speed/force 100. No sensor readback or automatic retry.',
            'Reported reconstructed tip is model-based, NOT independent measured accuracy.',
            'One slow checked moveJ and hold; no closing, extra lifting or return.',
            'No self-collision or unknown obstacle certification from nominal desk checks.'])
    def save():
        files.write_json(out/'summary.json', report)
    backend = None
    try:
        plan = load_plan(args.plan)
        config = real.load_config(Path(plan['config']))
        config.update(hand_speed_raw=100, hand_force_raw=100,
            joint_speed_rad_s=.10, joint_accel_rad_s2=.10)
        report.update(plan=str(args.plan.resolve()), measurement_target=plan['measurement_target'],
            effective_config=config, source_sha256={**plan['source_sha256'],
                **review.pin([args.plan, __file__, real.__file__, motion.__file__])})
        save()
        if not (args.controller_preflight or args.execute):
            report['status'] = 'inputs_validated_no_hardware'
            return 0
        backend = real.LiveBackend(config, out, report, save)
        backend.open_receiver()
        snapshot = backend.snapshot()
        report['initial_snapshot'] = snapshot
        model = support.build_model(plan['scene'], include_object=False)
        backend.open_control()
        real.check_tcp_offset(backend.control)
        commands = motion.preflight_goals(backend, snapshot, [pointer_goal(plan)], model, config, report)
        path, gap = render_checked(model, commands, plan, config, out)
        files.write_json(out/'execution.json', dict(commands=commands, target=plan['measurement_target']))
        report.update(execution_video=str(path), minimum_desk_clearance_mm=gap,
            execution_sha256=review.pin([path, out/'execution.json']), status='controller_path_checked_no_motion')
        save()
        if not args.execute:
            return 0
        review.show(path, True)
        print('A 点；掌心朝下，中指尖垂直投影对准十字。'
              f'本轮高度={plan["measurement_target"]["height_mm"]:g} mm；一次 moveJ，0.10 rad/s。\n'
              '确认桌面/相机未变、空手且张手与完整路径/线缆畅通，人员离开运动区，可立即急停。', flush=True)
        if not real.confirm('批准视频：低速张手，确认手型后移动到测量点？'):
            report['status'] = 'operator_cancelled_before_hand_command'
            return 0
        report['motion_confirmation_received'] = True
        review.verify(report['source_sha256']); review.verify(report['execution_sha256'])
        real.check_tcp_offset(backend.control)
        backend.held_origin = backend.snapshot()
        real.require_same_start(snapshot, backend.held_origin)
        backend.open_hand(plan['hand_raw'])
        if not real.confirm('请目视确认手指已张开伸直、没有卡住；Enter 执行已批准路径，其他输入停止：'):
            report['status'] = 'operator_cancelled_after_hand_open'
            return 0
        backend.move_phase(commands[0])
        final = backend.snapshot()
        reconstructed = pointer.UR5Bridge.tcp_T_to_hand_T(support.pose_matrix(final['tcp'])) @ np.r_[plan['reference_tip_in_hand_m'], 1.]
        report.update(status='at_measurement_pose_physical_error_pending', final_snapshot=final,
            reconstructed_tip_base_m=reconstructed[:3].tolist(),
            physical_measurement_status='awaiting_independent_operator_measurement')
        print('到位，保持姿态；没有自动返回。请在机械臂静止后测量垂直投影的水平偏差及实际高度。', flush=True)
        return 0
    except BaseException as exc:
        report.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], flush=True)
        return 1
    finally:
        if backend is not None:
            backend.close()
        save()
        print(f'Middle tip: {report["status"]}; {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
