#!/usr/bin/env python3
"""Single empty-hand Plan shape for manual gap measurement; offline by default.

No arm control interface, sweep, automatic opening/retry or feedback queries.
Execution uses receive-only UR state guarding and logs hand raw TX/RX.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def shape(width_mm):
    from rh56_controller.grasp_viz_core import GraspVizCore
    from rh56_controller.grasp_executor import GraspExecutor
    if not np.isfinite(width_mm) or not 95 <= width_mm <= 105:
        raise ValueError('This diagnostic is restricted to internal widths 95..105 mm')
    core = GraspVizCore(robot_mode=True, mink_viz=False, hand_write_only=True)
    try:
        result = core.closure.solve(core._mode, width_mm/1000)
        angles = GraspExecutor(None, None, core.fk, hand_write_only=True)._ctrl_to_real(result.ctrl_values)
        return dict(input_reference_width_mm=width_mm, returned_centroid_width_mm=result.width*1000,
                    angles=angles, ctrl_values=result.ctrl_values,
                    strategy='original plane4 Plan final shape only; wrist does not move')
    finally:
        core.cleanup()


def send_once(hand, angles, guard, pause=time.sleep):
    """Three explicit writes, no sensing retry or automatic error clearing."""
    guard()
    hand.force_set([100]*6)
    pause(.125)
    guard()
    hand.speed_set([100]*6)
    pause(.125)
    guard()
    hand.angle_set(angles)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner-width-mm', type=float, default=100.)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ur5_iterative_cube_lab.yaml',
                        help='Existing hardware identities and TCP bounds; does not change the preset')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-empty-clear', action='store_true',
                        help='Operator confirmed empty hand and clear finger path; permit single shape write')
    parser.add_argument('--start-snapshot', type=Path,
                        help='Fresh receive-only stationary UR snapshot required for execution')
    args = parser.parse_args(argv)
    if args.execute and (not args.confirm_empty_clear or args.start_snapshot is None):
        parser.error('Execution requires --confirm-empty-clear and --start-snapshot')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_plan_airgap/v1', status='starting', arm_motion_commands_sent=False,
                  hand_commands_attempted=False, measured_gap_mm=None,
                  physical_hand_completion_verified=False, force_raw=100, speed_raw=100,
                  assumptions=['Operator-confirmed empty hand and clear finger path; not an object grasp.',
                               'Arm stays fixed; no RTDE control script or wrist compensation.',
                               'Internal site distance is not physical contact-pad gap.',
                               'Write-only hand: missing ACK accepted; force/speed application and pose are not verified.',
                               'Five-second dwell is not motion feedback; operator must observe stability before measuring.',
                               'No retry, automatic release, error clearing, or higher force. Target may persist after exit.'])
    receiver = hand = log = None
    rc = 0
    try:
        from tools.run_vision_pick_cycle import (load_config, stationary_snapshot,
                                               check_stationary_origin, within)
        c = load_config(args.config)
        report['shape'] = shape(args.planner_width_mm)
        report['hardware'] = dict(ur_ip=c['ur_ip'], hand_port=c['hand_port'])
        print(json.dumps(report, indent=2), flush=True)
        if not args.execute:
            report['status'] = 'offline_preview'
        else:
            from tools.calibrate_ur5_external_camera_real import _open_rtde_receiver
            from tools.run_vision_plan_trial import SerialLog
            from rh56_controller.rh56_hand import RH56Hand
            expected = json.loads(args.start_snapshot.read_text())
            age = time.time()-expected['time_s']
            if not 0 <= age <= 300 or expected.get('ur_ip') != c['ur_ip']:
                raise ValueError('Snapshot must match UR and be less than five minutes old')
            receiver = _open_rtde_receiver(c['ur_ip'])
            origin = stationary_snapshot(receiver)
            check_stationary_origin(expected, origin)
            report['initial_arm_state'] = origin
            report['arm_samples'] = []

            def guard():
                sample = stationary_snapshot(receiver)
                check_stationary_origin(origin, sample)
                within(np.asarray(sample['tcp'][:3])*1000, c['tcp_bounds_mm'], 'Stationary air-gap TCP')
                report['arm_samples'].append(sample)

            guard()
            hand = RH56Hand(c['hand_port'], hand_id=1)
            hand.ser.exclusive = True
            log = (args.out/'hand_serial.jsonl').open('w')
            hand.ser = SerialLog(hand.ser, log)
            report['hand_commands_attempted'] = True
            send_once(hand, report['shape']['angles'], guard)
            report['angle_write_returned'] = True
            begin = time.monotonic()
            while time.monotonic()-begin < 5.:
                guard()
                time.sleep(.05)
            report['status'] = 'single_shape_sent_awaiting_manual_measurement'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        rc = 1
    finally:
        errors = []
        for resource, method in ((hand.ser if hand is not None else None, 'close'),
                                 (log,'close'), (receiver,'disconnect')):
            if resource is not None:
                try:
                    getattr(resource,method)()
                except Exception as exc:
                    errors.append(str(exc))
        report['cleanup_errors'] = errors
        if errors:
            report['status'] = 'failed'
            rc = 1
        (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (args.out/'summary.csv').open('w',newline='') as stream:
            keys = ['status','arm_motion_commands_sent','hand_commands_attempted','measured_gap_mm']
            writer = csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader(); writer.writerow({k:report[k] for k in keys})
        print(f'Air-gap check: {report["status"]}; report: {args.out / "summary.json"}',flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
