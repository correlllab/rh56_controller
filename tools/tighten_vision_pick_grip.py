#!/usr/bin/env python3
"""One bounded margin-1/force-100 -> margin-2/force-1000 grip adjustment.

Offline preview by default. Reuse the completed Plan's target and final pose,
never its open approach. No scan, release, lift, speed write or automatic retry.
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

from tools.run_vision_pick_cycle import LiveCycle, Plan, dump, finite, load_config, stationary_snapshot

CONFIRMATION = 'TIGHTEN CURRENT GRASP'


def validate_checkpoint(c, previous):
    """This operation is only for a completed closure interrupted before lift."""
    events = {(e['stage'], e['event']) for e in previous.get('events', [])}
    if (previous.get('schema') != 'rh56_vision_pick_cycle/v1'
            or previous.get('status') != 'failed' or previous.get('stage') != 'lift'
            or not {('grasp', 'complete'), ('settle', 'complete')} <= events
            or ('lift', 'complete') in events or previous.get('cleanup_errors')):
        raise ValueError('Expected completed closure with lift blocked and clean shutdown')
    old = previous['config']
    for key in set(c) | set(old):
        if key not in ('closure_margin_mm', 'hand_force_raw') and c.get(key) != old.get(key):
            raise ValueError(f'Grasp configuration changed: {key}')
    if (old['closure_margin_mm'], c['closure_margin_mm'], old['hand_force_raw'], c['hand_force_raw']) != (1, 2, 100, 1000):
        raise ValueError('Only the requested margin 1->2 mm, force 100->1000 adjustment is supported')
    for key in ('calibration', 'desk_plane', 'local_alignment', 'alignment_validation'):
        if hashlib.sha256(Path(c[key]).read_bytes()).hexdigest() != previous.get('source_sha256', {}).get(key):
            raise ValueError(f'Source changed: {key}')


def check_anchor(state, tcp, q=None):
    from tools.run_vision_plan_trial import rot_error
    actual, target = finite(state['tcp'], (6,)), finite(tcp, (6,))
    if (np.linalg.norm(actual[:3]-target[:3]) > .001
            or rot_error(actual, target) > np.deg2rad(.5)
            or (q is not None and np.max(np.abs(finite(state['q'], (6,))-q)) > .01)):
        raise RuntimeError('UR5 moved away from the completed grasp anchor; no automatic repositioning')


def validate_delta(old_tcp, new_tcp, old_command, new_command):
    from tools.run_vision_plan_trial import rot_error
    a, b = finite(old_tcp, (6,)), finite(new_tcp, (6,))
    old, new = finite(old_command, (6,)), finite(new_command, (6,))
    delta = (b[:3]-a[:3])*1000
    angle = float(np.rad2deg(rot_error(a, b)))
    if (np.linalg.norm(delta) > 4 or angle > 1 or not np.all(new <= old)
            or np.any(new < 0) or np.any(old > 1000)
            or old[0] != new[0] or old[5] != new[5] or np.max(old-new) > 10):
        raise ValueError('Adjustment is not the bounded nearby closure')
    return delta.tolist(), angle


def apply_adjustment(live, new_tcp, new_command):
    """Same arm-first/hand-second ordering as a Plan closure step."""
    live.report['stage'] = 'force'
    live.report['hardware_commands_attempted'] = True
    live.hand.force_set([1000]*6)
    live.report['force_write_completed'] = True
    live.wait_stationary(.5)  # Prior target may resume; does not verify the hand.
    live.report['stage'] = 'wrist'
    live.move_tcp(new_tcp)
    live.report['stage'] = 'tighten'
    live.hand.angle_set(new_command)
    live.report['angle_write_completed'] = True
    live.wait_stationary(.5)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Cycle summary after completed closure, before lift')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--confirm-adjust', help=f'Exact confirmation: {CONFIRMATION}')
    p.add_argument('--confirm-high-force', action='store_true')
    p.add_argument('--acknowledge-open-loop', action='store_true')
    p.add_argument('--acknowledge-shape-extrapolation', action='store_true')
    a = p.parse_args(argv)
    if a.execute and not (a.confirm_adjust == CONFIRMATION and a.confirm_high_force
                          and a.acknowledge_open_loop and a.acknowledge_shape_extrapolation):
        p.error('Execution requires exact --confirm-adjust and all three acknowledgements')
    return a


def main(argv=None):
    args = parse_args(argv)
    c = load_config(args.config)
    previous = json.loads(args.checkpoint.read_text())
    validate_checkpoint(c, previous)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_vision_grip_adjustment/v1', status='preview', stage='geometry',
        config=c, checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        hardware_commands_attempted=False, force_write_completed=False, angle_write_completed=False,
        automatic_lift=False, automatic_release=False, automatic_retry=False,
        physical_grip_verified=False, speed_write_sent=False,
        assumptions=['User reported the same cube loosely held at the completed grasp; no hand readback.',
                     'Use frozen target from completed grasp; no camera redetection or open approach.',
                     'Margin 2 mm is total; this adds only 1 mm to the prior closure margin.',
                     'Force 1000 is raw setting, not measured contact force; stalled fingers may resume.',
                     'Unchanged local mount/width fit is extrapolated to the narrower higher-force shape.',
                     'Same Plan final solution and arm-first/hand-second sequence; no model changes.',
                     'IK and bounded TCP travel do not certify finger/cube/table collision clearance.'])
    plan = old_plan = live = None
    failed = True
    try:
        # Recompute both endpoints with the same model; refuse changed provenance.
        old_plan = Plan(previous['config'], Path(previous['target_validation']))
        if (old_plan.commands != previous['plan']['hand_commands']
                or not np.allclose(old_plan.tcp, previous['plan']['tcp_waypoints'], rtol=0, atol=1e-10)):
            raise ValueError('Original Plan no longer matches the completed trial')
        plan = Plan(c, Path(previous['target_validation']))
        old_tcp, new_tcp = old_plan.tcp[-1], plan.tcp[-1]
        old_command, new_command = old_plan.commands[-1], plan.commands[-1]
        delta, angle = validate_delta(old_tcp, new_tcp, old_command, new_command)
        report.update(old_tcp=old_tcp, new_tcp=new_tcp, old_hand_command=old_command,
                      new_hand_command=new_command, tcp_delta_base_mm=delta,
                      rotation_change_deg=angle, plan=plan.report)
        print(json.dumps(report, indent=2), flush=True)
        if not args.execute:
            failed = False
            return 0
        from tools.calibrate_ur5_external_camera_real import _open_rtde_receiver
        from tools.run_vision_plan_trial import CheckedBridge, SerialLog
        from rh56_controller.rh56_hand import RH56Hand
        receiver = _open_rtde_receiver(c['ur_ip'])
        try:
            snapshot = stationary_snapshot(receiver)
            check_anchor(snapshot, old_tcp)
        finally:
            receiver.disconnect()
        # Local envelope inside the existing workspace, not the opening/scan route.
        local = dict(c)
        bounds = finite(c['tcp_bounds_mm'], (2, 3)).copy()
        anchor_mm = np.array(old_tcp[:3])*1000
        bounds[0] = np.maximum(bounds[0], anchor_mm-5)
        bounds[1] = np.minimum(bounds[1], anchor_mm+5)
        local.update(tcp_bounds_mm=bounds.tolist(), max_rotation_deg=2)
        live = LiveCycle(local, args, report)
        live.speed = .1 if c['speed_profile'] == 'planner' else .01
        live.bridge = CheckedBridge(ip=c['ur_ip'], speed=live.speed,
                                    accel=.2 if c['speed_profile'] == 'planner' else .02)
        report['controller_script_requested'] = True
        if not live.bridge.connect():
            raise RuntimeError('UR5 connection failed')
        check_anchor(stationary_snapshot(live.bridge._iface.recv), snapshot['tcp'], snapshot['q'])
        live.preflight([new_tcp], [5])
        live.timeout_s = 10
        check_anchor(stationary_snapshot(live.bridge._iface.recv), snapshot['tcp'], snapshot['q'])
        report['entry_snapshot'] = snapshot
        live.telemetry = (args.out/'telemetry.csv').open('w', newline='')
        live.telemetry_writer = csv.writer(live.telemetry)
        live.telemetry_writer.writerow(['time_s','stage','x','y','z','rx','ry','rz','safety_mode'])
        live.hand = RH56Hand(c['hand_port'], hand_id=1)
        live.hand.ser.exclusive = True
        live.serial_log = (args.out/'hand_serial.jsonl').open('w')
        live.hand.ser = SerialLog(live.hand.ser, live.serial_log)
        dump(args.out/'summary.json', report)
        print('Preflight passed: force 1000, small Plan wrist compensation, margin 2; no opening/lift.', flush=True)
        apply_adjustment(live, new_tcp, new_command)
        report['final_snapshot'] = stationary_snapshot(live.bridge._iface.recv)
        check_anchor(report['final_snapshot'], new_tcp)
        report['status'] = 'adjustment_commands_completed_grip_unverified'
        failed = False
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if live is not None:
            live.close(failed)
        for p in (plan, old_plan):
            if p is not None:
                p.close()
        dump(args.out/'summary.json', report)
        with (args.out/'summary.csv').open('w', newline='') as f:
            keys = ['status','stage','hardware_commands_attempted','force_write_completed',
                    'angle_write_completed','physical_grip_verified','automatic_lift']
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerow({k:report[k] for k in keys})
        print(f"Adjustment: {report['status']}; report: {args.out/'summary.json'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
