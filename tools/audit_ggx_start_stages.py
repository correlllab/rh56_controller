#!/usr/bin/env python3
"""Offline attribution of fixed starts versus positioning paths in a saved run.

Recompute endpoint/prefix checks using the unchanged production geometry and
first-solution IK policy. Preserve historical downstream results; do not search
for replacement routes, move points, infer new candidates, or contact devices.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Saved ten-point experiment directory')
    parser.add_argument('--prepared-point', default='P2', choices=[f'P{i}' for i in range(1, 11)],
                        help='Point with prepared.json containing the exact saved scene and scan snapshot')
    parser.add_argument('--out', type=Path)
    return parser.parse_args(argv)


def rejection_kinds(value):
    if isinstance(value, dict):
        if 'kind' in value:
            yield (value['kind'], value.get('phase'))
        for child in value.values():
            yield from rejection_kinds(child)
    elif isinstance(value, list):
        for child in value:
            yield from rejection_kinds(child)


def audit(args, out, report):
    import mujoco
    import numpy as np
    from tools import ggx_planning as planning, ggx_motion as motion
    from tools import ggx_real_support as support, ggx_template as template
    from tools import preflight_ggx_ur5_table as sim
    from tools.run_ggx_review import read_json, pin, verify, digest

    run = args.run.resolve()
    inputs = [run/'summary.json', run/'fixed_layout.json', run/args.prepared_point/'prepared.json']
    original, layout, prepared = map(read_json, inputs)
    source = prepared['source']
    if original['rank'] != prepared['rank'] or source['rank'] != original['rank']:
        raise ValueError('Saved run/scene/rank mismatch')
    if prepared.get('policy') != planning.policy():
        raise ValueError('Saved geometry/path policy differs from current implementation')
    # These wrappers changed only the UI/selection flow after the saved run.
    # They are NOT invoked by this diagnostic. Every other historical dependency,
    # including all geometry, IK, model, calibration and scene data, must match.
    wrappers = {str(ROOT/'tools/run_ggx_review.py'), str(ROOT/'tools/run_ggx_experiment.py')}
    drift = []
    for name, expected in source['pins'].items():
        actual = digest(name)
        if actual != expected:
            if name not in wrappers:
                raise ValueError(f'Historical geometry/data dependency changed: {name}')
            drift.append(dict(path=name, historical_sha256=expected, current_sha256=actual,
                              scope='Unused UI/orchestration wrapper; no historical approval inherited'))
    pins = {**pin(source['pins']), **pin([*inputs, __file__])}
    report.update(input_sha256=pins, historical_wrapper_drift=drift, rank=original['rank'],
                  scene=source['preflight'], snapshot=prepared['snapshot'], policy=planning.policy())
    model = support.build_model(source)
    config = source['trial_config']
    sim_args = template.sim_args(source)
    sim_args.ik_position_tol_mm = sim_args.ik_orientation_tol_deg = .05
    initial = sim.initial_qpos(model, np.asarray(prepared['snapshot']['q']))
    data = mujoco.MjData(model)
    historical = {t['point_id']: t for t in original['trials']}
    for point in layout['points']:
        old = historical[point['id']]
        if old['rank'] != original['rank'] or old['source_geometry'] != source['template_relocation']:
            raise ValueError('Trials do not share the same rank and relocated object scene')
        hint = sim.initial_qpos(model, np.asarray(point['q_nominal']))
        hand = support.UR5Bridge.tcp_T_to_hand_T(support.pose_matrix(point['tcp']))
        attempts = []
        q = next(planning.solutions(model, sim.target_eeff_transform(hand[:3, 3], hand[:3, :3]),
                                    sim_args, attempts, [hint, initial]), None)
        row = dict(point_id=point['id'], rank=original['rank'], ik_found=q is not None,
                   endpoint_pass=False, prefix_pass=False, historical_status=old['status'],
                   historical_reason=old.get('planning_reason'),
                   historical_failure_kinds=[dict(kind=k, phase=p) for k, p in
                       sorted(set(rejection_kinds(old.get('planning_attempts', []))), key=str)],
                   ik_attempts=attempts)
        if q is None:
            row['classification'] = 'start_ik_unresolved'
        else:
            endpoint = [sim.TrajectorySample('position_to_trial_start', 1., q.copy(), 0.)]
            endpoint_failures = planning.geometry_failures(model, endpoint, config, target_gate=True)
            prefix = [sim.TrajectorySample('start', 0., initial.copy(), 0.)]
            sim._append_joint_segment(model, prefix, initial, q, phase='position_to_trial_start',
                                      max_step_rad=np.radians(sim_args.path_step_deg), closure_alpha=0.)
            prefix_failures = planning.geometry_failures(model, prefix, config, target_gate=True)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            gap, nearest = motion.object_clearance(model, data)
            actual_tcp = support.fk_tcp(model, q[planning.arm_addresses(model)])
            pe, re = support.pose_error(actual_tcp, point['tcp'])
            row.update(endpoint_pass=not endpoint_failures, prefix_pass=not prefix_failures,
                       target_gap_mm=float(gap*1000), nearest_target_geom=nearest,
                       desk_gap_mm=float(endpoint[0].min_desk_clearance_m*1000),
                       endpoint_failures=endpoint_failures, prefix_failures=prefix_failures,
                       prefix_samples=len(prefix), tcp_error_mm=pe, tcp_error_deg=re,
                       q=q[planning.arm_addresses(model)].tolist(), tcp=actual_tcp)
            if endpoint_failures:
                row['classification'] = 'start_geometry_rejected'
            elif prefix_failures:
                row['classification'] = 'positioning_only_rejected'
            elif old['status'] == 'offline_prepared':
                row['classification'] = 'historical_full_route_pass'
            else:
                row['classification'] = 'historical_post_start_rejected'
            if old.get('planning_reason') == 'trial_positioning_unavailable' and not prefix_failures:
                raise ValueError(f'{point["id"]}: historical positioning rejection did not reproduce')
            if old.get('planning_reason') != 'trial_positioning_unavailable' and prefix_failures:
                raise ValueError(f'{point["id"]}: new unexpected positioning rejection')
        report['points'].append(row)
        print(f'{row["point_id"]}: {row["classification"]}; '
              f'endpoint={row["endpoint_pass"]}, scan-to-start={row["prefix_pass"]}, '
              f'target_gap_mm={row.get("target_gap_mm")}', flush=True)
    verify(pins)
    report.update(status='completed', counts={name: sum(r['classification'] == name for r in report['points'])
        for name in ('start_ik_unresolved', 'start_geometry_rejected', 'positioning_only_rejected',
                     'historical_post_start_rejected', 'historical_full_route_pass')})


def main(argv=None):
    import json
    args = parse_args(argv)
    out = (args.out or ROOT/'artifacts/ggx_start_stage_audits'/
           datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_start_stage_audit/v1', status='running', executable=False,
                  hardware_contacted=False, motion_authorized=False, points=[], assumptions=[
        'Saved nominal scene, scan snapshot, open hand and fixed point poses; no live measurements.',
        'Same first-solution start IK and sampled moveJ prefix as production, unchanged thresholds.',
        'Endpoint target overlap uses the existing enclosing box proxy, not physical contact evidence.',
        'No search for alternate positioning paths; downstream results are historical, not new solves.',
        'No camera mount/unknown obstacles or full self-collision certification; not a motion certificate.',
        'Non-consumable diagnostic only; original files, hashes and approvals remain untouched.'])
    try:
        audit(args, out, report)
        return 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr)
        return 1
    finally:
        (out/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        fields = ['point_id', 'rank', 'classification', 'endpoint_pass', 'prefix_pass',
                  'target_gap_mm', 'desk_gap_mm', 'nearest_target_geom', 'historical_status', 'historical_reason']
        with (out/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['points'])
        print(f'Offline start-stage audit: {report["status"]}; {out/"summary.json"}')


if __name__ == '__main__':
    raise SystemExit(main())
