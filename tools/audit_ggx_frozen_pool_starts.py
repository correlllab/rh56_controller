#!/usr/bin/env python3
"""Offline counterfactual: frozen GGX top-100 versus one historical template.

Reuse saved official/local screening, then try locally eligible candidates in
original rank order at each saved start. No inference, controller, or execution.
This audits candidate-pool coverage UNDER THE EXISTING CHECKER, not global IK
feasibility or physical success. Historical results and approvals are untouched.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--pool', type=Path, required=True, help='Completed top-100 local audit summary.json')
    p.add_argument('--archive-files', type=Path, required=True, help='Evidence archive files/ directory')
    p.add_argument('--prepared-point', default='P2')
    p.add_argument('--out', type=Path, required=True)
    return p.parse_args(argv)


def validate_pool(pool, official):
    rows = pool['feasibility_stages']
    if [r['rank'] for r in rows] != list(range(100)):
        raise ValueError('Require complete original top-100, in rank order')
    if [r['rank'] for r in official['candidates']] != list(range(100)):
        raise ValueError('Official filter must cover the same top-100')
    for a, b in zip(rows, official['candidates']):
        if a['official_scene_clear'] != b['official_scene_clear'] or a['confidence'] != b['confidence']:
            raise ValueError('Official/local rank lineage mismatch')
    eligible = [r['rank'] for r in rows if r['local_state'] == 'geometry_eligible']
    if eligible != pool['eligible_ranks']:
        raise ValueError('Eligible-rank ledger mismatch')
    if any(not rows[i]['official_scene_clear'] for i in eligible):
        raise ValueError('Local eligible candidate failed official screen')
    return rows


def select_first(rows, attempt):
    """Full top-100 ledger; stop path search after first pass, never rerank."""
    selected, ledger = None, []
    for row in rows:
        entry = dict(rank=row['rank'], confidence=row['confidence'], local_state=row['local_state'])
        if row['local_state'] != 'geometry_eligible':
            entry['status'] = 'saved_screen_rejection'
        elif selected is not None:
            entry['status'] = 'not_tested_after_first_pass'
        else:
            entry.update(attempt(row['rank']))
            if entry['status'] == 'nominal_route_pass':
                selected = row['rank']
        ledger.append(entry)
    return selected, ledger


def runtime_pins(historical, archive, report):
    """Validate route dependencies; load exact archived motion code if changed.

    Unused launchers/reset/layout builders are not called. Their changes are
    disclosed, not silently re-pinned as a historical execution approval.
    """
    import tools
    unused = {str(ROOT/'tools'/name) for name in (
        'run_ggx_review.py', 'run_ggx_experiment.py', 'ggx_reset.py', 'ggx_fixed_starts.py')}
    pins, drift = {}, []
    for name, expected in historical.items():
        actual = digest(name)
        if actual == expected:
            pins[name] = expected
            continue
        record = dict(path=name, historical_sha256=expected, current_sha256=actual)
        if name in unused:
            record['handling'] = 'Not invoked by offline audit (JSON/hash helpers only for review wrapper)'
            pins[name] = actual
        elif name == str(ROOT/'tools/ggx_motion.py'):
            frozen = (archive/'tools/ggx_motion.py').resolve()
            if digest(frozen) != expected:
                raise ValueError('Archive does not contain exact historical motion dependency')
            spec = importlib.util.spec_from_file_location('tools.ggx_motion', frozen)
            module = importlib.util.module_from_spec(spec)
            sys.modules['tools.ggx_motion'] = module
            spec.loader.exec_module(module)
            tools.ggx_motion = module
            pins[str(frozen)] = expected
            record.update(handling='Exact historical module loaded for this offline process only', archive=str(frozen))
        else:
            raise ValueError(f'Historical route dependency changed: {name}')
        drift.append(record)
    report.update(historical_source_sha256=historical, route_runtime_sha256=pins, dependency_drift=drift)
    return pins


def checkpoint(out, report):
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['point_id', 'historical_status', 'rank20_replay_pass',
                                                   'selected_rank', 'nominal_pool_pass', 'recovered_start'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in report['points'])


def audit(args, report):
    import numpy as np
    from tools import ggx_planning as planning, ggx_real_support as support

    run, pool_path = args.run.resolve(), args.pool.resolve()
    saved = read(run/args.prepared_point/'prepared.json')
    original, layout, pool = read(run/'summary.json'), read(run/'fixed_layout.json'), read(pool_path)
    official = read(pool['official_scene_filter'])
    rows = validate_pool(pool, official)
    if saved['policy'] != planning.policy() or pool['policy'] != planning.policy():
        raise ValueError('Historical local/route policy mismatch')
    if original['rank'] != 20 or saved['rank'] != 20:
        raise ValueError('This historical diagnostic expects the sugar rank-20 baseline')
    pins = runtime_pins(saved['source']['pins'], args.archive_files.resolve(), report)
    # Import after the frozen motion module is installed, so template uses it too.
    from tools import ggx_template as template

    paths = [pool_path, run/'summary.json', run/'fixed_layout.json', run/args.prepared_point/'prepared.json',
             run/'reference/prepared.json', Path(pool['official_scene_filter']), Path(official['candidate_file']), Path(__file__)]
    if digest(official['candidate_file']) != official['candidate_sha256']:
        raise ValueError('Frozen candidate file changed')
    base_reference = read(run/'reference/prepared.json')
    prepared_sources = {}
    for row in rows:
        if row['local_state'] != 'geometry_eligible':
            continue
        path = pool_path.parent/f"sim_rank_{row['rank']:03d}"/'summary.json'
        local = read(path); paths.append(path)
        if local['candidate_yaml_sha256'] != official['candidate_sha256']:
            raise ValueError('Candidate-file mismatch in local reference')
        if local['geometry_review_ranks'] != [row['rank']] or local['policy'] != planning.policy():
            raise ValueError('Local reference rank/policy mismatch')
        for key in ('object_center_m', 'object_size_mm', 'object_yaw_deg', 'desk_top_m', 'desk_envelope'):
            if local[key] != base_reference['source']['preflight'][key]:
                raise ValueError(f'Local reference scene differs: {key}')
        if row['rank'] == 20 and local['local_reference']['rows'] != base_reference['rows']:
            raise ValueError('Rank-20 reference changed')
        source = copy.deepcopy(saved['source'])
        source.update(rank=row['rank'], approval=None, source_kind='offline_pool_counterfactual', pins=pins)
        source['preflight'] = copy.deepcopy(local)
        # Reproduce exact historical object relocation; no new perception/yaw.
        for key in ('object_center_m', 'object_yaw_deg', 'table_height_m'):
            source['preflight'][key] = saved['source']['preflight'][key]
        for key in ('object_x_m', 'object_y_m', 'object_yaw_deg', 'table_height_m'):
            source['preflight']['arguments'][key] = saved['source']['preflight']['arguments'][key]
        reference = dict(base_reference, rank=row['rank'], source=source, rows=local['local_reference']['rows'])
        close = [r for r in reference['rows'] if r['phase'] == 'close'][-1]
        reference.update(closure_alpha=close['closure_alpha'], hand_close_raw=close['hand_raw'],
                         minimum_desk_clearance_mm=local['local_reference']['minimum_desk_clearance_mm'])
        delta = np.asarray(saved['source']['template_relocation']['delta_T'])
        model, canonical = template.canonical(None, source, reference, delta)
        prepared_sources[row['rank']] = (source, reference, model, canonical)
    # Saved evidence must match the archive, not just acquire a fresh hash now.
    archived_inputs = {}
    for path in paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        archived = args.archive_files.resolve()/path.resolve().relative_to(ROOT)
        if digest(archived) != digest(path):
            raise ValueError(f'Saved input differs from archived evidence: {path}')
        archived_inputs[str(archived)] = digest(archived)
    report.update(input_sha256={str(p): digest(p) for p in paths}, archived_input_sha256=archived_inputs,
                  eligible_ranks=pool['eligible_ranks'],
                  historical_pool_source_sha256=pool['source_sha256'], policy=planning.policy(),
                  fixed_layout=layout, snapshot=saved['snapshot'])
    # Ensure data references used during execution are bound as well as code.
    pins.update(report['input_sha256'])
    historical = {t['point_id']: t for t in original['trials']}
    for point in layout['points']:
        point_dir = args.out/point['id']; point_dir.mkdir()
        def attempt(rank):
            folder = point_dir/f'rank_{rank:03d}'; folder.mkdir()
            source, reference, model, canonical = prepared_sources[rank]
            start = time.monotonic()
            print(f"Pool audit: {point['id']} rank {rank}", flush=True)
            try:
                planning.execution(model, source, reference, canonical, saved['snapshot'],
                                   source['trial_config'], folder, point=point, video=False)
                value = dict(status='nominal_route_pass')
            except planning.PlanningUnavailable as error:
                value = dict(status='nominal_route_rejected', reason=error.reason, details=error.attempts)
            value['elapsed_s'] = time.monotonic()-start
            (folder/'audit_result.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
            print(f"  {value['status']} ({value['elapsed_s']:.1f}s)", flush=True)
            return value
        selected, ledger = select_first(rows, attempt)
        baseline = next(r for r in ledger if r['rank'] == 20)['status'] == 'nominal_route_pass'
        old_pass = historical[point['id']]['status'] == 'offline_prepared'
        report['points'].append(dict(point_id=point['id'], historical_status=historical[point['id']]['status'],
            rank20_replay_pass=baseline, selected_rank=selected, nominal_pool_pass=selected is not None,
            recovered_start=not old_pass and selected is not None, candidates=ledger))
        checkpoint(args.out, report)
        if baseline != old_pass:
            raise ValueError(f"Historical rank-20 reproducibility mismatch at {point['id']}; stop attribution")
    support.verify(pins)
    report.update(status='completed', recovered_starts=sum(p['recovered_start'] for p in report['points']),
                  nominal_pool_passes=sum(p['nominal_pool_pass'] for p in report['points']))


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_frozen_pool_start_audit/v1', status='running', executable=False,
        hardware_contacted=False, motion_authorized=False, robot_commands_sent=False,
        created_utc=datetime.now(timezone.utc).isoformat(), points=[], assumptions=[
            'Historical sugar scene, scan, fixed starts, nominal CB3 geometry and route policy are frozen.',
            'Reuse complete top-100 official/local screen; do not claim globally exhaustive IK search.',
            'For each start choose first locally eligible candidate with nominal route pass, in original rank order.',
            'No re-inference, reranking, threshold/model changes, controller queries or physical attempts.',
            'Offline prepared files are diagnostic only, not execution approvals.',
            'Historical fixed-template outcomes remain a separate protocol; nominal pass is not physical success.'])
    print('\n'.join(report['assumptions']), flush=True)
    try:
        audit(args, report)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        checkpoint(args.out, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
