"""Freeze GGX comparison inputs from a recorded Reflex run; no devices."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml

from tools import run_ggx_review as ui, run_ggx_real as real, ggx_template as template
from tools import ggx_fixed_starts as fixed

ROOT = Path(__file__).resolve().parents[1]


def uniform_wrist_layout(layout, settings, scan):
    """Change numerical branch hints only; keep every saved Cartesian start."""
    seed = template.vector(settings['seed_q'], 6).tolist()
    if not np.allclose(seed, template.vector(scan['q'], 6), rtol=0, atol=1e-12):
        raise ValueError('Common scan and start wrist-branch seeds differ')
    rv = template.vector(settings['tcp_rotvec'], 3)
    result = deepcopy(layout)
    previous = {}
    for point in result['points']:
        if not np.allclose(template.vector(point['tcp'], 6)[3:], rv, rtol=0, atol=1e-12):
            raise ValueError('Common wrist policy must not change saved TCP orientation')
        previous[point['id']] = point['q_nominal']
        point.update(q_nominal=list(seed), ik_status='unverified_seed_only',
                     wrist_branch_policy='common_new_wrist/v1')
    result['settings']['seed_q'] = list(seed)
    metadata = dict(revision='common_new_wrist/v1', original_q_nominal=previous,
        common_seed_q=seed, cartesian_starts_unchanged=True, historical_reflex_results_unchanged=True,
        selection='User-requested common new wrist branch, not optimized per candidate or object',
        limitation='IK seed is a branch preference, not a guarantee of every downstream joint solution; paths are rechecked')
    result['wrist_branch_policy'] = metadata
    return result, metadata


def load_reference(path):
    path = Path(path).resolve()
    old = ui.read_json(path)
    if old.get('schema') != 'rh56_reflex_ten_experiment/v1':
        raise ValueError('Expected a recorded Reflex ten-point summary')
    data_path = Path(old['template'])
    data = ui.read_json(data_path)
    layout_path = path.parent/'fixed_layout.json'
    layout = ui.read_json(layout_path)
    if (data.get('schema') != 'rh56_reflex_ten_template/v1'
            or layout.get('schema') != 'rh56_object_relative_starts/v1'
            or [p['id'] for p in layout['points']] != [f'P{i}' for i in range(1, 11)]):
        raise ValueError('Require complete saved object-relative Reflex P1–P10')
    if not np.allclose(layout['origin_grasp_center_mm'], data['target']['grasp_center_base_mm'], rtol=0, atol=1e-9):
        raise ValueError('Saved Reflex layout does not use this template grasp anchor')
    # Historical software is provenance, NOT authority to execute it. Actual
    # saved data/model/calibration must still match, even when old UI changed.
    historical = {**data['pins'], **old['pins']}
    critical = [data_path, layout_path, Path(data['crop']['point_cloud']),
                Path(data['crop']['source_preview']), Path(data['simulation_config'])]
    drift = []
    for name, expected in historical.items():
        current = ui.digest(name)
        if current != expected:
            p = Path(name)
            if p.suffix == '.py' or p.name == 'reflex_ten_starts_object_relative_lab.yaml':
                drift.append(dict(path=name, historical_sha256=expected, current_sha256=current,
                                  usage='historical provenance only; saved numeric layout is used'))
            else:
                raise ValueError(f'Reflex data/model/config changed: {name}')
    for p in critical:
        if historical.get(str(p.resolve())) != ui.digest(p):
            raise ValueError(f'Reflex run does not bind saved input: {p}')
    config_path = ROOT/'configs/reflex_real_lab.yaml'
    if historical.get(str(config_path)) != ui.digest(config_path):
        raise ValueError('Reflex motion configuration changed or was not recorded')
    config = real.load_config(config_path)
    simulation = yaml.safe_load(Path(data['simulation_config']).read_text())
    scans = []
    for name in historical:
        p = Path(name)
        if p.name in ('scan_snapshot.json', 'start_pose.json', 'reflex_scan_wrist_flip.json'):
            snap = ui.read_json(p)
            q = snap.get('entry_snapshot', snap).get('q')
            if q is not None and np.allclose(np.degrees(q), simulation['start_q_deg'], atol=1e-7, rtol=0):
                scans.append(p)
    if not scans:
        raise ValueError('Cannot recover the pinned Reflex scan snapshot matching simulation')
    historical_scan = scans[0]
    scan = ROOT/'configs/reflex_scan_wrist_flip.json'
    wrist_settings_path = ROOT/'configs/reflex_ten_starts_object_relative_lab.yaml'
    layout, wrist_policy = uniform_wrist_layout(layout, yaml.safe_load(wrist_settings_path.read_text()), ui.read_json(scan))
    wrist_policy.update(historical_scan=str(historical_scan), common_scan=str(scan),
                        historical_layout_source_sha256=dict(layout['source_sha256']))
    layout['source_sha256'].update(ui.pin([wrist_settings_path, scan]))
    hold = old.get('hold_seconds')
    if not isinstance(hold, (int, float)) or not np.isfinite(hold) or hold <= 0:
        raise ValueError('Reflex hold interval not recorded')
    if data['lift_mm'] != 80.:
        raise ValueError('This matched protocol expects the recorded 80 mm lift')
    reference = dict(perception=data['perception'], crop=data['crop'], pins={}, **deepcopy(data['scene']))
    pins = {name: ui.digest(name) for name in historical}
    pins.update(ui.pin([path, data_path, layout_path, config_path, historical_scan, scan, wrist_settings_path, __file__]))
    reference['pins'] = pins
    return dict(reflex_report=str(path), data=data, layout=layout, reference=reference,
                config=config, config_path=config_path, scan=scan, pins=pins,
                xy_search_px=old.get('xy_search_px', 120), hold_seconds=hold,
                wrist_branch_policy=wrist_policy,
                historical_code_drift=drift, object_label=data['object_label'],
                differences=[
                    'Reflex freezes one analytical grasp; GGX freezes original top-100 and selects per start.',
                    f"Reflex uses {data['target']['fingers']} fingers and prepares thumb before Pi; GGX retains its original five-finger OPEN/closure.",
                    'Reflex uses native thumb/remaining-finger timing; GGX keeps its interpolated closure timing.',
                    'Reflex permits intended finger contact during approach; GGX retains its existing target gate.',
                    'For round objects Reflex may rotate its proxy; GGX retains the measured enclosing proxy.',
                    'Saved Cartesian starts match; ALL GGX objects use the common new wrist seed and scan preset, including sugar.',
                    'Historical sugar Reflex used the old branch; this declared GGX accommodation does not rewrite that result.',
                    'Branch seeds are preferences; actual controller-resolved joints and all paths remain checked.'])


def matched_point(layout, point_id, reference_center, measured_xy):
    point = next(p for p in layout['points'] if p['id'] == point_id)
    delta = np.asarray(measured_xy)/1000 - np.asarray(reference_center[:2])
    return fixed.translate_point(point, delta), delta.tolist()


def freeze_pool(args, contract, out):
    """Generate ONCE from the identical cloud, then finish all 100 local checks."""
    from tools import run_ggx_feasibility as feasibility, ggx_cb3_candidates as cb3
    from tools.preview_tabletop_target import load_perception_config
    from tools.audit_ggx_frozen_pool_starts import validate_pool
    cloud = Path(contract['data']['crop']['point_cloud'])
    if args.pool_report:
        pool_path = args.pool_report
    else:
        batch = args.candidate_run
        if batch is None:
            batch = out/'generation'; batch.mkdir()
            ui.run_stage(batch, 'inference', 'run_ggx_cube_inference.py',
                ['--point-cloud', cloud, '--input-frame', 'ur_base', '--num-grasps', 1000,
                 '--topk', 100, '--save-all-candidates', '--seed', args.seed, '--out', batch/'inference'],
                python=args.ggx_python)
        inf_path = batch/'inference/summary.json'
        inf = ui.read_json(inf_path)
        validate_cloud(inf, cloud)
        # Fresh scene config from the shared crop, not from a different GGX object.
        # Keep GGX's validated enclosing proxy; record any round-Reflex difference.
        pc, _ = load_perception_config(Path(contract['data']['perception']['config']))
        config = ui.crop_sim_config(contract['data']['crop'], contract['data']['crop']['dimensions_prior_mm'],
                                   pc, ui.read_json(contract['scan']), Path(inf['candidate_file']))
        config['lift_mm'] = contract['data']['lift_mm']
        # The feasibility API accepts a batch directory. Bind the original
        # inference metadata without rewriting/re-signing its candidate files.
        stage = out/'batch'; (stage/'inference').mkdir(parents=True)
        template.write_json(stage/'inference/summary.json', inf)
        (stage/'simulation.yaml').write_text(yaml.safe_dump(config))
        code = feasibility.main(['--review-run', str(stage), '--real-config', str(contract['config_path']),
            '--max-candidates', '100', '--candidate-scope', 'grasp', '--ggx-python', str(args.ggx_python),
            '--out', str(out/'pool'), *([] if args.official_filter is None else ['--official-filter', str(args.official_filter)])])
        if code:
            raise RuntimeError('GGX pool screening failed; see pool/summary.json (not a grasp failure)')
        pool_path = out/'pool/summary.json'
    pool = ui.read_json(pool_path)
    if pool.get('status') != 'audit_completed':
        raise ValueError('Expected completed full-pool feasibility audit')
    ui.verify(pool['source_sha256'])
    official_path = Path(pool['official_scene_filter'])
    official = ui.read_json(official_path)
    inf = ui.read_json(Path(official['candidate_file']).parent/'summary.json')
    validate_cloud(inf, cloud)
    cb3.load_filter(official_path, Path(official['candidate_file']))
    # Existing audit omits eligible_ranks for an empty pool.
    pool.setdefault('eligible_ranks', [])
    rows = validate_pool(pool, official)
    pins = {**pool['source_sha256'], **ui.pin([pool_path, official_path])}
    for row in rows:
        if row.get('report'):
            pins.update(ui.pin([row['report']]))
    return pool_path, pool, pins


def validate_cloud(inference, cloud):
    if (inference.get('status') != 'candidates_generated_not_motion_validated'
            or inference.get('input_frame') != 'ur_base'
            or Path(inference['source_point_cloud']).resolve() != Path(cloud).resolve()
            or inference['source_point_cloud_sha256'] != ui.digest(cloud)
            or inference['candidate_sha256'] != ui.digest(inference['candidate_file'])):
        raise ValueError('GGX must use the identical pinned Reflex cloud and original candidate file')


def candidate_templates(pool, contract, pins, out):
    """Use the existing validated geometry-to-template handoff, no bypass."""
    from tools import ggx_real_support as support
    result = {}
    for row in pool['feasibility_stages']:
        if row['local_state'] != 'geometry_eligible':
            continue
        folder = out/f"rank_{row['rank']:03d}"; folder.mkdir()
        local_path = Path(row['report']).resolve()
        local = ui.read_json(local_path)
        if real.load_config(Path(local['real_config'])) != contract['config'] or local['arguments']['lift_mm'] != 80.:
            raise ValueError('Pool motion/lift settings differ from Reflex; re-screen the SAME candidates')
        bound = {**pins, **ui.pin([local_path])}
        proposal = folder/'candidate.json'
        template.write_json(proposal, dict(schema='rh56_ggx_execution_candidate/v1', rank=row['rank'],
            human_approved=False, executable=False, preflight=str(local_path), source_sha256=bound))
        support.read_candidate(proposal)
        template.create(proposal, contract['scan'], contract['config'], folder)
        result[row['rank']] = template.load(folder/'template.json')
    return result
