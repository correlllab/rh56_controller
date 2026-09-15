#!/usr/bin/env python3
"""Review saved GGX top-100 endpoints using only the official scene filter.

This is candidate selection, not a robot simulation or motion authorization.
The preview uses the same aggregate mesh and sampled surface as the official
filter, not the local articulated hand or an enclosing target proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = 'rh56_ggx_official_candidate_selection/v1'
PROTOCOL = 'official_ggx_top100_endpoint_only/v1'
LOCAL_CHECKS = ('hand_mapping', 'robot_ik', 'workspace', 'fixed_desk_path',
                'target_contact', 'actual_travel', 'closure', 'lift')


def not_evaluated():
    return {name: 'NOT_EVALUATED' for name in LOCAL_CHECKS}


def deferred_metadata(args, config, *, inference=None, source_pins=None):
    """Copy provenance for later local checks; do not construct a robot model."""
    from tools import run_ggx_review as ui
    metadata = dict(local_execution_checked=False)
    pins = {}
    required = ('object_x_m', 'object_y_m', 'table_height_m', 'object_size_mm', 'object_yaw_deg')
    if all(key in config for key in required):
        size = np.asarray(config['object_size_mm'], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('Invalid deferred target proxy dimensions')
        center = np.asarray([config['object_x_m'], config['object_y_m'],
                             config['table_height_m'] + size[2]/2000], dtype=float)
        if not np.isfinite(center).all() or not np.isfinite(config['object_yaw_deg']):
            raise ValueError('Invalid deferred target proxy frame')
        metadata.update(object_center_m=center.tolist(), object_size_mm=size.tolist(),
                        object_yaw_deg=config['object_yaw_deg'], table_height_m=config['table_height_m'])
    recorded_perception = None
    if inference is not None:
        crop_path = Path(inference['source_point_cloud']).parent/'summary.json'
        if crop_path.is_file():
            preview_path = Path(ui.read_json(crop_path)['source_preview']).resolve()
            recorded_perception = Path(ui.read_json(preview_path)['config']).resolve()
            for path in (crop_path.resolve(), preview_path, recorded_perception):
                if source_pins is None or str(path) not in source_pins:
                    raise ValueError('Reference crop/perception provenance is not pinned')
    for argument, key in (('real_config', 'real_config'), ('config', 'perception_config')):
        value = getattr(args, argument, None)
        if key == 'perception_config' and recorded_perception is not None:
            value = recorded_perception  # A CLI default must not replace a saved camera calibration.
        if value is not None:
            path = Path(value).resolve()
            metadata.update({key: str(path), f'{key}_sha256': ui.digest(path)})
            pins.update(ui.pin([path]))
            if key == 'real_config':
                real = yaml.safe_load(path.read_text())
                metadata['desk_envelope'] = real.get('desk_envelope')
                metadata['desk_edge_y_m'] = real.get('desk_edge_y_m')
    return metadata, pins


def load_inputs(filter_path, candidate_path):
    """Bind unchanged exporter ordering, official decisions, and preview assets."""
    from tools import run_ggx_review as ui
    from tools.run_ggx_official_filter import SCHEMA as FILTER_SCHEMA
    from rh56_controller.graspgenx_baseline import load_isaac_grasp_yaml

    filtered = ui.read_json(filter_path)
    candidate_path = Path(candidate_path).resolve()
    if (filtered.get('schema') != FILTER_SCHEMA or filtered.get('status') != 'completed'
            or filtered.get('executable') is not False
            or filtered.get('hardware_contacted') is not False
            or filtered.get('candidate_sha256') != ui.digest(candidate_path)
            or Path(filtered['candidate_file']).resolve() != candidate_path):
        raise ValueError('Missing/mismatched official scene filter')
    pins = dict(filtered['source_sha256'])
    ui.verify(pins)
    inference_path = Path(filtered['inference']).resolve()
    for path in (inference_path, candidate_path):
        if str(path) not in pins:
            raise ValueError('Official filter does not bind inference/candidate source')
    inference = ui.read_json(inference_path)
    if (inference.get('schema') != 'rh56_ggx_point_cloud_inference/v1'
            or inference.get('status') != 'candidates_generated_not_motion_validated'
            or inference.get('input_frame') != 'ur_base'
            or inference.get('topk') != 100
            or inference.get('candidate_sha256') != ui.digest(candidate_path)
            or Path(inference['candidate_file']).resolve() != candidate_path):
        raise ValueError('Expected original UR-base GGX top-100 inference')
    raw = yaml.safe_load(candidate_path.read_text())
    entries = raw.get('grasps', {})
    candidates = load_isaac_grasp_yaml(candidate_path)
    if not 1 <= len(candidates) <= 100 or inference.get('candidates') != len(candidates):
        raise ValueError('Candidate file exceeds/mismatches original top-100 export')
    if list(entries) != [c.name for c in candidates]:
        raise ValueError('Exporter order is not original descending confidence order')
    rows = filtered.get('candidates', [])
    if len(rows) != len(candidates) or [r.get('rank') for r in rows] != list(range(len(candidates))):
        raise ValueError('Official filter changed original rank order/count')
    for row, candidate in zip(rows, candidates):
        if (row.get('name') != candidate.name or row.get('confidence') != candidate.confidence
                or not np.isfinite(candidate.confidence)
                or candidate.position.shape != (3,)
                or not np.isfinite(candidate.position).all()
                or type(row.get('official_scene_clear')) is not bool):
            raise ValueError('Official filter changed candidate score/name/decision')

    def asset(name):
        matches = [Path(path) for path in pins if Path(path).name == name]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one pinned official asset: {name}')
        return matches[0]

    object_cloud = Path(inference['source_point_cloud']).resolve()
    if str(object_cloud) not in pins or ui.digest(object_cloud) != inference['source_point_cloud_sha256']:
        raise ValueError('Target preview cloud differs from inference')
    assets = dict(mesh=asset('coll_mesh.obj'), surface=asset('official_hand_surface_m.npy'),
                  scene=asset('scene_without_target_ur_base_m.npy'), object=object_cloud,
                  official_function=asset('collision_filter.py'))
    if Path(filtered.get('official_function', '')).resolve() != assets['official_function'].resolve():
        raise ValueError('Unbound official filter implementation')
    if not np.isfinite(filtered.get('threshold_mm', np.nan)) or filtered['threshold_mm'] <= 0:
        raise ValueError('Missing official distance threshold')
    descriptor = assets['mesh'].parent/'config.json'
    if not descriptor.is_file():
        raise ValueError('Official descriptor OPEN metadata missing')
    open_joints = ui.read_json(descriptor).get('open')
    if not isinstance(open_joints, dict) or not open_joints or not all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in open_joints.values()):
        raise ValueError('Invalid official descriptor OPEN dictionary')
    assets['descriptor'] = descriptor
    pins.update(ui.pin([descriptor]))
    return filtered, inference, candidates, assets, pins


def render_candidate(path, candidate, rank, assets):
    """Two views of exact saved official surface samples; not a solid rendering."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clouds = {key: np.load(assets[key], allow_pickle=False) for key in ('object', 'scene', 'surface')}
    for key, points in clouds.items():
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f'Invalid pinned preview points: {key}')
    hand = (clouds['surface'] @ candidate.rotation.T + candidate.position) * 1000
    target = clouds['object'] * 1000
    scene = clouds['scene'] * 1000
    all_points = np.vstack([hand, target])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    radius = np.max(np.ptp(all_points, axis=0)) / 2 + 15
    # View-only scene clipping never changes the saved filter inputs/decisions.
    visible_scene = scene[np.all(np.abs(scene - center) <= radius, axis=1)]
    fig = plt.figure(figsize=(12, 6))
    try:
        for index, (elev, azim) in enumerate(((25, -65), (25, 115)), start=1):
            ax = fig.add_subplot(1, 2, index, projection='3d')
            ax.scatter(*hand.T, s=2, color='#0077bb', alpha=.65)
            ax.scatter(*target[::max(1, len(target)//5000)].T, s=2, color='#dd9900', alpha=.8)
            ax.scatter(*visible_scene.T, s=.5, color='#777777', alpha=.3)
            ax.quiver(*(candidate.position*1000), *(candidate.approach_direction*45),
                      color='red', arrow_length_ratio=.15)
            ax.set(xlim=(center[0]-radius, center[0]+radius),
                   ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius),
                   xlabel='Base X (mm)', ylabel='Base Y (mm)', zlabel='Base Z (mm)')
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
        fig.suptitle(f'GGX original rank {rank}, score {candidate.confidence:.6f}\n'
                     'Official sampled hand surface | LOCAL EXECUTION NOT EVALUATED')
        fig.text(.5, .015, 'Orange: partial target observations; hidden geometry unknown. '
                 'Point preview, not solid mesh/path. Target excluded from filter. NOT MOTION APPROVAL.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .91))
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)


def official_candidates(args, out, report, config_path, source_pins, *, filter_path=None):
    """Yield (original rank, preview path, pins); never certify a local path."""
    from tools import run_ggx_review as ui, run_ggx_official_filter as official

    maximum = getattr(args, 'max_candidates', 100)
    selected_rank = getattr(args, 'candidate_rank', None)
    if type(maximum) is not int or not 1 <= maximum <= 100:
        raise ValueError('Official top-100 review budget must be 1..100; no lower-rank refill')
    if selected_rank is not None and (type(selected_rank) is not int or not 0 <= selected_rank < maximum):
        raise ValueError('Requested original rank is outside the top-100 review budget')
    ui.verify(source_pins)
    if filter_path is None:
        folder = out/'scene_input'; folder.mkdir()
        scene_input = official.prepare_scene(getattr(args, 'inference_report', out/'inference/summary.json'), folder)
        ui.run_stage(out, 'official_scene_filter', 'run_ggx_official_filter.py',
                     ['--scene-input', scene_input, '--threshold-mm', args.scene_collision_mm,
                      '--out', out/'official_scene_filter'], python=args.ggx_python)
        filter_path = out/'official_scene_filter/summary.json'
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get('candidate_frame') != 'ur-base':
        raise ValueError('Official candidate selection requires unchanged UR-base poses')
    candidate_path = Path(config['graspgenx_yaml']).resolve()
    filtered, inference, candidates, assets, pins = load_inputs(filter_path, candidate_path)
    requested_threshold = getattr(args, 'scene_collision_mm', filtered['threshold_mm'])
    if requested_threshold != filtered['threshold_mm']:
        raise ValueError('Saved official threshold differs from the declared protocol')
    metadata, metadata_pins = deferred_metadata(args, config, inference=inference, source_pins=pins)
    if selected_rank is not None and selected_rank >= len(candidates):
        raise ValueError('Requested original rank was not exported by GGX')
    pins.update(source_pins)
    pins.update(metadata_pins)
    pins.update(ui.pin([__file__, filter_path, config_path]))
    report.update(source_sha256=pins, official_scene_filter=str(Path(filter_path).resolve()),
                  feasibility_protocol=PROTOCOL, selection_scope='official_ggx_candidate',
                  feasibility_stages=[], local_checks=not_evaluated(),
                  simulation=dict(mode='official_endpoint_preview', model_loads=0),
                  executable=False, topk=100, review_budget=maximum,
                  full_batch_counts=dict(candidates=len(candidates),
                      official_scene_eligible=sum(r['official_scene_clear'] for r in filtered['candidates']),
                      official_rejected=sum(not r['official_scene_clear'] for r in filtered['candidates']),
                      local_evaluated=0, actual_travel_checked=0))
    report.setdefault('reviews', [])

    def checkpoint():
        rows = report['feasibility_stages']
        report['feasibility_counts'] = dict(candidates=len(rows),
            official_rejected=sum(not r['official_scene_clear'] for r in rows),
            official_scene_eligible=sum(r['official_scene_clear'] for r in rows),
            local_evaluated=0, actual_travel_checked=0)
        ui.save_report(out, report)
        with (out/'feasibility.csv').open('w', newline='') as stream:
            fields = ['rank', 'name', 'confidence', 'official_scene_clear', 'selection_state',
                      'local_state', 'report']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)

    checkpoint()
    for rank, candidate in enumerate(candidates[:maximum]):
        if selected_rank is not None and rank != selected_rank:
            continue
        ui.verify(pins)
        clear = filtered['candidates'][rank]['official_scene_clear']
        row = dict(rank=rank, name=candidate.name, confidence=candidate.confidence,
                   official_scene_clear=clear, local_state='NOT_EVALUATED',
                   selection_state='official_scene_eligible' if clear else 'official_scene_reject')
        report['feasibility_stages'].append(row)
        if not clear:
            report['reviews'].append(dict(rank=rank, decision='official_scene_reject',
                reason='rejected_by_official_scene_filter', report=str(filter_path)))
            print(f'rank {rank}: GGX 官方场景筛选拒绝；本地执行检查未进行。', flush=True)
            checkpoint(); continue
        folder = out/f'official_rank_{rank:03d}'; folder.mkdir()
        summary_path = folder/'summary.json'
        row['report'] = str(summary_path)
        summary = dict(schema=SCHEMA, status='official_candidate_reviewable',
            simulation_only=True, hardware_contacted=False, executable=False, executable_target=False,
            passed=False, automatic_pass_ranks=[], official_review_ranks=[rank],
            selection_scope='official_ggx_candidate', feasibility_protocol=PROTOCOL,
            actual_travel_checked=False, local_checks=not_evaluated(), feasibility=dict(row),
            rank=rank, name=candidate.name, confidence=candidate.confidence,
            topk=100, full_batch_counts=report['full_batch_counts'],
            candidate_frame='ur-base', candidate_yaml=str(candidate_path),
            candidate_yaml_sha256=ui.digest(candidate_path),
            candidate_pose=dict(position_base_m=candidate.position.tolist(),
                                rotation_base=candidate.rotation.tolist()),
            config=str(Path(config_path).resolve()), config_sha256=ui.digest(config_path),
            arguments=config, official_filter_report=str(Path(filter_path).resolve()),
            official_scene_threshold_mm=filtered['threshold_mm'],
            official_hand=dict(mesh=str(assets['mesh']), mesh_sha256=ui.digest(assets['mesh']),
                descriptor=str(assets['descriptor']), descriptor_sha256=ui.digest(assets['descriptor']),
                open_joint_positions_rad=ui.read_json(assets['descriptor'])['open'],
                metadata_provenance='Current pinned official descriptor; filter geometry is its pinned aggregate mesh.'),
            inference=str(Path(filtered['inference']).resolve()),
            preview_assets={key: str(value) for key, value in assets.items()},
            **metadata, source_sha256=pins, assumptions=[
                'Original GGX confidence/order/top-100 retained; no lower-rank refill or local reranking.',
                'Only official endpoint scene-distance filter decides candidate eligibility.',
                'Preview uses the exact saved surface samples from the pinned official aggregate collision mesh.',
                'Point preview only, not solid mesh/occlusion rendering or a local articulated-hand model.',
                'Target excluded by official filter; previewed partial target points are not collision validation.',
                'Robot IK/workspace/fixed desk/path/closure/lift and physical hand mapping NOT_EVALUATED.',
                'Candidate acceptance is NOT motion approval; separate local execution checks remain required.'])
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        with (folder/'candidates.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        view = summary_path
        if getattr(args, 'render_candidates', True):
            view = folder/'review_official.png'
            render_candidate(view, candidate, rank, assets)
        ui.verify(pins)
        review_pins = {**pins, **ui.pin([summary_path, folder/'candidates.csv', view])}
        print(f'rank {rank}: 官方候选可查看；本地手型/IK/桌面/路径未检查，不是运动批准。', flush=True)
        checkpoint()
        yield rank, view, review_pins
    checkpoint()


def reviewed_candidates(directory, *, require_video=True):
    """Reopen pinned endpoint selections without upgrading their validation level."""
    from tools import run_ggx_review as ui
    directory = Path(directory)
    path = directory/'summary.json'
    summary = ui.read_json(path)
    rank = summary.get('rank')
    if (summary.get('schema') != SCHEMA or summary.get('selection_scope') != 'official_ggx_candidate'
            or summary.get('status') != 'official_candidate_reviewable'
            or summary.get('executable') is not False or summary.get('passed') is not False
            or summary.get('hardware_contacted') is not False
            or summary.get('local_execution_checked') is not False
            or summary.get('actual_travel_checked') is not False
            or summary.get('automatic_pass_ranks') != []
            or type(rank) is not int or not 0 <= rank < 100
            or summary.get('official_review_ranks') != [rank]
            or summary.get('local_checks') != not_evaluated()):
        raise ValueError('Expected non-executable official-only candidate selection')
    pins = summary['source_sha256']
    ui.verify(pins)
    candidate = Path(summary['candidate_yaml']).resolve()
    filter_path = Path(summary['official_filter_report']).resolve()
    for bound in (candidate, filter_path, Path(summary['config']).resolve()):
        if str(bound) not in pins:
            raise ValueError('Saved official selection has unbound provenance')
    if ui.digest(candidate) != summary['candidate_yaml_sha256']:
        raise ValueError('Candidate changed after official selection')
    filtered, _, candidates, _, dependencies = load_inputs(filter_path, candidate)
    if rank >= len(candidates) or not filtered['candidates'][rank]['official_scene_clear']:
        raise ValueError('Selected candidate failed official scene filter')
    selected = candidates[rank]
    if (summary.get('name') != selected.name or summary.get('confidence') != selected.confidence
            or summary.get('candidate_pose') != dict(position_base_m=selected.position.tolist(),
                                                    rotation_base=selected.rotation.tolist())):
        raise ValueError('Saved selected rank/pose/score differs from original GGX')
    view = directory/'review_official.png'
    if not view.is_file():
        if require_video:
            raise ValueError('Missing official endpoint preview')
        view = path
    yield rank, view, {**pins, **dependencies, **ui.pin([path, directory/'candidates.csv', view])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-config', type=Path, required=True)
    parser.add_argument('--official-filter', type=Path, required=True)
    parser.add_argument('--real-config', type=Path, help='Optional pinned execution config, not evaluated')
    parser.add_argument('--config', type=Path, help='Optional pinned perception config, not evaluated')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--candidate-rank', type=int)
    parser.add_argument('--render-candidates', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    from tools import run_ggx_review as ui
    report = dict(schema=SCHEMA, status='running', hardware_contacted=False,
                  executable=False, robot_commands_sent=False, reviews=[])
    code = 0
    try:
        list(official_candidates(args, args.out, report, args.simulation_config,
                                ui.pin([args.simulation_config]), filter_path=args.official_filter))
        report['status'] = 'official_candidates_reviewable_not_executable'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}'); code = 1
    ui.save_report(args.out, report)
    print(f'Official candidate preview: {report["status"]}; {args.out/"summary.json"}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
