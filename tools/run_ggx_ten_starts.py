#!/usr/bin/env python3
"""Reuse one GGX grasp from the historical ten paper start centers.

Default: OFFLINE ten-point report, no camera/robot/hand access. --execute needs
--static-scene and fresh per-trial checked-video Enter. Full motion includes
positioning to the selected start; previous trials never authorize later ones.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ggx_template as template
from tools import run_ggx_real as real
from tools.run_ggx_review import pin, read_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--approval', type=Path, help='Create a frozen template from a saved visual approval')
    source.add_argument('--template', type=Path, help='Reuse template.json without inference or proxy refitting')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ggx_real_lab.yaml')
    parser.add_argument('--layout-config', type=Path, default=ROOT/'configs/ggx_ten_starts_lab.yaml')
    parser.add_argument('--start-snapshot', type=Path, default=ROOT / 'artifacts/vision_pick_start/start_pose.json',
                        help='Offline scan-start/IK seed only; execution always reads actual robot start')
    parser.add_argument('--point', choices=['all'] + [f'P{i}' for i in range(1, 11)], default='all')
    parser.add_argument('--origin-mm', type=float, nargs=3, help='UR-base grasp-center anchor XYZ; translates ALL ten starts only')
    parser.add_argument('--object-xy-mm', type=float, nargs=2, help='New same-object center XY in UR-base mm; Z follows calibrated desk')
    parser.add_argument('--object-yaw-deg', type=float, help='New same-object LONG-axis yaw; not hand yaw')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--static-scene', action='store_true', help='Operator confirms the current specified object pose; no RGB-D rechecks')
    parser.add_argument('--reset-first', action='store_true', help='Explicitly open/retreat/return, with separate support/clear-path Enter each trial')
    parser.add_argument('--phase', choices=['approach-only', 'grasp-lift'], default='approach-only')
    parser.add_argument('--allow-high-force', action='store_true')
    parser.add_argument('--video', action='store_true', help='Offline videos; execution always reviews its controller video')
    parser.add_argument('--open', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.execute and not args.static_scene:
        parser.error('--execute requires --static-scene: confirm this trial object pose at final Enter')
    if args.execute and args.point == 'all' and not args.reset_first:
        parser.error('Executing all ten requires --reset-first between trials (supported-object Enter each time)')
    if args.reset_first and not args.execute:
        parser.error('--reset-first requires --execute; offline commands never contact hardware')
    if args.static_scene and not args.execute:
        parser.error('--static-scene is an execution-only operator confirmation policy')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def plot_layout(out, points, source, grasp_center):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xyz = np.array([p['center_base_mm'] for p in points])
    ax.scatter(*xyz.T, label='Open-hand grasp-center starts')
    for point, pos in zip(points, xyz):
        ax.text(*pos, point['id'])
    target = np.array(source['preflight']['object_center_m'])*1000
    ax.scatter(*target, marker='s', label='Object geometric center')
    ax.scatter(*grasp_center, marker='*', s=100, label='Common open-hand grasp center at target')
    ax.set(xlabel='UR-base X (mm)', ylabel='UR-base Y (mm)', zlabel='UR-base Z (mm)')
    ax.set_box_aspect(np.maximum(np.ptp(np.vstack([xyz, target]), axis=0), 1))
    fig.legend(loc='upper center', fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, .89])
    fig.savefig(out / 'ten_starts.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def code_pins(config_path):
    assets = list((real.support.URDF.parent / 'meshes/ur5/collision').glob('*.stl'))
    assets += [p for p in (ROOT / 'h1_mujoco/inspire').rglob('*')
               if p.is_file() and p.suffix.lower() in ('.xml', '.stl', '.obj')]
    return pin([__file__, template.__file__, real.__file__, real.support.__file__,
        real.motion.__file__, real.sim.__file__, config_path, real.support.URDF,
        ROOT / 'tools/run_strategy_pregrasp_rate.py', ROOT / 'tools/demo_strategy_pregrasp_collision.py',
        ROOT / 'tools/ggx_reset.py', ROOT / 'tools/run_ggx_review.py',
        ROOT / 'tools/ggx_planning.py', ROOT / 'tools/ggx_cb3_candidates.py',
        ROOT / 'rh56_controller/grasp_geometry.py', ROOT / 'rh56_controller/graspgenx_baseline.py',
        ROOT / 'rh56_controller/ur5_bridge.py', ROOT / 'rh56_controller/rh56_hand.py',
        ROOT / 'rh56_controller/grasp_executor.py', ROOT / 'tools/run_vision_pick_cycle.py',
        ROOT / 'tools/run_vision_plan_trial.py', *assets])


def run(args, out, report, checkpoint):
    config = real.load_config(args.config)
    if args.execute:
        real.require_hand_setting_authorization(config, args.allow_high_force)
    if args.approval:
        template.create(args.approval, args.start_snapshot, config, out)
        template_path = out / 'template.json'
    else:
        template_path = args.template
    data, source, reference = template.load(template_path)
    source, delta = template.relocate(data, source, reference,
        xy_mm=args.object_xy_mm, yaw_deg=args.object_yaw_deg)
    source.update(desk_edge_y_m=config['desk_edge_y_m'], desk_envelope=config.get('desk_envelope'),
                  trial_config=config)
    pins = code_pins(args.config)
    model, common_rows = template.canonical(data, source, reference, delta)
    if args.origin_mm is None:
        from tools import ggx_fixed_starts as fixed
        setup = fixed.build(args.layout_config, config, Path(source['perception']['config']))
        origin, points = setup['origin_grasp_center_mm'], setup['points']
        pins.update({**setup['source_sha256'], **pin([fixed.__file__])})
    else:
        # Explicit legacy diagnostic override only; the experiment runner never
        # exposes this candidate-dependent start orientation/placement option.
        origin = args.origin_mm
        points = template.layout(data, source, common_rows, origin)
    import numpy as np
    final_tcp = [r for r in common_rows if r['phase'] == 'approach'][-1]['tcp']
    grasp_center = (real.support.pose_matrix(final_tcp) @ np.r_[data['open_grasp_center_in_tcp_m'], 1])[:3]*1000
    layout_path = out / 'layout.json'
    template.write_json(layout_path, dict(schema='rh56_ggx_ten_starts/v1', executable=False,
        frame='UR-base', units='mm', origin_grasp_center_mm=origin, points=points,
        historical_source='tools/run_strategy_pregrasp_rate.py:paper_approach_points_mm; approach-axis y-',
        transform_policy='Translation only: no rotation, scaling, clipping, dropping or reordering.',
        start_orientation_policy=('Setup-wide fixed scan orientation; independent of object/GGX'
                                  if args.origin_mm is None else 'Legacy candidate-dependent diagnostic override'),
        target_grasp_center_base_mm=grasp_center.tolist(),
        target_object_T=source['template_relocation']['object_T']))
    source['pins'].update(pin([layout_path]))
    report.update(template=str(template_path), rank=source['rank'], layout=str(layout_path),
        code_config_sha256=pins, target=source['template_relocation'],
        origin_grasp_center_mm=origin, trials=[])
    checkpoint()
    plot_layout(out, points, source, grasp_center)
    selected = points if args.point == 'all' else [p for p in points if p['id'] == args.point]
    print('十点：沿用历史 y- 布局，仅平移。固定模板抓姿/手形；不重新运行 GGX。', flush=True)
    print(f'抓取中心基准（UR-base mm）：{origin}', flush=True)
    print('起点 → pregrasp 为原 GGX moveJ，再 moveL 接近；不是旧论文的直线可达率实验。', flush=True)
    for point in selected:
        trial_out = out / point['id']
        trial_out.mkdir()
        trial = dict(schema='rh56_ggx_ten_start_trial/v1', point_id=point['id'],
            status='starting', execute_requested=args.execute, motion_confirmation_received=False,
            motion_commands_attempted=False, code_config_sha256=pins)
        report['trials'].append(trial)
        def save_trial():
            template.write_json(trial_out / 'summary.json', trial)
            checkpoint()
        save_trial()
        try:
            if args.reset_first:
                from types import SimpleNamespace
                from tools.ggx_reset import run_reset
                reset_args = SimpleNamespace(reset_first=True, capture_live=True, real_config=args.config,
                    config=Path(source['perception']['config']), start_snapshot=args.start_snapshot,
                    allow_high_force=args.allow_high_force)
                code, _ = run_reset(reset_args, trial_out / 'reset')
                if code:
                    trial['status'] = 'reset_cancelled_or_failed'
                    return code
                if not real.confirm('请把同一物体放到本轮指定坐标/方向；摆好并清场后继续。'):
                    trial['status'] = 'cancelled_before_trial'
                    return 2
            from types import SimpleNamespace
            run_args = SimpleNamespace(execute=args.execute, preflight_only=False,
                checked_review=bool(args.execute), static_scene=args.static_scene,
                start_snapshot=args.start_snapshot, phase=args.phase,
                video=args.video or args.execute, open=args.open)
            def prepare(s, snapshot, directory, video=False):
                return template.prepare_trial(data, s, reference, model, common_rows,
                                              point, snapshot, directory, video=video)
            code = real.run(run_args, config, source, trial_out, trial, save_trial, prepare_fn=prepare)
            if code and args.execute:
                return code
        except Exception as exc:
            trial.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f"{point['id']}: {trial['error']}", flush=True)
            if args.execute:
                return 1  # Never auto-retry a failed or rejected physical trial.
        finally:
            save_trial()
        print(f"{point['id']}: {trial['status']}", flush=True)
    report['status'] = 'completed' if all(t['status'] != 'failed' for t in report['trials']) else 'completed_with_blocked_points'
    return 0 if report['status'] == 'completed' else 2


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.out or ROOT / 'artifacts/ggx_ten_start_runs' / stamp
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_ten_start_run/v1', status='starting', execute_requested=args.execute,
        assumptions=['Default offline; no robot/camera/hand access.',
            'Same object; no GGX re-ranking, no re-estimation of proxy dimensions.',
            'Default object pose is the saved reference, NOT a fresh measurement.',
            'New XY/yaw must be supplied after object movement; not automatic visual relocalization.',
            'All starts use one frozen grasp orientation and hand closure sequence.',
            'Ten starts preserve historical pairwise displacement vectors in UR-base axes.',
            'New scan-to-start positioning segment is checked and shown in execution video.',
            'Sampled nominal geometry is not actual controller IK or pickup verification.',
            'Execution of each trial needs its own checked-video Enter. No automatic release after final trial.'])
    def checkpoint():
        template.write_json(out / 'summary.json', report)
        keys = ['point_id', 'status', 'execute_requested', 'motion_confirmation_received', 'motion_commands_attempted', 'error']
        with (out / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report.get('trials', []))
    checkpoint()
    try:
        code = run(args, out, report, checkpoint)
        if report['status'] == 'starting':
            report['status'] = 'stopped'
        return code
    except (KeyboardInterrupt, EOFError):
        report['status'] = 'interrupted'
        return 130
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr)
        return 1
    finally:
        checkpoint()
        print(f"十点实验：{report['status']}；{out / 'summary.json'}", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
