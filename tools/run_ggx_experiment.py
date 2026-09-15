#!/usr/bin/env python3
"""One object: GGX once -> fixed setup P1..P10, RGB XY relocation, collision skips.

Without --execute: no UR5/hand access. --capture-live enables camera capture
only. --execute includes separately confirmed reset and per-trial video Enter.
Change object by starting a NEW run (new GGX/template), not by reusing a template.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ggx_template as template, ggx_fixed_starts as fixed, ggx_xy_tracking as xy
from tools import run_ggx_real as real, run_ggx_review as review
from tools import run_ggx_ten_starts as ten


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--template', type=Path, help='Resume same object only, still remeasure XY')
    source.add_argument('--approval', type=Path, help='Existing visual approval for this object')
    source.add_argument('--candidate-run', type=Path,
                        help='Reuse saved simulation.yaml + inference; select among strict passes without new inference')
    parser.add_argument('--manual-select', action='store_true',
                        help='Show strict passing candidates in original rank order: Enter selects, n shows next; default selects first automatically')
    parser.add_argument('--official-filter', type=Path, help='Matching pinned official filter for --candidate-run')
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument('--capture-live', action='store_true')
    capture.add_argument('--capture', type=Path, help='Saved first capture for offline GGX generation')
    parser.add_argument('--trial-captures', type=Path, nargs='+', help='Offline replay: one saved capture per selected point')
    parser.add_argument('--dimensions-mm', type=float, nargs=3, metavar=('SHORT','LONG','HEIGHT'),
                        help='First generation only: physical size prior for the enclosing simulation proxy')
    parser.add_argument('--object-label', default='object')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ggx_real_lab.yaml')
    parser.add_argument('--perception-config', type=Path, default=ROOT/'configs/tabletop_vision_lab.yaml')
    parser.add_argument('--layout-config', type=Path, default=ROOT/'configs/ggx_ten_starts_lab.yaml')
    parser.add_argument('--background', type=Path, default=ROOT/'artifacts/tabletop_background/empty_20260909_01')
    parser.add_argument('--start-snapshot', type=Path, default=ROOT/'artifacts/vision_pick_start/start_pose.json')
    parser.add_argument('--ggx-python', type=Path, default=ROOT/'artifacts/ggx_lab/GraspGenX/.venv/bin/python')
    parser.add_argument('--point', choices=['all']+[f'P{i}' for i in range(1,11)], default='all')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-high-force', action='store_true')
    parser.add_argument('--phase', choices=['approach-only','grasp-lift'], default='grasp-lift')
    parser.add_argument('--video', action='store_true', help='Offline videos; live always shows the checked execution video')
    parser.add_argument('--open', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not (args.template or args.approval or args.candidate_run):
        if not (args.capture_live or args.capture) or args.dimensions_mm is None:
            parser.error('New object needs --capture-live (or --capture) and --dimensions-mm')
        from tools.prepare_ggx_tabletop_cloud import known_dimensions
        known_dimensions(args.dimensions_mm)
    elif args.dimensions_mm is not None or args.capture is not None:
        parser.error('A template/approval freezes this object; do not replace dimensions or first capture')
    if args.official_filter and not args.candidate_run:
        parser.error('--official-filter requires --candidate-run')
    if args.manual_select and (args.template or args.approval):
        parser.error('--manual-select needs new generation or --candidate-run, not a frozen selection')
    if args.execute and not args.capture_live:
        parser.error('--execute requires --capture-live for this run XY measurement; no old-XY fallback')
    if args.trial_captures and args.capture_live:
        parser.error('Saved trial captures cannot substitute for live measurements')
    count = 10 if args.point == 'all' else 1
    if args.trial_captures and len(args.trial_captures) != count:
        parser.error(f'Expected {count} saved trial capture(s)')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # Do not resolve a venv interpreter symlink to the system Python.
            setattr(args, key, value.absolute() if key == 'ggx_python' else value.resolve())
    if args.trial_captures:
        args.trial_captures = [p.resolve() for p in args.trial_captures]
    return args


def skip_collision(exc, trial):
    """Only typed, pre-motion geometric collisions are resumable."""
    from tools.ggx_planning import PlanningUnavailable
    return (isinstance(exc, (real.motion.PathCollision, PlanningUnavailable))
        and not trial.get('motion_commands_attempted') and not trial.get('hand_commands')
        and not trial.get('motion_confirmation_received') and not trial.get('cleanup_errors'))


def reset_robot(args, out):
    from tools.ggx_reset import run_reset
    return run_reset(SimpleNamespace(reset_first=True, capture_live=True,
        real_config=args.config, config=args.perception_config, start_snapshot=args.start_snapshot,
        allow_high_force=args.allow_high_force), out)


def generate_once(args, out, snapshot, *, review_main=review.main):
    if args.candidate_run:
        command = ['--candidate-run', str(args.candidate_run)]
        if args.official_filter:
            command += ['--official-filter', str(args.official_filter)]
    else:
        command = ['--capture-live'] if args.capture_live else ['--capture', str(args.capture)]
        command += ['--dimensions-mm', *map(str,args.dimensions_mm), '--crop-mode', 'foreground',
                    '--freeze-reference-yaw', '--background', str(args.background)]
    if not args.manual_select:
        command.append('--auto-select')
    command += ['--max-candidates', '100', '--no-fallback-review', '--feasibility-mode','official-cb3',
        '--real-config',str(args.config),'--config', str(args.perception_config),
        '--start-snapshot', str(snapshot),
        '--ggx-python', str(args.ggx_python), '--out', str(out)]
    if not args.open:
        command.append('--no-open')
    # Deliberately no --execute: this chooses the template, not a trial motion.
    rc = review_main(command)
    selection = review.read_json(out/'summary.json')
    expected = 'review_approved' if args.manual_select else 'auto_candidate_selected'
    if rc or selection['status'] != expected:
        raise RuntimeError(f'No strict GGX template selected; see {out / "summary.json"}')
    return out/'summary.json' if args.manual_select else Path(selection['selected_candidate'])


def capture_xy(args, source, out):
    from tools.preview_tabletop_target import load_perception_config
    c, _ = load_perception_config(args.perception_config)
    if not real.confirm('物体已放回附近、同一高度/朝向，手与夹具不遮挡；Enter 拍摄并仅更新 XY。'):
        raise KeyboardInterrupt
    capture = out/'capture'
    review.run_stage(out, 'capture', 'capture_tabletop_background.py',
        ['--out',capture,'--camera-serial',c['camera_serial'],'--scene','object',
         '--rgb-only-quality','--roi',*c['roi']])
    return capture


def run(args, out, report, checkpoint, *, reset_fn=reset_robot, generate_fn=generate_once,
        locate_fn=xy.locate, real_run=real.run, capture_fn=capture_xy):
    import numpy as np
    config = real.load_config(args.config)
    if args.execute:
        real.require_hand_setting_authorization(config, args.allow_high_force)
    # Entire setup is checked BEFORE reset/camera/inference. No object enters this API.
    layout = fixed.build(args.layout_config, config, args.perception_config)
    layout_path = out/'fixed_layout.json'
    template.write_json(layout_path, layout)
    pins = {**ten.code_pins(args.config), **layout['source_sha256'], **review.pin([
        __file__, fixed.__file__, xy.__file__, ROOT/'tools/prepare_ggx_tabletop_cloud.py',
        ROOT/'tools/capture_tabletop_background.py', ROOT/'tools/calibrate_ur5_external_camera_real.py',
        ROOT/'tools/preview_tabletop_target.py', ROOT/'tools/analyze_ur5_camera_colored_target.py', layout_path])}
    report.update(fixed_layout=str(layout_path), code_config_sha256=pins, trials=[])
    checkpoint()
    scan = args.start_snapshot

    def load_source(path):
        loaded = template.load(path)
        if Path(loaded[1]['perception']['config']).resolve() != args.perception_config:
            raise ValueError('Template and current perception config differ')
        return loaded

    # Reject invalid/stale saved inputs BEFORE asking to reset real hardware.
    loaded, template_path = None, args.template
    if template_path:
        loaded = load_source(template_path)
    elif args.approval:
        template.create(args.approval, scan, config, out)
        template_path = out/'template.json'
        loaded = load_source(template_path)
    report['ggx_inference_runs'] = 0
    if args.candidate_run and loaded is None:
        # Saved-batch errors must surface before any reset command.
        proposal = generate_fn(args, out/'initial_ggx', scan)
        report['initial_selection_report'] = str(out/'initial_ggx/summary.json')
        template.create(proposal, scan, config, out)
        template_path = out/'template.json'
        loaded = load_source(template_path)
    if args.execute:
        review.verify(pins)
        code, scan = reset_fn(args, out/'initial_reset')
        report['initial_reset'] = str(out/'initial_reset/summary.json')
        if code:
            report['status'] = 'reset_stopped'
            return code
    if loaded is None:
        report['initial_selection_report']=str(out/'initial_ggx/summary.json')
        report['initial_generation_attempted']=True
        checkpoint()
        try:
            approval = generate_fn(args, out/'initial_ggx', scan)
        finally:
            manifest=out/'initial_ggx/summary.json'
            inference=out/'initial_ggx/inference/summary.json'
            if manifest.is_file():
                selection=review.read_json(manifest)
                report['initial_selection_status']=selection['status']
                report['initial_feasibility_counts']=selection.get('feasibility_counts')
            if inference.is_file():
                report['ggx_inference_runs']=int(review.read_json(inference).get('status')=='candidates_generated_not_motion_validated')
            checkpoint()
        report['ggx_inference_runs'] = 1
        template.create(approval, scan, config, out)
        template_path = out/'template.json'
        loaded = load_source(template_path)
    data, original, reference = loaded
    points = layout['points'] if args.point == 'all' else [p for p in layout['points'] if p['id'] == args.point]
    report.update(template=str(template_path), rank=original['rank'], layout_independent_of_object=True)
    goal = (real.support.pose_matrix([r for r in reference['rows'] if r['phase']=='approach'][-1]['tcp'])
            @ np.r_[data['open_grasp_center_in_tcp_m'],1])[:3]*1000
    ten.plot_layout(out, layout['points'], original, goal)
    need_reset, measured, source, model, common = False, None, None, None, None
    print('固定十起点：位置/朝向/张手形状与物体和 GGX 无关；初始 yaw/Z 固定，每次放回只测 XY。', flush=True)
    for index, point in enumerate(points):
        directory = out/point['id']
        directory.mkdir()
        trial = dict(schema='rh56_ggx_fixed_start_trial/v1', point_id=point['id'],
            object_label=args.object_label, rank=original['rank'], status='starting',
            motion_confirmation_received=False, motion_commands_attempted=False,
            execute_requested=args.execute, code_config_sha256=pins)
        report['trials'].append(trial)
        def save():
            template.write_json(directory/'summary.json', trial)
            checkpoint()
        save()
        try:
            review.verify(pins)
            review.verify(original['pins'])
            if need_reset:
                code, scan = reset_fn(args, directory/'reset')
                if code:
                    trial['status'] = 'reset_stopped'
                    return code
                measured = None
                need_reset = False
            if measured is None or args.trial_captures:
                if args.capture_live:
                    capture = capture_fn(args, original, directory)
                else:
                    capture = args.trial_captures[index] if args.trial_captures else Path(original['perception']['capture'])
                measured = locate_fn(original, capture, directory/'xy')
                if measured.get('status') != 'localized':
                    raise ValueError('XY locator did not produce a valid measurement')
                source, delta = template.relocate(data, original, reference,
                    xy_mm=measured['object_xy_mm'], yaw_deg=None, lock_z=True)
                source.update(desk_edge_y_m=config['desk_edge_y_m'], desk_envelope=config.get('desk_envelope'), trial_config=config)
                source['pins'].update({**pins, **measured['input_sha256'], **review.pin([directory/'xy/summary.json'])})
                model, common = template.canonical(data, source, reference, delta)
            trial.update(xy=measured, xy_reused_after_no_motion=not (directory/'xy').exists(),
                         point=point, source_geometry=source['template_relocation'])
            save()
            live_args = SimpleNamespace(execute=args.execute, preflight_only=False,
                checked_review=bool(args.execute), static_scene=bool(args.execute),
                start_snapshot=scan, phase=args.phase, video=args.video or args.execute, open=args.open)
            def prepare(s, snapshot, folder, video=False):
                return template.prepare_trial(data,s,reference,model,common,point,snapshot,folder,video=video)
            code = real_run(live_args, config, source, directory, trial, save, prepare_fn=prepare)
            if code or trial.get('cleanup_errors'):
                report['status'] = 'stopped'
                return code or 1
            if args.execute:
                need_reset = True
        except Exception as exc:
            if skip_collision(exc, trial):
                from tools.ggx_planning import PlanningUnavailable
                if isinstance(exc, PlanningUnavailable):
                    trial.update(status='skipped_planning_unavailable',planning_reason=exc.reason,
                                 planning_attempts=exc.attempts,error=str(exc))
                    print(f"{point['id']} 本地路径未找到可执行解（{exc.reason}），未执行，记录并检查下一点。",flush=True)
                else:
                    trial.update(status='skipped_collision', collision_phase=exc.phase,
                                 obstacle=exc.obstacle, error=str(exc))
                    print(f"{point['id']} 路径碰撞（{exc.phase}/{exc.obstacle}），未执行，跳过。", flush=True)
            else:
                trial.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                report['status'] = 'failed'
                raise
        finally:
            save()
    skipped = sum(t['status'] in ('skipped_collision','skipped_planning_unavailable') for t in report['trials'])
    report.update(status='all_points_skipped' if skipped == len(points) else
                  'completed_with_skips' if skipped else 'completed', skipped_points=skipped)
    return 2 if skipped == len(points) else 0


def main(argv=None):
    args = parse_args(argv)
    out = args.out or ROOT/'artifacts/ggx_experiments'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_fixed_start_experiment/v1', status='starting',
        object_label=args.object_label, execute_requested=args.execute, live_camera_requested=args.capture_live,
        selection_policy='reuse_frozen_selection' if args.template or args.approval else
                         'manual_among_strict_passes' if args.manual_select else 'first_strict_pass',
        assumptions=['New object requires new initial GGX/approval; saved template is same-object only.',
            'Original GGX top-100 order retained; strict geometry passes only, no diagnostic contact-warning fallback.',
            'Manual selection records all displayed vetoes and approval; otherwise first strict pass is selected automatically.',
            'One setup-wide fixed ten-point layout, NOT optimized to make this object paths pass.',
            'Height and initial yaw held fixed; only XY is updated, no yaw correction.',
            'Consecutive pre-motion collision skips reuse the last XY: nothing has moved.',
            'After real motion: confirmed supported-object reset and a fresh capture/XY measurement.',
            'Typed local pre-motion planning exhaustion/collisions skip; controller, communication, localization, cleanup and motion faults stop.',
            'Offline without trial captures reuses the reference image; this is NOT a live placement measurement.',
            'No hand feedback; completed commands do not establish grasp success. No automatic release at end.'])
    def checkpoint():
        template.write_json(out/'summary.json', report)
        keys = ['object_label','point_id','rank','status','planning_reason','collision_phase','obstacle',
                'motion_confirmation_received','motion_commands_attempted','error']
        with (out/'summary.csv').open('w', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            writer.writeheader(); writer.writerows(report.get('trials', []))
    checkpoint()
    try:
        code = run(args,out,report,checkpoint)
        if report['status'] == 'starting':
            report['status'] = 'stopped'
        return code
    except (KeyboardInterrupt, EOFError):
        report['status'] = 'cancelled'
        return 130
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr)
        return 1
    finally:
        checkpoint()
        print(f"GGX 十点实验：{report['status']}；{out/'summary.json'}", flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
