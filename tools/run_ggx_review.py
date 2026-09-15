#!/usr/bin/env python3
"""Interactive GGX capture/generate/simulate/review launcher. Offline by default.

In offline mode Enter approves the displayed simulation only; n [reason] vetoes it, p reopens,
r invalidates the scene, q cancels. With --execute, numerical CB3/controller
checks precede the displayed execution video; its Enter authorizes motion.
Without --execute, Enter is only a visual opinion and cannot authorize motion.
Zero-pass fallback reviews are offline-only; --execute never shows a diagnostic-only Enter.
--reset-first explicitly enables confirmed open/return BEFORE a new live capture.
--feasibility-mode official-only instead previews official endpoint surface points;
its Enter selects a candidate, not a simulation/path pass. It cannot execute/reset.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANUAL_REVIEW_SCHEMA = 'rh56_ggx_manual_object_review/v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(paths):
    return {str(Path(p).resolve()): digest(p) for p in paths}


def verify(pins):
    for path, expected in pins.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f'Input/review artifact changed; regenerate: {path}')


def read_json(path):
    return json.loads(Path(path).read_text())


def save_report(out, report):
    (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                allow_nan=False) + '\n')
    keys = ['status', 'selected_rank', 'executable', 'robot_commands_sent']
    with (out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerow({key: report.get(key) for key in keys})


def show(path, auto_open):
    print(f'查看：{path}', flush=True)
    if auto_open:
        try:
            subprocess.Popen(['xdg-open', str(path)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except OSError:
            print('无法自动打开，请手动打开上面的文件。', flush=True)


def decision(video, pins, auto_open, *, manual_only=False, execution=False, official_only=False,
             local_preview=False):
    """Review exactly the pinned video; no permissive default on EOF/unknown keys."""
    verify(pins)
    show(video, auto_open)
    while True:
        prompt = ('Enter=选择此候选（本地预览，不授权运动）' if official_only and local_preview else
                  'Enter=选择此官方候选（未检查本地路径，不执行实机）' if official_only else
                  'Enter=记录人工认可（不执行实机）' if manual_only else
                  'Enter=批准并执行此 CB3 路径' if execution else 'Enter=批准此仿真')
        veto = '拒绝此候选' if execution else '下一个'
        answer = input(prompt + f'；n [原因]={veto}；p=重放；r=场景已动；q=退出：').strip()
        verify(pins)  # Detect edits made while the reviewer was paused.
        if answer == '':
            return 'approve', ''
        key, _, reason = answer.partition(' ')
        if key.lower() == 'n':
            return 'veto', reason or 'operator_veto_unspecified'
        if key.lower() in ('q', 'r'):
            return ('cancel', '') if key.lower() == 'q' else ('scene_invalidated', '')
        if key.lower() == 'p':
            show(video, auto_open)
        else:
            print('未批准；请输入 Enter、n、p、r 或 q。')


def run_stage(out, name, script, arguments, python=sys.executable, allowed=(0,)):
    command = [str(python), str(ROOT / 'tools' / script), *map(str, arguments)]
    print(f'GGX: {name}', flush=True)
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    with (out / f'{name}.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode not in allowed:
        diagnostic = out / name / 'summary.json'
        detail = ''
        if diagnostic.is_file():
            try:
                detail = read_json(diagnostic).get('error', '')
            except (ValueError, OSError, AttributeError):
                pass
        raise RuntimeError(f'{name} failed ({result.returncode})' +
                           (f': {detail}' if detail else '') + f'; see {out / (name + ".log")}')


def sim_config(patch, dimensions, perception, snapshot, candidate_file):
    import yaml
    from tools.analyze_ur5_camera_colored_target import plane_z_at_xy_mm
    q = snapshot.get('entry_snapshot', snapshot).get('q')
    if not isinstance(q, list) or len(q) != 6 or not all(math.isfinite(v) for v in q):
        raise ValueError('Saved start snapshot must contain six finite q radians')
    x, y, _ = patch['center_base_mm']
    normal = yaml.safe_load(Path(perception['desk_plane']).read_text())['plane']['normal_base']
    # Use the same measured desk-origin override as perception, not the raw fit intercept.
    z = plane_z_at_xy_mm(normal, perception['desk_z_mm'], x, y) / 1000
    return dict(candidate_frame='ur-base', graspgenx_yaml=str(candidate_file),
                object_x_m=x / 1000, object_y_m=y / 1000, table_height_m=z,
                object_size_mm=[dimensions[1], dimensions[0], dimensions[2]],
                object_yaw_deg=patch['short_axis_yaw_deg'] - 90,
                start_q_deg=[math.degrees(v) for v in q], video=True)


def crop_sim_config(crop, dimensions, perception, snapshot, candidate_file):
    """Keep real dimensions distinct from an explicitly enclosing non-flat proxy."""
    from tools.prepare_ggx_tabletop_cloud import FOREGROUND_SCHEMA, validate_foreground_report
    if crop.get('schema') == FOREGROUND_SCHEMA:
        frame, size = validate_foreground_report(crop)
        if list(dimensions) != crop['dimensions_prior_mm']:
            raise ValueError('Operator dimensions changed after foreground crop')
        return sim_config(frame, size, perception, snapshot, candidate_file)
    if crop.get('schema') != 'rh56_ggx_measured_crop/v1':
        raise ValueError('Unknown GGX crop schema')
    return sim_config(crop['top_patch'], dimensions, perception, snapshot, candidate_file)


def reviewed_candidates(directory, *, require_video=True):
    """Yield pinned automatic passes; never grants any motion authorization."""
    summary_path = directory / 'summary.json'
    summary = read_json(summary_path)
    if summary.get('schema') == 'rh56_ggx_official_candidate_selection/v1':
        from tools.ggx_official_candidates import reviewed_candidates as official_reviews
        yield from official_reviews(directory)
        return
    geometry_only = summary.get('schema') == 'rh56_ggx_cb3_grasp_geometry/v1'
    if (summary.get('schema') not in ('rh56_ggx_ur5_table_preflight/v1','rh56_ggx_cb3_scene_preflight/v1','rh56_ggx_cb3_grasp_geometry/v1')
            or summary.get('simulation_only') is not True):
        raise ValueError('Expected simulation-only GGX preflight report')
    if summary.get('object_collision_gate', {}).get('revision') != 'robot_object_before_closure/v2':
        raise ValueError('Outdated collision preflight: rerun with certified zero-distance handling')
    candidate_file = Path(summary['candidate_yaml'])
    if digest(candidate_file) != summary['candidate_yaml_sha256']:
        raise ValueError('Candidates changed since preflight')
    dependencies = [summary_path, directory / 'candidates.csv', candidate_file]
    for key in ('model', 'config'):
        if summary.get(key) is not None:
            path = Path(summary[key])
            if digest(path) != summary[f'{key}_sha256']:
                raise ValueError(f'{key} changed since preflight')
            dependencies.append(path)
    with (directory / 'candidates.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    ranks = sorted(int(row['rank']) for row in rows if row.get('geometry_eligible' if geometry_only else 'passed') == 'True')
    if ranks != sorted(summary['geometry_review_ranks'] if geometry_only else summary['automatic_pass_ranks']):
        raise ValueError('Candidate report and passing ranks disagree')
    dependency_pins = pin(dependencies)
    if geometry_only or summary['schema'] == 'rh56_ggx_cb3_scene_preflight/v1':
        from tools.ggx_cb3_candidates import load_filter
        source_pins = summary['source_sha256']
        verify(source_pins)
        filter_path = Path(summary['official_filter_report']).resolve()
        if str(filter_path) not in source_pins:
            raise ValueError('Saved CB3 review does not bind official scene filter')
        official = load_filter(filter_path, candidate_file)
        if (summary.get('feasibility', {}).get('local_state') != ('geometry_eligible' if geometry_only else 'passed_current_checks')
                or any(rank in summary.get('uncertain_ranks', [])
                       or not official['candidates'][rank]['official_scene_clear'] for rank in ranks)):
            raise ValueError('Uncertain/rejected CB3 candidate is not an automatic-pass review')
        dependency_pins.update(source_pins)
    for rank in ranks:
        verify(dependency_pins)
        video = directory / f'review_rank_{rank:03d}.mp4'
        trajectory = directory / f'trajectory_rank_{rank:03d}.csv'
        yield rank, video, {**dependency_pins, **pin([trajectory, *([video] if require_video else [])])}


def review_is_manual_only(video, pins):
    """Read the pinned review type, not a mutable flag outside the artifacts."""
    verify(pins)
    manifest = video.parent / 'summary.json'
    if str(manifest.resolve()) not in pins:
        raise ValueError('Review does not bind its simulation/diagnostic report')
    return read_json(manifest).get('schema') == MANUAL_REVIEW_SCHEMA


def manual_review_candidates(out, scene, sim_args, eligible, source_pins, report, *, execution=False):
    """Render already-computed rejected paths lazily; retain their rejection."""
    from tools import preflight_ggx_ur5_table as preflight
    report['manual_fallback'] = dict(trigger='zero_automatic_passes', review_only=True,
        eligible_ranks=[result.rank for result, samples, pins in eligible], shown_ranks=[],
        selection='confidence order; IK complete and desk clear; object collision unresolved')
    if execution:
        report['manual_fallback']['suppressed_for_execution'] = True
        report['status'] = 'no_executable_candidates'
        save_report(out, report)
        print('整批没有可执行候选；实机模式不播放仅供诊断的碰撞视频，也不请求无效的 Enter。'
              '本轮未进入抓取；已单独确认的复位不受影响。', flush=True)
        return
    save_report(out, report)
    print('整批没有自动通过项。下面仅人工查看 IK 成功、桌面检查通过的候选；'
          '物体碰撞未通过，Enter 不会触发实机执行。', flush=True)
    for result, samples, rejection_pins in eligible:
        verify(source_pins)
        verify(rejection_pins)
        if not preflight.manual_object_review_eligible(result, samples, sim_args):
            raise ValueError('Manual-review eligibility changed')
        directory = out / f'manual_rank_{result.rank:03d}'
        directory.mkdir(exist_ok=False)
        started = time.monotonic()
        video = directory / 'review_manual_only.mp4'
        trajectory = directory / 'trajectory.csv'
        preflight._write_csv(trajectory, preflight._trajectory_rows(scene.model, samples))
        print(f'GGX: manual_review_rank_{result.rank:03d} (object collision unresolved)', flush=True)
        # A display-only label; never mutate result.passed or the original reason.
        preflight.write_video(video, scene.model, samples,
            replace(result, reason='manual_object_collision_review'), sim_args, scene.object_center)
        verify(source_pins)
        verify(rejection_pins)
        manifest = directory / 'summary.json'
        manifest.write_text(json.dumps(dict(schema=MANUAL_REVIEW_SCHEMA,
            simulation_only=True, executable_target=False, automatic_pass=False,
            rank=result.rank, original_result=asdict(result),
            source_sha256={**source_pins, **rejection_pins}, render_s=time.monotonic()-started,
            assumptions=['Only shown because the complete batch produced zero automatic passes.',
                'Original IK and desk gates retained; object collision is unresolved, not declared false.',
                'No new inference, IK solve or path change for this video.',
                'Manual acceptance records a visual opinion only; NOT consumable by real execution.']),
            indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        report['manual_fallback']['shown_ranks'].append(result.rank)
        save_report(out, report)
        yield result.rank, video, {**source_pins, **rejection_pins, **pin([manifest, video, trajectory])}


def generate_candidates(args, out, report):
    import yaml
    from tools.preview_tabletop_target import load_perception_config
    perception, _ = load_perception_config(args.config)
    sources = [args.config, perception['calibration'], perception['desk_plane'], args.start_snapshot]
    if args.background:
        sources += [args.background / name for name in
                    ('camera_check_rgb.png', 'camera_check_depth_u16.png', 'camera_check.yaml')]
        if (args.background / 'depth_statistics.npz').exists():
            sources.append(args.background / 'depth_statistics.npz')
    source_pins = pin(sources)
    capture = args.capture
    if args.capture_live:
        print(('机械臂已复位，控制连接已关闭。请重新摆好物体，人员离开运动区；' if args.reset_first else
               '请先将机械臂移到拍摄位（未启用自动复位）；') +
              '保持相机和桌面固定，物体放稳，手与夹具不要遮挡物体。')
        if input('按 Enter 拍摄；其他输入取消：').strip():
            report['status'] = 'cancelled'
            return
        capture = out / 'capture'
        run_stage(out, 'capture', 'capture_tabletop_background.py',
                  ['--out', capture, '--camera-serial', perception['camera_serial'],
                   '--scene', 'object', '--roi', *perception['roi']])
    report['capture'] = str(capture)
    source_pins.update(pin(capture / name for name in
                          ('camera_check_rgb.png', 'camera_check_depth_u16.png', 'camera_check.yaml')))
    from tools.ggx_scene_check import reference_burst_paths
    source_pins.update(pin(reference_burst_paths(capture)))
    preview = out / 'perception'
    command = ['--capture', capture, '--config', args.config, '--out', preview]
    if args.background:
        command += ['--background', args.background]
    run_stage(out, 'perception', 'preview_tabletop_target.py', command)
    source_pins.update(pin([preview / 'summary.json', preview / 'overlay.png', preview / 'foreground.png']))
    verify(source_pins)
    show(preview / 'overlay.png', args.open)
    answer = input('输入目标物体的编号（# 后数字）；q 取消：').strip()
    verify(source_pins)
    if answer.lower() == 'q':
        report['status'] = 'cancelled'
        return
    component = int(answer)
    report['reviewed_component_id'] = component
    crop = out / 'crop'
    run_stage(out, 'crop', 'prepare_ggx_tabletop_cloud.py',
              ['--preview', preview / 'summary.json', '--component-id', component,
               '--dimensions-mm', *args.dimensions_mm, '--crop-mode', args.crop_mode, '--out', crop,
               *(['--freeze-reference-yaw'] if getattr(args, 'freeze_reference_yaw', False) else [])])
    crop_report = read_json(crop / 'summary.json')
    source_pins.update(pin([crop / 'summary.json', crop_report['point_cloud']]))
    if args.crop_mode == 'foreground':
        source_pins.update(pin([crop / 'input_overlay.png']))
        report['crop'] = dict(mode=args.crop_mode, report=str(crop / 'summary.json'),
                             collision_proxy=crop_report['collision_proxy'])
        print('前景模式：点云保持实测值；仿真使用包围外框，不是瓶子精确模型。', flush=True)
        for warning in crop_report.get('warnings', []):
            print(f'Crop warning: {warning}', flush=True)
        show(crop / 'input_overlay.png', args.open)
    verify(source_pins)
    inference = out / 'inference'
    run_stage(out, 'inference', 'run_ggx_cube_inference.py',
              ['--point-cloud', crop_report['point_cloud'], '--input-frame', 'ur_base',
               '--num-grasps', 1000, '--topk', 100, '--out', inference], python=args.ggx_python)
    inf = read_json(inference / 'summary.json')
    candidate_file = inference / 'graspgenx_candidates.yml'
    if (inf.get('status') != 'candidates_generated_not_motion_validated'
            or inf.get('input_frame') != 'ur_base'
            or digest(candidate_file) != inf['candidate_sha256']
            or digest(crop_report['point_cloud']) != inf['source_point_cloud_sha256']):
        raise ValueError('Inference inputs/frame/output mismatch')
    source_pins.update(pin([inference / 'summary.json', candidate_file]))
    config = crop_sim_config(crop_report, args.dimensions_mm, perception,
                             read_json(args.start_snapshot), candidate_file)
    config_path = out / 'simulation.yaml'
    config_path.write_text(yaml.safe_dump(config))
    source_pins.update(pin([config_path]))
    report['source_sha256'] = source_pins
    save_report(out, report)
    yield from simulate_candidates(args, out, report, config_path, source_pins)


def reuse_candidates(args, out, report):
    """New review of saved input/ranking, never inherits a historical approval."""
    config_path = args.candidate_run / 'simulation.yaml'
    args.inference_report = args.candidate_run / 'inference/summary.json'
    source_pins = pin([config_path, args.inference_report])
    report['reused_candidates'] = dict(directory=str(args.candidate_run), inference_rerun=False,
        old_approval_inherited=False, official_filter_rerun=not bool(args.official_filter))
    save_report(out, report)
    yield from simulate_candidates(args, out, report, config_path, source_pins)


def simulate_candidates(args, out, report, config_path, source_pins):
    """Lazy confidence-order checks in one process/model; never look ahead."""
    if getattr(args, 'feasibility_mode', 'legacy') == 'official-only':
        from tools.ggx_official_candidates import official_candidates
        yield from official_candidates(args, out, report, config_path, source_pins)
        return
    if getattr(args,'feasibility_mode','legacy') == 'official-cb3':
        from tools.ggx_cb3_candidates import simulate
        yield from simulate(args,out,report,config_path,source_pins,
                            filter_path=getattr(args, 'official_filter', None))
        return
    verify(source_pins)
    print('GGX: sim_init (load model once)', flush=True)
    started = time.monotonic()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from tools import preflight_ggx_ur5_table as preflight
    arguments = ['--config', str(config_path), '--no-rejected-video']
    if getattr(args, 'execute', False):
        arguments.append('--no-video')  # The actual checked CB3 video replaces this preview.
    sim_args = preflight.parse_args(arguments)
    scene = preflight.load_scene(sim_args)
    verify(source_pins)
    source_pins.update(scene.dependencies)
    report['source_sha256'] = source_pins
    report['simulation'] = dict(mode='in_process_fixed_scene/v1',
        initialization_s=time.monotonic() - started, model_loads=1,
        rejected_videos=False, convenience_video='copy_not_rerender', stages=[])
    save_report(out, report)
    print(f"GGX: sim_init completed ({report['simulation']['initialization_s']:.2f}s)", flush=True)
    count = min(args.max_candidates, len(scene.candidates))
    fallback_enabled = getattr(args, 'fallback_review', True)
    eligible = []
    automatic_passes = 0
    for rank in range(count):
        verify(source_pins)
        sim_out = out / f'sim_rank_{rank:03d}'
        stage = dict(rank=rank, status='running')
        report['simulation']['stages'].append(stage)
        save_report(out, report)
        print(f'GGX: sim_rank_{rank:03d}', flush=True)
        started = time.monotonic()
        evaluated = []
        try:
            with (out / f'sim_rank_{rank:03d}.log').open('x') as log:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        rc = preflight.main([*arguments, '--candidate-rank', str(rank),
                                             '--out', str(sim_out)], scene=scene, evaluated=evaluated)
                    except BaseException:
                        traceback.print_exc()
                        raise
            if rc not in (0, 2):
                raise RuntimeError(f'sim_rank_{rank:03d} failed ({rc}); see its log')
            stage['status'] = 'passed' if rc == 0 else 'rejected'
        except BaseException:
            stage['status'] = 'failed'
            raise
        finally:
            stage['elapsed_s'] = time.monotonic() - started
            save_report(out, report)
        verify(source_pins)
        sim = read_json(sim_out / 'summary.json')
        if bool(sim['passed']) != (rc == 0):
            raise ValueError('Preflight return code and report disagree')
        if len(evaluated) != 1 or evaluated[0][0].rank != rank:
            raise ValueError('Expected exactly one matching in-process evaluation')
        if not sim['passed']:
            with (sim_out / 'candidates.csv').open() as stream:
                reasons = '; '.join(row['reason'] for row in csv.DictReader(stream))
            report['reviews'].append(dict(rank=rank, decision='automatic_reject',
                                         reason=reasons, report=str(sim_out / 'candidates.csv')))
            save_report(out, report)
            print(f"rank {rank}: 未通过（{reasons}，{stage['elapsed_s']:.2f}s），自动检查下一个。", flush=True)
            if (fallback_enabled and automatic_passes == 0
                    and preflight.manual_object_review_eligible(*evaluated[0], sim_args)):
                dependencies = [sim_out / 'summary.json', sim_out / 'candidates.csv', sim_out / 'trajectory.csv']
                eligible.append((*evaluated[0], pin(dependencies)))
            continue
        automatic_passes += 1
        eligible.clear()  # Fallback is only for zero automatic passes, not vetoes.
        timing = sim['timings']
        print(f"rank {rank}: 通过；预检 {timing['evaluation_s']:.2f}s，"
              f"视频 {timing['render_s']:.2f}s。", flush=True)
        for result_rank, video, review_pins in reviewed_candidates(
                sim_out, require_video=not getattr(args, 'execute', False)):
            yield result_rank, video, {**source_pins, **review_pins}
    if fallback_enabled and automatic_passes == 0 and eligible:
        yield from manual_review_candidates(out, scene, sim_args,
                                           eligible, source_pins, report, execution=getattr(args, 'execute', False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture-live', action='store_true', help='Explicit camera-only fresh capture')
    mode.add_argument('--capture', type=Path, help='Offline saved RGB-D input')
    mode.add_argument('--review-run', type=Path, help='Offline UI review of an existing preflight directory')
    mode.add_argument('--candidate-run', type=Path,
                      help='Reuse saved simulation.yaml + inference for official-only or strict official-CB3 review; no inference/capture')
    parser.add_argument('--dimensions-mm', type=float, nargs=3, metavar=('SHORT', 'LONG', 'HEIGHT'))
    parser.add_argument('--crop-mode', choices=('flat-top', 'foreground'), default='flat-top',
                        help='flat-top: known rectangular box; foreground: selected non-flat opaque object')
    parser.add_argument('--freeze-reference-yaw', action='store_true',
                        help='Foreground: allow/freeze initial ambiguous box axis for XY-only repeated experiments')
    parser.add_argument('--feasibility-mode',choices=['legacy','official-cb3','official-only'],default='legacy',
                        help='official-only: original top-100 + official endpoint scene filter/hand preview; local execution assessed separately')
    parser.add_argument('--local-preview', action='store_true',
                        help='Official-only: automatically assess/render local path before the sole Enter; target-proxy contact gets a labeled local-only diagnostic, never motion permission')
    parser.add_argument('--auto-select', action='store_true',
                        help='Official-CB3 only: select first strict geometry pass in original top-100 order; no candidate videos/approval or hardware')
    parser.add_argument('--official-filter', type=Path,
                        help='With --candidate-run and official-cb3: reuse a matching pinned official filter')
    parser.add_argument('--scene-collision-mm',type=float,default=20.,
                        help='Official scene-distance threshold; predeclared protocol, default 20 mm')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/tabletop_vision_lab.yaml')
    parser.add_argument('--background', type=Path)
    parser.add_argument('--start-snapshot', type=Path, default=ROOT / 'artifacts/vision_pick_start/start_pose.json',
                        help='Saved q seed; with --reset-first also the recorded scan return target, never a live reading')
    parser.add_argument('--ggx-python', type=Path, default=ROOT / 'artifacts/ggx_lab/GraspGenX/.venv/bin/python')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--fallback-review', action=argparse.BooleanOptionalAction, default=True,
                        help='Offline only: if zero passes, show IK/desk-clear object-collision rejects for diagnosis; suppressed with --execute')
    parser.add_argument('--open', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--execute', action='store_true', help='Check the real CB3 path before video review; final Enter authorizes execution')
    parser.add_argument('--static-scene', action='store_true',
                        help='With --execute: operator confirms fixed reference; skip repeated camera captures and RGB/depth comparisons')
    parser.add_argument('--reset-first', action='store_true',
                        help='With --capture-live: explicitly permit separately confirmed open/retreat/return, then wait for Enter before capture')
    parser.add_argument('--real-config', type=Path, default=ROOT / 'configs/ggx_real_lab.yaml')
    parser.add_argument('--real-phase', choices=('approach-only', 'grasp-lift'), default='approach-only')
    parser.add_argument('--real-video', action=argparse.BooleanOptionalAction, default=None,
                        help='Legacy video override; --execute always requires the checked CB3 video')
    parser.add_argument('--allow-high-force', action='store_true',
                        help='Allow configured raw hand speed/force above 100, including reset opening; Enter still required')
    args = parser.parse_args(argv)
    if not math.isfinite(args.scene_collision_mm) or not 0 < args.scene_collision_mm <= 100:
        parser.error('Invalid positive official scene-distance threshold')
    if args.review_run and args.feasibility_mode != 'legacy':
        parser.error('An existing review retains its own recorded checker; do not override its protocol')
    if args.local_preview and (args.execute or args.reset_first
                               or (not args.review_run and args.feasibility_mode != 'official-only')):
        parser.error('--local-preview requires an offline official-only candidate flow; no --execute/--reset-first')
    if args.auto_select and (args.feasibility_mode != 'official-cb3' or args.execute
                            or args.reset_first or args.review_run or args.local_preview):
        parser.error('--auto-select requires official-cb3 generation/saved batch without hardware or diagnostic review')
    if args.auto_select:
        if args.max_candidates != 100:
            parser.error('--auto-select uses the fixed original top-100 protocol')
        args.render_candidates = False
        args.fallback_review = False
    if args.official_filter and not (args.candidate_run and args.feasibility_mode == 'official-cb3'):
        parser.error('--official-filter requires --candidate-run --feasibility-mode official-cb3')
    if args.candidate_run and args.feasibility_mode not in ('official-only', 'official-cb3'):
        parser.error('--candidate-run requires official-only or official-cb3')
    if args.candidate_run and (args.dimensions_mm is not None or args.freeze_reference_yaw
                               or args.background is not None or args.crop_mode != 'flat-top'):
        parser.error('--candidate-run retains the saved crop/dimensions/yaw/background; do not override them')
    if args.feasibility_mode == 'official-only' and (args.execute or args.reset_first):
        parser.error('Official-only selects a candidate, not motion; physical OPEN mapping is still unverified. Do not use --execute/--reset-first.')
    if args.freeze_reference_yaw and args.crop_mode != 'foreground':
        parser.error('--freeze-reference-yaw requires --crop-mode foreground')
    if args.static_scene and not args.execute:
        parser.error('--static-scene requires --execute')
    if args.reset_first and not args.capture_live:
        parser.error('--reset-first requires --capture-live; never combine reset with saved captures/approvals')
    if not 1 <= args.max_candidates <= 100:
        parser.error('--max-candidates must be 1..100')
    if not (args.review_run or args.candidate_run) and (args.dimensions_mm is None or
            not all(math.isfinite(v) and v > 0 for v in args.dimensions_mm)
            or args.dimensions_mm[0] > args.dimensions_mm[1]):
        parser.error('Generation requires measured positive --dimensions-mm SHORT LONG HEIGHT, SHORT <= LONG')
    if args.review_run and args.crop_mode != 'flat-top':
        parser.error('--crop-mode only applies to generation, not an existing review')
    if args.execute and args.real_video is False:
        parser.error('--execute now reviews the checked CB3 video; cannot use --no-real-video')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # Resolving a venv Python symlink selects the base interpreter and
            # silently loses that venv's Torch/GGX packages.
            setattr(args, key, value.absolute() if key == 'ggx_python' else value.resolve())
    return args


def review_checked_candidate(args, out, report, rank, video, pins, authorization):
    """Prepare an automatic pass before asking for visual/motion approval."""
    from tools import run_ggx_real
    verify(pins)
    candidate = out / f'execution_candidate_{rank:03d}.json'
    candidate.write_text(json.dumps(dict(schema='rh56_ggx_execution_candidate/v1',
        rank=rank, human_approved=False, executable=False,
        preflight=str((video.parent/'summary.json').resolve()), source_sha256=pins,
        assumptions=['Automatic pass only; human has NOT approved this candidate.',
                     'CB3/controller checks and a new checked-video Enter are required before motion.']),
        indent=2, allow_nan=False)+'\n')
    real_out = out / f'real_rank_{rank:03d}'
    command = ['--candidate', str(candidate), '--config', str(args.real_config),
               '--execute', '--phase', args.real_phase, '--out', str(real_out)]
    if not args.open:
        command.append('--no-open')
    if args.allow_high_force:
        command.append('--allow-high-force')
    if args.static_scene:
        command.append('--static-scene')
    rc = run_ggx_real.main(command, controller_authorization=authorization)
    result = read_json(real_out/'summary.json')
    report['reviews'].append(dict(rank=rank, scope='checked_cb3_execution',
        decision=result.get('checked_review', {}).get('decision', 'not_approved'),
        status=result['status'], report=str(real_out/'summary.json')))
    report['robot_commands_sent'] |= bool(result.get('motion_commands_attempted') or result.get('hand_commands'))
    report['status'] = result['status']
    if result.get('motion_confirmation_received'):
        report['selected_rank'] = rank
    save_report(out, report)
    return rc, result['status']


def main(argv=None):
    args = parse_args(argv)
    # Set before reset can import MuJoCo; formerly the sim subprocess did this.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if args.execute or args.reset_first:
        # Fail before reset/capture/inference, not after a long visual review.
        from tools.run_ggx_real import load_config, require_hand_setting_authorization
        require_hand_setting_authorization(load_config(args.real_config), args.allow_high_force)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.out or ROOT / 'artifacts/ggx_review_runs' / stamp
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rh56_ggx_interactive_review/v1', status='running',
                  executable=False, robot_commands_sent=False, selected_rank=None,
                  live_camera_requested=args.capture_live, reviews=[],
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  assumptions=[
                      'Offline Enter is visual-only; --execute shows the checked CB3 path before motion approval.',
                      'Manual object selection and known upright dimensions; crop_mode records flat-top or measured foreground. Not universal vision.',
                      'Saved joint seed; UR5e sampled desk + pre-closure object checks, not CB3/dynamic pickup validation.',
                      'Physical movement is not detected by file hashes; r invalidates the whole scene.',
                      'No implicit reset/release/retry; explicit --reset-first is separately confirmed before NEW capture.',
                      'No transfer of an approval to a later run.',
                      'Zero-pass manual fallback is visual-only; acceptance cannot authorize object-collision rejects for real motion.',
                  ])
    save_report(out, report)
    if args.feasibility_mode == 'official-cb3':
        report['assumptions'][2] = 'Official scene filter + start-independent CB3 grasp geometry; actual travel checked per trial, target proxy uncertainty not grasp failure.'
        save_report(out,report)
    if args.feasibility_mode == 'official-only':
        report['assumptions'] = [
            'Original GGX scores/order/top-100 unchanged; no local IK or target proxy selection filter.',
            'Official endpoint scene-distance filter and official OPEN geometry; not a swept-path certificate.',
            'Measured target is displayed, but excluded from official non-target scene filtering.',
            'Local execution and physical OPEN command/geometry correspondence remain separate and unverified.',
            'Enter selects one candidate only; no automatic replacement after local execution rejection.',
        ]
        save_report(out, report)
    print(('GGX 官方候选初审：静态终态点云预览；Enter 只选择候选，不授权运动。' if args.feasibility_mode == 'official-only' else
           'GGX 实机审核：先做 CB3/控制器检查，再看执行视频；Enter 会授权执行，n 换下一项。' if args.execute else
           'GGX 离线审核：Enter 只记录当前展示内容的选择；检查范围见下方，不发送实机动作。') +
          ' 物体移动后按 r，重新拍摄。', flush=True)
    rc = 2
    candidates = None
    controller_authorization = {}  # One explicit RTDE-only grant per invocation, never persisted/reused.
    try:
        if args.reset_first:
            from tools.ggx_reset import run_reset
            reset_out = out / 'reset'
            rc, scan_snapshot = run_reset(args, reset_out)
            reset_report = read_json(reset_out / 'summary.json')
            report['reset_report'] = str(reset_out / 'summary.json')
            report['robot_commands_sent'] = bool(reset_report.get('hand_commands')
                                                  or reset_report.get('motion_commands_attempted'))
            report['robot_commands_sent_semantics'] = 'reset command attempts; not hardware completion feedback'
            if rc != 0:
                report['status'] = 'reset_' + reset_report['status']
                return rc
            args.start_snapshot = scan_snapshot
            report['generation_start_snapshot'] = str(scan_snapshot)
            report['status'] = 'awaiting_placement_confirmation'
            save_report(out, report)
            # generate_candidates owns the placement Enter gate, BEFORE camera
            # startup. Return to normal status for its no-pass/cancel handling.
            report['status'] = 'running'
            rc = 2
        candidates = (reviewed_candidates(args.review_run) if args.review_run else
                      reuse_candidates(args, out, report) if args.candidate_run else
                      generate_candidates(args, out, report))
        for rank, video, pins in candidates:
            manual_only = review_is_manual_only(video, pins)
            candidate_report = video.parent/'summary.json'
            schema = read_json(candidate_report).get('schema')
            official_only = schema == 'rh56_ggx_official_candidate_selection/v1'
            geometry_only = schema == 'rh56_ggx_cb3_grasp_geometry/v1'
            if args.auto_select:
                from tools import ggx_real_support as support
                if manual_only or not geometry_only:
                    raise ValueError('Automatic template selection requires a strict local geometry pass')
                # Validate the full handoff, not just the generator yield or video name.
                support.read_candidate_source(candidate_report, rank, candidate_report,
                                              pins, 'unreviewed_candidate')
                proposal = out / 'selected_candidate.json'
                proposal.write_text(json.dumps(dict(schema='rh56_ggx_execution_candidate/v1',
                    rank=rank, human_approved=False, executable=False,
                    preflight=str(candidate_report.resolve()), source_sha256=pins,
                    assumptions=['First strict geometry pass in unchanged GGX top-100 order.',
                        'No target-contact diagnostic fallback; no actual travel or physical success claim.',
                        'Template selection is automatic, NOT human motion approval.']), indent=2)+'\n')
                report.update(status='auto_candidate_selected', selected_rank=rank,
                              selected_candidate=str(proposal.resolve()), human_approved=False,
                              executable=False, actual_travel_checked=False)
                print(f'自动选中原始 rank {rank}；严格局部检查通过，十点实际路径仍需分别检查。', flush=True)
                rc = 0
                break
            local_preview_record = None
            if args.local_preview and not official_only:
                raise ValueError('--local-preview requires a saved official-only candidate; no legacy checker fallback')
            if official_only:
                if args.execute:
                    raise ValueError('Official candidate selection is not motion approval; physical OPEN mapping is unverified. Use offline local assessment first.')
                if args.local_preview:
                    from tools import ggx_real_support as support, ggx_quick_preview as quick
                    source = support.read_candidate_source(candidate_report, rank, candidate_report,
                                                           pins, 'offline_diagnostic')
                    local_out = out / f'local_rank_{rank:03d}'
                    print(f'GGX: local_preview_rank_{rank:03d}', flush=True)
                    local = quick.prepare_preview(source, read_json(args.start_snapshot), local_out)
                    local_preview_record = dict(rank=rank, status=local['status'],
                        path_scope=local['path_scope'], report=str(local_out/'summary.json'))
                    report.setdefault('local_previews', []).append(local_preview_record)
                    save_report(out, report)
                    if not local.get('video'):
                        report['reviews'].append(dict(rank=rank, decision='local_preview_unavailable',
                            scope='official_ggx_candidate', official_candidate_retained=True,
                            reason=local.get('reason', local['status']), report=str(local_out/'summary.json')))
                        print(f'rank {rank}: 没有可显示的本地路径，记录原因并查看下一原始排名。', flush=True)
                        save_report(out, report)
                        continue
                    video = Path(local['video'])
                    pins = {**pins, **local.get('code_config_sha256', {}),
                            **pin([video, local_out/'summary.json', args.start_snapshot])}
                    if local['status'] == 'target_contact_warning_preview':
                        print('目标代理接触警告：显示局部接近/闭合预览；未验证从当前起点的行程，不是可执行通过项。', flush=True)
                    else:
                        print('已自动生成保存起点至抓取/抬升的本地路径预览；无需再运行第二条命令，未授权运动。', flush=True)
                else:
                    print('官方张开手型 + 实测点云的终态预览；不是运动仿真。本地手型、IK、桌面和接近路径均未评估。', flush=True)
            if geometry_only and not args.execute:
                print('局部接近已使用执行侧同一几何规则；未检查扫描位/十起点行程。Enter 只选择模板，不授权运动。',flush=True)
            if args.execute and manual_only:
                report['reviews'].append(dict(rank=rank, scope='manual_object_collision',
                                             decision='not_executable', reason='diagnostic_only_candidate'))
                save_report(out, report)
                print(f'rank {rank}: 仅供诊断，实机模式跳过，不显示执行确认。', flush=True)
                continue
            if args.execute and not manual_only:
                rc, status = review_checked_candidate(args, out, report, rank, video, pins,
                                                      controller_authorization)
                if status in ('cb3_review_vetoed', 'cb3_candidate_rejected'):
                    print(f'rank {rank}: {status}，继续下一候选。', flush=True)
                    report['status'] = 'running'
                    rc = 2
                    continue
                break
            options = dict(manual_only=manual_only)
            if official_only:
                options['official_only'] = True
                if args.local_preview:
                    options['local_preview'] = True
            action, reason = decision(video, pins, args.open, **options)
            row = dict(rank=rank, decision=action, reason=reason,
                                         scope='official_ggx_candidate' if official_only else 'manual_object_collision' if manual_only else 'grasp_geometry' if geometry_only else 'automatic_pass',
                                         video=str(video), reviewed_sha256=pins)
            if local_preview_record is not None:
                row.update(candidate_report=str(candidate_report.resolve()),
                           local_preview_report=local_preview_record['report'],
                           local_preview_status=local_preview_record['status'])
            report['reviews'].append(row)
            save_report(out, report)
            if action == 'approve':
                report.update(status='official_candidate_selected' if official_only else 'manual_review_accepted_not_executable' if manual_only else 'review_approved',
                              selected_rank=rank)
                rc = 0
                break
            if action != 'veto':
                report['status'] = action
                break
        else:
            if report['status'] == 'running':
                report['status'] = 'no_candidate_approved'
    except (EOFError, KeyboardInterrupt):
        report['status'] = 'cancelled'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(report['error'], file=sys.stderr, flush=True)
        rc = 1
    finally:
        if candidates is not None:
            candidates.close()  # Release the fixed sim scene before any real handoff.
        save_report(out, report)
        print(f"GGX: {report['status']}；记录：{out / 'summary.json'}", flush=True)
    if rc == 0 and report['status'] == 'manual_review_accepted_not_executable':
        print('已保存人工认可；此候选仍有未解决的物体碰撞，未进入实机流程（即使指定 --execute）。')
        return rc
    if args.execute:
        return rc  # Checked and executed in-process before leaving the candidate loop.
    if rc == 0 and report['status'] == 'auto_candidate_selected':
        print('自动筛选已保存，未记录人工批准、未连接实机；候选可用于十点模板。')
        return rc
    if rc == 0 and report['status'] == 'official_candidate_selected':
        print('候选和本地预览已保存；不要加 --execute，真实手型映射仍待确认。' if args.local_preview else
              '官方候选已选择；未证明本地可执行。可用 run_ggx_real.py --approval 本记录 --no-video 做离线本地评估；不要加 --execute。')
        return rc
    if rc == 0:
        print('审核已保存。未指定 --execute，因此不会连接实机；可用 run_ggx_real.py --approval 指定本记录继续。')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
