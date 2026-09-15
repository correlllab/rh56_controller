#!/usr/bin/env python3
"""Saved lab cube baseline: offline preview by default; --execute runs ONE supervised cycle.

Reuses run_vision_pick_cycle without changing the Plan, control, or motion rules.
See docs/iterative_cube_lab_zh.md. No unattended mode or automatic release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'configs/ur5_iterative_cube_baseline.json'


def verify_baseline(manifest_path=MANIFEST):
    """Fail before any hardware use if a pinned local dependency is missing/changed."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'rh56_iterative_cube_baseline/v1':
        raise ValueError('Unexpected baseline manifest schema')
    if not manifest.get('sha256') or manifest['config'] not in manifest['sha256']:
        raise ValueError('Baseline manifest must pin the configuration')
    for relative, expected in manifest['sha256'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Baseline path outside repository: {relative}')
        if not path.is_file():
            raise ValueError(f'Baseline file missing: {relative}; restore the saved lab files')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Baseline changed: {relative}; review before establishing a new baseline')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Show safety advice and wait for Enter before connecting hardware; open/reset/scan/grasp/lift once')
    parser.add_argument('--capture', type=Path, help='Offline RGB-D input; defaults to successful trial_04 capture (never used live)')
    parser.add_argument('--out', type=Path, help='New output directory; default artifacts/iterative_cube_runs/<UTC timestamp>')
    args = parser.parse_args(argv)
    if args.execute and args.capture:
        parser.error('--execute cannot use a saved capture; live mode always detects afresh')
    try:
        manifest = verify_baseline()
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.out or ROOT / 'artifacts/iterative_cube_runs' / stamp
    out = out.resolve()
    if out.exists():
        parser.error(f'Output already exists (never overwritten): {out}')
    # Lazy import: --help and failed verification never load the planner/hardware.
    from tools import run_vision_pick_cycle as cycle
    command = ['--config', str(ROOT / manifest['config']), '--out', str(out)]
    print('Saved iterative cube trial_04: 40 mm cube; original Plan wrist-following sequence.\n'
          '基座 XY 偏移 [0, -5] mm；闭合余量 2 mm；force 原始值 1000；planner 原速度。\n'
          '本轮结果目录：' + str(out), flush=True)
    if args.execute:
        print('\n执行前必须现场确认：\n'
              '1. 方块已放在桌上或可靠支撑物上，手中没有悬空物体；张手不会掉落物体。\n'
              '2. 人员离开运动区；张手、垂直退离、回扫描位、抓取和抬升路径均畅通。\n'
              '3. 相机、桌面、手的安装与成功试验一致；文件校验不能检测物理位移。\n'
              '4. 你全程观察并能操作急停；IK/边界检查不是碰撞检测。\n'
              '扫描位到抓取接近点不设固定距离上限；其他运动检查仍保留。\n'
              '本次授权包括 RTDE 控制脚本启动/停止、force=1000、原速度、无手反馈定时控制，\n'
              '以及局部安装/宽度校正在新闭合手形上的外推。\n'
              '流程：张手 → 必要时原地垂直退离（最多150 mm）→ 扫描位 → 新拍照 →\n'
              '原 Plan 抓取 → 垂直抬升50 mm → 保持。不会自动松手、重试或循环。\n', flush=True)
        try:
            if input('确认以上全部条件并执行一轮：按 Enter 继续；输入其他内容或 Ctrl-C 取消。').strip():
                print('已取消；未连接硬件。')
                return 2
        except (EOFError, KeyboardInterrupt):
            print('\n已取消；未连接硬件。')
            return 2
        # Enter explicitly acknowledges the printed scene, reset and force conditions.
        # Recheck after the operator pause. Never silently accept configuration drift.
        verify_baseline()
        command += ['--execute', '--confirm-cycle', cycle.CONFIRMATION,
                    '--acknowledge-open-loop', '--acknowledge-shape-extrapolation',
                    '--confirm-high-force', '--reset-from-low',
                    '--confirm-reset', cycle.RESET_CONFIRMATION]
    else:
        capture = args.capture or ROOT / manifest['preview_capture']
        print('离线预览：使用保存图像，不连接相机、手或机械臂；不代表当前现场安全。', flush=True)
        command += ['--capture', str(capture.resolve())]
    try:
        return cycle.main(command)
    finally:
        # The underlying runner owns directory creation and never overwrites a run.
        if (out / 'summary.json').is_file():
            record = dict(schema='rh56_iterative_cube_launch/v1', baseline=manifest,
                          execute=args.execute, confirmations_completed=args.execute,
                          confirmation_method='interactive_enter' if args.execute else 'none',
                          command=command,
                          assumptions=['Operator-supervised, write-only hand; no automatic release/retry.',
                                       'Checksums verify selected files, not physical alignment or all software dependencies.'])
            with (out / 'launcher.json').open('x', encoding='utf-8') as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')


if __name__ == '__main__':
    raise SystemExit(main())
