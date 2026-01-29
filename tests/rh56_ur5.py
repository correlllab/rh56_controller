#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run coordinated UR5 + RH56 trials with poses defined directly in this file.

Procedure (per trial):
  1) clear errors + open hand
  2) set hand "prepare posture" (from your presets)
  3) move arm to *_prep pose
  4) close hand (preset speed + close angles)
  5) move arm to *_end pose (lift)
  6) optional shake (between end and shake pose)
  7) return to *_prep pose (or "return_to_start" if enabled)
  8) open hand

Usage:
  python3 run_arm_hand_trial.py --trial tripod --shake 4
  python3 run_arm_hand_trial.py --trial precision
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# --- UR5 + RH56 ---
from magpie_control import ur5
from rh56_controller.rh56_hand import RH56Hand

# --- Reuse your hand presets/utilities from experiment.py ---
from experiment import build_presets, apply_angles, apply_speed, clear_and_open, DEFAULT_OPEN


# =========================
#  Poses
# =========================

pose_precision_prep = np.array([
    [-0.999,  0.005,  -0.034, -0.431],
    [ 0.033,  0.346,  -0.938, -0.390],
    [ 0.007, -0.938,  -0.346,  0.183],
    [ 0.,     0. ,    0. ,    1.   ]
], dtype=float)

pose_precision_end = np.array([
    [-0.999,  0.005,  -0.034, -0.431],
    [ 0.033,  0.346,  -0.938, -0.390],
    [ 0.007, -0.938,  -0.346,  0.300],
    [ 0.,     0. ,    0. ,    1.   ]
], dtype=float)

pose_precision_shake = np.array([
    [-0.333,  0.943, -0.029, -0.431],
    [ 0.342,  0.091, -0.935, -0.390],
    [ -0.879, -0.321, -0.353,  0.300],
    [ 0.,     0. ,    0. ,    1.   ]
], dtype=float)

pose_tripod_prep = np.array([
    [-0.993,  0.091, -0.071, -0.373],
    [ 0.112,  0.621, -0.776, -0.366],
    [-0.026, -0.779, -0.627,  0.245],#close to table 0.246, now higher
    [ 0.,     0.,     0.,     1.   ]
], dtype=float)

pose_tripod_end = np.array([
    [-0.993,  0.091, -0.071, -0.373],
    [ 0.112,  0.621, -0.776, -0.366],
    [-0.026, -0.779, -0.627,  0.400],
    [ 0.,     0.,     0.,     1.   ]
], dtype=float)

pose_tripod_shake = np.array([
    [-0.286,  0.956, -0.065, -0.373],
    [ 0.62,  0.133, -0.773, -0.366],
    [-0.73, -0.262, -0.631,  0.400],
    [ 0.,     0.,     0.,     1.   ]
], dtype=float)

# Poses for peg-in-hole task
pose_peg_0 = np.array([
    [-0.998,  0.009,  -0.064, -0.488],
    [ 0.056,  0.615,  -0.787, -0.318],
    [ 0.032, -0.789,  -0.614,  0.269],
    [ 0.,     0. ,    0. ,    1.   ]
], dtype=float)

TRIALS: Dict[str, Dict[str, Optional[np.ndarray]]] = {
    # "precision" == pinch-style in your naming; we’ll map it to preset key "1"
    "precision": {
        "prep": pose_precision_prep,
        "end": pose_precision_end,
        "shake": pose_precision_shake,  # no shake pose provided; can add later if you want
    },
    "tripod": {
        "prep": pose_tripod_prep,
        "end": pose_tripod_end,
        "shake": pose_tripod_shake,
    },
}

# Map trial name -> your preset mode key in experiment.py presets dict
# (based on your earlier description: 1=pinch, 2=tripod)
TRIAL_TO_MODE = {
    "precision": "1",
    "tripod": "2",
}


def _log(stage: str, msg: str = "") -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] [{stage}] {msg}".rstrip())


def _assert_pose(T: np.ndarray, name: str) -> None:
    if not isinstance(T, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray")
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError(f"{name} has NaN/Inf")


def shake_between(robot, A: np.ndarray, B: np.ndarray, cycles: int, linSpeed: float, linAccel: float) -> None:
    _log("SHAKE", f"{cycles} cycles")
    for _ in range(cycles):
        robot.moveL(A, linSpeed=linSpeed, linAccel=linAccel, asynch=False)
        robot.moveL(B, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

def sample_for(duration_s: float, period_s: float, hand: RH56Hand, robot):
    end_t = time.monotonic() + duration_s
    next_t = time.monotonic()
    count = 0
    first_count = 0
    reclosing_count = 0
    x = 1
    current_angle = []
    while time.monotonic() < end_t:
        data = hand.force_act()
        if first_count == 0:
            first = data
        diff_2 = abs(data[2] - first[2])
        diff_3 = abs(data[3] - first[3])
        diff = max(diff_2, diff_3)
        # print(f"Diff: {diff}, Reclosing count: {reclosing_count}")
        if (diff >(20.0)) and (reclosing_count == 0):
            apply_angles(hand, [1000, 1000, 0, 0, 850, 0], "re-closing during sample")
            x = x + 1
            # reclosing_count += 1
        # print(data)
        count = count + 1
        first_count += 1
        next_t += period_s
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
    print("First sample:", first)
    print("Last sample:", data)
    rate = count / duration_s
    print(f"Sampled {count} data points over {duration_s:.2f}s ({rate:.1f} Hz)")

import matplotlib.pyplot as plt

# 引入 experiment.py 中的控制函数 (确保 experiment.py 在同级目录)
try:
    from experiment import DEFAULT_OPEN, apply_angles, apply_speed
except ImportError:
    # 防止报错的简单定义
    DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
    def apply_angles(hand, angles, label): hand.angle_set(list(angles))
    def apply_speed(hand, speed, label): hand.speed_set([speed]*6)

# === 用户定义的参数 (保留不变) ===
IDX_FINGER_A = 3    # Index finger (用于监测插孔逻辑)
IDX_FINGER_B = 4    # Thumb finger (用于初始触发)

# === 逻辑阈值 (Raw 0-1000) ===
THUMB_SPIKE_THRESH = 50      # 拇指激增阈值
INDEX_STABLE_THRESH = 300    # 食指稳定力值
INDEX_STABLE_TIME = 2.0      # 稳定持续时间 (秒)
MODE_X_SPIKE = 75            # 插孔突增
MODE_X_DROP = 25             # 插孔骤降 (入位)
MODE_X_WINDOW = 0.5          # 滑动窗口时间

def run_peginhole_task(hand, robot, pose_lift, pose_return, close_angles):
    """
    执行 X 模式逻辑，同时保留原有的数据采集和绘图功能。
    """
    print(f"[{time.strftime('%H:%M:%S')}] START PEG-IN-HOLE TASK (X Mode)...")
    
    # 1. 初始化数据容器 (保留原有结构)
    times = []
    forces_finger_a = [] # Index (Mapped N)
    forces_finger_b = [] # Thumb (Mapped N)
    forces_wrist = []    # UR5 Wrist (N)
    phases = []          # 记录当前阶段方便画图分析
    
    # 2. 状态机变量
    current_phase = 0    # 0:检测拇指, 1:检测食指稳定, 2:检测插孔Drop, 3:完成
    start_t = time.monotonic()
    
    # Phase 0 变量
    thumb_baseline = None
    
    # Phase 1 变量
    stable_start_t = None
    
    # Phase 2 变量 (Mode X Windowing)
    window_samples = []
    window_start_t = time.monotonic()
    prev_avg = None
    spike_detected = False
    
    task_running = True
    
    try:
        while task_running:
            loop_start = time.monotonic()
            current_t = loop_start - start_t
            
            # =========================================================
            # part A: 数据读取 (完全保留你的逻辑)
            # =========================================================
            # 1. 读取 RH56
            hand_data = hand.force_act()
            
            # 2. 读取 UR5
            if hasattr(robot, 'get_tcp_force'):
                wrist_wrench = robot.get_tcp_force()
            elif hasattr(robot, 'get_force'):
                wrist_wrench = robot.get_force()
            else:
                wrist_wrench = [0.0] * 6 
            
            # 3. 数据处理与映射 (完全保留你的公式)
            # Index
            raw_index = hand_data[IDX_FINGER_A]
            f_a_newton = (raw_index * 0.007478) - 0.414
            
            # Thumb
            raw_thumb = hand_data[IDX_FINGER_B]
            f_b_newton = (raw_thumb * 0.012547) + 0.384
            
            # UR5 Wrist
            f_wrist_newton = np.linalg.norm(wrist_wrench[:3])

            # 4. 存储数据
            times.append(current_t)
            forces_finger_a.append(f_a_newton)
            forces_finger_b.append(f_b_newton)
            forces_wrist.append(f_wrist_newton)
            phases.append(current_phase)
            
            # =========================================================
            # part B: 核心控制逻辑 (X Mode State Machine)
            # =========================================================
            
            # --- 阶段 0: 外部推动 -> 检测大拇指激增 -> 闭合手指 ---
            robot.moveL(pose_peg_0, linSpeed=0.2, linAccel=0.5, asynch=False)  # no contact pose
            if current_phase == 0:
                robot.moveL(pose_peg_1, linSpeed=0.2, linAccel=0.5, asynch=True)  # move to contact pose
                if thumb_baseline is None: thumb_baseline = raw_thumb
                
                # 简单的基准线跟随 (适应缓慢漂移)
                if raw_thumb < thumb_baseline: thumb_baseline = raw_thumb
                else: thumb_baseline += (raw_thumb - thumb_baseline) * 0.05
                
                # 判断激增
                if (raw_thumb - thumb_baseline) > THUMB_SPIKE_THRESH:
                    print(f">>> [PHASE 0] Thumb Spike! Closing Hand.")
                    apply_speed(hand, 50, "slow close")
                    apply_angles(hand, close_angles, "Close Grip")
                    current_phase = 1
                    stable_start_t = None # 重置下一阶段计时器

            # --- 阶段 1: 抬起前检测 -> 食指 > 300 持续 2秒 -> 机械臂抬起 ---
            elif current_phase == 1:
                # 检查食指力度
                if raw_index > INDEX_STABLE_THRESH:
                    if stable_start_t is None:
                        stable_start_t = time.monotonic()
                    
                    # 检查持续时间
                    duration = time.monotonic() - stable_start_t
                    if duration >= INDEX_STABLE_TIME:
                        print(f">>> [PHASE 1] Stable > 300 for {duration:.1f}s. Moving Arm UP (Blocking).")
                        
                        # 动作：抬起 (阻塞/同步)，确保动作完成才继续
                        robot.moveL(pose_peg_lift, linSpeed=0.5, linAccel=0.5, asynch=False)
                        print(">>> Arm Lift Done. Preparing Return.")
                        
                        # 动作：开始回退 (异步)，进入下一阶段边动边测
                        # 这里稍微停顿一下确保状态切换
                        time.sleep(0.5) 
                        robot.moveL(pose_peg_insert, linSpeed=0.2, linAccel=0.5, asynch=False)
                        print(">>> Arm Returning (Async)... Monitoring Drop.")
                        
                        # 初始化 Phase 2 的窗口参数
                        current_phase = 2
                        window_samples = []
                        window_start_t = time.monotonic()
                        prev_avg = None
                        spike_detected = False
                else:
                    # 如果中途掉下来，计时器重置
                    stable_start_t = None

            # --- 阶段 2: 回退中 -> 检测 Spike + Drop -> 松手 ---
            elif current_phase == 2:
                # Mode X 滑动窗口逻辑
                window_samples.append(raw_index)
                
                if (time.monotonic() - window_start_t) >= MODE_X_WINDOW:
                    robot.moveL(pose_peg_2, linSpeed=0.2, linAccel=0.5, asynch=True)  # lateral move
                    if len(window_samples) > 0:
                        avg_force = sum(window_samples) / len(window_samples)
                        
                        if prev_avg is not None:
                            delta = avg_force - prev_avg
                            
                            if not spike_detected:
                                # 寻找激增 (接触)
                                if delta >= MODE_X_SPIKE:
                                    spike_detected = True
                                    print(f">>> [PHASE 2] Contact Spike (+{delta:.1f}). Waiting for Drop.")
                                    robot.moveL(pose_peg_3, linSpeed=0.2, linAccel=0.5, asynch=True)  # continue return
                            else:
                                # 寻找骤降 (入位)
                                drop = -delta
                                if drop >= MODE_X_DROP:
                                    print(f">>> [SUCCESS] Drop detected (-{drop:.1f}). Peg Inserted!")
                                    
                                    # 动作：松手
                                    # robot.stopL() # 立即停止机械臂
                                    apply_speed(hand, 1000, "fast open")
                                    apply_angles(hand, DEFAULT_OPEN, "Open Hand")
                                    
                                    task_running = False # 结束循环
                                    current_phase = 3
                        
                        prev_avg = avg_force
                        window_samples = []
                        window_start_t = time.monotonic()

            # 简单的频率控制 (约100Hz)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user.")
        robot.stopL()
    
    print(f"Task Finished. Collected {len(times)} data points.")
    
    # =========================
    #  绘图 (保留你的 Matplotlib 逻辑)
    # =========================
    plt.figure(figsize=(12, 8))
    
    # 上图：UR5 力度
    plt.subplot(2, 1, 1)
    plt.plot(times, forces_wrist, label='UR5 Wrist (N)', color='black')
    plt.title(f"Arm Force - Phase: {current_phase}")
    plt.grid(True)
    plt.legend()
    
    # 下图：手指力度 (你的映射数据)
    plt.subplot(2, 1, 2)
    # 用不同背景色标记阶段
    p_arr = np.array(phases)
    t_arr = np.array(times)
    if len(t_arr) > 0:
        plt.fill_between(t_arr, 0, max(forces_finger_b)*1.1, where=(p_arr==0), color='gray', alpha=0.2, label='Wait Thumb')
        plt.fill_between(t_arr, 0, max(forces_finger_b)*1.1, where=(p_arr==1), color='orange', alpha=0.2, label='Wait Stability')
        plt.fill_between(t_arr, 0, max(forces_finger_b)*1.1, where=(p_arr==2), color='green', alpha=0.2, label='Insertion')

    plt.plot(times, forces_finger_a, label=f'Index (Idx{IDX_FINGER_A}) Mapped', color='blue')
    plt.plot(times, forces_finger_b, label=f'Thumb (Idx{IDX_FINGER_B}) Mapped', color='red')
    
    plt.axhline(y=(300*0.007478 - 0.414), color='blue', linestyle=':', label='Stability Thresh (approx)')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    
    save_path = "peginhole_x_mode_plot.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show()

def run_trial(
    robot,
    hand: RH56Hand,
    trial_name: str,
    linSpeed: float,
    linAccel: float,
    settle_s: float,
    close_wait_s: float,
    shake_cycles: int,
    return_to_start: bool,
) -> None:
    if trial_name not in TRIALS:
        raise ValueError(f"Unknown trial '{trial_name}'. Options: {list(TRIALS.keys())}")

    poses = TRIALS[trial_name]
    prep = poses["prep"]
    end = poses["end"]
    shake_pose = poses["shake"]

    _assert_pose(prep, f"{trial_name}.prep")
    _assert_pose(end, f"{trial_name}.end")
    if shake_pose is not None:
        _assert_pose(shake_pose, f"{trial_name}.shake")

    mode_key = TRIAL_TO_MODE[trial_name]
    presets = build_presets()
    if mode_key not in presets:
        raise ValueError(f"Preset mode '{mode_key}' not in build_presets() keys: {list(presets.keys())}")
    preset = presets[mode_key]

    start_pose = robot.get_tcp_pose() if return_to_start else None

    # 1) safe start
    _log("INIT", "Clear errors + open hand")
    clear_and_open(hand)
    time.sleep(0.5)

    # 2) prep hand
    _log("HAND", f"Prepare posture: {trial_name} (mode {mode_key}: {preset.name})")
    apply_angles(hand, preset.prepare_angles, "Prepare posture")
    time.sleep(0.5)

    # 3) move arm to prep pose
    _log("ARM", f"MoveL -> {trial_name}.prep")
    robot.moveL(prep, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

    # 4) close hand
    hand.force_set([100] * 6)  # ensure forces are set
    time.sleep(2)
    _log("HAND", f"Close (speed={preset.close_speed})")
    # apply_speed(hand, preset.close_speed, "closing speed")
    apply_speed(hand, 50, "closing speed")
    apply_angles(hand, preset.close_angles, "closing angles")
    force = hand.force_act()
    time.sleep(5)
    while force [3] < 100 and force [2] < 100:
        apply_angles(hand, preset.close_angles, "closing angles")
        time.sleep(1)
        force = hand.force_act()
    # hand.force_set([200] * 6)
    # apply_speed(hand, 1000, "re-closing during sample")

    # 5) lift
    _log("ARM", f"MoveL -> {trial_name}.end (lift)")
    robot.moveL(end, linSpeed=0.5, linAccel=0.35, asynch=False)
    robot.moveL(pose_tripod_shake, linSpeed=0.5, linAccel=1.5, asynch=True)
    # time.sleep(5)
    sample_for(duration_s=500.0, period_s=0.006, hand=hand, robot=robot)

    while True:
        time.sleep(1)

    # 6) optional shake
    if shake_cycles > 0:
        if shake_pose is None:
            _log("SHAKE", "No shake pose provided for this trial; skipping")
            # time.sleep(5)
        else:
            # shake between end and shake_pose
            shake_between(robot, end, shake_pose, shake_cycles, linSpeed, linAccel)
            # return to end after shake
            time.sleep(5)
            robot.moveL(end, linSpeed=0.5, linAccel=3.0, asynch=False)

    # 7) return
    if return_to_start and start_pose is not None:
        _log("ARM", "Return to START pose")
        robot.moveL(start_pose, linSpeed=linSpeed, linAccel=linAccel, asynch=False)
    else:
        _log("ARM", f"Return to {trial_name}.prep")
        robot.moveL(prep, linSpeed=linSpeed, linAccel=linAccel, asynch=False)

    # 8) open hand
    time.sleep(2)
    _log("HAND", f"Open (speed={preset.restore_speed})")
    apply_speed(hand, preset.restore_speed, "opening speed")
    apply_angles(hand, DEFAULT_OPEN, "opening angles")
    time.sleep(1)

    # 9) lift arm higher to clear
    robot.moveL(end, linSpeed=linSpeed, linAccel=linAccel, asynch=False)
    clear_and_open(hand)
    time.sleep(0.5)
    clear_and_open(hand)
    _log("DONE", "Trial complete")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", choices=list(TRIALS.keys()), default="precision",
                    help="Which pose set to run")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="RH56 serial port")
    ap.add_argument("--hand-id", type=int, default=1, help="RH56 hand ID")

    ap.add_argument("--linSpeed", type=float, default=0.5)
    ap.add_argument("--linAccel", type=float, default=0.75)
    ap.add_argument("--settle-s", type=float, default=0.2)
    ap.add_argument("--close-wait-s", type=float, default=0.6)

    ap.add_argument("--shake", type=int, default=0, help="Shake cycles (tripod supports shake pose)")
    ap.add_argument("--return-to-start", action="store_true",
                    help="Return arm to starting TCP pose instead of returning to prep pose")
    return ap.parse_args()


def main():
    args = parse_args()

    _log("CONNECT", "Hand")
    hand = RH56Hand(port=args.port, hand_id=args.hand_id)

    _log("CONNECT", "UR5")
    robot = ur5.UR5_Interface()
    robot.start()

    try:
        run_trial(
            robot=robot,
            hand=hand,
            trial_name=args.trial,
            linSpeed=args.linSpeed,
            linAccel=args.linAccel,
            settle_s=args.settle_s,
            close_wait_s=args.close_wait_s,
            shake_cycles=args.shake,
            return_to_start=args.return_to_start,
        )
    finally:
        # Always leave the hand open
        try:
            _log("CLEANUP", "Open hand")
            clear_and_open(hand)
            apply_angles(hand, DEFAULT_OPEN, "final open")
        except Exception as e:
            _log("CLEANUP", f"Hand cleanup failed: {e}")

        # If your UR5 interface exposes stop/close, call it here.
        # try: robot.stop()
        # except: pass


if __name__ == "__main__":
    main()
